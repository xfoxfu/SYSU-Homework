// Copyright (c) 2013-2016 Sandstorm Development Group, Inc. and contributors
// Licensed under the MIT License:
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

use crate::pubsub_capnp::{publisher, subscriber, subscription};
use capnp::capability::Promise;
use capnp_rpc::pry;
use capnp_rpc::{rpc_twoparty_capnp, twoparty, RpcSystem};
use futures::{lock::Mutex, AsyncReadExt, FutureExt, StreamExt, TryFuture};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::{cell::RefCell, net::ToSocketAddrs};
use std::{rc::Rc, time::SystemTime};

struct SubscriberHandle {
    client: subscriber::Client<capnp::text::Owned>,
}

struct SubscriberMap {
    subscribers: HashMap<u64, SubscriberHandle>,
}

impl SubscriberMap {
    fn new() -> SubscriberMap {
        SubscriberMap {
            subscribers: HashMap::new(),
        }
    }
}

struct SubscriptionImpl {
    id: u64,
    subscribers: Rc<RefCell<SubscriberMap>>,
}

impl SubscriptionImpl {
    fn new(id: u64, subscribers: Rc<RefCell<SubscriberMap>>) -> SubscriptionImpl {
        SubscriptionImpl { id, subscribers }
    }
}

impl Drop for SubscriptionImpl {
    fn drop(&mut self) {
        info!("subscriber {} dropped", self.id);
        self.subscribers.borrow_mut().subscribers.remove(&self.id);
    }
}

impl subscription::Server for SubscriptionImpl {}

struct PublisherImpl {
    next_id: u64,
    subscribers: Rc<RefCell<SubscriberMap>>,
    concurrency_limit: Option<u8>,
}

#[allow(clippy::type_complexity)]
impl PublisherImpl {
    pub fn new(
        concurrency_limit: Option<u8>,
    ) -> (
        PublisherImpl,
        Rc<RefCell<SubscriberMap>>,
        Rc<Mutex<VecDeque<(String, SystemTime)>>>,
    ) {
        let subscribers = Rc::new(RefCell::new(SubscriberMap::new()));
        let messages = Rc::new(Mutex::new(VecDeque::new()));

        (
            PublisherImpl {
                next_id: 0,
                subscribers: subscribers.clone(),
                concurrency_limit,
            },
            subscribers,
            messages,
        )
    }
}

impl publisher::Server<capnp::text::Owned> for PublisherImpl {
    fn subscribe(
        &mut self,
        params: publisher::SubscribeParams<capnp::text::Owned>,
        mut results: publisher::SubscribeResults<capnp::text::Owned>,
    ) -> Promise<(), capnp::Error> {
        info!("new subscriber connected");
        if let Some(l) = self.concurrency_limit {
            let c = self.subscribers.borrow_mut().subscribers.len();
            if c >= l as usize {
                error!("subscriber count {} exceeded {}", c + 1, l);
                return Promise::err(capnp::Error::disconnected(
                    "exceeded concurrent subscriber limit".to_string(),
                ));
            }
        }
        self.subscribers.borrow_mut().subscribers.insert(
            self.next_id,
            SubscriberHandle {
                client: pry!(pry!(params.get()).get_subscriber()),
            },
        );

        results
            .get()
            .set_subscription(capnp_rpc::new_client(SubscriptionImpl::new(
                self.next_id,
                self.subscribers.clone(),
            )));

        self.next_id += 1;
        Promise::ok(())
    }
}

pub async fn main(opt: super::options::Server) -> Result<(), Box<dyn std::error::Error>> {
    let addr = opt
        .addr
        .to_socket_addrs()
        .unwrap()
        .next()
        .expect("could not parse address");

    tokio::task::LocalSet::new()
        .run_until(async move {
            let listener = tokio::net::TcpListener::bind(&addr).await?;
            let (publisher_impl, subscribers, messages) = PublisherImpl::new(opt.concurrency);
            let publisher: publisher::Client<_> = capnp_rpc::new_client(publisher_impl);
            info!("server listening on {:?}", addr);

            let handle_incoming = async move {
                loop {
                    let (stream, _) = listener.accept().await.unwrap();
                    stream.set_nodelay(true).unwrap();
                    let (reader, writer) =
                        tokio_util::compat::Tokio02AsyncReadCompatExt::compat(stream).split();
                    let network = twoparty::VatNetwork::new(
                        reader,
                        writer,
                        rpc_twoparty_capnp::Side::Server,
                        Default::default(),
                    );
                    let rpc_system =
                        RpcSystem::new(Box::new(network), Some(publisher.clone().client));

                    tokio::task::spawn_local(Box::pin(rpc_system.map(|_| ())));
                }
            };

            // Create messages once a second
            let messages_ref = messages.clone();
            let generate_messages = async move {
                loop {
                    let s = format!("system time is: {:?}", SystemTime::now());
                    info!("publish: {}", s);
                    messages_ref.lock().await.push_back((s, SystemTime::now()));
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            };

            // Cleanup unused messages once two seconds
            let messages_ref = messages.clone();
            let clean_up_messages = async move {
                if let Some(d) = opt.duration {
                    loop {
                        let mut cleans = Vec::new();
                        for (i, (_, t)) in messages_ref.lock().await.iter().enumerate() {
                            let elim = t.elapsed().unwrap().as_secs() > d;
                            debug!(
                                "check: {} {} => {}",
                                t.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs(),
                                t.elapsed().unwrap().as_secs(),
                                elim
                            );
                            if elim {
                                cleans.push(i);
                            }
                        }
                        for i in cleans {
                            messages_ref.lock().await.remove(i);
                        }
                        tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    }
                }
            };

            let messages_ref = messages.clone();
            let send_to_subscribers = async move {
                loop {
                    let subscribers_ref = subscribers.clone();
                    let subs = &mut subscribers_ref.borrow_mut().subscribers;
                    if !subs.is_empty() {
                        if let Some((s, ts)) = messages_ref.lock().await.pop_front() {
                            for (&idx, subscriber) in subs.iter_mut() {
                                let mut request = subscriber.client.publish_request();
                                let mut msg = request.get().init_message();
                                msg.set_send_ts(
                                    ts.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs(),
                                );
                                msg.set_content(s.as_str()).unwrap();
                                let subscribers_ref2 = subscribers.clone();
                                tokio::task::spawn_local(Box::pin(request.send().promise.map(
                                    move |r| {
                                        if let Err(e) = r {
                                            error!("Got error: {:?}. Dropping subscriber.", e);
                                            subscribers_ref2.borrow_mut().subscribers.remove(&idx);
                                        }
                                    },
                                )));
                            }
                        }
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
            };

            let _ = futures::future::join4(
                handle_incoming,
                send_to_subscribers,
                generate_messages,
                clean_up_messages,
            )
            .await;
            Ok(())
        })
        .await
}
