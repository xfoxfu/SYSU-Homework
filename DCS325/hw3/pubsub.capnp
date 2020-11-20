@0x81abe41b8006d644;

struct Message(T) {
  sendTs @0   : UInt64;
  duration @1 : UInt64;
  content @2  : T;
}

interface Subscription {}

interface Publisher(T) {
    # A source of messages of type T.

    subscribe @0 (subscriber: Subscriber(T)) -> (subscription: Subscription);
    # Registers `subscriber` to receive published messages. Dropping the returned `subscription`
    # signals to the `Publisher` that the subscriber is no longer interested in receiving messages.
}

interface Subscriber(T) {
    publish @0 (message: Message(T)) -> ();
    # Sends a message from a publisher to the subscriber. To help with flow control, the subscriber should not
    # return from this method until it is ready to process the next message.
}
