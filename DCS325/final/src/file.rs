use crate::directory::{
    self,
    proto::{DirEmpty, FileServerOff, FileServerOn},
};
use crate::DirServerClient;
use futures_core::Stream;
use futures_util::{pin_mut, StreamExt};
use prost::bytes::Bytes;
pub use proto::file_server_client::FileServerClient;
pub use proto::file_server_server::{FileServer, FileServerServer};
pub use proto::{
    Chunk, DeleteRequest, DownloadRequest, Empty, ListReply, ListRequest, MkdirRequest, Reply,
    UploadRequest,
};
use std::path::PathBuf;
use std::sync::Mutex;
use std::{collections::HashMap, pin::Pin};
use tokio::fs::File;
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio_stream::iter;
use tokio_util::io::ReaderStream;
use tonic::{
    transport::{Channel, Server},
    Request, Response, Status, Streaming,
};

pub mod proto {
    tonic::include_proto!("distfs.file");
}

pub struct FileServicer {
    id: i32,
    host: String,
    port: u16,
    location: String,
    root_dir: PathBuf,
    upconn: DirServerClient<Channel>,
}

impl FileServicer {
    pub fn new(
        id: i32,
        host: String,
        port: u16,
        location: String,
        root_dir: PathBuf,
        upconn: DirServerClient<Channel>,
    ) -> Self {
        // TODO: make connection to dir server
        Self {
            id,
            host,
            port,
            location,
            root_dir,
            upconn,
        }
    }

    pub async fn online(&self) -> Result<(), Status> {
        info!("FileServer is online");
        self.upconn
            .clone()
            .fileserver_online(FileServerOn {
                server_id: self.id,
                ip: self.host.clone(),
                port: self.port.to_string(),
                location: self.location.clone(),
            })
            .await
            .map(|_| ())
    }

    pub async fn offline(&self) -> Result<(), Status> {
        info!("FileServer is offline");
        self.upconn
            .clone()
            .fileserver_offline(FileServerOff { server_id: self.id })
            .await
            .map(|_| ())
    }
}

#[tonic::async_trait]
impl FileServer for FileServicer {
    type downloadStream = std::pin::Pin<
        Box<dyn Stream<Item = Result<Chunk, Status>> + Send + Sync + Unpin + 'static>,
    >;

    async fn upload(
        &self,
        request: Request<Streaming<UploadRequest>>,
    ) -> Result<Response<Reply>, Status> {
        let shards = request
            .into_inner()
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>()?;
        let peers = self
            .upconn
            .clone()
            .getfileserver(DirEmpty { empty: 0 })
            .await?;
        for peer in peers.into_inner().server_list.iter() {
            let mut client = FileServerClient::connect(format!("http://{}:{}", peer.ip, peer.port))
                .await
                .map_err(|e| Status::internal(e.to_string()))?;
            client.upload_without_syn(iter(shards.clone())).await?;
        }

        Result::Ok(Response::new(Reply { success: 1 }))
    }
    async fn upload_without_syn(
        &self,
        request: Request<Streaming<UploadRequest>>,
    ) -> Result<Response<Reply>, Status> {
        let mut stream = request.into_inner();
        let mut path = Option::None;
        let mut file = Option::None;
        while let Some(data) = stream.next().await {
            let data = data?;
            if path.is_none() {
                path = Some(data.target_path);
                file = Some(
                    File::create(self.root_dir.join(path.as_ref().unwrap()))
                        .await
                        .map_err(|e| Status::internal(e.to_string()))?,
                );
            } else if path.as_ref().unwrap() != &data.target_path {
                return Err(Status::invalid_argument("path not consistent"));
            }
            file.as_mut()
                .unwrap()
                .write(&data.buffer)
                .await
                .map_err(|e| Status::internal(e.to_string()))?;
        }
        if path.is_none() {
            return Err(Status::invalid_argument("no request present"));
        }
        Result::Ok(Response::new(Reply { success: 1 }))
    }
    async fn download(
        &self,
        request: Request<DownloadRequest>,
    ) -> Result<Response<Self::downloadStream>, Status> {
        let file = ReaderStream::new(
            File::open(self.root_dir.join(request.into_inner().download_path))
                .await
                .map_err(|e| Status::internal(e.to_string()))?,
        );
        Ok(Response::new(Pin::new(Box::new(file.map(|r| match r {
            Ok(buffer) => Ok(Chunk {
                buffer: buffer.into_iter().collect(),
            }),
            Err(e) => Err(Status::internal(e.to_string())),
        })))))
    }
    async fn delete(&self, request: Request<DeleteRequest>) -> Result<Response<Reply>, Status> {
        let request = request.into_inner();
        let peers = self
            .upconn
            .clone()
            .getfileserver(DirEmpty { empty: 0 })
            .await?;
        for peer in peers.into_inner().server_list.iter() {
            let mut client = FileServerClient::connect(format!("http://{}:{}", peer.ip, peer.port))
                .await
                .map_err(|e| Status::internal(e.to_string()))?;
            client.delete_without_syn(request.clone()).await?;
        }

        Result::Ok(Response::new(Reply { success: 1 }))
    }
    async fn delete_without_syn(
        &self,
        request: Request<DeleteRequest>,
    ) -> Result<Response<Reply>, Status> {
        tokio::fs::remove_file(self.root_dir.join(request.into_inner().delete_path)).await?;
        Ok(Response::new(Reply { success: 1 }))
    }
    async fn list(&self, request: Request<ListRequest>) -> Result<Response<ListReply>, Status> {
        let cur_path = self.root_dir.join(request.into_inner().cur_path);
        let dir = tokio_stream::wrappers::ReadDirStream::new(
            tokio::fs::read_dir(cur_path)
                .await
                .map_err(|e| Status::internal(e.to_string()))?,
        )
        .collect::<Vec<_>>()
        .await;
        let list = dir
            .into_iter()
            .map(|e| e.map(|d| d.path().to_str().unwrap().to_owned()))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(ListReply { list }))
    }

    async fn mkdir(&self, request: Request<MkdirRequest>) -> Result<Response<Reply>, Status> {
        let dir_path = self.root_dir.join(request.into_inner().dir_path);
        tokio::fs::create_dir_all(dir_path)
            .await
            .map_err(|e| Status::internal(e.to_string()))?;
        Result::Ok(Response::new(Reply { success: 1 }))
    }
}
