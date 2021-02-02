pub use proto::dir_server_client::DirServerClient;
pub use proto::dir_server_server::{DirServer, DirServerServer};
pub use proto::{
    DirEmpty, DirReply, FileServerInfo, FileServerList, FileServerOff, FileServerOn, LockRequest,
    LockResponse, UnlockRequest,
};
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;
use tonic::{transport::Server, Request, Response, Status};

pub mod proto {
    tonic::include_proto!("distfs.dir");
}

#[derive(Debug)]
pub struct DirServicer {
    host: String,
    port: u16,
    file_servers: Mutex<HashMap<i32, FileServerInfo>>,
    locks: Mutex<HashSet<String>>,
}

impl DirServicer {
    pub fn new(host: String, port: u16) -> Self {
        info!("DirServer online");
        let file_servers = Mutex::new(HashMap::new());
        let locks = Mutex::new(HashSet::new());
        Self {
            host,
            port,
            file_servers,
            locks,
        }
    }
}

impl Drop for DirServicer {
    fn drop(&mut self) {
        info!("DirServer offline");
    }
}

#[tonic::async_trait]
impl DirServer for DirServicer {
    async fn fileserver_online(
        &self,
        request: Request<FileServerOn>,
    ) -> Result<Response<DirReply>, Status> {
        let body = request.into_inner();
        if self
            .file_servers
            .lock()
            .unwrap()
            .contains_key(&body.server_id)
        {
            return Ok(Response::new(DirReply { success: 1 }));
        }
        self.file_servers.lock().unwrap().insert(
            body.server_id,
            FileServerInfo {
                server_id: body.server_id,
                ip: body.ip,
                port: body.port,
                location: body.location,
            },
        );
        Ok(Response::new(DirReply { success: 1 }))
    }

    async fn fileserver_offline(
        &self,
        request: Request<FileServerOff>,
    ) -> Result<Response<DirReply>, Status> {
        let body = request.into_inner();
        self.file_servers.lock().unwrap().remove(&body.server_id);
        Ok(Response::new(DirReply { success: 1 }))
    }

    async fn getfileserver(
        &self,
        _: Request<DirEmpty>,
    ) -> Result<Response<FileServerList>, Status> {
        let list = self
            .file_servers
            .lock()
            .unwrap()
            .iter()
            .map(|(_, s)| s.clone())
            .collect();
        Ok(Response::new(FileServerList { server_list: list }))
    }

    async fn lockfile(&self, req: Request<LockRequest>) -> Result<Response<LockResponse>, Status> {
        let client = req.remote_addr().unwrap().to_string();
        let req = req.into_inner();
        let file_path = req.file_path;
        if self.locks.lock().unwrap().contains(&file_path) {
            return Ok(Response::new(LockResponse { status: 1 }));
        }
        info!("client {} acquired lock on '{}'", client, file_path);
        self.locks.lock().unwrap().insert(file_path);
        Ok(Response::new(LockResponse { status: 0 }))
    }

    async fn unlockfile(
        &self,
        req: Request<UnlockRequest>,
    ) -> Result<Response<LockResponse>, Status> {
        let client = req.remote_addr().unwrap().to_string();
        let req = req.into_inner();
        let file_path = req.file_path;
        if !self.locks.lock().unwrap().contains(&file_path) {
            return Ok(Response::new(LockResponse { status: 3 }));
        }
        info!("client {} released lock on '{}'", client, file_path);
        self.locks.lock().unwrap().remove(&file_path);
        Ok(Response::new(LockResponse { status: 2 }))
    }
}
