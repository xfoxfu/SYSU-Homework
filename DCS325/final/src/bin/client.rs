#[macro_use]
extern crate log;

use anyhow::Result;
use distfs::directory::{DirEmpty, FileServerList, LockRequest, UnlockRequest};
use distfs::file::{
    Chunk, DeleteRequest, DownloadRequest, ListRequest, MkdirRequest, UploadRequest,
};
use distfs::{DirServerClient, FileServerClient};
use rand::Rng;
use std::io::BufRead;
use std::path::PathBuf;
use std::time::SystemTime;
use std::{collections::HashMap, path::Path};
use std::{convert::TryInto, str::FromStr};
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio_stream::StreamExt;
use tokio_util::io::ReaderStream;
use tonic::{transport::Channel, Status};

struct Client {
    root: PathBuf,
    cwd: PathBuf,
    buffer: HashMap<String, (SystemTime, Vec<u8>)>,
    ds: Option<DirServerClient<Channel>>,
    fs: Option<FileServerClient<Channel>>,
}

impl Client {
    pub fn new(root: PathBuf) -> Self {
        let cwd = PathBuf::new();
        let buffer = HashMap::new();
        let ds: Option<DirServerClient<Channel>> = Option::None;
        let fs: Option<FileServerClient<Channel>> = Option::None;
        Self {
            root,
            cwd,
            buffer,
            fs,
            ds,
        }
    }

    pub async fn connect(&mut self, target: impl Into<String>) -> anyhow::Result<()> {
        let mut ds = DirServerClient::connect(target.into()).await?;
        let fss = ds
            .getfileserver(DirEmpty { empty: 0 })
            .await?
            .into_inner()
            .server_list;
        let sid: usize = rand::thread_rng().gen_range(0..fss.len());
        info!("selected {} out of {}", sid, fss.len());
        let fs = &fss[sid];
        let target = format!("http://{}:{}", fs.ip, fs.port);
        info!("connecting to {}", target);
        self.ds = Some(ds);
        self.fs = Some(FileServerClient::connect(target.clone()).await?);
        info!("connected to {}", target);
        Ok(())
    }

    pub fn cwd(&self) -> String {
        self.cwd.clone().into_os_string().into_string().unwrap()
    }

    pub fn normalize_path(&self, path: impl AsRef<std::path::Path>) -> PathBuf {
        self.root.join(&self.cwd).join(path)
    }

    pub fn upstream_path(&self, path: impl AsRef<std::path::Path>) -> PathBuf {
        self.cwd.join(path)
    }

    pub async fn upload(&mut self, path: PathBuf) -> anyhow::Result<()> {
        self.lock(&path).await?;
        let mut buffer = Vec::new();
        File::open(self.normalize_path(&path))
            .await?
            .read_to_end(&mut buffer)
            .await?;
        let target_path = self
            .upstream_path(&path)
            .into_os_string()
            .into_string()
            .unwrap();
        self.fs
            .as_mut()
            .unwrap()
            .upload(tokio_stream::once(UploadRequest {
                target_path,
                buffer,
            }))
            .await?;

        self.unlock(&path).await?;
        Ok(())
    }

    pub async fn download(&mut self, path: PathBuf) -> anyhow::Result<()> {
        self.lock(&path).await?;
        let download_path = self
            .upstream_path(&path)
            .into_os_string()
            .into_string()
            .unwrap();
        let mut file = File::create(self.normalize_path(&path)).await?;
        let mut stream = self
            .fs
            .as_mut()
            .unwrap()
            .download(DownloadRequest { download_path })
            .await?
            .into_inner();
        while let Some(chunk) = stream.message().await? {
            file.write(&chunk.buffer).await?;
        }
        self.unlock(&path).await?;
        Ok(())
    }

    pub async fn delete(&mut self, path: PathBuf) -> anyhow::Result<()> {
        self.lock(&path).await?;
        let delete_path = self
            .upstream_path(&path)
            .into_os_string()
            .into_string()
            .unwrap();
        self.fs
            .as_mut()
            .unwrap()
            .delete(DeleteRequest { delete_path })
            .await?
            .into_inner();
        self.unlock(&path).await?;
        Ok(())
    }

    pub async fn mkdir(&mut self, path: PathBuf) -> anyhow::Result<()> {
        self.lock(&path).await?;
        let dir_path = self
            .upstream_path(&path)
            .into_os_string()
            .into_string()
            .unwrap();
        self.fs
            .as_mut()
            .unwrap()
            .mkdir(MkdirRequest { dir_path })
            .await?
            .into_inner();
        self.unlock(&path).await?;
        Ok(())
    }

    pub async fn list(&mut self, path: PathBuf) -> anyhow::Result<()> {
        self.lock(&path).await?;
        let cur_path = self
            .upstream_path(&path)
            .into_os_string()
            .into_string()
            .unwrap();
        let entries = self
            .fs
            .as_mut()
            .unwrap()
            .list(ListRequest { cur_path })
            .await?
            .into_inner();
        self.unlock(&path).await?;
        for entry in entries.list {
            println!("- {}", entry);
        }
        Ok(())
    }

    pub fn cd(&mut self, path: PathBuf) -> anyhow::Result<()> {
        self.cwd = self.cwd.join(path);
        Ok(())
    }

    pub async fn lock(&mut self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let file_path = self.upstream_path(&path).to_str().unwrap().to_string();
        if self
            .ds
            .as_mut()
            .unwrap()
            .lockfile(LockRequest { file_path })
            .await?
            .into_inner()
            .status
            != 0
        {
            return Err(anyhow::Error::msg("cannot acquire lock"));
        }
        Ok(())
    }

    pub async fn unlock(&mut self, path: impl AsRef<Path>) -> anyhow::Result<()> {
        let file_path = self.upstream_path(&path).to_str().unwrap().to_string();
        self.ds
            .as_mut()
            .unwrap()
            .unlockfile(UnlockRequest { file_path })
            .await?;
        Ok(())
    }

    pub fn print_help(&self) {
        println!("-------------------------------------------");
        println!("The available commands are as follows：");
        println!("ls: list file directories");
        println!("upload: upload files");
        println!("download: download files");
        println!("pwd: current access server path");
        println!("delete: delete files");
        println!("mkdir: create folder");
        println!("cd: change current path");
        println!("------------------------------------------");
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if let Err(std::env::VarError::NotPresent) = std::env::var("RUST_LOG") {
        std::env::set_var("RUST_LOG", "distfs=info,client=info")
    }
    pretty_env_logger::init();

    let root = std::env::args().nth(1).unwrap();
    let ds_loc = std::env::args().nth(2).unwrap();
    let mut client = Client::new(root.into());

    client.connect(ds_loc).await?;

    client.print_help();
    for line in std::io::stdin().lock().lines() {
        let mut line = line?;
        let (comm, path) = line.split_at(line.find(' ').unwrap_or_else(|| line.len()));
        let path = path.trim();
        match comm {
            "ls" => {
                client.list(PathBuf::from_str(&path)?).await?;
            }
            "upload" => {
                client.upload(PathBuf::from_str(&path)?).await?;
            }
            "download" => {
                client.download(PathBuf::from_str(&path)?).await?;
            }
            "pwd" => {
                println!("{}", client.cwd());
            }
            "delete" => {
                client.delete(PathBuf::from_str(&path)?).await?;
            }
            "mkdir" => {
                client.mkdir(PathBuf::from_str(&path)?).await?;
            }
            "cd" => {
                client.cd(PathBuf::from_str(&path)?)?;
            }
            _ => println!("invalid command"),
        }
        client.print_help();
    }

    Ok(())
}
