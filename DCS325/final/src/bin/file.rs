use distfs::directory::DirServerClient;
use distfs::file::{FileServerServer, FileServicer};
use tonic::transport::Server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if let Err(std::env::VarError::NotPresent) = std::env::var("RUST_LOG") {
        std::env::set_var("RUST_LOG", "distfs=info")
    }
    pretty_env_logger::init();

    let id: i32 = std::env::args().nth(1).unwrap().parse().unwrap();
    let port: u16 = std::env::args().nth(2).unwrap().parse().unwrap();
    let location: String = std::env::args().nth(3).unwrap().parse().unwrap();
    let root_dir: std::path::PathBuf = std::env::args().nth(4).unwrap().parse().unwrap();
    let upstream: String = std::env::args().nth(5).unwrap().parse().unwrap();

    let addr = format!("[::1]:{}", port).parse().unwrap();

    let upconn = DirServerClient::connect(upstream).await.unwrap();

    let fis = FileServicer::new(id, "localhost".to_owned(), port, location, root_dir, upconn);
    fis.online().await.unwrap();
    let svc = FileServerServer::new(fis);

    Server::builder().add_service(svc).serve(addr).await?;

    Ok(())
}
