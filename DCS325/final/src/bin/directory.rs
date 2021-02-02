use distfs::directory::{DirServerServer, DirServicer};
use tonic::transport::Server;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    if let Err(std::env::VarError::NotPresent) = std::env::var("RUST_LOG") {
        std::env::set_var("RUST_LOG", "distfs=info")
    }
    pretty_env_logger::init();

    let addr = "[::1]:10080".parse().unwrap();

    let dir = DirServicer::new("localhost".to_owned(), 10080);
    let svc = DirServerServer::new(dir);

    Server::builder().add_service(svc).serve(addr).await?;

    Ok(())
}
