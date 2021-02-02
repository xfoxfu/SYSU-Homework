#[macro_use]
extern crate log;

pub mod directory;
pub mod file;
// mod lock;

pub use directory::proto::dir_server_client::DirServerClient;
pub use file::proto::file_server_client::FileServerClient;
