// (Full example with detailed comments in examples/01d_quick_example.rs)
//
// This example demonstrates clap's full 'custom derive' style of creating arguments which is the
// simplest method of use, but sacrifices some flexibility.
use clap::Clap;

#[derive(Clap, Debug)]
#[clap(version = "1.0", author = "Yuze Fu <fuyz@mail2.sysu.edu.cn>")]
pub struct Opts {
    #[clap(subcommand)]
    pub subcmd: SubCommand,
}

#[derive(Clap, Debug)]
pub enum SubCommand {
    Client(Client),
    Server(Server),
}

#[derive(Clap, Debug)]
pub struct Client {
    /// target address
    pub addr: String,
}

#[derive(Clap, Debug)]
pub struct Server {
    /// maxium concurrent clients
    #[clap(short, long)]
    pub concurrency: Option<u8>,
    /// target address
    pub addr: String,
    /// message duration
    #[clap(short, long)]
    pub duration: Option<u64>,
}
