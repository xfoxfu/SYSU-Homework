use clap::Clap;

#[derive(Clap, Debug)]
#[clap(version = "1.0", author = "Yuze Fu <fuyz@mail2.sysu.edu.cn>")]
pub struct Options {
    /// grid size
    #[clap(short = 'n', long = "size", default_value = "11")]
    pub size: u32,

    /// search depth
    #[clap(short = 'd', long = "depth", default_value = "3")]
    pub depth: u8,

    #[clap(short = 'm', long = "machine-first")]
    pub machine_first: bool,
}
