extern crate capnpc;

fn main() {
    capnpc::CompilerCommand::new()
        .file("distkv.capnp")
        .run()
        .unwrap();
}
