extern crate capnpc;

fn main() {
    capnpc::CompilerCommand::new()
        .file("pubsub.capnp")
        .run()
        .unwrap();
}
