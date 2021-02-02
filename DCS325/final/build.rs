fn main() -> std::io::Result<()> {
    tonic_build::compile_protos("proto/DirectoryServer.proto")?;
    tonic_build::compile_protos("proto/FileServer.proto")?;
    Ok(())
}
