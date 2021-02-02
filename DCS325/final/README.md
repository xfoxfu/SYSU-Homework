# distfs

## 编译方法

```bash
cargo build --release
```

## 启动方法

系统要求：

```bash
mkdir -p misc/client
mkdir -p misc/fs1
mkdir -p misc/fs2
```

分别用作客户端和两个存储节点的工作目录。

服务：

```bash
goreman start
```

（需要系统安装有 `goreman`）

客户端：

```bash
cargo run --bin client misc/client http://localhost:10080
```
