# Async-unsafe detector

DCS242 Parallel Computing Homework 3

## How to compile

First, a LLVM binary must be accessible from current path. If not, LLVM path can be provided as follows:

```bash
export LLVM_SYS_90_PREFIX=/usr/local/opt/llvm
```

Then, a simple cargo build command can be executed:

```bash
cargo build
```

## How to use

A help can be retrieved:

```bash
cargo run -- --help
```

Normal use would be:

```bash
cargo run -- test.c
```

Where `test.c` is the filename.
