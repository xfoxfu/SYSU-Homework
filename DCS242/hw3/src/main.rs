use crate::rules::Rule;
use clap::Clap;
use llvm_ir::Module;
use std::io;
use std::io::Write;
use std::path::Path;
use std::process::Command;

mod context;
use context::Context;
mod rules;

/// This doc string acts as a help message when the user runs '--help'
/// as do all doc strings on fields
#[derive(Clap)]
#[clap(version = "1.0", author = "Y. Fu <i@xfox.me>")]
struct Opts {
    /// call llvm to generate ir
    #[clap(short = "c")]
    no_call_llvm: bool,

    /// Input file name.
    input: String,

    /// A level of verbosity, and can be used multiple times
    #[clap(short, long, parse(from_occurrences))]
    verbose: i32,
}

fn compute_relative<'a>(dir: Option<&'a str>) -> &'a str {
    if let Some(dir) = dir {
        let current_dir = std::env::current_dir().unwrap();
        if let Ok(path) = Path::new(dir).strip_prefix(current_dir) {
            let ret = path.to_str().unwrap();
            if ret.is_empty() {
                "."
            } else {
                ret
            }
        } else {
            dir
        }
    } else {
        "."
    }
}

macro_rules! check {
    ($rule:ty, $i:expr, $context:expr) => {
        if <$rule>::check($i, $context) == false {
            if let Some(debug_loc) = $i.get_debug_loc() {
                println!(
                    "violation of {} in {}/{}:{}:{}",
                    <$rule>::name(),
                    compute_relative(debug_loc.directory.as_deref()),
                    debug_loc.filename,
                    debug_loc.line,
                    debug_loc.col.unwrap_or(0)
                );
            }
            false
        } else {
            true
        }
    };
}

fn check_fn(f: &llvm_ir::Function, context: &mut Context) -> bool {
    use llvm_ir::debugloc::HasDebugLoc;

    let mut ret = true;
    for b in f.basic_blocks.iter() {
        for i in b.instrs.iter() {
            ret &= check!(rules::NoGlobal, i, context);
            ret &= check!(rules::NoMalloc, i, context);
            ret &= check!(rules::NoIO, i, context);
        }
    }
    ret
}

fn main() {
    let opts: Opts = Opts::parse();

    let mut bc_path = Path::new(&opts.input).to_path_buf();

    // optionally call llvm
    if !opts.no_call_llvm {
        bc_path = bc_path.with_extension("bc");

        if bc_path.exists() {
            panic!("file {} already exists", bc_path.to_str().unwrap());
        }

        let output = Command::new("clang")
            .args(&[
                "-c",
                "-emit-llvm",
                opts.input.as_str(),
                "-g",
                "-O1",
                "-o",
                bc_path.to_str().unwrap(),
            ])
            .output()
            .expect("failed to execute process");

        println!("llvm return: {}", output.status);
        io::stdout().write_all(&output.stdout).unwrap();
        io::stderr().write_all(&output.stderr).unwrap();

        assert!(output.status.success());
    }

    use std::path::Path;

    // let path = bc_path;
    let module = Module::from_bc_path(&bc_path).expect("cannot load llvm ir");

    let mut context = context::Context::new();

    for f in module.functions.iter() {
        let ret = check_fn(&f, &mut context);
        println!("{} => {}", f.name, ret);
    }

    if !opts.no_call_llvm {
        std::fs::remove_file(bc_path);
    }
}
