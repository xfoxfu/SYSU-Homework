use super::context::Context;
use llvm_ir::{Function, HasDebugLoc, Instruction};

pub trait Rule {
    fn name<'a>() -> &'a str;
    fn check(f: &Instruction, context: &mut Context) -> bool;
}

mod no_global;
pub use no_global::NoGlobal;

mod no_malloc;
pub use no_malloc::NoMalloc;

mod no_io;
pub use no_io::NoIO;
