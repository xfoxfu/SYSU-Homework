use super::Rule;
use crate::context::Context;
use llvm_ir::{instruction::Call, Constant, Instruction, Name, Operand};

pub struct NoMalloc;

impl Rule for NoMalloc {
    fn check(f: &Instruction, context: &mut Context) -> bool {
        if let Instruction::Call(Call {
            function:
                either::Either::Right(Operand::ConstantOperand(Constant::GlobalReference {
                    name: Name::Name(name),
                    ..
                })),
            ..
        }) = f
        {
            name != "malloc" && name != "free"
        } else {
            true
        }
    }
    fn name<'a>() -> &'a str {
        "No Heap Allocation"
    }
}
