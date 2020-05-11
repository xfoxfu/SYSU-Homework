use super::Rule;
use crate::context::Context;
use llvm_ir::{Constant, Instruction, Operand};

pub struct NoGlobal;

pub trait OperandCheck {
    fn is_global(&self) -> bool;
}

impl OperandCheck for Operand {
    fn is_global(&self) -> bool {
        if let Operand::ConstantOperand(Constant::GlobalReference { .. }) = self {
            true
        } else {
            false
        }
    }
}

impl OperandCheck for Vec<Operand> {
    fn is_global(&self) -> bool {
        for op in self.iter() {
            if op.is_global() {
                return true;
            }
        }
        false
    }
}

impl OperandCheck for Vec<(llvm_ir::operand::Operand, llvm_ir::name::Name)> {
    fn is_global(&self) -> bool {
        for (op, _) in self.iter() {
            if op.is_global() {
                return true;
            }
        }
        false
    }
}

impl OperandCheck
    for either::Either<llvm_ir::instruction::InlineAssembly, llvm_ir::operand::Operand>
{
    fn is_global(&self) -> bool {
        false
        // match self {
        //     either::Either::Left(_) => false, // this is not implemented by upstream, so just ignore
        //     either::Either::Right(op) => {
        //         if let Operand::ConstantOperand(Constant::GlobalReference {
        //             name: llvm_ir::Name::Name(name),
        //             ..
        //         }) = op
        //         {
        //             for safe_name in names.iter() {
        //                 if safe_name == name {
        //                     return false;
        //                 }
        //             }
        //             return true;
        //         }
        //         false
        //     }
        // }
    }
}

macro_rules! check_inst_match {
    ($out:ident, $($inst:ident => $($operand:ident)|*),*) => {
        match $out {
            $(llvm_ir::Instruction::$inst(llvm_ir::instruction::$inst { $($operand,)* .. }) =>
                !($($operand.is_global()||)* false)),*
        }
    }
}

impl Rule for NoGlobal {
    fn check(f: &Instruction, context: &mut Context) -> bool {
        check_inst_match!(f,
            Add => operand0 | operand1,
            Sub => operand0 | operand1,
            Mul => operand0 | operand1,
            UDiv => operand0 | operand1,
            SDiv => operand0 | operand1,
            URem => operand0 | operand1,
            SRem => operand0 | operand1,
            And => operand0 | operand1,
            Or => operand0 | operand1,
            Xor => operand0 | operand1,
            Shl => operand0 | operand1,
            LShr => operand0 | operand1,
            AShr => operand0 | operand1,
            FAdd => operand0 | operand1,
            FSub => operand0 | operand1,
            FMul => operand0 | operand1,
            FDiv => operand0 | operand1,
            FRem => operand0 | operand1,
            FNeg => operand,
            ExtractElement => vector | index,
            InsertElement => vector | element | index,
            ShuffleVector => operand0 | operand1,
            ExtractValue => aggregate,
            InsertValue => aggregate | element,
            Alloca => num_elements,
            Load => address,
            Store => address | value,
            Fence => ,
            CmpXchg => address | expected | replacement,
            AtomicRMW => address | value,
            GetElementPtr => address | indices,
            Trunc => operand,
            ZExt => operand,
            SExt => operand,
            FPTrunc => operand,
            FPExt => operand,
            FPToUI => operand,
            FPToSI => operand,
            UIToFP => operand,
            SIToFP => operand,
            PtrToInt => operand,
            IntToPtr => operand,
            BitCast => operand,
            AddrSpaceCast => operand,
            ICmp => operand0 | operand1,
            FCmp => operand0 | operand1,
            Phi => incoming_values,
            Select => condition | true_value | false_value,
            Call => function,
            VAArg => arg_list,
            LandingPad => ,
            CatchPad => catch_switch | args,
            CleanupPad => parent_pad | args
        )
    }
    fn name<'a>() -> &'a str {
        "No Global Variable Reference"
    }
}
