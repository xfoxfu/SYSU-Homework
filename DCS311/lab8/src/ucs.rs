use super::{CostFunction, GenericSearch};

pub struct UniformCost;

impl CostFunction for UniformCost {
    fn h(_cur: (usize, usize), _target: (usize, usize)) -> usize {
        0
    }
}

pub type UniformCostSearch<'a> = GenericSearch<'a, UniformCost>;
