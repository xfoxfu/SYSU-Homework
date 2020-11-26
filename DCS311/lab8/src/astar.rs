use super::{CostFunction, GenericSearch};

const D: usize = 1;

pub struct AStarCost;

impl CostFunction for AStarCost {
    fn h(cur: (usize, usize), target: (usize, usize)) -> usize {
        D * ((cur.0 as isize - target.0 as isize).abs() as usize
            + (cur.1 as isize - target.1 as isize).abs() as usize)
    }
}

pub type AStarSearch<'a> = GenericSearch<'a, AStarCost>;
