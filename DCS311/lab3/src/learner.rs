use crate::Case;

pub trait Learner {
    // TODO: find out how to perform this
    // fn new(cases: impl Iterator<Item = &'a Case>) -> Self;
    fn guess(&self, case: &Case) -> bool;
    fn iterate(&mut self) -> bool;
    fn iterate_n(&mut self, n: usize) -> usize {
        for i in 0..n {
            let r = self.iterate();
            if !r {
                return i + 1;
            }
        }
        n
    }
    fn print_debug(&self);
}
