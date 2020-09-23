mod c45_selector;
mod cart_selector;
mod id3_selector;

pub use c45_selector::C45Selector;
pub use cart_selector::CartSelector;
pub use id3_selector::Id3Selector;

use crate::case::*;

pub trait Selector {
    // TODO: figure out how to perform this
    // fn new(data: &[Case]) -> Self;
    fn best_fn<'f, F>(data: &[Case], fns: impl Iterator<Item = &'f F>) -> &'f F
    where
        F: Fn(&Case) -> bool;
}
