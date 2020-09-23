use selector::{C45Selector, CartSelector, Id3Selector};

mod builder;
mod case;
mod conf;
mod run;
mod sample;
mod selector;

#[cfg(test)]
pub(crate) mod test_utils;

fn main() {
    let op = std::env::args().nth(1);
    match op.as_deref() {
        Some("sample") => sample::sample(),
        Some("id3") => run::run::<Id3Selector>(),
        Some("c45") => run::run::<C45Selector>(),
        Some("cart") => run::run::<CartSelector>(),
        _ => panic!("unknown operation, use `sample`, `id3`, `c45` or `cart`"),
    }
}
