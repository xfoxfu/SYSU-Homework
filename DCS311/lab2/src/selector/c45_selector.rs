#![allow(dead_code)]

use super::{Id3Selector, Selector};
use crate::case::*;

pub struct C45Selector<'a> {
    pub id3: Id3Selector<'a>,
}

impl Selector for C45Selector<'_> {
    fn best_fn<'f, F>(data: &[Case], fns: impl Iterator<Item = &'f F>) -> Option<&'f F>
    where
        F: Fn(&Case) -> bool,
    {
        let s = C45Selector::new(data);
        s.best_ratio_binary_fn(fns)
    }
}

impl<'a> C45Selector<'a> {
    pub fn with_id3(id3: Id3Selector<'a>) -> Self {
        Self { id3 }
    }

    pub fn new(data: &'a [Case]) -> Self {
        Self::with_id3(Id3Selector::new(data))
    }
}

impl<'a> C45Selector<'a> {
    pub fn prob_if<F>(&self, selector: F) -> f64
    where
        F: Fn(&Case) -> bool,
    {
        self.id3.prob_if(selector)
    }
    pub fn probability_cond_if<F, G>(&self, selector: F, limit: G) -> f64
    where
        F: Fn(&Case) -> bool,
        G: Fn(&Case) -> bool,
    {
        self.id3.probability_cond_if(selector, limit)
    }

    pub fn entropy(&self, v: f64) -> f64 {
        self.id3.entropy(v)
    }

    pub fn entropy_fn<F>(&self, selector: F) -> f64
    where
        F: Fn(&Case) -> bool,
    {
        self.id3.entropy_fn(selector)
    }

    pub fn global_entropy(&self) -> f64 {
        self.id3.global_entropy()
    }

    pub fn gain_binary<F>(&self, selector: F) -> f64
    where
        F: Fn(&Case) -> bool,
    {
        self.id3.gain_binary(selector)
    }

    fn best_gain_fn<'f, F>(&self, fns: impl Iterator<Item = &'f F>) -> Option<&'f F>
    where
        F: Fn(&Case) -> bool,
    {
        self.id3.best_gain_binary_fn(fns)
    }
}

impl<'a> C45Selector<'a> {
    pub fn split_info_binary_fn<F>(&self, selector: F) -> f64
    where
        F: Fn(&Case) -> bool,
    {
        self.entropy_fn(|c| selector(c)) + self.entropy_fn(|c| !selector(c))
    }

    pub fn gain_ratio_fn<F>(&self, selector: F) -> f64
    where
        F: Fn(&Case) -> bool,
    {
        self.gain_binary(|c| selector(c)) / self.split_info_binary_fn(|c| selector(c))
    }

    pub fn best_ratio_binary_fn<'f, F>(&self, fns: impl Iterator<Item = &'f F>) -> Option<&'f F>
    where
        F: Fn(&Case) -> bool,
    {
        fns.map(|f| (f, self.gain_ratio_fn(f)))
            .filter(|(f, _)| self.prob_if(f) > f64::EPSILON && self.prob_if(f) < 1.0 - f64::EPSILON)
            .max_by(|(_, g1), (_, g2)| g1.partial_cmp(g2).unwrap())
            .map(|(f, _)| f)
    }
}

#[cfg(test)]
#[test]
fn test() {
    use crate::{assert_feq, test_utils::make_case};

    let cases = vec![
        make_case(Buying::High, Maint::High, Label::True),
        make_case(Buying::Low, Maint::High, Label::False),
        make_case(Buying::High, Maint::Low, Label::False),
        make_case(Buying::Low, Maint::Low, Label::False),
    ];
    let s = C45Selector::new(&cases);

    assert_feq!(
        s.split_info_binary_fn(|c| c.buying == Buying::High),
        -0.5 * s.id3.entropy(0.5) - 0.5 * s.id3.entropy(0.5)
    );
    assert_feq!(
        s.gain_ratio_fn(|c| c.buying == Buying::High),
        s.id3.gain_binary(|c| c.buying == Buying::High)
            / s.split_info_binary_fn(|c| c.buying == Buying::High)
    );
}
