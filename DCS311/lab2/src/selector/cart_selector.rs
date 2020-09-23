use crate::case::*;

use super::Selector;

pub struct CartSelector<'a> {
    data: &'a [Case],
}

impl Selector for CartSelector<'_> {
    fn best_fn<'f, F>(data: &[crate::case::Case], fns: impl Iterator<Item = &'f F>) -> &'f F
    where
        F: Fn(&crate::case::Case) -> bool,
    {
        CartSelector::new(data).best_gini_fn(fns)
    }
}

impl<'a> CartSelector<'a> {
    pub fn new(data: &'a [Case]) -> Self {
        Self { data }
    }

    pub fn prob_if<F>(&self, selector: F) -> f64
    where
        F: Fn(&Case) -> bool,
    {
        self.data.iter().filter(|c| selector(*c)).count() as f64 / self.data.len() as f64
    }
    pub fn probability_cond_if<F, G>(&self, selector: F, limit: G) -> f64
    where
        F: Fn(&Case) -> bool,
        G: Fn(&Case) -> bool,
    {
        self.prob_if(|c| selector(c) && limit(c)) / self.prob_if(|c| limit(c))
    }

    pub fn gini_fn(&self, cond: impl Fn(&Case) -> bool) -> f64 {
        let mut f = 1.0;
        f -= self
            .probability_cond_if(|c| c.label == Label::False, |c| cond(c))
            .powi(2);
        f -= self
            .probability_cond_if(|c| c.label == Label::True, |c| cond(c))
            .powi(2);
        f
    }

    pub fn gini_binary(&self, cond: impl Fn(&Case) -> bool) -> f64 {
        self.prob_if(|c| cond(c)) * self.gini_fn(|c| cond(c))
            + self.prob_if(|c| !cond(c)) * self.gini_fn(|c| !cond(c))
    }

    pub fn best_gini_fn<'f, F>(&self, fns: impl Iterator<Item = &'f F>) -> &'f F
    where
        F: Fn(&Case) -> bool,
    {
        fns.map(|f| (f, self.gini_binary(f)))
            .filter(|(f, _)| self.prob_if(f) > f64::EPSILON && self.prob_if(f) < 1.0 - f64::EPSILON)
            .min_by(|(_, g1), (_, g2)| (g1).partial_cmp(g2).unwrap())
            .unwrap()
            .0
    }
}
