use super::Selector;
use crate::case::*;

pub struct Id3Selector<'a> {
    data: &'a [Case],
    global_entropy: f64,
}

impl<'a> Selector for Id3Selector<'_> {
    fn best_fn<'f, F>(data: &[Case], fns: impl Iterator<Item = &'f F>) -> Option<&'f F>
    where
        F: Fn(&Case) -> bool,
    {
        let s = Id3Selector::new(data);
        s.best_gain_binary_fn(fns)
    }
}

impl<'a> Id3Selector<'a> {
    pub fn new(data: &'a [Case]) -> Self {
        Self {
            data,
            global_entropy: 0f64,
        }
        .global_entropy_inner()
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

    pub fn entropy(&self, v: f64) -> f64 {
        if v != 0.0 && !v.is_nan() {
            v * v.log2()
        } else {
            0.0
        }
    }

    pub fn entropy_fn<F>(&self, selector: F) -> f64
    where
        F: Fn(&Case) -> bool,
    {
        let mut sum = 0f64;

        sum += self.entropy(self.probability_cond_if(|c| c.label == Label::False, |c| selector(c)));
        sum += self.entropy(self.probability_cond_if(|c| c.label == Label::True, |c| selector(c)));
        sum += self
            .entropy(self.probability_cond_if(|c| c.label == Label::Unlabeled, |c| selector(c)));

        sum *= self.prob_if(selector);

        -sum
    }

    fn global_entropy_inner(self) -> Self {
        Self {
            data: self.data,
            global_entropy: self.entropy_fn(|_| true),
        }
    }

    pub fn global_entropy(&self) -> f64 {
        self.global_entropy
    }

    pub fn gain_binary<F>(&self, selector: F) -> f64
    where
        F: Fn(&Case) -> bool,
    {
        self.global_entropy() + self.entropy_fn(|c| selector(c)) + self.entropy_fn(|c| !selector(c))
    }

    pub fn best_gain_binary_fn<'f, F>(&self, fns: impl Iterator<Item = &'f F>) -> Option<&'f F>
    where
        F: Fn(&Case) -> bool,
    {
        fns.map(|f| (f, self.gain_binary(f)))
            .filter(|(f, _)| self.prob_if(f) > f64::EPSILON && self.prob_if(f) < 1.0 - f64::EPSILON)
            .max_by(|(_, g1), (_, g2)| (g1).partial_cmp(g2).unwrap())
            .map(|(f, _)| f)
    }
}

#[cfg(test)]
#[test]
fn basic() {
    use crate::assert_feq;
    use crate::test_utils::make_case;

    let cases = vec![
        make_case(Buying::High, Maint::High, Label::True),
        make_case(Buying::Low, Maint::High, Label::False),
        make_case(Buying::High, Maint::Low, Label::False),
        make_case(Buying::Low, Maint::Low, Label::False),
    ];
    let s = Id3Selector::new(&cases);

    assert_feq!(s.prob_if(|c| c.buying == Buying::High), 0.5);
    assert_feq!(s.prob_if(|_| true), 1.0);
    assert_feq!(s.entropy(0.5), 0.5 * 0.5f64.log2());
    assert_feq!(
        s.global_entropy(),
        -(0.25f64 * (0.25f64).log2() + 0.75f64 * (0.75f64).log2())
    );
    assert_feq!(
        s.entropy_fn(|c| c.buying == Buying::High),
        -0.5 * (s.entropy(0.5) + s.entropy(0.5))
    );
    assert_feq!(s.entropy_fn(|c| c.buying != Buying::High), 0.0);
    assert_feq!(
        s.entropy_fn(|c| c.maint == Maint::High),
        -0.5 * (s.entropy(0.5) + s.entropy(0.5))
    );
    assert_feq!(
        s.gain_binary(|c| c.buying == Buying::High),
        s.global_entropy()
            + s.entropy_fn(|c| c.buying == Buying::High)
            + s.entropy_fn(|c| c.buying != Buying::High)
    );
    assert_feq!(
        s.gain_binary(|c| c.maint == Maint::High),
        s.global_entropy()
            + s.entropy_fn(|c| c.maint == Maint::High)
            + s.entropy_fn(|c| c.maint != Maint::High)
    );
}
