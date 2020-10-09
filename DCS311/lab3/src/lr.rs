use crate::{Case, Learner, Vector40, Vector41};

fn pi(w: &Vector41, x: &Vector40) -> f64 {
    1.0 / (1.0 + (-w.dot(&x.insert_fixed_rows::<nalgebra::U1>(40, 1.0))).exp())
}

pub struct LRLearner<'a> {
    cases: Vec<&'a Case>,
    w: Vector41,
    eta: f64,
}

impl<'a> LRLearner<'a> {
    pub fn new(data: impl Iterator<Item = &'a Case>, eta: f64) -> Self {
        Self {
            cases: data.collect(),
            w: Vector41::zeros(),
            eta,
        }
    }

    pub fn pi(&self, x: &Vector40) -> f64 {
        pi(&self.w, x)
    }
}

impl<'a> Learner for LRLearner<'a> {
    fn guess(&self, case: &Case) -> bool {
        self.pi(&case.features) >= 0.5
    }

    fn iterate(&mut self) -> bool {
        for case in self.cases.iter() {
            self.w += self.eta
                * (case.y_lr() - self.pi(case.x()))
                * case.x().insert_fixed_rows::<nalgebra::U1>(40, 1.0);
        }
        true
    }

    fn print_debug(&self) {
        println!("{}", self.w.transpose())
    }
}
