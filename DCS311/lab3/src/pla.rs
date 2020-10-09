use crate::{Case, Learner, Vector40};

pub struct PLALearner<'a> {
    cases: Vec<&'a Case>,
    w: Vector40,
    b: f64,
    eta: f64,
}

impl<'a> PLALearner<'a> {
    pub fn new(data: impl Iterator<Item = &'a Case>, eta: f64) -> Self {
        Self {
            cases: data.collect(),
            w: Vector40::zeros(),
            b: 0f64,
            eta,
        }
    }
}

impl<'a> Learner for PLALearner<'a> {
    // fn new(cases: impl Iterator<Item = &'a Case>) -> Self {
    //     Self::new(cases, 0f64)
    // }

    fn guess(&self, case: &Case) -> bool {
        case.features.dot(&self.w) + self.b >= 0f64
    }

    fn iterate(&mut self) -> bool {
        for case in self.cases.iter() {
            if self.guess(case) == case.tag {
                continue;
            }
            self.w += self.eta * case.y_pla() * case.x();
            self.b += self.eta * case.y_pla();
            return true;
        }
        false
    }

    fn print_debug(&self) {
        println!("{} {}", self.w.transpose(), self.b)
    }
}
