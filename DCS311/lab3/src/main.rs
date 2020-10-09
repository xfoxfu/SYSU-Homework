use nalgebra::VectorN;
use std::io::{BufRead, BufReader};

mod case;
mod conf;
mod learner;
mod lr;
mod pla;
mod sample;

use case::Case;
use conf::*;
use learner::Learner;
use lr::LRLearner;
use pla::PLALearner;

type Vector40 = VectorN<f64, nalgebra::U40>;
type Vector41 = VectorN<f64, nalgebra::U41>;

fn main() {
    let op = std::env::args().nth(1);
    match op.as_deref() {
        Some("sample") => sample::sample(),
        Some("pla") => main_learner_pla().unwrap(),
        Some("lr") => main_learner_lr().unwrap(),
        _ => panic!("unknown operation, use `sample`, `pla` or `lr`"),
    }
}

macro_rules! main_fn {
    ($name:ident, $s:ident) => {
        fn $name() -> std::io::Result<()> {
            let tr_cases = BufReader::new(std::fs::File::open(DATA_POSITION).unwrap())
                .lines()
                .map(|s| s.unwrap().parse().unwrap())
                .collect::<Vec<_>>();
            let va_cases = BufReader::new(std::fs::File::open(VALIDATION_POSITION).unwrap())
                .lines()
                .map(|s| s.unwrap().parse().unwrap())
                .collect::<Vec<_>>();

            for eta_i in (10..=1000).step_by(10) {
                let eta = eta_i as f64 * 1e-8;
                for threshold in 0..50 {
                    let mut learner = $s::new(tr_cases.iter(), eta);
                    learner.iterate_n(threshold);

                    let va_total = va_cases.len();
                    let mut va_correct = 0;
                    for case in va_cases.iter() {
                        if learner.guess(case) == case.tag {
                            va_correct += 1;
                        }
                    }
                    println!(
                        "eta = {}, threshold = {}, correct rate = {:2.2}% ({:4}/{:4})",
                        eta,
                        threshold,
                        100.0 * va_correct as f64 / va_total as f64,
                        va_correct,
                        va_total
                    );
                }
            }

            Ok(())
        }
    };
}

main_fn!(main_learner_pla, PLALearner);
main_fn!(main_learner_lr, LRLearner);
