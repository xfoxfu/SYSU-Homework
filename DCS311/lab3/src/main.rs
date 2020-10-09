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

            let mut learner = $s::new(tr_cases.iter(), 1.0);
            for k in 0..10 {
                learner.iterate();
                println!("##### {:2} #####", k);
                for (i, case) in tr_cases.iter().enumerate() {
                    println!("{} => {}", i, learner.guess(case));
                }
            }

            Ok(())
        }
    };
}

main_fn!(main_learner_pla, PLALearner);
main_fn!(main_learner_lr, LRLearner);
