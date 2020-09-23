use crate::builder::Builder;
use crate::case::Case;
use crate::conf::{TEST_POSITION, TRAIN_POSITION, VALIDATION_POSITION};
use crate::selector::Selector;
use std::io::{BufRead, BufReader};
use std::str::FromStr;

fn read_data(path: &str) -> Vec<Case> {
    BufReader::new(std::fs::File::open(path).unwrap())
        .lines()
        .filter(|l| {
            if let Ok(ref s) = l {
                s != "buying,maint,dorrs,persons,lug_boot,safety,Label"
            } else {
                false
            }
        })
        .map(|l| Case::from_str(&l.unwrap()).unwrap())
        .collect()
}

pub fn run<S: Selector>() {
    let tr_data = read_data(TRAIN_POSITION);
    let va_data = read_data(VALIDATION_POSITION);
    let te_data = read_data(TEST_POSITION);

    let tree = Builder::<S>::with_data(tr_data).into_tree();

    let va_count = va_data.len();
    let mut va_accept = 0usize;
    for va in va_data.into_iter() {
        let predict = tree.traverse(&va);
        let actual = va.label;
        if predict == actual {
            va_accept += 1;
        }
    }
    println!(
        "accuracy = {} / {} = {}",
        va_accept,
        va_count,
        va_accept as f64 / va_count as f64
    );

    let va_count = te_data.len();
    let mut va_accept = 0usize;
    for va in te_data.into_iter() {
        let predict = tree.traverse(&va);
        let actual = va.label;
        if predict == actual {
            va_accept += 1;
        }
    }

    println!(
        "accuracy = {} / {} = {}",
        va_accept,
        va_count,
        va_accept as f64 / va_count as f64
    )
}
