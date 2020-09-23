use crate::case::*;
use crate::conf::*;
use rand::{distributions::Uniform, prelude::Distribution};
use std::io::{BufRead, BufReader, Write};
use std::str::FromStr;

pub fn sample() {
    let mut f_all = Vec::new();
    let mut t_all = Vec::new();

    for (case, rand) in BufReader::new(std::fs::File::open(DATA_POSITION).unwrap())
        .lines()
        .filter(|l| {
            if let Ok(ref s) = l {
                s != "buying,maint,dorrs,persons,lug_boot,safety,Label"
            } else {
                false
            }
        })
        .map(|l| Case::from_str(&l.unwrap()).unwrap())
        .zip(Uniform::new(1, 6).sample_iter(rand::thread_rng()))
    {
        if case.label == Label::False {
            let pos = f_all
                .binary_search_by_key(&rand, |(_, r)| *r)
                .unwrap_or_else(|e| e);
            f_all.insert(pos, (case, rand));
        } else if case.label == Label::True {
            let pos = t_all
                .binary_search_by_key(&rand, |(_, r)| *r)
                .unwrap_or_else(|e| e);
            t_all.insert(pos, (case, rand));
        } else {
            panic!("unexpected Label::Unspecified");
        }
    }

    println!("got False = {}, True = {}", f_all.len(), t_all.len());

    let (f_tr, f_rest) = f_all.split_at(f_all.len() * 3 / 5); // 6:4
    let (f_va, f_te) = f_rest.split_at(f_rest.len() / 2); // 1:1

    let (t_tr, t_rest) = t_all.split_at(t_all.len() * 3 / 5); // 6:4
    let (t_va, t_te) = t_rest.split_at(t_rest.len() / 2); // 1:1

    let mut tr = std::fs::File::create(TRAIN_POSITION).unwrap();
    let mut va = std::fs::File::create(VALIDATION_POSITION).unwrap();
    let mut te = std::fs::File::create(TEST_POSITION).unwrap();

    writeln!(tr, "buying,maint,dorrs,persons,lug_boot,safety,Label").unwrap();
    for (c, _) in f_tr.iter().chain(t_tr.iter()) {
        writeln!(tr, "{}", c).unwrap();
    }
    println!("write TR = {} + {}", f_tr.len(), t_tr.len());
    writeln!(va, "buying,maint,dorrs,persons,lug_boot,safety,Label").unwrap();
    for (c, _) in f_va.iter().chain(t_va.iter()) {
        writeln!(va, "{}", c).unwrap();
    }
    println!("write VA = {} + {}", f_va.len(), t_va.len());
    writeln!(te, "buying,maint,dorrs,persons,lug_boot,safety,Label").unwrap();
    for (c, _) in f_te.iter().chain(t_te.iter()) {
        writeln!(te, "{}", c).unwrap();
    }
    println!("write TE = {} + {}", f_te.len(), t_te.len());
}
