mod case;
mod layer;
mod network;
mod sigmoid;

use std::io::{BufRead, BufReader};

use case::Case;
use layer::{Context, Layer};
use network::Network;
use sigmoid::{Activate, Function};

// const TR_POSITION: &str = "dataset/train_tr.csv";
// const VA_POSITION: &str = "dataset/train_va.csv";
// const TE_DIR: &str = "dataset/train_va.csv";
const TR_POSITION: &str = "dataset/check_tr.csv";
const VA_POSITION: &str = "dataset/check_tr.csv";
const TE_DIR: &str = "dataset/check_va.csv";

fn main() -> std::io::Result<()> {
    let va_cases = BufReader::new(std::fs::File::open(VA_POSITION).unwrap())
        .lines()
        .map(|s| s.unwrap().parse::<Case>().unwrap().into_io())
        .collect::<Vec<_>>();

    let tr_cases = BufReader::new(std::fs::File::open(TR_POSITION).unwrap())
        .lines()
        .map(|s| s.unwrap().parse().unwrap());
    let mut learner = Network::new(1e-4, tr_cases);

    for e in 1..=1000 {
        learner.learn();

        let mut loss = 0f64;
        for (input, output) in va_cases.iter() {
            let output_hat = learner.guess(input);
            loss += (output_hat - output.get(0).unwrap()).powi(2);
        }
        loss /= va_cases.len() as f64;
        println!("{} {:8.2}", e, loss);
    }

    let te_cases = BufReader::new(std::fs::File::open(TE_DIR).unwrap())
        .lines()
        .map(|s| s.unwrap().parse::<Case>().unwrap().into_io())
        .collect::<Vec<_>>();
    for (input, output) in te_cases.iter() {
        let output_hat = learner.guess(input);
        println!("{:7.2} {:4}", output_hat, output.get(0).unwrap());
    }

    Ok(())
}
