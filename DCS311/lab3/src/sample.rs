use crate::conf::*;
use rand::{distributions::Uniform, prelude::Distribution};
use std::io::{BufRead, BufReader, Write};

pub fn sample() {
    let mut f_all = Vec::new();
    let mut t_all = Vec::new();

    // 读入原始数据，并且为每个数据关联一个随机数
    for (case, rand) in BufReader::new(std::fs::File::open(DATA_POSITION).unwrap())
        .lines()
        .map(|s| s.unwrap())
        .zip(Uniform::new(1, 6).sample_iter(rand::thread_rng()))
    {
        // 将数据集进行分层
        if case.trim().ends_with('0') {
            // Label 为 0 的部分，进行有序插入
            let pos = f_all
                .binary_search_by_key(&rand, |(_, r)| *r)
                .unwrap_or_else(|e| e);
            f_all.insert(pos, (case, rand));
        } else if case.trim().ends_with('1') {
            // Label 为 1 的部分，进行有序插入
            let pos = t_all
                .binary_search_by_key(&rand, |(_, r)| *r)
                .unwrap_or_else(|e| e);
            t_all.insert(pos, (case, rand));
        } else {
            panic!("unexpected Label::Unspecified");
        }
    }

    println!("got False = {}, True = {}", f_all.len(), t_all.len());

    // 划分训练集、验证集、测试集
    let (f_tr, f_rest) = f_all.split_at(f_all.len() * 3 / 5); // 6:4
    let (f_va, f_te) = f_rest.split_at(f_rest.len() / 2); // 1:1

    let (t_tr, t_rest) = t_all.split_at(t_all.len() * 3 / 5); // 6:4
    let (t_va, t_te) = t_rest.split_at(t_rest.len() / 2); // 1:1

    // 输出结果
    let mut tr = std::fs::File::create(TRAIN_POSITION).unwrap();
    let mut va = std::fs::File::create(VALIDATION_POSITION).unwrap();
    let mut te = std::fs::File::create(TEST_POSITION).unwrap();

    for (c, _) in f_tr.iter().chain(t_tr.iter()) {
        writeln!(tr, "{}", c).unwrap();
    }
    println!("write TR = {} + {}", f_tr.len(), t_tr.len());
    for (c, _) in f_va.iter().chain(t_va.iter()) {
        writeln!(va, "{}", c).unwrap();
    }
    println!("write VA = {} + {}", f_va.len(), t_va.len());
    for (c, _) in f_te.iter().chain(t_te.iter()) {
        writeln!(te, "{}", c).unwrap();
    }
    println!("write TE = {} + {}", f_te.len(), t_te.len());
}
