use crate::Emotion;
use std::collections::HashMap;

pub struct Case<'a, E: Emotion> {
    pub passage: HashMap<&'a str, f64>,
    pub emotion: E,
}

impl<'a, E: Emotion> Case<'a, E> {
    pub fn new(passage: &'a str) -> Self {
        let mut word_count = HashMap::new();
        // 对于每个单词
        for word in passage.split(' ') {
            if let Some(c) = word_count.get_mut(word) {
                // 如果包含计数信息，则增加 1
                *c += 1.0;
            } else {
                // 否则，插入计数信息，数值为 1
                word_count.insert(word, 1.0);
            }
        }

        Self {
            passage: word_count,
            emotion: E::default(),
        }
    }

    pub fn new_with_emotion(passage: &'a str, emotion: E) -> Self {
        Self {
            passage: Self::new(passage).passage,
            emotion,
        }
    }

    pub fn from_line(line: &'a str) -> Self {
        let first_comma = line.find(',').unwrap();
        let (words, emotion) = line.split_at(first_comma);
        let (_, emotion) = emotion.split_at(1);

        Self::new_with_emotion(words, emotion.parse().unwrap())
    }

    pub fn from_line_test(line: &'a str) -> (usize, Self) {
        let comma1 = line.find(',').unwrap();
        let (id, line) = line.split_at(comma1);
        let (_, line) = line.split_at(1);
        let comma2 = line.find(',').unwrap();
        let (words, emotion) = line.split_at(comma2);
        let (_, _) = emotion.split_at(1);

        (
            id.parse().unwrap(),
            Self::new_with_emotion(words, E::default()),
        )
    }

    pub fn parse_from_file(file: &'a str) -> Vec<Self> {
        file.split('\n')
            .filter(|l| !l.contains("Words (split by space)") && !l.is_empty())
            .map(Self::from_line)
            .collect()
    }

    pub fn parse_from_file_test(file: &'a str) -> Vec<(usize, Self)> {
        file.split('\n')
            .filter(|l| !l.contains("Words (split by space)") && !l.is_empty())
            .map(Self::from_line_test)
            .collect()
    }

    pub fn distance(&self, rhs: &Case<'a, E>, dist_p: u32) -> f64 {
        let p = dist_p as i32;
        if dist_p != 0 {
            let mut diff = 0.0; // 差异和
            for w in self.passage.keys().chain(rhs.passage.keys()) {
                if self.passage.contains_key(w) && rhs.passage.contains_key(w) {
                    // 在左右中均存在，则计算 |a_i - b_i|^p
                    diff += (*self.passage.get(w).unwrap() - *rhs.passage.get(w).unwrap())
                        .abs()
                        .powi(p);
                } else if self.passage.contains_key(w) {
                    // 否则，直接计算 |a_i|^p
                    diff += self.passage.get(w).unwrap().abs().powi(p);
                } else if rhs.passage.contains_key(w) {
                    // 否则，直接计算 |b_i|^p
                    diff += rhs.passage.get(w).unwrap().abs().powi(p);
                }
            }
            diff.powf(1.0 / p as f64)
        } else {
            let mut p = 0.0; // 内积
            for w in self.passage.keys() {
                if rhs.passage.contains_key(w) {
                    p += self.passage[w] * rhs.passage[w];
                }
            }
            let q = self.passage.values().map(|v| v * v).sum::<f64>() // a 的模长
                * rhs.passage.values().map(|v| v * v).sum::<f64>(); // b 的模长
            1f64 - p / (q as f64).sqrt()
        }
    }
}
