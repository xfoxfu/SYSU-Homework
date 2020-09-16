use crate::{safe_sub_abs, Emotion};
use std::collections::HashMap;

pub struct Case<'a, E: Emotion> {
    pub passage: HashMap<&'a str, usize>,
    pub emotion: E,
}

impl<'a, E: Emotion> Case<'a, E> {
    pub fn new(passage: &'a str) -> Self {
        let mut word_count = HashMap::new();
        for word in passage.split(' ') {
            if let Some(c) = word_count.get_mut(word) {
                *c += 1;
            } else {
                word_count.insert(word, 1);
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

    pub fn parse_from_file(file: &'a str) -> Vec<Self> {
        file.split('\n')
            .filter(|l| !l.starts_with("Words (split by space)") && !l.is_empty())
            .map(Self::from_line)
            .collect()
    }

    pub fn distance(&self, rhs: &Case<'a, E>, dist_p: u32) -> usize {
        if dist_p != 0 {
            let mut diff = 0usize;
            for w in self.passage.keys().chain(rhs.passage.keys()) {
                if self.passage.contains_key(w) && rhs.passage.contains_key(w) {
                    diff +=
                        safe_sub_abs(*self.passage.get(w).unwrap(), *rhs.passage.get(w).unwrap())
                            .pow(dist_p);
                } else if self.passage.contains_key(w) {
                    diff += self.passage.get(w).unwrap().pow(dist_p);
                } else if rhs.passage.contains_key(w) {
                    diff += rhs.passage.get(w).unwrap().pow(dist_p);
                }
            }
            diff
        } else {
            let mut p = 0;
            for w in self.passage.keys() {
                if rhs.passage.contains_key(w) {
                    p += self.passage[w] * rhs.passage[w];
                }
            }
            let q: usize = self.passage.values().map(|v| v * v).sum::<usize>()
                * rhs.passage.values().map(|v| v * v).sum::<usize>();
            ((1f64 - p as f64 / (q as f64).sqrt()) * 1_000_000f64) as usize
        }
    }
}
