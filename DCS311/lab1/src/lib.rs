use std::{collections::BinaryHeap, collections::HashMap, str::FromStr};

pub fn safe_sub_abs(lhs: usize, rhs: usize) -> usize {
    if lhs > rhs {
        lhs - rhs
    } else {
        rhs - lhs
    }
}

pub trait Emotion: Default + FromStr<Err = ()> + PartialEq + Clone {}

#[derive(Eq, PartialEq, Clone, Hash)]
pub enum ExactEmotion {
    Anger,
    Disgust,
    Fear,
    Joy,
    Sad,
    Surprise,
}

impl Default for ExactEmotion {
    fn default() -> Self {
        ExactEmotion::Anger
    }
}
impl FromStr for ExactEmotion {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let r = match s {
            "anger" => Self::Anger,
            "disgust" => Self::Disgust,
            "fear" => Self::Fear,
            "joy" => Self::Joy,
            "sad" => Self::Sad,
            "surprise" => Self::Surprise,
            _ => return Err(()),
        };
        Ok(r)
    }
}
impl Emotion for ExactEmotion {}
impl std::fmt::Display for ExactEmotion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Anger => "anger",
            Self::Disgust => "disgust",
            Self::Fear => "fear",
            Self::Joy => "joy",
            Self::Sad => "sad",
            Self::Surprise => "surprise",
        };
        f.write_str(s)
    }
}

#[derive(PartialEq, Clone)]
pub struct ProbEmotion {
    pub anger: f64,
    pub disgust: f64,
    pub fear: f64,
    pub joy: f64,
    pub sad: f64,
    pub surprise: f64,
}

impl Default for ProbEmotion {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }
}
impl FromStr for ProbEmotion {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut iter = s.split(',');
        let anger = iter.next().and_then(|v| v.parse().ok()).ok_or(())?;
        let disgust = iter.next().and_then(|v| v.parse().ok()).ok_or(())?;
        let fear = iter.next().and_then(|v| v.parse().ok()).ok_or(())?;
        let joy = iter.next().and_then(|v| v.parse().ok()).ok_or(())?;
        let sad = iter.next().and_then(|v| v.parse().ok()).ok_or(())?;
        let surprise = iter.next().and_then(|v| v.parse().ok()).ok_or(())?;

        Ok(Self::new(anger, disgust, fear, joy, sad, surprise))
    }
}
impl Emotion for ProbEmotion {}

impl ProbEmotion {
    pub fn new(anger: f64, disgust: f64, fear: f64, joy: f64, sad: f64, surprise: f64) -> Self {
        Self {
            anger,
            disgust,
            fear,
            joy,
            sad,
            surprise,
        }
    }

    pub fn from_vec(v: Vec<f64>) -> Self {
        Self::new(v[0], v[1], v[2], v[3], v[4], v[5])
    }

    pub fn into_vec(self) -> Vec<f64> {
        vec![
            self.anger,
            self.disgust,
            self.fear,
            self.joy,
            self.sad,
            self.surprise,
        ]
    }
}

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

#[derive(PartialEq)]
pub struct DistanceObject<'a, E: Emotion>(pub usize, pub &'a E);
impl<'a, E: Emotion> std::cmp::Eq for DistanceObject<'a, E> {}
impl<'a, E: Emotion> std::cmp::PartialOrd for DistanceObject<'a, E> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl<'a, E: Emotion> std::cmp::Ord for DistanceObject<'a, E> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<'a, E: Emotion> Case<'a, E> {
    pub fn predict_emotion(
        &self,
        threshold_k: usize,
        distance_p: u32,
        train_data: &'a [Self],
    ) -> impl Iterator<Item = DistanceObject<'a, E>> {
        let mut k_minimals = BinaryHeap::<DistanceObject<E>>::new();

        for case in train_data.iter() {
            let dist = self.distance(case, distance_p);

            if let Some(DistanceObject(max_dist, _)) = k_minimals.peek() {
                if *max_dist <= dist {
                    continue;
                }
            }

            k_minimals.push(DistanceObject(dist, &case.emotion));

            if k_minimals.len() > threshold_k {
                k_minimals.pop();
            }

            assert!(k_minimals.len() <= threshold_k);
        }
        // assert!(k_minimals.len() == threshold_k);

        k_minimals.into_iter()
    }
}
