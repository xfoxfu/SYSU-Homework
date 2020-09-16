use std::collections::{BinaryHeap, HashMap};
use std::vec::Vec;

#[derive(PartialEq, Clone)]
struct Emotion {
    pub anger: f64,
    pub disgust: f64,
    pub fear: f64,
    pub joy: f64,
    pub sad: f64,
    pub surprise: f64,
}

impl Emotion {
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

struct Case<'a> {
    pub passage: HashMap<&'a str, usize>,
    pub emotion: Emotion,
}

impl<'a> Case<'a> {
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
            emotion: Emotion {
                anger: 0f64,
                disgust: 0f64,
                fear: 0f64,
                joy: 0f64,
                sad: 0f64,
                surprise: 0f64,
            },
        }
    }

    pub fn new_with_emotion(
        passage: &'a str,
        anger: f64,
        disgust: f64,
        fear: f64,
        joy: f64,
        sad: f64,
        surprise: f64,
    ) -> Self {
        let mut s = Self::new(passage);

        s.emotion.anger = anger;
        s.emotion.disgust = disgust;
        s.emotion.fear = fear;
        s.emotion.joy = joy;
        s.emotion.sad = sad;
        s.emotion.surprise = surprise;

        s
    }

    pub fn from_line(line: &'a str) -> Self {
        let mut iter = line.split(',');
        let words = iter.next().unwrap();
        let anger = iter.next().and_then(|v| v.parse().ok()).unwrap();
        let disgust = iter.next().and_then(|v| v.parse().ok()).unwrap();
        let fear = iter.next().and_then(|v| v.parse().ok()).unwrap();
        let joy = iter.next().and_then(|v| v.parse().ok()).unwrap();
        let sad = iter.next().and_then(|v| v.parse().ok()).unwrap();
        let surprise = iter.next().and_then(|v| v.parse().ok()).unwrap();

        Self::new_with_emotion(words, anger, disgust, fear, joy, sad, surprise)
    }

    pub fn parse_from_file(file: &'a str) -> Vec<Self> {
        file.split('\n')
            .filter(|l| {
                *l != "Words (split by space),anger,disgust,fear,joy,sad,surprise" && !l.is_empty()
            })
            .map(Self::from_line)
            .collect()
    }

    pub fn distance(&self, rhs: &Case<'a>, dist_p: u32) -> usize {
        if dist_p != 0 {
            let mut diff = 0usize;
            for w in self.passage.keys().chain(rhs.passage.keys()) {
                if self.passage.contains_key(w) && rhs.passage.contains_key(w) {
                    diff +=
                        (self.passage.get(w).unwrap() - rhs.passage.get(w).unwrap()).pow(dist_p);
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

fn correlation_coefficient_single<'a, F>(
    emotions: impl Iterator<Item = (&'a Emotion, &'a Emotion)> + Clone,
    f: F,
) -> f64
where
    F: Fn(&Emotion) -> f64,
{
    let xy: f64 = emotions.clone().map(|(l, r)| f(l) * f(r)).sum();
    let xs: f64 = emotions.clone().map(|(l, _)| f(l)).sum();
    let ys: f64 = emotions.clone().map(|(_, r)| f(r)).sum();
    let x2: f64 = emotions.clone().map(|(l, _)| f(l) * f(l)).sum();
    let y2: f64 = emotions.clone().map(|(_, r)| f(r) * f(r)).sum();
    let n = emotions.count() as f64;

    (n * xy - xs * ys) / (f64::sqrt(n * x2 - xs * xs) * f64::sqrt(n * y2 - ys * ys))
}

fn correlation_coefficient<'a>(
    emotions: impl Iterator<Item = (&'a Emotion, &'a Emotion)> + Clone,
) -> f64 {
    (correlation_coefficient_single(emotions.clone(), |e| e.anger)
        + correlation_coefficient_single(emotions.clone(), |e| e.disgust)
        + correlation_coefficient_single(emotions.clone(), |e| e.fear)
        + correlation_coefficient_single(emotions.clone(), |e| e.joy)
        + correlation_coefficient_single(emotions.clone(), |e| e.sad)
        + correlation_coefficient_single(emotions, |e| e.surprise))
        / 6f64
}

#[derive(PartialEq)]
struct DistanceObject<'a>(usize, &'a Emotion);
impl<'a> std::cmp::Eq for DistanceObject<'a> {}
impl<'a> std::cmp::PartialOrd for DistanceObject<'a> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl<'a> std::cmp::Ord for DistanceObject<'a> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

fn predict_emotion(
    threshold_k: usize,
    distance_p: u32,
    train_data: &[Case],
    target_passage: &Case,
) -> Emotion {
    let mut k_minimals = BinaryHeap::<DistanceObject>::new();

    for case in train_data.iter() {
        let dist = target_passage.distance(case, distance_p);

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

    let mut emotion_vec = Vec::with_capacity(6);
    emotion_vec.resize(6, 0f64);
    for DistanceObject(cmp_dist, cmp_emotion) in k_minimals.into_iter() {
        let cmp_vec = cmp_emotion.clone().into_vec();
        for e in 0..6 {
            if cmp_dist != 0 {
                emotion_vec[e] += cmp_vec[e] / cmp_dist as f64;
            } else {
                // infinite override
                emotion_vec[e] = cmp_vec[e];
            }
        }
    }
    let sum_emotion: f64 = emotion_vec.iter().sum();
    for e in emotion_vec.iter_mut() {
        *e /= sum_emotion;
    }

    Emotion::from_vec(emotion_vec)
}

fn compute_with_kp<W: std::io::Write>(
    train_file: &str,
    validation_file: &str,
    test_file: &str,
    output_file: Option<&mut W>,
    threshold_k: usize,
    distance_p: u32,
) -> Result<f64, std::io::Error> {
    let train_data = Case::parse_from_file(train_file);

    let validation_data = Case::parse_from_file(validation_file);

    let predicted_emotions = validation_data
        .iter()
        .map(|case| predict_emotion(threshold_k, distance_p, &train_data, case))
        .collect::<Vec<_>>();

    let coefficient = correlation_coefficient(
        predicted_emotions
            .iter()
            .zip(validation_data.iter().map(|c| &c.emotion)),
    );

    println!(
        "P = {}, K = {}, Accuracy = {}",
        distance_p,
        threshold_k,
        // validation_ok_count as f64 / validation_total as f64
        coefficient
    );

    if let Some(fout) = output_file {
        // TODO:

        for line in test_file.split('\n') {
            if line == "textid,Words (split by space),anger,disgust,fear,joy,sad,surprise"
                || line.is_empty()
            {
                continue;
            }

            let id = line.split(',').next().unwrap();
            let passage = line.split(',').nth(1).unwrap();

            let case = Case::new(passage);

            let emotion = predict_emotion(threshold_k, distance_p, &train_data, &case);

            writeln!(
                fout,
                "{},{},{},{},{},{},{},{}",
                id,
                passage,
                emotion.anger,
                emotion.disgust,
                emotion.fear,
                emotion.joy,
                emotion.sad,
                emotion.surprise,
            )?;
        }
    }

    Ok(coefficient)
}

fn main() -> Result<(), std::io::Error> {
    let train_file = std::fs::read_to_string(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "./lab1_data/regression_dataset/train_set.csv".to_string()),
    )?;
    let validation_file = std::fs::read_to_string(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "./lab1_data/regression_dataset/validation_set.csv".to_string()),
    )?;
    let test_file = std::fs::read_to_string(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "./lab1_data/regression_dataset/test_set.csv".to_string()),
    )?;

    let mut best_ks = BinaryHeap::with_capacity(100);
    for p in 0..5 {
        for k in 1..20 {
            let accuracy = compute_with_kp::<std::fs::File>(
                &train_file,
                &validation_file,
                &test_file,
                None,
                k,
                p,
            )?;
            best_ks.push(((accuracy * 1e10f64) as usize, k, p));
        }
    }

    let (_accuracy, best_k, best_p) = best_ks.peek().cloned().unwrap();

    let mut output = std::fs::File::create("17341039_FUYuze_KNN_regression.csv")?;
    compute_with_kp(
        &train_file,
        &validation_file,
        &test_file,
        Some(&mut output),
        best_k,
        best_p,
    )?;

    Ok(())
}
