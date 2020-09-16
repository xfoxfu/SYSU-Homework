use lab1::ProbEmotion as Emotion;
use lab1::{Case, DistanceObject};
use std::collections::BinaryHeap;
use std::vec::Vec;

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

fn predict_emotion(
    threshold_k: usize,
    distance_p: u32,
    train_data: &[Case<Emotion>],
    target_passage: &Case<Emotion>,
) -> Emotion {
    let k_minimals = target_passage.predict_emotion(threshold_k, distance_p, train_data);

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
    train_data: &[Case<Emotion>],
    validation_data: &[Case<Emotion>],
    test_file: &str,
    output_file: Option<&mut W>,
    threshold_k: usize,
    distance_p: u32,
) -> Result<f64, std::io::Error> {
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
        distance_p, threshold_k, coefficient
    );

    if let Some(fout) = output_file {
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

    let train_data = Case::<Emotion>::parse_from_file(&train_file);
    let validation_data = Case::<Emotion>::parse_from_file(&validation_file);

    let mut best_ks = BinaryHeap::with_capacity(100);
    for p in 0..5 {
        for k in 1..20 {
            let accuracy = compute_with_kp::<std::fs::File>(
                &train_data,
                &validation_data,
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
        &train_data,
        &validation_data,
        &test_file,
        Some(&mut output),
        best_k,
        best_p,
    )?;

    Ok(())
}
