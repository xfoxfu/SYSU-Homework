use lab1::{Case, DistanceObject, ExactEmotion as Emotion};
use std::collections::{BinaryHeap, HashMap};

fn predict_emotion(
    threshold_k: usize,
    distance_p: u32,
    train_data: &[Case<Emotion>],
    target_passage: &Case<Emotion>,
) -> Emotion {
    let k_minimals = target_passage.predict_emotion(threshold_k, distance_p, train_data);

    let mut assume_emotions = HashMap::new();
    for DistanceObject(_dist, emotion) in k_minimals.into_iter() {
        if assume_emotions.contains_key(emotion) {
            *assume_emotions.get_mut(emotion).unwrap() += 1;
        } else {
            assume_emotions.insert(emotion, 1);
        }
    }

    let (emotion, _) = assume_emotions.into_iter().max_by_key(|(_, c)| *c).unwrap();

    emotion.clone()
}

fn compute_with_kp<W: std::io::Write>(
    train_data: &[Case<Emotion>],
    validation_data: &[Case<Emotion>],
    test_file: &str,
    output_file: Option<&mut W>,
    threshold_k: usize,
    distance_p: u32,
) -> Result<f64, std::io::Error> {
    let validation_total = validation_data.len();
    let mut validation_ok_count = 0usize;

    for target_passage in validation_data.iter() {
        let emotion = predict_emotion(threshold_k, distance_p, &train_data, &target_passage);

        if emotion == target_passage.emotion {
            validation_ok_count += 1;
        }
    }

    println!(
        "P = {}, K = {}, Accuracy = {}",
        distance_p,
        threshold_k,
        validation_ok_count as f64 / validation_total as f64
    );

    if let Some(fout) = output_file {
        for line in test_file.split('\n') {
            if line == "textid,Words (split by space),label" || line.is_empty() {
                continue;
            }

            let id = line.split(',').next().unwrap();
            let passage = line.split(',').nth(1).unwrap();

            let case = Case::new(passage);

            let emotion = predict_emotion(threshold_k, distance_p, &train_data, &case);

            writeln!(
                fout,
                "{},{},{}",
                id,
                line.split(',').nth(1).unwrap(),
                emotion
            )?;
        }
    }

    Ok(validation_ok_count as f64 / validation_total as f64)
}

fn main() -> Result<(), std::io::Error> {
    let train_file = std::fs::read_to_string(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "./lab1_data/classification_dataset/train_set.csv".to_string()),
    )?;
    let validation_file =
        std::fs::read_to_string(std::env::args().nth(1).unwrap_or_else(|| {
            "./lab1_data/classification_dataset/validation_set.csv".to_string()
        }))?;
    let test_file = std::fs::read_to_string(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "./lab1_data/classification_dataset/test_set.csv".to_string()),
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

    let mut output = std::fs::File::create("17341039_FUYuze_KNN_classification.csv")?;
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
