use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::vec::Vec;

fn passage_vector<'a>(passage: &[&'a str]) -> BTreeMap<&'a str, usize> {
    let mut word_count = BTreeMap::new();
    for word in passage.iter() {
        if let Some(c) = word_count.get_mut(*word) {
            *c += 1;
        } else {
            word_count.insert(*word, 1);
        }
    }

    word_count
}

fn passage_distance(distance_p: u32, lhs: &[&str], rhs: &[&str]) -> usize {
    let lvec = passage_vector(lhs);
    let rvec = passage_vector(rhs);

    let mut diff = 0usize;

    for (lw, lc) in lvec.iter() {
        if rvec.contains_key(lw) {
            diff += (lc + rvec[lw]).pow(distance_p);
        } else {
            diff += lc.pow(distance_p);
        }
    }
    for (rw, rc) in rvec.iter() {
        if !lvec.contains_key(rw) {
            diff += rc.pow(distance_p);
        }
    }

    diff
}

fn predict_emotion<'a>(
    threshold_k: usize,
    distance_p: u32,
    train_data: &'a BTreeMap<Vec<&str>, &str>,
    target_passage: &'a [&str],
) -> &'a str {
    let mut k_minimals = BinaryHeap::<(usize, &str)>::new();

    for (compare_passage, compare_emotion) in train_data.iter() {
        let dist = passage_distance(distance_p, &target_passage, compare_passage);

        if let Some((max_dist, _)) = k_minimals.peek() {
            if *max_dist <= dist {
                continue;
            }
        }

        k_minimals.push((dist, compare_emotion));

        if k_minimals.len() > threshold_k {
            k_minimals.pop();
        }

        assert!(k_minimals.len() <= threshold_k);
    }
    assert!(k_minimals.len() == threshold_k);

    let mut assume_emotions = HashMap::new();
    for (_dist, emotion) in k_minimals.into_iter() {
        if assume_emotions.contains_key(emotion) {
            *assume_emotions.get_mut(emotion).unwrap() += 1;
        } else {
            assume_emotions.insert(emotion, 1);
        }
    }

    let (emotion, _) = assume_emotions.into_iter().max_by_key(|(_, c)| *c).unwrap();

    emotion
}

fn compute_with_kp<W: std::io::Write>(
    train_file: &str,
    validation_file: &str,
    test_file: &str,
    output_file: Option<&mut W>,
    threshold_k: usize,
    distance_p: u32,
) -> Result<f64, std::io::Error> {
    let mut train_data = BTreeMap::new();

    for line in train_file.split('\n') {
        if line == "Words (split by space),label" || line.is_empty() {
            continue;
        }

        let passage = line.split(',').next().unwrap();
        let emotion = line.split(',').nth(1).unwrap();

        let words = passage.split(' ').collect::<Vec<_>>();

        train_data.insert(words, emotion);
    }

    let validation_total = validation_file
        .split('\n')
        .filter(|s| !s.trim().is_empty() && s != &"Words (split by space),label")
        .count();
    let mut validation_ok_count = 0usize;

    for line in validation_file.split('\n') {
        if line == "Words (split by space),label" || line.is_empty() {
            continue;
        }

        let target_passage = line
            .split(',')
            .next()
            .unwrap()
            .split(' ')
            .collect::<Vec<_>>();
        let target_emotion = line.split(',').nth(1).unwrap();

        let emotion = predict_emotion(threshold_k, distance_p, &train_data, &target_passage);

        if emotion == target_emotion {
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
            let target_passage = line
                .split(',')
                .nth(1)
                .unwrap()
                .split(' ')
                .collect::<Vec<_>>();

            let emotion = predict_emotion(threshold_k, distance_p, &train_data, &target_passage);

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

    let mut best_ks = BinaryHeap::with_capacity(100);
    for p in 1..5 {
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

    let mut output = std::fs::File::create("17341039_FUYuze_KNN_classification.csv")?;
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
