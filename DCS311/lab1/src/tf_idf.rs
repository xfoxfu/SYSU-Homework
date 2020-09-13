use std::collections::{BTreeMap, BTreeSet};
use std::vec::Vec;

fn print_comma_delimited<V>(
    target: &mut dyn std::io::Write,
    mut iter: impl Iterator<Item = V>,
) -> Result<(), std::io::Error>
where
    V: std::fmt::Display,
{
    if let Some(arg) = iter.next() {
        write!(target, "{}", arg)?;

        for arg in iter {
            write!(target, ",{}", arg)?;
        }
    }
    writeln!(target)?;

    Ok(())
}

fn main() -> Result<(), std::io::Error> {
    let file = std::fs::read_to_string(
        std::env::args()
            .nth(1)
            .unwrap_or_else(|| "./lab1_data/semeval.txt".to_string()),
    )?;

    let mut tf = Vec::new();
    let mut idf = BTreeMap::new();
    let mut words = BTreeSet::new();

    for line in file.split('\n') {
        if line.is_empty() {
            continue;
        }

        let passage = line.splitn(3, '\t').nth(2).unwrap();

        let mut word_count = BTreeMap::new();
        for word in passage.split(' ') {
            if let Some(c) = word_count.get_mut(word) {
                *c += 1;
            } else {
                word_count.insert(word.to_owned(), 1);
            }
        }

        for word in word_count.keys() {
            if !words.contains(word) {
                words.insert(word.to_owned());
            }

            if let Some(c) = idf.get_mut(word) {
                *c += 1;
            } else {
                idf.insert(word.to_owned(), 1);
            }
        }

        let len = passage.split(' ').count() as f64;
        tf.push(
            word_count
                .into_iter()
                .map(|(k, v)| (k, v as f64 / len))
                .collect::<BTreeMap<_, _>>(),
        );
    }

    let e = 1f64.exp();
    let passage_count = tf.len() as f64;
    let idf = idf
        .into_iter()
        .map(|(w, v)| (w, passage_count.log(e) - (v as f64 + 1.0).log(e)))
        .collect::<BTreeMap<_, _>>();

    for line in tf.iter_mut() {
        for (word, word_tf) in line.iter_mut() {
            *word_tf *= idf.get(word).copied().unwrap_or(0f64);
        }
    }
    let tf_idf = tf;

    let mut out = std::fs::File::create("17341039_FUYuze_TFIDF.txt")?;

    print_comma_delimited(&mut out, words.iter())?;

    for line in tf_idf.iter() {
        print_comma_delimited(&mut out, words.iter().map(|w| line.get(w).unwrap_or(&0f64)))?;
    }

    Ok(())
}
