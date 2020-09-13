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
    let file = std::fs::read_to_string("./lab1_data/semeval.txt")?;

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

        let len = word_count.len() as f64;
        tf.push(
            word_count
                .into_iter()
                .map(|(k, v)| (k, v as f64 / len))
                .collect::<BTreeMap<_, _>>(),
        )
    }

    let passage_count = tf.len() as f64;
    let idf = idf
        .into_iter()
        .map(|(w, v)| (w, passage_count.log2() / (v as f64).log2()))
        .collect::<BTreeMap<_, _>>();

    for line in tf.iter_mut() {
        for (word, word_tf) in line.iter_mut() {
            *word_tf *= idf.get(word).copied().unwrap_or(0f64);
        }
    }
    let tf_idf = tf;

    let mut out_f;
    let mut out_s;
    let out: &mut dyn std::io::Write = if let Some(f) = std::env::args().nth(1) {
        out_f = std::fs::File::create(f)?;
        &mut out_f
    } else {
        out_s = std::io::stdout();
        &mut out_s
    };

    print_comma_delimited(out, words.iter())?;

    for line in tf_idf.iter() {
        print_comma_delimited(out, words.iter().map(|w| line.get(w).unwrap_or(&0f64)))?;
    }

    Ok(())
}
