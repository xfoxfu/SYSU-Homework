use lab1::{Case, ExactEmotion, TfIdfContext};
use std::collections::BTreeSet;
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

    let mut words = BTreeSet::new();
    let mut articles = Vec::new();
    let mut ctx = TfIdfContext::new();

    // 1	all:148 anger:22 disgust:2 fear:60 joy:0 sad:64 surprise:0	mortar assault leav at least dead
    for line in file.lines() {
        let article = line.split('\t').nth(2).unwrap();
        let case = Case::<ExactEmotion>::new(article);
        for (word, _) in case.passage.iter() {
            words.insert(*word);
        }
        articles.push((article, case));
    }
    for (_, article) in articles.iter_mut() {
        ctx.add_article(article);
    }
    ctx.apply_idf();

    let mut out = std::fs::File::create("17341039_FUYuze_TFIDF.txt.2")?;

    print_comma_delimited(&mut out, words.iter())?;

    for (_, case) in articles.into_iter() {
        print_comma_delimited(
            &mut out,
            words.iter().map(|w| case.passage.get(w).unwrap_or(&0f64)),
        )?;
    }

    Ok(())
}
