use std::collections::BTreeMap;

use crate::{Case, Emotion};

#[derive(Default)]
pub struct TfIdfContext<'a, 'b, E: Emotion> {
    articles: Vec<&'b mut Case<'a, E>>,
    idf: BTreeMap<&'a str, f64>,
}

impl<'a, 'b, E: Emotion> TfIdfContext<'a, 'b, E> {
    pub fn new() -> Self {
        Self {
            articles: Vec::new(),
            idf: BTreeMap::new(),
        }
    }

    pub fn add_article(&mut self, article: &'b mut Case<'a, E>) {
        let wc = article.passage.iter().map(|(_, c)| c).sum::<f64>();

        // make TF
        for (_, c) in article.passage.iter_mut() {
            *c /= wc;
        }

        // count IDF
        for (w, _) in article.passage.iter() {
            if let Some(c) = self.idf.get_mut(w) {
                *c += 1.0;
            } else {
                self.idf.insert(w.to_owned(), 1.0);
            }
        }

        // add tracking
        self.articles.push(article);
    }

    fn mutate_idf(&mut self) {
        let pc = self.articles.len() as f64;
        let e = 1f64.exp();
        for (_, c) in self.idf.iter_mut() {
            *c = pc.log(e) - (*c + 1.0).log(e);
        }
    }

    pub fn apply_idf(mut self) {
        self.mutate_idf();

        for p in self.articles.into_iter() {
            for (w, c) in p.passage.iter_mut() {
                *c *= self.idf.get(w).unwrap();
            }
        }
    }
}
