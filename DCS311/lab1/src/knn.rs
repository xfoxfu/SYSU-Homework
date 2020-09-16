use crate::{Case, Emotion};
use std::collections::BinaryHeap;

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
