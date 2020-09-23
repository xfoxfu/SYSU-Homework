use crate::{Case, Emotion};
use std::collections::BinaryHeap;

#[derive(PartialEq)]
pub struct DistanceObject<'a, E: Emotion>(pub f64, pub &'a E);
impl<'a, E: Emotion> std::cmp::Eq for DistanceObject<'a, E> {}
impl<'a, E: Emotion> std::cmp::PartialOrd for DistanceObject<'a, E> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl<'a, E: Emotion> std::cmp::Ord for DistanceObject<'a, E> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap()
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

        // 对于训练数据集的每一个点
        for case in train_data.iter() {
            // 计算距离
            let dist = self.distance(case, distance_p);

            // 若超过当前队列最大点，则直接忽略
            if let Some(DistanceObject(max_dist, _)) = k_minimals.peek() {
                if *max_dist <= dist {
                    continue;
                }
            }

            // 否则加入队列
            k_minimals.push(DistanceObject(dist, &case.emotion));

            // 并且保证队列长度在 K 以下
            if k_minimals.len() > threshold_k {
                k_minimals.pop();
            }

            debug_assert!(k_minimals.len() <= threshold_k);
        }

        k_minimals.into_iter()
    }
}
