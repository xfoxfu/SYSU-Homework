mod case;
mod emotion;
mod knn;

pub use case::Case;
pub use emotion::{Emotion, ExactEmotion, ProbEmotion};
pub use knn::DistanceObject;

pub fn safe_sub_abs(lhs: usize, rhs: usize) -> usize {
    if lhs > rhs {
        lhs - rhs
    } else {
        rhs - lhs
    }
}
