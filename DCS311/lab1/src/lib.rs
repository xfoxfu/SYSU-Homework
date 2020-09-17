mod case;
mod emotion;
mod knn;
mod tf_idf;

pub use case::Case;
pub use emotion::{Emotion, ExactEmotion, ProbEmotion};
pub use knn::DistanceObject;
pub use tf_idf::TfIdfContext;

pub fn safe_sub_abs(lhs: usize, rhs: usize) -> usize {
    if lhs > rhs {
        lhs - rhs
    } else {
        rhs - lhs
    }
}
