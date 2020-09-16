use std::str::FromStr;

pub trait Emotion: Default + FromStr<Err = ()> + PartialEq + Clone {}

#[derive(Eq, PartialEq, Clone, Hash)]
pub enum ExactEmotion {
    Anger,
    Disgust,
    Fear,
    Joy,
    Sad,
    Surprise,
}

impl Default for ExactEmotion {
    fn default() -> Self {
        ExactEmotion::Anger
    }
}
impl FromStr for ExactEmotion {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let r = match s {
            "anger" => Self::Anger,
            "disgust" => Self::Disgust,
            "fear" => Self::Fear,
            "joy" => Self::Joy,
            "sad" => Self::Sad,
            "surprise" => Self::Surprise,
            _ => return Err(()),
        };
        Ok(r)
    }
}
impl Emotion for ExactEmotion {}
impl std::fmt::Display for ExactEmotion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Anger => "anger",
            Self::Disgust => "disgust",
            Self::Fear => "fear",
            Self::Joy => "joy",
            Self::Sad => "sad",
            Self::Surprise => "surprise",
        };
        f.write_str(s)
    }
}

#[derive(PartialEq, Clone)]
pub struct ProbEmotion {
    pub anger: f64,
    pub disgust: f64,
    pub fear: f64,
    pub joy: f64,
    pub sad: f64,
    pub surprise: f64,
}

impl Default for ProbEmotion {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }
}
impl FromStr for ProbEmotion {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut iter = s.split(',');
        let anger = iter.next().and_then(|v| v.parse().ok()).ok_or(())?;
        let disgust = iter.next().and_then(|v| v.parse().ok()).ok_or(())?;
        let fear = iter.next().and_then(|v| v.parse().ok()).ok_or(())?;
        let joy = iter.next().and_then(|v| v.parse().ok()).ok_or(())?;
        let sad = iter.next().and_then(|v| v.parse().ok()).ok_or(())?;
        let surprise = iter.next().and_then(|v| v.parse().ok()).ok_or(())?;

        Ok(Self::new(anger, disgust, fear, joy, sad, surprise))
    }
}
impl Emotion for ProbEmotion {}

impl ProbEmotion {
    pub fn new(anger: f64, disgust: f64, fear: f64, joy: f64, sad: f64, surprise: f64) -> Self {
        Self {
            anger,
            disgust,
            fear,
            joy,
            sad,
            surprise,
        }
    }

    pub fn from_vec(v: Vec<f64>) -> Self {
        Self::new(v[0], v[1], v[2], v[3], v[4], v[5])
    }

    pub fn into_vec(self) -> Vec<f64> {
        vec![
            self.anger,
            self.disgust,
            self.fear,
            self.joy,
            self.sad,
            self.surprise,
        ]
    }
}
