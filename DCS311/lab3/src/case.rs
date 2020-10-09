use crate::{Vector40, Vector41};
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq)]
pub struct Case {
    pub features: Vector40,
    pub tag: bool,
}

impl Case {
    pub fn new(features: Vector40, tag: bool) -> Self {
        Self { features, tag }
    }

    pub fn x(&self) -> &Vector40 {
        &self.features
    }
    pub fn y_pla(&self) -> f64 {
        if self.tag {
            1f64
        } else {
            -1f64
        }
    }
    pub fn y_lr(&self) -> f64 {
        if self.tag {
            1f64
        } else {
            0f64
        }
    }
}

#[derive(Debug)]
pub struct ParseError;

impl FromStr for Case {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let features = Vector40::from_iterator(s.split(',').map(|s| s.parse().unwrap()));
        let tag =
            (s[s.rfind(',').unwrap() + 1..].parse::<f64>().unwrap() - 1f64).abs() <= f64::EPSILON;
        Ok(Case::new(features, tag))
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
#[test]
pub fn parse() {
    assert_eq!(
        Case::from_str("0.11,-0.45,1.08,0.57,1.53,1.1,3.13,3.9,3.16,5.2,4.7,3.83,2.82,4.17,1.56,-1.66,-0.37,-3.14,0.1,-0.1,0.43,0.71,0.41,1.76,0.15,-1.04,-0.4,0.22,0.41,0.72,0.42,-0.58,0.31,1.25,-0.64,-0.08,0.31,-1.19,-0.63,-0.87,0").unwrap(),
        Case::new(Vector40::from_column_slice(&[0.11,-0.45,1.08,0.57,1.53,1.1,3.13,3.9,3.16,5.2,4.7,3.83,2.82,4.17,1.56,-1.66,-0.37,-3.14,0.1,-0.1,0.43,0.71,0.41,1.76,0.15,-1.04,-0.4,0.22,0.41,0.72,0.42,-0.58,0.31,1.25,-0.64,-0.08,0.31,-1.19,-0.63,-0.87]),false),
    )
}
