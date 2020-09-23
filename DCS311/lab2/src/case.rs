use std::{fmt::Display, str::FromStr};

#[derive(Debug, Copy, Clone)]
pub struct Case {
    pub buying: Buying,
    pub maint: Maint,
    pub doors: Doors,
    pub persons: Persons,
    pub lug_boot: LugBoot,
    pub safety: Safety,
    pub label: Label,
}

impl Case {
    pub fn new(
        buying: Buying,
        maint: Maint,
        doors: Doors,
        persons: Persons,
        lug_boot: LugBoot,
        safety: Safety,
        label: Label,
    ) -> Self {
        Self {
            buying,
            maint,
            doors,
            persons,
            lug_boot,
            safety,
            label,
        }
    }
}

pub type CasePredicateFn = fn(&Case) -> bool;

impl FromStr for Case {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut iter = s.splitn(7, ',');
        Ok(Self::new(
            /* buying */ iter.next().ok_or(()).and_then(Buying::from_str)?,
            /* maint */ iter.next().ok_or(()).and_then(Maint::from_str)?,
            /* doors */ iter.next().ok_or(()).and_then(Doors::from_str)?,
            /* persons */ iter.next().ok_or(()).and_then(Persons::from_str)?,
            /* lug_boot */ iter.next().ok_or(()).and_then(LugBoot::from_str)?,
            /* safety */ iter.next().ok_or(()).and_then(Safety::from_str)?,
            /* label */ iter.next().ok_or(()).and_then(Label::from_str)?,
        ))
    }
}

impl Display for Case {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{},{},{},{},{},{},{}",
            self.buying,
            self.maint,
            self.doors,
            self.persons,
            self.lug_boot,
            self.safety,
            self.label,
        ))
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum Buying {
    High,
    Low,
    Med,
    Vhigh,
}

impl FromStr for Buying {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let r = match s {
            "high" => Self::High,
            "low" => Self::Low,
            "med" => Self::Med,
            "vhigh" => Self::Vhigh,
            _ => return Err(()),
        };
        Ok(r)
    }
}

impl Display for Buying {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::High => "high",
            Self::Low => "low",
            Self::Med => "med",
            Self::Vhigh => "vhigh",
        };
        f.write_str(s)
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum Maint {
    High,
    Low,
    Med,
    Vhigh,
}

impl FromStr for Maint {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let r = match s {
            "high" => Self::High,
            "low" => Self::Low,
            "med" => Self::Med,
            "vhigh" => Self::Vhigh,
            _ => return Err(()),
        };
        Ok(r)
    }
}

impl Display for Maint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::High => "high",
            Self::Low => "low",
            Self::Med => "med",
            Self::Vhigh => "vhigh",
        };
        f.write_str(s)
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum Doors {
    Two,
    Three,
    Four,
    FiveOrMore,
}

impl FromStr for Doors {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let r = match s {
            "2" => Self::Two,
            "3" => Self::Three,
            "4" => Self::Four,
            "5more" => Self::FiveOrMore,
            _ => return Err(()),
        };
        Ok(r)
    }
}

impl Display for Doors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Two => "2",
            Self::Three => "3",
            Self::Four => "4",
            Self::FiveOrMore => "5more",
        };
        f.write_str(s)
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum Persons {
    Two,
    Four,
    More,
}

impl FromStr for Persons {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let r = match s {
            "2" => Self::Two,
            "4" => Self::Four,
            "more" => Self::More,
            _ => return Err(()),
        };
        Ok(r)
    }
}

impl Display for Persons {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Two => "2",
            Self::Four => "4",
            Self::More => "more",
        };
        f.write_str(s)
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum LugBoot {
    Big,
    Med,
    Small,
}

impl FromStr for LugBoot {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let r = match s {
            "big" => Self::Big,
            "med" => Self::Med,
            "small" => Self::Small,
            _ => return Err(()),
        };
        Ok(r)
    }
}

impl Display for LugBoot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Big => "big",
            Self::Med => "med",
            Self::Small => "small",
        };
        f.write_str(s)
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum Safety {
    High,
    Low,
    Med,
}

impl FromStr for Safety {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let r = match s {
            "high" => Self::High,
            "low" => Self::Low,
            "med" => Self::Med,
            _ => return Err(()),
        };
        Ok(r)
    }
}

impl Display for Safety {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::High => "high",
            Self::Low => "low",
            Self::Med => "med",
        };
        f.write_str(s)
    }
}

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub enum Label {
    False,
    True,
    Unlabeled,
}

impl FromStr for Label {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let r = match s {
            "0" => Self::False,
            "1" => Self::True,
            "?" => Self::Unlabeled,
            _ => return Err(()),
        };
        Ok(r)
    }
}

impl Display for Label {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::False => "0",
            Self::True => "1",
            Self::Unlabeled => "?",
        };
        f.write_str(s)
    }
}
