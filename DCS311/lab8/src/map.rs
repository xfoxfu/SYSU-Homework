use super::px;
use embedded_graphics::pixelcolor::Rgb888;
use embedded_graphics::prelude::*;
use embedded_graphics::primitives::Rectangle;
use embedded_graphics::style::{PrimitiveStyle, PrimitiveStyleBuilder};
use nalgebra::{DMatrix, Point2};
use std::convert::{TryFrom, TryInto};
use std::str::FromStr;

#[derive(thiserror::Error, Debug)]
pub enum ParseError {
    #[error("multiple start point ({0}, {1})")]
    TooMuchStart(usize, usize),
    #[error("invalid string {0}")]
    InvalidStr(String),
    #[error("invalid char {0}")]
    InvalidChar(char),
    #[error("too few lines in input string")]
    TooFew,
}

#[repr(u8)]
#[derive(Debug, Ord, PartialOrd, Eq, PartialEq, Copy, Clone)]
pub enum Cell {
    Road = 0x00,
    Wall = 0x01,
    Start = 0x10,
    End = 0x11,
    Explored = 0x12,
}

impl Cell {
    pub fn reachable(&self) -> bool {
        match self {
            Cell::Road | Cell::Start | Cell::End => true,
            Cell::Wall | Cell::Explored => false,
        }
    }

    pub fn get_color(&self) -> Rgb888 {
        match self {
            Cell::Road | Cell::Start => Rgb888::RED,
            Cell::Explored => Rgb888::YELLOW,
            Cell::Wall => Rgb888::BLACK,
            Cell::End => Rgb888::GREEN,
        }
    }
}

impl FromStr for Cell {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" => Ok(Cell::Road),
            "1" => Ok(Cell::Wall),
            "S" => Ok(Cell::Start),
            "E" => Ok(Cell::End),
            _ => Err(ParseError::InvalidStr(s.to_string())),
        }
    }
}

impl TryFrom<char> for Cell {
    type Error = ParseError;

    fn try_from(c: char) -> Result<Self, Self::Error> {
        match c {
            '0' => Ok(Cell::Road),
            '1' => Ok(Cell::Wall),
            'S' => Ok(Cell::Start),
            'E' => Ok(Cell::End),
            _ => Err(ParseError::InvalidChar(c)),
        }
    }
}

pub struct Map {
    inner: DMatrix<Cell>,
    current: (usize, usize),
    target: (usize, usize),
}

impl Map {
    pub fn current(&self) -> (usize, usize) {
        self.current
    }

    pub fn target(&self) -> (usize, usize) {
        self.target
    }

    pub fn get(&self, i: usize, j: usize) -> &Cell {
        self.inner.get((i, j)).unwrap()
    }

    pub fn get_mut(&mut self, i: usize, j: usize) -> &mut Cell {
        self.inner.get_mut((i, j)).unwrap()
    }

    pub fn explore(&mut self, i: usize, j: usize) {
        *self.inner.get_mut((i, j)).unwrap() = Cell::Explored;
        self.current = (i, j);
    }

    pub fn adjacent(&mut self, i: usize, j: usize) -> Vec<(usize, usize)> {
        let mut r = Vec::new();

        if i > 0 {
            r.push((i - 1, j));
        }
        if j > 0 {
            r.push((i, j - 1));
        }
        if i + 1 < self.inner.shape().0 {
            r.push((i + 1, j));
        }
        if j + 1 < self.inner.shape().1 {
            r.push((i, j + 1));
        }
        r
    }
}

impl FromStr for Map {
    type Err = ParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let m = s.lines().count();
        let n = s
            .lines()
            .next()
            .ok_or(ParseError::TooFew)?
            .trim()
            .chars()
            .count();
        let mut inner = unsafe { DMatrix::<Cell>::new_uninitialized(m, n) };
        let mut current = (0, 0);
        let mut target = (0, 0);
        for (i, l) in s.lines().enumerate() {
            for (j, c) in l.trim().chars().enumerate() {
                inner[(i, j)] = c.try_into()?;
                if inner[(i, j)] == Cell::Start {
                    if current.0 == 0 && current.1 == 0 {
                        current = (i, j);
                    } else {
                        return Err(ParseError::TooMuchStart(i, j));
                    }
                }
                if inner[(i, j)] == Cell::End {
                    if target.0 == 0 && target.1 == 0 {
                        target = (i, j);
                    } else {
                        return Err(ParseError::TooMuchStart(i, j));
                    }
                }
            }
        }
        Ok(Self {
            inner,
            current,
            target,
        })
    }
}

impl Map {
    pub fn draw_cell<D: DrawTarget<Rgb888>>(
        &self,
        display: &mut D,
        i: i32,
        j: i32,
        color: Rgb888,
    ) -> Result<(), D::Error> {
        Rectangle::new(
            Point::new(px(j as u32) as i32, px(i as u32) as i32),
            Point::new(px((j + 1) as u32) as i32, px((i + 1) as u32) as i32),
        )
        .into_styled(PrimitiveStyleBuilder::new().fill_color(color).build())
        .draw(display)?;

        Ok(())
    }
    pub fn draw<D: DrawTarget<Rgb888>>(&self, display: &mut D) -> Result<(), D::Error> {
        for (i, row) in self.inner.row_iter().enumerate() {
            for (j, cell) in row.column_iter().enumerate() {
                let color = cell[0].get_color();
                self.draw_cell(display, i as i32, j as i32, color)?;
            }
        }

        Ok(())
    }
}
