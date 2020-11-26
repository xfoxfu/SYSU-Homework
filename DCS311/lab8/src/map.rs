use super::px;
use embedded_graphics::pixelcolor::Rgb888;
use embedded_graphics::prelude::*;
use embedded_graphics::primitives::Rectangle;
use embedded_graphics::style::PrimitiveStyleBuilder;
use nalgebra::{DMatrix, Point2};
use std::convert::{TryFrom, TryInto};
use std::ops::{Deref, DerefMut};
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
    current: Point2<usize>,
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
        let mut current = Point2::origin();
        for (i, l) in s.lines().enumerate() {
            for (j, c) in l.trim().chars().enumerate() {
                inner[(i, j)] = c.try_into()?;
                if inner[(i, j)] == Cell::Start {
                    if current[0] == 0 && current[1] == 0 {
                        current = Point2::new(i, j);
                        inner[(i, j)] = Cell::Road;
                    } else {
                        return Err(ParseError::TooMuchStart(i, j));
                    }
                }
            }
        }
        Ok(Self { inner, current })
    }
}

impl Deref for Map {
    type Target = DMatrix<Cell>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for Map {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl Drawable<Rgb888> for Map {
    fn draw<D: DrawTarget<Rgb888>>(self, display: &mut D) -> Result<(), D::Error> {
        let cell_step = PrimitiveStyleBuilder::new().fill_color(Rgb888::RED).build();
        let cell_wall = PrimitiveStyleBuilder::new()
            .fill_color(Rgb888::BLACK)
            .build();
        let cell_current = PrimitiveStyleBuilder::new()
            .fill_color(Rgb888::YELLOW)
            .build();
        let cell_end = PrimitiveStyleBuilder::new()
            .fill_color(Rgb888::GREEN)
            .build();

        for (i, row) in self.inner.row_iter().enumerate() {
            for (j, cell) in row.column_iter().enumerate() {
                let style = if self.current[0] == i && self.current[1] == j {
                    &cell_current
                } else {
                    match cell[0] {
                        Cell::Road | Cell::Start => &cell_step,
                        Cell::Wall => &cell_wall,
                        Cell::End => &cell_end,
                    }
                };
                Rectangle::new(
                    Point::new(px(j as u32) as i32, px(i as u32) as i32),
                    Point::new(px((j + 1) as u32) as i32, px((i + 1) as u32) as i32),
                )
                .into_styled(style.to_owned())
                .draw(display)?;
            }
        }

        Ok(())
    }
}
