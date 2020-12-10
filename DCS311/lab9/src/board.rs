use crate::px;
use embedded_graphics::{
    drawable::Drawable,
    fonts::{Font8x16, Text},
    pixelcolor::{Rgb888, RgbColor},
    prelude::{Point, Primitive},
    primitives::{Circle, Rectangle},
    style::{PrimitiveStyleBuilder, TextStyleBuilder},
    DrawTarget,
};

#[derive(thiserror::Error, Debug)]
pub enum BoardError {
    #[error("size {0} not odd number")]
    InvalidSize(usize),
    #[error("current is not turn of human")]
    NotHumanTurn,
    #[error("current is not turn of machine")]
    NotMachineTurn,
    #[error("game has ended")]
    GameEnded,
    #[error("invalid placement {0:?} => {0:?}")]
    InvalidPlacement(CellState, CellState),
}

/// global game state
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum BoardState {
    HumanTake,
    MachineTake,
    HumanWin,
    HumanLose,
}

/// the color human plays
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum BoardHuman {
    White,
    Black,
}

/// cell state
#[derive(Debug, Eq, PartialEq, Copy, Clone)]
pub enum CellState {
    White,
    Black,
    Null,
}

impl CellState {
    pub fn get_color(&self) -> Rgb888 {
        match self {
            CellState::White => Rgb888::WHITE,
            CellState::Black => Rgb888::BLACK,
            CellState::Null => Rgb888::new(0xF5, 0x7C, 0x00),
        }
    }
}

/// a game
#[derive(Debug)]
pub struct Board {
    pub human: BoardHuman,
    pub state: BoardState,
    inner: Vec<CellState>,
    pub size: usize,
}

impl Board {
    pub fn new(size: u32, human: BoardHuman) -> Result<Self, BoardError> {
        let state = match human {
            BoardHuman::Black => BoardState::HumanTake,
            BoardHuman::White => BoardState::MachineTake,
        };
        let size = size as usize;
        if size % 2 == 0 {
            return Err(BoardError::InvalidSize(size));
        }
        let inner = vec![CellState::Null; (size * size) as usize];
        let mut board = Self {
            state,
            human,
            inner,
            size,
        };
        let mid = size / 2;
        board.set(mid, mid, CellState::White);
        board.set(mid, mid + 1, CellState::White);
        board.set(mid, mid - 1, CellState::Black);
        board.set(mid + 1, mid, CellState::Black);

        Ok(board)
    }

    pub fn get(&self, row: usize, col: usize) -> CellState {
        self.inner[row * self.size + col]
    }

    pub fn set(&mut self, row: usize, col: usize, val: CellState) {
        self.inner[row * self.size + col] = val;
    }

    pub fn place(&mut self, row: usize, col: usize, val: CellState) -> Result<(), BoardError> {
        let cell = self.get(row, col);
        if cell != CellState::Null {
            return Err(BoardError::InvalidPlacement(cell, val));
        }
        if val == CellState::Null {
            return Err(BoardError::InvalidPlacement(cell, val));
        }

        self.set(row, col, val);
        self.state = match self.state {
            BoardState::HumanTake => BoardState::MachineTake,
            BoardState::MachineTake => BoardState::HumanTake,
            _ => unreachable!(),
        };
        Ok(())
    }

    pub fn human_place(&mut self, row: usize, col: usize) -> Result<(), BoardError> {
        if !self.is_human_turn() {
            return Err(BoardError::NotHumanTurn);
        }
        self.place(row, col, self.human_color())
    }

    pub fn machine_place(&mut self, row: usize, col: usize) -> Result<(), BoardError> {
        if !self.is_machine_turn() {
            return Err(BoardError::NotHumanTurn);
        }
        self.place(row, col, self.machine_color())
    }

    pub fn is_human_turn(&self) -> bool {
        self.state == BoardState::HumanTake
    }

    pub fn is_machine_turn(&self) -> bool {
        self.state == BoardState::MachineTake
    }

    pub fn human_color(&self) -> CellState {
        match self.human {
            BoardHuman::Black => CellState::Black,
            BoardHuman::White => CellState::White,
        }
    }

    pub fn machine_color(&self) -> CellState {
        match self.human {
            BoardHuman::Black => CellState::White,
            BoardHuman::White => CellState::Black,
        }
    }
}

impl Board {
    pub fn draw_cell<D: DrawTarget<Rgb888>>(
        &self,
        display: &mut D,
        row: i32,
        col: i32,
        color: Rgb888,
    ) -> Result<(), D::Error> {
        Rectangle::new(
            Point::new(px(col as u32) as i32, px(row as u32) as i32),
            Point::new(px(col as u32 + 1) as i32, px(row as u32 + 1) as i32),
        )
        .into_styled(
            PrimitiveStyleBuilder::new()
                .stroke_color(Rgb888::BLACK)
                .stroke_width(1)
                .fill_color(Rgb888::new(0xF5, 0x7C, 0x00))
                .build(),
        )
        .draw(display)?;
        Circle::new(
            Point::new(
                (px(col as u32) + px(1) / 2) as i32,
                (px(row as u32) + px(1) / 2) as i32,
            ),
            px(1) / 2 - 1,
        )
        .into_styled(PrimitiveStyleBuilder::new().fill_color(color).build())
        .draw(display)?;

        Ok(())
    }

    pub fn draw<D: DrawTarget<Rgb888>>(&self, display: &mut D) -> Result<(), D::Error> {
        for row in 0..self.size {
            for col in 0..self.size {
                let color = self.get(row, col).get_color();
                self.draw_cell(display, row as i32, col as i32, color)?;
            }
        }
        Text::new(
            format!("State = {:?}", self.state).as_str(),
            Point::new(0, px(self.size as u32) as i32),
        )
        .into_styled(
            TextStyleBuilder::new(Font8x16)
                .text_color(Rgb888::YELLOW)
                .background_color(Rgb888::BLACK)
                .build(),
        )
        .draw(display)?;

        Ok(())
    }
}
