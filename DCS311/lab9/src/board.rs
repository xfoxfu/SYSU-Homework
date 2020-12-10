use crate::px;
use embedded_graphics::drawable::Drawable;
use embedded_graphics::fonts::{Font8x16, Text};
use embedded_graphics::pixelcolor::{Rgb888, RgbColor};
use embedded_graphics::prelude::{Point, Primitive};
use embedded_graphics::primitives::{Circle, Rectangle};
use embedded_graphics::style::{PrimitiveStyleBuilder, TextStyleBuilder};
use embedded_graphics::DrawTarget;

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
    #[error("recovery inconsistency {0:?} => {0:?}")]
    InvalidRecover(CellState, CellState),
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

    pub fn has_neighbor(&self, row: usize, col: usize) -> bool {
        for r in
            (std::cmp::max(0, row as isize - 2) as usize)..=std::cmp::min(self.size - 1, row + 2)
        {
            for c in (std::cmp::max(0, col as isize - 2) as usize)
                ..=std::cmp::min(self.size - 1, col + 2)
            {
                if self.get(r, c) != CellState::Null {
                    return true;
                }
            }
        }
        false
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
        let r = self.place(row, col, self.human_color());
        self.update_wins();
        r
    }

    pub fn machine_place(&mut self, row: usize, col: usize) -> Result<(), BoardError> {
        if !self.is_machine_turn() {
            return Err(BoardError::NotMachineTurn);
        }
        let r = self.place(row, col, self.machine_color());
        self.update_wins();
        r
    }

    pub fn current_try_place(
        &mut self,
        row: usize,
        col: usize,
    ) -> Result<(usize, usize, CellState, BoardState), BoardError> {
        let prev = (self.get(row, col), self.state);
        self.place(row, col, self.current_color())?;
        Ok((row, col, prev.0, prev.1))
    }

    pub fn current_recover(
        &mut self,
        (row, col, pcell, state): (usize, usize, CellState, BoardState),
    ) -> Result<(), BoardError> {
        let cell = self.get(row, col);
        // if cell != pcell {
        //     return Err(BoardError::InvalidRecover(cell, pcell));
        // }
        self.set(row, col, CellState::Null);
        self.state = state;
        Ok(())
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

    pub fn current_color(&self) -> CellState {
        match self.state {
            BoardState::HumanTake => self.human_color(),
            BoardState::MachineTake => self.machine_color(),
            _ => Err(BoardError::GameEnded).unwrap(),
        }
    }

    pub fn last_color(&self) -> CellState {
        match self.state {
            BoardState::HumanTake => self.machine_color(),
            BoardState::MachineTake => self.human_color(),
            _ => Err(BoardError::GameEnded).unwrap(),
        }
    }

    pub fn check_wins(&self, target: CellState) -> bool {
        for row in 0..self.size {
            for col in 0..self.size {
                if (row + 4 < self.size
                    && self.get(row, col) == target
                    && self.get(row + 1, col) == target
                    && self.get(row + 2, col) == target
                    && self.get(row + 3, col) == target
                    && self.get(row + 4, col) == target)
                    || (col + 4 < self.size
                        && self.get(row, col) == target
                        && self.get(row, col + 1) == target
                        && self.get(row, col + 2) == target
                        && self.get(row, col + 3) == target
                        && self.get(row, col + 4) == target)
                    || (row + 4 < self.size
                        && col + 4 < self.size
                        && self.get(row, col) == target
                        && self.get(row + 1, col + 1) == target
                        && self.get(row + 2, col + 2) == target
                        && self.get(row + 3, col + 3) == target
                        && self.get(row + 4, col + 4) == target)
                    || (row + 4 < self.size
                        && col >= 4
                        && self.get(row, col) == target
                        && self.get(row + 1, col - 1) == target
                        && self.get(row + 2, col - 2) == target
                        && self.get(row + 3, col - 3) == target
                        && self.get(row + 4, col - 4) == target)
                {
                    return true;
                }
            }
        }
        false
    }

    pub fn update_wins(&mut self) {
        if self.check_wins(self.human_color()) {
            self.state = BoardState::HumanWin;
        } else if self.check_wins(self.machine_color()) {
            self.state = BoardState::HumanLose;
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
            format!("State = {:<10?}     ", self.state).as_str(),
            Point::new(0, px(self.size as u32) as i32),
        )
        .into_styled(
            TextStyleBuilder::new(Font8x16)
                .text_color(Rgb888::YELLOW)
                .background_color(match self.state {
                    BoardState::HumanWin => Rgb888::GREEN,
                    BoardState::HumanLose => Rgb888::RED,
                    _ => Rgb888::BLACK,
                })
                .build(),
        )
        .draw(display)?;

        Ok(())
    }
}
