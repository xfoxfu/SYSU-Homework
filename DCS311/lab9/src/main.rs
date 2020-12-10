use anyhow::Result;
use embedded_graphics::pixelcolor::Rgb888;
use embedded_graphics::prelude::*;
use embedded_graphics_simulator::{
    OutputSettingsBuilder, SimulatorDisplay, SimulatorEvent, Window,
};
use std::io::Read;
use std::str::FromStr;

mod board;
mod opts;

use board::{Board, BoardHuman, BoardState};

pub(crate) const fn px(pt: u32) -> u32 {
    pt * 32
}

fn main() -> Result<()> {
    use clap::Clap;
    let opt = dbg!(opts::Options::parse());

    let mut display: SimulatorDisplay<Rgb888> =
        SimulatorDisplay::new(Size::new(px(opt.size), px(opt.size) + 16));
    let output_settings = OutputSettingsBuilder::new().build();
    let mut window = Window::new("Lab 9 FU Yuze", &output_settings);

    let mut board = board::Board::new(
        opt.size,
        match opt.machine_first {
            true => board::BoardHuman::White,
            false => board::BoardHuman::Black,
        },
    )?;

    'running: loop {
        window.update(&display);

        board.draw(&mut display)?;

        for event in window.events() {
            #[allow(clippy::single_match)]
            match event {
                SimulatorEvent::Quit => break 'running,
                SimulatorEvent::MouseButtonUp { point, .. } => {
                    let (col, row) = (point.x as u32 / px(1), point.y as u32 / px(1));
                    if row as usize >= board.size {
                        continue;
                    }
                    if board.is_human_turn() {
                        board.human_place(row as usize, col as usize)?;
                    } else {
                        board.machine_place(row as usize, col as usize)?;
                    }
                }
                _ => {}
            }
        }
    }

    Ok(())
}
