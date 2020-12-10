use anyhow::Result;
use embedded_graphics::pixelcolor::Rgb888;
use embedded_graphics::prelude::*;
use embedded_graphics_simulator::{
    OutputSettingsBuilder, SimulatorDisplay, SimulatorEvent, Window,
};

mod board;
mod eval;
mod opts;
mod search;
use board::{Board, BoardHuman, CellState};

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

    let mut board = Board::new(
        opt.size,
        match opt.machine_first {
            true => BoardHuman::White,
            false => BoardHuman::Black,
        },
    )?;

    'running: loop {
        window.update(&display);

        board.draw(&mut display)?;

        if board.is_machine_turn() {
            let (row, col) = search::search_strategy(&mut board, opt.depth);
            board.machine_place(row as usize, col as usize)?;
        }

        for event in window.events() {
            #[allow(clippy::single_match)]
            match event {
                SimulatorEvent::Quit => break 'running,
                SimulatorEvent::MouseButtonUp { point, .. } => {
                    let (col, row) = (point.x as u32 / px(1), point.y as u32 / px(1));
                    if row as usize >= board.size {
                        continue;
                    }
                    if board.get(row as usize, col as usize) != CellState::Null {
                        continue;
                    }

                    if board.is_human_turn() {
                        board.human_place(row as usize, col as usize)?;
                    }

                    use embedded_graphics::{
                        fonts::{Font8x16, Text},
                        style::TextStyleBuilder,
                    };

                    Text::new(
                        format!(
                            "H{:>8}M{:>8}",
                            eval::evaluate(&board, board.human_color()),
                            eval::evaluate(&board, board.machine_color())
                        )
                        .as_str(),
                        Point::new(
                            px(board.size as u32) as i32 - 18 * 8,
                            px(board.size as u32) as i32,
                        ),
                    )
                    .into_styled(
                        TextStyleBuilder::new(Font8x16)
                            .text_color(Rgb888::WHITE)
                            .background_color(Rgb888::BLACK)
                            .build(),
                    )
                    .draw(&mut display)?;
                }
                _ => {}
            }
        }
    }

    Ok(())
}
