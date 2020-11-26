use anyhow::Result;
use embedded_graphics::pixelcolor::Rgb888;
use embedded_graphics::prelude::*;
use embedded_graphics_simulator::{
    OutputSettingsBuilder, SimulatorDisplay, SimulatorEvent, Window,
};
use std::io::Read;
use std::str::FromStr;

mod astar;
mod map;
mod search;
mod ucs;

use astar::AStarSearch;
use map::{Cell, Map};
use search::{CostFunction, GenericSearch, Search};
use ucs::UniformCostSearch;

pub(crate) const fn px(pt: u32) -> u32 {
    pt * 16
}

fn main() -> Result<()> {
    let mut display: SimulatorDisplay<Rgb888> =
        SimulatorDisplay::new(Size::new(px(36), px(18) + 16));
    let output_settings = OutputSettingsBuilder::new().build();
    let mut window = Window::new("Lab 8 FU Yuze", &output_settings);

    let mut map = String::new();
    std::fs::File::open("MazeData.txt")?.read_to_string(&mut map)?;
    let mut map = Map::from_str(&map)?;

    map.draw(&mut display)?;
    let mut search: Box<dyn Search> = {
        let op = std::env::args().nth(1);
        match op.as_deref() {
            Some("astar") => Box::new(AStarSearch::new(&mut map)),
            Some("ucs") => Box::new(UniformCostSearch::new(&mut map)),
            _ => panic!("unknown operation, use `astar` or `ucs`"),
        }
    };
    let mut found = false;

    'running: loop {
        window.update(&display);

        if !found {
            if search.iterate()?.is_some() {
                found = true;
            }
            search.draw(&mut display)?;
        }
        for event in window.events() {
            #[allow(clippy::single_match)]
            match event {
                SimulatorEvent::Quit => break 'running,
                _ => {}
            }
        }
    }

    Ok(())
}
