use anyhow::Result;
use embedded_graphics::pixelcolor::Rgb888;
use embedded_graphics::prelude::*;
use embedded_graphics_simulator::{
    OutputSettingsBuilder, SimulatorDisplay, SimulatorEvent, Window,
};
use std::{io::Read, thread};
use std::{str::FromStr, time::Duration};

// mod astar;
mod map;
mod ucs;

// use astar::AStarSearch;
use map::{Cell, Map};
use ucs::UniformCostSearch;

pub(crate) const fn px(pt: u32) -> u32 {
    pt * 16
}

fn main() -> Result<()> {
    let mut display: SimulatorDisplay<Rgb888> = SimulatorDisplay::new(Size::new(px(36), px(18)));
    let output_settings = OutputSettingsBuilder::new()
        // .theme(BinaryColorTheme::OledBlue)
        .build();
    let mut window = Window::new("Lab 8 FU Yuze", &output_settings);

    let mut map = String::new();
    std::fs::File::open("MazeData.txt")?.read_to_string(&mut map)?;
    let mut map = Map::from_str(&map)?;

    map.draw(&mut display)?;
    let mut search = UniformCostSearch::new(&mut map);
    let mut found = false;

    'running: loop {
        window.update(&display);

        if !found {
            if let Some(route) = search.iterate()? {
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
        // thread::sleep(Duration::from_millis(1));
    }

    Ok(())
}
