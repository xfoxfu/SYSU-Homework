use anyhow::Result;
use embedded_graphics::pixelcolor::Rgb888;
use embedded_graphics::prelude::*;
use embedded_graphics_simulator::{
    OutputSettingsBuilder, SimulatorDisplay, SimulatorEvent, Window,
};
use std::{io::Read, thread};
use std::{str::FromStr, time::Duration};

mod map;
use map::{Cell, Map};

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
    let map = Map::from_str(&map)?;

    map.draw(&mut display)?;

    'running: loop {
        window.update(&display);

        for event in window.events() {
            match event {
                SimulatorEvent::MouseButtonUp { point, .. } => {
                    println!("Click event at ({}, {})", point.x, point.y);
                }
                SimulatorEvent::Quit => break 'running,
                _ => {}
            }

            thread::sleep(Duration::from_millis(200));
        }
    }

    Ok(())
}
