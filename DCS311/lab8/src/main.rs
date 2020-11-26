use anyhow::Result;
use embedded_graphics::fonts::Font6x8;
use embedded_graphics::pixelcolor::BinaryColor;
use embedded_graphics::prelude::*;
use embedded_graphics::{egcircle, egline, egtext, primitive_style, text_style};
use embedded_graphics_simulator::{
    OutputSettingsBuilder, SimulatorDisplay, SimulatorEvent, Window,
};
use std::thread;
use std::time::Duration;

const fn px(pt: u32) -> u32 {
    pt * 16
}

fn main() -> Result<()> {
    let mut display: SimulatorDisplay<BinaryColor> =
        SimulatorDisplay::new(Size::new(px(36), px(18)));
    let output_settings = OutputSettingsBuilder::new()
        // .theme(BinaryColorTheme::OledBlue)
        .build();
    let mut window = Window::new("Lab 8 FU Yuze", &output_settings);

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
