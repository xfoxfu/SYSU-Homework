#![no_std]
#![no_main]
#![feature(llvm_asm)]

extern crate rlibc;

#[repr(transparent)]
struct Buffer {
    chars: [[u32; 600]; 800],
}

const COLORS: [u32; 19] = [
    0x00f44336, 0x00e91e63, 0x009c27b0, 0x00673ab7, 0x003f51b5, 0x002196f3, 0x0003a9f4, 0x0000bcd4,
    0x00009688, 0x004caf50, 0x008bc34a, 0x00cddc39, 0x00ffeb3b, 0x00ffc107, 0x00ff9800, 0x00ff5722,
    0x00795548, 0x009e9e9e, 0x00607d8b,
];

#[export_name = "_start"]
pub extern "C" fn __impl_start(boot_info: &'static boot::BootInfo) {
    let mode = boot_info.graphic_info.mode;
    let (display_x, display_y) = mode.resolution();
    let (display_x, display_y) = (display_x as isize, display_y as isize);
    let fb_addr = boot_info.graphic_info.fb_addr;

    for i in 0..display_x * display_y {
        unsafe {
            *(fb_addr as *mut u32).offset(i).as_mut().unwrap() = 0x000F0F0F;
        }
    }

    let base_x: isize = 0;
    let base_y: isize = 0;
    let max_x: isize = display_x;
    let max_y: isize = display_y;

    let mut row = base_x;
    let mut col = base_y;

    let mut row_incr = 2;
    let mut col_incr = 1;
    let mut color = 0;
    for _ in 0..512 {
        if col >= base_x && col < max_x && row >= base_y && row < max_y {
            unsafe {
                *(fb_addr as *mut u32)
                    .offset(row * display_x + col)
                    .as_mut()
                    .unwrap() = COLORS[color];
            }
        } else {
            // warn!("invalid position ({}, {})", col, row);
        }

        row += row_incr;
        col += col_incr;

        if col <= base_x || col > max_x {
            col_incr = -col_incr;
        }
        if row <= base_y || row > max_y {
            row_incr = -row_incr;
        }

        if col <= base_x || col > max_x || row <= base_y || row > max_y {
            color += 1;
        }

        if color >= 19 {
            color = 0;
        }

        // wait for a little while
        for _ in 0..1_000_000 {
            unsafe { llvm_asm!("nop") }
        }
    }
}

/// This function is called on panic.
#[cfg(not(test))]
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    loop {}
}