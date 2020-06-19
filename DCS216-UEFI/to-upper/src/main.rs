#![no_std]
#![no_main]
#![feature(asm)]

extern crate rlibc;
extern crate xlibr;

#[export_name = "_start"]
pub extern "C" fn __impl_start() -> ! {
    // let arg: u64 = 42;
    // let mut tar: u64 = 0;
    // unsafe {
    //     asm!("int {id}", id = const 0x80, in("rax") arg, in("rbx") &mut tar, in("rcx") 54, in("rdx") 128);
    // }
    // let (width, height) = boot_info.graphic_info.mode.resolution();
    // let (hw, hh) = (width as isize / 2, height as isize / 2);
    // xlibr::display(boot_info, 0, 0, hw, hh, tar as usize)
    xlibr::sys_exit()
}
