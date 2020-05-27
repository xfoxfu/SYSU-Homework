#![no_std]
#![no_main]
#![feature(llvm_asm, abi_x86_interrupt, alloc_error_handler)]
#![feature(type_alias_impl_trait)]

use boot::BootInfo;
use x86_64::VirtAddr;

#[macro_use]
mod console;
mod allocator;
mod display;
mod drivers;
mod interrupts;
mod logging;
mod memory;
mod uefi_clock;

extern crate alloc;
extern crate rlibc;
#[macro_use]
extern crate log;
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate bitflags;

macro_rules! _svc {
    ($t: path) => {
        $t.lock().as_ref().unwrap()
    };
    ($t: path :mut) => {
        $t.lock().as_mut().unwrap()
    };
}

boot::entry_point!(kmain);

pub fn kmain(boot_info: &'static BootInfo) -> ! {
    display::initialize(&boot_info.graphic_info);
    _svc!(display::DISPLAY :mut).clear();

    console::initialize();
    println!("console initialized");

    logging::initialize();
    info!("logging initialized");

    interrupts::init();
    info!("interrupts initialized");

    let rs = unsafe { boot_info.system_table.runtime_services() };

    uefi_clock::initialize(rs);
    info!(
        "uefi clock initialized, now = {}",
        _svc!(uefi_clock::UEFI_CLOCK).now()
    );

    info!(
        "kernel loaded, firmware vendor={} version={:?}",
        boot_info.system_table.firmware_vendor(),
        boot_info.system_table.firmware_revision()
    );

    for mem in boot_info.memory_map.clone().iter {
        if mem.ty == boot::MemoryType::CONVENTIONAL {
            println!("{:?}", mem);
        }
    }

    unsafe {
        memory::init(
            VirtAddr::new_truncate(0xFFFF800000000000),
            &boot_info.memory_map,
        );
    }
    allocator::init_heap(
        memory::OFFSET_PAGE_TABLE.lock().as_mut().unwrap(),
        memory::FRAME_ALLOCATOR.lock().as_mut().unwrap(),
    );

    info!("memory allocator initialized");

    // allocate a number on the heap
    let heap_value = alloc::boxed::Box::new(41);
    println!("heap_value at {:p}", heap_value);

    // create a dynamically sized vector
    let mut vec = alloc::vec::Vec::new();
    for i in 0..500 {
        vec.push(i);
    }
    println!("vec at {:p}", vec.as_slice());

    // create a reference counted vector -> will be freed when count reaches 0
    let reference_counted = alloc::rc::Rc::new(alloc::vec![1, 2, 3]);
    let cloned_reference = reference_counted.clone();
    println!(
        "current reference count is {}",
        alloc::rc::Rc::strong_count(&cloned_reference)
    );
    core::mem::drop(reference_counted);
    println!(
        "reference count is {} now",
        alloc::rc::Rc::strong_count(&cloned_reference)
    );

    let mut buf = [0xFFu8; 512];
    let mut ide = drivers::ide::IDE::from_id(0);
    ide.init();
    ide.read_lba(0, 1, &mut buf);

    let part = fatpart::MBRPartitionTable::parse_sector(&buf).unwrap();
    info!("part: {:?}", part);
    info!("read = {}", part.partition0.begin_lba);
    ide.read_lba(part.partition0.begin_lba, 1, &mut buf);

    for i in 0..(512 / 16) {
        for j in 0..4 {
            for k in 0..4 {
                print!("{:02x}", buf[i * 16 + j * 4 + k]);
            }
            print!(" ");
        }
        println!();
    }

    info!("kernel exit, shutdown in 5s");

    _svc!(uefi_clock::UEFI_CLOCK).spin_wait_for_ns(5_000_000_000);

    unsafe {
        boot_info.system_table.runtime_services().reset(
            boot::ResetType::Shutdown,
            boot::UefiStatus::SUCCESS,
            None,
        );
    }
}
