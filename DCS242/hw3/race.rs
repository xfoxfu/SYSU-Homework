fn Thread1(Global: &mut isize) {
    *Global = 42;
}

fn main() {
    let mut Global: isize = 0;
    let th = std::thread::spawn(|| Thread1(&mut Global));
    Global = 43;
    th.join();
}
