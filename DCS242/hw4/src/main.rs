use crossbeam::thread;
use rand::Rng;
use std::time::SystemTime;

const MATRIX_M: usize = 1000;
const MATRIX_N: usize = 1000;

macro_rules! run_timed {
    ($f1:expr) => {{
        let now = SystemTime::now();
        $f1;
        let time = now.elapsed().unwrap().as_micros();
        time
    }};
}

fn main() {
    let mut rng = rand::thread_rng();

    let lhs = Matrix::new_rand(MATRIX_M, MATRIX_N, &mut rng);
    let rhs = Matrix::new_rand(MATRIX_N, 1, &mut rng);
    let mut ret = Matrix::new(MATRIX_N, 1);

    let time = run_timed!(thread_sum(&lhs, &rhs, &mut ret));
    println!("calculate finished, time = {}ms", time);

    for i in 0..MATRIX_M {
        let mut sum = 0;
        for k in 0..MATRIX_N {
            sum += lhs.get(i, k) * rhs.get(k, 0);
        }
        if ret.get(i, 0) != &sum {
            println!("ineq ret={} sum={}", ret.get(i, 0), sum);
        }
    }
    println!("calculate succeeded");
}

struct Matrix<T> {
    pub row_cnt: usize,
    pub col_cnt: usize,
    pub buf: Vec<T>,
}

impl Matrix<u32> {
    pub fn new_rand<T: Rng>(row_cnt: usize, col_cnt: usize, rng: &mut T) -> Self {
        let mut val = Self {
            row_cnt,
            col_cnt,
            buf: Vec::with_capacity(row_cnt * col_cnt),
        };
        for _ in 0..row_cnt * col_cnt {
            val.buf.push(rng.gen_range(0, 20));
        }
        val
    }
}

impl<T> Matrix<T>
where
    T: From<u32>,
{
    pub fn new(row_cnt: usize, col_cnt: usize) -> Self {
        let mut val = Self {
            row_cnt,
            col_cnt,
            buf: Vec::with_capacity(row_cnt * col_cnt),
        };
        for _ in 0..row_cnt * col_cnt {
            val.buf.push(0.into());
        }
        val
    }

    pub fn get(&self, row: usize, col: usize) -> &T {
        &self.buf[row * self.col_cnt + col]
    }

    pub fn buf(&self) -> &Vec<T> {
        &self.buf
    }
    pub fn buf_mut(&mut self) -> &mut Vec<T> {
        &mut self.buf
    }
}

fn thread_sum<'a, T>(lhs: &Matrix<T>, rhs: &Matrix<T>, ret: &mut Matrix<T>)
where
    T: Send + Sync + std::ops::AddAssign + std::ops::Mul<Output = T> + Copy,
    T: From<u32>,
{
    assert_eq!(lhs.row_cnt % THREAD_COUNT as usize, 0);
    assert_eq!(lhs.col_cnt, rhs.row_cnt);

    const THREAD_COUNT: u8 = 8;

    let mut lhs_buf = lhs.buf().as_slice();
    let rhs_buf = rhs.buf().as_slice();
    let mut ret_buf = ret.buf_mut().as_mut_slice();

    let block_size = lhs.row_cnt / THREAD_COUNT as usize;
    thread::scope(move |s| {
        for _ in 0..THREAD_COUNT {
            let (lhs_cur, lhs_rest) = lhs_buf.split_at(block_size * lhs.col_cnt);
            let (ret_cur, ret_rest) = ret_buf.split_at_mut(block_size);
            lhs_buf = lhs_rest;
            ret_buf = ret_rest;
            s.spawn(move |_| {
                thread_compute(lhs_cur, rhs_buf, ret_cur, lhs.col_cnt);
            });
        }
    })
    .unwrap()
}

fn thread_compute<T>(lhs: &[T], rhs: &[T], ret: &mut [T], atom_len: usize)
where
    T: Send + Sync + std::ops::AddAssign + std::ops::Mul<Output = T> + Copy,
    T: From<u32>,
{
    let cols = lhs.len() / atom_len;
    assert_eq!(atom_len, rhs.len());
    assert_eq!(cols * atom_len, lhs.len());
    // for each independent row
    for i in 0..cols {
        // for each cell in the row
        for k in 0..atom_len {
            ret[i] += lhs[i * atom_len + k] * rhs[k];
        }
    }
}
