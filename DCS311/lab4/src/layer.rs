use crate::{Activate, Function};
use nalgebra::{base::allocator::Allocator, DefaultAllocator, DimName, MatrixMN, VectorN, U1};

pub struct Context {
    pub eta: f64,
}

impl Context {
    pub fn new(eta: f64) -> Self {
        Self { eta }
    }
}

pub struct Layer<I: DimName, O: DimName>
where
    DefaultAllocator: Allocator<f64, I, O>,
    DefaultAllocator: Allocator<f64, O, I>,
    DefaultAllocator: Allocator<f64, I, U1>,
    DefaultAllocator: Allocator<f64, U1, I>,
    DefaultAllocator: Allocator<f64, O, U1>,
    DefaultAllocator: Allocator<f64, U1, O>,
{
    pub weight: MatrixMN<f64, O, I>,
    pub bias: VectorN<f64, O>,
}

impl<I: DimName, O: DimName> Layer<I, O>
where
    DefaultAllocator: Allocator<f64, I, O>,
    DefaultAllocator: Allocator<f64, O, I>,
    DefaultAllocator: Allocator<f64, I, U1>,
    DefaultAllocator: Allocator<f64, U1, I>,
    DefaultAllocator: Allocator<f64, O, U1>,
    DefaultAllocator: Allocator<f64, U1, O>,
{
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let dist = rand::distributions::Uniform::new(-1.0, 1.0);
        Self {
            weight: MatrixMN::<f64, O, I>::from_distribution(&dist, &mut rng),
            bias: VectorN::<f64, O>::from_distribution(&dist, &mut rng),
        }
    }

    /// 正向传播，输出没有激活的结果，即 $z^l$
    pub fn forward_pass(&self, input: &VectorN<f64, I>) -> VectorN<f64, O> {
        &self.weight * input + &self.bias
    }

    /// 正向传播，输出经过激活的结果，即 $a^l$
    pub fn forward_pass_activate(&self, input: &VectorN<f64, I>) -> VectorN<f64, O> {
        Function::activate_vec(&self.forward_pass(input))
    }

    pub fn error_grade<O2: DimName>(
        &self,
        output: &VectorN<f64, O>,
        next_grade: &VectorN<f64, O2>,
        next_layer: &Layer<O, O2>,
    ) -> VectorN<f64, O>
    where
        DefaultAllocator: Allocator<f64, O, O2>,
        DefaultAllocator: Allocator<f64, O2, O>,
        DefaultAllocator: Allocator<f64, O2, U1>,
        DefaultAllocator: Allocator<f64, U1, O2>,
    {
        Function::derive_vec(output).component_mul(&(next_layer.weight.transpose() * next_grade))
    }

    #[allow(clippy::type_complexity)]
    pub fn back_propogation<'e, O2: DimName>(
        &mut self,
        input: impl Iterator<Item = &'e VectorN<f64, I>>,
        output: impl Iterator<Item = &'e VectorN<f64, O>>,
        grade: impl Iterator<Item = &'e VectorN<f64, O2>>,
        next_layer: &Layer<O, O2>,
    ) -> (Vec<VectorN<f64, O>>, MatrixMN<f64, O, I>, VectorN<f64, O>)
    where
        DefaultAllocator: Allocator<f64, O, O2>,
        DefaultAllocator: Allocator<f64, O2, O>,
        DefaultAllocator: Allocator<f64, O2, U1>,
        DefaultAllocator: Allocator<f64, U1, O2>,
    {
        let mut dw = MatrixMN::<_, O, I>::zeros();
        let mut db = VectorN::<_, O>::zeros();
        let mut grades = Vec::new();

        // 对于每一组数据点
        for ((input, output), next_grade) in input.zip(output).zip(grade) {
            // 计算 $\delta$ 并累加
            let grade = self.error_grade(output, next_grade, next_layer);
            // 计算 $\Delta\matv{W}$ 并累加
            dw += &grade * input.transpose();
            // 计算 $\Delta\matv{b}$ 并累加
            db += &grade;
            grades.push(grade);
        }

        (grades, dw, db)
    }

    pub fn apply_dw_db(
        &mut self,
        dw: &MatrixMN<f64, O, I>,
        db: &VectorN<f64, O>,
        eta: f64,
        m: usize,
    ) {
        self.weight -= dw * eta / m as f64;
        self.bias -= db * eta / m as f64;
    }
}
