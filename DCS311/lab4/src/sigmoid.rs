use nalgebra::{DimName, VectorN};

pub trait Activate {
    fn activate(x: f64) -> f64;
    fn activate_vec<N: DimName>(x: &VectorN<f64, N>) -> VectorN<f64, N>
    where
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, N, nalgebra::U1>,
    {
        x.map(Self::activate)
    }

    fn derive(x: f64) -> f64;
    fn derive_vec<N: DimName>(x: &VectorN<f64, N>) -> VectorN<f64, N>
    where
        nalgebra::DefaultAllocator: nalgebra::base::allocator::Allocator<f64, N, nalgebra::U1>,
    {
        x.map(Self::derive)
    }
}

#[allow(dead_code)]
pub struct Sigmoid;

impl Activate for Sigmoid {
    fn activate(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn derive(x: f64) -> f64 {
        x * (1.0 - x)
    }
}

pub struct ReLU;

impl Activate for ReLU {
    fn activate(x: f64) -> f64 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }

    fn derive(x: f64) -> f64 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

pub type Function = ReLU;

#[cfg(test)]
#[test]
fn sigmoid() {
    assert_eq!(
        Sigmoid::activate_vec(&VectorN::<f64, nalgebra::U5>::new(0.1, 0.2, 0.4, 0.8, 1.0)),
        VectorN::<f64, nalgebra::U5>::new(
            Sigmoid::activate(0.1),
            Sigmoid::activate(0.2),
            Sigmoid::activate(0.4),
            Sigmoid::activate(0.8),
            Sigmoid::activate(1.0)
        )
    );
}
