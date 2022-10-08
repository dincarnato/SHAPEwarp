use std::iter::Sum;

use fftw::{
    array::{AlignedAllocable, AlignedVec},
    plan::C2CPlan,
    types::{Flag, Sign},
};
use num_complex::Complex;
use num_traits::{cast, float::FloatCore, Float, NumAssign, NumOps, NumRef, RefNum};

use crate::{mean_stddev, C2CPlanExt};

pub(crate) struct Mass<T: C2CPlanExt> {
    fw_plan: T::Plan,
    bw_plan: T::Plan,
    aligned_query: AlignedVec<Complex<T>>,
    query_transform: AlignedVec<Complex<T>>,
    product: AlignedVec<Complex<T>>,
    product_inverse: AlignedVec<Complex<T>>,
}

impl<T> Mass<T>
where
    T: C2CPlanExt
        + Float
        + NumRef
        + Sum
        + for<'a> Sum<&'a T>
        + ComplexExt
        + NumAssign
        + NumOps<Complex<T>, Complex<T>>
        + 'static,
    for<'a> &'a T: RefNum<T>,
    Complex<T>: AlignedAllocable + NumOps<T>,
{
    pub(crate) fn new(size: usize) -> Result<Self, fftw::error::Error> {
        let fw_plan: T::Plan = C2CPlan::aligned(&[size], Sign::Forward, Flag::ESTIMATE)?;
        let bw_plan: T::Plan = C2CPlan::aligned(&[size], Sign::Backward, Flag::ESTIMATE)?;
        let aligned_query = AlignedVec::new(size);
        let query_transform = aligned_query.clone();
        let product = aligned_query.clone();
        let product_inverse = aligned_query.clone();

        Ok(Self {
            fw_plan,
            bw_plan,
            aligned_query,
            query_transform,
            product,
            product_inverse,
        })
    }

    pub(crate) fn run(
        &mut self,
        db: &[T],
        db_transform: &AlignedVec<Complex<T>>,
        query: &[T],
    ) -> Result<Vec<Complex<T>>, fftw::error::Error> {
        let ts_len = db_transform.len();
        let query_len = query.len();

        query
            .iter()
            .rev()
            .copied()
            .zip(self.aligned_query.iter_mut())
            .for_each(|(q, y)| y.re = q);
        self.fw_plan
            .c2c(&mut self.aligned_query, &mut self.query_transform)?;

        self.product
            .iter_mut()
            .zip(&**db_transform)
            .zip(&*self.query_transform)
            .for_each(|((z, x), y)| *z = x * y);

        self.bw_plan
            .c2c(&mut self.product, &mut self.product_inverse)?;

        // Normalize results
        let scale_factor: T = T::one() / cast::<_, T>(ts_len).unwrap();
        for z in &mut *self.product_inverse {
            *z *= scale_factor;
        }

        let mean_sigma_x = db.windows(query_len).map(|window| mean_stddev(window, 0));
        let (mean_y, sigma_y) = mean_stddev(query, 0);

        let query_len_float: T = cast(query_len).unwrap();
        Ok(self.product_inverse[query_len - 1..ts_len]
            .iter()
            .zip(mean_sigma_x)
            .map(|(z, (mean_x, sigma_x))| {
                let squared = T::from_f64(2.).unwrap()
                    * (query_len_float
                        - (z - query_len_float * mean_x * mean_y) / (sigma_x * sigma_y));
                squared.sqrt()
            })
            .collect())
    }
}

pub(crate) trait ComplexExt {
    fn sqrt(&self) -> Self;
    fn powi(&self, n: i32) -> Self;
    fn is_finite(&self) -> bool;
}

impl ComplexExt for f64 {
    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
    }

    fn powi(&self, n: i32) -> Self {
        f64::powi(*self, n)
    }

    fn is_finite(&self) -> bool {
        f64::is_finite(*self)
    }
}

impl ComplexExt for f32 {
    fn sqrt(&self) -> Self {
        f32::sqrt(*self)
    }

    fn powi(&self, n: i32) -> Self {
        f32::powi(*self, n)
    }

    fn is_finite(&self) -> bool {
        f32::is_finite(*self)
    }
}

impl<T> ComplexExt for Complex<T>
where
    T: Float + FloatCore,
{
    fn sqrt(&self) -> Self {
        <Complex<T>>::sqrt(*self)
    }

    fn powi(&self, n: i32) -> Self {
        Complex::powi(self, n)
    }

    fn is_finite(&self) -> bool {
        Complex::is_finite(*self)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;
    use crate::transform_db;

    #[test]
    fn test_mass() {
        let ts = [1f64, 1., 1., 2., 1., 1., 4., 5.];
        let ts_t = transform_db(&ts).unwrap();
        let query = [2., 1., 1., 4.];
        let result = Mass::new(ts.len())
            .unwrap()
            .run(ts.as_ref(), &ts_t, query.as_ref())
            .unwrap();

        const EXPECTED: [Complex<f64>; 5] = [
            Complex::new(0.67640791, -1.37044402e-16),
            Complex::new(3.43092352, 0.00000000e+00),
            Complex::new(3.43092352, 1.02889035e-17),
            Complex::new(0., 0.),
            Complex::new(1.85113597, 1.21452707e-17),
        ];

        assert_abs_diff_eq!(&*result, EXPECTED.as_ref(), epsilon = 1e-7);
    }
}
