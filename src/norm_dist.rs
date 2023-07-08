use std::ops::Deref;

use num_traits::AsPrimitive;
use once_cell::unsync::OnceCell;

pub struct NormDist<D> {
    data: D,
    mean: OnceCell<f64>,
    stddev: OnceCell<f64>,
}

impl<D> NormDist<D> {
    #[inline]
    pub fn from_sample(data: D) -> Self {
        Self {
            data,
            mean: OnceCell::new(),
            stddev: OnceCell::new(),
        }
    }
}

impl<D, T> NormDist<D>
where
    D: Deref<Target = [T]>,
    T: AsPrimitive<f64>,
{
    pub fn z_score(&self, value: T) -> f64 {
        let mean = self.mean();
        let stddev = self.stddev();

        (value.as_() - mean) / stddev
    }

    pub fn mean(&self) -> f64 {
        *self.mean.get_or_init(|| {
            let len = self.data.len();
            if len == 0 {
                0.
            } else {
                // It is fine to evaluate the mean
                #[allow(clippy::cast_precision_loss)]
                let len_recip = (len as f64).recip();
                self.data.iter().map(|x| x.as_() * len_recip).sum()
            }
        })
    }

    pub fn stddev(&self) -> f64 {
        *self.stddev.get_or_init(|| {
            self.data.len().checked_sub(1).map_or(0., |adj_len| {
                // It is fine to evaluate the variance
                #[allow(clippy::cast_precision_loss)]
                let denominator = (adj_len as f64).recip();

                let mean = self.mean();
                let variance: f64 = self
                    .data
                    .iter()
                    .map(|x| (x.as_() - mean).powi(2) * denominator)
                    .sum();

                variance.sqrt()
            })
        })
    }
}
