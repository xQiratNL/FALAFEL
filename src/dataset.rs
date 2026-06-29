//! Dataset structure, plus functions for shuffling and creating batches.

use crate::prelude::*;
use crate::util::RandomMatrix;
use ndarray::iter::AxisChunksIter;
use ndarray::{Array2, ArrayView2, Axis, Ix2};
use rand::Rng;
use rand::prelude::{Distribution, SliceRandom};

#[derive(Clone)]
pub struct Dataset<T: Value> {
    samples: Array2<T>,
    labels: Array2<T>,
}

impl<T: Value> Dataset<T> {
    pub fn shuffle(&mut self, rng: &mut impl Rng) -> Vec<usize> {
        let mut indices = (0..self.samples.ncols()).collect::<Vec<_>>();
        indices.shuffle(rng);
        self.samples = self.samples.select(Axis(1), &indices);
        self.labels = self.labels.select(Axis(1), &indices);
        indices
    }

    pub fn batches(
        &self,
        batch_size: usize,
    ) -> (AxisChunksIter<'_, T, Ix2>, AxisChunksIter<'_, T, Ix2>) {
        (
            self.samples.axis_chunks_iter(Axis(1), batch_size),
            self.labels.axis_chunks_iter(Axis(1), batch_size),
        )
    }

    pub fn get_samples(&self) -> ArrayView2<'_, T> {
        self.samples.view()
    }

    pub fn get_labels(&self) -> ArrayView2<'_, T> {
        self.labels.view()
    }
}

impl<T: Value + RandomMatrix> Dataset<T> {
    pub fn new_random(
        n_samples: usize,
        n_features: usize,
        n_labels: usize,
        distribution: &impl Distribution<f64>,
        decimal_bits: usize,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            samples: T::random_matrix(n_features, n_samples, distribution, decimal_bits, rng),
            labels: T::random_matrix(n_labels, n_samples, distribution, decimal_bits, rng),
        }
    }
}

impl<T: Value> Dataset<T> {
    pub fn new(samples: Array2<T>, labels: Array2<T>) -> Self {
        assert_eq!(
            samples.shape()[1],
            labels.shape()[1],
            "Sample and label counts must match"
        );

        Self { samples, labels }
    }
}
