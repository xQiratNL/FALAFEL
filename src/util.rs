//! Handy utility functions

use crate::prelude::Value;
use ark_ff::PrimeField;
use ndarray::Order::ColumnMajor;
use ndarray::{Array2, Axis, Dim, Ix, StrideShape};
use num_traits::FromPrimitive;
use rand::Rng;
use rand::distributions::Distribution;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub trait RandomMatrix
where
    Self: Sized,
{
    fn random_matrix(
        n_rows: usize,
        n_cols: usize,
        distribution: &impl Distribution<f64>,
        decimal_bits: usize,
        rng: &mut impl Rng,
    ) -> Array2<Self>;
}

pub trait IntoField {
    fn to_field<F: PrimeField>(&self) -> F;
}

impl<T: Value> IntoField for T {
    fn to_field<F: PrimeField>(&self) -> F {
        if self.is_negative() {
            F::from_le_bytes_mod_order(self.abs().to_le_bytes().as_ref()).neg()
        } else {
            F::from_le_bytes_mod_order(self.to_le_bytes().as_ref())
        }
    }
}

impl<T: FromPrimitive> RandomMatrix for T {
    fn random_matrix(
        n_rows: usize,
        n_cols: usize,
        distribution: &impl Distribution<f64>,
        decimal_bits: usize,
        rng: &mut impl Rng,
    ) -> Array2<Self> {
        let trunc_factor = (1_usize << decimal_bits) as f64;
        Array2::from_shape_simple_fn((n_rows, n_cols), || {
            T::from_f64(distribution.sample(rng) * trunc_factor)
                .expect("This conversion should not fail!")
        })
    }
}

#[inline]
pub fn get_iv(client_id: usize, global_iter_count: usize) -> String {
    format!(
        "FALAFEL - Local Training Proof | Client {} | Global Iteration {}",
        client_id, global_iter_count
    )
}

#[inline]
pub fn permute<T: Clone, Sh: Into<StrideShape<Dim<[Ix; 2]>>>>(
    array: &[T],
    shape: Sh,
    permutation: &[usize],
) -> Vec<T> {
    Array2::from_shape_vec(shape, array.to_vec())
        .expect("This should fit exactly!")
        .select(Axis(1), &permutation)
        .flatten_with_order(ColumnMajor)
        .to_vec()
}

// Inspired by ark-poly-0.5.0/src/hyrax/utils.rs, but more efficient
#[inline]
pub fn flat_to_2dvec_almost_square_column_major<T: Copy + Send + Sync>(
    flat: &[T],
    n_rows: usize,
) -> Vec<Vec<T>> {
    ark_std::cfg_into_iter!((0..n_rows))
        .map(|mut idx| {
            let mut row = vec![];
            while idx < flat.len() {
                row.push(flat[idx]);
                idx += n_rows;
            }
            row
        })
        .collect::<Vec<_>>()
}
