use crate::prelude::*;
use ark_ff::{Field, PrimeField};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

// fold two lists of generators using two constants in bullet-proof kind fashion.
// i.e., we compute a new vector with entries: c_left * left_i + c_right * right_i, for all i.
// note: this assumes that both lists have the same length!!!
#[inline]
pub fn fold_generators<G: CurveGroup>(
    left: &[G::Affine],
    right: &[G::Affine],
    c_left: &G::ScalarField,
    c_right: &G::ScalarField,
) -> Vec<G::Affine> {
    ark_std::cfg_iter!(left)
        .zip(ark_std::cfg_iter!(right))
        .map(|(&l, &r)| (l * c_left + r * c_right).into_affine())
        .collect()
}

#[inline]
// compute the hadamard product of two slices (implicit zero padding is assumed for different lengths
pub fn hadamard<F: PrimeField>(a: &[F], b: &[F]) -> Vec<F> {
    ark_std::cfg_into_iter!(a)
        .zip(ark_std::cfg_into_iter!(b))
        .map(|(&a_val, b_val)| a_val * b_val)
        .collect()
}

#[inline]
// multiply each entry of a given slice with the scalar value c
pub fn scalar_times_slice<F: PrimeField>(slice: &[F], c: &F) -> Vec<F> {
    ark_std::cfg_into_iter!(slice).map(|&val| val * c).collect()
}

#[inline]
// fold two lists of scalars using two constants in bullet-proof kind fashion.
// i.e., we compute a new vector with entries: c_left * left_i + c_right * right_i, for all i.
// note: this assumes that the left slice has size greater or equal to the right slice
pub fn fold_scalars<F: PrimeField>(left: &[F], right: &[F], c_left: &F, c_right: &F) -> Vec<F> {
    let mut result: Vec<F> = ark_std::cfg_into_iter!(left)
        .zip(ark_std::cfg_into_iter!(right))
        .map(|(&l, &r)| l * c_left + r * c_right)
        .collect();
    // in the first iteration of the bullet-reduce x may be smaller (this is more efficient, but we do need to add this workaround here)
    if left.len() > right.len() {
        #[cfg(feature = "parallel")]
        result.par_extend(ark_std::cfg_into_iter!(left[right.len()..]).map(|&l| l * c_left));
        #[cfg(not(feature = "parallel"))]
        result.extend(ark_std::cfg_into_iter!(left[right.len()..]).map(|&l| l * c_left));
    }
    result
}

#[inline]
// multiply a row with a matrix, i.e., compute a linear combination of the rows of the matrix
// note: the first row of the matrix should have maximum length, the other rows may be shorter BUT NOT LONGER
pub fn row_mult_matrix<F: PrimeField>(row: &[F], matrix: &[Vec<F>]) -> Vec<F> {
    let mut result = matrix[0]
        .iter()
        .map(|&elem| elem * row[0])
        .collect::<Vec<_>>();
    for (row, &mult) in matrix[1..].iter().zip(row[1..].iter()) {
        ark_std::cfg_iter_mut!(result)
            .zip(ark_std::cfg_into_iter!(row))
            .for_each(|(a, b)| *a += mult * b);
    }
    result
}

#[inline]
// dot product, vectors need not be same size, since we can assume implicit zero-padding
pub fn dot<F: PrimeField>(v1: &[F], v2: &[F]) -> F {
    ark_std::cfg_iter!(v1)
        .zip(ark_std::cfg_iter!(v2))
        .map(|(&x, y)| x * y)
        .sum()
}

#[cfg(not(feature = "parallel"))]
// Based on distribute_powers_and_mul_by_const from ark-poly-0.5.0/src/domain/mod
// Compute an array where the i-th element is value^(i+1), i.e., [value, value^2, ..., value^n] if start_at_zero is false
// otherwise the i-th element is value^i, i.e., [1, value, value^2, ..., value^n-1].
#[inline]
pub fn powers<F: PrimeField>(value: &F, n: usize, start_at_zero: bool) -> Vec<F> {
    let mut result = vec![F::ONE; n];
    let to_fill = if start_at_zero {
        &mut result[1..]
    } else {
        result.as_mut_slice()
    };

    let mut pow = F::ONE;
    for entry in to_fill {
        pow *= value;
        *entry = pow;
    }
    result
}

#[cfg(feature = "parallel")]
// Based on distribute_powers_and_mul_by_const from ark-poly-0.5.0/src/domain/mod
// Compute an array where the i-th element is value^(i+1), i.e., [value, value^2, ..., value^n] if start_at_zero is false
// otherwise the i-th element is value^i, i.e., [1, value, value^2, ..., value^n-1].
#[inline]
pub fn powers<F: PrimeField>(value: &F, n: usize, start_at_zero: bool) -> Vec<F> {
    let min_parallel_chunk_size = 1024;
    let num_cpus_available = rayon::current_num_threads();
    let num_elem_per_thread = std::cmp::max(n / num_cpus_available, min_parallel_chunk_size);

    let mut result = vec![F::ONE; n];
    let to_fill = if start_at_zero {
        &mut result[1..]
    } else {
        result.as_mut_slice()
    };
    to_fill
        .par_chunks_mut(num_elem_per_thread)
        .enumerate()
        .for_each(|(i, chunk)| {
            let mut pow = value.pow([(i * num_elem_per_thread) as u64]);
            chunk.iter_mut().for_each(|entry| {
                pow *= value;
                *entry = pow;
            });
        });
    result
}

// Based on ark-poly-0.5.0/src/hyrax/utils.rs (since it is private), with optimizations:
// - precomputing the anti_val
// - removing the recursion (based on ark-poly-0.5.0/src/linear_codes/utils.rs)

// This function computes all evaluations of the MLE EQ(i, values) for i
// between 0...0 and 1...1 (n-bit strings).
// the order is LSB first, i.e., 00..00 > 10....0 > 01....0 > 11....0 > ............. > 01.....1 > 11.....1
// These are the \chi_b values from https://eprint.iacr.org/2017/1132.pdf Section 4 "Final step" that are used in Section 6.1 "Square-root commitment scheme", ordered from b=0, till b=2^n - 1, where n is the length of values
#[inline]
pub fn tensor_chi<F: Field>(values: &[F]) -> Vec<F> {
    let mut tail = vec![F::one()];

    for val in values.iter() {
        let anti_val = F::one() - val;
        tail = ark_std::cfg_iter!(tail)
            .map(|v| *v * anti_val)
            .chain(ark_std::cfg_iter!(tail).map(|v| *v * val))
            .collect();
    }

    tail
}

#[inline]
pub fn double_tensor_chi<F: Field>(values1: &[F], values2: &[F]) -> Vec<F> {
    let mut tail = vec![F::one()];

    for (val1, val2) in values1.iter().zip(values2) {
        let anti_val1 = F::one() - val1;
        let anti_val2 = F::one() - val2;
        let anti_val = anti_val1 * anti_val2;
        let val = *val1 * val2;
        tail = ark_std::cfg_iter!(tail)
            .map(|v| *v * anti_val)
            .chain(ark_std::cfg_iter!(tail).map(|v| *v * val))
            .collect();
    }

    tail
}

// computes c_0 + c_1 * x + c_2 x^2 + .... c_n * x^n
// assumes coeffs are in reverse order, i.e., (c_n, ..., c_1, c_0)
#[inline]
pub fn poly_eval<F: Field>(coeffs: &[F], x: &F) -> F {
    let mut monomial = F::ONE;
    coeffs
        .iter()
        .rev()
        .map(|&d| {
            let term = d * monomial;
            monomial *= x;
            term
        })
        .sum::<F>()
}
