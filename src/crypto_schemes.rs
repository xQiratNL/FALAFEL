//! Cryptographic primitives

use crate::prelude::*;
use crate::util::flat_to_2dvec_almost_square_column_major;
use ark_ff::PrimeField;
use ark_std::{UniformRand, log2};
use blake2::{Blake2s256, Digest};
use ndarray::{ArrayBase, Data, Ix2};
use num_traits::Euclid;
use rand::{CryptoRng, Rng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::cmp::max;
use std::ops::Mul;

pub struct Constants<F> {
    pub one_over_two: F,
    pub one_over_three: F,
    pub one_over_four: F,
    pub one_over_twentyseven: F,
}
impl<F: PrimeField> Constants<F> {
    pub fn new() -> Self {
        // compute consts for polynomial interpolation
        Self {
            one_over_two: F::from(2).inverse().expect("Inverse of two should exist."),
            one_over_three: F::from(3)
                .inverse()
                .expect("Inverse of three should exist."),
            one_over_four: F::from(4).inverse().expect("Inverse of four should exist."),
            one_over_twentyseven: F::from(27).inverse().expect("Inverse of 27 should exist."),
        }
    }
}

/// This object contains a long list of generators, which can be used to obtain sufficient
/// generators for all commitment schemes. I.e., the commitment schemes themselves should not do a
/// setup, they can just get the generators from here. This prevents unnecessary many generators.
pub struct Generators<G: CurveGroup> {
    pub reusable_gens: Vec<G::Affine>,
}

impl<G: CurveGroup> Generators<G> {
    pub fn new(n_pedersen_vec_gens: usize) -> Self {
        // The following block of code is taken from ark-poly-commit-0.5.0, specifically the
        // function `sample_generators` in the file src/ipa_pc/mod.rs. Minor adaptations were made
        // to fit our purposes.

        // vec_g should be power of two, we need one extra unique_g, and we need h
        let n_gens = (n_pedersen_vec_gens.next_power_of_two() + 2) as u64;

        let gens: Vec<_> = ark_std::cfg_into_iter!(0..n_gens)
            .map(|i| {
                let mut hash =
                    Blake2s256::digest([PROTOCOL_NAME, &i.to_le_bytes()].concat().as_slice());
                let mut g = G::Affine::from_random_bytes(&hash);
                let mut j = 0u64;
                while g.is_none() {
                    // PROTOCOL NAME, i, j
                    let mut bytes = PROTOCOL_NAME.to_vec();
                    bytes.extend(i.to_le_bytes());
                    bytes.extend(j.to_le_bytes());
                    hash = Blake2s256::digest(bytes.as_slice());
                    g = G::Affine::from_random_bytes(&hash);
                    j += 1;
                }
                let generator = g.unwrap();
                generator.mul_by_cofactor_to_group()
            })
            .collect();
        Self {
            reusable_gens: G::normalize_batch(&gens),
        }
    }
}

pub struct PedersenVectorCommitmentScheme<'a, G: CurveGroup> {
    g_vec: &'a [G::Affine],
    g_unique: &'a G::Affine,
    h: &'a G::Affine,
}

impl<'a, G: CurveGroup> PedersenVectorCommitmentScheme<'a, G> {
    pub fn setup(generators: &'a Generators<G>) -> Self {
        Self {
            g_vec: &generators.reusable_gens[..generators.reusable_gens.len() - 2],
            g_unique: &generators.reusable_gens[generators.reusable_gens.len() - 2],
            h: &generators.reusable_gens[generators.reusable_gens.len() - 1],
        }
    }
    pub fn vector_commit(
        &self,
        values: &[G::ScalarField],
        randomizer: &G::ScalarField,
    ) -> G::Affine {
        (G::msm_unchecked(&self.g_vec[0..values.len()], values) + self.h.mul(randomizer)).into()
    }

    pub fn value_commit_unique_g_same_h(
        &self,
        value: &G::ScalarField,
        randomizer: &G::ScalarField,
    ) -> G::Affine {
        (self.g_unique.mul(value) + self.h.mul(randomizer)).into()
    }

    pub fn get_g_vec(&self) -> &'a [G::Affine] {
        self.g_vec
    }

    pub fn get_g_unique(&self) -> &'a G::Affine {
        self.g_unique
    }

    pub fn get_h(&self) -> &'a G::Affine {
        self.h
    }
}

// The hyrax structs and implementations below are very loosely inspired by the files in ark-poly-0.5.0/src/hyrax/

// the dimension is the dimension of the underlying square matrix (i.e., number of rows/columns).
// Note1: We do not store zero-padding, so the number of columns may differ per row, and/or be smaller than the dimension. But it will always be structured such that the first rows have length x and the remaining ones have length <x, possible 0 (where x is some value between 1 and dim-1).
// Note2: The number of rows equals the dimension.
// Note3: We follow column major order (as defined in Hyrax), from the vector input that was given originally.
pub struct HyraxCommitment<G: CurveGroup> {
    pub row_coms: Vec<G::Affine>,
}

impl<G: CurveGroup> HyraxCommitment<G> {
    // get all row_coms as group elements (we use this a lot, so convenient to extract)
    #[inline]
    pub fn row_coms_as_group(&self) -> Vec<G> {
        self.row_coms
            .iter()
            .map(|x| x.into_group())
            .collect::<Vec<_>>()
    }
}
#[derive(Clone, Debug)]
pub struct HyraxCommitmentOpening<G: CurveGroup> {
    pub input_matrix: Vec<Vec<G::ScalarField>>,
    pub randomness: Vec<G::ScalarField>,
}

pub struct HyraxPolynomialCommitmentScheme<'a, G: CurveGroup> {
    pedersen_vector_commitment_scheme: PedersenVectorCommitmentScheme<'a, G>,
}

impl<'a, G: CurveGroup> HyraxPolynomialCommitmentScheme<'a, G> {
    pub fn setup(generators: &'a Generators<G>) -> Self {
        Self {
            pedersen_vector_commitment_scheme: PedersenVectorCommitmentScheme::setup(generators),
        }
    }

    pub fn combine_matrices<S: Data<Elem = G::ScalarField> + Sync>(
        &self,
        matrices: &[ArrayBase<S, Ix2>],
        padding: G::ScalarField,
        keep_matrices_separate: bool,
        custom_row_com_dim_log: Option<usize>,
    ) -> Vec<Vec<G::ScalarField>> {
        // figure out the dimensions
        let matrix_idx_dim_log = log2(matrices.len());
        let row_idx_dim_log = log2(
            matrices
                .iter()
                .map(|matrix| matrix.nrows())
                .max()
                .expect("Matrices should contain at least one matrix!"),
        );
        let col_idx_dim_log = log2(
            matrices
                .iter()
                .map(|matrix| matrix.ncols())
                .max()
                .expect("Matrices should have positive dimensions!"),
        );
        let total_dim_log = (matrix_idx_dim_log + row_idx_dim_log + col_idx_dim_log) as usize;
        let num_row_coms_log = if let Some(custom_row_com_dim_log) = custom_row_com_dim_log {
            total_dim_log - custom_row_com_dim_log
        } else {
            if keep_matrices_separate {
                max(total_dim_log / 2, matrix_idx_dim_log as usize)
            } else {
                total_dim_log / 2
            }
        };
        let num_row_coms = 1 << num_row_coms_log;
        let row_com_dim = 1 << (total_dim_log - num_row_coms_log);
        let n_matrices = 1 << matrix_idx_dim_log;
        let n_rows = 1 << row_idx_dim_log;

        // parse the matrices into one big matrix of num_row_coms x row_com_dimension, we need to add zero padding for the matrices themselves as otherwise it won't work
        ark_std::cfg_into_iter!((0..num_row_coms))
            .map(|mut idx_matrix_row_col| {
                let mut row = vec![];

                for _ in 0..row_com_dim {
                    let (idx_row_col, idx_matrix) = idx_matrix_row_col.div_rem_euclid(&n_matrices);
                    let (idx_col, idx_row) = idx_row_col.div_rem_euclid(&n_rows);
                    row.push(if let Some(matrix) = matrices.get(idx_matrix) {
                        matrix.get((idx_row, idx_col)).copied().unwrap_or(padding)
                    } else {
                        padding
                    });
                    idx_matrix_row_col += num_row_coms;
                }
                row
            })
            .collect::<Vec<_>>()
    }

    // we assume that the input is a vector of matrices, which may have different dimensions
    // all matrices themselves should be dense, i.e., each row in a matrix should have the same amount of elements.
    pub fn poly_commit_matrices<S: Data<Elem = G::ScalarField> + Sync>(
        &self,
        matrices: &[ArrayBase<S, Ix2>],
        padding: G::ScalarField,
        keep_matrices_separate: bool,
        custom_row_com_dim_log: Option<usize>,
        rng: &mut (impl Rng + CryptoRng),
    ) -> (HyraxCommitment<G>, HyraxCommitmentOpening<G>) {
        let matrix = self.combine_matrices(
            matrices,
            padding,
            keep_matrices_separate,
            custom_row_com_dim_log,
        );
        self.commit_directly(matrix, rng)
    }

    // this method is mainly for testing purposes, as we never commit to any vectors :)
    pub fn poly_commit_vec(
        &self,
        values: &[G::ScalarField],
        rng: &mut (impl Rng + CryptoRng),
    ) -> (HyraxCommitment<G>, HyraxCommitmentOpening<G>) {
        // hyrax wants a matrix that is of size 2^(l) x 2^(l'), where 2^(l +l') is the length of values and l' == l or l' == l + 1
        let n_rows = 1 << log2(values.len()) / 2;
        let matrix = flat_to_2dvec_almost_square_column_major(values, n_rows);

        // actual commit
        self.commit_directly(matrix, rng)
    }

    // we assume the matrix is already formatted in the right way (incl. zero rows at the bottom, to make sure the number of rows is a power of 2.)
    // the length of each row should not be larger than the number of rows
    #[inline]
    pub fn commit_directly(
        &self,
        matrix: Vec<Vec<G::ScalarField>>,
        rng: &mut (impl Rng + CryptoRng),
    ) -> (HyraxCommitment<G>, HyraxCommitmentOpening<G>) {
        let commitment_opening = HyraxCommitmentOpening::<G> {
            // no need to parallelize this, it is a short vector, and otherwise we need to make one rng per thread, which probably costs more
            randomness: (0..matrix.len())
                .map(|_| G::ScalarField::rand(rng))
                .collect::<Vec<_>>(),
            input_matrix: matrix,
        };

        let commitment = HyraxCommitment {
            row_coms: ark_std::cfg_iter!(&commitment_opening.input_matrix)
                .zip(ark_std::cfg_iter!(&commitment_opening.randomness))
                .map(|(row, row_randomness)| {
                    self.pedersen_vector_commitment_scheme
                        .vector_commit(row, row_randomness)
                })
                .collect::<Vec<_>>(),
        };

        (commitment, commitment_opening)
    }
}

#[cfg(test)]
mod test {
    use crate::client::ClientSchemes;
    use crate::crypto_schemes::Generators;
    use ark_ec::PrimeGroup;
    use ark_ff::AdditiveGroup;
    use ndarray::array;
    use rand::SeedableRng;
    use rand_chacha::ChaChaRng;

    type G = ark_curve25519::EdwardsProjective;

    #[test]
    fn test_matrix_commit() {
        // schemes setup
        let mut crypto_rng = ChaChaRng::from_entropy();
        let generators = Generators::<G>::new(100);
        let schemes = ClientSchemes::setup(&generators);

        // this is a pretty decent test case that should include all possibilities
        let matrices = vec![
            array![
                [
                    <G as PrimeGroup>::ScalarField::from(1),
                    <G as PrimeGroup>::ScalarField::from(8),
                ],
                [
                    <G as PrimeGroup>::ScalarField::from(4),
                    <G as PrimeGroup>::ScalarField::from(10),
                ],
            ],
            array![
                [
                    <G as PrimeGroup>::ScalarField::from(2),
                    <G as PrimeGroup>::ScalarField::from(9),
                ],
                [
                    <G as PrimeGroup>::ScalarField::from(5),
                    <G as PrimeGroup>::ScalarField::from(11),
                ],
                [
                    <G as PrimeGroup>::ScalarField::from(7),
                    <G as PrimeGroup>::ScalarField::from(12),
                ],
            ],
            array![
                [<G as PrimeGroup>::ScalarField::from(3)],
                [<G as PrimeGroup>::ScalarField::from(6)],
            ],
        ];

        let found = schemes.poly_commit.poly_commit_matrices(
            matrices.as_slice(),
            <G as PrimeGroup>::ScalarField::ZERO,
            false,
            None,
            &mut crypto_rng,
        );

        let expected = schemes.poly_commit.poly_commit_vec(
            &[
                1, 2, 3, 0, 4, 5, 6, 0, 0, 7, 0, 0, 0, 0, 0, 0, 8, 9, 0, 0, 10, 11, 0, 0, 0, 12, 0,
                0, 0, 0, 0, 0,
            ]
            .map(|x| <G as PrimeGroup>::ScalarField::from(x)),
            &mut crypto_rng,
        );
        assert_eq!(expected.1.input_matrix, found.1.input_matrix);
    }
}
