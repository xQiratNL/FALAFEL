use crate::client::ClientSchemes;
use crate::crypto_schemes::HyraxCommitmentOpening;
use crate::proof::hyrax_poly_eval::{
    HyraxPolyEvalProof, HyraxPolyEvalProofSeparator, HyraxTargetCommitment,
};
use crate::proof::sumcheck::util::{
    collapse, get_coeffs_degree_2, get_coeffs_degree_3, multi_collapse, point_evals_degree_2,
    point_evals_degree_3,
};
use crate::proof::sumcheck::{SumCheckProof, SumCheckProofSeparator};
use crate::proof::util::{dot, double_tensor_chi, poly_eval, tensor_chi};
use crate::util::flat_to_2dvec_almost_square_column_major;
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{AdditiveGroup, Field};
use ark_std::UniformRand;
use num_traits::Euclid;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::codecs::arkworks_algebra::{
    FieldDomainSeparator, GroupDomainSeparator, GroupToUnitDeserialize, GroupToUnitSerialize,
    UnitToField,
};
use spongefish::{DomainSeparator, DuplexSpongeInterface, ProofResult, ProverState, VerifierState};
use std::cmp::min;

pub trait BackwardPassSumCheck2ProofSeparator<G: CurveGroup> {
    fn add_backward_pass_sum_check2(
        self,
        r_1_len: usize,
        r_2_len: usize,
        r_3_len: usize,
        r_4_len: usize,
    ) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> BackwardPassSumCheck2ProofSeparator<G>
    for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    fn add_backward_pass_sum_check2(
        mut self,
        r_1_len: usize,
        r_2_len: usize,
        r_3_len: usize,
        r_4_len: usize,
    ) -> Self {
        // we assume the commitments to the a, b, c matrices have already been sent
        self = self
            .challenge_scalars(r_1_len, "r_1")
            .challenge_scalars(r_2_len, "r_2")
            .challenge_scalars(r_4_len, "r_4")
            .add_hyrax_target_commitment();
        for round in 0..r_3_len {
            self = self
                .add_points(1, &format!("r_3; round {round} coefficients commitment"))
                .challenge_scalars(1, &format!("r_3; round {round}"));
        }
        for round in 0..r_4_len {
            self = self
                .add_points(
                    1,
                    &format!("r_4_same; round {round} coefficients commitment"),
                )
                .challenge_scalars(1, &format!("r_4_same; round {round}"))
        }
        for round in 0..r_2_len {
            self = self
                .add_points(
                    1,
                    &format!("r_2_same; round {round} coefficients commitment"),
                )
                .challenge_scalars(1, &format!("r_2_same; round {round}"))
        }
        for round in 0..r_4_len {
            self = self
                .add_points(
                    1,
                    &format!("r_4_next; round {round} coefficients commitment"),
                )
                .challenge_scalars(1, &format!("r_4_next; round {round}"))
        }
        for round in 0..r_1_len {
            self = self
                .add_points(
                    1,
                    &format!("r_1_same; round {round} coefficients commitment"),
                )
                .challenge_scalars(1, &format!("r_1_same; round {round}"))
        }
        self.add_hyrax_target_commitment()
            .add_hyrax_target_commitment()
            .add_hyrax_target_commitment()
            .add_points(2, "target commitments to b_1 * b_2 and a^T * (b_1 * b_2)")
            .add_sum_check(None, r_3_len + r_4_len + r_2_len, r_4_len + r_1_len)
    }
}

// proof for C = A^T matmul (B_1 hadamard B_2), where the first column of A (i.e., the first row of A^T) is set to zero.
pub struct BackwardPassSumCheck2Proof<'a, G: CurveGroup> {
    // dimensions
    pub r_1_len: usize,
    pub r_2_len: usize,
    pub r_3_len: usize,
    pub r_4_len: usize,

    // other constants
    pub zero_step_idx: usize,
    pub mat_idx_shift: usize,

    // witnesses
    pub opening_a_with_public: Option<&'a Vec<Vec<G::ScalarField>>>,
    pub opening_b1_with_public: Option<&'a HyraxCommitmentOpening<G>>,
    pub opening_b2_with_public: Option<&'a HyraxCommitmentOpening<G>>,
    pub opening_c_with_public: Option<&'a HyraxCommitmentOpening<G>>,
}

impl<'a, G: CurveGroup> BackwardPassSumCheck2Proof<'a, G> {
    // Algorithm 2 in our paper
    pub fn prove(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<[HyraxTargetCommitment<G>; 4]> {
        // 1: already should have happened before calling this

        // 2: obtain challenges for C (r_1, r_2, r_4);
        let mut r_1 = vec![G::ScalarField::ZERO; self.r_1_len];
        prover_state.fill_challenge_scalars(&mut r_1)?;
        let mut r_2 = vec![G::ScalarField::ZERO; self.r_2_len];
        prover_state.fill_challenge_scalars(&mut r_2)?;
        let mut r_4 = vec![G::ScalarField::ZERO; self.r_4_len];
        prover_state.fill_challenge_scalars(&mut r_4)?;

        // 3 + 4: evaluate C on the challenge, DEFER poly eval (step 4)
        let c_target_proof = HyraxPolyEvalProof {
            points: &[r_4.as_slice(), r_1.as_slice(), r_2.as_slice()].concat(),
            commitment_opening_input_matrix: self.opening_c_with_public.map(|x| &x.input_matrix),
        }
        .target_commitment_prove(prover_state, client_schemes)?;

        let test = tensor_chi(&[r_4.as_slice(), r_1.as_slice(), r_2.as_slice()].concat());
        let test_matrix = flat_to_2dvec_almost_square_column_major(
            &test,
            1 << (self.r_4_len + self.r_1_len + self.r_2_len) / 2,
        );
        debug_assert_eq!(
            test_matrix
                .iter()
                .zip(&self.opening_c_with_public.as_ref().unwrap().input_matrix)
                .map(|(x, y)| dot(&x, &y))
                .sum::<G::ScalarField>(),
            c_target_proof.ltr.unwrap()
        );

        // 5-8: generate sum-check polynomials and remaining challenges (r_3, r_4_same, r_2_same, r_4_next, r_1_same)
        // we use the approach from Section 5.4.2 (mentioned in Section 8.2.2) of https://eprint.iacr.org/2013/351.pdf
        // we also optimize based on the suggestions from https://eprint.iacr.org/2017/242.pdf, specifically Section 3.2, Appendix A & B

        // A: initialize the DP arrays in the order that's easiest to collapse.
        // I.e., the randomness ordering should adhere to r_4 > r_1 > r_2 > r_3 > r_4_same > r_2_same > r_4_next > r_1_same

        // Initialize the DP array for beta, with r_2 given
        let mut beta_dp_array = tensor_chi(&r_2);

        // Initialize the DP array for gamma, with r_4 given
        let r_4_full_len = 1_usize << self.r_4_len;
        let mut gamma_dp_array = vec![G::ScalarField::ZERO; r_4_full_len.pow(2)];
        let mut gamma_values = tensor_chi(&r_4);
        for idx in (0..gamma_values.len()).step_by(self.zero_step_idx) {
            gamma_values[idx] = G::ScalarField::ZERO;
            if idx + 1 < gamma_values.len() {
                gamma_values[idx + 1] = G::ScalarField::ZERO; // we don't have an intermediate gradient for the first layer
            }
        }
        // since the last layer is tackled by the MSE we also zero out those entries
        for idx in (self.zero_step_idx - 1..gamma_values.len()).step_by(self.zero_step_idx) {
            gamma_values[idx] = G::ScalarField::ZERO;
        }
        for (entry, value) in gamma_dp_array
            .iter_mut()
            .skip(self.mat_idx_shift * r_4_full_len)
            .step_by(1 + r_4_full_len)
            .zip(gamma_values)
        {
            *entry = value;
        }

        // Initialize the DP array for delta, with r_1 given, for zero-ing out the first column of A, i.e., the first row of A^T
        let mut delta_dp_array = tensor_chi(&r_1);
        delta_dp_array[0] = G::ScalarField::ZERO;

        // useful constants for initializing DP arrays for matrices
        // A matrix
        let a_matrix_len: usize = 1 << self.r_4_len;
        let a_row_len: usize = 1 << self.r_3_len; // row of A is column of A^T
        let a_col_len: usize = 1 << self.r_1_len; // column of A is row of A^T
        // B matrices
        let b_matrix_len: usize = 1 << self.r_4_len;
        let b_row_len: usize = 1 << self.r_3_len;
        let b_col_len: usize = 1 << self.r_2_len;

        // Initialize the DP array for A^T in order column, matrix, row, i.e., (r_3, r_4_same, r_1_same).
        let opening_a = self
            .opening_a_with_public
            .ok_or_else(|| spongefish::ProofError::SerializationError)?;
        let mut a_t_dp_array = vec![G::ScalarField::ZERO; a_matrix_len * a_row_len * a_col_len];
        ark_std::cfg_iter_mut!(a_t_dp_array)
            .enumerate()
            .for_each(|(idx, entry)| {
                let row_idx = idx % a_row_len; // row of A is column of A^T
                let matrix_idx = (idx / a_row_len) % a_matrix_len;
                let col_idx = idx / (a_row_len * a_matrix_len); // column of A is row of A^T

                let opening_full_idx =
                    col_idx * (a_matrix_len * a_row_len) + row_idx * a_matrix_len + matrix_idx;
                let (opening_col, opening_row) = opening_full_idx.div_rem_euclid(&opening_a.len());

                *entry = opening_a[opening_row][opening_col]
            });

        // initialize DP array for B_1 in order row, column, matrix, i.e., (r_3, r_2_same, r_4_next)
        let opening_b1 = self
            .opening_b1_with_public
            .ok_or_else(|| spongefish::ProofError::SerializationError)?;
        let mut b1_dp_array = vec![G::ScalarField::ZERO; b_matrix_len * b_row_len * b_col_len];
        ark_std::cfg_iter_mut!(b1_dp_array)
            .enumerate()
            .for_each(|(idx, entry)| {
                let row_idx = idx % b_row_len;
                let col_idx = (idx / b_row_len) % b_col_len;
                let matrix_idx = idx / (b_row_len * b_col_len);

                let opening_full_idx =
                    col_idx * (b_matrix_len * b_row_len) + row_idx * b_matrix_len + matrix_idx;
                let (opening_col, opening_row) =
                    opening_full_idx.div_rem_euclid(&opening_b1.input_matrix.len());

                *entry = opening_b1.input_matrix[opening_row][opening_col]
            });

        // initialize DP array for B_2 in order row, matrix, column, i.e., (r_3, r_4_same, r_2_same)
        let opening_b2 = self
            .opening_b2_with_public
            .ok_or_else(|| spongefish::ProofError::SerializationError)?;
        let mut b2_dp_array = vec![G::ScalarField::ZERO; b_matrix_len * b_row_len * b_col_len];
        ark_std::cfg_iter_mut!(b2_dp_array)
            .enumerate()
            .for_each(|(idx, entry)| {
                let row_idx = idx % b_row_len;
                let matrix_idx = (idx / b_row_len) % b_matrix_len;
                let col_idx = idx / (b_row_len * b_matrix_len);

                let opening_full_idx =
                    col_idx * (b_matrix_len * b_row_len) + row_idx * b_matrix_len + matrix_idx;
                let (opening_col, opening_row) =
                    opening_full_idx.div_rem_euclid(&opening_b2.input_matrix.len());

                *entry = opening_b2.input_matrix[opening_row][opening_col]
            });

        // B: Collapse over r_4, r_1, r_2 as those are fixed
        // Beta is already given over r_2, so no need to collapse
        // Gamma is already given over r_4, so no need to collapse
        // Delta is already given over r_1, so no need to collapse

        // step-by-step computation of polynomials in most efficient order first r_3, then r_4_same, r_2_same, r_4_next, r_1_same
        // this also collapses the DP arrays after each step

        // in the r_3 part we construct polynomials of degree 3 (so we interpolate on 4 points)
        let mut r_3 = vec![G::ScalarField::ZERO; self.r_3_len];
        let mut cms_coeffs_r_3 = vec![G::Affine::zero(); self.r_3_len];
        let mut coeffs_r_3 = vec![[G::ScalarField::ZERO; 4]; self.r_3_len];
        let mut r_coeffs_r_3 = vec![G::ScalarField::ZERO; self.r_3_len];
        for r_3_idx in 0..self.r_3_len {
            // evaluate polynomial and compute coefficients
            let [eval_minus1, eval_0, eval_1, eval_2] = self.eval_r_3(
                &beta_dp_array,
                &gamma_dp_array,
                &delta_dp_array,
                &a_t_dp_array,
                &b1_dp_array,
                &b2_dp_array,
            );
            let coeffs = get_coeffs_degree_3(
                eval_minus1,
                eval_0,
                eval_1,
                eval_2,
                &client_schemes.constants,
            );

            debug_assert_eq!(
                eval_0 + eval_1,
                if r_3_idx == 0 {
                    c_target_proof.ltr.unwrap()
                } else {
                    poly_eval(&coeffs_r_3[r_3_idx - 1], &r_3[r_3_idx - 1])
                }
            );

            // commit to coefficients
            let r_coeffs = G::ScalarField::rand(prover_state.rng());
            let cm_coeffs_r_3 = client_schemes
                .vector_commit
                .vector_commit(&coeffs, &r_coeffs);

            // get new randomness
            prover_state.add_points(&[cm_coeffs_r_3.into_group()])?;
            let [r_3_entry]: [_; 1] = prover_state.challenge_scalars()?;

            // collapse on new randomness and add to r_3 array
            a_t_dp_array = collapse(&a_t_dp_array, &r_3_entry);
            b1_dp_array = collapse(&b1_dp_array, &r_3_entry);
            b2_dp_array = collapse(&b2_dp_array, &r_3_entry);
            r_3[r_3_idx] = r_3_entry;

            cms_coeffs_r_3[r_3_idx] = cm_coeffs_r_3;
            coeffs_r_3[r_3_idx] = coeffs;
            r_coeffs_r_3[r_3_idx] = r_coeffs;
        }

        // in the r_4_same part we construct polynomials of degree 3 (so we interpolate on 4 points)
        let mut r_4_same = vec![G::ScalarField::ZERO; self.r_4_len];
        let mut cms_coeffs_r_4_same = vec![G::Affine::zero(); self.r_4_len];
        let mut coeffs_r_4_same = vec![[G::ScalarField::ZERO; 4]; self.r_4_len];
        let mut r_coeffs_r_4_same = vec![G::ScalarField::ZERO; self.r_4_len];
        for r_4_same_idx in 0..self.r_4_len {
            // evaluate polynomial and compute coefficients
            let [eval_minus1, eval_0, eval_1, eval_2] = self.eval_r_4_same(
                &beta_dp_array,
                &gamma_dp_array,
                &delta_dp_array,
                &a_t_dp_array,
                &b1_dp_array,
                &b2_dp_array,
            );
            let coeffs = get_coeffs_degree_3(
                eval_minus1,
                eval_0,
                eval_1,
                eval_2,
                &client_schemes.constants,
            );

            debug_assert_eq!(
                eval_0 + eval_1,
                if r_4_same_idx == 0 {
                    poly_eval(coeffs_r_3.last().unwrap(), r_3.last().unwrap())
                } else {
                    poly_eval(
                        &coeffs_r_4_same[r_4_same_idx - 1],
                        &r_4_same[r_4_same_idx - 1],
                    )
                }
            );

            // commit to coefficients
            let r_coeffs = G::ScalarField::rand(prover_state.rng());
            let cm_coeffs_r_4_same = client_schemes
                .vector_commit
                .vector_commit(&coeffs, &r_coeffs);

            // get new randomness
            prover_state.add_points(&[cm_coeffs_r_4_same.into_group()])?;
            let [r_4_same_entry]: [_; 1] = prover_state.challenge_scalars()?;

            // collapse on new randomness and add to r_4_same array
            gamma_dp_array = collapse(&gamma_dp_array, &r_4_same_entry);
            a_t_dp_array = collapse(&a_t_dp_array, &r_4_same_entry);
            b2_dp_array = collapse(&b2_dp_array, &r_4_same_entry);
            r_4_same[r_4_same_idx] = r_4_same_entry;

            cms_coeffs_r_4_same[r_4_same_idx] = cm_coeffs_r_4_same;
            coeffs_r_4_same[r_4_same_idx] = coeffs;
            r_coeffs_r_4_same[r_4_same_idx] = r_coeffs;
        }

        // in the r_2_same part we construct polynomials of degree 3 (so we interpolate on 4 points)
        let mut r_2_same = vec![G::ScalarField::ZERO; self.r_2_len];
        let mut cms_coeffs_r_2_same = vec![G::Affine::zero(); self.r_2_len];
        let mut coeffs_r_2_same = vec![[G::ScalarField::ZERO; 4]; self.r_2_len];
        let mut r_coeffs_r_2_same = vec![G::ScalarField::ZERO; self.r_2_len];
        let delta_a_t_value = self.compute_delta_a_t_value(&delta_dp_array, &a_t_dp_array);
        for r_2_same_idx in 0..self.r_2_len {
            // evaluate polynomial and compute coefficients
            let [eval_minus1, eval_0, eval_1, eval_2] = self.eval_r_2_same(
                &beta_dp_array,
                &gamma_dp_array,
                &delta_a_t_value,
                &b1_dp_array,
                &b2_dp_array,
            );
            let coeffs = get_coeffs_degree_3(
                eval_minus1,
                eval_0,
                eval_1,
                eval_2,
                &client_schemes.constants,
            );

            debug_assert_eq!(
                eval_0 + eval_1,
                if r_2_same_idx == 0 {
                    poly_eval(coeffs_r_4_same.last().unwrap(), r_4_same.last().unwrap())
                } else {
                    poly_eval(
                        &coeffs_r_2_same[r_2_same_idx - 1],
                        &r_2_same[r_2_same_idx - 1],
                    )
                }
            );

            // commit to coefficients
            let r_coeffs = G::ScalarField::rand(prover_state.rng());
            let cm_coeffs_r_2_same = client_schemes
                .vector_commit
                .vector_commit(&coeffs, &r_coeffs);

            // get new randomness
            prover_state.add_points(&[cm_coeffs_r_2_same.into_group()])?;
            let [r_2_same_entry]: [_; 1] = prover_state.challenge_scalars()?;

            // collapse on new randomness and add to r_2_same array
            beta_dp_array = collapse(&beta_dp_array, &r_2_same_entry);
            b1_dp_array = collapse(&b1_dp_array, &r_2_same_entry);
            b2_dp_array = collapse(&b2_dp_array, &r_2_same_entry);
            r_2_same[r_2_same_idx] = r_2_same_entry;

            cms_coeffs_r_2_same[r_2_same_idx] = cm_coeffs_r_2_same;
            coeffs_r_2_same[r_2_same_idx] = coeffs;
            r_coeffs_r_2_same[r_2_same_idx] = r_coeffs;
        }

        // in the r_4_next part we construct polynomials of degree 2 (so we interpolate on 3 points)
        let mut r_4_next = vec![G::ScalarField::ZERO; self.r_4_len];
        let mut cms_coeffs_r_4_next = vec![G::Affine::zero(); self.r_4_len];
        let mut coeffs_r_4_next = vec![[G::ScalarField::ZERO; 3]; self.r_4_len];
        let mut r_coeffs_r_4_next = vec![G::ScalarField::ZERO; self.r_4_len];
        let beta_b2_value = beta_dp_array[0] * b2_dp_array[0];
        for r_4_next_idx in 0..self.r_4_len {
            // evaluate polynomial and compute coefficients
            let [eval_minus1, eval_0, eval_1] = self.eval_r_4_next(
                &gamma_dp_array,
                &delta_a_t_value,
                &b1_dp_array,
                &beta_b2_value,
            );
            let coeffs =
                get_coeffs_degree_2(eval_minus1, eval_0, eval_1, &client_schemes.constants);

            debug_assert_eq!(
                eval_0 + eval_1,
                if r_4_next_idx == 0 {
                    poly_eval(coeffs_r_2_same.last().unwrap(), r_2_same.last().unwrap())
                } else {
                    poly_eval(
                        &coeffs_r_4_next[r_4_next_idx - 1],
                        &r_4_next[r_4_next_idx - 1],
                    )
                }
            );

            // commit to coefficients
            let r_coeffs = G::ScalarField::rand(prover_state.rng());
            let cm_coeffs_r_4_next = client_schemes
                .vector_commit
                .vector_commit(&coeffs, &r_coeffs);

            // get new randomness
            prover_state.add_points(&[cm_coeffs_r_4_next.into_group()])?;
            let [r_4_next_entry]: [_; 1] = prover_state.challenge_scalars()?;

            // collapse on new randomness and add to r_4_next array
            gamma_dp_array = collapse(&gamma_dp_array, &r_4_next_entry);
            b1_dp_array = collapse(&b1_dp_array, &r_4_next_entry);
            r_4_next[r_4_next_idx] = r_4_next_entry;

            cms_coeffs_r_4_next[r_4_next_idx] = cm_coeffs_r_4_next;
            coeffs_r_4_next[r_4_next_idx] = coeffs;
            r_coeffs_r_4_next[r_4_next_idx] = r_coeffs;
        }

        // in the r_1_same part we construct polynomials of degree 2 (so we interpolate on 3 points)
        let mut r_1_same = vec![G::ScalarField::ZERO; self.r_1_len];
        let mut cms_coeffs_r_1_same = vec![G::Affine::zero(); self.r_1_len];
        let mut coeffs_r_1_same = vec![[G::ScalarField::ZERO; 3]; self.r_1_len];
        let mut r_coeffs_r_1_same = vec![G::ScalarField::ZERO; self.r_1_len];
        let gamma_b_value = gamma_dp_array[0] * beta_b2_value * b1_dp_array[0];
        for r_1_same_idx in 0..self.r_1_len {
            // evaluate polynomial and compute coefficients
            let [eval_minus1, eval_0, eval_1] =
                self.eval_r_1_same(&delta_dp_array, &a_t_dp_array, &gamma_b_value);
            let coeffs =
                get_coeffs_degree_2(eval_minus1, eval_0, eval_1, &client_schemes.constants);

            debug_assert_eq!(
                eval_0 + eval_1,
                if r_1_same_idx == 0 {
                    poly_eval(coeffs_r_4_next.last().unwrap(), r_4_next.last().unwrap())
                } else {
                    poly_eval(
                        &coeffs_r_1_same[r_1_same_idx - 1],
                        &r_1_same[r_1_same_idx - 1],
                    )
                }
            );

            // commit to coefficients
            let r_coeffs = G::ScalarField::rand(prover_state.rng());
            let cm_coeffs_r_1_same = client_schemes
                .vector_commit
                .vector_commit(&coeffs, &r_coeffs);

            // get new randomness
            prover_state.add_points(&[cm_coeffs_r_1_same.into_group()])?;
            let [r_1_same_entry]: [_; 1] = prover_state.challenge_scalars()?;

            // collapse on new randomness and add to r_1_same array
            delta_dp_array = collapse(&delta_dp_array, &r_1_same_entry);
            a_t_dp_array = collapse(&a_t_dp_array, &r_1_same_entry);
            r_1_same[r_1_same_idx] = r_1_same_entry;

            cms_coeffs_r_1_same[r_1_same_idx] = cm_coeffs_r_1_same;
            coeffs_r_1_same[r_1_same_idx] = coeffs;
            r_coeffs_r_1_same[r_1_same_idx] = r_coeffs;
        }

        // 9: evaluate A, B_1, and B_2 on the appropriate challenges from r_1,...,r_4_next
        // no need for beta, gamma, or delta, as the verifier knows the polynomial so can compute it directly
        // note: for a_t we have swapped r_3 and r_1_same order, since we compute the transpose of a and not a
        let a_t_target_proof = HyraxPolyEvalProof {
            points: &[r_4_same.as_slice(), r_3.as_slice(), r_1_same.as_slice()].concat(),
            commitment_opening_input_matrix: self.opening_a_with_public,
        }
        .target_commitment_prove(prover_state, client_schemes)?;
        let b1_target_proof = HyraxPolyEvalProof {
            points: &[r_4_next.as_slice(), r_3.as_slice(), r_2_same.as_slice()].concat(),
            commitment_opening_input_matrix: self.opening_b1_with_public.map(|x| &x.input_matrix),
        }
        .target_commitment_prove(prover_state, client_schemes)?;
        let b2_target_proof = HyraxPolyEvalProof {
            points: &[r_4_same.as_slice(), r_3.as_slice(), r_2_same.as_slice()].concat(),
            commitment_opening_input_matrix: self.opening_b2_with_public.map(|x| &x.input_matrix),
        }
        .target_commitment_prove(prover_state, client_schemes)?;

        let b1_b2 =
            b1_target_proof.ltr.expect("Should exist") * b2_target_proof.ltr.expect("Should exist");
        let a_t_b1_b2 = a_t_target_proof.ltr.expect("Should exist") * b1_b2;
        let r_b1_b2 = G::ScalarField::rand(prover_state.rng());
        let r_a_t_b1_b2 = G::ScalarField::rand(prover_state.rng());
        let cm_b1_b2 = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&b1_b2, &r_b1_b2);
        let cm_a_t_b1_b2 = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&a_t_b1_b2, &r_a_t_b1_b2);
        prover_state.add_points(&[cm_b1_b2.into_group(), cm_a_t_b1_b2.into_group()])?;

        debug_assert_eq!(
            poly_eval(coeffs_r_1_same.last().unwrap(), r_1_same.last().unwrap()),
            a_t_b1_b2 * beta_dp_array[0] * gamma_dp_array[0] * delta_dp_array[0]
        );

        // 10: sum-check proof
        SumCheckProof {
            cms_coeffs_degree4: None,
            cms_coeffs_degree3: &[
                cms_coeffs_r_3.as_slice(),
                cms_coeffs_r_4_same.as_slice(),
                cms_coeffs_r_2_same.as_slice(),
            ]
            .concat(),
            cms_coeffs_degree2: &[
                cms_coeffs_r_4_next.as_slice(),
                cms_coeffs_r_1_same.as_slice(),
            ]
            .concat(),
            cm_a1: &a_t_target_proof.omega.expect("This should exist"),
            cm_a2: None,
            cm_b1: &b1_target_proof.omega.expect("This should exist"),
            cm_b2: &b2_target_proof.omega.expect("This should exist"),
            cm_c: &c_target_proof.omega.expect("This should exist"),
            cm_a1_a2: None,
            cm_b1_b2: &cm_b1_b2,
            cm_a_b: &cm_a_t_b1_b2,
            multiplier: &(beta_dp_array[0] * gamma_dp_array[0] * delta_dp_array[0]),
            r_degree_4: None,
            r_degree_3: &[r_3.as_slice(), r_4_same.as_slice(), r_2_same.as_slice()].concat(),
            r_degree_2: &[r_4_next.as_slice(), r_1_same.as_slice()].concat(),
            coeffs_degree4: None,
            r_coeffs_degree4: None,
            coeffs_degree3: Some(
                &[
                    coeffs_r_3.as_slice(),
                    coeffs_r_4_same.as_slice(),
                    coeffs_r_2_same.as_slice(),
                ]
                .concat(),
            ),
            r_coeffs_degree3: Some(
                &[
                    r_coeffs_r_3.as_slice(),
                    r_coeffs_r_4_same.as_slice(),
                    r_coeffs_r_2_same.as_slice(),
                ]
                .concat(),
            ),
            coeffs_degree2: Some(
                &[coeffs_r_4_next.as_slice(), coeffs_r_1_same.as_slice()].concat(),
            ),
            r_coeffs_degree2: Some(
                &[r_coeffs_r_4_next.as_slice(), r_coeffs_r_1_same.as_slice()].concat(),
            ),
            a1: Some(&a_t_target_proof.ltr.expect("Should exist")),
            r_a1: Some(&a_t_target_proof.r_omega.expect("Should exist")),
            a2: None,
            r_a2: None,
            b1: Some(&b1_target_proof.ltr.expect("Should exist")),
            r_b1: Some(&b1_target_proof.r_omega.expect("Should exist")),
            b2: Some(&b2_target_proof.ltr.expect("Should exist")),
            r_b2: Some(&b2_target_proof.r_omega.expect("Should exist")),
            r_c: Some(&c_target_proof.r_omega.expect("Should exist")),
            a1_a2: None,
            r_a1_a2: None,
            b1_b2: Some(&b1_b2),
            r_b1_b2: Some(&r_b1_b2),
            r_a_b: Some(&r_a_t_b1_b2),
        }
        .prove(false, prover_state, client_schemes)?;

        Ok([
            a_t_target_proof,
            b1_target_proof,
            b2_target_proof,
            c_target_proof,
        ])
    }

    pub fn verify(
        self,
        verifier_state: &mut VerifierState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<[HyraxTargetCommitment<G>; 4]> {
        // 1: already got all commitments before this is called

        // 2
        let mut r_1 = vec![G::ScalarField::ZERO; self.r_1_len];
        verifier_state.fill_challenge_scalars(&mut r_1)?;
        let mut r_2 = vec![G::ScalarField::ZERO; self.r_2_len];
        verifier_state.fill_challenge_scalars(&mut r_2)?;
        let mut r_4 = vec![G::ScalarField::ZERO; self.r_4_len];
        verifier_state.fill_challenge_scalars(&mut r_4)?;

        // 3 + 4
        let c_target_proof = HyraxPolyEvalProof {
            points: &[r_4.as_slice(), r_1.as_slice(), r_2.as_slice()].concat(),
            commitment_opening_input_matrix: None,
        }
        .target_commitment_verify(verifier_state)?;

        // 5-8

        // r_3 part
        let mut cms_coeffs_r_3 = vec![G::Affine::zero(); self.r_3_len];
        let mut r_3 = vec![G::ScalarField::ZERO; self.r_3_len];
        for r_3_idx in 0..self.r_3_len {
            cms_coeffs_r_3[r_3_idx] = verifier_state
                .next_points::<1>()?
                .map(|cm: G| cm.into_affine())[0];
            r_3[r_3_idx] = verifier_state.challenge_scalars::<1>()?[0];
        }

        // r_4_same part
        let mut cms_coeffs_r_4_same = vec![G::Affine::zero(); self.r_4_len];
        let mut r_4_same = vec![G::ScalarField::ZERO; self.r_4_len];
        for r_4_same_idx in 0..self.r_4_len {
            cms_coeffs_r_4_same[r_4_same_idx] = verifier_state
                .next_points::<1>()?
                .map(|cm: G| cm.into_affine())[0];
            r_4_same[r_4_same_idx] = verifier_state.challenge_scalars::<1>()?[0];
        }

        // r_2_same part
        let mut cms_coeffs_r_2_same = vec![G::Affine::zero(); self.r_2_len];
        let mut r_2_same = vec![G::ScalarField::ZERO; self.r_2_len];
        for r_2_same_idx in 0..self.r_2_len {
            cms_coeffs_r_2_same[r_2_same_idx] = verifier_state
                .next_points::<1>()?
                .map(|cm: G| cm.into_affine())[0];
            r_2_same[r_2_same_idx] = verifier_state.challenge_scalars::<1>()?[0];
        }

        // r_4_next part
        let mut cms_coeffs_r_4_next = vec![G::Affine::zero(); self.r_4_len];
        let mut r_4_next = vec![G::ScalarField::ZERO; self.r_4_len];
        for r_4_next_idx in 0..self.r_4_len {
            cms_coeffs_r_4_next[r_4_next_idx] = verifier_state
                .next_points::<1>()?
                .map(|cm: G| cm.into_affine())[0];
            r_4_next[r_4_next_idx] = verifier_state.challenge_scalars::<1>()?[0];
        }

        // r_1_same part
        let mut cms_coeffs_r_1_same = vec![G::Affine::zero(); self.r_1_len];
        let mut r_1_same = vec![G::ScalarField::ZERO; self.r_1_len];
        for r_1_same_idx in 0..self.r_1_len {
            cms_coeffs_r_1_same[r_1_same_idx] = verifier_state
                .next_points::<1>()?
                .map(|cm: G| cm.into_affine())[0];
            r_1_same[r_1_same_idx] = verifier_state.challenge_scalars::<1>()?[0];
        }

        // 9
        // note: for a_t we have swapped r_3 and r_1 order, since we compute the transpose of a and not a
        let a_t_target_proof = HyraxPolyEvalProof {
            points: &[r_4_same.as_slice(), r_3.as_slice(), r_1_same.as_slice()].concat(),
            commitment_opening_input_matrix: None,
        }
        .target_commitment_verify(verifier_state)?;
        let b1_target_proof = HyraxPolyEvalProof {
            points: &[r_4_next.as_slice(), r_3.as_slice(), r_2_same.as_slice()].concat(),
            commitment_opening_input_matrix: None,
        }
        .target_commitment_verify(verifier_state)?;
        let b2_target_proof = HyraxPolyEvalProof {
            points: &[r_4_same.as_slice(), r_3.as_slice(), r_2_same.as_slice()].concat(),
            commitment_opening_input_matrix: None,
        }
        .target_commitment_verify(verifier_state)?;
        let [cm_b1_b2, cm_a_t_b1_b2]: [_; 2] =
            verifier_state.next_points()?.map(|cm: G| cm.into_affine());

        // evaluate beta, gamma, and delta
        // for beta we can use the simple formulation
        let beta_eval = ark_std::cfg_iter!(r_2)
            .zip(ark_std::cfg_iter!(r_2_same))
            .map(|(&x, y)| x * y + (G::ScalarField::ONE - x) * (G::ScalarField::ONE - y))
            .product::<G::ScalarField>();
        // for gamma we can use the simple formulation for the (r_4, r_4_same) part and use the regular way to link that to r_4_next
        let mut gamma_dp_array = double_tensor_chi(&r_4, &r_4_same);
        for idx in (0..gamma_dp_array.len()).step_by(self.zero_step_idx) {
            gamma_dp_array[idx] = G::ScalarField::ZERO;
            if idx + 1 < gamma_dp_array.len() {
                gamma_dp_array[idx + 1] = G::ScalarField::ZERO; // we don't have an intermediate gradient for the first layer
            }
        }
        // since the last layer is tackled by the MSE we also zero out those entries
        for idx in (self.zero_step_idx - 1..gamma_dp_array.len()).step_by(self.zero_step_idx) {
            gamma_dp_array[idx] = G::ScalarField::ZERO;
        }
        gamma_dp_array.rotate_right(self.mat_idx_shift);
        gamma_dp_array[0] = G::ScalarField::ZERO;
        gamma_dp_array = multi_collapse(&gamma_dp_array, &r_4_next);

        // for delta we can use the simple formulation, but we need to subtract one value (for when r_1=0...0 and r_1_same=0...0)
        let mut delta_eval = ark_std::cfg_iter!(r_1)
            .zip(ark_std::cfg_iter!(r_1_same))
            .map(|(&x, y)| x * y + (G::ScalarField::ONE - x) * (G::ScalarField::ONE - y))
            .product::<G::ScalarField>();
        delta_eval -= ark_std::cfg_iter!(r_1)
            .zip(ark_std::cfg_iter!(r_1_same))
            .map(|(&x, y)| (G::ScalarField::ONE - x) * (G::ScalarField::ONE - y))
            .product::<G::ScalarField>();

        // 10
        SumCheckProof {
            cms_coeffs_degree4: None,
            cms_coeffs_degree3: &[
                cms_coeffs_r_3.as_slice(),
                cms_coeffs_r_4_same.as_slice(),
                cms_coeffs_r_2_same.as_slice(),
            ]
            .concat(),
            cms_coeffs_degree2: &[
                cms_coeffs_r_4_next.as_slice(),
                cms_coeffs_r_1_same.as_slice(),
            ]
            .concat(),
            cm_a1: &a_t_target_proof.omega.expect("This should exist"),
            cm_a2: None,
            cm_b1: &b1_target_proof.omega.expect("This should exist"),
            cm_b2: &b2_target_proof.omega.expect("This should exist"),
            cm_c: &c_target_proof.omega.expect("This should exist"),
            cm_a1_a2: None,
            cm_b1_b2: &cm_b1_b2,
            cm_a_b: &cm_a_t_b1_b2,
            multiplier: &(beta_eval * gamma_dp_array[0] * delta_eval),
            r_degree_4: None,
            r_degree_3: &[r_3.as_slice(), r_4_same.as_slice(), r_2_same.as_slice()].concat(),
            r_degree_2: &[r_4_next.as_slice(), r_1_same.as_slice()].concat(),
            coeffs_degree4: None,
            r_coeffs_degree4: None,
            coeffs_degree3: None,
            r_coeffs_degree3: None,
            coeffs_degree2: None,
            r_coeffs_degree2: None,
            a1: None,
            r_a1: None,
            a2: None,
            r_a2: None,
            b1: None,
            r_b1: None,
            b2: None,
            r_b2: None,
            r_c: None,
            a1_a2: None,
            r_a1_a2: None,
            b1_b2: None,
            r_b1_b2: None,
            r_a_b: None,
        }
        .verify(false, verifier_state, client_schemes)?;

        Ok([
            a_t_target_proof,
            b1_target_proof,
            b2_target_proof,
            c_target_proof,
        ])
    }

    fn eval_r_3(
        &self,
        beta_dp_array: &[G::ScalarField],
        gamma_dp_array: &[G::ScalarField],
        delta_dp_array: &[G::ScalarField],
        a_t_dp_array: &[G::ScalarField],
        b1_dp_array: &[G::ScalarField],
        b2_dp_array: &[G::ScalarField],
    ) -> [G::ScalarField; 4] {
        // constants
        let r_1_max = 1 << self.r_1_len;
        let r_2_max = 1 << self.r_2_len;
        let r_4_max = 1 << self.r_4_len;
        let r_3_max = a_t_dp_array.len() / (r_1_max * r_4_max);

        let evals = ark_std::cfg_into_iter!(0..r_3_max).step_by(2).fold(
            #[cfg(feature = "parallel")]
            || [G::ScalarField::ZERO; 4],
            #[cfg(not(feature = "parallel"))]
            [G::ScalarField::ZERO; 4],
            |mut evals_tmp, r_3| {
                // evals
                let a_b_evals =
                    (0..r_4_max - 1).fold([G::ScalarField::ZERO; 4], |mut a_b_evals_tmp, r_4| {
                        // indices
                        let gamma_idx = r_4 + (r_4 + self.mat_idx_shift) * r_4_max;

                        // evals
                        let delta_a_t_evals = (0..r_1_max).fold(
                            [G::ScalarField::ZERO; 4],
                            |mut delta_a_t_evals_tmp, r_1| {
                                // indices
                                let inner_a_idx = r_3 + r_4 * r_3_max + r_1 * r_3_max * r_4_max;

                                // evals
                                let a_t_evals = point_evals_degree_3(
                                    &a_t_dp_array[inner_a_idx],
                                    &a_t_dp_array[inner_a_idx + 1],
                                );

                                // summing the evals
                                delta_a_t_evals_tmp
                                    .iter_mut()
                                    .zip(a_t_evals)
                                    .for_each(|(s, a_t_eval)| *s += delta_dp_array[r_1] * a_t_eval);
                                delta_a_t_evals_tmp
                            },
                        );
                        let b_evals =
                            (0..r_2_max).fold([G::ScalarField::ZERO; 4], |mut b_evals_tmp, r_2| {
                                // indices
                                let inner_b1_idx =
                                    r_3 + r_2 * r_3_max + (r_4 + 1) * r_3_max * r_2_max;
                                let inner_b2_idx = r_3 + r_4 * r_3_max + r_2 * r_3_max * r_4_max;

                                // evals
                                let b1_evals = point_evals_degree_3(
                                    &b1_dp_array[inner_b1_idx],
                                    &b1_dp_array[inner_b1_idx + 1],
                                );
                                let b2_evals = point_evals_degree_3(
                                    &b2_dp_array[inner_b2_idx],
                                    &b2_dp_array[inner_b2_idx + 1],
                                );

                                // summing the evals
                                b_evals_tmp.iter_mut().zip(b1_evals).zip(b2_evals).for_each(
                                    |((s, b1_eval), b2_eval)| {
                                        *s += beta_dp_array[r_2] * b1_eval * b2_eval
                                    },
                                );
                                b_evals_tmp
                            });

                        // summing the evals
                        a_b_evals_tmp
                            .iter_mut()
                            .zip(delta_a_t_evals)
                            .zip(b_evals)
                            .for_each(|((s, delta_a_t_eval), b_eval)| {
                                *s += gamma_dp_array[gamma_idx] * delta_a_t_eval * b_eval
                            });
                        a_b_evals_tmp
                    });

                // summing the evals
                evals_tmp
                    .iter_mut()
                    .zip(a_b_evals)
                    .for_each(|(s, r)| *s += r);
                evals_tmp
            },
        );
        // in the parallel setting we need to reduce the results of the threads once more
        #[cfg(feature = "parallel")]
        {
            evals.reduce(
                || [G::ScalarField::ZERO; 4],
                |mut evals_final, x| {
                    evals_final.iter_mut().zip(x).for_each(|(a, b)| *a += b);
                    evals_final
                },
            )
        }
        #[cfg(not(feature = "parallel"))]
        evals
    }

    fn eval_r_4_same(
        &self,
        beta_dp_array: &[G::ScalarField],
        gamma_dp_array: &[G::ScalarField],
        delta_dp_array: &[G::ScalarField],
        a_t_dp_array: &[G::ScalarField],
        b1_dp_array: &[G::ScalarField],
        b2_dp_array: &[G::ScalarField],
    ) -> [G::ScalarField; 4] {
        // constants
        let r_1_max = 1 << self.r_1_len;
        let r_2_max = 1 << self.r_2_len;
        let r_4_next_max = 1 << self.r_4_len;
        let r_4_same_max = gamma_dp_array.len() / (r_4_next_max);
        let r_4_next_fp_max = r_4_next_max / r_4_same_max;

        let evals = ark_std::cfg_into_iter!(0..r_4_same_max).step_by(2).fold(
            #[cfg(feature = "parallel")]
            || [G::ScalarField::ZERO; 4],
            #[cfg(not(feature = "parallel"))]
            [G::ScalarField::ZERO; 4],
            |mut evals_tmp, r_4_same| {
                // evals
                let delta_a_t_evals =
                    (0..r_1_max).fold([G::ScalarField::ZERO; 4], |mut delta_a_t_evals_tmp, r_1| {
                        // indices
                        let inner_a_idx = r_4_same + r_1 * r_4_same_max;

                        // evals
                        let a_t_evals = point_evals_degree_3(
                            &a_t_dp_array[inner_a_idx],
                            &a_t_dp_array[inner_a_idx + 1],
                        );

                        // summing the evals
                        delta_a_t_evals_tmp
                            .iter_mut()
                            .zip(a_t_evals)
                            .for_each(|(s, a_t_eval)| *s += delta_dp_array[r_1] * a_t_eval);
                        delta_a_t_evals_tmp
                    });
                let gamma_b_evals = (r_4_same * r_4_next_fp_max + 1
                    ..min(
                        (r_4_same + 1) * r_4_next_fp_max + r_4_next_fp_max + 1,
                        r_4_next_max,
                    ))
                    .fold(
                        [G::ScalarField::ZERO; 4],
                        |mut gamma_b_evals_tmp, r_4_next| {
                            // indices
                            let gamma_idx = r_4_same + r_4_next * r_4_same_max;

                            // evals
                            let gamma_evals = point_evals_degree_3(
                                &gamma_dp_array[gamma_idx],
                                &gamma_dp_array[gamma_idx + 1],
                            );
                            let b_evals = (0..r_2_max).fold(
                                [G::ScalarField::ZERO; 4],
                                |mut b_evals_tmp, r_2| {
                                    // indices
                                    let inner_b1_idx = r_2 + r_4_next * r_2_max;
                                    let inner_b2_idx = r_4_same + r_2 * r_4_same_max;

                                    //evals
                                    let b2_evals = point_evals_degree_3(
                                        &b2_dp_array[inner_b2_idx],
                                        &b2_dp_array[inner_b2_idx + 1],
                                    );

                                    //summing the evals
                                    b_evals_tmp.iter_mut().zip(b2_evals).for_each(
                                        |(s, b2_eval)| {
                                            *s += beta_dp_array[r_2]
                                                * b1_dp_array[inner_b1_idx]
                                                * b2_eval
                                        },
                                    );
                                    b_evals_tmp
                                },
                            );

                            // summing the evals
                            gamma_b_evals_tmp
                                .iter_mut()
                                .zip(gamma_evals)
                                .zip(b_evals)
                                .for_each(|((s, gamma_eval), b_eval)| *s += gamma_eval * b_eval);
                            gamma_b_evals_tmp
                        },
                    );

                // summing the evals
                evals_tmp
                    .iter_mut()
                    .zip(delta_a_t_evals)
                    .zip(gamma_b_evals)
                    .for_each(|((s, delta_a_t_eval), gamma_b_eval)| {
                        *s += delta_a_t_eval * gamma_b_eval
                    });
                evals_tmp
            },
        );

        // in the parallel setting we need to reduce the results of the threads once more
        #[cfg(feature = "parallel")]
        {
            evals.reduce(
                || [G::ScalarField::ZERO; 4],
                |mut evals_final, x| {
                    evals_final.iter_mut().zip(x).for_each(|(a, b)| *a += b);
                    evals_final
                },
            )
        }
        #[cfg(not(feature = "parallel"))]
        evals
    }

    fn compute_delta_a_t_value(
        &self,
        delta_dp_array: &[G::ScalarField],
        a_t_dp_array: &[G::ScalarField],
    ) -> G::ScalarField {
        ark_std::cfg_into_iter!(delta_dp_array)
            .zip(ark_std::cfg_into_iter!(a_t_dp_array))
            .map(|(&delta_value, a_t_value)| delta_value * a_t_value)
            .sum()
    }

    fn eval_r_2_same(
        &self,
        beta_dp_array: &[G::ScalarField],
        gamma_dp_array: &[G::ScalarField],
        delta_a_t_value: &G::ScalarField,
        b1_dp_array: &[G::ScalarField],
        b2_dp_array: &[G::ScalarField],
    ) -> [G::ScalarField; 4] {
        // constants
        let r_4_next_max = 1 << self.r_4_len;
        let r_2_same_max = beta_dp_array.len();

        let gamma_b_evals = ark_std::cfg_into_iter!(0..r_2_same_max).step_by(2).fold(
            #[cfg(feature = "parallel")]
            || [G::ScalarField::ZERO; 4],
            #[cfg(not(feature = "parallel"))]
            [G::ScalarField::ZERO; 4],
            |mut gamma_b_evals_tmp, r_2_same| {
                // evals
                let gamma_b1_evals = (0..r_4_next_max).fold(
                    [G::ScalarField::ZERO; 4],
                    |mut gamma_b1_evals_tmp, r_4_next| {
                        // indices
                        let inner_b1_idx = r_2_same + r_4_next * r_2_same_max;

                        // evals
                        let b1_evals = point_evals_degree_3(
                            &b1_dp_array[inner_b1_idx],
                            &b1_dp_array[inner_b1_idx + 1],
                        );

                        // summing the evals
                        gamma_b1_evals_tmp
                            .iter_mut()
                            .zip(b1_evals)
                            .for_each(|(s, b1_eval)| *s += gamma_dp_array[r_4_next] * b1_eval);
                        gamma_b1_evals_tmp
                    },
                );
                let beta_evals =
                    point_evals_degree_3(&beta_dp_array[r_2_same], &beta_dp_array[r_2_same + 1]);
                let b2_evals =
                    point_evals_degree_3(&b2_dp_array[r_2_same], &b2_dp_array[r_2_same + 1]);

                // summing the evals
                gamma_b_evals_tmp
                    .iter_mut()
                    .zip(beta_evals)
                    .zip(gamma_b1_evals)
                    .zip(b2_evals)
                    .for_each(|(((s, beta_eval), gamma_b1_eval), b2_eval)| {
                        *s += beta_eval * gamma_b1_eval * b2_eval
                    });
                gamma_b_evals_tmp
            },
        );

        // in the parallel setting we need to reduce the results of the threads once more
        let mut evals;
        #[cfg(feature = "parallel")]
        {
            evals = gamma_b_evals.reduce(
                || [G::ScalarField::ZERO; 4],
                |mut gamma_b_evals_final, x| {
                    gamma_b_evals_final
                        .iter_mut()
                        .zip(x)
                        .for_each(|(a, b)| *a += b);
                    gamma_b_evals_final
                },
            )
        }
        #[cfg(not(feature = "parallel"))]
        {
            evals = gamma_b_evals;
        }

        // multiply with delta_a_t_value for the final result
        evals.iter_mut().for_each(|eval| *eval *= delta_a_t_value);
        evals
    }

    fn eval_r_4_next(
        &self,
        gamma_dp_array: &[G::ScalarField],
        delta_a_t_value: &G::ScalarField,
        b1_dp_array: &[G::ScalarField],
        beta_b2_value: &G::ScalarField,
    ) -> [G::ScalarField; 3] {
        let r_4_next_max = gamma_dp_array.len();

        let gamma_b1_evals = ark_std::cfg_into_iter!(0..r_4_next_max).step_by(2).fold(
            #[cfg(feature = "parallel")]
            || [G::ScalarField::ZERO; 3],
            #[cfg(not(feature = "parallel"))]
            [G::ScalarField::ZERO; 3],
            |mut gamma_b1_evals_tmp, r_4_next| {
                // evals
                let gamma_evals =
                    point_evals_degree_2(&gamma_dp_array[r_4_next], &gamma_dp_array[r_4_next + 1]);
                let b1_evals =
                    point_evals_degree_2(&b1_dp_array[r_4_next], &b1_dp_array[r_4_next + 1]);

                gamma_b1_evals_tmp
                    .iter_mut()
                    .zip(gamma_evals)
                    .zip(b1_evals)
                    .for_each(|((s, gamma_eval), b1_eval)| *s += gamma_eval * b1_eval);
                gamma_b1_evals_tmp
            },
        );

        // in the parallel setting we need to reduce the results of the threads once more
        let mut evals;
        #[cfg(feature = "parallel")]
        {
            evals = gamma_b1_evals.reduce(
                || [G::ScalarField::ZERO; 3],
                |mut gamma_b1_evals_final, x| {
                    gamma_b1_evals_final
                        .iter_mut()
                        .zip(x)
                        .for_each(|(a, b)| *a += b);
                    gamma_b1_evals_final
                },
            )
        }
        #[cfg(not(feature = "parallel"))]
        {
            evals = gamma_b1_evals;
        }

        // multiply with delta_a_t_value and beta_b2_value for the final result
        let multiplier = *delta_a_t_value * beta_b2_value;
        evals.iter_mut().for_each(|eval| *eval *= multiplier);
        evals
    }

    fn eval_r_1_same(
        &self,
        delta_dp_array: &[G::ScalarField],
        a_t_dp_array: &[G::ScalarField],
        gamma_b_value: &G::ScalarField,
    ) -> [G::ScalarField; 3] {
        let r_1_same_max = delta_dp_array.len();

        let delta_a_t_evals = ark_std::cfg_into_iter!(0..r_1_same_max).step_by(2).fold(
            #[cfg(feature = "parallel")]
            || [G::ScalarField::ZERO; 3],
            #[cfg(not(feature = "parallel"))]
            [G::ScalarField::ZERO; 3],
            |mut delta_a_t_evals_tmp, r_1_same| {
                // evals
                let delta_evals =
                    point_evals_degree_2(&delta_dp_array[r_1_same], &delta_dp_array[r_1_same + 1]);
                let a_t_evals =
                    point_evals_degree_2(&a_t_dp_array[r_1_same], &a_t_dp_array[r_1_same + 1]);

                delta_a_t_evals_tmp
                    .iter_mut()
                    .zip(delta_evals)
                    .zip(a_t_evals)
                    .for_each(|((s, delta_eval), a_t_eval)| *s += delta_eval * a_t_eval);
                delta_a_t_evals_tmp
            },
        );

        // in the parallel setting we need to reduce the results of the threads once more
        let mut evals;
        #[cfg(feature = "parallel")]
        {
            evals = delta_a_t_evals.reduce(
                || [G::ScalarField::ZERO; 3],
                |mut delta_a_t_evals_final, x| {
                    delta_a_t_evals_final
                        .iter_mut()
                        .zip(x)
                        .for_each(|(a, b)| *a += b);
                    delta_a_t_evals_final
                },
            )
        }
        #[cfg(not(feature = "parallel"))]
        {
            evals = delta_a_t_evals;
        }

        // multiply with gamma_b_value for the final result
        evals.iter_mut().for_each(|eval| *eval *= gamma_b_value);
        evals
    }
}
