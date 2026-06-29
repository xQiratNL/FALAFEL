use crate::client::ClientSchemes;
use crate::crypto_schemes::HyraxCommitmentOpening;
use crate::proof::hyrax_poly_eval::{
    HyraxPolyEvalProof, HyraxPolyEvalProofSeparator, HyraxTargetCommitment,
};
use crate::proof::sumcheck::util::{
    collapse, get_coeffs_degree_2, get_coeffs_degree_3, get_coeffs_degree_4, multi_collapse,
    point_evals_degree_2, point_evals_degree_3, point_evals_degree_4,
};
use crate::proof::sumcheck::{SumCheckProof, SumCheckProofSeparator};
use crate::proof::util::{double_tensor_chi, poly_eval, tensor_chi};
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

pub trait BackwardPassSumCheck1ProofSeparator<G: CurveGroup> {
    fn add_backward_pass_sum_check1(
        self,
        r_1_len: usize,
        r_2_len: usize,
        r_3_len: usize,
        r_4_len: usize,
    ) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> BackwardPassSumCheck1ProofSeparator<G>
    for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    fn add_backward_pass_sum_check1(
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
                    &format!("r_4_prev; round {round} coefficients commitment"),
                )
                .challenge_scalars(1, &format!("r_4_prev; round {round}"))
        }
        for round in 0..r_2_len {
            self = self
                .add_points(
                    1,
                    &format!("r_2_same; round {round} coefficients commitment"),
                )
                .challenge_scalars(1, &format!("r_2_same; round {round}"))
        }
        for round in 0..r_1_len {
            self = self
                .add_points(
                    1,
                    &format!("r_1_same; round {round} coefficients commitment"),
                )
                .challenge_scalars(1, &format!("r_1_same; round {round}"))
        }
        for round in 0..r_4_len {
            self = self
                .add_points(
                    1,
                    &format!("r_4_same; round {round} coefficients commitment"),
                )
                .challenge_scalars(1, &format!("r_4_same; round {round}"))
        }
        for round in 0..r_4_len {
            self = self
                .add_points(
                    1,
                    &format!("r_4_next; round {round} coefficients commitment"),
                )
                .challenge_scalars(1, &format!("r_4_next; round {round}"))
        }
        self.add_hyrax_target_commitment()
            .add_hyrax_target_commitment()
            .add_hyrax_target_commitment()
            .add_hyrax_target_commitment()
            .add_points(
                3,
                "target commitments to a_1 * a_2, b_1 * b_2 and (a_1 * a_2) * (b_1 * b_2)^T",
            )
            .add_sum_check(
                Some(r_3_len),
                r_4_len + r_2_len + r_1_len,
                r_4_len + r_4_len,
            )
    }
}

// proof for C = (A_1 hadamard A_2) matmul (B_1 hadamard B_2)^T
pub struct BackwardPassSumCheck1Proof<'a, G: CurveGroup> {
    // dimensions
    pub r_1_len: usize,
    pub r_2_len: usize,
    pub r_3_len: usize,
    pub r_4_len: usize,

    // other constants
    pub zero_step_idx: usize,
    pub mat_idx_shift: usize,

    // witnesses
    pub opening_a1_with_public: Option<&'a HyraxCommitmentOpening<G>>,
    pub opening_a2_with_public: Option<&'a HyraxCommitmentOpening<G>>,
    pub opening_b1_with_public: Option<&'a HyraxCommitmentOpening<G>>,
    pub opening_b2_with_public: Option<&'a HyraxCommitmentOpening<G>>,
    pub opening_c_with_public: Option<&'a HyraxCommitmentOpening<G>>,
}

impl<'a, G: CurveGroup> BackwardPassSumCheck1Proof<'a, G> {
    // Algorithm 2 in our paper
    pub fn prove(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<(
        [HyraxTargetCommitment<G>; 5],
        [Option<Vec<G::ScalarField>>; 4],
    )> {
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

        // 5-8: generate sum-check polynomials and remaining challenges (r_3, r_4_prev, r_2_same, r_1_same, r_4_same, r_4_next)
        // we use the approach from Section 5.4.2 (mentioned in Section 8.2.2) of https://eprint.iacr.org/2013/351.pdf
        // we also optimize based on the suggestions from https://eprint.iacr.org/2017/242.pdf, specifically Section 3.2, Appendix A & B

        // A: initialize the DP arrays in the order that's easiest to collapse.
        // I.e., the randomness ordering should adhere to r_4 > r_1 > r_2 > r_3 > r_4_prev > r_2_same > r_1_same > r_4_same > r_4_next

        // Initialize the DP array for beta_a, with r_1 given
        let mut beta_a_dp_array = tensor_chi(&r_1);

        // Initialize the DP array for beta_b, with r_2 given
        let mut beta_b_dp_array = tensor_chi(&r_2);

        // Initialize the DP array for gamma, with r_4 given
        let r_4_full_len = 1_usize << self.r_4_len;
        let mut gamma_dp_array = vec![G::ScalarField::ZERO; r_4_full_len.pow(3)];
        let mut gamma_values = tensor_chi(&r_4);
        for idx in (0..gamma_values.len()).step_by(self.zero_step_idx) {
            gamma_values[idx] = G::ScalarField::ZERO;
        }
        for idx in (0..gamma_values.len())
            .skip(self.zero_step_idx - 1)
            .step_by(self.zero_step_idx)
        {
            gamma_values[idx] = G::ScalarField::ZERO;
        }
        gamma_values.rotate_left(self.mat_idx_shift);

        for (entry, value) in gamma_dp_array
            .iter_mut()
            .skip(self.mat_idx_shift * r_4_full_len + 2 * self.mat_idx_shift * r_4_full_len.pow(2))
            .step_by(1 + r_4_full_len + r_4_full_len.pow(2))
            .zip(gamma_values)
        {
            *entry = value;
        }

        // useful constants for initializing DP arrays for matrices
        // A matrix
        let a_matrix_len: usize = 1 << self.r_4_len;
        let a_row_len: usize = 1 << self.r_1_len;
        let a_col_len: usize = 1 << self.r_3_len;
        // B matrices
        let b_matrix_len: usize = 1 << self.r_4_len;
        let b_row_len: usize = 1 << self.r_2_len; // row of B is column of B^T
        let b_col_len: usize = 1 << self.r_3_len; // column of B is row of B^T

        // Initialize the DP array for A_1 in order column, row, matrix, i.e., (r_3, r_1_same, r_4_next)
        let opening_a1 = self
            .opening_a1_with_public
            .ok_or_else(|| spongefish::ProofError::SerializationError)?;
        let mut a1_dp_array = vec![G::ScalarField::ZERO; a_matrix_len * a_row_len * a_col_len];
        ark_std::cfg_iter_mut!(a1_dp_array)
            .enumerate()
            .for_each(|(idx, entry)| {
                let col_idx = idx % a_col_len;
                let row_idx = (idx / a_col_len) % a_row_len;
                let matrix_idx = idx / (a_col_len * a_row_len);

                let opening_full_idx =
                    col_idx * (a_matrix_len * a_row_len) + row_idx * a_matrix_len + matrix_idx;
                let (opening_col, opening_row) =
                    opening_full_idx.div_rem_euclid(&opening_a1.input_matrix.len());

                *entry = opening_a1.input_matrix[opening_row][opening_col]
            });
        let a1_dp_array_cloned = Some(a1_dp_array.clone());

        // Initialize the DP array for A_2 in order column, row, matrix, i.e., (r_3, r_1_same, r_4_same)
        let opening_a2 = self
            .opening_a2_with_public
            .ok_or_else(|| spongefish::ProofError::SerializationError)?;
        let mut a2_dp_array = vec![G::ScalarField::ZERO; a_matrix_len * a_row_len * a_col_len];
        ark_std::cfg_iter_mut!(a2_dp_array)
            .enumerate()
            .for_each(|(idx, entry)| {
                let col_idx = idx % a_col_len;
                let row_idx = (idx / a_col_len) % a_row_len;
                let matrix_idx = idx / (a_col_len * a_row_len);

                let opening_full_idx =
                    col_idx * (a_matrix_len * a_row_len) + row_idx * a_matrix_len + matrix_idx;
                let (opening_col, opening_row) =
                    opening_full_idx.div_rem_euclid(&opening_a2.input_matrix.len());

                *entry = opening_a2.input_matrix[opening_row][opening_col]
            });
        let a2_dp_array_cloned = None; // since we don't use it, no need to clone it

        // Initialize the DP array for B_1^T in order row, matrix, column, i.e., (r_3, r_4_prev, r_2_same)
        let opening_b1 = self
            .opening_b1_with_public
            .ok_or_else(|| spongefish::ProofError::SerializationError)?;
        let mut b1_t_dp_array = vec![G::ScalarField::ZERO; b_matrix_len * b_row_len * b_col_len];
        ark_std::cfg_iter_mut!(b1_t_dp_array)
            .enumerate()
            .for_each(|(idx, entry)| {
                let col_idx = idx % b_col_len; // column of B_1 is row of B_1^T
                let matrix_idx = (idx / b_col_len) % b_matrix_len;
                let row_idx = idx / (b_col_len * b_matrix_len); // row of B_1 is column of B_1^T

                let opening_full_idx =
                    col_idx * (b_matrix_len * b_row_len) + row_idx * b_matrix_len + matrix_idx;
                let (opening_col, opening_row) =
                    opening_full_idx.div_rem_euclid(&opening_b1.input_matrix.len());

                *entry = opening_b1.input_matrix[opening_row][opening_col]
            });
        let b1_dp_array_cloned = None; // since we don't use it, no need to clone it

        // Initialize the DP array for B_1^T in order row, matrix, column, i.e., (r_3, r_4_prev, r_2_same)
        let opening_b2 = self
            .opening_b2_with_public
            .ok_or_else(|| spongefish::ProofError::SerializationError)?;
        let mut b2_t_dp_array = vec![G::ScalarField::ZERO; b_matrix_len * b_row_len * b_col_len];
        ark_std::cfg_iter_mut!(b2_t_dp_array)
            .enumerate()
            .for_each(|(idx, entry)| {
                let col_idx = idx % b_col_len; // column of B_2 is row of B_2^T
                let matrix_idx = (idx / b_col_len) % b_matrix_len;
                let row_idx = idx / (b_col_len * b_matrix_len); // row of B_2 is column of B_2^T

                let opening_full_idx =
                    col_idx * (b_matrix_len * b_row_len) + row_idx * b_matrix_len + matrix_idx;
                let (opening_col, opening_row) =
                    opening_full_idx.div_rem_euclid(&opening_b2.input_matrix.len());

                *entry = opening_b2.input_matrix[opening_row][opening_col]
            });
        let b2_dp_array_cloned = None; // since we don't use it, no need to clone it

        // B: Collapse over r_4, r_1, r_2 as those are fixed
        // beta_a is already given over r_1, so no need to collapse
        // beta_b is already given over r_2, so no need to collapse
        // gamma is already given over r_4, so no need to collapse

        // step-by-step computation of polynomials in most efficient order first r_3, then r_4_prev, r_2_same, r_1_same, r_4_same, r_4_next
        // this also collapses the DP arrays after each step

        // in the r_3 part we construct polynomials of degree 4 (so we interpolate on 5 points)
        let mut r_3 = vec![G::ScalarField::ZERO; self.r_3_len];
        let mut cms_coeffs_r_3 = vec![G::Affine::zero(); self.r_3_len];
        let mut coeffs_r_3 = vec![[G::ScalarField::ZERO; 5]; self.r_3_len];
        let mut r_coeffs_r_3 = vec![G::ScalarField::ZERO; self.r_3_len];
        for r_3_idx in 0..self.r_3_len {
            // evaluate polynomial and compute coefficients
            let [eval_minus2, eval_minus1, eval_0, eval_1, eval_2] = self.eval_r_3(
                &beta_a_dp_array,
                &beta_b_dp_array,
                &gamma_dp_array,
                &a1_dp_array,
                &a2_dp_array,
                &b1_t_dp_array,
                &b2_t_dp_array,
            );
            let coeffs = get_coeffs_degree_4(
                eval_minus2,
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
            a1_dp_array = collapse(&a1_dp_array, &r_3_entry);
            a2_dp_array = collapse(&a2_dp_array, &r_3_entry);
            b1_t_dp_array = collapse(&b1_t_dp_array, &r_3_entry);
            b2_t_dp_array = collapse(&b2_t_dp_array, &r_3_entry);
            r_3[r_3_idx] = r_3_entry;

            cms_coeffs_r_3[r_3_idx] = cm_coeffs_r_3;
            coeffs_r_3[r_3_idx] = coeffs;
            r_coeffs_r_3[r_3_idx] = r_coeffs;
        }

        // in the r_4_prev part we construct polynomials of degree 3 (so we interpolate on 4 points)
        let mut r_4_prev = vec![G::ScalarField::ZERO; self.r_4_len];
        let mut cms_coeffs_r_4_prev = vec![G::Affine::zero(); self.r_4_len];
        let mut coeffs_r_4_prev = vec![[G::ScalarField::ZERO; 4]; self.r_4_len];
        let mut r_coeffs_r_4_prev = vec![G::ScalarField::ZERO; self.r_4_len];
        let a_dp_array = self.compute_a_dp_array(&beta_a_dp_array, &a1_dp_array, &a2_dp_array); // we can precompute this, as it is the same for each iteration
        for r_4_prev_idx in 0..self.r_4_len {
            // evaluate polynomial and compute coefficients
            let [eval_minus1, eval_0, eval_1, eval_2] = self.eval_r_4_prev(
                &beta_b_dp_array,
                &gamma_dp_array,
                &a_dp_array,
                &b1_t_dp_array,
                &b2_t_dp_array,
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
                if r_4_prev_idx == 0 {
                    poly_eval(coeffs_r_3.last().unwrap(), r_3.last().unwrap())
                } else {
                    poly_eval(
                        &coeffs_r_4_prev[r_4_prev_idx - 1],
                        &r_4_prev[r_4_prev_idx - 1],
                    )
                }
            );

            // commit to coefficients
            let r_coeffs = G::ScalarField::rand(prover_state.rng());
            let cm_coeffs_r_4_prev = client_schemes
                .vector_commit
                .vector_commit(&coeffs, &r_coeffs);

            // get new randomness
            prover_state.add_points(&[cm_coeffs_r_4_prev.into_group()])?;
            let [r_4_prev_entry]: [_; 1] = prover_state.challenge_scalars()?;

            // collapse on new randomness and add to r_4_prev array
            gamma_dp_array = collapse(&gamma_dp_array, &r_4_prev_entry);
            b1_t_dp_array = collapse(&b1_t_dp_array, &r_4_prev_entry);
            b2_t_dp_array = collapse(&b2_t_dp_array, &r_4_prev_entry);
            r_4_prev[r_4_prev_idx] = r_4_prev_entry;

            cms_coeffs_r_4_prev[r_4_prev_idx] = cm_coeffs_r_4_prev;
            coeffs_r_4_prev[r_4_prev_idx] = coeffs;
            r_coeffs_r_4_prev[r_4_prev_idx] = r_coeffs;
        }

        // in the r_2_same part we construct polynomials of degree 3 (so we interpolate on 4 points)
        let mut r_2_same = vec![G::ScalarField::ZERO; self.r_2_len];
        let mut cms_coeffs_r_2_same = vec![G::Affine::zero(); self.r_2_len];
        let mut coeffs_r_2_same = vec![[G::ScalarField::ZERO; 4]; self.r_2_len];
        let mut r_coeffs_r_2_same = vec![G::ScalarField::ZERO; self.r_2_len];
        let gamma_a_value = self.compute_gamma_a_value(&gamma_dp_array, &a_dp_array);
        for r_2_same_idx in 0..self.r_2_len {
            // evaluate polynomial and compute coefficients
            let [eval_minus1, eval_0, eval_1, eval_2] = self.eval_r_2_same(
                &beta_b_dp_array,
                &gamma_a_value,
                &b1_t_dp_array,
                &b2_t_dp_array,
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
                    poly_eval(coeffs_r_4_prev.last().unwrap(), r_4_prev.last().unwrap())
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
            beta_b_dp_array = collapse(&beta_b_dp_array, &r_2_same_entry);
            b1_t_dp_array = collapse(&b1_t_dp_array, &r_2_same_entry);
            b2_t_dp_array = collapse(&b2_t_dp_array, &r_2_same_entry);
            r_2_same[r_2_same_idx] = r_2_same_entry;

            cms_coeffs_r_2_same[r_2_same_idx] = cm_coeffs_r_2_same;
            coeffs_r_2_same[r_2_same_idx] = coeffs;
            r_coeffs_r_2_same[r_2_same_idx] = r_coeffs;
        }

        // in the r_1_same part we construct polynomials of degree 3 (so we interpolate on 4 points)
        let mut r_1_same = vec![G::ScalarField::ZERO; self.r_1_len];
        let mut cms_coeffs_r_1_same = vec![G::Affine::zero(); self.r_1_len];
        let mut coeffs_r_1_same = vec![[G::ScalarField::ZERO; 4]; self.r_1_len];
        let mut r_coeffs_r_1_same = vec![G::ScalarField::ZERO; self.r_1_len];
        let b_t_value = beta_b_dp_array[0] * b1_t_dp_array[0] * b2_t_dp_array[0];
        for r_1_same_idx in 0..self.r_1_len {
            // evaluate polynomial and compute coefficients
            let [eval_minus1, eval_0, eval_1, eval_2] = self.eval_r_1_same(
                &beta_a_dp_array,
                &gamma_dp_array,
                &a1_dp_array,
                &a2_dp_array,
                &b_t_value,
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
                if r_1_same_idx == 0 {
                    poly_eval(coeffs_r_2_same.last().unwrap(), r_2_same.last().unwrap())
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
            beta_a_dp_array = collapse(&beta_a_dp_array, &r_1_same_entry);
            a1_dp_array = collapse(&a1_dp_array, &r_1_same_entry);
            a2_dp_array = collapse(&a2_dp_array, &r_1_same_entry);
            r_1_same[r_1_same_idx] = r_1_same_entry;

            cms_coeffs_r_1_same[r_1_same_idx] = cm_coeffs_r_1_same;
            coeffs_r_1_same[r_1_same_idx] = coeffs;
            r_coeffs_r_1_same[r_1_same_idx] = r_coeffs;
        }

        // in the r_4_same part we construct polynomials of degree 2 (so we interpolate on 3 points)
        let mut r_4_same = vec![G::ScalarField::ZERO; self.r_4_len];
        let mut cms_coeffs_r_4_same = vec![G::Affine::zero(); self.r_4_len];
        let mut coeffs_r_4_same = vec![[G::ScalarField::ZERO; 3]; self.r_4_len];
        let mut r_coeffs_r_4_same = vec![G::ScalarField::ZERO; self.r_4_len];
        let beta_a_b_t_value = beta_a_dp_array[0] * b_t_value;
        for r_4_same_idx in 0..self.r_4_len {
            // evaluate polynomial and compute coefficients
            let [eval_minus1, eval_0, eval_1] = self.eval_r_4_same(
                &gamma_dp_array,
                &a1_dp_array,
                &a2_dp_array,
                &beta_a_b_t_value,
            );
            let coeffs =
                get_coeffs_degree_2(eval_minus1, eval_0, eval_1, &client_schemes.constants);

            debug_assert_eq!(
                eval_0 + eval_1,
                if r_4_same_idx == 0 {
                    poly_eval(coeffs_r_1_same.last().unwrap(), r_1_same.last().unwrap())
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
            a2_dp_array = collapse(&a2_dp_array, &r_4_same_entry);
            r_4_same[r_4_same_idx] = r_4_same_entry;

            cms_coeffs_r_4_same[r_4_same_idx] = cm_coeffs_r_4_same;
            coeffs_r_4_same[r_4_same_idx] = coeffs;
            r_coeffs_r_4_same[r_4_same_idx] = r_coeffs;
        }

        // in the r_4_next part we construct polynomials of degree 2 (so we interpolate on 3 points)
        let mut r_4_next = vec![G::ScalarField::ZERO; self.r_4_len];
        let mut cms_coeffs_r_4_next = vec![G::Affine::zero(); self.r_4_len];
        let mut coeffs_r_4_next = vec![[G::ScalarField::ZERO; 3]; self.r_4_len];
        let mut r_coeffs_r_4_next = vec![G::ScalarField::ZERO; self.r_4_len];
        let beta_a_a2_b_t_value = a2_dp_array[0] * beta_a_b_t_value;
        for r_4_next_idx in 0..self.r_4_len {
            // evaluate polynomial and compute coefficients
            let [eval_minus1, eval_0, eval_1] =
                self.eval_r_4_next(&gamma_dp_array, &a1_dp_array, &beta_a_a2_b_t_value);
            let coeffs =
                get_coeffs_degree_2(eval_minus1, eval_0, eval_1, &client_schemes.constants);

            debug_assert_eq!(
                eval_0 + eval_1,
                if r_4_next_idx == 0 {
                    poly_eval(coeffs_r_4_same.last().unwrap(), r_4_same.last().unwrap())
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
            a1_dp_array = collapse(&a1_dp_array, &r_4_next_entry);
            r_4_next[r_4_next_idx] = r_4_next_entry;

            cms_coeffs_r_4_next[r_4_next_idx] = cm_coeffs_r_4_next;
            coeffs_r_4_next[r_4_next_idx] = coeffs;
            r_coeffs_r_4_next[r_4_next_idx] = r_coeffs;
        }

        // 9: evaluate A_1, A_2, B_1, and B_2 on the appropriate challenges from r_1,..,r_4_next
        // no need for beta_a, beta_b, gamma, as the verifier knows the polynomial so can compute it directly
        // note: for b_1_t and b_2_t we have swapped r_3 and r_2_same order, since we compute the transpose of b and not b
        let a1_target_proof = HyraxPolyEvalProof {
            points: &[r_4_next.as_slice(), r_1_same.as_slice(), r_3.as_slice()].concat(),
            commitment_opening_input_matrix: self.opening_a1_with_public.map(|x| &x.input_matrix),
        }
        .target_commitment_prove(prover_state, client_schemes)?;
        let a2_target_proof = HyraxPolyEvalProof {
            points: &[r_4_same.as_slice(), r_1_same.as_slice(), r_3.as_slice()].concat(),
            commitment_opening_input_matrix: self.opening_a2_with_public.map(|x| &x.input_matrix),
        }
        .target_commitment_prove(prover_state, client_schemes)?;
        let b1_t_target_proof = HyraxPolyEvalProof {
            points: &[r_4_prev.as_slice(), r_2_same.as_slice(), r_3.as_slice()].concat(),
            commitment_opening_input_matrix: self.opening_b1_with_public.map(|x| &x.input_matrix),
        }
        .target_commitment_prove(prover_state, client_schemes)?;
        let b2_t_target_proof = HyraxPolyEvalProof {
            points: &[r_4_prev.as_slice(), r_2_same.as_slice(), r_3.as_slice()].concat(),
            commitment_opening_input_matrix: self.opening_b2_with_public.map(|x| &x.input_matrix),
        }
        .target_commitment_prove(prover_state, client_schemes)?;

        // compute a commitment to a_1 * a_2, b_1 * b_2 and (a_1 * a_2) * (b_1 * b_2)
        let a1_a2 =
            a1_target_proof.ltr.expect("Should exist") * a2_target_proof.ltr.expect("Should exist");
        let b1_b2 = b1_t_target_proof.ltr.expect("Should exist")
            * b2_t_target_proof.ltr.expect("Should exist");
        let a_b = a1_a2 * b1_b2;
        let r_a1_a2 = G::ScalarField::rand(prover_state.rng());
        let r_b1_b2 = G::ScalarField::rand(prover_state.rng());
        let r_a_b = G::ScalarField::rand(prover_state.rng());
        let cm_a1_a2 = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&a1_a2, &r_a1_a2);
        let cm_b1_b2 = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&b1_b2, &r_b1_b2);
        let cm_a_b = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&a_b, &r_a_b);
        prover_state.add_points(&[
            cm_a1_a2.into_group(),
            cm_b1_b2.into_group(),
            cm_a_b.into_group(),
        ])?;

        debug_assert_eq!(
            poly_eval(coeffs_r_4_next.last().unwrap(), r_4_next.last().unwrap()),
            a_b * beta_a_dp_array[0] * beta_b_dp_array[0] * gamma_dp_array[0]
        );

        // 10: sum-check proof
        SumCheckProof {
            cms_coeffs_degree4: Some(&cms_coeffs_r_3),
            cms_coeffs_degree3: &[
                cms_coeffs_r_4_prev.as_slice(),
                cms_coeffs_r_2_same.as_slice(),
                cms_coeffs_r_1_same.as_slice(),
            ]
            .concat(),
            cms_coeffs_degree2: &[
                cms_coeffs_r_4_same.as_slice(),
                cms_coeffs_r_4_next.as_slice(),
            ]
            .concat(),
            cm_a1: &a1_target_proof.omega.expect("This should exist"),
            cm_a2: Some(&a2_target_proof.omega.expect("This should exist")),
            cm_b1: &b1_t_target_proof.omega.expect("This should exist"),
            cm_b2: &b2_t_target_proof.omega.expect("This should exist"),
            cm_c: &c_target_proof.omega.expect("This should exist"),
            cm_a1_a2: Some(&cm_a1_a2),
            cm_b1_b2: &cm_b1_b2,
            cm_a_b: &cm_a_b,
            multiplier: &(beta_a_dp_array[0] * beta_b_dp_array[0] * gamma_dp_array[0]),
            r_degree_4: Some(&r_3),
            r_degree_3: &[
                r_4_prev.as_slice(),
                r_2_same.as_slice(),
                r_1_same.as_slice(),
            ]
            .concat(),
            r_degree_2: &[r_4_same.as_slice(), r_4_next.as_slice()].concat(),
            coeffs_degree4: Some(Some(&coeffs_r_3)),
            r_coeffs_degree4: Some(Some(&r_coeffs_r_3)),
            coeffs_degree3: Some(
                &[
                    coeffs_r_4_prev.as_slice(),
                    coeffs_r_2_same.as_slice(),
                    coeffs_r_1_same.as_slice(),
                ]
                .concat(),
            ),
            r_coeffs_degree3: Some(
                &[
                    r_coeffs_r_4_prev.as_slice(),
                    r_coeffs_r_2_same.as_slice(),
                    r_coeffs_r_1_same.as_slice(),
                ]
                .concat(),
            ),
            coeffs_degree2: Some(
                &[coeffs_r_4_same.as_slice(), coeffs_r_4_next.as_slice()].concat(),
            ),
            r_coeffs_degree2: Some(
                &[r_coeffs_r_4_same.as_slice(), r_coeffs_r_4_next.as_slice()].concat(),
            ),
            a1: Some(&a1_target_proof.ltr.expect("Should exist")),
            r_a1: Some(&a1_target_proof.r_omega.expect("Should exist")),
            a2: Some(Some(&a2_target_proof.ltr.expect("Should exist"))),
            r_a2: Some(Some(&a2_target_proof.r_omega.expect("Should exist"))),
            b1: Some(&b1_t_target_proof.ltr.expect("Should exist")),
            r_b1: Some(&b1_t_target_proof.r_omega.expect("Should exist")),
            b2: Some(&b2_t_target_proof.ltr.expect("Should exist")),
            r_b2: Some(&b2_t_target_proof.r_omega.expect("Should exist")),
            r_c: Some(&c_target_proof.r_omega.expect("Should exist")),
            a1_a2: Some(Some(&a1_a2)),
            r_a1_a2: Some(Some(&r_a1_a2)),
            b1_b2: Some(&b1_b2),
            r_b1_b2: Some(&r_b1_b2),
            r_a_b: Some(&r_a_b),
        }
        .prove(true, prover_state, client_schemes)?;

        Ok((
            [
                a1_target_proof,
                a2_target_proof,
                b1_t_target_proof,
                b2_t_target_proof,
                c_target_proof,
            ],
            [
                a1_dp_array_cloned,
                a2_dp_array_cloned,
                b1_dp_array_cloned,
                b2_dp_array_cloned,
            ],
        ))
    }

    pub fn verify(
        self,
        verifier_state: &mut VerifierState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<[HyraxTargetCommitment<G>; 5]> {
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

        // r_4_prev part
        let mut cms_coeffs_r_4_prev = vec![G::Affine::zero(); self.r_4_len];
        let mut r_4_prev = vec![G::ScalarField::ZERO; self.r_4_len];
        for r_4_prev_idx in 0..self.r_4_len {
            cms_coeffs_r_4_prev[r_4_prev_idx] = verifier_state
                .next_points::<1>()?
                .map(|cm: G| cm.into_affine())[0];
            r_4_prev[r_4_prev_idx] = verifier_state.challenge_scalars::<1>()?[0];
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

        // r_1_same part
        let mut cms_coeffs_r_1_same = vec![G::Affine::zero(); self.r_1_len];
        let mut r_1_same = vec![G::ScalarField::ZERO; self.r_1_len];
        for r_1_same_idx in 0..self.r_1_len {
            cms_coeffs_r_1_same[r_1_same_idx] = verifier_state
                .next_points::<1>()?
                .map(|cm: G| cm.into_affine())[0];
            r_1_same[r_1_same_idx] = verifier_state.challenge_scalars::<1>()?[0];
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

        // r_4_next part
        let mut cms_coeffs_r_4_next = vec![G::Affine::zero(); self.r_4_len];
        let mut r_4_next = vec![G::ScalarField::ZERO; self.r_4_len];
        for r_4_next_idx in 0..self.r_4_len {
            cms_coeffs_r_4_next[r_4_next_idx] = verifier_state
                .next_points::<1>()?
                .map(|cm: G| cm.into_affine())[0];
            r_4_next[r_4_next_idx] = verifier_state.challenge_scalars::<1>()?[0];
        }

        // 9
        // note: for b_1_t and b_2_t we have swapped r_3 and r_2_same order, since we compute the transpose of b and not b
        let a1_target_proof = HyraxPolyEvalProof {
            points: &[r_4_next.as_slice(), r_1_same.as_slice(), r_3.as_slice()].concat(),
            commitment_opening_input_matrix: None,
        }
        .target_commitment_verify(verifier_state)?;
        let a2_target_proof = HyraxPolyEvalProof {
            points: &[r_4_same.as_slice(), r_1_same.as_slice(), r_3.as_slice()].concat(),
            commitment_opening_input_matrix: None,
        }
        .target_commitment_verify(verifier_state)?;
        let b1_t_target_proof = HyraxPolyEvalProof {
            points: &[r_4_prev.as_slice(), r_2_same.as_slice(), r_3.as_slice()].concat(),
            commitment_opening_input_matrix: None,
        }
        .target_commitment_verify(verifier_state)?;
        let b2_t_target_proof = HyraxPolyEvalProof {
            points: &[r_4_prev.as_slice(), r_2_same.as_slice(), r_3.as_slice()].concat(),
            commitment_opening_input_matrix: None,
        }
        .target_commitment_verify(verifier_state)?;
        let [cm_a1_a2, cm_b1_b2, cm_a_b]: [_; 3] =
            verifier_state.next_points()?.map(|cm: G| cm.into_affine());

        // evaluate beta_a, beta_b and gamma
        // for beta_a we can use the simple formulation
        let beta_a_eval = ark_std::cfg_iter!(r_1)
            .zip(ark_std::cfg_iter!(r_1_same))
            .map(|(&x, y)| x * y + (G::ScalarField::ONE - x) * (G::ScalarField::ONE - y))
            .product::<G::ScalarField>();

        // for beta_b we can use the simple formulation
        let beta_b_eval = ark_std::cfg_iter!(r_2)
            .zip(ark_std::cfg_iter!(r_2_same))
            .map(|(&x, y)| x * y + (G::ScalarField::ONE - x) * (G::ScalarField::ONE - y))
            .product::<G::ScalarField>();

        // for gamma we can use the simple formulation for the (r_4, r_4_same) part and use the regular way to link that to r_4_prev and r_4_next
        let r_4_full_len = 1_usize << self.r_4_len;
        let mut gamma_dp_array = vec![G::ScalarField::ZERO; r_4_full_len.pow(2)];
        let mut gamma_values = double_tensor_chi(&r_4, &r_4_same);
        for idx in (0..gamma_values.len()).step_by(self.zero_step_idx) {
            gamma_values[idx] = G::ScalarField::ZERO;
        }
        for idx in (0..gamma_values.len())
            .skip(self.zero_step_idx - 1)
            .step_by(self.zero_step_idx)
        {
            gamma_values[idx] = G::ScalarField::ZERO;
        }
        gamma_values.rotate_left(self.mat_idx_shift);
        for (entry, value) in gamma_dp_array
            .iter_mut()
            .skip(2 * self.mat_idx_shift * r_4_full_len)
            .step_by(1 + r_4_full_len)
            .zip(gamma_values)
        {
            *entry = value;
        }
        gamma_dp_array = multi_collapse(&gamma_dp_array, &r_4_prev);
        gamma_dp_array = multi_collapse(&gamma_dp_array, &r_4_next);

        // 10: sum-check proof
        SumCheckProof {
            cms_coeffs_degree4: Some(&cms_coeffs_r_3),
            cms_coeffs_degree3: &[
                cms_coeffs_r_4_prev.as_slice(),
                cms_coeffs_r_2_same.as_slice(),
                cms_coeffs_r_1_same.as_slice(),
            ]
            .concat(),
            cms_coeffs_degree2: &[
                cms_coeffs_r_4_same.as_slice(),
                cms_coeffs_r_4_next.as_slice(),
            ]
            .concat(),
            cm_a1: &a1_target_proof.omega.expect("This should exist"),
            cm_a2: Some(&a2_target_proof.omega.expect("This should exist")),
            cm_b1: &b1_t_target_proof.omega.expect("This should exist"),
            cm_b2: &b2_t_target_proof.omega.expect("This should exist"),
            cm_c: &c_target_proof.omega.expect("This should exist"),
            cm_a1_a2: Some(&cm_a1_a2),
            cm_b1_b2: &cm_b1_b2,
            cm_a_b: &cm_a_b,
            multiplier: &(beta_a_eval * beta_b_eval * gamma_dp_array[0]),
            r_degree_4: Some(&r_3),
            r_degree_3: &[
                r_4_prev.as_slice(),
                r_2_same.as_slice(),
                r_1_same.as_slice(),
            ]
            .concat(),
            r_degree_2: &[r_4_same.as_slice(), r_4_next.as_slice()].concat(),
            coeffs_degree4: Some(None),
            r_coeffs_degree4: Some(None),
            coeffs_degree3: None,
            r_coeffs_degree3: None,
            coeffs_degree2: None,
            r_coeffs_degree2: None,
            a1: None,
            r_a1: None,
            a2: Some(None),
            r_a2: Some(None),
            b1: None,
            r_b1: None,
            b2: None,
            r_b2: None,
            r_c: None,
            a1_a2: Some(None),
            r_a1_a2: Some(None),
            b1_b2: None,
            r_b1_b2: None,
            r_a_b: None,
        }
        .verify(true, verifier_state, client_schemes)?;

        Ok([
            a1_target_proof,
            a2_target_proof,
            b1_t_target_proof,
            b2_t_target_proof,
            c_target_proof,
        ])
    }

    fn eval_r_3(
        &self,
        beta_a_dp_array: &[G::ScalarField],
        beta_b_dp_array: &[G::ScalarField],
        gamma_dp_array: &[G::ScalarField],
        a1_dp_array: &[G::ScalarField],
        a2_dp_array: &[G::ScalarField],
        b1_t_dp_array: &[G::ScalarField],
        b2_t_dp_array: &[G::ScalarField],
    ) -> [G::ScalarField; 5] {
        let r_1_max = 1 << self.r_1_len;
        let r_2_max = 1 << self.r_2_len;
        let r_4_max: usize = 1 << self.r_4_len;
        let r_4_max_double = r_4_max.pow(2);
        let r_3_max = a1_dp_array.len() / (r_1_max * r_4_max);

        let evals = ark_std::cfg_into_iter!(0..r_3_max).step_by(2).fold(
            #[cfg(feature = "parallel")]
            || [G::ScalarField::ZERO; 5],
            #[cfg(not(feature = "parallel"))]
            [G::ScalarField::ZERO; 5],
            |mut evals_tmp, r_3| {
                // evals
                let a_b_evals = (1..r_4_max - 1).fold(
                    [G::ScalarField::ZERO; 5],
                    |mut a_b_evals_tmp, r_4_same| {
                        // indices
                        let r_4_prev = r_4_same - self.mat_idx_shift;
                        let r_4_next = r_4_same + self.mat_idx_shift;
                        let gamma_idx = r_4_prev + r_4_same * r_4_max + r_4_next * r_4_max_double;

                        // evals
                        let a_evals =
                            (0..r_1_max).fold([G::ScalarField::ZERO; 5], |mut a_evals_tmp, r_1| {
                                // indices
                                let inner_a1_idx =
                                    r_3 + r_1 * r_3_max + r_4_next * r_3_max * r_1_max;
                                let inner_a2_idx =
                                    r_3 + r_1 * r_3_max + r_4_same * r_3_max * r_1_max;

                                // evals
                                let a1_evals = point_evals_degree_4(
                                    &a1_dp_array[inner_a1_idx],
                                    &a1_dp_array[inner_a1_idx + 1],
                                );
                                let a2_evals = point_evals_degree_4(
                                    &a2_dp_array[inner_a2_idx],
                                    &a2_dp_array[inner_a2_idx + 1],
                                );

                                // summing the evals
                                a_evals_tmp.iter_mut().zip(a1_evals).zip(a2_evals).for_each(
                                    |((s, a1_eval), a2_eval)| {
                                        *s += beta_a_dp_array[r_1] * a1_eval * a2_eval
                                    },
                                );
                                a_evals_tmp
                            });
                        let b_t_evals = (0..r_2_max).fold(
                            [G::ScalarField::ZERO; 5],
                            |mut b_t_evals_tmp, r_2| {
                                // indices
                                let inner_b_t_idx =
                                    r_3 + r_4_prev * r_3_max + r_2 * r_3_max * r_4_max; // b1 and b2 are the same

                                // evals
                                let b1_evals = point_evals_degree_4(
                                    &b1_t_dp_array[inner_b_t_idx],
                                    &b1_t_dp_array[inner_b_t_idx + 1],
                                );
                                let b2_evals = point_evals_degree_4(
                                    &b2_t_dp_array[inner_b_t_idx],
                                    &b2_t_dp_array[inner_b_t_idx + 1],
                                );

                                //summing the evals
                                b_t_evals_tmp
                                    .iter_mut()
                                    .zip(b1_evals)
                                    .zip(b2_evals)
                                    .for_each(|((s, b1_eval), b2_eval)| {
                                        *s += beta_b_dp_array[r_2] * b1_eval * b2_eval
                                    });
                                b_t_evals_tmp
                            },
                        );

                        // summing the evals;
                        a_b_evals_tmp
                            .iter_mut()
                            .zip(a_evals)
                            .zip(b_t_evals)
                            .for_each(|((s, a_eval), b_t_eval)| {
                                *s += gamma_dp_array[gamma_idx] * a_eval * b_t_eval
                            });
                        a_b_evals_tmp
                    },
                );

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
                || [G::ScalarField::ZERO; 5],
                |mut evals_final, x| {
                    evals_final.iter_mut().zip(x).for_each(|(a, b)| *a += b);
                    evals_final
                },
            )
        }
        #[cfg(not(feature = "parallel"))]
        evals
    }

    fn compute_a_dp_array(
        &self,
        beta_a_dp_array: &[G::ScalarField],
        a1_dp_array: &[G::ScalarField],
        a2_dp_array: &[G::ScalarField],
    ) -> Vec<G::ScalarField> {
        let r_1_max = 1 << self.r_1_len;
        let mut result = ark_std::cfg_chunks!(a1_dp_array, r_1_max)
            .skip(2)
            .zip(ark_std::cfg_chunks!(a2_dp_array, r_1_max).skip(1))
            .map(|(a1_chunk, a2_chunk)| {
                a1_chunk
                    .into_iter()
                    .zip(a2_chunk)
                    .zip(beta_a_dp_array)
                    .map(|((&a1_val, a2_val), beta_a_val)| a1_val * a2_val * beta_a_val)
                    .sum::<G::ScalarField>()
            })
            .collect::<Vec<_>>();
        result.insert(0, G::ScalarField::ZERO);
        result.push(G::ScalarField::ZERO);
        result
    }

    fn eval_r_4_prev(
        &self,
        beta_b_dp_array: &[G::ScalarField],
        gamma_dp_array: &[G::ScalarField],
        a_dp_array: &[G::ScalarField],
        b1_t_dp_array: &[G::ScalarField],
        b2_t_dp_array: &[G::ScalarField],
    ) -> [G::ScalarField; 4] {
        let r_2_max = 1 << self.r_2_len;
        let r_4_same_max = 1 << self.r_4_len;
        let r_4_prev_max = b1_t_dp_array.len() / r_2_max;
        let r_4_same_fp_max = r_4_same_max / r_4_prev_max;
        let r_4_max_double = r_4_prev_max * r_4_same_max;

        let evals = ark_std::cfg_into_iter!(0..r_4_prev_max).step_by(2).fold(
            #[cfg(feature = "parallel")]
            || [G::ScalarField::ZERO; 4],
            #[cfg(not(feature = "parallel"))]
            [G::ScalarField::ZERO; 4],
            |mut evals_tmp, r_4_prev| {
                // evals
                let gamma_a_evals = (r_4_prev * r_4_same_fp_max + 1
                    ..min(
                        (r_4_prev + 1) * r_4_same_fp_max + r_4_same_fp_max + 1,
                        r_4_same_max - self.mat_idx_shift,
                    ))
                    .fold(
                        [G::ScalarField::ZERO; 4],
                        |mut gamma_a_evals_tmp, r_4_same| {
                            // indices
                            let r_4_next = r_4_same + self.mat_idx_shift;
                            let gamma_idx =
                                r_4_prev + r_4_same * r_4_prev_max + r_4_next * r_4_max_double;

                            // evals
                            let gamma_evals = point_evals_degree_3(
                                &gamma_dp_array[gamma_idx],
                                &gamma_dp_array[gamma_idx + 1],
                            );

                            // summing the evals
                            gamma_a_evals_tmp.iter_mut().zip(gamma_evals).for_each(
                                |(s, gamma_eval)| *s += gamma_eval * a_dp_array[r_4_same],
                            );
                            gamma_a_evals_tmp
                        },
                    );
                let b_t_evals =
                    (0..r_2_max).fold([G::ScalarField::ZERO; 4], |mut b_t_evals_tmp, r_2| {
                        // indices
                        let inner_b_t_idx = r_4_prev + r_2 * r_4_prev_max;

                        // evals
                        let b1_evals = point_evals_degree_3(
                            &b1_t_dp_array[inner_b_t_idx],
                            &b1_t_dp_array[inner_b_t_idx + 1],
                        );
                        let b2_evals = point_evals_degree_3(
                            &b2_t_dp_array[inner_b_t_idx],
                            &b2_t_dp_array[inner_b_t_idx + 1],
                        );

                        // summing the evals
                        b_t_evals_tmp
                            .iter_mut()
                            .zip(b1_evals)
                            .zip(b2_evals)
                            .for_each(|((s, b1_eval), b2_eval)| {
                                *s += beta_b_dp_array[r_2] * b1_eval * b2_eval
                            });
                        b_t_evals_tmp
                    });

                // summing the evals
                evals_tmp
                    .iter_mut()
                    .zip(gamma_a_evals)
                    .zip(b_t_evals)
                    .for_each(|((s, gamma_a_eval), b_t_eval)| *s += gamma_a_eval * b_t_eval);
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

    fn compute_gamma_a_value(
        &self,
        gamma_dp_array: &[G::ScalarField],
        a_dp_array: &[G::ScalarField],
    ) -> G::ScalarField {
        let r_4_max = 1 << self.r_4_len;
        ark_std::cfg_iter!(a_dp_array[..a_dp_array.len() - 1])
            .enumerate()
            .map(|(r_4_same, a_val)| {
                let r_4_next = r_4_same + 1;
                gamma_dp_array[r_4_same + r_4_next * r_4_max] * a_val
            })
            .sum()
    }

    fn eval_r_2_same(
        &self,
        beta_b_dp_array: &[G::ScalarField],
        gamma_a_value: &G::ScalarField,
        b1_t_dp_array: &[G::ScalarField],
        b2_t_dp_array: &[G::ScalarField],
    ) -> [G::ScalarField; 4] {
        let r_2_max = beta_b_dp_array.len();

        let b_t_evals = ark_std::cfg_into_iter!(0..r_2_max).step_by(2).fold(
            #[cfg(feature = "parallel")]
            || [G::ScalarField::ZERO; 4],
            #[cfg(not(feature = "parallel"))]
            [G::ScalarField::ZERO; 4],
            |mut b_t_evals_tmp, r_2_same| {
                // evals
                let beta_b_evals = point_evals_degree_3(
                    &beta_b_dp_array[r_2_same],
                    &beta_b_dp_array[r_2_same + 1],
                );
                let b1_t_evals =
                    point_evals_degree_3(&b1_t_dp_array[r_2_same], &b1_t_dp_array[r_2_same + 1]);
                let b2_t_evals =
                    point_evals_degree_3(&b2_t_dp_array[r_2_same], &b2_t_dp_array[r_2_same + 1]);

                // summing the evals
                b_t_evals_tmp
                    .iter_mut()
                    .zip(beta_b_evals)
                    .zip(b1_t_evals)
                    .zip(b2_t_evals)
                    .for_each(|(((s, beta_b_eval), b1_eval), b2_eval)| {
                        *s += beta_b_eval * b1_eval * b2_eval
                    });
                b_t_evals_tmp
            },
        );

        // in the parallel setting we need to reduce the results of the threads once more
        let mut evals;
        #[cfg(feature = "parallel")]
        {
            evals = b_t_evals.reduce(
                || [G::ScalarField::ZERO; 4],
                |mut b_evals_final, x| {
                    b_evals_final.iter_mut().zip(x).for_each(|(a, b)| *a += b);
                    b_evals_final
                },
            )
        }
        #[cfg(not(feature = "parallel"))]
        {
            evals = b_t_evals;
        }
        // multiply with gamma_a_value for the final result
        evals.iter_mut().for_each(|eval| *eval *= gamma_a_value);
        evals
    }

    fn eval_r_1_same(
        &self,
        beta_a_dp_array: &[G::ScalarField],
        gamma_dp_array: &[G::ScalarField],
        a1_dp_array: &[G::ScalarField],
        a2_dp_array: &[G::ScalarField],
        b_t_value: &G::ScalarField,
    ) -> [G::ScalarField; 4] {
        let r_1_max = beta_a_dp_array.len();
        let r_4_max = 1 << self.r_4_len;

        let gamma_a_evals = ark_std::cfg_into_iter!(0..r_1_max).step_by(2).fold(
            #[cfg(feature = "parallel")]
            || [G::ScalarField::ZERO; 4],
            #[cfg(not(feature = "parallel"))]
            [G::ScalarField::ZERO; 4],
            |mut gamma_a_evals_tmp, r_1| {
                // evals
                let beta_a_evals =
                    point_evals_degree_3(&beta_a_dp_array[r_1], &beta_a_dp_array[r_1 + 1]);
                let gamma_a_evals = (0..r_4_max - 1).fold(
                    [G::ScalarField::ZERO; 4],
                    |mut a_evals_tmp, r_4_same| {
                        // indices
                        let r_4_next = r_4_same + self.mat_idx_shift;
                        let gamma_idx = r_4_same + r_4_next * r_4_max;
                        let inner_a1_idx = r_1 + r_4_next * r_1_max;
                        let inner_a2_idx = r_1 + r_4_same * r_1_max;

                        //evals
                        let a1_evals = point_evals_degree_3(
                            &a1_dp_array[inner_a1_idx],
                            &a1_dp_array[inner_a1_idx + 1],
                        );
                        let a2_evals = point_evals_degree_3(
                            &a2_dp_array[inner_a2_idx],
                            &a2_dp_array[inner_a2_idx + 1],
                        );

                        // summing
                        a_evals_tmp.iter_mut().zip(a1_evals).zip(a2_evals).for_each(
                            |((s, a1_eval), a2_eval)| {
                                *s += gamma_dp_array[gamma_idx] * a1_eval * a2_eval
                            },
                        );
                        a_evals_tmp
                    },
                );

                // summing the evals
                gamma_a_evals_tmp
                    .iter_mut()
                    .zip(beta_a_evals)
                    .zip(gamma_a_evals)
                    .for_each(|((s, beta_a_eval), gamma_a_eval)| *s += beta_a_eval * gamma_a_eval);
                gamma_a_evals_tmp
            },
        );

        // in the parallel setting we need to reduce the results of the threads once more
        let mut evals;
        #[cfg(feature = "parallel")]
        {
            evals = gamma_a_evals.reduce(
                || [G::ScalarField::ZERO; 4],
                |mut gamma_a_evals_final, x| {
                    gamma_a_evals_final
                        .iter_mut()
                        .zip(x)
                        .for_each(|(a, b)| *a += b);
                    gamma_a_evals_final
                },
            )
        }
        #[cfg(not(feature = "parallel"))]
        {
            evals = gamma_a_evals;
        }
        // multiply with b_t_value for the final result
        evals.iter_mut().for_each(|eval| *eval *= b_t_value);
        evals
    }

    fn eval_r_4_same(
        &self,
        gamma_dp_array: &[G::ScalarField],
        a1_dp_array: &[G::ScalarField],
        a2_dp_array: &[G::ScalarField],
        beta_a_b_t_value: &G::ScalarField,
    ) -> [G::ScalarField; 3] {
        let r_4_next_max = 1 << self.r_4_len;
        let r_4_same_max = a2_dp_array.len();
        let r_4_next_fp_max = r_4_next_max / r_4_same_max;

        let gamma_a_evals = ark_std::cfg_into_iter!(0..r_4_same_max).step_by(2).fold(
            #[cfg(feature = "parallel")]
            || [G::ScalarField::ZERO; 3],
            #[cfg(not(feature = "parallel"))]
            [G::ScalarField::ZERO; 3],
            |mut gamma_a_evals_tmp, r_4_same| {
                // evals
                let gamma_a1_evals = (r_4_same * r_4_next_fp_max + 1
                    ..min(
                        (r_4_same + 1) * r_4_next_fp_max + r_4_next_fp_max + 1,
                        r_4_next_max,
                    ))
                    .fold(
                        [G::ScalarField::ZERO; 3],
                        |mut gamma_a1_evals_tmp, r_4_next| {
                            // indices
                            let gamma_idx = r_4_same + r_4_next * r_4_same_max;

                            // evals
                            let gamma_evals = point_evals_degree_2(
                                &gamma_dp_array[gamma_idx],
                                &gamma_dp_array[gamma_idx + 1],
                            );

                            // summing the evals
                            gamma_a1_evals_tmp.iter_mut().zip(gamma_evals).for_each(
                                |(s, gamma_eval)| *s += gamma_eval * a1_dp_array[r_4_next],
                            );
                            gamma_a1_evals_tmp
                        },
                    );
                let a2_evals =
                    point_evals_degree_2(&a2_dp_array[r_4_same], &a2_dp_array[r_4_same + 1]);

                // summing the evals
                gamma_a_evals_tmp
                    .iter_mut()
                    .zip(gamma_a1_evals)
                    .zip(a2_evals)
                    .for_each(|((s, gamma_a1_eval), a2_eval)| *s += gamma_a1_eval * a2_eval);
                gamma_a_evals_tmp
            },
        );

        // in the parallel setting we need to reduce the results of the threads once more
        let mut evals;
        #[cfg(feature = "parallel")]
        {
            evals = gamma_a_evals.reduce(
                || [G::ScalarField::ZERO; 3],
                |mut gamma_a_evals_final, x| {
                    gamma_a_evals_final
                        .iter_mut()
                        .zip(x)
                        .for_each(|(a, b)| *a += b);
                    gamma_a_evals_final
                },
            )
        }
        #[cfg(not(feature = "parallel"))]
        {
            evals = gamma_a_evals;
        }
        // multiply with beta_a_b_t_value for the final result
        evals.iter_mut().for_each(|eval| *eval *= beta_a_b_t_value);
        evals
    }

    fn eval_r_4_next(
        &self,
        gamma_dp_array: &[G::ScalarField],
        a1_dp_array: &[G::ScalarField],
        beta_a_a2_b_t_value: &G::ScalarField,
    ) -> [G::ScalarField; 3] {
        let r_4_next_max = a1_dp_array.len();

        let gamma_a1_evals = ark_std::cfg_into_iter!(0..r_4_next_max).step_by(2).fold(
            #[cfg(feature = "parallel")]
            || [G::ScalarField::ZERO; 3],
            #[cfg(not(feature = "parallel"))]
            [G::ScalarField::ZERO; 3],
            |mut gamma_a1_evals_tmp, r_4_next| {
                // evals
                let gamma_evals =
                    point_evals_degree_2(&gamma_dp_array[r_4_next], &gamma_dp_array[r_4_next + 1]);
                let a1_evals =
                    point_evals_degree_2(&a1_dp_array[r_4_next], &a1_dp_array[r_4_next + 1]);

                // summing the evals
                gamma_a1_evals_tmp
                    .iter_mut()
                    .zip(gamma_evals)
                    .zip(a1_evals)
                    .for_each(|((s, gamma_eval), a1_eval)| *s += gamma_eval * a1_eval);
                gamma_a1_evals_tmp
            },
        );

        // in the parallel setting we need to reduce the results of the threads once more
        let mut evals;
        #[cfg(feature = "parallel")]
        {
            evals = gamma_a1_evals.reduce(
                || [G::ScalarField::ZERO; 3],
                |mut gamma_a1_evals_final, x| {
                    gamma_a1_evals_final
                        .iter_mut()
                        .zip(x)
                        .for_each(|(a, b)| *a += b);
                    gamma_a1_evals_final
                },
            )
        }
        #[cfg(not(feature = "parallel"))]
        {
            evals = gamma_a1_evals;
        }
        // multiply with beta_a_a2_b_t_value for the final result
        evals
            .iter_mut()
            .for_each(|eval| *eval *= beta_a_a2_b_t_value);
        evals
    }
}
