use crate::client::ClientSchemes;
use crate::crypto_schemes::{HyraxCommitment, HyraxCommitmentOpening};
use crate::prelude::*;
use crate::proof::backward_pass_sumcheck1::{
    BackwardPassSumCheck1Proof, BackwardPassSumCheck1ProofSeparator,
};
use crate::proof::backward_pass_sumcheck2::{
    BackwardPassSumCheck2Proof, BackwardPassSumCheck2ProofSeparator,
};
use crate::proof::commitments::LocalTrainingCommitments;
use crate::proof::hyrax_poly_eval::HyraxTargetCommitment;
use crate::proof::scaling::{ScalingProof, ScalingProofSeparator};
use crate::proof::{LocalTrainingProof, LocalTrainingProofConstants, ProofWitnessFull};
use crate::util::IntoField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::codecs::arkworks_algebra::{FieldDomainSeparator, GroupDomainSeparator};
use spongefish::{DomainSeparator, DuplexSpongeInterface, ProofResult, ProverState, VerifierState};
use std::slice::from_ref;

pub trait BackwardPassSeparator<G: CurveGroup> {
    fn add_backward_pass_proof<T: Value>(self, constants: &LocalTrainingProofConstants<T>) -> Self;
}

pub struct BackwardPassDeferredEvals<G: CurveGroup> {
    pub r_hat_one: HyraxTargetCommitment<G>,
    pub c_one_one: HyraxTargetCommitment<G>,
    pub c_one_two: HyraxTargetCommitment<G>,
    pub t_hat: HyraxTargetCommitment<G>,
    pub g_bar: HyraxTargetCommitment<G>,
    pub w: HyraxTargetCommitment<G>,
    pub r_hat_two: HyraxTargetCommitment<G>,
    pub c_two: HyraxTargetCommitment<G>,
    pub r: HyraxTargetCommitment<G>,

    // cloned dp arrays (for efficiency) - prover only
    pub r_hat_dp_array: Option<Vec<G::ScalarField>>, // order: column, row, matrix
}

impl<G: CurveGroup, H: DuplexSpongeInterface> BackwardPassSeparator<G> for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    fn add_backward_pass_proof<T: Value>(self, constants: &LocalTrainingProofConstants<T>) -> Self {
        self
            // a
            .add_backward_pass_sum_check1(
                constants.cm_r_dimensions.rows_log,
                constants.cm_c_dimensions.rows_log,
                constants.cm_r_dimensions.columns_log,
                constants.cm_r_dimensions.matrices_log,
            )
            // b
            .add_backward_pass_sum_check2(
                constants.cm_r_dimensions.rows_log,
                constants.cm_r_dimensions.columns_log,
                constants.cm_r_dimensions.rows_log,
                constants.cm_r_dimensions.matrices_log,
            )
            // c
            .add_scaling(
                constants.cm_r_dimensions.num_row_coms,
                constants.cm_r_dimensions.row_com_dimension,
                constants.cm_r_dimensions.row_com_dimension,
                constants.total_bits,
            )
            // d
            .add_scaling(
                constants.cm_r_final_dimensions.num_row_coms,
                constants.cm_r_final_dimensions.row_com_dimension,
                constants.cm_r_final_dimensions.row_com_dimension,
                constants.total_bits,
            )
    }
}

impl<'a, G: CurveGroup, T: Value> ProofWitnessFull<'a, G, T> {
    pub fn prove_backward_pass(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<G>,
        commitments: &LocalTrainingCommitments<G>,
    ) -> ProofResult<BackwardPassDeferredEvals<G>> {
        // a: proof of sumcheck for layer gradient
        let (
            [r_hat_one, c_one_one, c_one_two, t_hat, g_bar],
            [r_hat_dp_array, _c_dp_array, _c_dp_array_2, _t_hat_dp_array],
        ) = BackwardPassSumCheck1Proof {
            r_1_len: self.constants.cm_r_dimensions.rows_log,
            r_2_len: self.constants.cm_c_dimensions.rows_log,
            r_3_len: self.constants.cm_r_dimensions.columns_log,
            r_4_len: self.constants.cm_r_dimensions.matrices_log,
            zero_step_idx: self.constants.n_layers + 2,
            mat_idx_shift: 1,
            opening_a1_with_public: commitments.opening_r_hat_with_public.as_ref(),
            opening_a2_with_public: commitments.opening_c_with_public.as_ref(),
            opening_b1_with_public: commitments.opening_c_with_public.as_ref(),
            opening_b2_with_public: commitments.opening_t_hat_with_public.as_ref(),
            opening_c_with_public: commitments.opening_g_bar.as_ref(),
        }
        .prove(prover_state, client_schemes)?;

        // b: proof of sumcheck for intermediate gradient
        let [w, r_hat_two, c_two, r] = BackwardPassSumCheck2Proof {
            r_1_len: self.constants.cm_r_dimensions.rows_log,
            r_2_len: self.constants.cm_r_dimensions.columns_log,
            r_3_len: self.constants.cm_r_dimensions.rows_log,
            r_4_len: self.constants.cm_r_dimensions.matrices_log,
            zero_step_idx: self.constants.n_layers + 2,
            mat_idx_shift: 1,
            opening_a_with_public: commitments.w_with_public_input_matrix.as_ref(),
            opening_b1_with_public: commitments.opening_r_hat_with_public.as_ref(),
            opening_b2_with_public: commitments.opening_c_with_public.as_ref(),
            opening_c_with_public: commitments.opening_r.as_ref(),
        }
        .prove(prover_state, client_schemes)?;

        // c: proof of scaling for intermediate gradient
        ScalingProof {
            n_a: self.constants.cm_r_dimensions.row_com_dimension,
            n_b: self.constants.cm_r_dimensions.row_com_dimension,
            a_take: None,
            q: self.constants.total_bits,
            s: self.constants.decimal_bits, // we truncate once
            cm_a_mat_unscaled_all: from_ref(&commitments.cm_r_hat),
            cm_b_mat_unscaled: &commitments.cm_r,
            opening_a_mat_unscaled_all: commitments.opening_r_hat.as_ref().map(|x| from_ref(x)),
            opening_b_mat_unscaled: commitments.opening_r.as_ref(),
            sign_bits_proof: None,
        }
        .prove(prover_state, client_schemes)?;

        // d: proof of scaling for r_hat_final
        let double_batch_size_as_field = T::from_f64(
            2. * (1 << self.constants.decimal_bits) as f64 / self.constants.batch_size as f64,
        )
        .unwrap()
        .to_field::<G::ScalarField>();
        let double_final_batch_size_as_field = T::from_f64(
            2. * (1 << self.constants.decimal_bits) as f64
                / (self.constants.n_samples % self.constants.batch_size) as f64,
        )
        .unwrap_or(
            T::from_f64(
                2. * (1 << self.constants.decimal_bits) as f64 / self.constants.batch_size as f64,
            )
            .unwrap(),
        )
        .to_field::<G::ScalarField>();
        let cm_r_final = HyraxCommitment {
            row_coms: G::normalize_batch(
                &ark_std::cfg_chunks!(
                    commitments.cm_u.row_coms,
                    self.constants.cm_batch_labels_dimensions.num_row_coms
                )
                .flat_map(|u_row_coms| {
                    ark_std::cfg_iter!(u_row_coms)
                        .zip(ark_std::cfg_iter!(commitments.cm_batch_labels.row_coms))
                        .map(|(&row_com_u, row_com_y)| {
                            (row_com_u - row_com_y).mul(double_batch_size_as_field)
                        })
                })
                .collect::<Vec<_>>(),
            ),
        };
        let opening_u = commitments.opening_u.as_ref().expect("This should exist");
        let opening_batch_labels = commitments
            .opening_batch_labels
            .as_ref()
            .expect("This should exist");
        let opening_r_final = HyraxCommitmentOpening {
            input_matrix: ark_std::cfg_chunks!(
                opening_u.input_matrix,
                self.constants.cm_batch_labels_dimensions.num_row_coms
            )
            .flat_map(|u_rows| {
                ark_std::cfg_iter!(u_rows)
                    .zip(ark_std::cfg_iter!(opening_batch_labels.input_matrix))
                    .enumerate()
                    .map(|(idx, (u_row, y_row))| {
                        let multiplier = if (idx + 1) % self.constants.n_batches == 0 {
                            double_final_batch_size_as_field
                        } else {
                            double_batch_size_as_field
                        };
                        ark_std::cfg_iter!(u_row)
                            .zip(ark_std::cfg_iter!(y_row))
                            .map(|(&u_val, y_val)| (u_val - y_val) * multiplier)
                            .collect::<Vec<_>>()
                    })
            })
            .collect::<Vec<_>>(),
            randomness: ark_std::cfg_chunks!(
                opening_u.randomness,
                self.constants.cm_batch_labels_dimensions.num_row_coms
            )
            .flat_map(|u_rs| {
                ark_std::cfg_iter!(u_rs)
                    .zip(ark_std::cfg_iter!(opening_batch_labels.randomness))
                    .enumerate()
                    .map(|(idx, (&u_r, y_r))| {
                        let multiplier = if (idx + 1) % self.constants.n_batches == 0 {
                            double_final_batch_size_as_field
                        } else {
                            double_batch_size_as_field
                        };
                        (u_r - y_r) * multiplier
                    })
            })
            .collect::<Vec<_>>(),
        };

        ScalingProof {
            n_a: self.constants.cm_r_final_dimensions.row_com_dimension,
            n_b: self.constants.cm_u_dimensions.row_com_dimension,
            a_take: None,
            q: self.constants.total_bits,
            s: self.constants.decimal_bits, // we truncate once
            cm_a_mat_unscaled_all: from_ref(&commitments.cm_r_hat_final),
            cm_b_mat_unscaled: &cm_r_final,
            opening_a_mat_unscaled_all: commitments
                .opening_r_hat_final
                .as_ref()
                .map(|x| from_ref(x)),
            opening_b_mat_unscaled: Some(&opening_r_final),
            sign_bits_proof: None,
        }
        .prove(prover_state, client_schemes)?;

        Ok(BackwardPassDeferredEvals {
            r_hat_one,
            c_one_one,
            c_one_two,
            t_hat,
            g_bar,
            w,
            r_hat_two,
            c_two,
            r,
            r_hat_dp_array,
        })
    }
}

impl<G: CurveGroup, T: Value> LocalTrainingProof<G, T> {
    pub fn verify_backward_pass<'a>(
        &self,
        verifier_state: &mut VerifierState,
        commitments: &LocalTrainingCommitments<G>,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<BackwardPassDeferredEvals<G>> {
        // a: proof of sumcheck for layer gradient
        let [r_hat_one, c_one_one, c_one_two, t_hat, g_bar] = BackwardPassSumCheck1Proof {
            r_1_len: self.constants.cm_r_dimensions.rows_log,
            r_2_len: self.constants.cm_c_dimensions.rows_log,
            r_3_len: self.constants.cm_r_dimensions.columns_log,
            r_4_len: self.constants.cm_r_dimensions.matrices_log,
            zero_step_idx: self.constants.n_layers + 2,
            mat_idx_shift: 1,
            opening_a1_with_public: None,
            opening_a2_with_public: None,
            opening_b1_with_public: None,
            opening_b2_with_public: None,
            opening_c_with_public: None,
        }
        .verify(verifier_state, client_schemes)?;

        // b: proof of sumcheck for intermediate gradient
        let [w, r_hat_two, c_two, r] = BackwardPassSumCheck2Proof {
            r_1_len: self.constants.cm_r_dimensions.rows_log,
            r_2_len: self.constants.cm_r_dimensions.columns_log,
            r_3_len: self.constants.cm_r_dimensions.rows_log,
            r_4_len: self.constants.cm_r_dimensions.matrices_log,
            zero_step_idx: self.constants.n_layers + 2,
            mat_idx_shift: 1,
            opening_a_with_public: None,
            opening_b1_with_public: None,
            opening_b2_with_public: None,
            opening_c_with_public: None,
        }
        .verify(verifier_state, client_schemes)?;

        // c: proof of scaling for intermediate gradient
        ScalingProof {
            n_a: self.constants.cm_r_dimensions.row_com_dimension,
            n_b: self.constants.cm_r_dimensions.row_com_dimension,
            a_take: None,
            q: self.constants.total_bits,
            s: self.constants.decimal_bits, // we truncate once
            cm_a_mat_unscaled_all: from_ref(&commitments.cm_r_hat),
            cm_b_mat_unscaled: &commitments.cm_r,
            opening_a_mat_unscaled_all: None,
            opening_b_mat_unscaled: None,
            sign_bits_proof: None,
        }
        .verify(verifier_state, client_schemes)?;

        // d: proof of scaling for r_hat_final
        let double_batch_size_as_field = T::from_f64(
            2. * (1 << self.constants.decimal_bits) as f64 / self.constants.batch_size as f64,
        )
        .unwrap()
        .to_field::<G::ScalarField>();
        let double_final_batch_size_as_field = T::from_f64(
            2. * (1 << self.constants.decimal_bits) as f64
                / (self.constants.n_samples % self.constants.batch_size) as f64,
        )
        .unwrap_or(
            T::from_f64(
                2. * (1 << self.constants.decimal_bits) as f64 / self.constants.batch_size as f64,
            )
            .unwrap(),
        )
        .to_field::<G::ScalarField>();
        let cm_r_final = HyraxCommitment {
            row_coms: G::normalize_batch(
                &ark_std::cfg_chunks!(
                    commitments.cm_u.row_coms,
                    self.constants.cm_batch_labels_dimensions.num_row_coms
                )
                .flat_map(|u_row_coms| {
                    ark_std::cfg_iter!(u_row_coms)
                        .zip(ark_std::cfg_iter!(commitments.cm_batch_labels.row_coms))
                        .enumerate()
                        .map(|(idx, (&row_com_u, row_com_y))| {
                            let multiplier = if (idx + 1) % self.constants.n_batches == 0 {
                                double_final_batch_size_as_field
                            } else {
                                double_batch_size_as_field
                            };
                            (row_com_u - row_com_y).mul(multiplier)
                        })
                })
                .collect::<Vec<_>>(),
            ),
        };

        ScalingProof {
            n_a: self.constants.cm_r_final_dimensions.row_com_dimension,
            n_b: self.constants.cm_u_dimensions.row_com_dimension,
            a_take: None,
            q: self.constants.total_bits,
            s: self.constants.decimal_bits, // we truncate once
            cm_a_mat_unscaled_all: from_ref(&commitments.cm_r_hat_final),
            cm_b_mat_unscaled: &cm_r_final,
            opening_a_mat_unscaled_all: None,
            opening_b_mat_unscaled: None,
            sign_bits_proof: None,
        }
        .verify(verifier_state, client_schemes)?;

        Ok(BackwardPassDeferredEvals {
            r_hat_one,
            c_one_one,
            c_one_two,
            t_hat,
            g_bar,
            w,
            r_hat_two,
            c_two,
            r,
            r_hat_dp_array: None,
        })
    }
}
