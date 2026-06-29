use crate::client::ClientSchemes;
use crate::prelude::*;
use crate::proof::commitments::LocalTrainingCommitments;
use crate::proof::forward_pass_sumcheck::{
    ForwardPassSumCheckProof, ForwardPassSumCheckProofSeparator,
};
use crate::proof::hyrax_poly_eval::HyraxTargetCommitment;
use crate::proof::scaling::{ScalingProof, ScalingProofSeparator, SignBitsProof};
use crate::proof::{LocalTrainingProof, LocalTrainingProofConstants, ProofWitnessFull};
use spongefish::codecs::arkworks_algebra::{FieldDomainSeparator, GroupDomainSeparator};
use spongefish::{DomainSeparator, DuplexSpongeInterface, ProofResult, ProverState, VerifierState};
use std::slice::from_ref;

pub trait ForwardPassSeparator<G: CurveGroup> {
    fn add_forward_pass_proof<T: Value>(self, constants: &LocalTrainingProofConstants<T>) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> ForwardPassSeparator<G> for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    fn add_forward_pass_proof<T: Value>(self, constants: &LocalTrainingProofConstants<T>) -> Self {
        // a: sumchecks
        self.add_forward_pass_sum_check(
            constants.cm_t_dimensions.rows_log,
            constants.cm_t_dimensions.columns_log,
            constants.cm_c_dimensions.rows_log,
            constants.cm_t_dimensions.matrices_log,
        )
        // b: proof of scaling + sign bits
        .add_scaling(
            constants.cm_t_dimensions.num_row_coms,
            constants.cm_t_dimensions.row_com_dimension,
            constants.cm_t_dimensions.row_com_dimension,
            constants.total_bits,
        )
    }
}

pub struct ForwardPassDeferredEvals<G: CurveGroup> {
    pub w: HyraxTargetCommitment<G>,
    pub c: HyraxTargetCommitment<G>,
    pub t_hat: HyraxTargetCommitment<G>,
    pub t: HyraxTargetCommitment<G>,

    // cloned dp arrays (for efficiency) - prover only
    pub w_dp_array: Option<Vec<G::ScalarField>>, // order: row, column, matrix
    pub c_dp_array: Option<Vec<G::ScalarField>>, // order: row, matrix, column
    pub t_hat_dp_array: Option<Vec<G::ScalarField>>, // order: row, matrix, column
}

impl<'a, G: CurveGroup, T: Value> ProofWitnessFull<'a, G, T> {
    pub fn prove_forward_pass(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<G>,
        commitments: &LocalTrainingCommitments<G>,
    ) -> ProofResult<ForwardPassDeferredEvals<G>> {
        // a: sumchecks
        let ([w, c, t_hat, t], [w_dp_array, c_dp_array, t_hat_dp_array]) =
            ForwardPassSumCheckProof {
                r_1_len: self.constants.cm_t_dimensions.rows_log,
                r_2_len: self.constants.cm_t_dimensions.columns_log,
                r_3_len: self.constants.cm_c_dimensions.rows_log,
                r_4_len: self.constants.cm_t_dimensions.matrices_log,
                zero_step_idx: self.constants.n_layers + 2,
                mat_idx_shift: 1,
                opening_a_with_public_input_matrix: commitments.w_with_public_input_matrix.as_ref(),
                opening_b1_with_public: commitments.opening_c_with_public.as_ref(),
                opening_b2_with_public: commitments.opening_t_hat_with_public.as_ref(),
                opening_c_with_public: commitments.opening_t_with_public.as_ref(),
            }
            .prove(prover_state, client_schemes)?;

        // b: proof of scaling + sign bits
        ScalingProof {
            n_a: self.constants.cm_t_dimensions.row_com_dimension,
            n_b: self.constants.cm_t_dimensions.row_com_dimension,
            a_take: None,
            q: self.constants.total_bits,
            s: self.constants.decimal_bits, // we truncate once
            cm_a_mat_unscaled_all: from_ref(&commitments.cm_t_hat),
            cm_b_mat_unscaled: &commitments.cm_t,
            opening_a_mat_unscaled_all: commitments.opening_t_hat.as_ref().map(|x| from_ref(x)),
            opening_b_mat_unscaled: commitments.opening_t.as_ref(),
            sign_bits_proof: Some(SignBitsProof {
                cm_c_mat: &commitments.cm_c,
                opening_c_mat: commitments.opening_c.as_ref(),
            }),
        }
        .prove(prover_state, client_schemes)?;

        Ok(ForwardPassDeferredEvals {
            w,
            c,
            t_hat,
            t,
            w_dp_array,
            c_dp_array,
            t_hat_dp_array,
        })
    }
}

impl<G: CurveGroup, T: Value> LocalTrainingProof<G, T> {
    pub fn verify_forward_pass<'a>(
        &self,
        verifier_state: &mut VerifierState,
        commitments: &LocalTrainingCommitments<G>,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<ForwardPassDeferredEvals<G>> {
        // a: sumchecks
        let [w, c, t_hat, t] = ForwardPassSumCheckProof {
            r_1_len: self.constants.cm_t_dimensions.rows_log,
            r_2_len: self.constants.cm_t_dimensions.columns_log,
            r_3_len: self.constants.cm_c_dimensions.rows_log,
            r_4_len: self.constants.cm_t_dimensions.matrices_log,
            zero_step_idx: self.constants.n_layers + 2,
            mat_idx_shift: 1,
            opening_a_with_public_input_matrix: None,
            opening_b1_with_public: None,
            opening_b2_with_public: None,
            opening_c_with_public: None,
        }
        .verify(verifier_state, client_schemes)?;

        // b: proof of scaling + sign bits
        ScalingProof {
            n_a: self.constants.cm_t_dimensions.row_com_dimension,
            n_b: self.constants.cm_t_dimensions.row_com_dimension,
            a_take: None,
            q: self.constants.total_bits,
            s: self.constants.decimal_bits, // we truncate once
            cm_a_mat_unscaled_all: from_ref(&commitments.cm_t_hat),
            cm_b_mat_unscaled: &commitments.cm_t,
            opening_a_mat_unscaled_all: None,
            opening_b_mat_unscaled: None,
            sign_bits_proof: Some(SignBitsProof {
                cm_c_mat: &commitments.cm_c,
                opening_c_mat: None,
            }),
        }
        .verify(verifier_state, client_schemes)?;

        Ok(ForwardPassDeferredEvals {
            w,
            c,
            t_hat,
            t,
            w_dp_array: None,
            c_dp_array: None,
            t_hat_dp_array: None,
        })
    }
}
