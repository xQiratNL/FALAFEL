use crate::client::ClientSchemes;
use crate::crypto_schemes::{HyraxCommitment, HyraxCommitmentOpening};
use crate::prelude::*;
use crate::proof::commitments::LocalTrainingCommitments;
use crate::proof::scaling::{ScalingProof, ScalingProofSeparator};
use crate::proof::{LocalTrainingProof, LocalTrainingProofConstants, ProofWitnessFull};
use crate::util::IntoField;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::codecs::arkworks_algebra::{FieldDomainSeparator, GroupDomainSeparator};
use spongefish::{DomainSeparator, DuplexSpongeInterface, ProofResult, ProverState, VerifierState};
use std::ops::Mul;

pub trait WeightUpdateSeparator<G: CurveGroup> {
    fn add_weight_update_proof<T: Value>(self, constants: &LocalTrainingProofConstants<T>) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> WeightUpdateSeparator<G> for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    fn add_weight_update_proof<T: Value>(self, constants: &LocalTrainingProofConstants<T>) -> Self {
        self
            // a
            .add_scaling(
                constants.cm_g_bar_dimensions.num_row_coms,
                constants
                    .cm_g_hat_single_iteration_dimensions
                    .row_com_dimension,
                constants.cm_g_bar_dimensions.row_com_dimension,
                constants.total_bits,
            )
        // b: done implicitly
    }
}

impl<'a, G: CurveGroup, T: Value> ProofWitnessFull<'a, G, T> {
    pub fn prove_weight_update(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<G>,
        commitments: &LocalTrainingCommitments<G>,
    ) -> ProofResult<()> {
        // a: proof of scaling
        let learning_rate_as_field = self.constants.learning_rate.to_field::<G::ScalarField>();
        let cm_g_bar_times_learning_parameter = HyraxCommitment {
            row_coms: G::normalize_batch(
                &ark_std::cfg_iter!(commitments.cm_g_bar.row_coms)
                    .map(|row_com| row_com.mul(learning_rate_as_field))
                    .collect::<Vec<_>>(),
            ),
        };
        let opening_g_bar = commitments
            .opening_g_bar
            .as_ref()
            .expect("This should exist");
        let opening_g_bar_times_learning_parameter = HyraxCommitmentOpening {
            input_matrix: ark_std::cfg_iter!(opening_g_bar.input_matrix)
                .map(|row| {
                    ark_std::cfg_iter!(row)
                        .map(|value| learning_rate_as_field * value)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
            randomness: ark_std::cfg_iter!(opening_g_bar.randomness)
                .map(|r| learning_rate_as_field * r)
                .collect::<Vec<_>>(),
        };
        ScalingProof {
            n_a: self
                .constants
                .cm_g_hat_single_iteration_dimensions
                .row_com_dimension,
            n_b: self.constants.cm_g_bar_dimensions.row_com_dimension,
            a_take: Some(self.constants.n_layers + 2),
            q: self.constants.total_bits,
            s: self.constants.decimal_bits * 2, // we truncate twice
            cm_a_mat_unscaled_all: &commitments.cm_g_hat_all,
            cm_b_mat_unscaled: &cm_g_bar_times_learning_parameter,
            opening_a_mat_unscaled_all: commitments
                .opening_g_hat_all
                .as_ref()
                .map(|x| x.as_slice()),
            opening_b_mat_unscaled: Some(&opening_g_bar_times_learning_parameter),
            sign_bits_proof: None,
        }
        .prove(prover_state, client_schemes)?;

        // b: weight update
        // this is already done implicitly

        Ok(())
    }
}

impl<G: CurveGroup, T: Value> LocalTrainingProof<G, T> {
    pub fn verify_weight_update<'a>(
        &self,
        verifier_state: &mut VerifierState,
        commitments: &LocalTrainingCommitments<G>,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        // a: proof of scaling
        let learning_rate_as_field = self.constants.learning_rate.to_field::<G::ScalarField>();
        let cm_g_bar_times_learning_parameter = HyraxCommitment {
            row_coms: G::normalize_batch(
                &ark_std::cfg_iter!(commitments.cm_g_bar.row_coms)
                    .map(|row_com| row_com.mul(learning_rate_as_field))
                    .collect::<Vec<_>>(),
            ),
        };
        ScalingProof {
            n_a: self
                .constants
                .cm_g_hat_single_iteration_dimensions
                .row_com_dimension,
            n_b: self.constants.cm_g_bar_dimensions.row_com_dimension,
            a_take: Some(self.constants.n_layers + 2),
            q: self.constants.total_bits,
            s: self.constants.decimal_bits * 2, // we truncate twice
            cm_a_mat_unscaled_all: &commitments.cm_g_hat_all,
            cm_b_mat_unscaled: &cm_g_bar_times_learning_parameter,
            opening_a_mat_unscaled_all: None,
            opening_b_mat_unscaled: None,
            sign_bits_proof: None,
        }
        .verify(verifier_state, client_schemes)?;

        // b: weight update
        // this is already done implicitly

        Ok(())
    }
}
