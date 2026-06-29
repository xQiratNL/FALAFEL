mod backward_pass;
mod backward_pass_sumcheck1;
mod backward_pass_sumcheck2;
mod batch_correctness;
pub mod bench;
mod bulletproof_ipa;
pub mod commitments;
mod dot_product;
mod equality;
mod forward_pass;
mod forward_pass_sumcheck;
mod hyrax_poly_eval;
mod opening;
mod poly_evals;
mod product;
mod scaling;
mod schnorr;
mod sumcheck;
mod util;
pub mod weight_aggregation;
mod weight_update;

use crate::auditor::verify_dataset_signature;
use crate::client::ClientSchemes;
use crate::crypto_schemes::{HyraxCommitment, HyraxCommitmentOpening};
use crate::dataset::Dataset;
use crate::neural_network::FeedForward;
use crate::prelude::*;
use crate::proof::backward_pass::BackwardPassSeparator;
use crate::proof::batch_correctness::BatchCorrectnessSeparator;
use crate::proof::forward_pass::ForwardPassSeparator;
use crate::proof::poly_evals::PolyEvalsSeparator;
use crate::proof::weight_update::WeightUpdateSeparator;
use ark_std::log2;
use ndarray::Array2;
use rand::{CryptoRng, Rng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use spongefish::codecs::arkworks_algebra::{
    FieldDomainSeparator, GroupDomainSeparator, GroupToUnitDeserialize,
};
use spongefish::{DefaultHash, DomainSeparator, DuplexSpongeInterface, ProofResult};
use std::cmp::max;

#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
pub struct HyraxCmMatricesDimensions {
    pub num_row_coms: usize,
    pub row_com_dimension: usize,
    pub total_log: usize,
    matrices_log: usize,
    rows_log: usize,
    columns_log: usize,
}

impl HyraxCmMatricesDimensions {
    pub(crate) fn new(
        n_matrices: usize,
        n_rows_max: usize,
        n_columns_max: usize,
        keep_matrices_separate: bool,
        custom_row_com_dim_log: Option<usize>,
    ) -> Self {
        let matrices_log = log2(n_matrices) as usize;
        let rows_log = log2(n_rows_max) as usize;
        let columns_log = log2(n_columns_max) as usize;
        let total_log = matrices_log + rows_log + columns_log;
        let num_row_coms_log = if let Some(custom_row_com_dim_log) = custom_row_com_dim_log {
            total_log - custom_row_com_dim_log
        } else {
            if keep_matrices_separate {
                max(total_log / 2, matrices_log)
            } else {
                total_log / 2
            }
        };
        Self {
            num_row_coms: 1 << num_row_coms_log,
            row_com_dimension: 1 << (total_log - num_row_coms_log),
            total_log,
            matrices_log,
            rows_log,
            columns_log,
        }
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(bound(deserialize = "T: Value"))]
pub struct LocalTrainingProofConstants<T: Value> {
    learning_rate: T,
    total_bits: usize,
    decimal_bits: usize,
    batch_size: usize,
    n_features: usize,
    n_samples: usize,
    n_batches: usize,
    n_layers: usize,
    n_epochs: usize,
    n_hidden: Vec<usize>,
    n_labels: usize,
    cm_batch_samples_dimensions: HyraxCmMatricesDimensions,
    cm_batch_labels_dimensions: HyraxCmMatricesDimensions,
    cm_u_dimensions: HyraxCmMatricesDimensions,
    pub cm_w_single_iteration_dimensions: HyraxCmMatricesDimensions,
    cm_g_hat_single_iteration_dimensions: HyraxCmMatricesDimensions,
    cm_w_full_dimensions: HyraxCmMatricesDimensions,
    cm_c_dimensions: HyraxCmMatricesDimensions,
    cm_t_dimensions: HyraxCmMatricesDimensions,
    cm_g_bar_dimensions: HyraxCmMatricesDimensions,
    cm_r_dimensions: HyraxCmMatricesDimensions,
    cm_r_final_dimensions: HyraxCmMatricesDimensions,
}

impl<T: Value> LocalTrainingProofConstants<T> {
    pub fn new(
        total_bits: usize,
        decimal_bits: usize,
        n_local_iter: usize,
        mut batch_size: usize,
        n_features: usize,
        n_samples: usize,
        n_hidden: &[usize],
        n_labels: usize,
        nn: &FeedForward<T>,
    ) -> Self {
        // batch matrices
        let n_batches = (n_samples - 1) / batch_size + 1;
        batch_size = if n_batches == 1 {
            n_samples
        } else {
            batch_size
        };
        // W matrices (and those with the same dimensions)
        let w_rows = std::iter::once(n_features + 1)
            .chain(
                nn.layers[0..nn.layers.len() - 1]
                    .iter()
                    .map(|layer| layer.weights.nrows() + 1),
            )
            .chain(std::iter::once(nn.layers.last().unwrap().weights.nrows()))
            .max()
            .unwrap();
        let w_cols = std::iter::once(1)
            .chain(nn.layers.iter().map(|layer| layer.weights.ncols()))
            .chain(std::iter::once(nn.layers.last().unwrap().weights.nrows()))
            .max()
            .unwrap();
        // T matrices (and those with the same dimensions)
        let t_rows = w_cols;
        let t_cols = batch_size;

        // other stuff
        let n_epochs = n_local_iter * n_batches;
        let n_layers = nn.layers.len();
        let n_matrices_per_layer = 2 + n_layers;
        let n_matrices = n_epochs * n_matrices_per_layer;

        // actual struct
        let cm_g_hat_single_iteration_dimensions =
            HyraxCmMatricesDimensions::new(n_matrices_per_layer, w_rows, w_cols, true, None);
        Self {
            learning_rate: nn.learning_rate,
            total_bits,
            decimal_bits,
            batch_size,
            n_features,
            n_samples,
            n_batches,
            n_layers,
            n_epochs,
            n_hidden: n_hidden.to_vec(),
            n_labels,
            cm_batch_samples_dimensions: HyraxCmMatricesDimensions::new(
                n_batches,
                n_features + 1,
                batch_size,
                true,
                None,
            ),
            cm_batch_labels_dimensions: HyraxCmMatricesDimensions::new(
                n_batches, n_labels, batch_size, true, None,
            ),
            cm_u_dimensions: HyraxCmMatricesDimensions::new(n_epochs, n_labels, t_cols, true, None),
            cm_w_single_iteration_dimensions: HyraxCmMatricesDimensions::new(
                n_matrices_per_layer,
                w_rows,
                w_cols,
                true,
                None,
            ),
            cm_g_hat_single_iteration_dimensions,
            cm_w_full_dimensions: HyraxCmMatricesDimensions::new(
                n_matrices, w_rows, w_cols, false, None,
            ),
            cm_c_dimensions: HyraxCmMatricesDimensions::new(
                n_matrices, t_rows, t_cols, false, None,
            ),
            cm_t_dimensions: HyraxCmMatricesDimensions::new(
                n_matrices, t_rows, t_cols, false, None,
            ),
            cm_g_bar_dimensions: HyraxCmMatricesDimensions::new(
                n_matrices,
                w_rows,
                w_cols,
                false,
                Some(log2(cm_g_hat_single_iteration_dimensions.row_com_dimension) as usize),
            ),
            cm_r_dimensions: HyraxCmMatricesDimensions::new(
                n_matrices, t_rows, t_cols, false, None,
            ),
            cm_r_final_dimensions: HyraxCmMatricesDimensions::new(
                n_epochs, n_labels, t_cols, true, None,
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LocalTrainingProof<G: CurveGroup, T: Value> {
    pub dataset_signature: AuditorSignature<G>,
    pub proof: Vec<u8>,
    pub constants: LocalTrainingProofConstants<T>,
}

pub struct AggregationProof {
    pub proof: Vec<u8>,
}

// note: the input matrix is the first entry in the post_activations_vector
pub struct ProofWitnessStep<T: Value> {
    pub weights_after: Vec<Array2<T>>,
    pub post_activations_vector: Vec<Array2<T>>,
    pub pre_activations_vector: Vec<Array2<T>>,
    pub layer_gradients: Vec<Array2<T>>,
    pub intermediate_gradients: Vec<Array2<T>>,
}

// this should be all the witness values we use in the proofs, so that everything is easy to access
// note: this also includes some constants for ease of access
pub struct ProofWitnessFull<'a, G: CurveGroup, T: Value> {
    // public values
    pub cm_dataset_samples: &'a G::Affine,
    pub cm_dataset_labels: &'a G::Affine,

    // witnesses
    pub dataset_shuffled: Dataset<T>,
    pub initial_weights: &'a [Array2<T>],
    pub opening_initial_weights: HyraxCommitmentOpening<G>,
    pub feed_forward_steps: &'a [ProofWitnessStep<T>],
    pub r_dataset_samples: &'a G::ScalarField,
    pub r_dataset_labels: &'a G::ScalarField,
    pub dataset_scalar_samples_permuted: &'a [G::ScalarField],
    pub dataset_scalar_labels_permuted: &'a [G::ScalarField],
    pub permutation: &'a [usize],

    // constants
    pub constants: LocalTrainingProofConstants<T>,
}

pub trait LocalTrainingProofSeparator<G: CurveGroup, T: Value> {
    fn new_local_training_proof_separator(
        iv: &str,
        constants: &LocalTrainingProofConstants<T>,
    ) -> Self;
    fn add_statements(self, constants: &LocalTrainingProofConstants<T>) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface, T: Value> LocalTrainingProofSeparator<G, T>
    for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    fn new_local_training_proof_separator(
        iv: &str,
        constants: &LocalTrainingProofConstants<T>,
    ) -> Self {
        Self::new(iv)
            .add_statements(constants)
            .ratchet()
            .add_batch_proof(constants)
            .ratchet()
            .add_forward_pass_proof(constants)
            .ratchet()
            .add_backward_pass_proof(constants)
            .ratchet()
            .add_weight_update_proof(constants)
            .ratchet()
            .add_poly_eval_proofs(constants)
    }

    // we add all the commitments at once here, as some are used in multiple places, this ensures that all works correctly.
    fn add_statements(self, constants: &LocalTrainingProofConstants<T>) -> Self {
        self.add_points(2, "full dataset commitment (samples + labels)")
            .add_points(
                constants.cm_batch_samples_dimensions.num_row_coms,
                "batch samples commitments",
            )
            .add_points(
                constants.cm_batch_labels_dimensions.num_row_coms,
                "batch labels commitments",
            )
            .add_points(constants.cm_u_dimensions.num_row_coms, "U matrices")
            .add_points(constants.cm_c_dimensions.num_row_coms, "C matrices")
            .add_points(constants.cm_t_dimensions.num_row_coms, "T_hat matrices")
            .add_points(constants.cm_t_dimensions.num_row_coms, "T matrices")
            .add_points(constants.cm_g_bar_dimensions.num_row_coms, "G_bar matrices")
            .add_points(constants.cm_r_dimensions.num_row_coms, "R_hat matrices")
            .add_points(
                constants.cm_r_final_dimensions.num_row_coms,
                "R_hat_final matrices",
            )
            .add_points(constants.cm_r_dimensions.num_row_coms, "R matrices")
            .add_points(
                constants.n_epochs * constants.cm_g_hat_single_iteration_dimensions.num_row_coms,
                "G_hat matrices",
            )
    }
}

//This separator is for the server to simply extract the commitments from a local training proof of a client. The server does not need to verify the entire proof.
pub trait LocalTrainingProofCommitmentsSeparator<G: CurveGroup> {
    fn new_local_training_proof_commitments_separator<T: Value>(
        iv: &str,
        constants: &LocalTrainingProofConstants<T>,
    ) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> LocalTrainingProofCommitmentsSeparator<G>
    for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    fn new_local_training_proof_commitments_separator<T: Value>(
        iv: &str,
        constants: &LocalTrainingProofConstants<T>,
    ) -> Self {
        Self::new(iv).add_statements(constants)
    }
}

impl<'a, G: CurveGroup, T: Value> ProofWitnessFull<'a, G, T> {
    // based on algorithm 5 in the paper
    pub fn prove(
        self,
        iv: &str,
        dataset_signature: AuditorSignature<G>,
        client_schemes: &ClientSchemes<G>,
        rng: &mut (impl Rng + CryptoRng),
    ) -> ProofResult<(LocalTrainingProof<G, T>, HyraxCommitmentOpening<G>)> {
        // initialize domain separator + prover state
        let domain_separator: DomainSeparator<DefaultHash> =
            LocalTrainingProofSeparator::<G, T>::new_local_training_proof_separator(
                iv,
                &self.constants,
            );
        let mut prover_state = domain_separator.to_prover_state();

        // commitments: all commitments should be made here so we can distribute them efficiently;
        let commitments = self.compute_commitments(client_schemes, rng);

        // add statement (i.e., the commitments)
        self.add_commitments(&mut prover_state, &commitments)?;
        prover_state.ratchet()?;
        //compute the actual proofs
        self.prove_batch_correctness(&mut prover_state, client_schemes, &commitments)?;
        prover_state.ratchet()?;
        let deferred_proofs_fp =
            self.prove_forward_pass(&mut prover_state, client_schemes, &commitments)?;
        prover_state.ratchet()?;
        let deferred_proofs_bp =
            self.prove_backward_pass(&mut prover_state, client_schemes, &commitments)?;
        prover_state.ratchet()?;
        self.prove_weight_update(&mut prover_state, client_schemes, &commitments)?;
        prover_state.ratchet()?;
        self.prove_poly_evals(
            &mut prover_state,
            client_schemes,
            &commitments,
            deferred_proofs_fp,
            deferred_proofs_bp,
        )?;
        Ok((
            LocalTrainingProof {
                dataset_signature,
                proof: prover_state.narg_string().to_vec(),
                constants: self.constants,
            },
            HyraxCommitmentOpening {
                input_matrix: commitments
                    .w_final_input_matrix
                    .expect("The prover should have the witness!"),
                randomness: ark_std::cfg_into_iter!(
                    (0..self.opening_initial_weights.randomness.len())
                )
                .map(|idx| {
                    self.opening_initial_weights.randomness[idx]
                        - ark_std::cfg_iter!(
                            commitments
                                .opening_g_hat_all
                                .as_ref()
                                .expect("This should exist")
                        )
                        .map(|opening_g_hat| opening_g_hat.randomness[idx])
                        .sum::<G::ScalarField>()
                })
                .collect::<Vec<_>>(),
            },
        ))
    }
}

impl<G: CurveGroup, T: Value> LocalTrainingProof<G, T> {
    pub fn verify<'a>(
        &self,
        iv: &str,
        pp_signature: &AuditorSignatureParameters<G>,
        auditor_pk: &AuditorSignaturePublicKey<G>,
        cm_initial_weights: &HyraxCommitment<G>,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        // note: we assume the IV is correctly generated by the verifier and not be the cheater
        // initialize domain separator + verifier state
        let domain_separator: DomainSeparator<DefaultHash> =
            LocalTrainingProofSeparator::<G, T>::new_local_training_proof_separator(
                iv,
                &self.constants,
            );
        let mut verifier_state = domain_separator.to_verifier_state(&self.proof);

        let [cm_dataset_samples, cm_dataset_labels]: [_; 2] =
            verifier_state.next_points()?.map(|cm: G| cm.into_affine());

        // verify signature on dataset commitment from auditor
        if !verify_dataset_signature(
            pp_signature,
            auditor_pk,
            &(cm_dataset_samples, cm_dataset_labels),
            &self.dataset_signature,
        )
        .unwrap_or(false)
        {
            return Err(spongefish::ProofError::InvalidProof);
        }

        // get all the commitments for the ZKPs
        let commitments = self.obtain_commitments(&mut verifier_state)?;
        verifier_state.ratchet()?;

        // verify actual ZKPs
        self.verify_batch_correctness(
            &mut verifier_state,
            &commitments,
            client_schemes,
            &cm_dataset_samples,
            &cm_dataset_labels,
        )?;
        verifier_state.ratchet()?;
        let deferred_proofs_fp =
            self.verify_forward_pass(&mut verifier_state, &commitments, client_schemes)?;
        verifier_state.ratchet()?;
        let deferred_proofs_bp =
            self.verify_backward_pass(&mut verifier_state, &commitments, client_schemes)?;
        verifier_state.ratchet()?;
        self.verify_weight_update(&mut verifier_state, &commitments, client_schemes)?;
        verifier_state.ratchet()?;
        self.verify_poly_evals(
            &mut verifier_state,
            &commitments,
            &cm_initial_weights,
            client_schemes,
            deferred_proofs_fp,
            deferred_proofs_bp,
        )?;

        Ok(())
    }
}
