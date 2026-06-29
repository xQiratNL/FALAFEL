//! Server structure, this is the main place where all server side functionality is tied together.

use crate::auditor::Auditor;
use crate::client::ClientSchemes;
use crate::crypto_schemes::{Generators, HyraxCommitment, HyraxCommitmentOpening};
use crate::message::{
    HyraxCommitmentOpeningSerializable, InitializeMessage, LocalUpdateMessage,
    LocalUpdateMessageSerializable,
};
use crate::neural_network::{ActivationFunction, FeedForward};
use crate::prelude::*;
use crate::proof::commitments::{LocalTrainingCommitments, create_w_initial_matrices};
use crate::proof::{
    AggregationProof, HyraxCmMatricesDimensions, LocalTrainingProof,
    LocalTrainingProofCommitmentsSeparator, LocalTrainingProofConstants,
};
use crate::util::get_iv;
use ark_ff::AdditiveGroup;
use ark_std::UniformRand;
use ndarray::Array2;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::codecs::arkworks_algebra::GroupToUnitDeserialize;
use spongefish::{DefaultHash, DomainSeparator, ProofResult};
use std::ops::Mul;

pub struct Server<'a, G: CurveGroup, T: Value> {
    global_iter: usize,
    client_datasizes: Vec<T>,
    commitment_dimensions: &'a HyraxCmMatricesDimensions,
    constants: LocalTrainingProofConstants<T>,
    client_schemes: ClientSchemes<'a, G>,
    auditor: &'a Auditor<'a, G>,
    neural_network: Option<FeedForward<T>>,
    local_proofs_iterations: Vec<
        Option<(
            LocalUpdateMessage<G, T>,
            Vec<Array2<T>>,
            HyraxCommitmentOpening<G>,
        )>,
    >,
    zero_commitments: Option<(Vec<G::Affine>, Vec<G::Scalar>)>,
    rng: ChaChaRng,
    activation_functions: Vec<ActivationFunction>,
    learning_rate: T,
}

impl<'a, G: CurveGroup, T: Value> Server<'a, G, T> {
    pub fn new(
        client_datasizes: &'a Vec<usize>,
        commitment_dimensions: &'a HyraxCmMatricesDimensions,
        constants: LocalTrainingProofConstants<T>,
        generators: &'a Generators<G>,
        auditor: &'a Auditor<'a, G>,
        activation_functions: Vec<ActivationFunction>,
        learning_rate: T,
    ) -> Self {
        let mut server = Self {
            global_iter: 0,
            client_datasizes: client_datasizes
                .iter()
                .map(|s| T::from_usize(*s).unwrap())
                .collect::<Vec<T>>(),
            commitment_dimensions,
            constants,
            client_schemes: ClientSchemes::setup(generators),
            auditor,
            neural_network: None,
            local_proofs_iterations: vec![None; client_datasizes.len()],
            zero_commitments: None,
            rng: ChaChaRng::from_entropy(),
            activation_functions,
            learning_rate,
        };
        server.preprocess_zero_commitments(server.commitment_dimensions.num_row_coms);
        server
    }

    pub fn preprocess_zero_commitments(&mut self, amount: usize) {
        let rng = &mut self.rng;
        let randomizers = (0..amount)
            .into_iter()
            .map(|_| G::ScalarField::rand(rng))
            .collect::<Vec<_>>();

        self.zero_commitments = Some((
            randomizers
                .clone()
                .into_iter()
                .map(|x| {
                    self.client_schemes
                        .vector_commit
                        .get_h()
                        .mul(x)
                        .into_affine()
                })
                .collect::<_>(),
            randomizers,
        ));
    }

    pub fn initialize_neural_network(&mut self, nn: FeedForward<T>) -> FalafelResult<Vec<u8>> {
        let (cm_initial_weights, r_initial_weights) = self.commit_to_initial_weights(&nn);
        let message = InitializeMessage {
            neural_network: nn,
            cm_initial_weights,
            r_initial_weights: r_initial_weights.randomness,
        }
        .to_serializable()?;
        let bytes = bitcode::serialize(&message)
            .map_err(|e| FalafelError::SerializationError(e.to_string()))?;

        self.neural_network = Some(message.neural_network);
        Ok(bytes)
    }

    pub fn receive_and_verify_local_update(
        &mut self,
        data: (Vec<u8>, Vec<u8>, Vec<u8>),
        cm_initial_weights: &HyraxCommitment<G>,
    ) -> FalafelResult<bool> {
        //Process a local update step from a client. Verify their proof and store their updated weights until all are received.

        //Need to check consistency of witness with the proof (e.g. recompute the commitment based on the opening and verify whether it is the same is in the proof)
        let (message, weights, witness): (
            LocalUpdateMessageSerializable<T>,
            Vec<Array2<T>>,
            HyraxCommitmentOpeningSerializable,
        ) = (
            bitcode::deserialize(&data.0)
                .map_err(|e| FalafelError::DeserializationError(e.to_string()))?,
            bitcode::deserialize(&data.1)
                .map_err(|e| FalafelError::DeserializationError(e.to_string()))?,
            bitcode::deserialize(&data.2)
                .map_err(|e| FalafelError::DeserializationError(e.to_string()))?,
        );
        let (message, witness): (LocalUpdateMessage<G, T>, HyraxCommitmentOpening<G>) =
            (message.to_regular()?, witness.to_regular()?);

        if message.global_iter != self.global_iter {
            return Err(FalafelError::ProofError(
                "incorrect iteration count".to_string(),
            ));
        }

        match &message.proof.verify(
            &get_iv(message.client_id, message.global_iter),
            self.auditor.get_pp(),
            self.auditor.get_pk(),
            cm_initial_weights,
            &self.client_schemes,
        ) {
            Ok(_) => {
                let idx = message.client_id;
                self.local_proofs_iterations[idx] = Some((message, weights, witness));
                Ok(true)
            }
            Err(_) => Err(FalafelError::ProofError(format!(
                "Server failed to verify local proof of {}",
                message.client_id
            ))),
        }
    }

    pub fn extract_network_commitments_local_proof(
        &self,
        iv: &str,
        client_proof: LocalTrainingProof<G, T>,
        cm_initial_weights: &HyraxCommitment<G>,
    ) -> FalafelResult<(LocalTrainingCommitments<G>, HyraxCommitment<G>)> {
        let domain_separator: DomainSeparator<DefaultHash> = LocalTrainingProofCommitmentsSeparator::<G>::new_local_training_proof_commitments_separator(
            iv,
            &self.constants,
        );
        let mut verifier_state = domain_separator.to_verifier_state(&client_proof.proof);

        let [_cm_dataset_samples]: [G; 1] = verifier_state
            .next_points()
            .map_err(|e| FalafelError::DeserializationError(e.to_string()))?;

        let commitments = client_proof
            .obtain_commitments(&mut verifier_state)
            .unwrap();

        let cm_w_final = HyraxCommitment {
            row_coms: G::normalize_batch(
                &ark_std::cfg_into_iter!((0..cm_initial_weights.row_coms.len()))
                    .map(|idx| {
                        cm_initial_weights.row_coms[idx]
                            - ark_std::cfg_iter!(commitments.cm_g_hat_all)
                                .map(|cm_g_hat| cm_g_hat.row_coms[idx])
                                .sum::<G>()
                    })
                    .collect::<Vec<_>>(),
            ),
        };

        Ok((commitments, cm_w_final))
    }

    pub fn aggregate_and_prove(
        &mut self,
        cm_initial_weights: &HyraxCommitment<G>,
    ) -> FalafelResult<(
        FeedForward<T>,
        ProofResult<(AggregationProof, Vec<G::ScalarField>)>,
    )> {
        let iv = &get_iv(usize::MAX, self.global_iter);

        if self.local_proofs_iterations.len() == self.client_datasizes.len() {
            //Once all weight updates have been received for the iteration, aggregate them and generate the proof.
            let weights: Vec<FeedForward<T>> = self
                .local_proofs_iterations
                .iter()
                .map(|x| x.as_ref().unwrap())
                .map(|(_localupdate, weights, _witness)| {
                    FeedForward::new_from_weights(
                        weights.clone(),
                        self.activation_functions.clone(),
                        self.learning_rate,
                    )
                    .unwrap()
                })
                .collect();

            let weights_aggregated = weights.into_iter().sum();

            let client_weight_commitments: Vec<Vec<G::Affine>> = self
                .local_proofs_iterations
                .iter()
                .map(|x| x.as_ref().unwrap())
                .map(|(localupdate, _, _)| {
                    self.extract_network_commitments_local_proof(
                        iv,
                        localupdate.proof.clone(),
                        cm_initial_weights,
                    )
                    .unwrap()
                    .1
                    .row_coms
                })
                .collect();

            let client_commitment_witnesses: Vec<Vec<G::ScalarField>> = self
                .local_proofs_iterations
                .iter()
                .map(|x| x.as_ref().unwrap())
                .map(|(_, _, witness)| witness.randomness.clone())
                .collect();

            let proof = AggregationProof::prove_weight_aggregation(
                iv,
                &self.client_schemes,
                client_weight_commitments,
                client_commitment_witnesses,
                self.zero_commitments
                    .clone()
                    .ok_or_else(|| {
                        format!("Missing zero commitments in iteration {}", self.global_iter)
                    })
                    .unwrap(),
                self.commitment_dimensions,
            );

            self.global_iter += 1;
            Ok((weights_aggregated, proof))
        } else {
            Err(FalafelError::OtherError(
                "Not all local weight updates have been received".to_string(),
            ))
        }
    }

    fn commit_to_initial_weights(
        &mut self,
        nn: &FeedForward<T>,
    ) -> (HyraxCommitment<G>, HyraxCommitmentOpening<G>) {
        let w_initial_matrices = create_w_initial_matrices::<_, G>(
            &nn.layers
                .iter()
                .map(|layer| &layer.weights)
                .collect::<Vec<_>>(),
        );
        self.client_schemes.poly_commit.poly_commit_matrices(
            &w_initial_matrices,
            G::ScalarField::ZERO,
            true,
            None,
            &mut self.rng,
        )
    }
}
