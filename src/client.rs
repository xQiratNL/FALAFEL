//! Client structure, this is the main place where all client side functionality is tied together.

use crate::crypto_schemes::{
    Constants, Generators, HyraxCommitment, HyraxCommitmentOpening,
    HyraxPolynomialCommitmentScheme, PedersenVectorCommitmentScheme,
};
use crate::dataset::Dataset;
use crate::message::{InitializeMessage, InitializeMessageSerializable, LocalUpdateMessage};
use crate::neural_network::FeedForward;
use crate::prelude::*;
use crate::proof::bench::ClientTimedResult;
use crate::proof::commitments::create_w_initial_matrices;
use crate::proof::{LocalTrainingProofConstants, ProofWitnessFull};
use crate::util::{IntoField, get_iv, permute};
use ark_ff::AdditiveGroup;
use ark_std::UniformRand;
use ndarray::Order::ColumnMajor;
use ndarray::{Array2, Axis, ShapeBuilder, concatenate};
use rand::{CryptoRng, Rng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

pub struct ClientSchemes<'a, G: CurveGroup> {
    pub vector_commit: PedersenVectorCommitmentScheme<'a, G>,
    pub poly_commit: HyraxPolynomialCommitmentScheme<'a, G>,
    pub constants: Constants<G::ScalarField>,
}

impl<'a, G: CurveGroup> ClientSchemes<'a, G> {
    pub fn setup(generators: &'a Generators<G>) -> Self {
        Self {
            vector_commit: PedersenVectorCommitmentScheme::setup(generators),
            poly_commit: HyraxPolynomialCommitmentScheme::setup(generators),
            constants: Constants::new(),
        }
    }
}

pub struct Client<'a, T: Value, G: CurveGroup> {
    pub id: usize,
    pub schemes: ClientSchemes<'a, G>,
    dataset: Dataset<T>,
    dataset_shuffled: Dataset<T>,
    dataset_permutation: Option<Vec<usize>>,
    cm_dataset_samples: Option<G::Affine>,
    cm_dataset_labels: Option<G::Affine>,
    r_dataset_samples: Option<G::ScalarField>,
    r_dataset_labels: Option<G::ScalarField>,
    dataset_scalar_samples: Option<Vec<G::ScalarField>>,
    dataset_scalar_labels: Option<Vec<G::ScalarField>>,
    dataset_scalar_samples_permuted: Option<Vec<G::ScalarField>>,
    dataset_scalar_labels_permuted: Option<Vec<G::ScalarField>>,
    dataset_signature: Option<AuditorSignature<G>>,
    neural_network: Option<FeedForward<T>>,
    cm_initial_weights: Option<HyraxCommitment<G>>,
    r_initial_weights: Option<Vec<G::ScalarField>>,
}

impl<'a, T: Value, G: CurveGroup> Client<'a, T, G> {
    pub fn new(id: usize, dataset: Dataset<T>, generators: &'a Generators<G>) -> Self {
        Self {
            id,
            schemes: ClientSchemes::setup(generators),
            dataset: dataset.clone(),
            dataset_shuffled: dataset,
            dataset_permutation: None,
            cm_initial_weights: None,
            r_initial_weights: None,
            cm_dataset_samples: None,
            cm_dataset_labels: None,
            r_dataset_samples: None,
            r_dataset_labels: None,
            dataset_scalar_samples: None,
            dataset_scalar_labels: None,
            dataset_scalar_samples_permuted: None,
            dataset_scalar_labels_permuted: None,
            dataset_signature: None,
            neural_network: None,
        }
    }

    pub fn commit_dataset(&mut self, rng: &mut (impl Rng + CryptoRng)) {
        let samples = self.dataset.get_samples();
        let labels = self.dataset.get_labels();

        #[cfg(not(feature = "parallel"))]
        // ColumnMajor order is VERY IMPORTANT to keep here, other things are also in the same order, and it is also most sensible, code-wise
        let dataset_scalar_samples = concatenate![
            Axis(0),
            Array2::from_elem((1, samples.ncols()), T::zero()),
            samples
        ]
        .flatten_with_order(ColumnMajor)
        .view()
        .into_iter()
        .map(|value| value.to_field())
        .collect::<Vec<_>>();
        #[cfg(not(feature = "parallel"))]
        let dataset_scalar_labels = labels
            .flatten_with_order(ColumnMajor)
            .view()
            .into_iter()
            .map(|value| value.to_field())
            .collect::<Vec<_>>();

        #[cfg(feature = "parallel")]
        // ColumnMajor order is VERY IMPORTANT to keep here, other things are also in the same order, and it is also most sensible, code-wise
        let dataset_scalar_samples = concatenate![
            Axis(0),
            Array2::from_elem((1, samples.ncols()), T::zero()),
            samples
        ]
        .flatten_with_order(ColumnMajor)
        .view()
        .into_par_iter()
        .map(|value| value.to_field())
        .collect::<Vec<_>>();
        #[cfg(feature = "parallel")]
        let dataset_scalar_labels = labels
            .flatten_with_order(ColumnMajor)
            .view()
            .into_par_iter()
            .map(|value| value.to_field())
            .collect::<Vec<_>>();

        let randomizer_samples = G::ScalarField::rand(rng);
        let randomizer_labels = G::ScalarField::rand(rng);
        self.cm_dataset_samples = Some(
            self.schemes
                .vector_commit
                .vector_commit(&dataset_scalar_samples, &randomizer_samples),
        );
        self.cm_dataset_labels = Some(
            self.schemes
                .vector_commit
                .vector_commit(&dataset_scalar_labels, &randomizer_labels),
        );
        self.dataset_scalar_samples = Some(dataset_scalar_samples);
        self.dataset_scalar_labels = Some(dataset_scalar_labels);
        self.r_dataset_samples = Some(randomizer_samples);
        self.r_dataset_labels = Some(randomizer_labels);
    }

    pub fn get_dataset_cm(&self) -> FalafelResult<(G::Affine, G::Affine)> {
        if let (Some(cm_samples), Some(cm_labels)) =
            (self.cm_dataset_samples, self.cm_dataset_labels)
        {
            Ok((cm_samples, cm_labels))
        } else {
            Err(FalafelError::UnsetError("dataset commitment".to_string()))
        }
    }

    pub fn store_dataset_signature(&mut self, signature: AuditorSignature<G>) {
        self.dataset_signature = Some(signature);
    }

    pub fn initialize_neural_network(&mut self, message: &[u8]) -> FalafelResult<()> {
        let message: InitializeMessageSerializable<T> = bitcode::deserialize(message)
            .map_err(|e| FalafelError::DeserializationError(e.to_string()))?;
        let message: InitializeMessage<T, G> = message.to_regular()?;
        self.neural_network = Some(message.neural_network);
        self.cm_initial_weights = Some(message.cm_initial_weights);
        self.r_initial_weights = Some(message.r_initial_weights);
        Ok(())
    }

    pub fn shuffle_dataset(&mut self, rng: &mut impl Rng) -> FalafelResult<()> {
        // re-initialize to original one + permute
        self.dataset_shuffled = self.dataset.clone();
        let permutation = self.dataset_shuffled.shuffle(rng);
        let n_features = self.dataset.get_samples().nrows();
        let n_labels = self.dataset.get_labels().nrows();
        let n_samples = self.dataset.get_samples().ncols();
        let sample_shape = (n_features + 1, n_samples).strides((1, n_features + 1));
        let label_shape = (n_labels, n_samples).strides((1, n_labels));
        self.dataset_scalar_samples_permuted = Some(permute(
            self.dataset_scalar_samples.as_ref().ok_or_else(|| {
                FalafelError::UnsetError("client.permuted_dataset_scalar_samples".to_string())
            })?,
            sample_shape,
            &permutation,
        ));
        self.dataset_scalar_labels_permuted = Some(permute(
            self.dataset_scalar_labels.as_ref().ok_or_else(|| {
                FalafelError::UnsetError("client.permuted_dataset_scalar_labels".to_string())
            })?,
            label_shape,
            &permutation,
        ));
        self.dataset_permutation = Some(permutation);
        Ok(())
    }

    pub fn update_neural_network_weights(&mut self, weights: Vec<Array2<T>>) -> FalafelResult<()> {
        self.neural_network
            .as_mut()
            .ok_or_else(|| FalafelError::UnsetError("client.neural_network".to_string()))?
            .set_weights(weights)
    }

    pub fn set_new_network(
        &mut self,
        cm_new_weights: Vec<G::Affine>,
        new_network: FeedForward<T>,
        r_new_weights: Vec<G::ScalarField>,
    ) {
        self.cm_initial_weights = Some(HyraxCommitment {
            row_coms: cm_new_weights,
        });
        self.r_initial_weights = Some(r_new_weights);
        self.neural_network = Some(new_network);
    }

    pub fn prove_local_training_full(
        &mut self,
        n_local_iter: usize,
        batch_size: usize,
        global_iter_count: usize,
        total_bits: usize,
        decimal_bits: usize,
        rng: &mut (impl Rng + CryptoRng),
    ) -> FalafelResult<(Vec<u8>, Vec<u8>, Vec<u8>)> {
        // compute witness for training steps
        let nn = self
            .neural_network
            .as_mut()
            .ok_or_else(|| FalafelError::UnsetError("client.neural_network".to_string()))?;
        let initial_weights = nn
            .layers
            .iter()
            .map(|layer| layer.weights.clone())
            .collect::<Vec<_>>();
        let feed_forward_steps = nn.local_training_full(
            &self.dataset_shuffled,
            n_local_iter,
            batch_size,
            decimal_bits,
        )?;

        // compute constants
        let n_samples = self.dataset.get_samples().ncols();
        let n_features = self.dataset.get_samples().nrows();
        let mut n_hidden = nn
            .layers
            .iter()
            .map(|layer| layer.weights.nrows())
            .collect::<Vec<_>>();
        let n_labels = n_hidden.pop().expect("This should exist!");

        let constants = LocalTrainingProofConstants::new(
            total_bits,
            decimal_bits,
            n_local_iter,
            batch_size,
            n_features,
            n_samples,
            &n_hidden,
            n_labels,
            &nn,
        );

        // construct full witness
        let witness = ProofWitnessFull {
            initial_weights: &initial_weights,
            opening_initial_weights: HyraxCommitmentOpening {
                input_matrix: self.schemes.poly_commit.combine_matrices(
                    &create_w_initial_matrices::<_, G>(
                        &initial_weights.iter().map(|x| x).collect::<Vec<_>>(),
                    ),
                    G::ScalarField::ZERO,
                    true,
                    None,
                ),
                randomness: self
                    .r_initial_weights
                    .as_ref()
                    .ok_or_else(|| {
                        FalafelError::UnsetError("client.r_initial_weights".to_string())
                    })?
                    .clone(),
            },
            feed_forward_steps: &feed_forward_steps,
            cm_dataset_samples: &self
                .cm_dataset_samples
                .ok_or_else(|| FalafelError::UnsetError("client.commitment_samples".to_string()))?,
            cm_dataset_labels: &self
                .cm_dataset_labels
                .ok_or_else(|| FalafelError::UnsetError("client.commitment_labels".to_string()))?,
            r_dataset_samples: &self.r_dataset_samples.ok_or_else(|| {
                FalafelError::UnsetError("client.commitment_r_samples".to_string())
            })?,
            r_dataset_labels: &self.r_dataset_labels.ok_or_else(|| {
                FalafelError::UnsetError("client.commitment_r_labels".to_string())
            })?,
            dataset_scalar_samples_permuted: self
                .dataset_scalar_samples_permuted
                .as_ref()
                .ok_or_else(|| {
                    FalafelError::UnsetError("client.dataset_scalar_samples".to_string())
                })?,
            dataset_scalar_labels_permuted: self
                .dataset_scalar_labels_permuted
                .as_ref()
                .ok_or_else(|| {
                    FalafelError::UnsetError("client.dataset_scalar_labels".to_string())
                })?,
            permutation: self.dataset_permutation.as_ref().ok_or_else(|| {
                FalafelError::UnsetError("client.dataset_permutation".to_string())
            })?,
            constants,
            dataset_shuffled: self.dataset_shuffled.clone(),
        };

        // compute the proof
        let dataset_signature = self
            .dataset_signature
            .clone()
            .ok_or_else(|| FalafelError::UnsetError("client.commitment_r".to_string()))?;
        let (proof, opening_w_final) = witness
            .prove(
                &get_iv(self.id, global_iter_count),
                dataset_signature,
                &self.schemes,
                rng,
            )
            .map_err(|e| FalafelError::ProofError(e.to_string()))?;

        let message_all = LocalUpdateMessage {
            client_id: self.id,
            global_iter: global_iter_count,
            proof,
        }
        .to_serializable()?;
        let message_bytes_all = bitcode::serialize(&message_all)
            .map_err(|e| FalafelError::SerializationError(e.to_string()))?;
        let final_local_weights = feed_forward_steps.last().unwrap().weights_after.clone();
        let final_local_weights_serialized = bitcode::serialize(&final_local_weights)
            .map_err(|e| FalafelError::SerializationError(e.to_string()))?;

        let witness_bytes_server = bitcode::serialize(&opening_w_final.to_serializable()?)
            .map_err(|e| FalafelError::SerializationError(e.to_string()))?;

        Ok((
            message_bytes_all,
            final_local_weights_serialized,
            witness_bytes_server,
        ))
    }

    pub fn prove_local_training_full_timed(
        &mut self,
        n_local_iter: usize,
        batch_size: usize,
        global_iter_count: usize,
        total_bits: usize,
        decimal_bits: usize,
        rng: &mut (impl Rng + CryptoRng),
    ) -> FalafelResult<ClientTimedResult> {
        let nn = self
            .neural_network
            .as_mut()
            .ok_or_else(|| FalafelError::UnsetError("client.neural_network".to_string()))?;
        let initial_weights = nn
            .layers
            .iter()
            .map(|layer| layer.weights.clone())
            .collect::<Vec<_>>();
        let feed_forward_steps = nn.local_training_full(
            &self.dataset_shuffled,
            n_local_iter,
            batch_size,
            decimal_bits,
        )?;

        let n_samples = self.dataset.get_samples().ncols();
        let n_features = self.dataset.get_samples().nrows();
        let mut n_hidden = nn
            .layers
            .iter()
            .map(|layer| layer.weights.nrows())
            .collect::<Vec<_>>();
        let n_labels = n_hidden.pop().expect("This should exist!");

        let constants = LocalTrainingProofConstants::new(
            total_bits,
            decimal_bits,
            n_local_iter,
            batch_size,
            n_features,
            n_samples,
            &n_hidden,
            n_labels,
            &nn,
        );

        let witness = ProofWitnessFull {
            initial_weights: &initial_weights,
            opening_initial_weights: HyraxCommitmentOpening {
                input_matrix: self.schemes.poly_commit.combine_matrices(
                    &create_w_initial_matrices::<_, G>(
                        &initial_weights.iter().map(|x| x).collect::<Vec<_>>(),
                    ),
                    G::ScalarField::ZERO,
                    true,
                    None,
                ),
                randomness: self
                    .r_initial_weights
                    .as_ref()
                    .ok_or_else(|| {
                        FalafelError::UnsetError("client.r_initial_weights".to_string())
                    })?
                    .clone(),
            },
            feed_forward_steps: &feed_forward_steps,
            cm_dataset_samples: &self
                .cm_dataset_samples
                .ok_or_else(|| FalafelError::UnsetError("client.commitment_samples".to_string()))?,
            cm_dataset_labels: &self
                .cm_dataset_labels
                .ok_or_else(|| FalafelError::UnsetError("client.commitment_labels".to_string()))?,
            r_dataset_samples: &self.r_dataset_samples.ok_or_else(|| {
                FalafelError::UnsetError("client.commitment_r_samples".to_string())
            })?,
            r_dataset_labels: &self.r_dataset_labels.ok_or_else(|| {
                FalafelError::UnsetError("client.commitment_r_labels".to_string())
            })?,
            dataset_scalar_samples_permuted: self
                .dataset_scalar_samples_permuted
                .as_ref()
                .ok_or_else(|| {
                    FalafelError::UnsetError("client.dataset_scalar_samples".to_string())
                })?,
            dataset_scalar_labels_permuted: self
                .dataset_scalar_labels_permuted
                .as_ref()
                .ok_or_else(|| {
                    FalafelError::UnsetError("client.dataset_scalar_labels".to_string())
                })?,
            permutation: self.dataset_permutation.as_ref().ok_or_else(|| {
                FalafelError::UnsetError("client.dataset_permutation".to_string())
            })?,
            constants,
            dataset_shuffled: self.dataset_shuffled.clone(),
        };

        let dataset_signature = self
            .dataset_signature
            .clone()
            .ok_or_else(|| FalafelError::UnsetError("client.commitment_r".to_string()))?;
        let timed = witness
            .prove_timed(
                &get_iv(self.id, global_iter_count),
                dataset_signature,
                &self.schemes,
                rng,
            )
            .map_err(|e| FalafelError::ProofError(e.to_string()))?;

        let timings = timed.timings;

        let message_all = LocalUpdateMessage {
            client_id: self.id,
            global_iter: global_iter_count,
            proof: timed.proof,
        }
        .to_serializable()?;
        let message_bytes_all = bitcode::serialize(&message_all)
            .map_err(|e| FalafelError::SerializationError(e.to_string()))?;
        let final_local_weights = feed_forward_steps.last().unwrap().weights_after.clone();
        let final_local_weights_serialized = bitcode::serialize(&final_local_weights)
            .map_err(|e| FalafelError::SerializationError(e.to_string()))?;
        let witness_bytes_server = bitcode::serialize(&timed.opening.to_serializable()?)
            .map_err(|e| FalafelError::SerializationError(e.to_string()))?;

        Ok(ClientTimedResult {
            serialized: (
                message_bytes_all,
                final_local_weights_serialized,
                witness_bytes_server,
            ),
            timings,
        })
    }
}
