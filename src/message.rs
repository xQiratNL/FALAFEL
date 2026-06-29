//! Message structs, each struct contains all data is contained in a single message sent by a client
//! or server.

use crate::crypto_schemes::HyraxCommitment;
use crate::crypto_schemes::HyraxCommitmentOpening;
use crate::neural_network::FeedForward;
use crate::prelude::*;
use crate::proof::{LocalTrainingProof, LocalTrainingProofConstants};
use ark_crypto_primitives::signature::schnorr;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use serde::{Deserialize, Serialize};

pub struct InitializeMessage<T: Value, G: CurveGroup> {
    pub neural_network: FeedForward<T>,
    pub cm_initial_weights: HyraxCommitment<G>,
    pub r_initial_weights: Vec<G::ScalarField>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(deserialize = "T: Value"))]
pub struct InitializeMessageSerializable<T: Value> {
    pub neural_network: FeedForward<T>,
    pub cm_initial_weights_serialized: Vec<u8>,
    pub r_initial_weights_serialized: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct LocalUpdateMessage<G: CurveGroup, T: Value> {
    pub client_id: usize,
    pub global_iter: usize,
    pub proof: LocalTrainingProof<G, T>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(deserialize = "T: Value"))]
pub struct LocalUpdateMessageSerializable<T: Value> {
    pub client_id: usize,
    pub global_iter: usize,
    pub proof: LocalTrainingProofSerializable<T>,
}

#[derive(Serialize, Deserialize)]
#[serde(bound(deserialize = "T: Value"))]
pub struct LocalTrainingProofSerializable<T: Value> {
    dataset_signature_serialized: AuditorSignatureSerialized,
    proof: Vec<u8>,
    constants: LocalTrainingProofConstants<T>,
}

#[derive(Serialize, Deserialize)]
pub struct SchnorrSignatureSerialized {
    prover_response_serialized: Vec<u8>,
    verifier_challenge_serialized: Vec<u8>,
}

#[derive(Serialize, Deserialize)]
pub struct HyraxCommitmentOpeningSerializable {
    pub input_matrix_serialized: Vec<u8>,
    pub randomness_serialized: Vec<u8>,
}

impl<G: CurveGroup> HyraxCommitmentOpening<G> {
    pub fn to_serializable(self) -> FalafelResult<HyraxCommitmentOpeningSerializable> {
        let mut input_matrix_serialized = vec![];
        let mut randomness_serialized = vec![];

        self.input_matrix
            .serialize_compressed(&mut input_matrix_serialized)
            .map_err(|e| FalafelError::SerializationError(e.to_string()))?;

        self.randomness
            .serialize_compressed(&mut randomness_serialized)
            .map_err(|e| FalafelError::SerializationError(e.to_string()))?;

        Ok(HyraxCommitmentOpeningSerializable {
            input_matrix_serialized,
            randomness_serialized,
        })
    }
}

impl HyraxCommitmentOpeningSerializable {
    pub fn to_regular<G: CurveGroup>(self) -> FalafelResult<HyraxCommitmentOpening<G>> {
        Ok(HyraxCommitmentOpening {
            input_matrix: <Vec<Vec<G::ScalarField>>>::deserialize_compressed(
                self.input_matrix_serialized.as_slice(),
            )
            .map_err(|e| FalafelError::DeserializationError(e.to_string()))?,
            randomness: <Vec<G::ScalarField>>::deserialize_compressed(
                self.randomness_serialized.as_slice(),
            )
            .map_err(|e| FalafelError::DeserializationError(e.to_string()))?,
        })
    }
}

impl<T: Value, G: CurveGroup> InitializeMessage<T, G> {
    pub fn to_serializable(self) -> FalafelResult<InitializeMessageSerializable<T>> {
        let mut cm_initial_weights_serialized = vec![];
        self.cm_initial_weights
            .row_coms
            .serialize_compressed(&mut cm_initial_weights_serialized)
            .map_err(|e| FalafelError::SerializationError(e.to_string()))?;
        let mut r_initial_weights_serialized = vec![];
        self.r_initial_weights
            .serialize_compressed(&mut r_initial_weights_serialized)
            .map_err(|e| FalafelError::SerializationError(e.to_string()))?;
        Ok(InitializeMessageSerializable {
            neural_network: self.neural_network,
            cm_initial_weights_serialized,
            r_initial_weights_serialized,
        })
    }
}

impl<T: Value> InitializeMessageSerializable<T> {
    pub fn to_regular<G: CurveGroup>(self) -> FalafelResult<InitializeMessage<T, G>> {
        Ok(InitializeMessage {
            neural_network: self.neural_network,
            cm_initial_weights: HyraxCommitment {
                row_coms: Vec::<G::Affine>::deserialize_compressed(
                    self.cm_initial_weights_serialized.as_slice(),
                )
                .map_err(|e| FalafelError::DeserializationError(e.to_string()))?,
            },
            r_initial_weights: Vec::<G::ScalarField>::deserialize_compressed(
                self.r_initial_weights_serialized.as_slice(),
            )
            .map_err(|e| FalafelError::DeserializationError(e.to_string()))?,
        })
    }
}

impl<G: CurveGroup, T: Value> LocalUpdateMessage<G, T> {
    pub fn to_serializable(self) -> FalafelResult<LocalUpdateMessageSerializable<T>> {
        Ok(LocalUpdateMessageSerializable {
            client_id: self.client_id,
            global_iter: self.global_iter,
            proof: self.proof.to_serializable()?,
        })
    }
}

impl<T: Value> LocalUpdateMessageSerializable<T> {
    pub fn to_regular<G: CurveGroup>(self) -> FalafelResult<LocalUpdateMessage<G, T>> {
        Ok(LocalUpdateMessage {
            client_id: self.client_id,
            global_iter: self.global_iter,
            proof: self.proof.to_regular()?,
        })
    }
}

impl<G: CurveGroup, T: Value> LocalTrainingProof<G, T> {
    fn to_serializable(self) -> FalafelResult<LocalTrainingProofSerializable<T>> {
        Ok(LocalTrainingProofSerializable {
            dataset_signature_serialized: AuditorSignatureSerialized::from_signature(
                self.dataset_signature,
            )?,
            proof: self.proof,
            constants: self.constants,
        })
    }
}

impl<T: Value> LocalTrainingProofSerializable<T> {
    fn to_regular<G: CurveGroup>(self) -> FalafelResult<LocalTrainingProof<G, T>> {
        Ok(LocalTrainingProof {
            dataset_signature: self.dataset_signature_serialized.to_regular()?,
            proof: self.proof,
            constants: self.constants,
        })
    }
}

impl SchnorrSignatureSerialized {
    fn from_signature<G: CurveGroup>(signature: schnorr::Signature<G>) -> FalafelResult<Self> {
        let mut prover_response_serialized = vec![];
        signature
            .prover_response
            .serialize_compressed(&mut prover_response_serialized)
            .map_err(|e| FalafelError::SerializationError(e.to_string()))?;

        let mut verifier_challenge_serialized = vec![];
        signature
            .verifier_challenge
            .serialize_compressed(&mut verifier_challenge_serialized)
            .map_err(|e| FalafelError::SerializationError(e.to_string()))?;

        Ok(Self {
            prover_response_serialized,
            verifier_challenge_serialized,
        })
    }

    fn to_regular<G: CurveGroup>(self) -> FalafelResult<schnorr::Signature<G>> {
        Ok(schnorr::Signature {
            prover_response: G::ScalarField::deserialize_compressed(
                self.prover_response_serialized.as_slice(),
            )
            .map_err(|e| FalafelError::DeserializationError(e.to_string()))?,
            verifier_challenge: G::ScalarField::deserialize_compressed(
                self.verifier_challenge_serialized.as_slice(),
            )
            .map_err(|e| FalafelError::DeserializationError(e.to_string()))?,
        })
    }
}
