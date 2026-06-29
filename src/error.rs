//! Error types for FALAFEL

use std::fmt::{Display, Formatter};

#[derive(Debug, Clone)]
pub enum FalafelError {
    ArkworksError(String),
    InputError(String),
    UnsetError(String),
    SerializationError(String),
    ProofError(String),
    VerificationError(String),
    InvalidConfig(String),
    DeserializationError(String),
    OtherError(String),
    SetupLockedError,
}

impl Display for FalafelError {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        match self {
            FalafelError::InputError(msg) => {
                write!(f, "Invalid input: {}", msg)
            }
            FalafelError::UnsetError(msg) => {
                write!(
                    f,
                    "Tried to access/change `{}`, but this variable has not yet been set.",
                    msg
                )
            }
            FalafelError::InvalidConfig(msg) => {
                write!(f, "Invalid configuration: {}", msg)
            }
            FalafelError::OtherError(msg) => {
                write!(f, "Error: {}", msg)
            }
            FalafelError::SetupLockedError => {
                write!(
                    f,
                    "Tried to create new generators, but the setup phase has already been locked. \
                        Most likely, you are trying to generate new parameters after the setup phase. \
                        For performance measurement reasons, all generators should be sampled during the setup phase."
                )
            }
            FalafelError::ArkworksError(msg) => write!(f, "Arkworks error: {}", msg),
            FalafelError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            FalafelError::DeserializationError(msg) => write!(f, "Deserialization error: {}", msg),
            FalafelError::ProofError(msg) => write!(f, "Proof error: {}", msg),
            FalafelError::VerificationError(msg) => write!(f, "Verification error: {}", msg),
        }
    }
}
