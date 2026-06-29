//! Convenient re-exports and aliases that we often need in this crate.

use ark_crypto_primitives::signature::schnorr;
use ark_crypto_primitives::signature::schnorr::Schnorr;
use blake2::Blake2s256;
use ndarray::LinalgScalar;
use num_traits::{FromPrimitive, Signed, ToBytes};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::fmt::Debug;

// convenient re-exports
pub use crate::error::*;
use crate::message::SchnorrSignatureSerialized;
pub use ark_crypto_primitives::signature::SignatureScheme;
pub use ark_ec::{AffineRepr, CurveGroup};
pub use ark_ff::Field;

// consts
pub const PROTOCOL_NAME: &[u8] = b"FALAFEL";

// a convenient trait for values we can use to compute with (e.g., fixed point, floating point, ...)
pub trait Value:
    LinalgScalar
    + Signed
    + FromPrimitive
    + Default
    + ToBytes
    + Send
    + Sync
    + Serialize
    + DeserializeOwned
    + Debug
{
}

impl<T> Value for T where
    T: LinalgScalar
        + Signed
        + FromPrimitive
        + Default
        + ToBytes
        + Send
        + Sync
        + Serialize
        + DeserializeOwned
        + Debug
{
}

// convenient aliases
pub type FalafelResult<T> = Result<T, FalafelError>;
pub type AuditorSignatureScheme<G> = Schnorr<G, Blake2s256>;
pub type AuditorSignatureParameters<G> = schnorr::Parameters<G, Blake2s256>;
pub type AuditorSignaturePublicKey<G> = schnorr::PublicKey<G>;
pub type AuditorSignatureSecretKey<G> = schnorr::SecretKey<G>;
pub type AuditorSignature<G> = schnorr::Signature<G>;
pub type AuditorSignatureSerialized = SchnorrSignatureSerialized;
