//! Auditor structure, emulates an auditor that audit the dataset and signs it after. We only
//! implement the signing parts.

use crate::prelude::*;
use ark_serialize::CanonicalSerialize;
use ark_std::rand::Rng;
use rand::CryptoRng;

pub struct Auditor<'a, G: CurveGroup> {
    parameters: &'a AuditorSignatureParameters<G>,
    pk: AuditorSignaturePublicKey<G>,
    sk: AuditorSignatureSecretKey<G>,
}

impl<'a, G: CurveGroup> Auditor<'a, G> {
    pub fn new(
        parameters: &'a AuditorSignatureParameters<G>,
        rng: &mut (impl Rng + CryptoRng),
    ) -> FalafelResult<Self> {
        AuditorSignatureScheme::keygen(parameters, rng)
            .map(|(pk, sk)| Self { parameters, pk, sk })
            .map_err(|e| FalafelError::ArkworksError(e.to_string()))
    }

    pub fn get_pp(&self) -> &AuditorSignatureParameters<G> {
        &self.parameters
    }

    pub fn get_pk(&self) -> &AuditorSignaturePublicKey<G> {
        &self.pk
    }

    pub fn sign_dataset(
        &self,
        cm_dataset: &(G::Affine, G::Affine),
        rng: &mut (impl Rng + CryptoRng),
    ) -> FalafelResult<AuditorSignature<G>> {
        AuditorSignatureScheme::sign(
            self.parameters,
            &self.sk,
            &dataset_cm_to_bytes::<G>(cm_dataset)?,
            rng,
        )
        .map_err(|e| FalafelError::ArkworksError(e.to_string()))
    }
}

fn dataset_cm_to_bytes<G: CurveGroup>(
    cm_dataset: &(G::Affine, G::Affine),
) -> FalafelResult<Vec<u8>> {
    let mut message = vec![];
    cm_dataset.serialize_compressed(&mut message).map_err(|e| FalafelError::OtherError(format!("Tried to convert group elements to byte vector but failed unexpectedly, with the following message: {e}.")))?;
    Ok(message)
}

pub fn verify_dataset_signature<G: CurveGroup>(
    pp_signature: &AuditorSignatureParameters<G>,
    auditor_pk: &AuditorSignaturePublicKey<G>,
    cm_dataset: &(G::Affine, G::Affine),
    signature: &AuditorSignature<G>,
) -> FalafelResult<bool> {
    AuditorSignatureScheme::verify(
        pp_signature,
        auditor_pk,
        &dataset_cm_to_bytes::<G>(cm_dataset)?,
        signature,
    )
    .map_err(|e| FalafelError::ArkworksError(e.to_string()))
}
