use crate::client::ClientSchemes;
use ark_ec::CurveGroup;
use ark_std::UniformRand;
use spongefish::codecs::arkworks_algebra::{
    FieldDomainSeparator, FieldToUnitDeserialize, FieldToUnitSerialize, GroupDomainSeparator,
    GroupToUnitDeserialize, GroupToUnitSerialize, UnitToField,
};
use spongefish::{
    DomainSeparator, DuplexSpongeInterface, ProofError, ProofResult, ProverState, VerifierState,
};
use std::ops::Mul;

pub trait BatchEqualityProofSeparator<G: CurveGroup> {
    fn add_batch_equality_proof(self) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> BatchEqualityProofSeparator<G> for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    fn add_batch_equality_proof(self) -> Self {
        self.add_points(1, "alpha")
            .challenge_scalars(1, "c")
            .add_scalars(1, "z")
    }
}

pub struct BatchEqualityProof<'a, G: CurveGroup> {
    // commitments
    pub cms_lhs: Option<&'a [G::Affine]>,
    pub cms_rhs: Option<&'a [G::Affine]>,

    // witnesses
    pub rs_lhs: Option<&'a [G::ScalarField]>,
    pub rs_rhs: Option<&'a [G::ScalarField]>,
}

impl<'a, G: CurveGroup> BatchEqualityProof<'a, G> {
    // Based on ideas from Figure 9 in https://eprint.iacr.org/2017/1132.pdf and https://www.michaelstraka.com/batchedschnorr
    pub fn prove(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        let h = client_schemes.vector_commit.get_h();

        // 1
        let t = G::ScalarField::rand(&mut prover_state.rng());
        let alpha = h.mul(t);
        prover_state.add_points(&[alpha])?;

        // 2
        let [c]: [G::ScalarField; 1] = prover_state.challenge_scalars()?;

        // 3
        let rs_lhs = self.rs_lhs.ok_or_else(|| ProofError::SerializationError)?;
        let rs_rhs = self.rs_rhs.ok_or_else(|| ProofError::SerializationError)?;
        let mut z = t;
        let mut c_to_the_power_i = c;
        for (&r_lhs, r_rhs) in rs_lhs.iter().zip(rs_rhs) {
            z += c_to_the_power_i * (r_lhs - r_rhs);
            c_to_the_power_i *= c;
        }
        prover_state.add_scalars(&[z])?;

        // 4 is verifier only
        Ok(())
    }

    pub fn verify(
        &self,
        verifier_state: &mut VerifierState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        // 1
        let [alpha]: [G; 1] = verifier_state.next_points()?;

        // 2
        let [c]: [G::ScalarField; 1] = verifier_state.challenge_scalars()?;

        // 3
        let [z]: [G::ScalarField; 1] = verifier_state.next_scalars()?;

        // 4
        let h = client_schemes.vector_commit.get_h();
        let cms_lhs = self.cms_lhs.ok_or_else(|| ProofError::SerializationError)?;
        let cms_rhs = self.cms_rhs.ok_or_else(|| ProofError::SerializationError)?;

        let mut rhs = alpha;
        let mut c_to_the_power_i = c;
        for (&cm_lhs, cm_rhs) in cms_lhs.iter().zip(cms_rhs) {
            rhs += (cm_lhs - cm_rhs).mul(c_to_the_power_i);
            c_to_the_power_i *= c;
        }

        if h.mul(z) != rhs {
            return Err(ProofError::InvalidProof);
        }

        Ok(())
    }
}
