use crate::client::ClientSchemes;
use ark_ec::{AffineRepr, CurveGroup};
use ark_std::UniformRand;
use spongefish::codecs::arkworks_algebra::{
    FieldDomainSeparator, FieldToUnitDeserialize, FieldToUnitSerialize, GroupDomainSeparator,
    GroupToUnitDeserialize, GroupToUnitSerialize, UnitToField,
};
use spongefish::{
    DomainSeparator, DuplexSpongeInterface, ProofError, ProofResult, ProverState, VerifierState,
};
use std::ops::Mul;

pub trait BatchOpeningProofSeparator<G: CurveGroup> {
    fn add_batch_opening_proof(self) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> BatchOpeningProofSeparator<G> for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    fn add_batch_opening_proof(self) -> Self {
        self.add_points(1, "alpha")
            .challenge_scalars(1, "c")
            .add_scalars(2, "z1, z2")
    }
}

pub struct BatchOpeningProof<'a, G: CurveGroup> {
    // commitments
    pub cms: &'a [G::Affine],

    // witnesses
    pub xs: Option<&'a [G::ScalarField]>,
    pub rs: Option<&'a [G::ScalarField]>,
}

impl<'a, G: CurveGroup> BatchOpeningProof<'a, G> {
    // Based on ideas from Figure 8 in https://eprint.iacr.org/2017/1132.pdf and https://www.michaelstraka.com/batchedschnorr
    pub fn prove(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        // 1
        let t1 = G::ScalarField::rand(&mut prover_state.rng());
        let t2 = G::ScalarField::rand(&mut prover_state.rng());

        let alpha = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&t1, &t2);
        prover_state.add_points(&[alpha.into_group()])?;

        // 2
        let [c]: [G::ScalarField; 1] = prover_state.challenge_scalars()?;

        // 3
        let xs = self.xs.ok_or_else(|| ProofError::SerializationError)?;
        let rs = self.rs.ok_or_else(|| ProofError::SerializationError)?;
        let mut z1 = t1;
        let mut z2 = t2;
        let mut c_to_the_power_i = c;
        for (x, r) in xs.iter().zip(rs) {
            z1 += c_to_the_power_i * x;
            z2 += c_to_the_power_i * r;
            c_to_the_power_i *= c;
        }
        prover_state.add_scalars(&[z1, z2])?;

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
        let [z1, z2]: [G::ScalarField; 2] = verifier_state.next_scalars()?;

        // 4
        let mut rhs = alpha;
        let mut c_to_the_power_i = c;
        for cm in self.cms {
            rhs += cm.mul(c_to_the_power_i);
            c_to_the_power_i *= c;
        }

        if client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&z1, &z2)
            != rhs.into_affine()
        {
            return Err(ProofError::InvalidProof);
        }

        Ok(())
    }
}
