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

pub trait ProductProofSeparator<G: CurveGroup> {
    fn add_product_proof(self) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> ProductProofSeparator<G> for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    fn add_product_proof(self) -> Self {
        // we assume the statement (x, y, z commitments) has already been sent
        self.add_points(3, "alpha, beta, delta")
            .challenge_scalars(1, "c")
            .add_scalars(5, "zs")
    }
}

pub struct ProductProof<'a, G: CurveGroup> {
    // commitments
    pub cm_x: &'a G::Affine,
    pub cm_y: &'a G::Affine,
    pub cm_z: &'a G::Affine,

    // witnesses
    pub x: Option<&'a G::ScalarField>,
    pub r_x: Option<&'a G::ScalarField>,
    pub y: Option<&'a G::ScalarField>,
    pub r_y: Option<&'a G::ScalarField>,
    pub r_z: Option<&'a G::ScalarField>,
}

impl<'a, G: CurveGroup> ProductProof<'a, G> {
    // Figure 5 in https://eprint.iacr.org/2017/1132.pdf
    pub fn prove(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        // 1
        let bs = (0..5)
            .into_iter()
            .map(|_| G::ScalarField::rand(&mut prover_state.rng()))
            .collect::<Vec<_>>();
        let alpha = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&bs[0], &bs[1]);
        let beta = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&bs[2], &bs[3]);
        let delta: G = self.cm_x.mul(bs[2]) + client_schemes.vector_commit.get_h().mul(bs[4]);
        prover_state.add_points(&[alpha.into_group(), beta.into_group(), delta])?;

        // 2
        let [c]: [G::ScalarField; 1] = prover_state.challenge_scalars()?;

        // 3
        let x = self.x.ok_or_else(|| ProofError::SerializationError)?;
        let y = self.y.ok_or_else(|| ProofError::SerializationError)?;
        let r_x = self.r_x.ok_or_else(|| ProofError::SerializationError)?;
        let r_y = self.r_y.ok_or_else(|| ProofError::SerializationError)?;
        let r_z = self.r_z.ok_or_else(|| ProofError::SerializationError)?;
        let zs = [
            bs[0] + c * x,
            bs[1] + c * r_x,
            bs[2] + c * y,
            bs[3] + c * r_y,
            bs[4] + c * (*r_z - *r_x * y),
        ];
        prover_state.add_scalars(&zs)?;

        // 4 is verifier only
        Ok(())
    }

    pub fn verify(
        &self,
        verifier_state: &mut VerifierState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        // 1
        let [alpha, beta, delta]: [G; 3] = verifier_state.next_points()?;

        // 2
        let [c]: [G::ScalarField; 1] = verifier_state.challenge_scalars()?;

        // 3
        let zs: [G::ScalarField; 5] = verifier_state.next_scalars()?;

        // 4
        // grab the relevant generators directly, since we compute custom 'commitments'
        let g = client_schemes.vector_commit.get_g_unique();
        let h = client_schemes.vector_commit.get_h();

        // check 1
        if alpha + self.cm_x.mul(&c) != g.mul(zs[0]) + h.mul(zs[1]) {
            return Err(ProofError::InvalidProof);
        }
        // check 2
        if beta + self.cm_y.mul(&c) != g.mul(zs[2]) + h.mul(zs[3]) {
            return Err(ProofError::InvalidProof);
        }
        // check 3
        if delta + self.cm_z.mul(&c) != self.cm_x.mul(&zs[2]) + h.mul(zs[4]) {
            return Err(ProofError::InvalidProof);
        }

        Ok(())
    }
}
