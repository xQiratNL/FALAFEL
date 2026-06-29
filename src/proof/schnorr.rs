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

pub trait SchnorrSeparator<G: CurveGroup> {
    fn add_schnorr_proof(self, repetitions: usize) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> SchnorrSeparator<G> for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    fn add_schnorr_proof(self, repetitions: usize) -> Self {
        self.add_points(repetitions, "a")
            .challenge_scalars(1, "c")
            .add_scalars(repetitions, "z")
    }
}

pub struct SchnorrProof<'a, G: CurveGroup> {
    //Commitment
    pub cm_x: &'a Vec<G::Affine>,

    //Witnesses
    pub r_x: Option<&'a Vec<G::ScalarField>>,
}

impl<'a, G: CurveGroup> SchnorrProof<'a, G> {
    pub fn prove(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        let a_r = vec![G::ScalarField::rand(&mut prover_state.rng()); self.cm_x.len()];
        let a: Vec<G::Affine> = a_r
            .clone()
            .into_iter()
            .map(|x| client_schemes.vector_commit.get_h().mul(x).into_affine())
            .collect();

        //Add the random a's to the domain separator
        let a_points: Vec<G> = a.into_iter().map(|x| x.into_group()).collect();
        prover_state.add_points(&a_points)?;

        let [c]: [G::ScalarField; 1] = prover_state.challenge_scalars()?;

        let r_x = self.r_x.ok_or_else(|| ProofError::SerializationError)?;
        let z: Vec<G::ScalarField> = a_r
            .clone()
            .into_iter()
            .zip(r_x.iter())
            .map(|(x, y)| c * y + x)
            .collect();
        prover_state.add_scalars(&z)?;
        Ok(())
    }

    pub fn verify(
        &self,
        verifier_state: &mut VerifierState,
        client_schemes: &ClientSchemes<'a, G>,
        dimensions: usize,
    ) -> ProofResult<()> {
        let mut a = vec![G::ZERO; dimensions];
        verifier_state.fill_next_points(&mut a)?;

        let [c]: [G::ScalarField; 1] = verifier_state.challenge_scalars()?;
        let mut z = vec![G::ScalarField::default(); dimensions];
        verifier_state.fill_next_scalars(&mut z)?;

        let h = client_schemes.vector_commit.get_h();
        let lhs: Vec<_> = z
            .into_iter()
            .map(|x| h.mul(G::ScalarField::from(x)))
            .collect();

        let rhs: Vec<_> = a
            .into_iter()
            .zip(self.cm_x.iter())
            .map(|(x, y)| x + y.mul(&c))
            .collect();

        if lhs != rhs {
            return Err(ProofError::InvalidProof);
        }
        Ok(())
    }
}
