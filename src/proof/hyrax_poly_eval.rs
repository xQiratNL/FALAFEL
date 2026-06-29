use crate::client::ClientSchemes;
use crate::crypto_schemes::{HyraxCommitment, HyraxCommitmentOpening};
use crate::prelude::*;
use crate::proof::dot_product::{DotProductProof, DotProductProofSeparator};
use crate::proof::util::{dot, row_mult_matrix, tensor_chi};
use ark_std::{UniformRand, log2};
use spongefish::codecs::arkworks_algebra::{
    FieldDomainSeparator, GroupDomainSeparator, GroupToUnitDeserialize, GroupToUnitSerialize,
};
use spongefish::{DomainSeparator, DuplexSpongeInterface, ProofResult, ProverState, VerifierState};

pub trait HyraxPolyEvalProofSeparator<G: CurveGroup> {
    fn add_hyrax_target_commitment(self) -> Self;
    fn add_hyrax_poly_eval_proof(self, dim: usize) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> HyraxPolyEvalProofSeparator<G> for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    // this described only the target commitment part of the point evaluation on a hyrax poly commitment
    fn add_hyrax_target_commitment(self) -> Self {
        // we assume that the hyrax polynomial commitment + the random points have already been sent
        // the polynomial commitment should have been sent all the way in the beginning, and the random points during some sum-check proof
        self.add_points(1, "target commitment (evaluation result)")
    }

    // this describes the dot product part of the point evaluation on a hyrax polynomial commitment,
    // dim equals the value of row_com_dimension
    fn add_hyrax_poly_eval_proof(self, dim: usize) -> Self {
        // we assume that the hyrax polynomial commitment + the random points have already been sent
        // we assume that the target commitment (evaluation result) has already been sent
        self.add_dot_product(dim)
    }
}

// The hyrax structs and implementations below are very loosely inspired by the files in ark-poly-0.5.0/src/hyrax/
// Overall we follow Section 6.1 "Square-root commitment scheme" from https://eprint.iacr.org/2017/1132.pdf

pub struct HyraxPolyEvalProof<'a, G: CurveGroup> {
    pub points: &'a [G::ScalarField],
    pub commitment_opening_input_matrix: Option<&'a Vec<Vec<G::ScalarField>>>, // witness
}

pub struct HyraxTargetCommitment<G: CurveGroup> {
    // target commitment to L \cdot T \cdot R
    pub omega: Option<G::Affine>,
    pub ltr: Option<G::ScalarField>,
    pub r_omega: Option<G::ScalarField>,

    // intermediate vector L \cdot T
    pub lt: Option<Vec<G::ScalarField>>,

    // L vector and R vector
    pub l: Option<Vec<G::ScalarField>>,
    pub r: Option<Vec<G::ScalarField>>,

    // in some situations we also need the points, so it's good to store them
    pub points: Option<Vec<G::ScalarField>>,
}

impl<G: CurveGroup> HyraxTargetCommitment<G> {
    pub fn eval_prove<'a>(
        &self,
        commitment_opening: &HyraxCommitmentOpening<G>,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
        recompute_lt: bool,
    ) -> ProofResult<()> {
        let r = self.r.as_ref().expect("Should exist");

        // 1-3: target commitment was already computed and set

        // postponed part of 2: Compute T' and the randomness part of its opening r_lt.
        let l = self.l.as_ref().expect("Should exist");
        let r_lt = dot(l, &commitment_opening.randomness);

        // 4: dot product proof
        // note: dim is a power of two, so g_vec and a_vec will have the correct length
        if recompute_lt {
            let lt = row_mult_matrix(&l, &commitment_opening.input_matrix);
            DotProductProof {
                g_vec: &client_schemes.vector_commit.get_g_vec()[..r.len()],
                a_vec: r,
                xi: None,
                r_xi: Some(&r_lt),
                tau: None,
                r_tau: Some(self.r_omega.as_ref().expect("Should exist")),
                x_vec: Some(&lt),
            }
            .prove(prover_state, client_schemes)?;
        } else {
            DotProductProof {
                g_vec: &client_schemes.vector_commit.get_g_vec()[..r.len()],
                a_vec: r,
                xi: None,
                r_xi: Some(&r_lt),
                tau: None,
                r_tau: Some(self.r_omega.as_ref().expect("Should exist")),
                x_vec: Some(self.lt.as_ref().expect("Should exist")),
            }
            .prove(prover_state, client_schemes)?;
        }

        Ok(())
    }

    pub fn eval_verify<'a>(
        &self,
        commitment: &HyraxCommitment<G>,
        verifier_state: &mut VerifierState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        // 1-3: target commitment was already received

        // the numbered items don't correspond to the paper or anything, but help to structure the prover/verifier
        if let (Some(l), Some(r)) = (self.l.as_ref(), self.r.as_ref()) {
            // 2: Compute T'
            let t_prime = G::msm_unchecked(&commitment.row_coms, &l).into_affine();

            // 4: dot product verification
            DotProductProof {
                g_vec: &client_schemes.vector_commit.get_g_vec()[..r.len()],
                a_vec: &r,
                xi: Some(&t_prime),
                r_xi: None,
                tau: Some(self.omega.as_ref().expect("This should exist")),
                r_tau: None,
                x_vec: None,
            }
            .verify(verifier_state, client_schemes)?;
        } else {
            let log2_l_len = log2(commitment.row_coms.len()) as usize;
            let points = self.points.as_ref().expect("This should exist");

            // 1: compute L and R vectors
            // note: we can use tensor_chi for R, since we consider only the most significant half of the bits for chi there, which equals the ones used in tensor_chi
            let l = tensor_chi(&points[..log2_l_len]);
            let r = tensor_chi(&points[log2_l_len..]);

            // 2: Compute T'
            let t_prime = G::msm_unchecked(&commitment.row_coms, &l).into_affine();

            // 4: dot product verification
            DotProductProof {
                g_vec: &client_schemes.vector_commit.get_g_vec()[..r.len()],
                a_vec: &r,
                xi: Some(&t_prime),
                r_xi: None,
                tau: Some(self.omega.as_ref().expect("This should exist")),
                r_tau: None,
                x_vec: None,
            }
            .verify(verifier_state, client_schemes)?;
        }

        Ok(())
    }
}

impl<'a, G: CurveGroup> HyraxPolyEvalProof<'a, G> {
    pub fn target_commitment_prove(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<HyraxTargetCommitment<G>> {
        let commitment_opening_input_matrix = self
            .commitment_opening_input_matrix
            .ok_or_else(|| spongefish::ProofError::SerializationError)?;
        let log2_l_len = log2(commitment_opening_input_matrix.len()) as usize;

        // the numbered items don't correspond to the paper or anything, but help to structure the prover/verifier
        // 1: compute L and R vectors
        // note: we can use tensor_chi for R, since we consider only the most significant half of the bits for chi there, which equals the ones used in tensor_chi
        let l = tensor_chi(&self.points[..log2_l_len]);
        let r = tensor_chi(&self.points[log2_l_len..]);

        // 2: Compute lt = L \cdot T (we postpone the evaluation of  r_lt and T' ( commitment with opening (lt, r_lt) ) until later.
        let lt = row_mult_matrix(&l, &commitment_opening_input_matrix);

        // 3: Compute omega, i.e., the commitment to eval:=L \cdot T \cdot R (the evaluation of this polynomial at the random points).
        let eval = dot(&lt, &r);
        let r_omega = G::ScalarField::rand(prover_state.rng());
        let omega = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&eval, &r_omega);
        prover_state.add_points(&[omega.into_group()])?;

        Ok(HyraxTargetCommitment {
            omega: Some(omega),
            ltr: Some(eval),
            r_omega: Some(r_omega),
            lt: Some(lt),
            l: Some(l),
            r: Some(r),
            points: Some(self.points.to_vec()),
        })
    }

    pub fn target_commitment_verify(
        &self,
        verifier_state: &mut VerifierState,
    ) -> ProofResult<HyraxTargetCommitment<G>> {
        // 1-2: we postpone step 1 and 2 until later

        // 3: get omega (target commitment)
        let [omega]: [_; 1] = verifier_state.next_points()?.map(|cm: G| cm.into_affine());

        Ok(HyraxTargetCommitment {
            omega: Some(omega),
            ltr: None,
            r_omega: None,
            lt: None,
            l: None,
            r: None,
            points: Some(self.points.to_vec()),
        })
    }
}

#[cfg(test)]
mod test {
    use crate::client::ClientSchemes;
    use crate::crypto_schemes::Generators;
    use crate::proof::hyrax_poly_eval::{HyraxPolyEvalProof, HyraxPolyEvalProofSeparator};
    use ark_ec::PrimeGroup;
    use ark_std::{UniformRand, log2};
    use rand_chacha::ChaChaRng;
    use rand_chacha::rand_core::SeedableRng;
    use spongefish::{DefaultHash, DomainSeparator};
    use std::cmp::max;

    type G = ark_curve25519::EdwardsProjective;

    // this function does 10 random hyrax poly commits on randomly (each time) generated vectors and
    // evaluation points for the polynomial.
    // then it performs a proof and runs the verifier to see whether the proof verifies (this does
    // not guarantee that the polynomial evaluation is correct, but does make it highly likely!,
    // as then both prover and verifier would contain the same mistakes)
    fn hyrax_poly_commit_prover_verifier(vec_len: usize) {
        // schemes setup
        let mut crypto_rng = ChaChaRng::from_entropy();
        let generators = Generators::<G>::new(1 << max(log2(vec_len), 1).next_multiple_of(2) / 2);
        let schemes = ClientSchemes::setup(&generators);

        // just do 10 random runs
        for _ in 0..10 {
            // get random values to commit to and commit
            let vector = (0..vec_len)
                .map(|_| <G as PrimeGroup>::ScalarField::rand(&mut crypto_rng))
                .collect::<Vec<_>>();

            // prep the prover
            let (cm, cm_open) = schemes
                .poly_commit
                .poly_commit_vec(&vector, &mut crypto_rng);
            let domain_separator = DomainSeparator::<DefaultHash>::new("test");
            let domain_separator =
                <DomainSeparator as HyraxPolyEvalProofSeparator<G>>::add_hyrax_target_commitment(
                    domain_separator,
                );
            let domain_separator =
                <DomainSeparator as HyraxPolyEvalProofSeparator<G>>::add_hyrax_poly_eval_proof(
                    domain_separator,
                    cm_open.input_matrix[0].len(),
                );

            // generate the random points for evaluation
            let eval_points = (0..log2(cm_open.input_matrix.len() * cm_open.input_matrix[0].len()))
                .map(|_| <G as PrimeGroup>::ScalarField::rand(&mut crypto_rng))
                .collect::<Vec<_>>();

            // create the proof
            let mut prover_state = domain_separator.to_prover_state();
            let target_commitment = HyraxPolyEvalProof {
                points: &eval_points,
                commitment_opening_input_matrix: Some(&cm_open.input_matrix),
            }
            .target_commitment_prove(&mut prover_state, &schemes)
            .unwrap();
            target_commitment
                .eval_prove(&cm_open, &mut prover_state, &schemes, false)
                .unwrap();
            let proof = prover_state.narg_string();

            // verify
            let mut verifier_state = domain_separator.to_verifier_state(proof);
            let target_commitment = HyraxPolyEvalProof {
                points: &eval_points,
                commitment_opening_input_matrix: None,
            }
            .target_commitment_verify(&mut verifier_state)
            .unwrap();

            target_commitment
                .eval_verify(&cm, &mut verifier_state, &schemes)
                .unwrap();
        }
    }

    #[test]
    fn hyrax_poly_commit_prover_verifier_non_double_two_power() {
        // Here we test for commitments to vectors of sizes the scheme was not defined for (i.e., not size 2^(2x), for some positive integer x )
        // specifically we check for size 2^x for odd x and sizes that are no two power at all (either a bit more, a bit less, or somewhere in between)
        // Since we keep zero-padding implicit, these tests are important to check we use this implicit padding
        // properly.

        // these are two power, but not double
        hyrax_poly_commit_prover_verifier(1);
        hyrax_poly_commit_prover_verifier(2);
        hyrax_poly_commit_prover_verifier(8);
        hyrax_poly_commit_prover_verifier(32);

        // these are close to double two powers
        hyrax_poly_commit_prover_verifier(7);
        hyrax_poly_commit_prover_verifier(9);
        hyrax_poly_commit_prover_verifier(15);
        hyrax_poly_commit_prover_verifier(16);

        // arbitrary
        hyrax_poly_commit_prover_verifier(20);
        hyrax_poly_commit_prover_verifier(89);
        hyrax_poly_commit_prover_verifier(103);
    }

    #[test]
    fn hyrax_poly_commit_prover_verifier_double_two_power() {
        // this test checks the scheme for vector sizes that are double powers of two, e.g. 2^2x for some integral x.
        // The scheme in the paper is defined for this, so no padding is needed.
        hyrax_poly_commit_prover_verifier(4);
        hyrax_poly_commit_prover_verifier(16);
        hyrax_poly_commit_prover_verifier(64);
    }
}
