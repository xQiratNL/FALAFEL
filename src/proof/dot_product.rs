use crate::client::ClientSchemes;
use crate::prelude::*;
use crate::proof::util::{dot, fold_generators, fold_scalars, scalar_times_slice};
use ark_ff::{AdditiveGroup, Field, batch_inversion};
use ark_std::{UniformRand, log2};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::ProofError;
use spongefish::codecs::arkworks_algebra::{
    FieldDomainSeparator, FieldToUnitDeserialize, FieldToUnitSerialize, GroupDomainSeparator,
    GroupToUnitDeserialize, GroupToUnitSerialize, UnitToField,
};
use spongefish::{DomainSeparator, DuplexSpongeInterface, ProofResult, ProverState, VerifierState};
use std::borrow::Cow;
use std::cmp::min;
use std::ops::Mul;

pub trait DotProductProofSeparator<G: CurveGroup> {
    fn add_dot_product(self, n: usize) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> DotProductProofSeparator<G> for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    // this describes the dot product protocol, i.e., verifier/prover messages
    fn add_dot_product(mut self, n: usize) -> Self {
        // we assume the statement (xi, tau, a_vec) has already been added
        for round in 0..log2(n) {
            self = self
                .add_points(2, &format!("Round {round} messages"))
                .challenge_scalars(1, &format!("Round {round} challenge"));
        }
        self.add_points(2, "commitment for dot product")
            .challenge_scalars(1, "challenge for dot product")
            .add_scalars(2, "response for dot product")
    }
}

pub struct DotProductProof<'a, G: CurveGroup> {
    pub g_vec: &'a [G::Affine],
    pub a_vec: &'a [G::ScalarField],
    pub xi: Option<&'a G::Affine>,           // verifier only
    pub r_xi: Option<&'a G::ScalarField>,    // witness
    pub tau: Option<&'a G::Affine>,          // verifier only
    pub r_tau: Option<&'a G::ScalarField>,   // witness
    pub x_vec: Option<&'a [G::ScalarField]>, // witness
}

impl<'a, G: CurveGroup> DotProductProof<'a, G> {
    // based on https://eprint.iacr.org/2017/1132.pdf Figure 7
    // note: a_vec and g_vec should have equal length and that length should be a power of 2, x_vec may be shorter (we just pretend zero padding then)
    // note 2: we assume that the statement (xi, tau, a_vec) has already been sent to the verifier
    pub fn prove(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        let r_tau = self.r_tau.ok_or_else(|| ProofError::SerializationError)?;
        let r_xi = self.r_xi.ok_or_else(|| ProofError::SerializationError)?;
        let x_vec = self.x_vec.ok_or_else(|| ProofError::SerializationError)?;

        // 1
        let reduced_values = BulletReduce {
            g_vec: Cow::from(self.g_vec),
            a_vec: Cow::from(self.a_vec),
            gamma: None,
            r_gamma: Some(*r_tau + r_xi),
            x_vec: Some(Cow::from(x_vec)),
        }
        .reduce(prover_state, client_schemes)?;

        // 2
        let d = G::ScalarField::rand(prover_state.rng());
        let r_delta = G::ScalarField::rand(prover_state.rng());
        let r_beta = G::ScalarField::rand(prover_state.rng());
        let delta = reduced_values.g_hat.mul(d) + client_schemes.vector_commit.get_h().mul(r_delta);
        let beta = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&d, &r_beta)
            .into_group();
        prover_state.add_points(&[delta, beta])?;

        // 3
        let [c]: [G::ScalarField; 1] = prover_state.challenge_scalars()?;

        // 4
        let z_1 = d + c * reduced_values.y_hat;
        let z_2 = reduced_values.a_hat * (c * reduced_values.r_gamma_hat + r_beta) + r_delta;
        prover_state.add_scalars(&[z_1, z_2])?;

        // 5 is verifier only

        Ok(())
    }

    pub fn verify(
        &self,
        verifier_state: &mut VerifierState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        // 1
        let reduced_values = BulletReduce::<G> {
            g_vec: Cow::from(self.g_vec),
            a_vec: Cow::from(self.a_vec),
            gamma: Some(
                (*self.tau.expect("This should exist") + self.xi.expect("This should exist"))
                    .into_affine(),
            ),
            r_gamma: None,
            x_vec: None,
        }
        .verifier_reduce(verifier_state)?;

        // 2
        let [delta, beta]: [_; 2] = verifier_state.next_points()?.map(|cm: G| cm.into_affine());

        // 3
        let [c]: [G::ScalarField; 1] = verifier_state.challenge_scalars()?;

        // 4
        let [z_1, z_2]: [G::ScalarField; 2] = verifier_state.next_scalars()?;

        // 5
        // grab the relevant generators directly, since we compute custom 'commitments'
        let g = client_schemes.vector_commit.get_g_unique();
        let h = client_schemes.vector_commit.get_h();

        // verification check
        let lhs: G = (reduced_values.gamma_hat.mul(c) + beta).mul(reduced_values.a_hat) + delta;
        let rhs: G = (reduced_values.g_hat + g.mul(reduced_values.a_hat)).mul(z_1) + h.mul(z_2);
        if lhs != rhs {
            return Err(ProofError::InvalidProof);
        }
        Ok(())
    }
}

// note: we use Cow's here, since for the first iteration the inputs will be references, but all other iterations will have owned values
// this way it is most code/memory efficient JUST MAKE SURE WE USE ALL THE COWS BY REFERENCE AND NOT CLONE THEM
#[derive(Debug)]
pub struct BulletReduce<'a, G: CurveGroup> {
    pub g_vec: Cow<'a, [G::Affine]>,
    pub a_vec: Cow<'a, [G::ScalarField]>,
    pub gamma: Option<G::Affine>,
    pub r_gamma: Option<G::ScalarField>,
    pub x_vec: Option<Cow<'a, [G::ScalarField]>>,
}

#[derive(Debug)]
pub struct BulletReduceOutput<G: CurveGroup> {
    pub g_hat: G::Affine,
    pub a_hat: G::ScalarField,
    pub r_gamma_hat: G::ScalarField,
    pub y_hat: G::ScalarField,
}

#[derive(Debug)]
pub struct BulletReduceVerifierOutput<G: CurveGroup> {
    pub g_hat: G::Affine,
    pub a_hat: G::ScalarField,
    pub gamma_hat: G::Affine,
}

impl<'a, G: CurveGroup> BulletReduce<'a, G> {
    // this is the bullet reduce step of the proof of dot product
    // based on: https://eprint.iacr.org/2017/1132.pdf Figure 8 (minor adaptations made)
    // note: the length of a and g should be the same and a power of 2
    pub fn reduce(
        &mut self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<BulletReduceOutput<G>> {
        // grab the relevant generators directly, since we compute custom 'commitments'
        let g = client_schemes.vector_commit.get_g_unique();
        let h = client_schemes.vector_commit.get_h();

        loop {
            let n = self.a_vec.len();
            let r_gamma = self.r_gamma.ok_or_else(|| ProofError::SerializationError)?;
            let x_vec = self
                .x_vec
                .as_ref()
                .ok_or_else(|| ProofError::SerializationError)?;

            // 1
            if n == 1 {
                return Ok(BulletReduceOutput {
                    g_hat: self.g_vec[0],
                    a_hat: self.a_vec[0],
                    r_gamma_hat: r_gamma,
                    y_hat: self.a_vec[0] * x_vec[0],
                });
            }

            // 2
            let r_gamma_left = G::ScalarField::rand(prover_state.rng());
            let r_gamma_right = G::ScalarField::rand(prover_state.rng());

            // we take the min of n/2 and x_vec in case x_vec is short (since we do not store zero values)
            let x_vec_bound = min(n / 2, x_vec.len());
            let gamma_left = h.mul(r_gamma_left)
                + g.mul(dot(&x_vec[..x_vec_bound], &self.a_vec[n / 2..]))
                + G::msm_unchecked(&self.g_vec[n / 2..], &x_vec[..x_vec_bound]);
            // we set a lot to zero if x_vec smaller than or equal to n/2 (since we do not store zero values)
            let gamma_right = if x_vec.len() > n / 2 {
                h.mul(r_gamma_right)
                    + g.mul(dot(&x_vec[n / 2..], &self.a_vec[..n / 2]))
                    + G::msm_unchecked(&self.g_vec[..n / 2], &x_vec[n / 2..])
            } else {
                h.mul(r_gamma_right)
            };

            prover_state.add_points(&[gamma_left, gamma_right])?;

            // 3
            let [c]: [G::ScalarField; 1] = prover_state.challenge_scalars()?;
            let c_inv = c.inverse().expect("Challenge may not be zero!");

            // 4 + 5
            self.a_vec = Cow::from(fold_scalars(
                &self.a_vec[..n / 2],
                &self.a_vec[n / 2..],
                &c_inv,
                &c,
            ));
            self.g_vec = Cow::from(fold_generators::<G>(
                &self.g_vec[..n / 2],
                &self.g_vec[n / 2..],
                &c_inv,
                &c,
            ));

            // folding should be different if x_vec is small (note: we don't store zeroes)
            self.x_vec = if x_vec.len() >= n / 2 {
                Some(Cow::from(fold_scalars(
                    &x_vec[..n / 2],
                    &x_vec[n / 2..],
                    &c,
                    &c_inv,
                )))
            } else {
                Some(Cow::from(scalar_times_slice(x_vec, &c)))
            };
            // note: no need to compute y' so I'm skipping that
            self.r_gamma =
                Some(r_gamma_left * c.square() + r_gamma + r_gamma_right * c_inv.square());
        }
    }

    pub fn verifier_reduce(
        &mut self,
        verifier_state: &mut VerifierState,
    ) -> ProofResult<BulletReduceVerifierOutput<G>> {
        let mut gamma_l_vec = vec![];
        let mut gamma_r_vec = vec![];
        let mut c_vec = vec![];

        // set useful variables
        let n = self.a_vec.len();
        for _ in 0..log2(n) {
            // 2
            let [gamma_left, gamma_right]: [_; 2] =
                verifier_state.next_points()?.map(|cm: G| cm.into_affine());

            gamma_l_vec.push(gamma_left);
            gamma_r_vec.push(gamma_right);

            // 3
            let [c]: [G::ScalarField; 1] = verifier_state.challenge_scalars()?;

            c_vec.push(c);

            // 4 + 5
            // postponed
        }

        // SINGLE MSM TRICK VERSION of 1
        // compute c_inv_vec and s_0
        let mut c_inv_vec = c_vec.clone();
        batch_inversion(&mut c_inv_vec);
        let s_0 = ark_std::cfg_iter!(c_inv_vec).product::<G::ScalarField>();

        // compute c_sq_vec
        ark_std::cfg_iter_mut!(c_vec).for_each(|c| {
            c.square_in_place();
        });
        let c_sq_vec = c_vec;

        // compute s
        let mut s = Vec::with_capacity(n);
        s.push(s_0);
        (1..=log2(n)).for_each(|i_log_ub| {
            let i_from = 1 << (i_log_ub - 1);
            let i_to = 1 << i_log_ub;
            let mut s_chunk = vec![G::ScalarField::ZERO; i_to - i_from];
            ark_std::cfg_into_iter!((i_from..i_to))
                .zip(ark_std::cfg_iter_mut!(s_chunk))
                .for_each(|(i, s_val)| {
                    let k_exp = (64 - 1 - (i as u64).leading_zeros()) as usize;
                    let k = 1 << k_exp;
                    *s_val = s[i - k] * c_sq_vec[c_sq_vec.len() - 1 - k_exp];
                });
            s.append(&mut s_chunk);
        });

        // compute neg_x_inv_sq_vec
        ark_std::cfg_iter_mut!(c_inv_vec).for_each(|c_inv| {
            c_inv.square_in_place();
        });
        let c_inv_sq_vec = c_inv_vec;

        Ok(BulletReduceVerifierOutput {
            g_hat: G::msm_unchecked(&self.g_vec, &s).into_affine(),
            a_hat: dot(&self.a_vec, &s),
            gamma_hat: (G::msm_unchecked(
                &[gamma_l_vec.as_slice(), gamma_r_vec.as_slice()].concat(),
                &[c_sq_vec.as_slice(), c_inv_sq_vec.as_slice()].concat(),
            ) + self.gamma.expect("This should exist"))
            .into_affine(),
        })
    }
}
