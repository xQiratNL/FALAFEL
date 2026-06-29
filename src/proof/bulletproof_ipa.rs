use crate::client::ClientSchemes;
use crate::proof::util::{dot, fold_generators, fold_scalars};
use ark_ec::CurveGroup;
use ark_ff::{AdditiveGroup, Field, batch_inversion};
use ark_std::log2;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::codecs::arkworks_algebra::{
    FieldDomainSeparator, FieldToUnitDeserialize, FieldToUnitSerialize, GroupDomainSeparator,
    GroupToUnitDeserialize, GroupToUnitSerialize, UnitToField,
};
use spongefish::{
    DomainSeparator, DuplexSpongeInterface, ProofError, ProofResult, ProverState, VerifierState,
};
use std::borrow::Cow;
use std::cmp::min;
use std::ops::Mul;
use std::slice::from_ref;
use std::vec;

pub trait BulletproofInnerProductArgumentSeparator<G: CurveGroup> {
    fn add_bulletproof_ipa(self, n: usize) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> BulletproofInnerProductArgumentSeparator<G>
    for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    fn add_bulletproof_ipa(mut self, n: usize) -> Self {
        // protocol 1
        self = self.challenge_scalars(1, "Protocol 1: x");
        // protocol 2
        for round in 0..log2(n) {
            self = self
                .add_points(2, &format!("Protocol 2: Round {round} points L, R"))
                .challenge_scalars(1, &format!("Protocol 2: Round {round} challenge x"));
        }
        self.add_scalars(2, "Final round scalars a, b")
    }
}

pub struct BulletproofInnerProductArgument<'a, G: CurveGroup> {
    pub g_vec: &'a [G::Affine],
    pub h_vec: &'a [G::Affine],
    pub p: &'a G::Affine,
    pub c: &'a G::ScalarField,
    pub a_vec: Option<&'a [G::ScalarField]>,
    pub b_vec: Option<&'a [G::ScalarField]>,
}

impl<'a, G: CurveGroup> BulletproofInnerProductArgument<'a, G> {
    // based on https://eprint.iacr.org/2017/1066.pdf Protocol 1
    // note: the lengths of g_vec and h_vec should be the same and a power of 2, a and b vectors may have a shorter (non-zero) length than g_vec and h_vec, we assume implicit zero-padding then
    pub fn prove(
        &mut self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        // get u generator
        let u = client_schemes.vector_commit.get_g_unique();

        // 6, 7
        let [x]: [G::ScalarField; 1] = prover_state.challenge_scalars()?;

        // 8
        let p_prime = *self.p + u.mul(x * self.c);

        // 9
        BulletproofRecursiveArgument::<G> {
            g_vec: Cow::from(self.g_vec),
            h_vec: Cow::from(self.h_vec),
            u: u.mul(x).into_affine(),
            p: p_prime.into_affine(),
            a_vec: self.a_vec.map(|a| Cow::from(a)),
            b_vec: self.b_vec.map(|b| Cow::from(b)),
        }
        .prove(prover_state)
    }

    pub fn verify(
        &mut self,
        verifier_state: &mut VerifierState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        // get u generator
        let u = client_schemes.vector_commit.get_g_unique();

        // 6, 7
        let [x]: [G::ScalarField; 1] = verifier_state.challenge_scalars()?;

        // 8
        let p_prime = *self.p + u.mul(x * self.c);

        // 9
        BulletproofRecursiveArgument::<G> {
            g_vec: Cow::from(self.g_vec),
            h_vec: Cow::from(self.h_vec),
            u: u.mul(x).into_affine(),
            p: p_prime.into_affine(),
            a_vec: None,
            b_vec: None,
        }
        .verify(verifier_state)
    }
}

pub struct BulletproofRecursiveArgument<'a, G: CurveGroup> {
    pub g_vec: Cow<'a, [G::Affine]>,
    pub h_vec: Cow<'a, [G::Affine]>,
    pub u: G::Affine,
    pub p: G::Affine,
    pub a_vec: Option<Cow<'a, [G::ScalarField]>>,
    pub b_vec: Option<Cow<'a, [G::ScalarField]>>,
}

impl<'a, G: CurveGroup> BulletproofRecursiveArgument<'a, G> {
    // based on https://eprint.iacr.org/2017/1066.pdf Protocol 2
    // note: the lengths of g_vec and h_vec should be the same and a power of 2, a and b vectors may have a shorter (non-zero) length than g_vec and h_vec, we assume implicit zero-padding then
    pub fn prove(&mut self, prover_state: &mut ProverState) -> ProofResult<()> {
        loop {
            // set useful variables
            let n = self.g_vec.len();
            let a_vec = self
                .a_vec
                .as_ref()
                .ok_or_else(|| ProofError::SerializationError)?;
            let b_vec = self
                .b_vec
                .as_ref()
                .ok_or_else(|| ProofError::SerializationError)?;

            // 14
            if n == 1 {
                // 15
                prover_state.add_scalars(&[a_vec[0], b_vec[0]])?;

                // 16, 17 is verifier only
                return Ok(());
            }

            // 18 (n>1)
            // 19
            // 20
            let n_prime = n >> 1;
            // 21
            let c_l = dot(&a_vec[..n_prime], &b_vec[n_prime..]);
            // 22
            let c_r = dot(&a_vec[n_prime..], &b_vec[..n_prime]);
            // 23
            let l = G::msm_unchecked(
                &[
                    &self.g_vec[n_prime..n_prime + min(n_prime, a_vec.len())],
                    &self.h_vec[..min(n_prime, b_vec.len() - n_prime)],
                    from_ref(&self.u),
                ]
                .concat(),
                &[&a_vec[..n_prime], &b_vec[n_prime..], from_ref(&c_l)].concat(),
            );

            // 24
            let r = G::msm_unchecked(
                &[
                    &self.g_vec[..min(n_prime, a_vec.len() - n_prime)],
                    &self.h_vec[n_prime..n_prime + min(n_prime, b_vec.len())],
                    from_ref(&self.u),
                ]
                .concat(),
                &[&a_vec[n_prime..], &b_vec[..n_prime], from_ref(&c_r)].concat(),
            );
            // 25
            prover_state.add_points(&[l, r])?;

            // 26, 27
            let [x]: [G::ScalarField; 1] = prover_state.challenge_scalars()?;
            let x_inv = x.inverse().expect("Challenge may not be zero!");

            // 28
            // 29
            self.g_vec = Cow::from(fold_generators::<G>(
                &self.g_vec[..n_prime],
                &self.g_vec[n_prime..],
                &x_inv,
                &x,
            ));
            // 30
            self.h_vec = Cow::from(fold_generators::<G>(
                &self.h_vec[..n_prime],
                &self.h_vec[n_prime..],
                &x,
                &x_inv,
            ));
            // 31
            self.p = (l.mul(x.square()) + self.p + r.mul(x_inv.square())).into_affine();

            // 32
            // 33
            self.a_vec = Some(Cow::from(fold_scalars(
                &a_vec[..n_prime],
                &a_vec[n_prime..],
                &x,
                &x_inv,
            )));
            // 34
            self.b_vec = Some(Cow::from(fold_scalars(
                &b_vec[..n_prime],
                &b_vec[n_prime..],
                &x_inv,
                &x,
            )));

            // 35: we loop back
        }
    }

    pub fn verify(&mut self, verifier_state: &mut VerifierState) -> ProofResult<()> {
        let mut l_vec = vec![];
        let mut r_vec = vec![];
        let mut x_vec = vec![];

        // set useful variables
        let n = self.g_vec.len();
        for _ in 0..log2(n) {
            // 18 (n>1)
            // 19-24 is prover only (except computing n_prime (step 20))

            // 25
            let [l, r]: [G; 2] = verifier_state.next_points()?;

            l_vec.push(l);
            r_vec.push(r);

            // 26, 27
            let [x]: [G::ScalarField; 1] = verifier_state.challenge_scalars()?;

            // note: this vector is still x_vec here
            x_vec.push(x);

            // 28
            // 29
            // postponed
            // 30
            // postponed
            // 31
            // postponed

            // 32-34 is prover only

            // 35: we loop back
        }
        // 14
        // 15
        let [a, b]: [G::ScalarField; 2] = verifier_state.next_scalars()?;

        // SINGLE MSM TRICK VERSION of 16, 17
        // compute x_inv_vec and s_0
        let mut x_inv_vec = x_vec.clone();
        batch_inversion(&mut x_inv_vec);
        let s_0 = ark_std::cfg_iter!(x_inv_vec).product::<G::ScalarField>();

        // compute x_sq_vec
        ark_std::cfg_iter_mut!(x_vec).for_each(|x| {
            x.square_in_place();
        });
        let mut x_sq_vec = x_vec;

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
                    *s_val = s[i - k] * x_sq_vec[x_sq_vec.len() - 1 - k_exp];
                });
            s.append(&mut s_chunk);
        });

        // compute b_s_inv
        // s_inv is the same as s.rev()
        let b_s_inv = ark_std::cfg_iter!(s)
            .rev()
            .map(|s_val| b * s_val)
            .collect::<Vec<_>>();

        // compute a_s
        ark_std::cfg_iter_mut!(s)
            .map(|s_val| *s_val *= a)
            .collect::<Vec<_>>();
        let a_s = s;

        // compute neg_x_sq_vec
        ark_std::cfg_iter_mut!(x_sq_vec).for_each(|x_sq| {
            x_sq.neg_in_place();
        });
        let neg_x_sq_vec = x_sq_vec;

        // compute neg_x_inv_sq_vec
        ark_std::cfg_iter_mut!(x_inv_vec).for_each(|x_inv| {
            let x_inv_sq = x_inv.square_in_place();
            x_inv_sq.neg_in_place();
        });
        let neg_x_inv_sq_vec = x_inv_vec;

        let l_vec = G::normalize_batch(&l_vec);
        let r_vec = G::normalize_batch(&r_vec);

        if self.p
            == G::msm_unchecked(
                &[&self.g_vec, &self.h_vec, from_ref(&self.u), &l_vec, &r_vec].concat(),
                &[
                    &a_s,
                    &b_s_inv,
                    from_ref(&(a * b)),
                    &neg_x_sq_vec,
                    &neg_x_inv_sq_vec,
                ]
                .concat(),
            )
            .into_affine()
        {
            Ok(())
        } else {
            Err(ProofError::InvalidProof)
        }
    }
}
