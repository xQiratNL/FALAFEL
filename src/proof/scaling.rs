use crate::client::ClientSchemes;
use crate::crypto_schemes::{HyraxCommitment, HyraxCommitmentOpening};
use crate::prelude::{PEDERSEN_DIVISOR_LARGE, PEDERSEN_DIVISOR_SMALL};
use crate::proof::bulletproof_ipa::{
    BulletproofInnerProductArgument, BulletproofInnerProductArgumentSeparator,
};
use crate::proof::dot_product::{DotProductProof, DotProductProofSeparator};
use crate::proof::util::{dot, hadamard, powers};
use ark_ec::{AffineRepr, CurveGroup};
use ark_ff::{AdditiveGroup, BigInteger, Field, PrimeField};
use ark_std::UniformRand;
use ark_std::iterable::Iterable;
use rand::SeedableRng;
use rand_chacha::ChaChaRng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::codecs::arkworks_algebra::{
    FieldDomainSeparator, FieldToUnitDeserialize, FieldToUnitSerialize, GroupDomainSeparator,
    GroupToUnitDeserialize, GroupToUnitSerialize, UnitToField,
};
use spongefish::{
    DomainSeparator, DuplexSpongeInterface, ProofError, ProofResult, ProverState, VerifierState,
};
use std::cmp::{max, min};
use std::ops::Mul;
use std::slice::from_ref;

pub trait ScalingProofSeparator<G: CurveGroup> {
    fn add_scaling(self, m_b: usize, n_a: usize, n_b: usize, q: usize) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> ScalingProofSeparator<G> for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    // note: the proof format is independent of whether we also include scaling bits
    fn add_scaling(mut self, m_b: usize, n_a: usize, n_b: usize, q: usize) -> Self {
        let pedersen_divisor = if PEDERSEN_DIVISOR_LARGE <= m_b {
            PEDERSEN_DIVISOR_LARGE
        } else {
            PEDERSEN_DIVISOR_SMALL
        };
        for rep_idx in 0..pedersen_divisor {
            let m_b = if rep_idx + 1 == pedersen_divisor {
                m_b - (pedersen_divisor - 1)
                    * ((m_b as f64 / pedersen_divisor as f64).ceil() as usize)
            } else {
                (m_b as f64 / pedersen_divisor as f64).ceil() as usize
            };

            let n = max(n_a, n_b);
            let m = n_b * m_b / n;

            self = self
                .add_points(2, "cm_a, cm_s")
                .challenge_scalars(2, "y, z")
                .add_points(1, "cm_v")
                .add_dot_product(n)
                .add_points(2, "cm_t1, cm_t2")
                .challenge_scalars(1, "x")
                .add_scalars(3, "tau_x, mu, t_hat")
                .add_bulletproof_ipa(q * n * m)
        }
        self
    }
}

pub struct ScalingProof<'a, G: CurveGroup> {
    // constants
    pub n_a: usize,
    pub n_b: usize,
    pub a_take: Option<usize>,

    // bit sizes
    pub q: usize,
    pub s: usize,

    // commitments
    pub cm_a_mat_unscaled_all: &'a [HyraxCommitment<G>],
    pub cm_b_mat_unscaled: &'a HyraxCommitment<G>,

    // witnesses
    pub opening_a_mat_unscaled_all: Option<&'a [HyraxCommitmentOpening<G>]>,
    pub opening_b_mat_unscaled: Option<&'a HyraxCommitmentOpening<G>>,

    // optional if we also want to include sign bits
    pub sign_bits_proof: Option<SignBitsProof<'a, G>>,
}

pub struct SignBitsProof<'a, G: CurveGroup> {
    pub cm_c_mat: &'a HyraxCommitment<G>,
    pub opening_c_mat: Option<&'a HyraxCommitmentOpening<G>>,
}

impl<'a, G: CurveGroup> ScalingProof<'a, G> {
    // Algorithm 3 in our paper
    pub fn prove(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        // useful constants
        let q_prime = self.q - 1;
        let m_a = self.cm_a_mat_unscaled_all[0].row_coms.len();
        let m_b = self.cm_b_mat_unscaled.row_coms.len();
        let two_to_the_q_prime_minus_s = G::ScalarField::from(2).pow([(q_prime - self.s) as u64]);
        let two_to_the_q_prime = G::ScalarField::from(2).pow([q_prime as u64]);

        // INTERMEZZO - for a version where a consists of multiple HyraxCommitments and corresponding openings
        let a_take = self.a_take.unwrap_or(m_a);
        let a_chunk_size = a_take.next_power_of_two();
        let n_matrices = a_take * self.cm_a_mat_unscaled_all.len();

        // useful variables
        let opening_a_mat_unscaled_all = self
            .opening_a_mat_unscaled_all
            .ok_or_else(|| ProofError::SerializationError)?;
        let opening_b_mat_unscaled = self
            .opening_b_mat_unscaled
            .ok_or_else(|| ProofError::SerializationError)?;

        let to_pad = n_matrices.next_power_of_two() - n_matrices;
        let pad_vec = vec![G::ScalarField::ZERO; self.n_a];
        let a_input_matrix_combined_filtered_sorted =
            ark_std::cfg_into_iter!((0..m_a / a_chunk_size))
                .flat_map(|chunk_idx| {
                    let base_idx = chunk_idx * a_chunk_size;
                    ark_std::cfg_iter!(opening_a_mat_unscaled_all)
                        .flat_map(move |opening_a_mat_unscaled| {
                            &opening_a_mat_unscaled.input_matrix[base_idx..base_idx + a_take]
                        })
                        .chain(ark_std::cfg_into_iter!(vec![&pad_vec; to_pad]))
                })
                .collect::<Vec<_>>();
        let a_randomness_combined_filtered_sorted =
            ark_std::cfg_into_iter!((0..m_a / a_chunk_size))
                .flat_map(|chunk_idx| {
                    let base_idx = chunk_idx * a_chunk_size;
                    ark_std::cfg_iter!(opening_a_mat_unscaled_all)
                        .flat_map(move |opening_a_mat_unscaled| {
                            ark_std::cfg_iter!(
                                opening_a_mat_unscaled.randomness[base_idx..base_idx + a_take]
                            )
                            .copied()
                        })
                        .chain(ark_std::cfg_into_iter!(vec![G::ScalarField::ZERO; to_pad]))
                })
                .collect::<Vec<_>>();

        // 1: verifier only

        let pedersen_divisor = if PEDERSEN_DIVISOR_LARGE <= m_b {
            PEDERSEN_DIVISOR_LARGE
        } else {
            PEDERSEN_DIVISOR_SMALL
        };
        for rep_idx in 0..pedersen_divisor {
            let rep_len = (m_b as f64 / pedersen_divisor as f64).ceil() as usize;
            let m_b = if rep_idx + 1 == pedersen_divisor {
                m_b - (pedersen_divisor - 1)
                    * ((m_b as f64 / pedersen_divisor as f64).ceil() as usize)
            } else {
                (m_b as f64 / pedersen_divisor as f64).ceil() as usize
            };

            // get the generators (note we get a two power for g_vec and h_vec as this is needed for the IPA, in the rest the latter part is ignored due to zips)
            let num_generators_full = (self.q * self.n_b * m_b).next_power_of_two();
            let g_vec = &client_schemes.vector_commit.get_g_vec()[..num_generators_full];
            let h_vec = &client_schemes.vector_commit.get_g_vec()
                [num_generators_full..2 * num_generators_full];
            let h = client_schemes.vector_commit.get_h();

            // 2
            // compute a_l, a_r, alpha and cm_a
            let mut a_l = vec![false; self.q * self.n_b * m_b];
            let mut a_r_negated = vec![false; self.q * self.n_b * m_b];

            ark_std::cfg_iter!(
                opening_b_mat_unscaled.input_matrix[rep_idx * rep_len..rep_idx * rep_len + m_b]
            )
            .zip(ark_std::cfg_chunks_mut!(a_l, self.q * self.n_b))
            .zip(ark_std::cfg_chunks_mut!(a_r_negated, self.q * self.n_b))
            .for_each(|((row_unscaled, a_l_chunk), a_r_negated_chunk)| {
                row_unscaled
                    .iter()
                    // note: we also make the implicit zeroes explicit here
                    .chain(std::iter::repeat(&G::ScalarField::ZERO))
                    .zip(a_l_chunk.chunks_mut(self.q))
                    .zip(a_r_negated_chunk.chunks_mut(self.q))
                    .for_each(
                        |((val_unscaled, a_l_chunk_chunk), a_r_negated_chunk_chunk)| {
                            // debug_assert!(
                            //     !(two_to_the_q_prime + val_unscaled)
                            //         .into_bigint()
                            //         .to_bits_le()[self.q..]
                            //         .iter()
                            //         .any(|x| *x)
                            // );
                            (two_to_the_q_prime + val_unscaled)
                                .into_bigint()
                                .to_bits_le()
                                .iter()
                                .zip(a_l_chunk_chunk)
                                .zip(a_r_negated_chunk_chunk)
                                .for_each(|((&bit, a_l_entry), a_r_negated_entry)| {
                                    if bit {
                                        *a_l_entry = true;
                                    } else {
                                        *a_r_negated_entry = true;
                                    }
                                })
                        },
                    )
            });
            let alpha = G::ScalarField::rand(prover_state.rng());
            let cm_a = ark_std::cfg_iter!(g_vec)
                .zip(ark_std::cfg_iter!(a_l))
                .filter_map(|(g, &a)| if a { Some(g) } else { None })
                .sum::<G>()
                - ark_std::cfg_iter!(h_vec)
                    .zip(ark_std::cfg_iter!(a_r_negated))
                    .filter_map(|(g, &a_negated)| if a_negated { Some(g) } else { None })
                    .sum::<G>()
                + h.mul(alpha);

            // compute s_l, s_r, rho, and cm_s
            let min_parallel_chunk_size = 1024;
            let num_cpus_available = rayon::current_num_threads();
            let num_elem_per_thread = max(
                self.q * self.n_b * m_b / num_cpus_available,
                min_parallel_chunk_size,
            );
            let n_chunks = ((self.q * self.n_b * m_b - 1) / num_elem_per_thread) + 1;

            let mut rngs = (0..n_chunks)
                .map(|_| ChaChaRng::from_rng(prover_state.rng()).unwrap())
                .collect::<Vec<_>>();

            let mut s_l = vec![G::ScalarField::ZERO; self.q * self.n_b * m_b];
            let mut s_r = vec![G::ScalarField::ZERO; self.q * self.n_b * m_b];
            ark_std::cfg_chunks_mut!(&mut s_l, num_elem_per_thread)
                .zip(ark_std::cfg_iter_mut!(rngs))
                .for_each(|(s_l_chunk, rng)| {
                    for s_l_entry in s_l_chunk {
                        *s_l_entry = G::ScalarField::rand(rng)
                    }
                });
            ark_std::cfg_chunks_mut!(&mut s_r, num_elem_per_thread)
                .zip(ark_std::cfg_iter_mut!(rngs))
                .for_each(|(s_r_chunk, rng)| {
                    for s_r_entry in s_r_chunk {
                        *s_r_entry = G::ScalarField::rand(rng)
                    }
                });

            let rho = G::ScalarField::rand(prover_state.rng());
            let cm_s = G::msm_unchecked(
                &[&g_vec[..s_l.len()], &h_vec[..s_r.len()], from_ref(h)].concat(),
                &[&s_l, &s_r, from_ref(&rho)].concat(),
            );
            prover_state.add_points(&[cm_a, cm_s])?;

            // 3
            let [y, z]: [G::ScalarField; 2] = prover_state.challenge_scalars()?;

            // INTERMEZZO: compute n and m
            let n = max(self.n_a, self.n_b);
            let m = self.n_b * m_b / n;

            // debug_assert_eq!(opening_b_mat_unscaled.input_matrix.len(), m);
            debug_assert_eq!(opening_b_mat_unscaled.input_matrix[0].len(), n);
            // debug_assert_eq!(a_input_matrix_combined_filtered_sorted.len(), m);
            debug_assert_eq!(a_input_matrix_combined_filtered_sorted[0].len(), n);
            // 4
            // compute z powers and other useful stuff
            let inner_z_powers = powers(&z, n, true);
            let z_to_the_n = inner_z_powers[n - 1] * z;
            let outer_z_powers = powers(&z_to_the_n, m, true);
            let z_to_the_nm = outer_z_powers[m - 1] * z_to_the_n;

            let outer_powers_sum = ark_std::cfg_iter!(outer_z_powers).sum::<G::ScalarField>();
            let outer_powers_times_two_to_the_q_prime = outer_powers_sum * two_to_the_q_prime;
            let outer_powers_times_two_to_the_q_prime_minus_s =
                outer_powers_sum * two_to_the_q_prime_minus_s;
            let outer_scaling_scaled_again = outer_powers_times_two_to_the_q_prime
                + z_to_the_nm * outer_powers_times_two_to_the_q_prime_minus_s;

            // compute b part
            let mut x_vec = vec![G::ScalarField::ZERO; n];
            opening_b_mat_unscaled.input_matrix[rep_idx * rep_len..rep_idx * rep_len + m_b]
                .iter()
                .zip(&outer_z_powers)
                .for_each(|(b_row, &outer_z_power)| {
                    ark_std::cfg_iter_mut!(&mut x_vec)
                        .zip(ark_std::cfg_iter!(b_row))
                        .for_each(|(x_val, b_val)| *x_val += outer_z_power * b_val)
                });
            // compute a part
            let outer_z_powers_times_z_to_the_nm = ark_std::cfg_iter!(outer_z_powers)
                .map(|x| z_to_the_nm * x)
                .collect::<Vec<_>>();
            a_input_matrix_combined_filtered_sorted[rep_idx * rep_len..rep_idx * rep_len + m_b]
                .iter()
                .zip(&outer_z_powers_times_z_to_the_nm)
                .for_each(|(a_row, &outer_z_power)| {
                    ark_std::cfg_iter_mut!(&mut x_vec)
                        .zip(ark_std::cfg_iter!(a_row))
                        .for_each(|(x_val, a_val)| *x_val += outer_z_power * a_val)
                });

            ark_std::cfg_iter_mut!(x_vec).for_each(|value| *value += outer_scaling_scaled_again);

            // optional<with scaling>: compute c part and add it to v
            if let Some(sign_bits_proof) = &self.sign_bits_proof {
                let opening_c_mat = sign_bits_proof
                    .opening_c_mat
                    .ok_or_else(|| ProofError::SerializationError)?;
                let z_to_the_2nm = z_to_the_nm.square();

                let outer_z_powers_times_z_to_the_2nm = ark_std::cfg_iter!(outer_z_powers)
                    .map(|x| z_to_the_2nm * x)
                    .collect::<Vec<_>>();
                opening_c_mat.input_matrix[rep_idx * rep_len..rep_idx * rep_len + m_b]
                    .iter()
                    .zip(&outer_z_powers_times_z_to_the_2nm)
                    .for_each(|(c_row, &outer_z_power)| {
                        ark_std::cfg_iter_mut!(&mut x_vec)
                            .zip(ark_std::cfg_iter!(c_row))
                            .for_each(|(x_val, c_val)| *x_val += outer_z_power * c_val)
                    });
            }
            // compute v
            let v = dot(&x_vec, &inner_z_powers);

            // send commitment to v
            let gamma = G::ScalarField::rand(prover_state.rng());
            let cm_v = client_schemes
                .vector_commit
                .value_commit_unique_g_same_h(&v, &gamma);
            prover_state.add_points(&[cm_v.into_group()])?;

            // 5
            // compute the randomness of the combined commitment
            let mut r_xi = dot(
                &opening_b_mat_unscaled.randomness[rep_idx * rep_len..rep_idx * rep_len + m_b],
                &outer_z_powers,
            ) + z_to_the_nm
                * dot(
                    &a_randomness_combined_filtered_sorted
                        [rep_idx * rep_len..rep_idx * rep_len + m_b],
                    &outer_z_powers,
                );
            // optional<with scaling>: add the c commitment and randomness
            if let Some(sign_bits_proof) = &self.sign_bits_proof {
                let opening_c_mat = sign_bits_proof
                    .opening_c_mat
                    .ok_or_else(|| ProofError::SerializationError)?;

                let z_to_the_2nm = z_to_the_nm.square();

                r_xi += z_to_the_2nm
                    * dot(
                        &opening_c_mat.randomness[rep_idx * rep_len..rep_idx * rep_len + m_b],
                        &outer_z_powers,
                    );
            }

            DotProductProof {
                g_vec: &client_schemes.vector_commit.get_g_vec()[..n],
                a_vec: &inner_z_powers,
                xi: None,
                r_xi: Some(&r_xi),
                tau: None,
                r_tau: Some(&gamma),
                x_vec: Some(&x_vec),
            }
            .prove(prover_state, client_schemes)?;

            // debug_assert_eq!(
            //     a_r_negated,
            //     a_l.iter()
            //         .map(|&x| x - G::ScalarField::ONE)
            //         .collect::<Vec<_>>()
            // );
            // debug_assert!(
            //     hadamard(&a_l, &a_r_negated)
            //         .iter()
            //         .all(|&x| x == G::ScalarField::ZERO)
            // );
            // debug_assert_eq!(
            //     {
            //         let two_powers = &powers(&G::ScalarField::from(2), self.q, true);
            //         let mut z_dbg = G::ScalarField::ONE;
            //         a_l.chunks(self.q)
            //             .map(|chunk| {
            //                 let mut result = z_dbg * dot(chunk, &two_powers)
            //                     + z_dbg
            //                         * z_to_the_nm
            //                         * dot(&chunk[self.s..], &two_powers[..self.q - self.s]);
            //                 if self.sign_bits_proof.is_some() {
            //                     result += z_dbg * z_to_the_nm * z_to_the_nm * chunk.last().unwrap()
            //                 }
            //                 z_dbg *= z;
            //                 result
            //             })
            //             .sum::<G::ScalarField>()
            //     },
            //     v
            // );

            // 6
            // we also pre-compute some stuff for step 8
            let y_powers = powers(&y, self.q * n * m, true);
            let z_powers_for_polynomial = &powers(&z, 2 * n * m + 1, false);

            // coefficient vectors of constant
            let l0 = ark_std::cfg_iter!(a_l)
                .map(|&a_l_val| if a_l_val { G::ScalarField::ONE - z } else { -z })
                .collect::<Vec<_>>();
            let r0 = ark_std::cfg_chunks!(y_powers, self.q)
                .zip(ark_std::cfg_chunks!(a_r_negated, self.q))
                .zip(ark_std::cfg_iter!(z_powers_for_polynomial[1..n * m + 1]).cloned())
                .zip(ark_std::cfg_iter!(z_powers_for_polynomial[n * m + 1..]).cloned())
                .map(
                    |(
                        ((y_vals, a_r_negated_vals), mut z_power_first_term),
                        mut z_power_second_term,
                    )| {
                        // optional<with scaling>: z power for the third term
                        let z_power_third_term = if self.sign_bits_proof.is_some() {
                            Some(z_power_second_term * z_to_the_nm)
                        } else {
                            None
                        };
                        y_vals.iter().zip(a_r_negated_vals).enumerate().map(
                            move |(idx, (y_power, &a_r_negated_val))| {
                                let mut result = (if a_r_negated_val {
                                    z - G::ScalarField::ONE
                                } else {
                                    z
                                }) * y_power
                                    + z_power_first_term;
                                z_power_first_term.double_in_place();
                                if idx >= self.s {
                                    result += z_power_second_term;
                                    z_power_second_term.double_in_place();
                                }
                                // optional<with scaling>: add a third term for the MSB only
                                if let Some(z_power_third_term) = z_power_third_term
                                    && idx == q_prime
                                {
                                    result += z_power_third_term;
                                }
                                result
                            },
                        )
                    },
                );
            #[cfg(feature = "parallel")]
            let r0 = r0.flatten_iter().collect::<Vec<_>>();
            #[cfg(not(feature = "parallel"))]
            let r0 = r0.flatten().collect::<Vec<_>>();

            // coefficient vectors of X
            let l1 = s_l;
            let r1 = hadamard(&y_powers, &s_r);

            // compute t1 and t2 and send commitments to both
            let t1 = dot(&l0, &r1) + dot(&l1, &r0);
            let t2 = dot(&l1, &r1);
            let tau1 = G::ScalarField::rand(prover_state.rng());
            let cm_t1 = client_schemes
                .vector_commit
                .value_commit_unique_g_same_h(&t1, &tau1);
            let tau2 = G::ScalarField::rand(prover_state.rng());
            let cm_t2 = client_schemes
                .vector_commit
                .value_commit_unique_g_same_h(&t2, &tau2);
            prover_state.add_points(&[cm_t1.into_group(), cm_t2.into_group()])?;

            // 7
            let [x]: [G::ScalarField; 1] = prover_state.challenge_scalars()?;

            // 8
            let z_squared = z.square();
            let x_squared = x.square();

            // compute l(x) and r(x)
            let l = ark_std::cfg_iter!(l0)
                .zip(ark_std::cfg_iter!(l1))
                .map(|(l0_val, &l1_val)| l1_val * x + l0_val)
                .collect::<Vec<_>>();
            let r = ark_std::cfg_iter!(r0)
                .zip(ark_std::cfg_iter!(r1))
                .map(|(r0_val, &r1_val)| r1_val * x + r0_val)
                .collect::<Vec<_>>();

            // compute tau_x, mu and t_hat and send to the verifier
            let tau_x = tau1 * x + tau2 * x_squared + z_squared * gamma;
            let mu = alpha + rho * x;
            let t_hat = dot(&l, &r);
            prover_state.add_scalars(&[tau_x, mu, t_hat])?;

            // 9
            // todo: maybe we can store this somewhere, so no need to recompute
            let sum_of_g_vec = ark_std::cfg_iter!(g_vec[..self.q * n * m]).sum::<G>();
            let y_inv = y.inverse().expect("Challenge may not be zero!");
            let y_inv_powers = powers(&y_inv, num_generators_full, true);
            // note: also for h_prime_vec we need a power of two for the length
            let h_prime_vec = G::normalize_batch(
                &ark_std::cfg_iter!(h_vec)
                    .zip(ark_std::cfg_into_iter!(y_inv_powers))
                    .map(|(h_val, y_inv_power)| h_val.mul(y_inv_power))
                    .collect::<Vec<_>>(),
            );
            // note: for powers_for_p we have pulled out one z, hence the difference with the computation in step 6
            let powers_for_p = ark_std::cfg_chunks!(y_powers, self.q)
                .zip(ark_std::cfg_iter!(z_powers_for_polynomial[..n * m]).cloned())
                .zip(ark_std::cfg_iter!(z_powers_for_polynomial[n * m..]).cloned()) // last one gets skipped (as it should) cuz of zip
                .map(
                    |((y_vals, mut z_power_first_term), mut z_power_second_term)| {
                        // optional<with scaling>: z power for the third term
                        let z_power_third_term = if self.sign_bits_proof.is_some() {
                            Some(z_power_second_term * z_to_the_nm)
                        } else {
                            None
                        };

                        y_vals.iter().enumerate().map(move |(idx, y_power)| {
                            let mut result = z_power_first_term + y_power;
                            z_power_first_term.double_in_place();
                            if idx >= self.s {
                                result += z_power_second_term;
                                z_power_second_term.double_in_place();
                            }
                            // optional<with scaling>: add a third term for the MSB only
                            if let Some(z_power_third_term) = z_power_third_term
                                && idx == q_prime
                            {
                                result += z_power_third_term;
                            }
                            result
                        })
                    },
                );
            #[cfg(feature = "parallel")]
            let powers_for_p = powers_for_p.flatten_iter().collect::<Vec<_>>();
            #[cfg(not(feature = "parallel"))]
            let powers_for_p = powers_for_p.flatten().collect::<Vec<_>>();
            let p = cm_a
                + cm_s.mul(&x)
                + sum_of_g_vec.mul(-z)
                + G::msm_unchecked(&h_prime_vec, &powers_for_p).mul(z);

            // 10 is verifier only

            // debug_assert_eq!(
            //     G::msm_unchecked(&g_vec, &l) + G::msm_unchecked(&h_prime_vec, &r) + h.mul(mu),
            //     p
            // );
            // debug_assert_eq!(dot(&l, &r), t_hat);

            // 11
            BulletproofInnerProductArgument {
                g_vec,
                h_vec: &h_prime_vec,
                p: &(p + h.mul(-mu)).into_affine(),
                c: &t_hat,
                a_vec: Some(&l),
                b_vec: Some(&r),
            }
            .prove(prover_state, client_schemes)?;
        }

        Ok(())
    }

    pub fn verify(
        self,
        verifier_state: &mut VerifierState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        // useful constants
        let q_prime = self.q - 1;
        let m_a = self.cm_a_mat_unscaled_all[0].row_coms.len();
        let m_b = self.cm_b_mat_unscaled.row_coms.len();
        let two_to_the_q_prime_minus_s = G::ScalarField::from(2).pow([(q_prime - self.s) as u64]);
        let two_to_the_q_prime = G::ScalarField::from(2).pow([q_prime as u64]);

        // INTERMEZZO - for a version where a consists of multiple HyraxCommitments and corresponding openings
        let a_take = self.a_take.unwrap_or(m_a);
        let a_chunk_size = a_take.next_power_of_two();
        let n_matrices = a_take * self.cm_a_mat_unscaled_all.len();

        // compute filtered version of commitments
        let to_pad = n_matrices.next_power_of_two() - n_matrices;

        let cm_a_mat_unscaled_combined_filtered_sorted =
            ark_std::cfg_into_iter!((0..m_a / a_chunk_size))
                .flat_map(|chunk_idx| {
                    let base_idx = chunk_idx * a_chunk_size;
                    ark_std::cfg_iter!(self.cm_a_mat_unscaled_all)
                        .flat_map(move |cm_a_mat| {
                            ark_std::cfg_iter!(cm_a_mat.row_coms[base_idx..base_idx + a_take])
                                .copied()
                        })
                        .chain(ark_std::cfg_into_iter!(vec![G::Affine::zero(); to_pad]))
                })
                .collect::<Vec<_>>();

        let pedersen_divisor = if PEDERSEN_DIVISOR_LARGE <= m_b {
            PEDERSEN_DIVISOR_LARGE
        } else {
            PEDERSEN_DIVISOR_SMALL
        };
        for rep_idx in 0..pedersen_divisor {
            let rep_len = (m_b as f64 / pedersen_divisor as f64).ceil() as usize;
            let m_b = if rep_idx + 1 == pedersen_divisor {
                m_b - (pedersen_divisor - 1)
                    * ((m_b as f64 / pedersen_divisor as f64).ceil() as usize)
            } else {
                (m_b as f64 / pedersen_divisor as f64).ceil() as usize
            };

            // useful variables
            // todo: maybe we can store this somewhere, so there's no need to recompute
            let sum_of_hyrax_gs_small = ark_std::cfg_iter!(
                &client_schemes.vector_commit.get_g_vec()[..min(self.n_a, self.n_b)]
            )
            .sum::<G>();
            let sum_of_hyrax_gs_big = sum_of_hyrax_gs_small
                + ark_std::cfg_iter!(
                    &client_schemes.vector_commit.get_g_vec()
                        [min(self.n_a, self.n_b)..max(self.n_a, self.n_b)]
                )
                .sum::<G>();
            let (sum_of_hyrax_gs_a, sum_of_hyrax_gs_b) = if self.n_a < self.n_b {
                (sum_of_hyrax_gs_small, sum_of_hyrax_gs_big)
            } else {
                (sum_of_hyrax_gs_big, sum_of_hyrax_gs_small)
            };

            let cm_two_to_the_q_prime_b = sum_of_hyrax_gs_b.mul(two_to_the_q_prime).into_affine();
            let cm_two_to_the_q_prime_minus_s_a = sum_of_hyrax_gs_a
                .mul(two_to_the_q_prime_minus_s)
                .into_affine();
            let cm_a_mat = ark_std::cfg_iter!(
                cm_a_mat_unscaled_combined_filtered_sorted
                    [rep_idx * rep_len..rep_idx * rep_len + m_b]
            )
            .map(|cm| (cm_two_to_the_q_prime_minus_s_a + cm).into_affine())
            .collect::<Vec<_>>();
            let cm_b_mat = ark_std::cfg_iter!(
                self.cm_b_mat_unscaled.row_coms[rep_idx * rep_len..rep_idx * rep_len + m_b]
            )
            .map(|cm| (cm_two_to_the_q_prime_b + cm).into_affine())
            .collect::<Vec<_>>();

            // 1
            // get the generators (note we get a two power for g_vec and h_vec as this is needed for the IPA, in the rest the latter part is ignored due to zips)
            let num_generators_full = (self.q * self.n_b * m_b).next_power_of_two();
            let g_vec = &client_schemes.vector_commit.get_g_vec()[..num_generators_full];
            let h_vec = &client_schemes.vector_commit.get_g_vec()
                [num_generators_full..2 * num_generators_full];
            let h = client_schemes.vector_commit.get_h();

            // 2
            let [cm_a, cm_s]: [G; 2] = verifier_state.next_points()?;

            // 3
            let [y, z]: [G::ScalarField; 2] = verifier_state.challenge_scalars()?;

            // INTERMEZZO: compute n and m
            let n = max(self.n_a, self.n_b);
            let m = self.n_b * m_b / n;

            // 4
            // compute z powers and other useful stuff
            let inner_z_powers = powers(&z, n, true);
            let z_to_the_n = inner_z_powers[n - 1] * z;
            let outer_z_powers = powers(&z_to_the_n, m, true);
            let z_to_the_nm = outer_z_powers[m - 1] * z_to_the_n;

            // receive cm_v
            let [cm_v]: [_; 1] = verifier_state.next_points()?.map(|cm: G| cm.into_affine());

            // 5
            // compute the combined commitment
            let mut xi = G::msm_unchecked(&cm_b_mat, &outer_z_powers)
                + G::msm_unchecked(&cm_a_mat, &outer_z_powers).mul(z_to_the_nm);

            if let Some(sign_bits_proof) = &self.sign_bits_proof {
                let z_to_the_2nm = z_to_the_nm.square();
                let cm_c_mat =
                    &sign_bits_proof.cm_c_mat.row_coms[rep_idx * rep_len..rep_idx * rep_len + m_b];

                xi += G::msm_unchecked(&cm_c_mat, &outer_z_powers).mul(z_to_the_2nm);
            }

            let xi = xi.into_affine();

            DotProductProof {
                g_vec: &client_schemes.vector_commit.get_g_vec()[..n],
                a_vec: &inner_z_powers,
                xi: Some(&xi),
                r_xi: None,
                tau: Some(&cm_v),
                r_tau: None,
                x_vec: None,
            }
            .verify(verifier_state, client_schemes)?;

            // 6
            // we also pre-compute some stuff for step 8
            let y_powers = powers(&y, self.q * n * m, true);
            let z_powers_for_polynomial = &powers(&z, 2 * n * m + 1, false);

            let [cm_t1, cm_t2]: [G; 2] = verifier_state.next_points()?;

            // 7
            let [x]: [G::ScalarField; 1] = verifier_state.challenge_scalars()?;

            // 8
            let z_squared = z.square();
            let x_squared = x.square();

            let [tau_x, mu, t_hat]: [G::ScalarField; 3] = verifier_state.next_scalars()?;

            // 9
            let sum_of_g_vec = ark_std::cfg_iter!(g_vec[0..self.q * n * m]).sum::<G>();
            let y_inv = y.inverse().expect("Challenge may not be zero!");
            let y_inv_powers = powers(&y_inv, num_generators_full, true);
            // note: also for h_prime_vec we need a power of two for the length
            let h_prime_vec = G::normalize_batch(
                &ark_std::cfg_iter!(h_vec)
                    .zip(ark_std::cfg_into_iter!(y_inv_powers))
                    .map(|(h_val, y_inv_power)| h_val.mul(y_inv_power))
                    .collect::<Vec<_>>(),
            );
            // note: for powers_for_p we have pulled out one z, hence the difference with the computation in step 6
            let powers_for_p = ark_std::cfg_chunks!(y_powers, self.q)
                .zip(ark_std::cfg_iter!(z_powers_for_polynomial[..n * m]).cloned())
                .zip(ark_std::cfg_iter!(z_powers_for_polynomial[n * m..]).cloned()) // last one gets skipped (as it should) cuz of zip
                .map(
                    |((y_vals, mut z_power_first_term), mut z_power_second_term)| {
                        // optional<with scaling>: z power for the third term
                        let z_power_third_term = if self.sign_bits_proof.is_some() {
                            Some(z_power_second_term * z_to_the_nm)
                        } else {
                            None
                        };

                        y_vals.iter().enumerate().map(move |(idx, y_power)| {
                            let mut result = z_power_first_term + y_power;
                            z_power_first_term.double_in_place();
                            if idx >= self.s {
                                result += z_power_second_term;
                                z_power_second_term.double_in_place();
                            }
                            // optional<with scaling>: add a third term for the MSB only
                            if let Some(z_power_third_term) = z_power_third_term
                                && idx == q_prime
                            {
                                result += z_power_third_term;
                            }
                            result
                        })
                    },
                );
            #[cfg(feature = "parallel")]
            let powers_for_p = powers_for_p.flatten_iter().collect::<Vec<_>>();
            #[cfg(not(feature = "parallel"))]
            let powers_for_p = powers_for_p.flatten().collect::<Vec<_>>();
            let p = cm_a
                + cm_s.mul(&x)
                + sum_of_g_vec.mul(-z)
                + G::msm_unchecked(&h_prime_vec, &powers_for_p).mul(z);

            // 10
            // compute delta
            let y_powers_sum = ark_std::cfg_iter!(y_powers).sum::<G::ScalarField>();
            let mut k_y_z = y_powers_sum
                + (two_to_the_q_prime.double() - G::ScalarField::ONE)
                    * ark_std::cfg_iter!(z_powers_for_polynomial[..n * m]).sum::<G::ScalarField>()
                + (two_to_the_q_prime_minus_s.double() - G::ScalarField::ONE)
                    * ark_std::cfg_iter!(z_powers_for_polynomial[n * m..n * m * 2])
                        .sum::<G::ScalarField>();
            if self.sign_bits_proof.is_some() {
                k_y_z += z_to_the_nm
                    * ark_std::cfg_iter!(z_powers_for_polynomial[n * m..n * m * 2])
                        .sum::<G::ScalarField>();
            }
            k_y_z *= -z_squared;
            let delta_y_z = z * y_powers_sum + k_y_z;

            // get the base for single value commitment: g
            let g = client_schemes.vector_commit.get_g_unique();

            // perform the check
            let lhs = client_schemes
                .vector_commit
                .value_commit_unique_g_same_h(&t_hat, &tau_x);
            let rhs = g.mul(delta_y_z) + cm_v.mul(z_squared) + cm_t1.mul(x) + cm_t2.mul(x_squared);
            if lhs != rhs.into_affine() {
                return Err(ProofError::InvalidProof);
            }

            // 11
            BulletproofInnerProductArgument {
                g_vec,
                h_vec: &h_prime_vec,
                p: &(p + h.mul(-mu)).into_affine(),
                c: &t_hat,
                a_vec: None,
                b_vec: None,
            }
            .verify(verifier_state, client_schemes)?;
        }

        Ok(())
    }
}
