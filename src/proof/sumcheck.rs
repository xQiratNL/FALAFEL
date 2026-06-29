use crate::client::ClientSchemes;
use crate::prelude::*;
use crate::proof::product::{ProductProof, ProductProofSeparator};
use crate::proof::util::poly_eval;
use ark_ff::AdditiveGroup;
use ark_std::UniformRand;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::codecs::arkworks_algebra::{
    FieldDomainSeparator, FieldToUnitDeserialize, FieldToUnitSerialize, GroupDomainSeparator,
    GroupToUnitDeserialize, GroupToUnitSerialize, UnitToField,
};
use spongefish::{
    DomainSeparator, DuplexSpongeInterface, ProofError, ProofResult, ProverState, VerifierState,
};
use std::ops::Mul;

pub mod util {
    use crate::crypto_schemes::Constants;
    use ark_ff::PrimeField;
    #[cfg(feature = "parallel")]
    use rayon::prelude::*;

    #[inline]
    pub fn point_evals_degree_2<F: PrimeField>(left: &F, right: &F) -> [F; 3] {
        let eval_minus1 = left.double() - right; // 2L - R
        let eval_0 = *left; // L
        let eval_1 = *right; // R
        [eval_minus1, eval_0, eval_1]
    }

    #[inline]
    pub fn point_evals_degree_3<F: PrimeField>(left: &F, right: &F) -> [F; 4] {
        let eval_minus1 = left.double() - right; // 2L - R
        let eval_0 = *left; // L
        let eval_1 = *right; // R
        let eval_2 = right.double() - left; // 2R - L
        [eval_minus1, eval_0, eval_1, eval_2]
    }

    #[inline]
    pub fn point_evals_degree_4<F: PrimeField>(left: &F, right: &F) -> [F; 5] {
        // pre-compute for re-using
        let left_double = left.double();
        let right_double = right.double();

        // evals
        let eval_minus2 = left_double + left - right_double; // 3L - 2R
        let eval_minus1 = left_double - right; // 2L - R
        let eval_0 = *left; // L
        let eval_1 = *right; // R
        let eval_2 = right_double - left; // 2R - L
        [eval_minus2, eval_minus1, eval_0, eval_1, eval_2]
    }

    pub fn get_coeffs_degree_2<F: PrimeField>(
        eval_minus1: F,
        eval_0: F,
        eval_1: F,
        constants: &Constants<F>,
    ) -> [F; 3] {
        // interpolation
        // useful points for determining coefficients
        let c_0_plus_c_2 = (eval_1 + eval_minus1) * constants.one_over_two;

        // determine coefficients
        let c_0 = eval_0;
        let c_2 = c_0_plus_c_2 - eval_0;
        let c_1 = (eval_1 - eval_minus1) * constants.one_over_two;

        [c_2, c_1, c_0]
    }

    pub fn get_coeffs_degree_3<F: PrimeField>(
        eval_minus1: F,
        eval_0: F,
        eval_1: F,
        eval_2: F,
        constants: &Constants<F>,
    ) -> [F; 4] {
        // interpolation
        // useful points for determining coefficients
        let neg_eval_minus1 = -eval_minus1;
        let c_0_plus_c_2 = (eval_1 + eval_minus1) * constants.one_over_two;
        let c_1_plus_c_3 = (eval_1 + &neg_eval_minus1) * constants.one_over_two;
        let c_1_plus_c_2_plus_three_times_c_3 =
            (eval_2 + neg_eval_minus1) * constants.one_over_three;

        // determine coefficients
        let c_0 = eval_0;
        let c_2 = c_0_plus_c_2 - eval_0;
        let c_3 = (c_1_plus_c_2_plus_three_times_c_3 - c_1_plus_c_3 - c_2) * constants.one_over_two;
        let c_1 = c_1_plus_c_3 - c_3;

        [c_3, c_2, c_1, c_0]
    }

    pub fn get_coeffs_degree_4<F: PrimeField>(
        eval_minus2: F,
        eval_minus1: F,
        eval_0: F,
        eval_1: F,
        eval_2: F,
        constants: &Constants<F>,
    ) -> [F; 5] {
        // interpolation
        // useful points for determining coefficients
        let c_0_plus_c_2_plus_c_4 = (eval_1 + eval_minus1) * constants.one_over_two;
        let c_1_plus_c_3 = (eval_1 - eval_minus1) * constants.one_over_two;
        let c_0_plus_four_c_2_plus_sixteen_c_4 = (eval_2 + eval_minus2) * constants.one_over_two;
        let c_1_plus_four_c_3 = (eval_2 - eval_minus2) * constants.one_over_four;

        // determine coefficients
        let c_0 = eval_0;
        let c_3 = (c_1_plus_four_c_3 - c_1_plus_c_3) * constants.one_over_three;
        let c_1 = c_1_plus_c_3 - c_3;

        // intermediate points
        let c_2_plus_c_4 = c_0_plus_c_2_plus_c_4 - c_0;
        let c_2_plus_five_c_4 =
            (c_0_plus_four_c_2_plus_sixteen_c_4 - c_0_plus_c_2_plus_c_4) * constants.one_over_three;

        // final coefficients
        let c_4 = (c_2_plus_five_c_4 - c_2_plus_c_4) * constants.one_over_four;
        let c_2 = c_2_plus_c_4 - c_4;

        [c_4, c_3, c_2, c_1, c_0]
    }

    // iterate over value in values with update array = (1 - value) * odd indexed entries + value * even indexed entries
    pub fn multi_collapse<F: PrimeField>(dp_array: &[F], values: &[F]) -> Vec<F> {
        let mut dp_array = collapse(dp_array, &values[0]);
        if values.len() >= 1 {
            for r_value in &values[1..] {
                dp_array = collapse(&dp_array, r_value)
            }
        }
        dp_array
    }

    // compute (1 - value) * odd indexed entries + value * even indexed entries
    pub fn collapse<F: PrimeField>(dp_array: &[F], value: &F) -> Vec<F> {
        let one_minus_value = F::ONE - value;
        ark_std::cfg_chunks!(dp_array, 2)
            .map(|left_right| one_minus_value * left_right[0] + *value * left_right[1])
            .collect()
    }
}

pub trait SumCheckProofSeparator<G: CurveGroup> {
    fn add_sum_check(self, len_deg4: Option<usize>, len_deg3: usize, len_deg2: usize) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> SumCheckProofSeparator<G> for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    fn add_sum_check(mut self, len_deg4: Option<usize>, len_deg3: usize, len_deg2: usize) -> Self {
        // we assume all statement elements have been sent already
        self = self.add_product_proof().add_product_proof();
        if let Some(len_deg4) = len_deg4 {
            self = self
                .add_product_proof()
                .add_points(len_deg4, "degree 4 deltas")
        }
        self = self
            .add_points(len_deg3, "degree 3 deltas")
            .add_points(len_deg2, "degree 2 deltas")
            .challenge_scalars(len_deg4.unwrap_or(0) + len_deg3 + len_deg2 + 1, "rhos")
            .add_points(1, "cm_p")
            .challenge_scalars(1, "c");
        if let Some(len_deg4) = len_deg4 {
            for round in 0..len_deg4 {
                self = self.add_scalars(5, &format!("deg 4; round {round} zs"))
            }
        }
        for round in 0..len_deg3 {
            self = self.add_scalars(4, &format!("deg 3; round {round} zs"))
        }
        for round in 0..len_deg2 {
            self = self.add_scalars(3, &format!("deg 2; round {round} zs"))
        }
        self.add_scalars(len_deg4.unwrap_or(0) + len_deg3 + len_deg2, "z_deltas")
            .add_scalars(1, "z_P")
    }
}

pub struct SumCheckProof<'a, G: CurveGroup> {
    // commitments
    pub cms_coeffs_degree4: Option<&'a [G::Affine]>, // only for degree 4
    pub cms_coeffs_degree3: &'a [G::Affine],
    pub cms_coeffs_degree2: &'a [G::Affine],
    pub cm_a1: &'a G::Affine, // this is cm_a in a sumcheck of degree 3
    pub cm_a2: Option<&'a G::Affine>, // only for degree 4
    pub cm_b1: &'a G::Affine,
    pub cm_b2: &'a G::Affine,
    pub cm_c: &'a G::Affine,
    pub cm_a1_a2: Option<&'a G::Affine>, // only for degree 4
    pub cm_b1_b2: &'a G::Affine,
    pub cm_a_b: &'a G::Affine,

    // public value
    pub multiplier: &'a G::ScalarField,

    // eval inputs
    pub r_degree_4: Option<&'a [G::ScalarField]>, // only for degree 4
    pub r_degree_3: &'a [G::ScalarField],
    pub r_degree_2: &'a [G::ScalarField],

    // witnesses
    pub coeffs_degree4: Option<Option<&'a [[G::ScalarField; 5]]>>, // only for degree 4
    pub r_coeffs_degree4: Option<Option<&'a [G::ScalarField]>>,    // only for degree 4
    pub coeffs_degree3: Option<&'a [[G::ScalarField; 4]]>,
    pub r_coeffs_degree3: Option<&'a [G::ScalarField]>,
    pub coeffs_degree2: Option<&'a [[G::ScalarField; 3]]>,
    pub r_coeffs_degree2: Option<&'a [G::ScalarField]>,
    pub a1: Option<&'a G::ScalarField>,
    pub r_a1: Option<&'a G::ScalarField>,
    pub a2: Option<Option<&'a G::ScalarField>>, // only for degree 4
    pub r_a2: Option<Option<&'a G::ScalarField>>, // only for degree 4
    pub b1: Option<&'a G::ScalarField>,
    pub r_b1: Option<&'a G::ScalarField>,
    pub b2: Option<&'a G::ScalarField>,
    pub r_b2: Option<&'a G::ScalarField>,
    pub r_c: Option<&'a G::ScalarField>,
    pub a1_a2: Option<Option<&'a G::ScalarField>>, // only for degree 4
    pub r_a1_a2: Option<Option<&'a G::ScalarField>>, // only for degree 4
    pub b1_b2: Option<&'a G::ScalarField>,
    pub r_b1_b2: Option<&'a G::ScalarField>,
    pub r_a_b: Option<&'a G::ScalarField>,
}

impl<'a, G: CurveGroup> SumCheckProof<'a, G> {
    // Algorithm 8 in our paper
    pub fn prove(
        &self,
        is_degree_4: bool,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        // unwrap useful values
        let r_c = self.r_c.ok_or_else(|| ProofError::SerializationError)?;
        let r_a_b = self.r_a_b.ok_or_else(|| ProofError::SerializationError)?;

        // 1: input already given

        // 2
        if is_degree_4 {
            ProductProof::<G> {
                cm_x: self.cm_a1, // this also exists for degree 4
                cm_y: self.cm_a2.expect("This should exist for degree 4!"),
                cm_z: self.cm_a1_a2.expect("This should exist for degree 4!"),
                x: self.a1,     // this also exists for degree 4
                r_x: self.r_a1, // this also exists for degree 4
                y: self.a2.expect("This should exist for degree 4!"),
                r_y: self.r_a2.expect("This should exist for degree 4!"),
                r_z: self.r_a1_a2.expect("This should exist for degree 4!"),
            }
            .prove(prover_state, client_schemes)?;
        }
        ProductProof::<G> {
            cm_x: self.cm_b1,
            cm_y: self.cm_b2,
            cm_z: self.cm_b1_b2,
            x: self.b1,
            r_x: self.r_b1,
            y: self.b2,
            r_y: self.r_b2,
            r_z: self.r_b1_b2,
        }
        .prove(prover_state, client_schemes)?;
        ProductProof::<G> {
            cm_x: if is_degree_4 {
                self.cm_a1_a2.expect("This should exist for degree 4!")
            } else {
                self.cm_a1
            },
            cm_y: self.cm_b1_b2,
            cm_z: self.cm_a_b,
            x: if is_degree_4 {
                self.a1_a2.expect("This should exist for degree 4!")
            } else {
                self.a1
            },
            r_x: if is_degree_4 {
                self.r_a1_a2.expect("This should exist for degree 4!")
            } else {
                self.r_a1
            },
            y: self.b1_b2,
            r_y: self.r_b1_b2,
            r_z: self.r_a_b,
        }
        .prove(prover_state, client_schemes)?;

        // 3: commit to randomness
        let ds_degree4 = if is_degree_4 {
            (0..self
                .cms_coeffs_degree4
                .expect("This should exist for degree 4!")
                .len())
                .into_iter()
                .map(|_| {
                    (0..5)
                        .into_iter()
                        .map(|_| G::ScalarField::rand(&mut prover_state.rng()))
                        .collect::<Vec<_>>()
                        .try_into()
                        .unwrap()
                })
                .collect::<Vec<[_; 5]>>()
        } else {
            vec![]
        };
        let ds_degree3 = (0..self.cms_coeffs_degree3.len())
            .into_iter()
            .map(|_| {
                (0..4)
                    .into_iter()
                    .map(|_| G::ScalarField::rand(&mut prover_state.rng()))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<[_; 4]>>();
        let ds_degree2 = (0..self.cms_coeffs_degree2.len())
            .into_iter()
            .map(|_| {
                (0..3)
                    .into_iter()
                    .map(|_| G::ScalarField::rand(&mut prover_state.rng()))
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap()
            })
            .collect::<Vec<[_; 3]>>();
        let r_deltas_degree4 = if is_degree_4 {
            (0..self
                .cms_coeffs_degree4
                .expect("This should exist for degree 4!")
                .len())
                .into_iter()
                .map(|_| G::ScalarField::rand(&mut prover_state.rng()))
                .collect::<Vec<_>>()
        } else {
            vec![]
        };
        let r_deltas_degree3 = (0..self.cms_coeffs_degree3.len())
            .into_iter()
            .map(|_| G::ScalarField::rand(&mut prover_state.rng()))
            .collect::<Vec<_>>();
        let r_deltas_degree2 = (0..self.cms_coeffs_degree2.len())
            .into_iter()
            .map(|_| G::ScalarField::rand(&mut prover_state.rng()))
            .collect::<Vec<_>>();
        // actual commitments
        let deltas_degree4 = ark_std::cfg_iter!(ds_degree4)
            .zip(ark_std::cfg_iter!(r_deltas_degree4))
            .map(|(d, r_delta)| client_schemes.vector_commit.vector_commit(d, r_delta))
            .collect::<Vec<_>>();
        let deltas_degree3 = ark_std::cfg_iter!(ds_degree3)
            .zip(ark_std::cfg_iter!(r_deltas_degree3))
            .map(|(d, r_delta)| client_schemes.vector_commit.vector_commit(d, r_delta))
            .collect::<Vec<_>>();
        let deltas_degree2 = ark_std::cfg_iter!(ds_degree2)
            .zip(ark_std::cfg_iter!(r_deltas_degree2))
            .map(|(d, r_delta)| client_schemes.vector_commit.vector_commit(d, r_delta))
            .collect::<Vec<_>>();
        if is_degree_4 {
            prover_state.add_points(
                &deltas_degree4
                    .iter()
                    .map(|x| x.into_group())
                    .collect::<Vec<_>>(),
            )?;
        }
        prover_state.add_points(
            &deltas_degree3
                .iter()
                .map(|x| x.into_group())
                .collect::<Vec<_>>(),
        )?;
        prover_state.add_points(
            &deltas_degree2
                .iter()
                .map(|x| x.into_group())
                .collect::<Vec<_>>(),
        )?;

        // 4
        let mut rhos = vec![
            G::ScalarField::ZERO;
            self.r_degree_4.map_or(0, |x| x.len())
                + self.r_degree_3.len()
                + self.r_degree_2.len()
                + 1
        ];
        prover_state.fill_challenge_scalars(&mut rhos)?;

        // 5: compute the inner product of the linear combination of the conditions
        let p = self.ip_with_lc_of_m(&rhos, &ds_degree4, &ds_degree3, &ds_degree2);
        let r_p = G::ScalarField::rand(prover_state.rng());
        let cm_p = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&p, &r_p);
        prover_state.add_points(&[cm_p.into_group()])?;

        // 6
        let [c]: [G::ScalarField; 1] = prover_state.challenge_scalars()?;

        // 7
        // compute and send all z values
        let coeffs_degree4 = if is_degree_4 {
            self.coeffs_degree4
                .expect("This should exist for degree 4!")
                .ok_or_else(|| ProofError::SerializationError)?
        } else {
            &[]
        };
        let coeffs_degree3 = self
            .coeffs_degree3
            .ok_or_else(|| ProofError::SerializationError)?;
        let coeffs_degree2 = self
            .coeffs_degree2
            .ok_or_else(|| ProofError::SerializationError)?;
        let zs = ark_std::cfg_iter!(coeffs_degree4)
            .map(|x| x.as_slice())
            .chain(ark_std::cfg_iter!(coeffs_degree3).map(|x| x.as_slice()))
            .chain(ark_std::cfg_iter!(coeffs_degree2).map(|x| x.as_slice()))
            .zip(
                ark_std::cfg_iter!(ds_degree4)
                    .map(|x| x.as_slice())
                    .chain(ark_std::cfg_iter!(ds_degree3).map(|x| x.as_slice()))
                    .chain(ark_std::cfg_iter!(ds_degree2).map(|x| x.as_slice())),
            )
            .map(|(coeffs, ds)| {
                coeffs
                    .iter()
                    .zip(ds.iter())
                    .map(|(coeff, d)| c * coeff + d)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for z in zs {
            prover_state.add_scalars(&z)?;
        }

        // compute and send all z_delta values
        let r_coeffs_degree4 = if is_degree_4 {
            self.r_coeffs_degree4
                .expect("This should exist for degree 4!")
                .ok_or_else(|| ProofError::SerializationError)?
        } else {
            &[]
        };
        let r_coeffs_degree3 = self
            .r_coeffs_degree3
            .ok_or_else(|| ProofError::SerializationError)?;
        let r_coeffs_degree2 = self
            .r_coeffs_degree2
            .ok_or_else(|| ProofError::SerializationError)?;
        let z_deltas = ark_std::cfg_iter!(r_coeffs_degree4)
            .chain(ark_std::cfg_iter!(r_coeffs_degree3))
            .chain(ark_std::cfg_iter!(r_coeffs_degree2))
            .zip(
                ark_std::cfg_iter!(r_deltas_degree4)
                    .chain(ark_std::cfg_iter!(r_deltas_degree3))
                    .chain(ark_std::cfg_iter!(r_deltas_degree2)),
            )
            .map(|(r_coeff, r_delta)| c * r_coeff + r_delta)
            .collect::<Vec<_>>();
        prover_state.add_scalars(&z_deltas)?;

        // compute and send z_P
        let z_p = r_p + c * (*r_c * rhos[0] + *r_a_b * self.multiplier * rhos.last().unwrap());
        prover_state.add_scalars(&[z_p])?;

        // 8: verifier only

        Ok(())
    }

    pub fn verify(
        &self,
        is_degree_4: bool,
        verifier_state: &mut VerifierState,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<()> {
        // 1: input already given

        // 2
        if is_degree_4 {
            ProductProof::<G> {
                cm_x: self.cm_a1, // this also exists for degree 4
                cm_y: self.cm_a2.expect("This should exist for degree 4!"),
                cm_z: self.cm_a1_a2.expect("This should exist for degree 4!"),
                x: None,
                r_x: None,
                y: None,
                r_y: None,
                r_z: None,
            }
            .verify(verifier_state, client_schemes)?;
        }
        ProductProof::<G> {
            cm_x: self.cm_b1,
            cm_y: self.cm_b2,
            cm_z: self.cm_b1_b2,
            x: None,
            r_x: None,
            y: None,
            r_y: None,
            r_z: None,
        }
        .verify(verifier_state, client_schemes)?;
        ProductProof::<G> {
            cm_x: if is_degree_4 {
                self.cm_a1_a2.expect("This should exist for degree 4!")
            } else {
                self.cm_a1
            },
            cm_y: self.cm_b1_b2,
            cm_z: self.cm_a_b,
            x: None,
            r_x: None,
            y: None,
            r_y: None,
            r_z: None,
        }
        .verify(verifier_state, client_schemes)?;

        // 3
        let deltas_degree4 = if is_degree_4 {
            let mut deltas_degree4 = vec![
                G::ZERO;
                self.r_degree_4
                    .expect("This should exist for degree 4!")
                    .len()
            ];
            verifier_state.fill_next_points(&mut deltas_degree4)?;
            let deltas_degree4 = G::normalize_batch(&deltas_degree4);
            deltas_degree4
        } else {
            vec![]
        };
        let mut deltas_degree3 = vec![G::ZERO; self.r_degree_3.len()];
        verifier_state.fill_next_points(&mut deltas_degree3)?;
        let deltas_degree3 = G::normalize_batch(&deltas_degree3);
        let mut deltas_degree2 = vec![G::ZERO; self.r_degree_2.len()];
        verifier_state.fill_next_points(&mut deltas_degree2)?;
        let deltas_degree2 = G::normalize_batch(&deltas_degree2);

        // 4
        let mut rhos = vec![
            G::ScalarField::ZERO;
            self.r_degree_4.map_or(0, |x| x.len())
                + self.r_degree_3.len()
                + self.r_degree_2.len()
                + 1
        ];
        verifier_state.fill_challenge_scalars(&mut rhos)?;

        // 5
        let [cm_p]: [G; 1] = verifier_state.next_points()?;
        let cm_p = cm_p.into_affine();

        // 6
        let [c]: [G::ScalarField; 1] = verifier_state.challenge_scalars()?;

        // 7
        // all z_values = (zs_degree4, zs_degree3, zs_degree2)
        let zs_degree4 = if is_degree_4 {
            let mut zs_degree4 = vec![
                [G::ScalarField::ZERO; 5];
                self.r_degree_4
                    .expect("This should exist for degree 4!")
                    .len()
            ];
            for round_zs in &mut zs_degree4 {
                *round_zs = verifier_state.next_scalars()?;
            }
            zs_degree4
        } else {
            vec![]
        };
        let mut zs_degree3 = vec![[G::ScalarField::ZERO; 4]; self.r_degree_3.len()];
        for round_zs in &mut zs_degree3 {
            *round_zs = verifier_state.next_scalars()?;
        }
        let mut zs_degree2 = vec![[G::ScalarField::ZERO; 3]; self.r_degree_2.len()];
        for round_zs in &mut zs_degree2 {
            *round_zs = verifier_state.next_scalars()?;
        }

        // z_delta & z_P
        let mut z_deltas = vec![
            G::ScalarField::ZERO;
            self.r_degree_4.map_or(0, |x| x.len())
                + self.r_degree_3.len()
                + self.r_degree_2.len()
        ];
        verifier_state.fill_next_scalars(&mut z_deltas)?;
        let [z_p]: [G::ScalarField; 1] = verifier_state.next_scalars()?;

        // 8

        // check 1: for each round
        if ark_std::cfg_iter!(zs_degree4)
            .map(|x| x.as_slice())
            .chain(ark_std::cfg_iter!(zs_degree3).map(|x| x.as_slice()))
            .chain(ark_std::cfg_iter!(zs_degree2).map(|x| x.as_slice()))
            .zip(ark_std::cfg_iter!(z_deltas))
            .zip(
                ark_std::cfg_iter!(if is_degree_4 {
                    self.cms_coeffs_degree4
                        .expect("This should exist for degree 4!")
                } else {
                    &[]
                })
                .chain(ark_std::cfg_iter!(self.cms_coeffs_degree3))
                .chain(ark_std::cfg_iter!(self.cms_coeffs_degree2)),
            )
            .zip(
                ark_std::cfg_iter!(deltas_degree4)
                    .chain(ark_std::cfg_iter!(deltas_degree3))
                    .chain(ark_std::cfg_iter!(deltas_degree2)),
            )
            .any(|(((zs, z_delta), cm_coeffs), delta)| {
                client_schemes.vector_commit.vector_commit(zs, z_delta)
                    != (cm_coeffs.mul(&c) + delta).into_affine()
            })
        {
            return Err(ProofError::InvalidProof);
        }

        // check 2
        let lhs = client_schemes.vector_commit.value_commit_unique_g_same_h(
            &self.ip_with_lc_of_m(&rhos, &zs_degree4, &zs_degree3, &zs_degree2),
            &z_p,
        );
        let rhs = cm_p
            + (self.cm_c.mul(rhos[0]) + self.cm_a_b.mul(*self.multiplier * rhos.last().unwrap()))
                .mul(c);
        if lhs != rhs.into_affine() {
            return Err(ProofError::InvalidProof);
        }

        Ok(())
    }

    // computes the inner product for step 5/8 of the sum-check protocol
    fn ip_with_lc_of_m(
        &self,
        rhos: &[G::ScalarField],
        ip_vector_degree4: &[[G::ScalarField; 5]],
        ip_vector_degree3: &[[G::ScalarField; 4]],
        ip_vector_degree2: &[[G::ScalarField; 3]],
    ) -> G::ScalarField {
        // rho_1: (c_4) + c_3 + c_2 + c_1 + 2*c_0
        let first_coefficients = if let Some(coeffs) = ip_vector_degree4.first() {
            coeffs.as_slice()
        } else {
            ip_vector_degree3[0].as_slice()
        };
        let mut p = rhos[0]
            * (first_coefficients.iter().sum::<G::ScalarField>()
                + first_coefficients.last().unwrap());

        // rho_i: 1 < i <= n: ((c_4 + ) c_3 + ) c_2 + c_1 + 2*c_0 - ( ((c_4*r^4) + c_3*r^3) + c_2*r^2 + c_1*r + c_0)
        p += ark_std::cfg_iter!(rhos[1..rhos.len() - 1])
            .zip(
                ark_std::cfg_iter!(ip_vector_degree4)
                    .map(|x| x.as_slice())
                    .chain(ark_std::cfg_iter!(ip_vector_degree3).map(|x| x.as_slice()))
                    .chain(ark_std::cfg_iter!(ip_vector_degree2).map(|x| x.as_slice())),
            )
            .zip(
                ark_std::cfg_iter!(if let Some(r) = self.r_degree_4 {
                    r
                } else {
                    &[]
                })
                .chain(ark_std::cfg_iter!(self.r_degree_3))
                .chain(ark_std::cfg_iter!(self.r_degree_2)),
            )
            .zip(
                ark_std::cfg_iter!(ip_vector_degree4)
                    .map(|x| x.as_slice())
                    .chain(ark_std::cfg_iter!(ip_vector_degree3).map(|x| x.as_slice()))
                    .chain(ark_std::cfg_iter!(ip_vector_degree2).map(|x| x.as_slice()))
                    .skip(1),
            )
            .map(|(((&rho, prev_ds), prev_r), cur_ds)| {
                rho * (cur_ds.iter().sum::<G::ScalarField>() + cur_ds.last().unwrap()
                    - poly_eval(prev_ds, prev_r))
            })
            .sum::<G::ScalarField>();

        // rho_{n+1}: c_2*r^2 + c_1*r + c_0
        p + *rhos.last().unwrap()
            * poly_eval(
                ip_vector_degree2.last().unwrap(),
                self.r_degree_2.last().unwrap(),
            )
    }
}
