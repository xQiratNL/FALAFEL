use crate::client::ClientSchemes;
use crate::crypto_schemes::{Constants, HyraxCommitment};
use crate::prelude::*;
use crate::proof::backward_pass::BackwardPassDeferredEvals;
use crate::proof::commitments::LocalTrainingCommitments;
use crate::proof::dot_product::DotProductProof;
use crate::proof::equality::{BatchEqualityProof, BatchEqualityProofSeparator};
use crate::proof::forward_pass::ForwardPassDeferredEvals;
use crate::proof::hyrax_poly_eval::{HyraxPolyEvalProofSeparator, HyraxTargetCommitment};
use crate::proof::opening::{BatchOpeningProof, BatchOpeningProofSeparator};
use crate::proof::sumcheck::util::multi_collapse;
use crate::proof::util::{dot, row_mult_matrix, tensor_chi};
use crate::proof::{
    HyraxCmMatricesDimensions, LocalTrainingProof, LocalTrainingProofConstants, ProofWitnessFull,
};
use ark_ff::{AdditiveGroup, Field, PrimeField};
use ark_poly::DenseUVPolynomial;
use ark_poly::univariate::{DenseOrSparsePolynomial, DensePolynomial};
use ark_std::{UniformRand, log2};
use ndarray::{Array2, s};
use num_traits::Zero;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::codecs::arkworks_algebra::{
    FieldDomainSeparator, GroupDomainSeparator, GroupToUnitDeserialize, GroupToUnitSerialize,
    UnitToField,
};
use spongefish::{DomainSeparator, DuplexSpongeInterface, ProofResult, ProverState, VerifierState};
use std::ops::Mul;

pub trait PolyEvalsSeparator<G: CurveGroup> {
    fn add_combined_poly_eval_proof(
        self,
        n_poly_evals: usize,
        dimensions: &HyraxCmMatricesDimensions,
        additional_eval_size: Option<&HyraxCmMatricesDimensions>,
    ) -> Self;
    fn add_poly_eval_proofs<T: Value>(self, constants: &LocalTrainingProofConstants<T>) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> PolyEvalsSeparator<G> for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    fn add_combined_poly_eval_proof(
        mut self,
        n_poly_evals: usize,
        dimensions: &HyraxCmMatricesDimensions,
        additional_eval_size: Option<&HyraxCmMatricesDimensions>,
    ) -> Self {
        self = self
            .add_points(
                (n_poly_evals - 1) * dimensions.total_log + 1,
                "coefficients of curve-restricted polynomial",
            )
            .add_batch_opening_proof()
            .add_batch_equality_proof()
            .challenge_scalars(1, "tau for curve-restricted polynomial");
        if let Some(size) = additional_eval_size {
            self = self
                .add_points(1, "target commitment (evaluation result)")
                .add_hyrax_poly_eval_proof(size.row_com_dimension);
        }
        self.add_hyrax_poly_eval_proof(dimensions.row_com_dimension)
    }

    fn add_poly_eval_proofs<T: Value>(self, constants: &LocalTrainingProofConstants<T>) -> Self {
        self
            // W
            .add_points(
                constants.cm_w_full_dimensions.total_log + 1,
                "coefficients of curve-restricted polynomial",
            )
            .add_batch_opening_proof()
            .add_batch_equality_proof()
            .challenge_scalars(1, "tau for curve_restricted polynomial")
            .add_hyrax_poly_eval_proof(constants.cm_w_single_iteration_dimensions.row_com_dimension)
            // C
            .add_combined_poly_eval_proof(4, &constants.cm_c_dimensions, None)
            // T_hat
            .add_combined_poly_eval_proof(
                2,
                &constants.cm_t_dimensions,
                Some(&constants.cm_batch_samples_dimensions),
            )
            // T
            .add_points(1, "target commitment (evaluation result of u)")
            .add_hyrax_poly_eval_proof(constants.cm_u_dimensions.row_com_dimension)
            .add_hyrax_poly_eval_proof(constants.cm_t_dimensions.row_com_dimension)
            // R_hat
            .add_combined_poly_eval_proof(
                2,
                &constants.cm_r_dimensions,
                Some(&constants.cm_r_final_dimensions),
            )
            // G_bar
            .add_hyrax_poly_eval_proof(constants.cm_g_bar_dimensions.row_com_dimension)
            // R
            .add_hyrax_poly_eval_proof(constants.cm_r_dimensions.row_com_dimension)
    }
}

impl<'a, G: CurveGroup, T: Value> ProofWitnessFull<'a, G, T> {
    pub fn prove_poly_evals(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<G>,
        commitments: &LocalTrainingCommitments<G>,
        forward_pass_deferred_evals: ForwardPassDeferredEvals<G>,
        backward_pass_deferred_evals: BackwardPassDeferredEvals<G>,
    ) -> ProofResult<()> {
        // W: two times
        self.eval_prove_combined_w_target_commitment(
            prover_state,
            client_schemes,
            commitments,
            &forward_pass_deferred_evals,
            &backward_pass_deferred_evals,
        )?;

        // C: four times
        let combined_c_target_commitment = self.compute_combined_c_target_commitment(
            prover_state,
            client_schemes,
            commitments,
            &forward_pass_deferred_evals,
            &backward_pass_deferred_evals,
        )?;
        combined_c_target_commitment.eval_prove(
            commitments.opening_c.as_ref().expect("This should exist"),
            prover_state,
            client_schemes,
            true,
        )?;

        // T_hat: two times
        let combined_t_hat_target_commitment = self.compute_combined_t_hat_target_commitment(
            prover_state,
            client_schemes,
            commitments,
            &forward_pass_deferred_evals,
            &backward_pass_deferred_evals,
        )?;
        combined_t_hat_target_commitment.eval_prove(
            commitments
                .opening_t_hat
                .as_ref()
                .expect("This should exist"),
            prover_state,
            client_schemes,
            true,
        )?;

        // T: one time
        let mut t_target_commitment = forward_pass_deferred_evals.t;
        self.update_t_target_commitment(
            prover_state,
            client_schemes,
            commitments,
            &mut t_target_commitment,
        )?;
        t_target_commitment.eval_prove(
            commitments.opening_t.as_ref().expect("This should exist"),
            prover_state,
            client_schemes,
            true,
        )?;

        // R_hat: two times
        let combined_r_hat_target_commitment = self.compute_combined_r_hat_target_commitment(
            prover_state,
            client_schemes,
            commitments,
            &backward_pass_deferred_evals,
        )?;
        combined_r_hat_target_commitment.eval_prove(
            commitments
                .opening_r_hat
                .as_ref()
                .expect("This should exist"),
            prover_state,
            client_schemes,
            true,
        )?;

        // G_bar: one time
        backward_pass_deferred_evals.g_bar.eval_prove(
            commitments
                .opening_g_bar
                .as_ref()
                .expect("This should exist"),
            prover_state,
            client_schemes,
            false,
        )?;

        // R: one time
        backward_pass_deferred_evals.r.eval_prove(
            commitments.opening_r.as_ref().expect("This should exist"),
            prover_state,
            client_schemes,
            false,
        )?;

        Ok(())
    }

    fn coefficients_opening_prove(
        evals: &[G::ScalarField],
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<G>,
    ) -> ProofResult<Vec<G::ScalarField>> {
        // then we determine the coefficients based on the evals
        let restricted_poly_coeffs = lagrange_interpolation(evals);
        // finally we commit to the coefficients and send them
        let r_restricted_poly_coeffs = (0..evals.len())
            .map(|_| G::ScalarField::rand(&mut prover_state.rng()))
            .collect::<Vec<_>>();
        let cms_restricted_poly_coeffs = ark_std::cfg_iter!(restricted_poly_coeffs)
            .zip(ark_std::cfg_iter!(r_restricted_poly_coeffs))
            .map(|(coeff, r)| {
                client_schemes
                    .vector_commit
                    .value_commit_unique_g_same_h(coeff, r)
            })
            .collect::<Vec<_>>();
        prover_state.add_points(
            &cms_restricted_poly_coeffs
                .iter()
                .map(|x| x.into_group())
                .collect::<Vec<_>>(),
        )?;

        // batched proof of opening for commitments to coefficients of the curve-restricted polynomial
        BatchOpeningProof {
            cms: &cms_restricted_poly_coeffs,
            xs: Some(&restricted_poly_coeffs),
            rs: Some(&r_restricted_poly_coeffs),
        }
        .prove(prover_state, client_schemes)?;

        Ok(r_restricted_poly_coeffs)
    }

    fn degree_3_equality_prove(
        r_restricted_poly_coeffs: &[G::ScalarField],
        target_commitments: &[&HyraxTargetCommitment<G>; 4],
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<G>,
    ) -> ProofResult<()> {
        // do a batched equality proof to verify the 4 evaluations against the curve-restricted polynomial
        let two = G::ScalarField::from(2);
        let three = G::ScalarField::from(3);
        let mut rs_lhs = vec![r_restricted_poly_coeffs[0]; 4];
        let mut two_to_the_power_i = two;
        let mut three_to_the_power_i = three;
        for r in r_restricted_poly_coeffs.iter().skip(1) {
            rs_lhs[1] += r;
            rs_lhs[2] += two_to_the_power_i * r;
            rs_lhs[3] += three_to_the_power_i * r;
            two_to_the_power_i *= two;
            three_to_the_power_i *= three;
        }
        let rs_rhs = target_commitments.map(|x| x.r_omega.expect("This should exist"));
        BatchEqualityProof {
            cms_lhs: None,
            cms_rhs: None,
            rs_lhs: Some(&rs_lhs),
            rs_rhs: Some(&rs_rhs),
        }
        .prove(prover_state, client_schemes)
    }

    fn degree_1_equality_prove(
        r_restricted_poly_coeffs: &[G::ScalarField],
        target_commitments: &[&HyraxTargetCommitment<G>; 2],
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<G>,
    ) -> ProofResult<()> {
        // do a batched equality proof to verify the 2 evaluations against curve-restricted T_hat
        let mut rs_lhs = vec![r_restricted_poly_coeffs[0]; 2];
        for r in r_restricted_poly_coeffs.iter().skip(1) {
            rs_lhs[1] += r;
        }
        let rs_rhs = target_commitments.map(|x| x.r_omega.expect("This should exist"));
        BatchEqualityProof {
            cms_lhs: None,
            cms_rhs: None,
            rs_lhs: Some(&rs_lhs),
            rs_rhs: Some(&rs_rhs),
        }
        .prove(prover_state, client_schemes)
    }

    fn random_point<const N: usize>(
        curve_coeffs: &[[G::ScalarField; N]],
        prover_state: &mut ProverState,
    ) -> ProofResult<(G::ScalarField, Vec<G::ScalarField>)> {
        // determine the random point on the line
        let [tau]: [G::ScalarField; 1] = prover_state.challenge_scalars()?;
        let points = curve_eval(tau, curve_coeffs);
        Ok((tau, points))
    }

    fn new_target_commitment(
        tau: G::ScalarField,
        points: Vec<G::ScalarField>,
        r_restricted_poly_coeffs: &[G::ScalarField],
        r_shared_commitment: Option<G::ScalarField>, // we assume that the shared commitment is not in the commitment, but is in the sum-check result
        poly_commitment: &HyraxCommitment<G>,
    ) -> ProofResult<HyraxTargetCommitment<G>> {
        // compute the HyraxTargetCommitment from the obtained info and run the evaluation proof
        let log2_l_len = log2(poly_commitment.row_coms.len()) as usize;
        let c_l = tensor_chi(&points[..log2_l_len]);
        let c_r = tensor_chi(&points[log2_l_len..]);

        let mut r_omega = r_restricted_poly_coeffs[0];
        let mut tau_to_the_power_i = tau;
        for r in r_restricted_poly_coeffs.iter().skip(1) {
            r_omega += tau_to_the_power_i * r;
            tau_to_the_power_i *= tau;
        }

        Ok(HyraxTargetCommitment {
            omega: None,
            ltr: None,
            r_omega: Some(if let Some(r_shared_commitment) = r_shared_commitment {
                r_omega - r_shared_commitment
            } else {
                r_omega
            }),
            lt: None,
            l: Some(c_l),
            r: Some(c_r),
            points: Some(points),
        })
    }

    fn update_t_target_commitment(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
        commitments: &LocalTrainingCommitments<G>,
        t_target_cm: &mut HyraxTargetCommitment<G>,
    ) -> ProofResult<()> {
        let t_points = t_target_cm.points.as_ref().unwrap();

        // evaluate the U matrices
        let (t_points_matrix, t_points_row_col) =
            t_points.split_at(self.constants.cm_t_dimensions.matrices_log);
        let (t_points_row, t_points_col) =
            t_points_row_col.split_at(self.constants.cm_t_dimensions.rows_log);
        let u_points_row_col = [
            &t_points_row[..self.constants.cm_u_dimensions.rows_log],
            &t_points_col[..self.constants.cm_u_dimensions.columns_log],
        ]
        .concat();

        let remaining_row_col_multiplier = t_points_row[self.constants.cm_u_dimensions.rows_log..]
            .iter()
            .chain(&t_points_col[self.constants.cm_u_dimensions.columns_log..])
            .map(|point| G::ScalarField::ONE - point)
            .product::<G::ScalarField>();

        let row_col_split_idx = u_points_row_col.len()
            - log2(self.constants.cm_u_dimensions.row_com_dimension) as usize;
        let mut u_points_matrix_l = tensor_chi(&t_points_matrix)
            .into_iter()
            .skip(self.constants.n_layers + 1)
            .step_by(self.constants.n_layers + 2)
            .collect::<Vec<_>>();
        u_points_matrix_l.resize(
            1 << self.constants.cm_u_dimensions.matrices_log,
            G::ScalarField::ZERO,
        );
        let u_points_row_col_l = tensor_chi(&u_points_row_col[..row_col_split_idx]);
        let u_points_r = tensor_chi(&u_points_row_col[row_col_split_idx..]);
        let u_points_l = ark_std::cfg_iter!(u_points_row_col_l)
            .flat_map(|&row_col_val| {
                ark_std::cfg_iter!(u_points_matrix_l)
                    .map(move |matrix_val| row_col_val * matrix_val * remaining_row_col_multiplier)
            })
            .collect::<Vec<_>>();

        let opening_u = commitments.opening_u.as_ref().expect("This should exist");
        let lt = row_mult_matrix(&u_points_l, &opening_u.input_matrix);
        let eval = dot(&lt, &u_points_r);

        let r_u_eval = G::ScalarField::rand(prover_state.rng());
        let u_eval = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&eval, &r_u_eval);
        prover_state.add_points(&[u_eval.into_group()])?;

        HyraxTargetCommitment {
            omega: None,
            ltr: None,
            r_omega: Some(r_u_eval),
            lt: Some(lt),
            l: Some(u_points_l),
            r: Some(u_points_r),
            points: None,
        }
        .eval_prove(opening_u, prover_state, client_schemes, false)?;

        t_target_cm.r_omega = Some(t_target_cm.r_omega.unwrap() - r_u_eval);

        Ok(())
    }

    fn compute_combined_r_hat_target_commitment(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
        commitments: &LocalTrainingCommitments<G>,
        backward_pass_deferred_evals: &BackwardPassDeferredEvals<G>,
    ) -> ProofResult<HyraxTargetCommitment<G>> {
        // compute R_hat restricted to the curve going through the 2 different points that R_hat is evaluated on
        // this will be a function of degree log(full dimension of R_hat)
        let r_hat_target_cms = [
            &backward_pass_deferred_evals.r_hat_one,
            &backward_pass_deferred_evals.r_hat_two,
        ];
        let r_hat_points = r_hat_target_cms.map(|x| {
            x.points
                .as_ref()
                .map(|y| y.as_slice())
                .expect("This should exist")
        });
        // first, we compute the curve going through the 2 different points
        let r_hat_curve_coeffs = degree_1_curve_coeffs(r_hat_points);

        // then we compute sufficient evals
        let mut evals = r_hat_target_cms
            .map(|x| x.ltr.expect("This should exist"))
            .to_vec();
        // we make sure the ordering of r_hat_curve_coeffs matches that of the r_hat_dp_array we re=use from the sumcheck phase
        let (r_hat_curve_coeffs_matrix_part, r_hat_curve_coeffs_row_col_part) =
            r_hat_curve_coeffs.split_at(self.constants.cm_r_dimensions.matrices_log);
        let (r_hat_curve_coeffs_row_part, r_hat_curve_coeffs_col_part) =
            r_hat_curve_coeffs_row_col_part.split_at(self.constants.cm_r_dimensions.rows_log);
        for point in 2..=self.constants.cm_r_dimensions.total_log {
            evals.push(curve_constrained_eval(
                G::ScalarField::from(point as u64),
                &[
                    r_hat_curve_coeffs_col_part,
                    r_hat_curve_coeffs_row_part,
                    r_hat_curve_coeffs_matrix_part,
                ]
                .concat(),
                backward_pass_deferred_evals
                    .r_hat_dp_array
                    .as_ref()
                    .expect("This should exist"),
            ))
        }

        let r_r_hat_restricted_coeffs =
            Self::coefficients_opening_prove(&evals, prover_state, client_schemes)?;

        Self::degree_1_equality_prove(
            &r_r_hat_restricted_coeffs,
            &r_hat_target_cms,
            prover_state,
            client_schemes,
        )?;

        let (tau, r_hat_points) = Self::random_point(&r_hat_curve_coeffs, prover_state)?;

        // evaluate the final r_hat_matrices
        let (r_hat_points_matrix, r_hat_points_row_col) =
            r_hat_points.split_at(self.constants.cm_r_dimensions.matrices_log);
        let (r_hat_points_row, r_hat_points_col) =
            r_hat_points_row_col.split_at(self.constants.cm_r_dimensions.rows_log);
        let r_hat_final_points_row_col = [
            &r_hat_points_row[..self.constants.cm_r_final_dimensions.rows_log],
            &r_hat_points_col[..self.constants.cm_r_final_dimensions.columns_log],
        ]
        .concat();

        let remaining_row_col_multiplier = r_hat_points_row
            [self.constants.cm_r_final_dimensions.rows_log..]
            .iter()
            .chain(&r_hat_points_col[self.constants.cm_r_final_dimensions.columns_log..])
            .map(|point| G::ScalarField::ONE - point)
            .product::<G::ScalarField>();

        let row_col_split_idx = r_hat_final_points_row_col.len()
            - log2(self.constants.cm_r_final_dimensions.row_com_dimension) as usize;
        let mut r_hat_final_points_matrix_l = tensor_chi(&r_hat_points_matrix)
            .into_iter()
            .skip(self.constants.n_layers + 1)
            .step_by(self.constants.n_layers + 2)
            .collect::<Vec<_>>();
        r_hat_final_points_matrix_l.resize(
            1 << self.constants.cm_r_final_dimensions.matrices_log,
            G::ScalarField::ZERO,
        );
        let r_hat_final_points_row_col_l =
            tensor_chi(&r_hat_final_points_row_col[..row_col_split_idx]);
        let r_hat_final_points_r = tensor_chi(&r_hat_final_points_row_col[row_col_split_idx..]);
        let r_hat_final_points_l = ark_std::cfg_iter!(r_hat_final_points_row_col_l)
            .flat_map(|&row_col_val| {
                ark_std::cfg_iter!(r_hat_final_points_matrix_l)
                    .map(move |matrix_val| row_col_val * matrix_val * remaining_row_col_multiplier)
            })
            .collect::<Vec<_>>();

        let opening_r_hat_final = commitments
            .opening_r_hat_final
            .as_ref()
            .expect("This should exist");
        let lt = row_mult_matrix(&r_hat_final_points_l, &opening_r_hat_final.input_matrix);
        let eval = dot(&lt, &r_hat_final_points_r);

        let r_r_final_eval = G::ScalarField::rand(prover_state.rng());
        let r_final_eval = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&eval, &r_r_final_eval);
        prover_state.add_points(&[r_final_eval.into_group()])?;

        HyraxTargetCommitment {
            omega: None,
            ltr: None,
            r_omega: Some(r_r_final_eval),
            lt: Some(lt),
            l: Some(r_hat_final_points_l),
            r: Some(r_hat_final_points_r),
            points: None,
        }
        .eval_prove(opening_r_hat_final, prover_state, client_schemes, false)?;

        Self::new_target_commitment(
            tau,
            r_hat_points,
            &r_r_hat_restricted_coeffs,
            Some(r_r_final_eval),
            &commitments.cm_r_hat,
        )
    }

    fn compute_combined_t_hat_target_commitment(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
        commitments: &LocalTrainingCommitments<G>,
        forward_pass_deferred_evals: &ForwardPassDeferredEvals<G>,
        backward_pass_deferred_evals: &BackwardPassDeferredEvals<G>,
    ) -> ProofResult<HyraxTargetCommitment<G>> {
        // compute T_hat restricted to the curve going through the 2 different points that T_hat is evaluated on
        // this will be a function of degree log(full dimension of T_hat)
        let t_hat_target_cms = [
            &forward_pass_deferred_evals.t_hat,
            &backward_pass_deferred_evals.t_hat,
        ];
        let t_hat_points = t_hat_target_cms.map(|x| {
            x.points
                .as_ref()
                .map(|y| y.as_slice())
                .expect("This should exist")
        });
        // first, we compute the curve going through the 2 different points
        let t_hat_curve_coeffs = degree_1_curve_coeffs(t_hat_points);

        // then we compute sufficient evals
        let mut evals = t_hat_target_cms
            .map(|x| x.ltr.expect("This should exist"))
            .to_vec();
        // we make sure the ordering of t_hat_curve_coeffs matches that of the t_hat_dp_array that we re-use from the sumcheck phase
        let (t_hat_curve_coeffs_matrix_part, t_hat_curve_coeffs_row_col_part) =
            t_hat_curve_coeffs.split_at(self.constants.cm_t_dimensions.matrices_log);
        let (t_hat_curve_coeffs_row_part, t_hat_curve_coeffs_col_part) =
            t_hat_curve_coeffs_row_col_part.split_at(self.constants.cm_t_dimensions.rows_log);
        for point in 2..=self.constants.cm_t_dimensions.total_log {
            evals.push(curve_constrained_eval(
                G::ScalarField::from(point as u64),
                &[
                    t_hat_curve_coeffs_row_part,
                    t_hat_curve_coeffs_matrix_part,
                    t_hat_curve_coeffs_col_part,
                ]
                .concat(),
                forward_pass_deferred_evals
                    .t_hat_dp_array
                    .as_ref()
                    .expect("This should exist"),
            ))
        }

        let r_t_hat_restricted_coeffs =
            Self::coefficients_opening_prove(&evals, prover_state, client_schemes)?;

        Self::degree_1_equality_prove(
            &r_t_hat_restricted_coeffs,
            &t_hat_target_cms,
            prover_state,
            client_schemes,
        )?;

        let (tau, t_hat_points) = Self::random_point(&t_hat_curve_coeffs, prover_state)?;

        // also evaluate the batch matrices
        let (t_hat_points_matrix, t_hat_points_row_col) =
            t_hat_points.split_at(self.constants.cm_t_dimensions.matrices_log);
        let (t_hat_points_row, t_hat_points_col) =
            t_hat_points_row_col.split_at(self.constants.cm_t_dimensions.rows_log);
        let batch_points_row_col = [
            &t_hat_points_row[..self.constants.cm_batch_samples_dimensions.rows_log],
            &t_hat_points_col[..self.constants.cm_batch_samples_dimensions.columns_log],
        ]
        .concat();

        let remaining_row_col_multiplier = t_hat_points_row
            [self.constants.cm_batch_samples_dimensions.rows_log..]
            .iter()
            .chain(&t_hat_points_col[self.constants.cm_batch_samples_dimensions.columns_log..])
            .map(|point| G::ScalarField::ONE - point)
            .product::<G::ScalarField>();

        let row_col_split_idx = batch_points_row_col.len()
            - log2(self.constants.cm_batch_samples_dimensions.row_com_dimension) as usize;
        let t_hat_points_matrix = tensor_chi(&t_hat_points_matrix);
        let batch_points_row_col_l = tensor_chi(&batch_points_row_col[..row_col_split_idx]);
        let batch_points_r = tensor_chi(&batch_points_row_col[row_col_split_idx..]);
        let mut batch_points_matrix_l =
            vec![
                G::ScalarField::ZERO;
                1 << self.constants.cm_batch_samples_dimensions.matrices_log
            ];
        for epoch_idx in 0..self.constants.n_epochs {
            batch_points_matrix_l[epoch_idx % self.constants.n_batches] +=
                t_hat_points_matrix[epoch_idx * (self.constants.n_layers + 2)]
        }
        let batch_points_l = ark_std::cfg_iter!(batch_points_row_col_l)
            .flat_map(|&row_col_val| {
                ark_std::cfg_iter!(batch_points_matrix_l)
                    .map(move |matrix_val| row_col_val * matrix_val * remaining_row_col_multiplier)
            })
            .collect::<Vec<_>>();

        let opening_batch_samples = commitments
            .opening_batch_samples
            .as_ref()
            .expect("This should exist");
        let lt = row_mult_matrix(&batch_points_l, &opening_batch_samples.input_matrix);
        let eval = dot(&lt, &batch_points_r);

        let r_batch_eval = G::ScalarField::rand(prover_state.rng());
        let batch_eval = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&eval, &r_batch_eval);
        prover_state.add_points(&[batch_eval.into_group()])?;

        HyraxTargetCommitment {
            omega: None,
            ltr: None,
            r_omega: Some(r_batch_eval),
            lt: Some(lt),
            l: Some(batch_points_l),
            r: Some(batch_points_r),
            points: None,
        }
        .eval_prove(opening_batch_samples, prover_state, client_schemes, false)?;

        Self::new_target_commitment(
            tau,
            t_hat_points,
            &r_t_hat_restricted_coeffs,
            Some(r_batch_eval),
            &commitments.cm_t_hat,
        )
    }

    fn compute_combined_c_target_commitment(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<G>,
        commitments: &LocalTrainingCommitments<G>,
        forward_pass_deferred_evals: &ForwardPassDeferredEvals<G>,
        backward_pass_deferred_evals: &BackwardPassDeferredEvals<G>,
    ) -> ProofResult<HyraxTargetCommitment<G>> {
        // compute C restricted to the curve going through the 4 different points that C is evaluated on
        // this will be a function of degree 3*log(full dimension of C)
        let c_target_cms = [
            &forward_pass_deferred_evals.c,
            &backward_pass_deferred_evals.c_one_one,
            &backward_pass_deferred_evals.c_one_two,
            &backward_pass_deferred_evals.c_two,
        ];
        let c_points = c_target_cms.map(|x| {
            x.points
                .as_ref()
                .map(|y| y.as_slice())
                .expect("This should exist.")
        });
        // first, we compute the curve going through the 4 different points
        let c_curve_coeffs = degree_3_curve_coeffs(c_points, &client_schemes.constants);

        // then we compute sufficient evals
        let mut evals = c_target_cms
            .map(|x| x.ltr.expect("This should exist"))
            .to_vec();
        // we make sure the ordering of c_curve_coeffs matches that of the c_dp_array that we re-use from the sumcheck phase
        let (c_curve_coeffs_matrix_part, c_curve_coeffs_row_col_part) =
            c_curve_coeffs.split_at(self.constants.cm_c_dimensions.matrices_log);
        let (c_curve_coeffs_row_part, c_curve_coeffs_col_part) =
            c_curve_coeffs_row_col_part.split_at(self.constants.cm_c_dimensions.rows_log);
        for point in 4..=3 * self.constants.cm_c_dimensions.total_log {
            evals.push(curve_constrained_eval(
                G::ScalarField::from(point as u64),
                &[
                    c_curve_coeffs_row_part,
                    c_curve_coeffs_matrix_part,
                    c_curve_coeffs_col_part,
                ]
                .concat(),
                forward_pass_deferred_evals
                    .c_dp_array
                    .as_ref()
                    .expect("This should exist"),
            ))
        }

        let r_c_restricted_coeffs =
            Self::coefficients_opening_prove(&evals, prover_state, client_schemes)?;

        Self::degree_3_equality_prove(
            &r_c_restricted_coeffs,
            &c_target_cms,
            prover_state,
            client_schemes,
        )?;

        let (tau, c_points) = Self::random_point(&c_curve_coeffs, prover_state)?;

        Self::new_target_commitment(
            tau,
            c_points,
            &r_c_restricted_coeffs,
            None,
            &commitments.cm_c,
        )
    }

    fn eval_prove_combined_w_target_commitment(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<'a, G>,
        commitments: &LocalTrainingCommitments<G>,
        forward_pass_deferred_evals: &ForwardPassDeferredEvals<G>,
        backward_pass_deferred_evals: &BackwardPassDeferredEvals<G>,
    ) -> ProofResult<()> {
        // compute W restricted to the curve going through the 2 different points that W is evaluated on
        // this will be a function of degree log(full dimension of W)
        let w_target_cms = [
            &forward_pass_deferred_evals.w,
            &backward_pass_deferred_evals.w,
        ];
        let w_points = w_target_cms.map(|x| {
            x.points
                .as_ref()
                .map(|y| y.as_slice())
                .expect("This should exist")
        });
        // first, we compute the curve going through the 2 different points
        let w_curve_coeffs = degree_1_curve_coeffs(w_points);

        // then we compute sufficient evals
        let mut evals = w_target_cms
            .map(|x| x.ltr.expect("This should exist"))
            .to_vec();
        // we make sure the ordering of the w_curve_coeffs matches that of the w_dp_array that we re-use from the sumcheck phase
        let (w_curve_coeffs_matrix_part, w_curve_coeffs_row_col_part) =
            w_curve_coeffs.split_at(self.constants.cm_w_full_dimensions.matrices_log);
        let (w_curve_coeffs_row_part, w_curve_coeffs_col_part) =
            w_curve_coeffs_row_col_part.split_at(self.constants.cm_w_full_dimensions.rows_log);
        for point in 2..=self.constants.cm_w_full_dimensions.total_log {
            evals.push(curve_constrained_eval(
                G::ScalarField::from(point as u64),
                &[
                    w_curve_coeffs_row_part,
                    w_curve_coeffs_col_part,
                    w_curve_coeffs_matrix_part,
                ]
                .concat(),
                forward_pass_deferred_evals
                    .w_dp_array
                    .as_ref()
                    .expect("This should exist"),
            ))
        }

        let r_w_restricted_coeffs =
            Self::coefficients_opening_prove(&evals, prover_state, client_schemes)?;

        Self::degree_1_equality_prove(
            &r_w_restricted_coeffs,
            &w_target_cms,
            prover_state,
            client_schemes,
        )?;

        let (tau, w_points) = Self::random_point(&w_curve_coeffs, prover_state)?;

        // evaluate W
        let (w_points_matrix, w_points_row_col) =
            w_points.split_at(self.constants.cm_w_full_dimensions.matrices_log);

        let row_col_split_idx = w_points_row_col.len()
            - log2(
                self.constants
                    .cm_w_single_iteration_dimensions
                    .row_com_dimension,
            ) as usize;
        let w_points_matrix_l = tensor_chi(&w_points_matrix);
        let w_single_points_row_col_l = tensor_chi(&w_points_row_col[..row_col_split_idx]);
        let w_single_points_r = tensor_chi(&w_points_row_col[row_col_split_idx..]);

        let padding = vec![
            G::ScalarField::ZERO;
            (1 << self.constants.cm_w_single_iteration_dimensions.matrices_log)
                - (self.constants.n_layers + 2)
        ];
        let mut w_single_points_l_all = w_points_matrix_l
            .chunks_exact(self.constants.n_layers + 2)
            .take(self.constants.n_epochs)
            .rev()
            .map(|w_points_matrix_l_chunk| {
                ark_std::cfg_iter!(w_single_points_row_col_l)
                    .flat_map(|&row_col_val| {
                        ark_std::cfg_iter!(w_points_matrix_l_chunk)
                            .chain(ark_std::cfg_iter!(padding))
                            .map(move |matrix_val| row_col_val * matrix_val)
                    })
                    .collect::<Vec<_>>()
            });

        // evaluate the g_hat part
        let g_hat_opening_all = commitments
            .opening_g_hat_all
            .as_ref()
            .expect("This should exist");
        let mut g_hat_opening_all = g_hat_opening_all.iter().rev().skip(1);

        // initial step to initialize the sums
        let mut w_single_points_l_sum = w_single_points_l_all.next().unwrap();
        let (l_g_hat_sum, mut r_lt) = if let Some(g_hat_opening) = g_hat_opening_all.next() {
            let mut l_g_hat_sum =
                row_mult_matrix(&w_single_points_l_sum, &g_hat_opening.input_matrix);
            let mut r_lt = dot(&w_single_points_l_sum, &g_hat_opening.randomness);

            for g_hat_opening in g_hat_opening_all {
                ark_std::cfg_iter_mut!(w_single_points_l_sum)
                    .zip(ark_std::cfg_into_iter!(
                        w_single_points_l_all.next().unwrap()
                    ))
                    .for_each(|(s, e)| *s += e);
                let l_g_hat = row_mult_matrix(&w_single_points_l_sum, &g_hat_opening.input_matrix);
                ark_std::cfg_iter_mut!(l_g_hat_sum)
                    .zip(ark_std::cfg_into_iter!(l_g_hat))
                    .for_each(|(s, e)| *s += e);
                r_lt += dot(&w_single_points_l_sum, &g_hat_opening.randomness);
            }
            ark_std::cfg_iter_mut!(w_single_points_l_sum)
                .zip(ark_std::cfg_into_iter!(
                    w_single_points_l_all.next().unwrap()
                ))
                .for_each(|(s, e)| *s += e);

            (Some(l_g_hat_sum), r_lt)
        } else {
            (None, G::ScalarField::ZERO)
        };

        // compute w - g_hat_sum for all values
        let mut lt = row_mult_matrix(
            &w_single_points_l_sum,
            &self.opening_initial_weights.input_matrix,
        );
        if let Some(l_g_hat_sum) = l_g_hat_sum {
            ark_std::cfg_iter_mut!(lt)
                .zip(ark_std::cfg_into_iter!(l_g_hat_sum))
                .for_each(|(s, e)| *s -= e);
        }
        r_lt = dot(
            &w_single_points_l_sum,
            &self.opening_initial_weights.randomness,
        ) - r_lt;

        // compute r_tau
        let mut r_omega = r_w_restricted_coeffs[0];
        let mut tau_to_the_power_i = tau;
        for r in r_w_restricted_coeffs.iter().skip(1) {
            r_omega += tau_to_the_power_i * r;
            tau_to_the_power_i *= tau;
        }

        // we need to do a custom version of eval_prove
        DotProductProof {
            g_vec: &client_schemes.vector_commit.get_g_vec()[..w_single_points_r.len()],
            a_vec: &w_single_points_r,
            xi: None,
            r_xi: Some(&r_lt),
            tau: None,
            r_tau: Some(&r_omega),
            x_vec: Some(&lt),
        }
        .prove(prover_state, client_schemes)
    }
}

impl<G: CurveGroup, T: Value> LocalTrainingProof<G, T> {
    pub fn verify_poly_evals<'a>(
        &self,
        verifier_state: &mut VerifierState,
        commitments: &LocalTrainingCommitments<G>,
        cm_initial_weights: &HyraxCommitment<G>,
        client_schemes: &ClientSchemes<'a, G>,
        forward_pass_deferred_evals: ForwardPassDeferredEvals<G>,
        backward_pass_deferred_evals: BackwardPassDeferredEvals<G>,
    ) -> ProofResult<()> {
        // W: two times
        self.eval_verify_combined_w_target_commitment(
            verifier_state,
            commitments,
            cm_initial_weights,
            client_schemes,
            &forward_pass_deferred_evals,
            &backward_pass_deferred_evals,
        )?;

        // C: four times
        let combined_c_target_commitment = self.compute_combined_c_target_commitment(
            verifier_state,
            commitments,
            client_schemes,
            &forward_pass_deferred_evals,
            &backward_pass_deferred_evals,
        )?;
        combined_c_target_commitment.eval_verify(
            &commitments.cm_c,
            verifier_state,
            client_schemes,
        )?;

        // T_hat: two times
        let combined_t_hat_target_commitment = self.compute_combined_t_hat_target_commitment(
            verifier_state,
            commitments,
            client_schemes,
            &forward_pass_deferred_evals,
            &backward_pass_deferred_evals,
        )?;
        combined_t_hat_target_commitment.eval_verify(
            &commitments.cm_t_hat,
            verifier_state,
            client_schemes,
        )?;

        // T: one time
        let mut t_target_commitment = forward_pass_deferred_evals.t;
        self.update_t_target_commitment(
            verifier_state,
            commitments,
            client_schemes,
            &mut t_target_commitment,
        )?;
        t_target_commitment.eval_verify(&commitments.cm_t, verifier_state, client_schemes)?;

        // R_hat: two times
        let combined_r_hat_target_commitment = self.compute_combined_r_hat_target_commitment(
            verifier_state,
            commitments,
            client_schemes,
            &backward_pass_deferred_evals,
        )?;
        combined_r_hat_target_commitment.eval_verify(
            &commitments.cm_r_hat,
            verifier_state,
            client_schemes,
        )?;

        // G_bar: one time
        backward_pass_deferred_evals.g_bar.eval_verify(
            &commitments.cm_g_bar,
            verifier_state,
            client_schemes,
        )?;

        // R: one time
        backward_pass_deferred_evals.r.eval_verify(
            &commitments.cm_r,
            verifier_state,
            client_schemes,
        )?;

        Ok(())
    }

    fn coefficients_opening_verify(
        degree: usize,
        verifier_state: &mut VerifierState,
        client_schemes: &ClientSchemes<G>,
    ) -> ProofResult<Vec<G::Affine>> {
        let mut cms_restricted_poly_coeffs = vec![G::zero(); degree + 1];
        verifier_state.fill_next_points(&mut cms_restricted_poly_coeffs)?;
        let cms_restricted_poly_coeffs = G::normalize_batch(&cms_restricted_poly_coeffs);

        // batched proof of opening for commitments to coefficients of the curve-restricted polynomial
        BatchOpeningProof {
            cms: &cms_restricted_poly_coeffs,
            xs: None,
            rs: None,
        }
        .verify(verifier_state, client_schemes)?;

        Ok(cms_restricted_poly_coeffs)
    }

    fn degree_3_equality_verify(
        cms_restricted_poly_coeffs: &[G::Affine],
        target_commitments: &[&HyraxTargetCommitment<G>; 4],
        verifier_state: &mut VerifierState,
        client_schemes: &ClientSchemes<G>,
    ) -> ProofResult<()> {
        // do a batched equality proof to verify the 4 evaluations against the curve-restricted polynomial
        let two = G::ScalarField::from(2);
        let three = G::ScalarField::from(3);
        let mut cm_lhs = vec![cms_restricted_poly_coeffs[0].into_group(); 4];
        let mut two_to_the_power_i = two;
        let mut three_to_the_power_i = three;
        for cm in cms_restricted_poly_coeffs.iter().skip(1) {
            cm_lhs[1] += cm;
            cm_lhs[2] += cm.mul(two_to_the_power_i);
            cm_lhs[3] += cm.mul(three_to_the_power_i);
            two_to_the_power_i *= two;
            three_to_the_power_i *= three;
        }
        let cm_lhs = G::normalize_batch(&cm_lhs);
        let cm_rhs = target_commitments.map(|x| x.omega.expect("This should exist"));
        BatchEqualityProof {
            cms_lhs: Some(&cm_lhs),
            cms_rhs: Some(&cm_rhs),
            rs_lhs: None,
            rs_rhs: None,
        }
        .verify(verifier_state, client_schemes)
    }

    fn degree_1_equality_verify(
        cms_restricted_poly_coeffs: &[G::Affine],
        target_commitments: &[&HyraxTargetCommitment<G>; 2],
        verifier_state: &mut VerifierState,
        client_schemes: &ClientSchemes<G>,
    ) -> ProofResult<()> {
        // do a batched equality proof to verify the 2 evaluations against the curve-restricted polynomial
        let mut cm_lhs = vec![cms_restricted_poly_coeffs[0].into_group(); 2];
        for cm in cms_restricted_poly_coeffs.iter().skip(1) {
            cm_lhs[1] += cm;
        }
        let cm_lhs = G::normalize_batch(&cm_lhs);
        let cm_rhs = target_commitments.map(|x| x.omega.expect("This should exist"));
        BatchEqualityProof {
            cms_lhs: Some(&cm_lhs),
            cms_rhs: Some(&cm_rhs),
            rs_lhs: None,
            rs_rhs: None,
        }
        .verify(verifier_state, client_schemes)
    }

    fn random_point_degree_3(
        target_commitments: &[&HyraxTargetCommitment<G>; 4],
        verifier_state: &mut VerifierState,
        client_schemes: &ClientSchemes<G>,
    ) -> ProofResult<(G::ScalarField, Vec<G::ScalarField>)> {
        let [tau]: [G::ScalarField; 1] = verifier_state.challenge_scalars()?;
        let points = target_commitments.map(|x| {
            x.points
                .as_ref()
                .map(|y| y.as_slice())
                .expect("This should exist")
        });
        // first, we compute the curve going through the 4 different points
        let curve_coeffs = degree_3_curve_coeffs(points, &client_schemes.constants);
        // then, we compute the random point
        let points = curve_eval(tau, &curve_coeffs);
        Ok((tau, points))
    }

    fn random_point_degree_1(
        target_commitments: &[&HyraxTargetCommitment<G>; 2],
        verifier_state: &mut VerifierState,
    ) -> ProofResult<(G::ScalarField, Vec<G::ScalarField>)> {
        let [tau]: [G::ScalarField; 1] = verifier_state.challenge_scalars()?;
        let points = target_commitments.map(|x| {
            x.points
                .as_ref()
                .map(|y| y.as_slice())
                .expect("This should exist.")
        });
        // first, we compute the curve going through the 2 different points
        let curve_coeffs = degree_1_curve_coeffs(points);
        // then, we compute the random point
        let points = curve_eval(tau, &curve_coeffs);
        Ok((tau, points))
    }

    fn new_target_commitment(
        tau: G::ScalarField,
        l: Vec<G::ScalarField>,
        r: Vec<G::ScalarField>,
        cms_restricted_poly_coeffs: &[G::Affine],
        public_value: G::ScalarField, // we assume that the public value is not in the commitment, but is in the sum-check result
        shared_commitment: Option<G::Affine>, // we assume that the shared commitment is not in the commitment, but is in the sum-check result
        client_schemes: &ClientSchemes<G>,
    ) -> HyraxTargetCommitment<G> {
        let mut zeta = cms_restricted_poly_coeffs[0].into_group();
        let mut tau_to_the_power_i = tau;
        for cm in cms_restricted_poly_coeffs.iter().skip(1) {
            zeta += cm.mul(tau_to_the_power_i);
            tau_to_the_power_i *= tau;
        }

        HyraxTargetCommitment {
            omega: Some(
                if let Some(shared_commitment) = shared_commitment {
                    zeta - shared_commitment
                        - client_schemes
                            .vector_commit
                            .get_g_unique()
                            .mul(public_value)
                } else {
                    zeta - client_schemes
                        .vector_commit
                        .get_g_unique()
                        .mul(public_value)
                }
                .into_affine(),
            ),
            ltr: None,
            r_omega: None,
            lt: None,
            l: Some(l),
            r: Some(r),
            points: None,
        }
    }

    fn compute_c_and_t_hat_zero_matrices(&self) -> Vec<Array2<G::ScalarField>> {
        std::iter::repeat_n(self.constants.batch_size, self.constants.n_batches - 1)
            .chain(std::iter::once({
                let remainder = self.constants.n_samples % self.constants.batch_size;
                if remainder == 0 {
                    self.constants.batch_size
                } else {
                    remainder
                }
            }))
            .map(|col| {
                std::iter::once(&self.constants.n_features)
                    .chain(self.constants.n_hidden.iter())
                    .map(|x| x + 1)
                    .chain(std::iter::repeat_n(self.constants.n_labels, 2))
                    .map(move |row| (row, col))
            })
            .flatten()
            .collect::<Vec<_>>()
            .repeat(self.constants.n_epochs / self.constants.n_batches)
            .into_iter()
            .map(|dims| Array2::from_elem(dims, G::ScalarField::ZERO))
            .collect::<Vec<_>>()
    }

    fn compute_w_matrices(&self) -> Vec<Array2<G::ScalarField>> {
        std::iter::once((self.constants.n_features + 1, 1))
            .chain(
                std::iter::once(self.constants.n_features)
                    .chain(self.constants.n_hidden.iter().copied())
                    .zip(
                        self.constants
                            .n_hidden
                            .iter()
                            .copied()
                            .chain(std::iter::once(self.constants.n_labels)),
                    )
                    .map(|(n_in, n_out)| (n_out + 1, n_in + 1)),
            )
            .chain(std::iter::once((
                self.constants.n_labels,
                self.constants.n_labels,
            )))
            .collect::<Vec<_>>()
            .repeat(self.constants.n_epochs)
            .into_iter()
            .map(|dims| Array2::from_elem(dims, G::ScalarField::ZERO))
            .collect::<Vec<_>>()
    }

    fn update_t_target_commitment<'a>(
        &self,
        verifier_state: &mut VerifierState,
        commitments: &LocalTrainingCommitments<G>,
        client_schemes: &ClientSchemes<'a, G>,
        t_target_cm: &mut HyraxTargetCommitment<G>,
    ) -> ProofResult<()> {
        let t_points = t_target_cm.points.as_ref().unwrap();

        // evaluate the U matrices
        let (t_points_matrix, t_points_row_col) =
            t_points.split_at(self.constants.cm_t_dimensions.matrices_log);
        let (t_points_row, t_points_col) =
            t_points_row_col.split_at(self.constants.cm_t_dimensions.rows_log);
        let u_points_row_col = [
            &t_points_row[..self.constants.cm_u_dimensions.rows_log],
            &t_points_col[..self.constants.cm_u_dimensions.columns_log],
        ]
        .concat();

        let remaining_row_col_multiplier = t_points_row[self.constants.cm_u_dimensions.rows_log..]
            .iter()
            .chain(&t_points_col[self.constants.cm_u_dimensions.columns_log..])
            .map(|point| G::ScalarField::ONE - point)
            .product::<G::ScalarField>();

        let row_col_split_idx = u_points_row_col.len()
            - log2(self.constants.cm_u_dimensions.row_com_dimension) as usize;
        let mut u_points_matrix_l = tensor_chi(&t_points_matrix)
            .into_iter()
            .skip(self.constants.n_layers + 1)
            .step_by(self.constants.n_layers + 2)
            .collect::<Vec<_>>();
        u_points_matrix_l.resize(
            1 << self.constants.cm_u_dimensions.matrices_log,
            G::ScalarField::ZERO,
        );
        let u_points_row_col_l = tensor_chi(&u_points_row_col[..row_col_split_idx]);
        let u_points_r = tensor_chi(&u_points_row_col[row_col_split_idx..]);
        let u_points_l = ark_std::cfg_iter!(u_points_row_col_l)
            .flat_map(|&row_col_val| {
                ark_std::cfg_iter!(u_points_matrix_l)
                    .map(move |matrix_val| row_col_val * matrix_val * remaining_row_col_multiplier)
            })
            .collect::<Vec<_>>();

        let [u_eval]: [_; 1] = verifier_state.next_points()?.map(|x: G| x.into_affine());

        HyraxTargetCommitment {
            omega: Some(u_eval),
            ltr: None,
            r_omega: None,
            lt: None,
            l: Some(u_points_l),
            r: Some(u_points_r),
            points: None,
        }
        .eval_verify(&commitments.cm_u, verifier_state, client_schemes)?;

        t_target_cm.omega = Some((t_target_cm.omega.unwrap() - u_eval).into_affine());

        Ok(())
    }

    fn compute_combined_r_hat_target_commitment<'a>(
        &self,
        verifier_state: &mut VerifierState,
        commitments: &LocalTrainingCommitments<G>,
        client_schemes: &ClientSchemes<'a, G>,
        backward_pass_deferred_evals: &BackwardPassDeferredEvals<G>,
    ) -> ProofResult<HyraxTargetCommitment<G>> {
        // obtain R_hat restricted to the curve going through the 2 different points that R_hat is evaluated on
        // this will be a function of degree log(full dimension of R_hat)
        let r_hat_target_cms = [
            &backward_pass_deferred_evals.r_hat_one,
            &backward_pass_deferred_evals.r_hat_two,
        ];
        let cms_r_hat_restricted_coeffs = Self::coefficients_opening_verify(
            self.constants.cm_r_dimensions.total_log,
            verifier_state,
            client_schemes,
        )?;

        Self::degree_1_equality_verify(
            &cms_r_hat_restricted_coeffs,
            &r_hat_target_cms,
            verifier_state,
            client_schemes,
        )?;

        // determine the random point on the line
        let (tau, r_hat_points) = Self::random_point_degree_1(&r_hat_target_cms, verifier_state)?;

        // evaluate the final r_hat_matrices
        let (r_hat_points_matrix, r_hat_points_row_col) =
            r_hat_points.split_at(self.constants.cm_r_dimensions.matrices_log);
        let (r_hat_points_row, r_hat_points_col) =
            r_hat_points_row_col.split_at(self.constants.cm_r_dimensions.rows_log);
        let r_hat_final_points_row_col = [
            &r_hat_points_row[..self.constants.cm_r_final_dimensions.rows_log],
            &r_hat_points_col[..self.constants.cm_r_final_dimensions.columns_log],
        ]
        .concat();

        let remaining_row_col_multiplier = r_hat_points_row
            [self.constants.cm_r_final_dimensions.rows_log..]
            .iter()
            .chain(&r_hat_points_col[self.constants.cm_r_final_dimensions.columns_log..])
            .map(|point| G::ScalarField::ONE - point)
            .product::<G::ScalarField>();

        let row_col_split_idx = r_hat_final_points_row_col.len()
            - log2(self.constants.cm_r_final_dimensions.row_com_dimension) as usize;
        let mut r_hat_final_points_matrix_l = tensor_chi(&r_hat_points_matrix)
            .into_iter()
            .skip(self.constants.n_layers + 1)
            .step_by(self.constants.n_layers + 2)
            .collect::<Vec<_>>();
        r_hat_final_points_matrix_l.resize(
            1 << self.constants.cm_r_final_dimensions.matrices_log,
            G::ScalarField::ZERO,
        );
        let r_hat_final_points_row_col_l =
            tensor_chi(&r_hat_final_points_row_col[..row_col_split_idx]);
        let r_hat_final_points_r = tensor_chi(&r_hat_final_points_row_col[row_col_split_idx..]);
        let r_hat_final_points_l = ark_std::cfg_iter!(r_hat_final_points_row_col_l)
            .flat_map(|&row_col_val| {
                ark_std::cfg_iter!(r_hat_final_points_matrix_l)
                    .map(move |matrix_val| row_col_val * matrix_val * remaining_row_col_multiplier)
            })
            .collect::<Vec<_>>();

        let [r_final_eval]: [_; 1] = verifier_state.next_points()?.map(|x: G| x.into_affine());
        HyraxTargetCommitment {
            omega: Some(r_final_eval),
            ltr: None,
            r_omega: None,
            lt: None,
            l: Some(r_hat_final_points_l),
            r: Some(r_hat_final_points_r),
            points: None,
        }
        .eval_verify(&commitments.cm_r_hat_final, verifier_state, client_schemes)?;

        // compute the HyraxTargetCommitment from the obtained info and run the evaluation proof
        let log2_l_len = log2(commitments.cm_t_hat.row_coms.len()) as usize;
        Ok(Self::new_target_commitment(
            tau,
            tensor_chi(&r_hat_points[..log2_l_len]),
            tensor_chi(&r_hat_points[log2_l_len..]),
            &cms_r_hat_restricted_coeffs,
            G::ScalarField::ZERO, // zero since there is no public value for this one, everything is in the shared commitment
            Some(r_final_eval), // positive since it is not included in the commitment, but it is included in the sum-check eval
            client_schemes,
        ))
    }

    fn compute_combined_t_hat_target_commitment<'a>(
        &self,
        verifier_state: &mut VerifierState,
        commitments: &LocalTrainingCommitments<G>,
        client_schemes: &ClientSchemes<'a, G>,
        forward_pass_deferred_evals: &ForwardPassDeferredEvals<G>,
        backward_pass_deferred_evals: &BackwardPassDeferredEvals<G>,
    ) -> ProofResult<HyraxTargetCommitment<G>> {
        // obtain T_hat restricted to the curve going through the 2 different points that T_hat is evaluated on
        // this will be a function of degree log(full dimension of T_hat)
        let t_hat_target_cms = [
            &forward_pass_deferred_evals.t_hat,
            &backward_pass_deferred_evals.t_hat,
        ];
        let cms_t_hat_restricted_coeffs = Self::coefficients_opening_verify(
            self.constants.cm_t_dimensions.total_log,
            verifier_state,
            client_schemes,
        )?;

        Self::degree_1_equality_verify(
            &cms_t_hat_restricted_coeffs,
            &t_hat_target_cms,
            verifier_state,
            client_schemes,
        )?;

        // determine the random point on the line
        let (tau, t_hat_points) = Self::random_point_degree_1(&t_hat_target_cms, verifier_state)?;

        // compute the public values so they are easy to evaluate below
        let log2_l_len = log2(commitments.cm_t_hat.row_coms.len()) as usize;
        let t_hat_l = tensor_chi(&t_hat_points[..log2_l_len]);
        let t_hat_r = tensor_chi(&t_hat_points[log2_l_len..]);

        // first construct zero matrices of the right size (we need that for the padding anyhow)
        let mut public_t_hat_matrices = self.compute_c_and_t_hat_zero_matrices();
        // then add the public values to those matrices
        // first row of all T-hat is all 1's (apart from the final two matrices of each iteration)
        for (idx, matrix) in public_t_hat_matrices.iter_mut().enumerate() {
            if idx % (self.constants.n_layers + 2) < self.constants.n_layers {
                matrix.slice_mut(s![0, ..]).fill(G::ScalarField::ONE);
            }
        }
        let t_hat_public_and_padding_matrix = client_schemes.poly_commit.combine_matrices(
            &public_t_hat_matrices,
            G::ScalarField::ONE,
            false,
            None,
        );

        let t_hat_l_public_and_padding =
            row_mult_matrix(&t_hat_l, &t_hat_public_and_padding_matrix);
        let t_hat_l_public_and_padding_r = dot(&t_hat_l_public_and_padding, &t_hat_r);

        // also evaluate the batch matrices
        let (t_hat_points_matrix, t_hat_points_row_col) =
            t_hat_points.split_at(self.constants.cm_t_dimensions.matrices_log);
        let (t_hat_points_row, t_hat_points_col) =
            t_hat_points_row_col.split_at(self.constants.cm_t_dimensions.rows_log);
        let batch_points_row_col = [
            &t_hat_points_row[..self.constants.cm_batch_samples_dimensions.rows_log],
            &t_hat_points_col[..self.constants.cm_batch_samples_dimensions.columns_log],
        ]
        .concat();

        let remaining_row_col_multiplier = t_hat_points_row
            [self.constants.cm_batch_samples_dimensions.rows_log..]
            .iter()
            .chain(&t_hat_points_col[self.constants.cm_batch_samples_dimensions.columns_log..])
            .map(|point| G::ScalarField::ONE - point)
            .product::<G::ScalarField>();

        let row_col_split_idx = batch_points_row_col.len()
            - log2(self.constants.cm_batch_samples_dimensions.row_com_dimension) as usize;
        let t_hat_points_matrix = tensor_chi(&t_hat_points_matrix);
        let batch_points_row_col_l = tensor_chi(&batch_points_row_col[..row_col_split_idx]);
        let batch_points_r = tensor_chi(&batch_points_row_col[row_col_split_idx..]);
        let mut batch_points_matrix_l =
            vec![
                G::ScalarField::ZERO;
                1 << self.constants.cm_batch_samples_dimensions.matrices_log
            ];
        for epoch_idx in 0..self.constants.n_epochs {
            batch_points_matrix_l[epoch_idx % self.constants.n_batches] +=
                t_hat_points_matrix[epoch_idx * (self.constants.n_layers + 2)]
        }
        let batch_points_l = ark_std::cfg_iter!(batch_points_row_col_l)
            .flat_map(|&row_col_val| {
                ark_std::cfg_iter!(batch_points_matrix_l)
                    .map(move |matrix_val| row_col_val * matrix_val * remaining_row_col_multiplier)
            })
            .collect::<Vec<_>>();

        let [batch_eval]: [_; 1] = verifier_state.next_points()?.map(|x: G| x.into_affine());
        HyraxTargetCommitment {
            omega: Some(batch_eval),
            ltr: None,
            r_omega: None,
            lt: None,
            l: Some(batch_points_l),
            r: Some(batch_points_r),
            points: None,
        }
        .eval_verify(
            &commitments.cm_batch_samples,
            verifier_state,
            client_schemes,
        )?;

        // compute the HyraxTargetCommitment from the obtained info and run the evaluation proof
        Ok(Self::new_target_commitment(
            tau,
            t_hat_l,
            t_hat_r,
            &cms_t_hat_restricted_coeffs,
            t_hat_l_public_and_padding_r, // positive since it is not included in the commitment, but is included in the sum-check eval
            Some(batch_eval), // positive since it is not included in the commitment, but is included in the sum-check eval
            client_schemes,
        ))
    }

    fn compute_combined_c_target_commitment<'a>(
        &self,
        verifier_state: &mut VerifierState,
        commitments: &LocalTrainingCommitments<G>,
        client_schemes: &ClientSchemes<'a, G>,
        forward_pass_deferred_evals: &ForwardPassDeferredEvals<G>,
        backward_pass_deferred_evals: &BackwardPassDeferredEvals<G>,
    ) -> ProofResult<HyraxTargetCommitment<G>> {
        // obtain C restricted to the curve going through the 4 different points that C is evaluated on
        // this will be a function of degree 3*log(full dimension of C)
        let c_target_cms = [
            &forward_pass_deferred_evals.c,
            &backward_pass_deferred_evals.c_one_one,
            &backward_pass_deferred_evals.c_one_two,
            &backward_pass_deferred_evals.c_two,
        ];
        let cms_c_restricted_coeffs = Self::coefficients_opening_verify(
            3 * self.constants.cm_c_dimensions.total_log,
            verifier_state,
            client_schemes,
        )?;

        Self::degree_3_equality_verify(
            &cms_c_restricted_coeffs,
            &c_target_cms,
            verifier_state,
            client_schemes,
        )?;

        // determine the random point on the line
        let (tau, c_points) =
            Self::random_point_degree_3(&c_target_cms, verifier_state, client_schemes)?;

        // compute the public values so they are easy to evaluate below
        let log2_l_len = log2(commitments.cm_c.row_coms.len()) as usize;
        let c_l = tensor_chi(&c_points[..log2_l_len]);
        let c_r = tensor_chi(&c_points[log2_l_len..]);

        let zero_c_matrices = self.compute_c_and_t_hat_zero_matrices();
        let c_padding_matrix = client_schemes.poly_commit.combine_matrices(
            &zero_c_matrices,
            G::ScalarField::ONE,
            false,
            None,
        );

        let c_l_padding = row_mult_matrix(&c_l, &c_padding_matrix);
        let c_l_padding_r = dot(&c_l_padding, &c_r);

        // compute the HyraxTargetCommitment from the obtained info and run the evaluation proof
        Ok(Self::new_target_commitment(
            tau,
            c_l,
            c_r,
            &cms_c_restricted_coeffs,
            -c_l_padding_r, // negative since it is included in the commitment and not in the sum-check eval
            None,
            client_schemes,
        ))
    }

    fn eval_verify_combined_w_target_commitment<'a>(
        &self,
        verifier_state: &mut VerifierState,
        commitments: &LocalTrainingCommitments<G>,
        cm_initial_weights: &HyraxCommitment<G>,
        client_schemes: &ClientSchemes<'a, G>,
        forward_pass_deferred_evals: &ForwardPassDeferredEvals<G>,
        backward_pass_deferred_evals: &BackwardPassDeferredEvals<G>,
    ) -> ProofResult<()> {
        // obtain W restricted to the curve going through the 2 different points that W is evaluate don
        // this will be a function of degree log(full dimension of W)
        let w_target_cms = [
            &forward_pass_deferred_evals.w,
            &backward_pass_deferred_evals.w,
        ];
        let cms_w_restricted_coeffs = Self::coefficients_opening_verify(
            self.constants.cm_w_full_dimensions.total_log,
            verifier_state,
            client_schemes,
        )?;

        Self::degree_1_equality_verify(
            &cms_w_restricted_coeffs,
            &w_target_cms,
            verifier_state,
            client_schemes,
        )?;

        let (tau, w_points) = Self::random_point_degree_1(&w_target_cms, verifier_state)?;

        // compute the public values so they are easy to evaluate below
        let log2_l_len = log2(self.constants.cm_w_full_dimensions.num_row_coms) as usize;
        let w_l = tensor_chi(&w_points[..log2_l_len]);
        let w_r = tensor_chi(&w_points[log2_l_len..]);

        // construct the public matrices
        let mut public_w_matrices = self.compute_w_matrices();
        // then add the public values to those matrices
        // W, in each iteration, for l = L+1 is zero with all 1's diagonal
        for matrix in public_w_matrices[self.constants.n_layers + 1..]
            .iter_mut()
            .step_by(self.constants.n_layers + 2)
        {
            *matrix = Array2::from_diag_elem(matrix.nrows(), G::ScalarField::ONE);
        }
        let w_public_matrix = client_schemes.poly_commit.combine_matrices(
            &public_w_matrices,
            G::ScalarField::ZERO,
            false,
            None,
        );

        let w_l_public = row_mult_matrix(&w_l, &w_public_matrix);
        let w_l_public_r = dot(&w_l_public, &w_r);

        // evaluate W
        let (w_points_matrix, w_points_row_col) =
            w_points.split_at(self.constants.cm_w_full_dimensions.matrices_log);

        let row_col_split_idx = w_points_row_col.len()
            - log2(
                self.constants
                    .cm_w_single_iteration_dimensions
                    .row_com_dimension,
            ) as usize;
        let w_points_matrix_l = tensor_chi(&w_points_matrix);
        let w_single_points_row_col_l = tensor_chi(&w_points_row_col[..row_col_split_idx]);
        let w_single_points_r = tensor_chi(&w_points_row_col[row_col_split_idx..]);

        let padding = vec![
            G::ScalarField::ZERO;
            (1 << self.constants.cm_w_single_iteration_dimensions.matrices_log)
                - (self.constants.n_layers + 2)
        ];
        let mut w_single_points_l_all = w_points_matrix_l
            .chunks_exact(self.constants.n_layers + 2)
            .take(self.constants.n_epochs)
            .rev()
            .map(|w_points_matrix_l_chunk| {
                ark_std::cfg_iter!(w_single_points_row_col_l)
                    .flat_map(|&row_col_val| {
                        ark_std::cfg_iter!(w_points_matrix_l_chunk)
                            .chain(ark_std::cfg_iter!(padding))
                            .map(move |matrix_val| row_col_val * matrix_val)
                    })
                    .collect::<Vec<_>>()
            });

        // evaluate the g_hat part
        let cm_g_hat_all = &commitments.cm_g_hat_all;
        let mut cm_g_hat_all = cm_g_hat_all.iter().rev().skip(1);

        // initial step to initialize the sums
        let mut w_single_points_l_sum = w_single_points_l_all.next().unwrap();
        let mut t_prime = if let Some(cm_g_hat) = cm_g_hat_all.next() {
            let mut t_prime = G::msm_unchecked(&cm_g_hat.row_coms, &w_single_points_l_sum);

            for cm_g_hat in cm_g_hat_all {
                ark_std::cfg_iter_mut!(w_single_points_l_sum)
                    .zip(ark_std::cfg_into_iter!(
                        w_single_points_l_all.next().unwrap()
                    ))
                    .for_each(|(s, e)| *s += e);
                t_prime += G::msm_unchecked(&cm_g_hat.row_coms, &w_single_points_l_sum);
            }
            ark_std::cfg_iter_mut!(w_single_points_l_sum)
                .zip(ark_std::cfg_into_iter!(
                    w_single_points_l_all.next().unwrap()
                ))
                .for_each(|(s, e)| *s += e);
            t_prime
        } else {
            G::ZERO
        };

        // compute w - g_hat_sum
        t_prime = G::msm_unchecked(&cm_initial_weights.row_coms, &w_single_points_l_sum) - t_prime;

        // compute omega (zeta - public value)
        let mut zeta = cms_w_restricted_coeffs[0].into_group();
        let mut tau_to_the_power_i = tau;
        for cm in cms_w_restricted_coeffs.iter().skip(1) {
            zeta += cm.mul(tau_to_the_power_i);
            tau_to_the_power_i *= tau;
        }

        let omega = zeta
            - client_schemes
                .vector_commit
                .get_g_unique()
                .mul(w_l_public_r);

        // we need to do a custom version of eval_verify
        DotProductProof {
            g_vec: &client_schemes.vector_commit.get_g_vec()[..w_single_points_r.len()],
            a_vec: &w_single_points_r,
            xi: Some(&t_prime.into_affine()),
            r_xi: None,
            tau: Some(&omega.into_affine()),
            r_tau: None,
            x_vec: None,
        }
        .verify(verifier_state, client_schemes)
    }
}

fn degree_1_curve_coeffs<F: PrimeField>(points: [&[F]; 2]) -> Vec<[F; 2]> {
    ark_std::cfg_iter!(points[0])
        .zip(ark_std::cfg_iter!(points[1]))
        .map(|(&r0_val, &r1_val)| {
            let c0 = r0_val;
            let c1 = r1_val - r0_val;
            [c0, c1]
        })
        .collect()
}

fn degree_3_curve_coeffs<F: PrimeField>(
    points: [&[F]; 4],
    constants: &Constants<F>,
) -> Vec<[F; 4]> {
    let minus_two = -F::from(2);
    let minus_six = -F::from(6);

    ark_std::cfg_iter!(points[0])
        .zip(ark_std::cfg_iter!(points[1]))
        .zip(ark_std::cfg_iter!(points[2]))
        .zip(ark_std::cfg_iter!(points[3]))
        .map(|(((&r0_val, r1_val), r2_val), r3_val)| {
            let c0 = r0_val;
            let neg_c0 = -c0;

            let c3 = constants.one_over_two
                * (constants.one_over_three * (neg_c0 + r3_val) + r1_val - r2_val);

            let c2 = constants.one_over_two * (c0 + minus_two * r1_val + r2_val + minus_six * c3);

            let c1 = neg_c0 + r1_val - c2 - c3;

            [c0, c1, c2, c3]
        })
        .collect()
}

fn curve_eval<const N: usize, F: PrimeField>(point: F, curve_coeffs: &[[F; N]]) -> Vec<F> {
    let mut point_powers = [F::ONE; N];
    point_powers[1] = point;
    for idx in 2..N {
        point_powers[idx] = point_powers[idx - 1] * point
    }

    ark_std::cfg_iter!(curve_coeffs)
        .map(|coeffs| coeffs.iter().zip(point_powers).map(|(&x, y)| x * y).sum())
        .collect::<Vec<_>>()
}

fn curve_constrained_eval<const N: usize, F: PrimeField>(
    point: F,
    curve_coeffs: &[[F; N]],
    dp_array: &[F],
) -> F {
    let dp_eval_point = curve_eval(point, curve_coeffs);

    multi_collapse(dp_array, &dp_eval_point)[0]
}

// we assume the evaluations are on the points 0,...,n-1, for a vector of length n
// we return the coefficients in order c0,...,cn
fn lagrange_interpolation<F: PrimeField>(evaluations: &[F]) -> Vec<F> {
    let points = (0..evaluations.len() as u64)
        .map(F::from)
        .collect::<Vec<_>>();

    // compute all elements of the Lagrange basis polynomial
    let displacements = points
        .iter()
        .map(|&point| DensePolynomial::from_coefficients_vec(vec![-point, F::ONE]))
        .collect::<Vec<_>>();
    let displacements_prod = DenseOrSparsePolynomial::from(displacements.iter().fold(
        DensePolynomial::from_coefficients_vec(vec![F::ONE]),
        |prod, displacement| prod.naive_mul(displacement),
    ));
    let barycentric_weights = points.iter().enumerate().map(|(cur_idx, &cur_point)| {
        points
            .iter()
            .enumerate()
            .fold(F::ONE, |mut weight, (other_idx, other_point)| {
                if other_idx != cur_idx {
                    weight *= cur_point - other_point
                }
                weight
            })
            .inverse()
            .expect("Inverse should exist")
    });

    let mut interpolation = DensePolynomial::zero();
    for ((evaluation, displacement), barycentric_weight) in evaluations
        .iter()
        .zip(displacements)
        .zip(barycentric_weights)
    {
        let (div, rem) = displacements_prod
            .divide_with_q_and_r(&DenseOrSparsePolynomial::from(displacement))
            .expect("This division should not fail.");
        debug_assert!(rem.is_zero());
        interpolation += &(div * (barycentric_weight * evaluation));
    }

    let mut result_coeffs = interpolation.coeffs;
    result_coeffs.resize(evaluations.len(), F::zero());
    result_coeffs
}
