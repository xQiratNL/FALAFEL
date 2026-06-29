use crate::client::ClientSchemes;
use crate::prelude::*;
use crate::proof::commitments::LocalTrainingCommitments;
use crate::proof::dot_product::{DotProductProof, DotProductProofSeparator};
use crate::proof::util::{dot, powers};
use crate::proof::{LocalTrainingProof, LocalTrainingProofConstants, ProofWitnessFull};
use crate::util::permute;
use ark_ff::{AdditiveGroup, Field, PrimeField};
use ark_std::UniformRand;
use ndarray::Order::ColumnMajor;
use ndarray::{Array2, ShapeBuilder};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::codecs::arkworks_algebra::{
    FieldDomainSeparator, GroupDomainSeparator, GroupToUnitDeserialize, GroupToUnitSerialize,
    UnitToField,
};
use spongefish::{
    ByteDomainSeparator, BytesToUnitDeserialize, BytesToUnitSerialize, DomainSeparator,
    DuplexSpongeInterface, ProofResult, ProverState, VerifierState,
};
use std::cmp::max;
use std::ops::Mul;

pub trait BatchCorrectnessSeparator<G: CurveGroup> {
    fn add_batch_proof<T: Value>(self, constants: &LocalTrainingProofConstants<T>) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> BatchCorrectnessSeparator<G> for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    // this describes the batch correctness product protocol, i.e., verifier/prover messages
    fn add_batch_proof<T: Value>(self, constants: &LocalTrainingProofConstants<T>) -> Self {
        self.add_bytes(size_of::<usize>() * constants.n_samples, "permutation")
            // note: we assume that the rest of the statement (cm_d, batch commitments) has already been added
            // challenge
            .challenge_scalars(1, "challenge")
            // response
            .add_points(2, "target commitments (sums of components)")
            // dot product proofs
            .add_dot_product(max(
                constants.cm_batch_samples_dimensions.row_com_dimension,
                constants.cm_batch_labels_dimensions.row_com_dimension,
            ))
            .add_dot_product((constants.n_features + 1) * constants.n_samples)
            .add_dot_product(constants.n_labels * constants.n_samples)
    }
}

impl<'a, G: CurveGroup, T: Value> ProofWitnessFull<'a, G, T> {
    // based on: Algorithm 8 our work
    // note: we assume that the statement (cm_d, batch commitments) has already been sent to the verifier
    pub fn prove_batch_correctness(
        &self,
        prover_state: &mut ProverState,
        client_schemes: &ClientSchemes<G>,
        commitments: &LocalTrainingCommitments<G>,
    ) -> ProofResult<()> {
        let opening_batch_samples = commitments
            .opening_batch_samples
            .as_ref()
            .ok_or_else(|| spongefish::ProofError::SerializationError)?;
        let opening_batch_labels = commitments
            .opening_batch_labels
            .as_ref()
            .ok_or_else(|| spongefish::ProofError::SerializationError)?;

        // 0
        // we only need to add the permutation
        prover_state.add_bytes(
            &ark_std::cfg_into_iter!(self.permutation)
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        // all the commitments already done in the commitment phase (so elsewhere)

        // 1
        // challenge
        let [z]: [G::ScalarField; 1] = prover_state.challenge_scalars()?;

        // 2
        // response:  compute + store powers of z and compute t
        // note: we need one power of z for each element in the hyrax poly commit
        let n_scalars_with_padding = self.constants.cm_batch_samples_dimensions.row_com_dimension
            * self.constants.cm_batch_samples_dimensions.num_row_coms
            + self.constants.cm_batch_labels_dimensions.row_com_dimension
                * self.constants.cm_batch_labels_dimensions.num_row_coms;
        let powers_of_z = powers(&z, n_scalars_with_padding, false);

        let t_1 = ark_std::cfg_iter!(&opening_batch_samples.input_matrix)
            .zip(ark_std::cfg_chunks!(
                powers_of_z[..self.constants.cm_batch_samples_dimensions.row_com_dimension
                    * self.constants.cm_batch_samples_dimensions.num_row_coms],
                self.constants.cm_batch_samples_dimensions.row_com_dimension
            ))
            .map(|(x, y)| dot(x, y))
            .sum::<G::ScalarField>();
        let t_2 = ark_std::cfg_iter!(&opening_batch_labels.input_matrix)
            .zip(ark_std::cfg_chunks!(
                powers_of_z[self.constants.cm_batch_samples_dimensions.row_com_dimension
                    * self.constants.cm_batch_samples_dimensions.num_row_coms..],
                self.constants.cm_batch_labels_dimensions.row_com_dimension
            ))
            .map(|(x, y)| dot(x, y))
            .sum::<G::ScalarField>();
        // commit to t and send it
        let r_t_1 = G::ScalarField::rand(prover_state.rng());
        let r_t_2 = G::ScalarField::rand(prover_state.rng());
        let cm_t_1 = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&t_1, &r_t_1);
        let cm_t_2 = client_schemes
            .vector_commit
            .value_commit_unique_g_same_h(&t_2, &r_t_2);
        prover_state.add_points(&[cm_t_1.into_group(), cm_t_2.into_group()])?;

        // 3: compute the opening of cm_d_hat
        let offset_samples = self.constants.cm_batch_samples_dimensions.row_com_dimension;
        let offset_labels = self.constants.cm_batch_labels_dimensions.row_com_dimension;
        let mut z_offset_powers_samples = powers(
            &z.pow([offset_samples as u64]),
            self.constants.cm_batch_samples_dimensions.num_row_coms + 1,
            true,
        );
        let extra_z_power_labels = z_offset_powers_samples.pop().unwrap();
        let z_offset_powers_labels = powers(
            &z.pow([offset_labels as u64]),
            self.constants.cm_batch_labels_dimensions.num_row_coms,
            true,
        );

        // compute the opening of cm_d_hat (d_hat_vec, r_d_hat)
        let r_d_hat = dot(&opening_batch_samples.randomness, &z_offset_powers_samples)
            + extra_z_power_labels * dot(&opening_batch_labels.randomness, &z_offset_powers_labels);

        let mut d_hat_vec = vec![G::ScalarField::ZERO; max(offset_samples, offset_labels)];
        for (input_row_labels, &z_power_labels) in opening_batch_labels
            .input_matrix
            .iter()
            .zip(&z_offset_powers_labels)
        {
            ark_std::cfg_iter_mut!(d_hat_vec)
                .zip(ark_std::cfg_into_iter!(input_row_labels))
                .for_each(|(a, b)| *a += z_power_labels * b);
        }

        ark_std::cfg_iter_mut!(d_hat_vec).for_each(|val| *val *= extra_z_power_labels);

        for (input_row_samples, &z_power_samples) in opening_batch_samples
            .input_matrix
            .iter()
            .zip(&z_offset_powers_samples)
        {
            ark_std::cfg_iter_mut!(d_hat_vec)
                .zip(ark_std::cfg_into_iter!(input_row_samples))
                .for_each(|(a, b)| *a += z_power_samples * b);
        }

        // 4: first dot product proof
        DotProductProof {
            g_vec: &client_schemes.vector_commit.get_g_vec()[..max(offset_samples, offset_labels)],
            a_vec: &powers_of_z[..max(offset_samples, offset_labels)],
            xi: None,
            r_xi: Some(&r_d_hat),
            tau: None,
            r_tau: Some(&(r_t_1 + r_t_2)),
            x_vec: Some(&d_hat_vec),
        }
        .prove(prover_state, client_schemes)?;

        // 5: second dot product proof (we differ from the paper by permuting g and x, rather than taking the inverse permutation of z, but that has the same effect)
        // we need to select the right powers of z (the ones actually used in the other dot product, i.e., the hyrax commitment added a bunch of padding, we need to take that into account here).
        let row_dim_samples = 1 << self.constants.cm_batch_samples_dimensions.rows_log;
        let matrix_dim_samples = 1 << self.constants.cm_batch_samples_dimensions.matrices_log;

        let powers_of_z_matrix_samples = Array2::from_shape_vec(
            (
                self.constants.cm_batch_samples_dimensions.num_row_coms,
                offset_samples,
            ),
            powers_of_z[..self.constants.cm_batch_samples_dimensions.row_com_dimension
                * self.constants.cm_batch_samples_dimensions.num_row_coms]
                .to_vec(),
        )
        .unwrap()
        .flatten_with_order(ColumnMajor)
        .to_vec();

        let mut powers_of_z_selected_samples = vec![];
        for idx_batch in 0..self.constants.n_batches {
            for idx_col in 0..if idx_batch == self.constants.n_batches - 1 {
                self.constants.n_samples - idx_batch * self.constants.batch_size
            } else {
                self.constants.batch_size
            } {
                for idx_row in 0..self.constants.n_features + 1 {
                    powers_of_z_selected_samples.push(
                        powers_of_z_matrix_samples[idx_batch
                            + idx_row * matrix_dim_samples
                            + idx_col * matrix_dim_samples * row_dim_samples],
                    );
                }
            }
        }
        // we need a power of two for the length, so add dummies if not so
        powers_of_z_selected_samples.resize(
            powers_of_z_selected_samples.len().next_power_of_two(),
            G::ScalarField::ZERO,
        );

        // we also need to permute the generators according to the given permutation
        let sample_shape = (self.constants.n_features + 1, self.constants.n_samples)
            .strides((1, self.constants.n_features + 1));
        // for g_vec we then need to append dummy generators since we need a 2 power number of generators
        let g_vec_samples = &client_schemes.vector_commit.get_g_vec()
            [..self.constants.n_samples * (self.constants.n_features + 1)];
        let mut g_vec_permuted_samples = permute(g_vec_samples, sample_shape, self.permutation);
        g_vec_permuted_samples.extend_from_slice(
            &client_schemes.vector_commit.get_g_vec()[self.constants.n_samples
                * (self.constants.n_features + 1)
                ..powers_of_z_selected_samples.len()],
        );

        debug_assert_eq!(
            dot(
                &self.dataset_scalar_samples_permuted,
                &powers_of_z_selected_samples
            ),
            t_1
        );

        DotProductProof {
            g_vec: &g_vec_permuted_samples,
            a_vec: &powers_of_z_selected_samples,
            xi: None,
            r_xi: Some(self.r_dataset_samples),
            tau: None,
            r_tau: Some(&r_t_1),
            x_vec: Some(&self.dataset_scalar_samples_permuted),
        }
        .prove(prover_state, client_schemes)?;

        // 6: third dot product proof (we differ from the paper by permuting g and x, rather than taking the inverse permutation of z, but that has the same effect)
        // we need to select the right powers of z (the ones actually used in the other dot product, i.e., the hyrax commitment added a bunch of padding, we need to take that into account here).
        let row_dim_labels = 1 << self.constants.cm_batch_labels_dimensions.rows_log;
        let matrix_dim_labels = 1 << self.constants.cm_batch_labels_dimensions.matrices_log;

        let powers_of_z_matrix_labels = Array2::from_shape_vec(
            (
                self.constants.cm_batch_labels_dimensions.num_row_coms,
                offset_labels,
            ),
            powers_of_z[..self.constants.cm_batch_labels_dimensions.row_com_dimension
                * self.constants.cm_batch_labels_dimensions.num_row_coms]
                .to_vec(),
        )
        .unwrap()
        .flatten_with_order(ColumnMajor)
        .to_vec();

        let mut powers_of_z_selected_labels = vec![];
        for idx_batch in 0..self.constants.n_batches {
            for idx_col in 0..if idx_batch == self.constants.n_batches - 1 {
                self.constants.n_samples - idx_batch * self.constants.batch_size
            } else {
                self.constants.batch_size
            } {
                for idx_row in 0..self.constants.n_labels {
                    powers_of_z_selected_labels.push(
                        powers_of_z_matrix_labels[idx_batch
                            + idx_row * matrix_dim_labels
                            + idx_col * matrix_dim_labels * row_dim_labels],
                    );
                }
            }
        }
        // we need a power of two for the length, so add dummies if not so
        powers_of_z_selected_labels.resize(
            powers_of_z_selected_labels.len().next_power_of_two(),
            G::ScalarField::ZERO,
        );

        // we also need to permute the generators according to the given permutation
        let label_shape = (self.constants.n_labels, self.constants.n_samples)
            .strides((1, self.constants.n_labels));
        // for g_vec we then need to append dummy generators since we need a 2 power number of generators
        let g_vec_labels = &client_schemes.vector_commit.get_g_vec()
            [..self.constants.n_samples * self.constants.n_labels];
        let mut g_vec_permuted_labels = permute(g_vec_labels, label_shape, self.permutation);
        g_vec_permuted_labels.extend_from_slice(
            &client_schemes.vector_commit.get_g_vec()[self.constants.n_samples
                * self.constants.n_labels
                ..powers_of_z_selected_labels.len()],
        );

        debug_assert_eq!(
            dot(
                &self.dataset_scalar_labels_permuted,
                &powers_of_z_selected_labels
            ),
            t_2 * extra_z_power_labels.inverse().unwrap()
        );

        DotProductProof {
            g_vec: &g_vec_permuted_labels,
            a_vec: &powers_of_z_selected_labels,
            xi: None,
            r_xi: Some(self.r_dataset_labels),
            tau: None,
            r_tau: Some(&(r_t_2 * extra_z_power_labels.inverse().unwrap())),
            x_vec: Some(&self.dataset_scalar_labels_permuted),
        }
        .prove(prover_state, client_schemes)?;

        Ok(())
    }
}

impl<G: CurveGroup, T: Value> LocalTrainingProof<G, T> {
    // based on: Algorithm 8 our work
    pub fn verify_batch_correctness<'a>(
        &self,
        verifier_state: &mut VerifierState,
        commitments: &LocalTrainingCommitments<G>,
        client_schemes: &ClientSchemes<'a, G>,
        cm_dataset_samples: &G::Affine,
        cm_dataset_labels: &G::Affine,
    ) -> ProofResult<()> {
        // 0
        // the batch commitments are already included in commitments
        // we just need the permutation
        let mut permutation = vec![0; size_of::<usize>() * self.constants.n_samples];
        verifier_state.fill_next_bytes(&mut permutation)?;
        let permutation = ark_std::cfg_chunks!(permutation, size_of::<usize>())
            .map(|chunk| usize::from_le_bytes(chunk.try_into().expect("This should fit!")))
            .collect::<Vec<_>>();

        // 1
        let [z]: [G::ScalarField; 1] = verifier_state.challenge_scalars()?;

        // 2
        // we only have to get cm_t and compute the powers of z here for verifying steps 4 and 5
        // note: we need one power of z for each element in the hyrax poly commit
        let n_scalars_with_padding = self.constants.cm_batch_samples_dimensions.row_com_dimension
            * self.constants.cm_batch_samples_dimensions.num_row_coms
            + self.constants.cm_batch_labels_dimensions.row_com_dimension
                * self.constants.cm_batch_labels_dimensions.num_row_coms;
        let powers_of_z = powers(&z, n_scalars_with_padding, false);
        let [cm_t_1, cm_t_2]: [_; 2] = verifier_state.next_points()?.map(|cm: G| cm.into_affine());

        // 3
        // compute cm_d_hat
        let offset_samples = self.constants.cm_batch_samples_dimensions.row_com_dimension;
        let offset_labels = self.constants.cm_batch_labels_dimensions.row_com_dimension;
        let mut z_offset_powers_samples = powers(
            &z.pow([offset_samples as u64]),
            self.constants.cm_batch_samples_dimensions.num_row_coms + 1,
            true,
        );
        let extra_z_power_labels = z_offset_powers_samples.pop().unwrap();
        let z_offset_powers_labels = powers(
            &z.pow([offset_labels as u64]),
            self.constants.cm_batch_labels_dimensions.num_row_coms,
            true,
        );

        let cm_d_hat = (G::msm_bigint(
            &commitments.cm_batch_samples.row_coms,
            &ark_std::cfg_iter!(z_offset_powers_samples)
                .map(|value| value.into_bigint())
                .collect::<Vec<_>>(),
        ) + G::msm_bigint(
            &commitments.cm_batch_labels.row_coms,
            &ark_std::cfg_iter!(z_offset_powers_labels)
                .map(|value| value.into_bigint())
                .collect::<Vec<_>>(),
        )
        .mul(extra_z_power_labels))
        .into_affine();

        // 4: first dot product proof
        DotProductProof::<G> {
            g_vec: &client_schemes.vector_commit.get_g_vec()[..max(offset_samples, offset_labels)],
            a_vec: &powers_of_z[..max(offset_samples, offset_labels)],
            xi: Some(&cm_d_hat),
            r_xi: None,
            tau: Some(&(cm_t_1 + cm_t_2).into_affine()),
            r_tau: None,
            x_vec: None,
        }
        .verify(verifier_state, client_schemes)?;

        // 5: second dot product proof (we differ from the paper by permuting g and (for the prover also x), rather than taking the inverse permutation of z, but that has the same effect)
        // we need to select the right powers of z (the ones actually used in the other dot product, i.e., the hyrax commitment added a bunch of padding, we need to take that into account here).
        let row_dim_samples = 1 << self.constants.cm_batch_samples_dimensions.rows_log;
        let matrix_dim_samples = 1 << self.constants.cm_batch_samples_dimensions.matrices_log;

        let powers_of_z_matrix_samples = Array2::from_shape_vec(
            (
                self.constants.cm_batch_samples_dimensions.num_row_coms,
                offset_samples,
            ),
            powers_of_z[..self.constants.cm_batch_samples_dimensions.row_com_dimension
                * self.constants.cm_batch_samples_dimensions.num_row_coms]
                .to_vec(),
        )
        .unwrap()
        .flatten_with_order(ColumnMajor)
        .to_vec();

        let mut powers_of_z_selected_samples = vec![];
        for idx_batch in 0..self.constants.n_batches {
            for idx_col in 0..if idx_batch == self.constants.n_batches - 1 {
                self.constants.n_samples - idx_batch * self.constants.batch_size
            } else {
                self.constants.batch_size
            } {
                for idx_row in 0..self.constants.n_features + 1 {
                    powers_of_z_selected_samples.push(
                        powers_of_z_matrix_samples[idx_batch
                            + idx_row * matrix_dim_samples
                            + idx_col * matrix_dim_samples * row_dim_samples],
                    );
                }
            }
        }
        // we need a power of two for the length, so add dummies if not so
        powers_of_z_selected_samples.resize(
            powers_of_z_selected_samples.len().next_power_of_two(),
            G::ScalarField::ZERO,
        );

        // we also need to permute the generators according to the given permutation
        let sample_shape = (self.constants.n_features + 1, self.constants.n_samples)
            .strides((1, self.constants.n_features + 1));
        // for g_vec we then need to append dummy generators since we need a 2 power number of generators
        let g_vec_samples = &client_schemes.vector_commit.get_g_vec()
            [..self.constants.n_samples * (self.constants.n_features + 1)];
        let mut g_vec_permuted_samples = permute(g_vec_samples, sample_shape, &permutation);
        g_vec_permuted_samples.extend_from_slice(
            &client_schemes.vector_commit.get_g_vec()[self.constants.n_samples
                * (self.constants.n_features + 1)
                ..powers_of_z_selected_samples.len()],
        );

        DotProductProof::<G> {
            g_vec: &g_vec_permuted_samples,
            a_vec: &powers_of_z_selected_samples,
            xi: Some(cm_dataset_samples),
            r_xi: None,
            tau: Some(&cm_t_1),
            r_tau: None,
            x_vec: None,
        }
        .verify(verifier_state, client_schemes)?;

        // 6: third dot product proof (we differ from the paper by permuting g and (for the prover also x), rather than taking the inverse permutation of z, but that has the same effect)
        // we need to select the right powers of z (the ones actually used in the other dot product, i.e., the hyrax commitment added a bunch of padding, we need to take that into account here).
        let row_dim_labels = 1 << self.constants.cm_batch_labels_dimensions.rows_log;
        let matrix_dim_labels = 1 << self.constants.cm_batch_labels_dimensions.matrices_log;

        let powers_of_z_matrix_labels = Array2::from_shape_vec(
            (
                self.constants.cm_batch_labels_dimensions.num_row_coms,
                offset_labels,
            ),
            powers_of_z[..self.constants.cm_batch_labels_dimensions.row_com_dimension
                * self.constants.cm_batch_labels_dimensions.num_row_coms]
                .to_vec(),
        )
        .unwrap()
        .flatten_with_order(ColumnMajor)
        .to_vec();

        let mut powers_of_z_selected_labels = vec![];
        for idx_batch in 0..self.constants.n_batches {
            for idx_col in 0..if idx_batch == self.constants.n_batches - 1 {
                self.constants.n_samples - idx_batch * self.constants.batch_size
            } else {
                self.constants.batch_size
            } {
                for idx_row in 0..self.constants.n_labels {
                    powers_of_z_selected_labels.push(
                        powers_of_z_matrix_labels[idx_batch
                            + idx_row * matrix_dim_labels
                            + idx_col * matrix_dim_labels * row_dim_labels],
                    );
                }
            }
        }
        // we need a power of two for the length, so add dummies if not so
        powers_of_z_selected_labels.resize(
            powers_of_z_selected_labels.len().next_power_of_two(),
            G::ScalarField::ZERO,
        );

        // we also need to permute the generators according to the given permutation
        let label_shape = (self.constants.n_labels, self.constants.n_samples)
            .strides((1, self.constants.n_labels));
        // for g_vec we then need to append dummy generators since we need a 2 power number of generators
        let g_vec_labels = &client_schemes.vector_commit.get_g_vec()
            [..self.constants.n_samples * self.constants.n_labels];
        let mut g_vec_permuted_labels = permute(g_vec_labels, label_shape, &permutation);
        g_vec_permuted_labels.extend_from_slice(
            &client_schemes.vector_commit.get_g_vec()[self.constants.n_samples
                * self.constants.n_labels
                ..powers_of_z_selected_labels.len()],
        );

        DotProductProof::<G> {
            g_vec: &g_vec_permuted_labels,
            a_vec: &powers_of_z_selected_labels,
            xi: Some(cm_dataset_labels),
            r_xi: None,
            tau: Some(
                &cm_t_2
                    .mul(extra_z_power_labels.inverse().unwrap())
                    .into_affine(),
            ),
            r_tau: None,
            x_vec: None,
        }
        .verify(verifier_state, client_schemes)?;

        Ok(())
    }
}
