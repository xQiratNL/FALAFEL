use crate::client::ClientSchemes;
use crate::crypto_schemes::{HyraxCommitment, HyraxCommitmentOpening};
use crate::prelude::*;
use crate::proof::{LocalTrainingProof, ProofWitnessFull};
use crate::util::IntoField;
use ark_ff::{AdditiveGroup, Field};
use ark_std::log2;
use ndarray::{Array, Array2, Axis, Dim, concatenate, s};
use rand::{CryptoRng, Rng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::codecs::arkworks_algebra::{GroupToUnitDeserialize, GroupToUnitSerialize};
use spongefish::{ProofResult, ProverState, VerifierState};

pub struct LocalTrainingCommitments<G: CurveGroup> {
    // commitments
    pub cm_batch_samples: HyraxCommitment<G>,
    pub cm_batch_labels: HyraxCommitment<G>,
    pub cm_u: HyraxCommitment<G>,
    pub cm_c: HyraxCommitment<G>,
    pub cm_t_hat: HyraxCommitment<G>,
    pub cm_t: HyraxCommitment<G>,
    pub cm_g_bar: HyraxCommitment<G>,
    pub cm_r_hat: HyraxCommitment<G>,
    pub cm_r_hat_final: HyraxCommitment<G>,
    pub cm_r: HyraxCommitment<G>,
    pub cm_g_hat_all: Vec<HyraxCommitment<G>>,

    // witnesses
    pub opening_batch_samples: Option<HyraxCommitmentOpening<G>>,
    pub opening_batch_labels: Option<HyraxCommitmentOpening<G>>,
    pub opening_u: Option<HyraxCommitmentOpening<G>>,
    pub w_with_public_input_matrix: Option<Vec<Vec<G::ScalarField>>>,
    pub w_final_input_matrix: Option<Vec<Vec<G::ScalarField>>>,
    pub opening_c: Option<HyraxCommitmentOpening<G>>,
    pub opening_c_with_public: Option<HyraxCommitmentOpening<G>>,
    pub opening_t_hat: Option<HyraxCommitmentOpening<G>>,
    pub opening_t_hat_with_public: Option<HyraxCommitmentOpening<G>>,
    pub opening_t: Option<HyraxCommitmentOpening<G>>,
    pub opening_t_with_public: Option<HyraxCommitmentOpening<G>>,
    pub opening_g_bar: Option<HyraxCommitmentOpening<G>>,
    pub opening_r_hat: Option<HyraxCommitmentOpening<G>>,
    pub opening_r_hat_with_public: Option<HyraxCommitmentOpening<G>>,
    pub opening_r_hat_final: Option<HyraxCommitmentOpening<G>>,
    pub opening_r: Option<HyraxCommitmentOpening<G>>,
    pub opening_g_hat_all: Option<Vec<HyraxCommitmentOpening<G>>>,
}

impl<'a, G: CurveGroup, T: Value> ProofWitnessFull<'a, G, T> {
    pub fn compute_commitments(
        &self,
        client_schemes: &ClientSchemes<G>,
        rng: &mut (impl Rng + CryptoRng),
    ) -> LocalTrainingCommitments<G> {
        // batch commitments
        let batch_matrices_samples = self
            .dataset_shuffled
            .batches(self.constants.batch_size)
            .0
            .map(|matrix| matrix.mapv(|value| value.to_field()))
            // add a dummy first row with all zeroes
            .map(|matrix| {
                concatenate![
                    Axis(0),
                    Array2::from_elem((1, matrix.ncols()), G::ScalarField::ZERO,),
                    matrix
                ]
            })
            .collect::<Vec<_>>();
        let (cm_batch_samples, opening_batch_samples) =
            client_schemes.poly_commit.poly_commit_matrices(
                &batch_matrices_samples,
                G::ScalarField::ZERO,
                true,
                None,
                rng,
            );
        let batch_matrices_labels = self
            .dataset_shuffled
            .batches(self.constants.batch_size)
            .1
            .map(|matrix| matrix.mapv(|value| value.to_field()))
            .collect::<Vec<_>>();
        let (cm_batch_labels, opening_batch_labels) =
            client_schemes.poly_commit.poly_commit_matrices(
                &batch_matrices_labels,
                G::ScalarField::ZERO,
                true,
                None,
                rng,
            );

        // U (final layer only) for each iteration
        let u_matrices = self
            .feed_forward_steps
            .iter()
            .map(|step| {
                step.post_activations_vector
                    .last()
                    .unwrap()
                    .mapv(|value| value.to_field())
            })
            .collect::<Vec<_>>();
        let (cm_u, opening_u) = client_schemes.poly_commit.poly_commit_matrices(
            &u_matrices,
            G::ScalarField::ZERO,
            true,
            None,
            rng,
        );

        let trunc_factor =
            T::from_usize(1 << self.constants.decimal_bits).expect("This should fit!");
        // R_hat_final (final layer only) for each iteration
        let r_hat_final_matrices = self
            .feed_forward_steps
            .iter()
            .map(|step| {
                step.intermediate_gradients.last().unwrap().mapv(|value| {
                    if value.is_positive() {
                        (value / trunc_factor).to_field()
                    } else {
                        ((value - trunc_factor + T::one()) / trunc_factor).to_field()
                    }
                })
            })
            .collect::<Vec<_>>();
        let (cm_r_hat_final, opening_r_hat_final) = client_schemes
            .poly_commit
            .poly_commit_matrices(&r_hat_final_matrices, G::ScalarField::ZERO, true, None, rng);

        // W
        let mut w_matrices = std::iter::once(
            // dummy first W matrix of all zeroes
            Dim([
                self.feed_forward_steps[0].post_activations_vector[0].nrows(),
                1,
            ]),
        )
        // initial weights shaped matrix of all zeroes plus dummy final W matrix (all matrices, but the last one, have a dummy first row)
        .chain(
            self.initial_weights[0..self.initial_weights.len() - 1]
                .iter()
                .map(|matrix| Dim([matrix.nrows() + 1, matrix.ncols()])),
        )
        .chain(std::iter::once(
            self.initial_weights.last().unwrap().raw_dim(),
        ))
        // add a dummy final W matrix of all zeroes
        .chain(std::iter::once(Dim([
            self.initial_weights.last().unwrap().nrows(),
            self.initial_weights.last().unwrap().nrows(),
        ])))
        .map(|dim| Array2::from_elem(dim, G::ScalarField::ZERO))
        // chain all weights from the iterations (we skip the final round as we don't need it)
        .chain(
            self.feed_forward_steps[..self.feed_forward_steps.len() - 1]
                .iter()
                .map(|step| step.weights_after.as_slice())
                .enumerate()
                .flat_map(|(idx, step)| {
                    // for each round add a dummy first W matrix of all zeroes
                    std::iter::once(Array2::from_elem(
                        Dim([
                            self.feed_forward_steps[idx].post_activations_vector[0].nrows(),
                            1,
                        ]),
                        G::ScalarField::ZERO,
                    ))
                    // chain the weight matrices of the layers
                    .chain(
                        step.iter()
                            .map(|matrix| matrix.mapv(|value| value.to_field()))
                            // add a dummy first row with all zeroes
                            .enumerate()
                            .map(|(idx, matrix)| {
                                if idx == step.len() - 1 {
                                    matrix
                                } else {
                                    concatenate![
                                        Axis(0),
                                        Array2::from_elem(
                                            (1, matrix.ncols()),
                                            G::ScalarField::ZERO
                                        ),
                                        matrix
                                    ]
                                }
                            }),
                    )
                    // for each round add a dummy final W matrix of all zeroes
                    .chain(std::iter::once(Array2::from_elem(
                        [
                            self.initial_weights.last().unwrap().nrows(),
                            self.initial_weights.last().unwrap().nrows(),
                        ],
                        G::ScalarField::ZERO,
                    )))
                }),
        )
        .collect::<Vec<_>>();

        // W final
        let w_final_matrices = std::iter::once(
            // add a dummy first W matrix of all zeros
            Array2::from_elem(
                Dim([
                    self.feed_forward_steps
                        .last()
                        .unwrap()
                        .post_activations_vector[0]
                        .nrows(),
                    1,
                ]),
                G::ScalarField::ZERO,
            ),
        )
        .chain(
            self.feed_forward_steps
                .last()
                .unwrap()
                .weights_after
                .iter()
                .map(|matrix| matrix.mapv(|value| value.to_field()))
                // add a dummy first row with all zeroes
                .enumerate()
                .map(|(idx, matrix)| {
                    if idx == self.feed_forward_steps.last().unwrap().weights_after.len() - 1 {
                        matrix
                    } else {
                        concatenate![
                            Axis(0),
                            Array2::from_elem((1, matrix.ncols()), G::ScalarField::ZERO),
                            matrix
                        ]
                    }
                }),
        )
        // for each round add a dummy final W matrix of all zeroes
        .chain(std::iter::once(Array2::from_elem(
            [
                self.initial_weights.last().unwrap().nrows(),
                self.initial_weights.last().unwrap().nrows(),
            ],
            G::ScalarField::ZERO,
        )))
        .collect::<Vec<_>>();

        // T hat
        let mut t_hat_matrices = self
            .feed_forward_steps
            .iter()
            .flat_map(|step| {
                // for each round add the first matrix to be of the same shape as the batch matrix (plus additional dummy row)
                std::iter::once(Array2::from_elem(
                    step.post_activations_vector[0].raw_dim(), // dummy row is included
                    G::ScalarField::ZERO,
                ))
                .chain(
                    step.pre_activations_vector
                        .iter()
                        .map(|matrix| {
                            matrix.mapv(|value| {
                                if value.is_positive() {
                                    (value / trunc_factor).to_field()
                                } else {
                                    ((value - trunc_factor + T::one()) / trunc_factor).to_field()
                                }
                            })
                        })
                        // add a dummy first row with all zeroes (to all but the final matrix)
                        .enumerate()
                        .map(|(idx, matrix)| {
                            if idx == step.pre_activations_vector.len() - 1 {
                                matrix
                            } else {
                                concatenate![
                                    Axis(0),
                                    Array2::from_elem((1, matrix.ncols()), G::ScalarField::ZERO),
                                    matrix
                                ]
                            }
                        })
                        // for each round add a finaly dummy matrix with the shape of the post activations of the final matrix
                        .chain(std::iter::once(Array2::from_elem(
                            step.post_activations_vector.last().unwrap().raw_dim(),
                            G::ScalarField::ZERO,
                        ))),
                )
            })
            .collect::<Vec<_>>();
        let (cm_t_hat, opening_t_hat) = client_schemes.poly_commit.poly_commit_matrices(
            &t_hat_matrices,
            G::ScalarField::ZERO,
            false,
            None,
            rng,
        );

        // T
        let mut t_matrices = self
            .feed_forward_steps
            .iter()
            .flat_map(|step| {
                // for each round add the first matrix to be of the same shape as the batch matrix (plus additional dummy row)
                std::iter::once(Array2::from_elem(
                    step.post_activations_vector[0].raw_dim(), // dummy row is included
                    G::ScalarField::ZERO,
                ))
                .chain(
                    step.pre_activations_vector
                        .iter()
                        .map(|matrix| matrix.mapv(|value| value.to_field()))
                        // add a dummy first row with all zeroes (to all but the final matrix)
                        .enumerate()
                        .map(|(idx, matrix)| {
                            if idx == step.pre_activations_vector.len() - 1 {
                                matrix
                            } else {
                                concatenate![
                                    Axis(0),
                                    Array2::from_elem((1, matrix.ncols()), G::ScalarField::ZERO),
                                    matrix
                                ]
                            }
                        })
                        // for each round add a final dummy matrix with the shape of the post activations of the final matrix
                        .chain(std::iter::once(Array2::from_elem(
                            step.post_activations_vector.last().unwrap().raw_dim(),
                            G::ScalarField::ZERO,
                        ))),
                )
            })
            .collect::<Vec<_>>();
        let (cm_t, opening_t) = client_schemes.poly_commit.poly_commit_matrices(
            &t_matrices,
            G::ScalarField::ZERO,
            false,
            None,
            rng,
        );

        // C
        let c_matrices = self
            .feed_forward_steps
            .iter()
            .flat_map(|step| {
                // for each round add a dummy first matrix of all zeroes (this one already has an additional dummy row of zeroes)
                std::iter::once(Array2::from_elem(
                    step.post_activations_vector[0].raw_dim(), // dummy row is included
                    G::ScalarField::ONE,
                ))
                .chain(
                    step.pre_activations_vector
                        .iter()
                        .map(|matrix| {
                            matrix.mapv(|value| {
                                if value.is_negative() {
                                    G::ScalarField::ZERO
                                } else {
                                    G::ScalarField::ONE
                                }
                            })
                        })
                        // add a dummy first row with all zeroes (to all but the final matrix)
                        .enumerate()
                        .map(|(idx, matrix)| {
                            if idx == step.pre_activations_vector.len() - 1 {
                                matrix
                            } else {
                                concatenate![
                                    Axis(0),
                                    Array2::from_elem((1, matrix.ncols()), G::ScalarField::ONE),
                                    matrix
                                ]
                            }
                        })
                        .chain(
                            // for each round add a final matrix with the same shape as that of the previous round
                            {
                                let last_matrix = step.post_activations_vector.last().unwrap();
                                std::iter::once(Array2::from_elem(
                                    (last_matrix.nrows(), last_matrix.ncols()),
                                    G::ScalarField::ONE,
                                ))
                            },
                        ),
                )
            })
            .collect::<Vec<_>>();
        let (cm_c, opening_c) = client_schemes.poly_commit.poly_commit_matrices(
            &c_matrices,
            G::ScalarField::ONE,
            false,
            None,
            rng,
        );

        // G bar
        let g_bar_matrices = self
            .feed_forward_steps
            .iter()
            .flat_map(|step| {
                // for each round add a dummy first matrix of all zeroes
                std::iter::once(Array2::from_elem(
                    Dim([step.post_activations_vector[0].nrows(), 1]),
                    G::ScalarField::ZERO,
                ))
                // chain the weight matrices of the layers
                .chain(
                    step.layer_gradients
                        .iter()
                        .map(|matrix| matrix.mapv(|value| value.to_field()))
                        // add a dummy first row with all zeroes (to all but the final matrix)
                        .enumerate()
                        .map(|(idx, matrix)| {
                            if idx == step.layer_gradients.len() - 1 {
                                matrix
                            } else {
                                concatenate![
                                    Axis(0),
                                    Array2::from_elem((1, matrix.ncols()), G::ScalarField::ZERO),
                                    matrix
                                ]
                            }
                        }),
                )
                // for each round add a dummy final matrix of all zeroes
                .chain(std::iter::once(Array2::from_elem(
                    [
                        self.initial_weights.last().unwrap().nrows(),
                        self.initial_weights.last().unwrap().nrows(),
                    ],
                    G::ScalarField::ZERO,
                )))
            })
            .collect::<Vec<_>>();
        let (cm_g_bar, opening_g_bar) = client_schemes.poly_commit.poly_commit_matrices(
            &g_bar_matrices,
            G::ScalarField::ZERO,
            true,
            Some(log2(
                self.constants
                    .cm_g_hat_single_iteration_dimensions
                    .row_com_dimension,
            ) as usize),
            rng,
        );

        // R hat
        let mut r_hat_matrices = self
            .feed_forward_steps
            .iter()
            .flat_map(|step|
                // add a first dummy matrix
                std::iter::once(Array2::from_elem(
                    step.post_activations_vector[0].raw_dim(), // dummy row is included
                    G::ScalarField::ZERO,
                ))
                    // for each round add the second matrix to be of the same shape as the batch matrix (plus additional dummy row)
                    .chain(
                        std::iter::once(Array2::from_elem(
                            step.post_activations_vector[0].raw_dim(), // dummy row is included
                            G::ScalarField::ZERO,
                        ))
                    )
                    .chain(step.intermediate_gradients[0..step.intermediate_gradients.len() - 1].iter().map(|matrix| matrix.mapv(|value| {
                        if value.is_positive() {
                            (value / trunc_factor).to_field()
                        } else {
                            ((value - trunc_factor + T::one()) / trunc_factor).to_field()
                        }
                    }))
                        // add a dummy first row with all zeroes
                        .map(|matrix| {
                            concatenate![
                                    Axis(0),
                                    Array2::from_elem((1, matrix.ncols()), G::ScalarField::ZERO),
                                    matrix
                                ]
                        })
                    )
                    // the final matrix is separate (so we add it as zeroes here)
                    .chain(std::iter::once(Array::from_elem(
                        step.intermediate_gradients.last().unwrap().raw_dim(),
                        G::ScalarField::ZERO,
                    ))))
            .collect::<Vec<_>>();
        let (cm_r_hat, opening_r_hat) = client_schemes.poly_commit.poly_commit_matrices(
            &r_hat_matrices,
            G::ScalarField::ZERO,
            false,
            None,
            rng,
        );

        // R
        let r_matrices = self
            .feed_forward_steps
            .iter()
            .flat_map(|step|
                // add a first dummy matrix
                std::iter::once(Array2::from_elem(
                    step.post_activations_vector[0].raw_dim(), // dummy row is included
                    G::ScalarField::ZERO,
                ))
                    // for each round add the second matrix to be of the same shape as the batch matrix (plus additional dummy row)
                    .chain(
                        std::iter::once(Array2::from_elem(
                            step.post_activations_vector[0].raw_dim(), // dummy row is included
                            G::ScalarField::ZERO,
                        ))
                    )
                    .chain(step.intermediate_gradients[0..step.intermediate_gradients.len() - 1].iter().map(|matrix| matrix.mapv(|value| value.to_field()))
                        // add a dummy first row with all zeroes
                        .map(|matrix| {
                            concatenate![
                                    Axis(0),
                                    Array2::from_elem((1, matrix.ncols()), G::ScalarField::ZERO),
                                    matrix
                                ]
                        })
                    )
                    // the final matrix is separate (so we add it as zeroes here)
                    .chain(std::iter::once(Array::from_elem(
                        step.intermediate_gradients.last().unwrap().raw_dim(),
                        G::ScalarField::ZERO,
                    )))
            )
            .collect::<Vec<_>>();
        let (cm_r, opening_r) = client_schemes.poly_commit.poly_commit_matrices(
            &r_matrices,
            G::ScalarField::ZERO,
            false,
            None,
            rng,
        );

        // G hat
        let learning_rate = self.constants.learning_rate;
        let double_trunc_factor =
            T::from_usize(1 << 2 * self.constants.decimal_bits).expect("This should fit!");
        let g_hat_matrices_all = self
            .feed_forward_steps
            .iter()
            .map(|step| {
                // for each round add a dummy first matrix of all zeroes
                std::iter::once(Array2::from_elem(
                    Dim([step.post_activations_vector[0].nrows(), 1]),
                    G::ScalarField::ZERO,
                ))
                // chain the weight matrices of the layers
                .chain(
                    step.layer_gradients
                        .iter()
                        .map(|matrix| {
                            matrix.mapv(|value| {
                                if value.is_positive() {
                                    ((value * learning_rate) / double_trunc_factor).to_field()
                                } else {
                                    (((value * learning_rate) - double_trunc_factor + T::one())
                                        / double_trunc_factor)
                                        .to_field()
                                }
                            })
                        })
                        // add a dummy first row with all zeroes (to all but the final matrix)
                        .enumerate()
                        .map(|(idx, matrix)| {
                            if idx == step.layer_gradients.len() - 1 {
                                matrix
                            } else {
                                concatenate![
                                    Axis(0),
                                    Array2::from_elem((1, matrix.ncols()), G::ScalarField::ZERO),
                                    matrix
                                ]
                            }
                        }),
                )
                // for each round add a dummy final matrix of all zeroes
                .chain(std::iter::once(Array2::from_elem(
                    [
                        self.initial_weights.last().unwrap().nrows(),
                        self.initial_weights.last().unwrap().nrows(),
                    ],
                    G::ScalarField::ZERO,
                )))
                .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut cm_g_hat_all = vec![];
        let mut opening_g_hat_all = vec![];
        g_hat_matrices_all.iter().for_each(|g_hat_matrices| {
            let (cm_g_hat, opening_g_hat) = client_schemes.poly_commit.poly_commit_matrices(
                &g_hat_matrices,
                G::ScalarField::ZERO,
                true,
                None,
                rng,
            );
            cm_g_hat_all.push(cm_g_hat);
            opening_g_hat_all.push(opening_g_hat);
        });

        // construct opening matrices with public values/values from other commitments
        // first row of all T-hat is all 1's (apart from the final two matrices of each iteration)
        for (idx, matrix) in t_hat_matrices.iter_mut().enumerate() {
            if idx % (self.constants.n_layers + 2) < self.constants.n_layers {
                matrix.slice_mut(s![0, ..]).fill(G::ScalarField::ONE);
            }
        }

        // T_hat, in each iteration, for l=0 gets the appropriate batch matrix (below the dummy row)
        // W, in each iteration, for l = L+1 is zero with all 1's diagonal
        // T, in each iteration, for l = L+1 is equal to the U matrix
        // R_hat, in each iteration, for l = L+1 is equal to the separated final R_hat matrix
        for (idx, (((t_hat_matrix, w_matrix), t_matrix), r_hat_matrix)) in t_hat_matrices
            .iter_mut()
            .zip(&mut w_matrices[self.constants.n_layers + 1..])
            .zip(&mut t_matrices[self.constants.n_layers + 1..])
            .zip(&mut r_hat_matrices[self.constants.n_layers + 1..])
            .step_by(self.constants.n_layers + 2)
            .enumerate()
        {
            t_hat_matrix
                .slice_mut(s![1.., ..])
                .assign(&batch_matrices_samples[idx % self.constants.n_batches].slice(s![1.., ..]));
            *w_matrix = Array2::from_diag_elem(w_matrix.nrows(), G::ScalarField::ONE);
            *t_matrix = self.feed_forward_steps[idx]
                .post_activations_vector
                .last()
                .unwrap()
                .mapv(|value| value.to_field());
            *r_hat_matrix = self.feed_forward_steps[idx]
                .intermediate_gradients
                .last()
                .unwrap()
                .mapv(|value| {
                    if value.is_positive() {
                        (value / trunc_factor).to_field()
                    } else {
                        ((value - trunc_factor + T::one()) / trunc_factor).to_field()
                    }
                });
        }

        // W, in the first iteration, for l = 1,...,L (i.e. not equal to 0 or L+1) should equal the initial weight matrices
        for (matrix, initial_weights) in w_matrices[1..=self.constants.n_layers]
            .iter_mut()
            .zip(self.initial_weights)
        {
            if matrix.raw_dim() == initial_weights.raw_dim() {
                // since the weight matrix at layer L doesn't have a dummy row
                matrix.assign(&initial_weights.mapv(|value| value.to_field()));
            } else {
                matrix
                    .slice_mut(s![1.., ..])
                    .assign(&initial_weights.mapv(|value| value.to_field()));
            }
        }

        for base in 0..self.constants.n_epochs {
            for idx in base * (self.constants.n_layers + 2)
                ..base * (self.constants.n_layers + 2) + (self.constants.n_layers + 1)
            {
                debug_assert_eq!(
                    t_matrices[idx + 1],
                    (&w_matrices[idx + 1]).dot(&(&c_matrices[idx] * &t_hat_matrices[idx])),
                    "idx: {}",
                    idx
                );
            }
        }

        let trunc_factor_as_scalar: G::ScalarField = trunc_factor.to_field();
        debug_assert!(
            opening_t
                .input_matrix
                .iter()
                .zip(&opening_t_hat.input_matrix)
                .all(
                    |(t_row, t_hat_row)| t_row.iter().zip(t_hat_row).all(|(t_val, t_hat_val)| {
                        let remainder = *t_val - *t_hat_val * trunc_factor_as_scalar;
                        remainder >= G::ScalarField::ZERO && remainder < trunc_factor_as_scalar
                    })
                )
        );
        for row in &opening_t.input_matrix {
            for &elem in row {
                let test =
                    elem + G::ScalarField::from(2).pow([(self.constants.total_bits - 1) as u64]);
                debug_assert!(
                    test <= G::ScalarField::from(2).pow([(self.constants.total_bits) as u64])
                );
            }
        }

        for base in 0..self.constants.n_epochs {
            for idx in (base * (self.constants.n_layers + 2)
                ..base * (self.constants.n_layers + 2) + self.constants.n_layers)
                .rev()
            {
                debug_assert_eq!(
                    g_bar_matrices[idx + 1],
                    (&r_hat_matrices[idx + 2] * &c_matrices[idx + 1])
                        .dot(&(&c_matrices[idx] * &t_hat_matrices[idx]).t()),
                    "idx: {}",
                    idx
                );

                // skip the first layer, since no intermediate gradient there
                if idx % (self.constants.n_layers + 2) != 0 {
                    debug_assert_eq!(
                        r_matrices[idx + 1],
                        concatenate![
                            Axis(1),
                            Array2::from_elem(
                                (w_matrices[idx + 1].nrows(), 1),
                                G::ScalarField::ZERO
                            ),
                            w_matrices[idx + 1].slice(s![.., 1..])
                        ]
                        .t()
                        .dot(&(&r_hat_matrices[idx + 2] * &c_matrices[idx + 1])),
                        "idx: {}",
                        idx
                    );
                }
            }
        }

        for row in &opening_r.input_matrix {
            for &elem in row {
                let test =
                    elem + G::ScalarField::from(2).pow([(self.constants.total_bits - 1) as u64]);
                debug_assert!(
                    test <= G::ScalarField::from(2).pow([(self.constants.total_bits) as u64])
                );
            }
        }

        debug_assert!(
            opening_r
                .input_matrix
                .iter()
                .zip(&opening_r_hat.input_matrix)
                .all(
                    |(r_row, r_hat_row)| r_row.iter().zip(r_hat_row).all(|(r_val, r_hat_val)| {
                        let remainder = *r_val - *r_hat_val * trunc_factor_as_scalar;
                        remainder >= G::ScalarField::ZERO && remainder < trunc_factor_as_scalar
                    })
                )
        );

        for row in &opening_g_bar.input_matrix {
            for &elem in row {
                let test = elem
                    + G::ScalarField::from(2).pow([(self.constants.total_bits * 2 - 1) as u64]);
                debug_assert!(
                    test <= G::ScalarField::from(2).pow([(self.constants.total_bits * 2) as u64])
                );
            }
        }

        // create struct containing all relevant commitment information
        LocalTrainingCommitments {
            cm_batch_samples,
            cm_batch_labels,
            cm_u,
            cm_c,
            cm_t_hat,
            cm_t,
            cm_g_bar,
            cm_r_hat,
            cm_r_hat_final,
            cm_r,
            cm_g_hat_all,
            opening_batch_samples: Some(opening_batch_samples),
            opening_batch_labels: Some(opening_batch_labels),
            opening_u: Some(opening_u),
            w_with_public_input_matrix: Some(client_schemes.poly_commit.combine_matrices(
                &w_matrices,
                G::ScalarField::ZERO,
                false,
                None,
            )),
            w_final_input_matrix: Some(client_schemes.poly_commit.combine_matrices(
                &w_final_matrices,
                G::ScalarField::ZERO,
                true,
                None,
            )),
            opening_c_with_public: Some(HyraxCommitmentOpening {
                input_matrix: client_schemes.poly_commit.combine_matrices(
                    &c_matrices,
                    G::ScalarField::ZERO,
                    false,
                    None,
                ),
                randomness: opening_c.randomness.clone(),
            }),
            opening_c: Some(opening_c),
            opening_t_hat_with_public: Some(HyraxCommitmentOpening {
                input_matrix: client_schemes.poly_commit.combine_matrices(
                    &t_hat_matrices,
                    G::ScalarField::ONE,
                    false,
                    None,
                ),
                randomness: opening_t_hat.randomness.clone(),
            }),
            opening_t_hat: Some(opening_t_hat),
            opening_t_with_public: Some(HyraxCommitmentOpening {
                input_matrix: client_schemes.poly_commit.combine_matrices(
                    &t_matrices,
                    G::ScalarField::ZERO,
                    false,
                    None,
                ),
                randomness: opening_t.randomness.clone(),
            }),
            opening_t: Some(opening_t),
            opening_g_bar: Some(opening_g_bar),
            opening_r_hat_with_public: Some(HyraxCommitmentOpening {
                input_matrix: client_schemes.poly_commit.combine_matrices(
                    &r_hat_matrices,
                    G::ScalarField::ZERO,
                    false,
                    None,
                ),
                randomness: opening_r_hat.randomness.clone(),
            }),
            opening_r_hat: Some(opening_r_hat),
            opening_r_hat_final: Some(opening_r_hat_final),
            opening_r: Some(opening_r),
            opening_g_hat_all: Some(opening_g_hat_all),
        }
    }

    pub fn add_commitments(
        &self,
        prover_state: &mut ProverState,
        commitments: &LocalTrainingCommitments<G>,
    ) -> ProofResult<()> {
        // add commitments to the transcript
        prover_state.add_points(&[
            self.cm_dataset_samples.into_group(),
            self.cm_dataset_labels.into_group(),
        ])?;
        prover_state.add_points(&commitments.cm_batch_samples.row_coms_as_group())?;
        prover_state.add_points(&commitments.cm_batch_labels.row_coms_as_group())?;
        prover_state.add_points(&commitments.cm_u.row_coms_as_group())?;
        prover_state.add_points(&commitments.cm_c.row_coms_as_group())?;
        prover_state.add_points(&commitments.cm_t_hat.row_coms_as_group())?;
        prover_state.add_points(&commitments.cm_t.row_coms_as_group())?;
        prover_state.add_points(&commitments.cm_g_bar.row_coms_as_group())?;
        prover_state.add_points(&commitments.cm_r_hat.row_coms_as_group())?;
        prover_state.add_points(&commitments.cm_r_hat_final.row_coms_as_group())?;
        prover_state.add_points(&commitments.cm_r.row_coms_as_group())?;
        prover_state.add_points(
            &ark_std::cfg_iter!(commitments.cm_g_hat_all)
                .flat_map(|x| x.row_coms_as_group())
                .collect::<Vec<_>>(),
        )?;
        Ok(())
    }
}

impl<G: CurveGroup, T: Value> LocalTrainingProof<G, T> {
    pub fn obtain_commitments(
        &self,
        verifier_state: &mut VerifierState,
    ) -> ProofResult<LocalTrainingCommitments<G>> {
        let mut cm_batch_samples =
            vec![G::ZERO; self.constants.cm_batch_samples_dimensions.num_row_coms];
        verifier_state.fill_next_points(&mut cm_batch_samples)?;
        let mut cm_batch_labels =
            vec![G::ZERO; self.constants.cm_batch_labels_dimensions.num_row_coms];
        verifier_state.fill_next_points(&mut cm_batch_labels)?;
        let mut cm_u = vec![G::ZERO; self.constants.cm_u_dimensions.num_row_coms];
        verifier_state.fill_next_points(&mut cm_u)?;
        let mut cm_c = vec![G::ZERO; self.constants.cm_c_dimensions.num_row_coms];
        verifier_state.fill_next_points(&mut cm_c)?;
        let mut cm_t_hat = vec![G::ZERO; self.constants.cm_t_dimensions.num_row_coms];
        verifier_state.fill_next_points(&mut cm_t_hat)?;
        let mut cm_t = vec![G::ZERO; self.constants.cm_t_dimensions.num_row_coms];
        verifier_state.fill_next_points(&mut cm_t)?;
        let mut cm_g_bar = vec![G::ZERO; self.constants.cm_g_bar_dimensions.num_row_coms];
        verifier_state.fill_next_points(&mut cm_g_bar)?;
        let mut cm_r_hat = vec![G::ZERO; self.constants.cm_r_dimensions.num_row_coms];
        verifier_state.fill_next_points(&mut cm_r_hat)?;
        let mut cm_r_hat_final = vec![G::ZERO; self.constants.cm_r_final_dimensions.num_row_coms];
        verifier_state.fill_next_points(&mut cm_r_hat_final)?;
        let mut cm_r = vec![G::ZERO; self.constants.cm_r_dimensions.num_row_coms];
        verifier_state.fill_next_points(&mut cm_r)?;
        let mut cm_g_hat_all = vec![
            G::ZERO;
            self.constants.n_epochs
                * self
                    .constants
                    .cm_g_hat_single_iteration_dimensions
                    .num_row_coms
        ];
        verifier_state.fill_next_points(&mut cm_g_hat_all)?;

        // store the result
        Ok(LocalTrainingCommitments {
            cm_batch_samples: HyraxCommitment {
                row_coms: G::normalize_batch(&cm_batch_samples),
            },
            cm_batch_labels: HyraxCommitment {
                row_coms: G::normalize_batch(&cm_batch_labels),
            },
            cm_u: HyraxCommitment {
                row_coms: G::normalize_batch(&cm_u),
            },
            cm_c: HyraxCommitment {
                row_coms: G::normalize_batch(&cm_c),
            },
            cm_t_hat: HyraxCommitment {
                row_coms: G::normalize_batch(&cm_t_hat),
            },
            cm_t: HyraxCommitment {
                row_coms: G::normalize_batch(&cm_t),
            },
            cm_g_bar: HyraxCommitment {
                row_coms: G::normalize_batch(&cm_g_bar),
            },
            cm_r_hat: HyraxCommitment {
                row_coms: G::normalize_batch(&cm_r_hat),
            },
            cm_r_hat_final: HyraxCommitment {
                row_coms: G::normalize_batch(&cm_r_hat_final),
            },
            cm_r: HyraxCommitment {
                row_coms: G::normalize_batch(&cm_r),
            },
            cm_g_hat_all: G::normalize_batch(&cm_g_hat_all)
                .chunks_exact(
                    self.constants
                        .cm_g_hat_single_iteration_dimensions
                        .num_row_coms,
                )
                .map(|chunk| HyraxCommitment {
                    row_coms: chunk.to_vec(),
                })
                .collect::<Vec<_>>(),
            opening_batch_samples: None,
            opening_batch_labels: None,
            opening_u: None,
            w_with_public_input_matrix: None,
            w_final_input_matrix: None,
            opening_c: None,
            opening_c_with_public: None,
            opening_t_hat: None,
            opening_t_hat_with_public: None,
            opening_t: None,
            opening_t_with_public: None,
            opening_g_bar: None,
            opening_r_hat: None,
            opening_r_hat_with_public: None,
            opening_r_hat_final: None,
            opening_r: None,
            opening_g_hat_all: None,
        })
    }
}

pub fn create_w_initial_matrices<T: Value, G: CurveGroup>(
    initial_weights: &[&Array2<T>],
) -> Vec<Array2<G::ScalarField>> {
    std::iter::once(
        // add a dummy first W matrix of all zeros
        Array2::from_elem(Dim([initial_weights[0].ncols(), 1]), G::ScalarField::ZERO),
    )
    .chain(
        initial_weights
            .iter()
            .map(|matrix| matrix.mapv(|value| value.to_field()))
            // add a dummy first row with all zeroes (all matrices, but the last)
            .enumerate()
            .map(|(idx, matrix)| {
                if idx == initial_weights.len() - 1 {
                    matrix
                } else {
                    concatenate![
                        Axis(0),
                        Array2::from_elem((1, matrix.ncols()), G::ScalarField::ZERO),
                        matrix
                    ]
                }
            }),
    )
    // for each round add a dummy final W matrix of all zeroes
    .chain(std::iter::once(Array2::from_elem(
        [
            initial_weights.last().unwrap().nrows(),
            initial_weights.last().unwrap().nrows(),
        ],
        G::ScalarField::ZERO,
    )))
    .collect::<Vec<_>>()
}
