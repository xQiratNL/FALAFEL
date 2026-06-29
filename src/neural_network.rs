//! All functionality for feed-forward neural networks on any numeric value.

use crate::dataset::Dataset;
use crate::prelude::*;
use crate::proof::ProofWitnessStep;
use crate::util::RandomMatrix;
use ndarray::{Array2, ArrayView2, Axis, concatenate, s};
use rand::Rng;
use rand::prelude::Distribution;
use serde::{Deserialize, Serialize};
use std::iter::Sum;
use std::ops::{Add, Mul};

#[derive(Copy, Clone, Serialize, Deserialize, Debug)]
pub enum ActivationFunction {
    ReLU,
    Identity,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(bound(deserialize = "T: Value"))]
pub struct DenseLayer<T: Value> {
    pub weights: Array2<T>,
    pub activation_function: ActivationFunction,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
#[serde(bound(deserialize = "T: Value"))]
pub struct FeedForward<T: Value> {
    pub layers: Vec<DenseLayer<T>>,
    pub learning_rate: T,
}

impl<T: Value> DenseLayer<T> {
    fn new_from_weights(weights: Array2<T>, activation_function: ActivationFunction) -> Self {
        Self {
            weights,
            activation_function,
        }
    }

    pub fn set_weights(&mut self, weights: Array2<T>) -> FalafelResult<()> {
        if weights.shape() != self.weights.shape() {
            return Err(FalafelError::InputError("Cannot set weights in this layer, since the new matrix has a different shape than the current.".to_string()));
        }
        self.weights = weights;
        Ok(())
    }
}

impl<T: Value + RandomMatrix> DenseLayer<T> {
    fn new_with_random_weights(
        n_in: usize,
        n_out: usize,
        activation_function: ActivationFunction,
        distribution: &impl Distribution<f64>,
        decimal_bits: usize,
        rng: &mut impl Rng,
    ) -> Self {
        Self {
            weights: T::random_matrix(n_out, n_in + 1, distribution, decimal_bits, rng), // the "+ 1" is for the bias
            activation_function,
        }
    }
}

impl<T: Value> Add for DenseLayer<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let sum = self.weights + other.weights;
        Self {
            weights: sum,
            activation_function: self.activation_function,
        }
    }
}

impl<T: Value> Mul<T> for DenseLayer<T> {
    type Output = Self;

    fn mul(self, other: T) -> Self {
        let result = self.weights.mapv(|v| v * other);
        Self {
            weights: result,
            activation_function: self.activation_function,
        }
    }
}

impl<T: Value> FeedForward<T> {
    pub fn new_from_weights(
        weights: Vec<Array2<T>>,
        activation_functions: Vec<ActivationFunction>,
        learning_rate: T,
    ) -> FalafelResult<Self> {
        if weights.len() != activation_functions.len() {
            return Err(FalafelError::InputError("Cannot create FeedForward network, different number of weight matrices and activation_functions provided.".to_string()));
        }
        Ok(Self {
            layers: weights
                .into_iter()
                .zip(activation_functions.iter())
                .map(|(w, &f)| DenseLayer::new_from_weights(w, f))
                .collect(),
            learning_rate,
        })
    }

    pub fn set_weights(&mut self, weights_vector: Vec<Array2<T>>) -> FalafelResult<()> {
        if weights_vector.len() != self.layers.len() {
            return Err(FalafelError::InputError("Cannot set weights in this feed-forward network, the number of new weight matrices is less than the number of layers.".to_string()));
        }
        self.layers
            .iter_mut()
            .zip(weights_vector.into_iter())
            .try_for_each(|(layer, weights)| layer.set_weights(weights))
    }

    /// Note: This assumes that the initial weights (or weights obtained from a central
    /// aggregation), have been set already. This method only offers the local training for
    /// `n_epochs` epochs given the dataset that is split in k batches of size `batch_size`.
    /// I.e., the total number of training steps run is k times `n_epochs`.
    pub fn local_training_full(
        &mut self,
        dataset: &Dataset<T>,
        n_epochs: usize,
        batch_size: usize,
        decimal_bits: usize,
    ) -> FalafelResult<Vec<ProofWitnessStep<T>>> {
        let mut proof_data = vec![];
        for _epoch_idx in 0..n_epochs {
            let (batch_inputs, batch_labels) = dataset.batches(batch_size);
            for (_batch_idx, (batch_inputs, batch_labels)) in batch_inputs
                .into_iter()
                .zip(batch_labels.into_iter())
                .enumerate()
            {
                proof_data.push(self.local_training_single_step(
                    batch_inputs,
                    batch_labels,
                    decimal_bits,
                )?);
            }
        }
        Ok(proof_data)
    }

    pub fn local_training_single_step(
        &mut self,
        batch_inputs: ArrayView2<T>,
        batch_labels: ArrayView2<T>,
        decimal_bits: usize,
    ) -> FalafelResult<ProofWitnessStep<T>> {
        // actual training loop
        let (post_activations_vector, pre_activations_vector) =
            self.forward_pass(batch_inputs, decimal_bits);
        let (layer_gradients, intermediate_gradients) = self.backward_pass(
            &post_activations_vector,
            &pre_activations_vector,
            batch_labels,
            decimal_bits,
        )?;
        // this updates the weights
        self.weight_update(&layer_gradients, decimal_bits);

        // store the weights from after the update step
        let weights_after = self
            .layers
            .iter()
            .map(|layer| layer.weights.clone())
            .collect::<Vec<_>>();

        // note: `batch_inputs` is the first entry in the post_activations_vector
        Ok(ProofWitnessStep {
            weights_after,
            post_activations_vector,
            pre_activations_vector,
            layer_gradients,
            intermediate_gradients,
        })
    }

    fn forward_pass(
        &self,
        batch_inputs: ArrayView2<T>,
        decimal_bits: usize,
    ) -> (Vec<Array2<T>>, Vec<Array2<T>>) {
        let trunc_factor = T::from_usize(1 << decimal_bits).expect("This should fit!");

        let mut post_activations_vector = Vec::with_capacity(self.layers.len() + 1);
        let mut pre_activations_vector = Vec::with_capacity(self.layers.len());

        // Define the batch inputs as the first post activations to simplify the iteration.
        // Add row of ones to include bias in the weight matrix.
        post_activations_vector.push(concatenate![
            Axis(0),
            Array2::ones((1, batch_inputs.ncols())),
            batch_inputs
        ]);

        // Iterate the forward pass through the layers.
        for (n_layer, layer) in self.layers.iter().enumerate() {
            pre_activations_vector.push(layer.weights.dot(&post_activations_vector[n_layer]));
            let post_activations = match layer.activation_function {
                ActivationFunction::ReLU => pre_activations_vector[n_layer].mapv(|elem| {
                    if elem.is_positive() {
                        elem / trunc_factor
                    } else {
                        T::zero()
                    }
                }),
                ActivationFunction::Identity => pre_activations_vector[n_layer].clone(),
            };
            if n_layer != self.layers.len() - 1 {
                // Again, we prepend a row of ones to match our convention of including the
                // bias as the first column in the weight matrix.
                post_activations_vector.push(concatenate![
                    Axis(0),
                    Array2::ones((1, pre_activations_vector[n_layer].ncols())),
                    post_activations
                ]);
            } else {
                // In the final iteration, we don’t prepend a row of ones since there’s no subsequent layer.
                // So no additional bias term is needed.
                post_activations_vector.push(post_activations);
            }
        }

        // Return relevant vectors.
        (post_activations_vector, pre_activations_vector)
    }

    fn backward_pass(
        &self,
        post_activations_vector: &[Array2<T>],
        pre_activations_vector: &[Array2<T>],
        batch_labels: ArrayView2<T>,
        decimal_bits: usize,
    ) -> FalafelResult<(Vec<Array2<T>>, Vec<Array2<T>>)> {
        let trunc_factor = T::from_usize(1 << decimal_bits).expect("This should fit!");

        let mut layer_gradients = vec![Array2::default((0, 0)); self.layers.len()];
        let mut intermediate_gradients = vec![Array2::default((0, 0)); self.layers.len()];

        // compute the first intermediate gradient
        let scaling_factor = T::from_f64(
            2. * (1 << decimal_bits) as f64 / batch_labels.ncols() as f64,
        )
        .ok_or_else(|| {
            FalafelError::InputError(
                "Could not convert 2/batch_size to the used number format!".to_string(),
            )
        })?;

        // we start with the derivative of the loss function
        intermediate_gradients[self.layers.len() - 1] =
            &post_activations_vector[self.layers.len()] - &batch_labels;
        intermediate_gradients[self.layers.len() - 1].mapv_inplace(|elem| elem * scaling_factor);

        // iterate backwards through the layers
        for (n_layer, layer) in self.layers.iter().enumerate().rev() {
            let activation_function_derivatives = match layer.activation_function {
                ActivationFunction::ReLU => pre_activations_vector[n_layer].mapv(|elem| {
                    if elem.is_negative() {
                        T::zero()
                    } else {
                        T::one()
                    }
                }),
                ActivationFunction::Identity => {
                    // Derivative of identity function is always 1
                    Array2::from_elem(pre_activations_vector[n_layer].dim(), T::one())
                }
            };

            let intermediate_gradient_this_layer_truncated =
                intermediate_gradients[n_layer].mapv(|value| {
                    if value.is_positive() {
                        value / trunc_factor
                    } else {
                        (value - trunc_factor + T::one()) / trunc_factor
                    }
                });

            layer_gradients[n_layer] = (&intermediate_gradient_this_layer_truncated
                * &activation_function_derivatives)
                .dot(&post_activations_vector[n_layer].t());
            // we don't need to compute the intermediate gradient after the first layer
            if n_layer != 0 {
                // we slice off the bias, since it is not used for the intermediate gradient
                intermediate_gradients[n_layer - 1] = layer.weights.slice(s![.., 1..]).t().dot(
                    &(&intermediate_gradient_this_layer_truncated
                        * &activation_function_derivatives),
                )
            }
        }
        Ok((layer_gradients, intermediate_gradients))
    }

    fn weight_update(&mut self, layer_gradients: &[Array2<T>], decimal_bits: usize) {
        let double_trunc_factor = T::from_usize(1 << 2 * decimal_bits).expect("This should fit!");
        for (layer, gradient) in self.layers.iter_mut().zip(layer_gradients) {
            layer.weights = &layer.weights
                - gradient.mapv(|elem| {
                    if elem.is_positive() {
                        (elem * self.learning_rate) / double_trunc_factor
                    } else {
                        ((elem * self.learning_rate) - double_trunc_factor + T::one())
                            / double_trunc_factor
                    }
                })
        }
    }
}

impl<T: Value + RandomMatrix> FeedForward<T> {
    pub fn new_with_random_weights(
        n_input: usize,
        n_output: usize,
        layer_sizes: Vec<usize>,
        activation_functions: Vec<ActivationFunction>,
        learning_rate: T,
        distribution: &impl Distribution<f64>,
        decimal_bits: usize,
        rng: &mut impl Rng,
    ) -> FalafelResult<Self> {
        if layer_sizes.len() + 1 != activation_functions.len() {
            return Err(FalafelError::InputError("Cannot create FeedForward network, different number of layers and activation_functions provided.".to_string()));
        }
        Ok(Self {
            layers: std::iter::once(n_input)
                .chain(layer_sizes.iter().copied())
                .zip(layer_sizes.iter().copied().chain(std::iter::once(n_output)))
                .zip(activation_functions)
                .map(|((n_in, n_out), f)| {
                    DenseLayer::new_with_random_weights(
                        n_in,
                        n_out,
                        f,
                        distribution,
                        decimal_bits,
                        rng,
                    )
                })
                .collect(),
            learning_rate,
        })
    }
}

impl<T: Value> Add for FeedForward<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let added_layers = self
            .layers
            .into_iter()
            .zip(other.layers.into_iter())
            .map(|(x, y)| x + y)
            .collect();
        Self {
            layers: added_layers,
            learning_rate: self.learning_rate,
        }
    }
}

impl<T: Value> Sum for FeedForward<T> {
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        iter.next()
            .map(|first| iter.fold(first, |a, b| a + b))
            .expect("Cannot sum an empty list of networks")
    }
}

#[cfg(test)]
mod test {
    use crate::dataset::Dataset;
    use crate::neural_network::ActivationFunction::ReLU;
    use crate::neural_network::FeedForward;
    use crate::prelude::Value;
    use crate::tests::fixtures::{
        complex_network_relu_relu, simple_network_identity_identity, simple_network_identity_relu,
        simple_network_relu_identity, simple_network_relu_relu,
    };
    use crate::util::RandomMatrix;
    use approx::assert_abs_diff_eq;
    use fixed::types::{I32F0, I32F32};
    use ndarray::{Array2, array};
    use rand_chacha::ChaChaRng;
    use rand_chacha::rand_core::SeedableRng;
    use rand_distr::StandardNormal;
    use rstest::rstest;

    fn scalar_mul_fixed_point_array(arr: &Array2<I32F32>, scalar: I32F32) -> Array2<I32F32> {
        arr.mapv(|x| x * scalar)
    }

    #[test]
    fn local_training_with_fxp() {
        // This test ensures that we can also train using the fixed crate.
        local_training_random_data::<I32F32>();
        local_training_random_data::<I32F0>();
    }

    #[test]
    fn local_training_with_fp() {
        // This test ensures that we can train using floating points.
        local_training_random_data::<f64>();
        local_training_random_data::<f32>();
    }

    #[test]
    fn local_training_with_int() {
        // This test ensures that we can train using ints.
        local_training_random_data::<i128>();
        local_training_random_data::<i64>();
        local_training_random_data::<i32>();
    }

    fn local_training_random_data<T: Value + RandomMatrix>() {
        let mut rng = ChaChaRng::from_entropy();
        let mut nn = FeedForward::<T>::new_with_random_weights(
            5,
            10,
            vec![3, 2, 20, 30, 4],
            vec![ReLU; 6],
            T::from_f64(0.01).unwrap(),
            &StandardNormal,
            0,
            &mut rng,
        )
        .expect("Could not instantiate neural network!");
        let dataset = Dataset::<T>::new_random(100, 5, 10, &StandardNormal, 0, &mut rng);
        nn.local_training_full(&dataset, 3, 9, 0).unwrap();
    }

    // Test forward pass with floating point values.
    #[rstest]
    #[case::f64_case_1(simple_network_relu_relu::<f64>(), array![[3.0]], array![[0.0]])]
    #[case::f64_case_2(simple_network_relu_relu::<f64>(), array![[0.0]], array![[0.75]])]
    #[case::f64_case_3(simple_network_relu_relu::<f64>(), array![[-1.0]], array![[0.75]])]
    #[case::f64_case_4(complex_network_relu_relu::<f64>(), array![[1.0], [2.0]], array![[4.5], [16.0]]
    )]
    #[case::f64_case_5(complex_network_relu_relu::<f64>(), array![[0.0], [0.0]], array![[0.0], [0.0]]
    )]
    #[case::f64_case_6(complex_network_relu_relu::<f64>(), array![[-1.0], [1.0]], array![[0.0], [0.0]]
    )]
    #[case::f64_case_7(complex_network_relu_relu::<f64>(), array![[10.0], [-1.0]], array![[37.5], [70.0]]
    )]
    fn feedforward_output_matches_expected_f64(
        #[case] network: FeedForward<f64>,
        #[case] input: Array2<f64>,
        #[case] expected_output: Array2<f64>,
    ) {
        let (post_activations, _) = network.forward_pass(input.view(), 0);
        let output = post_activations.last().unwrap();
        assert_abs_diff_eq!(*output, &expected_output, epsilon = 1e-6);
    }

    // Test forward pass with fixed point values.
    #[rstest]
    #[case::i32f32_case_1(
        simple_network_relu_relu::<I32F32>(),
        array![[I32F32::from_num(3.0)]],
        array![[I32F32::from_num(0.0)]]
    )]
    #[case::i32f32_case_2(
        simple_network_relu_relu::<I32F32>(),
        array![[I32F32::from_num(0.0)]],
        array![[I32F32::from_num(0.75)]]
    )]
    #[case::i32f32_case_3(
        simple_network_relu_relu::<I32F32>(),
        array![[I32F32::from_num(-1.0)]],
        array![[I32F32::from_num(0.75)]]
    )]
    #[case::i32f32_case_4(
        complex_network_relu_relu::<I32F32>(),
        array![[I32F32::from_num(1.0)], [I32F32::from_num(2.0)]],
        array![[I32F32::from_num(4.5)], [I32F32::from_num(16.0)]]
    )]
    #[case::i32f32_case_5(
        complex_network_relu_relu::<I32F32>(),
        array![[I32F32::from_num(0.0)], [I32F32::from_num(0.0)]],
        array![[I32F32::from_num(0.0)], [I32F32::from_num(0.0)]]
    )]
    #[case::i32f32_case_6(
        complex_network_relu_relu::<I32F32>(),
        array![[I32F32::from_num(-1.0)], [I32F32::from_num(1.0)]],
        array![[I32F32::from_num(0.0)], [I32F32::from_num(0.0)]]
    )]
    #[case::i32f32_case_7(
        complex_network_relu_relu::<I32F32>(),
        array![[I32F32::from_num(10.0)], [I32F32::from_num(-1.0)]],
        array![[I32F32::from_num(37.5)], [I32F32::from_num(70.0)]]
    )]
    fn feedforward_output_matches_expected(
        #[case] network: FeedForward<I32F32>,
        #[case] input: Array2<I32F32>,
        #[case] expected_output: Array2<I32F32>,
    ) {
        let (post_activations, _) = network.forward_pass(input.view(), 0);
        let output = post_activations.last().unwrap();

        // Compare element-wise with a small epsilon
        let epsilon = I32F32::from_num(1e-6);
        for (a, b) in output.iter().zip(expected_output.iter()) {
            assert!((*a - *b).abs() <= epsilon, "Expected {:?}, got {:?}", b, a);
        }
    }

    // Test backward pass with floating point values.
    #[rstest]
    #[case::f64_case_1(
        simple_network_relu_relu::<f64>(),
        array![[3.0]],
        array![[-6.0]],
        // All gradients are zero so no weight update is done.
        vec![
            array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], // shape (3, 2)
            array![[0.0, 0.0, 0.0, 0.0]]               // shape (1, 4)
        ],
        vec![
            array![[0.5, 1.0], [-0.5, 2.0], [0.1, -1.0]], // shape (3, 2)
            array![[0.2, 1.0, -1.0, 0.5]]                // shape (1, 4)
        ]
    )]
    #[case::f64_case_2(
        simple_network_relu_relu::<f64>(),
        array![[-10.0]],
        array![[-6.0]],
        vec![
            array![[0.0, 0.0], [0.0, 0.0], [11.25, -112.5]], // shape (3, 2)
            array![[22.5, 0.0, 0.0, 227.25]]                // shape (1, 4)
        ],
        vec![
            array![
                [0.5 - 0.01 * 0.0, 1.0 - 0.01 * 0.0],
                [-0.5 - 0.01 * 0.0, 2.0 - 0.01 * 0.0],
                [0.1 - 0.01 * 11.25, -1.0 - 0.01 * -112.5]
            ], // shape (3, 2)
            array![[
                0.2 - 0.01 * 22.5,
                1.0,
                -1.0,
                0.5 - 0.01 * 227.25
            ]] // shape (1, 4)
        ]
    )]
    #[case::f64_case_3(
        simple_network_relu_relu::<f64>(),
        array![[1.0]],
        array![[-2.0]],
        vec![
            array![[4.4, 4.4], [-4.4, -4.4], [0.0, 0.0]], // shape (3, 2)
            array![[4.4, 6.6, 6.6, 0.0]]                 // shape (1, 4)
        ],
        vec![
            array![
                [0.5 - 0.01 * 4.4, 1.0 - 0.01 * 4.4],
                [-0.5 - 0.01 * -4.4, 2.0 - 0.01 * -4.4],
                [0.1, -1.0]
            ], // shape (3, 2)
            array![
                [0.2 - 0.01 * 4.4, 1.0 - 0.01 * 6.6, -1.0 - 0.01 * 6.6, 0.5 - 0.01 * 0.0]
            ] // shape (1, 4)
        ]
    )]
    #[case::f64_case_4(
        simple_network_relu_relu::<f64>(),
        array![[-2.0]],
        array![[6.0]],
        vec![
            array![[0.0, 0.0], [0.0, 0.0], [-4.75, 9.5]], // shape (3, 2)
            array![[-9.5, 0.0, 0.0, -19.95]]             // shape (1, 4)
        ],
        vec![
            array![
                [0.5, 1.0],
                [-0.5, 2.0],
                [0.1 - 0.01 * -4.75, -1.0 - 0.01 * 9.5]
            ], // shape (3, 2)
            array![[
                0.2 - 0.01 * -9.5,
                1.0,
                -1.0,
                0.5 - 0.01 * -19.95
            ]] // shape (1, 4)
        ]
    )]
    #[case::f64_case_5(
        complex_network_relu_relu::<f64>(),
        array![[0.0],[0.0]],
        array![[6.0],[1.0]],
        // Gradients don't vanish (as we keep the ReLU gradient at 1 at x= 0).
        vec![
            array![
                [24.0, 0.0, 0.0],
                [0.0, 0.0, 0.0]
            ], // shape (2, 3)
            array![
                [-12.0, -6.0, 0.0],
                [0.0, 0.0, 0.0]
            ] // shape (2, 3)
        ],
        vec![
            array![
                [0.26, 1.0, 1.0],
                [-0.5, 2.0, 1.0]
            ], // shape (2, 3)
            array![
                [1.12, -1.94, 3.0],
                [-5.0, 4.0, 2.0]
            ] // shape (2, 3)
        ]
    )]
    #[case::f64_case_6(
        complex_network_relu_relu::<f64>(),
        array![[1.0],[-1.0]],
        array![[3.0],[-7.0]],
        vec![
            array![
                [6.0, 6.0, -6.0],
                [-9.0, -9.0, 9.0]
            ], // shape (2, 3)
            array![
                [-3.0, -1.5, -1.5],
                [0.0, 0.0, 0.0]
            ], // shape (2, 3)
        ],
        vec![
            array![
                [0.5, 1.0, 1.0],
                [-0.5, 2.0, 1.0]
            ] - 0.01 * array![
                [6.0, 6.0, -6.0],
                [-9.0, -9.0, 9.0]
            ],
            array![
                [1.0, -2.0, 3.0],
                [-5.0, 4.0, 2.0]
            ] - 0.01 * array![
                [-3.0, -1.5, -1.5],
                [0.0, 0.0, 0.0]
            ]
        ]
    )]
    fn backward_pass_output_matches_expected_f64(
        #[case] mut network: FeedForward<f64>,
        #[case] input_to_nn: Array2<f64>,
        #[case] wanted_output_of_nn: Array2<f64>,
        #[case] expected_gradient: Vec<Array2<f64>>,
        #[case] expected_new_weights: Vec<Array2<f64>>, // Hidden + Output Weights
    ) {
        let (post_activations, pre_activations) = network.forward_pass(input_to_nn.view(), 0);
        let (layer_gradients, _) = network
            .backward_pass(
                &post_activations,
                &pre_activations,
                wanted_output_of_nn.view(),
                0,
            )
            .unwrap();
        for (grad, exp_grad) in layer_gradients.iter().zip(expected_gradient.iter()) {
            assert_abs_diff_eq!(grad, exp_grad, epsilon = 1e-6);
        }

        network.weight_update(&layer_gradients, 0);

        for (weights, exp_weights) in network
            .layers
            .iter()
            .map(|l| &l.weights)
            .zip(expected_new_weights.iter())
        {
            assert_abs_diff_eq!(weights, exp_weights, epsilon = 1e-6);
        }
    }

    // Test backward pass with fixed point values.
    #[rstest]
    #[case::fixed_case_1(
        simple_network_relu_relu::<I32F32>(),
        array![[I32F32::from_num(3.0)]],
        array![[I32F32::from_num(-6.0)]],
        // All gradients zero, no weight update.
        vec![
            array![
                [I32F32::from_num(0.0), I32F32::from_num(0.0)],
                [I32F32::from_num(0.0), I32F32::from_num(0.0)],
                [I32F32::from_num(0.0), I32F32::from_num(0.0)]
            ],
            array![
                [I32F32::from_num(0.0), I32F32::from_num(0.0), I32F32::from_num(0.0), I32F32::from_num(0.0)]
            ]
        ],
        vec![
            array![
                [I32F32::from_num(0.5), I32F32::from_num(1.0)],
                [I32F32::from_num(-0.5), I32F32::from_num(2.0)],
                [I32F32::from_num(0.1), I32F32::from_num(-1.0)]
            ],
            array![
                [I32F32::from_num(0.2), I32F32::from_num(1.0), I32F32::from_num(-1.0), I32F32::from_num(0.5)]
            ]
        ]
    )]
    #[case::fixed_case_2(
        simple_network_relu_relu::<I32F32>(),
        array![[I32F32::from_num(-10.0)]],
        array![[I32F32::from_num(-6.0)]],
        vec![
            array![
                [I32F32::from_num(0.0), I32F32::from_num(0.0)],
                [I32F32::from_num(0.0), I32F32::from_num(0.0)],
                [I32F32::from_num(11.25), I32F32::from_num(-112.5)]
            ],
            array![
                [I32F32::from_num(22.5), I32F32::from_num(0.0), I32F32::from_num(0.0), I32F32::from_num(227.25)]
            ]
        ],
        vec![
            array![
                [
                    I32F32::from_num(0.5) - I32F32::from_num(0.01) * I32F32::from_num(0.0),
                    I32F32::from_num(1.0) - I32F32::from_num(0.01) * I32F32::from_num(0.0)
                ],
                [
                    I32F32::from_num(-0.5) - I32F32::from_num(0.01) * I32F32::from_num(0.0),
                    I32F32::from_num(2.0) - I32F32::from_num(0.01) * I32F32::from_num(0.0)
                ],
                [
                    I32F32::from_num(0.1) - I32F32::from_num(0.01) * I32F32::from_num(11.25),
                    I32F32::from_num(-1.0) - I32F32::from_num(0.01) * I32F32::from_num(-112.5)
                ]
            ],
            array![
                [
                    I32F32::from_num(0.2) - I32F32::from_num(0.01) * I32F32::from_num(22.5),
                    I32F32::from_num(1.0),
                    I32F32::from_num(-1.0),
                    I32F32::from_num(0.5) - I32F32::from_num(0.01) * I32F32::from_num(227.25)
                ]
            ]
        ]
    )]
    #[case::fixed_case_3(
        simple_network_relu_relu::<I32F32>(),
        array![[I32F32::from_num(1.0)]],
        array![[I32F32::from_num(-2.0)]],
        vec![
            array![
                [I32F32::from_num(4.4), I32F32::from_num(4.4)],
                [I32F32::from_num(-4.4), I32F32::from_num(-4.4)],
                [I32F32::from_num(0.0), I32F32::from_num(0.0)]
            ],
            array![
                [I32F32::from_num(4.4), I32F32::from_num(6.6), I32F32::from_num(6.6), I32F32::from_num(0.0)]
            ]
        ],
        vec![
            array![
                [
                    I32F32::from_num(0.5) - I32F32::from_num(0.01) * I32F32::from_num(4.4),
                    I32F32::from_num(1.0) - I32F32::from_num(0.01) * I32F32::from_num(4.4)
                ],
                [
                    I32F32::from_num(-0.5) - I32F32::from_num(0.01) * I32F32::from_num(-4.4),
                    I32F32::from_num(2.0) - I32F32::from_num(0.01) * I32F32::from_num(-4.4)
                ],
                [I32F32::from_num(0.1), I32F32::from_num(-1.0)]
            ],
            array![
                [
                    I32F32::from_num(0.2) - I32F32::from_num(0.01) * I32F32::from_num(4.4),
                    I32F32::from_num(1.0) - I32F32::from_num(0.01) * I32F32::from_num(6.6),
                    I32F32::from_num(-1.0) - I32F32::from_num(0.01) * I32F32::from_num(6.6),
                    I32F32::from_num(0.5) - I32F32::from_num(0.01) * I32F32::from_num(0.0)
                ]
            ]
        ]
    )]
    #[case::fixed_case_4(
        simple_network_relu_relu::<I32F32>(),
        array![[I32F32::from_num(-2.0)]],
        array![[I32F32::from_num(6.0)]],
        vec![
            array![
                [I32F32::from_num(0.0), I32F32::from_num(0.0)],
                [I32F32::from_num(0.0), I32F32::from_num(0.0)],
                [I32F32::from_num(-4.75), I32F32::from_num(9.5)]
            ],
            array![
                [I32F32::from_num(-9.5), I32F32::from_num(0.0), I32F32::from_num(0.0), I32F32::from_num(-19.95)]
            ]
        ],
        vec![
            array![
                [
                    I32F32::from_num(0.5),
                    I32F32::from_num(1.0)
                ],
                [
                    I32F32::from_num(-0.5),
                    I32F32::from_num(2.0)
                ],
                [
                    I32F32::from_num(0.1) - I32F32::from_num(0.01) * I32F32::from_num(-4.75),
                    I32F32::from_num(-1.0) - I32F32::from_num(0.01) * I32F32::from_num(9.5)
                ]
            ],
            array![
                [
                    I32F32::from_num(0.2) - I32F32::from_num(0.01) * I32F32::from_num(-9.5),
                    I32F32::from_num(1.0),
                    I32F32::from_num(-1.0),
                    I32F32::from_num(0.5) - I32F32::from_num(0.01) * I32F32::from_num(-19.95)
                ]
            ]
        ]
    )]
    #[case::fixed_case_5(
        complex_network_relu_relu::<I32F32>(),
        array![[I32F32::from_num(0.0)], [I32F32::from_num(0.0)]],
        array![[I32F32::from_num(6.0)], [I32F32::from_num(1.0)]],
        // Gradients don't vanish (as we keep the ReLU gradient at 1 at x= 0).
        vec![
            array![
                [I32F32::from_num(24.0), I32F32::from_num(0.0), I32F32::from_num(0.0)],
                [I32F32::from_num(0.0), I32F32::from_num(0.0), I32F32::from_num(0.0)]
            ],
            array![
                [I32F32::from_num(-12.0), I32F32::from_num(-6.0), I32F32::from_num(0.0)],
                [I32F32::from_num(0.0), I32F32::from_num(0.0), I32F32::from_num(0.0)]
            ]
        ],
        vec![
            array![
                [I32F32::from_num(0.26), I32F32::from_num(1.0), I32F32::from_num(1.0)],
                [I32F32::from_num(-0.5), I32F32::from_num(2.0), I32F32::from_num(1.0)]
            ],
            array![
                [I32F32::from_num(1.12), I32F32::from_num(-1.94), I32F32::from_num(3.0)],
                [I32F32::from_num(-5.0), I32F32::from_num(4.0), I32F32::from_num(2.0)]
            ]
        ]
    )]
    #[case::fixed_case_6(
        complex_network_relu_relu::<I32F32>(),
        array![[I32F32::from_num(1.0)], [I32F32::from_num(-1.0)]],
        array![[I32F32::from_num(3.0)], [I32F32::from_num(-7.0)]],
        vec![
            array![
                [I32F32::from_num(6.0), I32F32::from_num(6.0), I32F32::from_num(-6.0)],
                [I32F32::from_num(-9.0), I32F32::from_num(-9.0), I32F32::from_num(9.0)]
            ],
            array![
                [I32F32::from_num(-3.0), I32F32::from_num(-1.5), I32F32::from_num(-1.5)],
                [I32F32::from_num(0.0), I32F32::from_num(0.0), I32F32::from_num(0.0)]
            ]
        ],
        vec![
            &array![
                [I32F32::from_num(0.5), I32F32::from_num(1.0), I32F32::from_num(1.0)],
                [I32F32::from_num(-0.5), I32F32::from_num(2.0), I32F32::from_num(1.0)]
            ] - &scalar_mul_fixed_point_array(
                &array![
                    [I32F32::from_num(6.0), I32F32::from_num(6.0), I32F32::from_num(-6.0)],
                    [I32F32::from_num(-9.0), I32F32::from_num(-9.0), I32F32::from_num(9.0)]
                ],
                I32F32::from_num(0.01)
            ),

            &array![
                [I32F32::from_num(1.0), I32F32::from_num(-2.0), I32F32::from_num(3.0)],
                [I32F32::from_num(-5.0), I32F32::from_num(4.0), I32F32::from_num(2.0)]
            ] - &scalar_mul_fixed_point_array(
                &array![
                    [I32F32::from_num(-3.0), I32F32::from_num(-1.5), I32F32::from_num(-1.5)],
                    [I32F32::from_num(0.0), I32F32::from_num(0.0), I32F32::from_num(0.0)]
                ],
                I32F32::from_num(0.01)
            )
        ]
    )]

    fn backward_pass_output_matches_expected_fixed(
        #[case] mut network: FeedForward<I32F32>,
        #[case] input_to_nn: Array2<I32F32>,
        #[case] wanted_output_of_nn: Array2<I32F32>,
        #[case] expected_gradient: Vec<Array2<I32F32>>,
        #[case] expected_new_weights: Vec<Array2<I32F32>>, // Hidden + Output Weights
    ) {
        let (post_activations, pre_activations) = network.forward_pass(input_to_nn.view(), 0);
        let (layer_gradients, _) = network
            .backward_pass(
                &post_activations,
                &pre_activations,
                wanted_output_of_nn.view(),
                0,
            )
            .unwrap();
        let epsilon = I32F32::from_num(1e-6);

        for (grad_matrix, exp_grad_matrix) in layer_gradients.iter().zip(expected_gradient.iter()) {
            for (grad_val, exp_grad_val) in grad_matrix.iter().zip(exp_grad_matrix.iter()) {
                assert!(
                    (*grad_val - *exp_grad_val).abs() <= epsilon,
                    "Expected {:?}, got {:?}",
                    exp_grad_val,
                    grad_val
                );
            }
        }
        network.weight_update(&layer_gradients, 0);

        for (weights_matrix, exp_weights_matrix) in network
            .layers
            .iter()
            .map(|l| &l.weights)
            .zip(expected_new_weights.iter())
        {
            for (w, ew) in weights_matrix.iter().zip(exp_weights_matrix.iter()) {
                assert!(
                    (*w - *ew).abs() <= epsilon,
                    "Expected weight {:?}, got {:?}",
                    ew,
                    w
                );
            }
        }
    }

    // Test forward pass with different activation functions
    #[rstest]
    // Identity/Identity cases
    #[case::f64_identity_identity_case_1(
        simple_network_identity_identity::<f64>(),
        array![[1.0]],
        array![[-0.25]]
    )]
    #[case::f64_identity_identity_case_2(
        simple_network_identity_identity::<f64>(),
        array![[0.0]],
        array![[1.25]]
    )]
    #[case::f64_identity_identity_case_3(
        simple_network_identity_identity::<f64>(),
        array![[-1.0]],
        array![[2.75]]
    )]
    // Identity/ReLU cases
    #[case::f64_identity_relu_case_1(
        simple_network_identity_relu::<f64>(),
        array![[1.0]],
        array![[0.0]]
    )]
    #[case::f64_identity_relu_case_2(
        simple_network_identity_relu::<f64>(),
        array![[0.0]],
        array![[1.25]]
    )]
    #[case::f64_identity_relu_case_3(
        simple_network_identity_relu::<f64>(),
        array![[-1.0]],
        array![[2.75]]
    )]
    // ReLU/Identity cases
    #[case::f64_relu_identity_case_1(
        simple_network_relu_identity::<f64>(),
        array![[1.0]],
        array![[0.2]]
    )]
    #[case::f64_relu_identity_case_2(
        simple_network_relu_identity::<f64>(),
        array![[0.0]],
        array![[0.75]]
    )]
    #[case::f64_relu_identity_case_3(
        simple_network_relu_identity::<f64>(),
        array![[-1.0]],
        array![[0.75]]
    )]

    fn test_forward_pass_f64_different_activations(
        #[case] network: FeedForward<f64>,
        #[case] input: Array2<f64>,
        #[case] expected_output: Array2<f64>,
    ) {
        let (post_activations, _) = network.forward_pass(input.view(), 0);
        let output = post_activations.last().unwrap();

        // Compare element-wise with a small epsilon
        let epsilon = 1e-6;
        for (a, b) in output.iter().zip(expected_output.iter()) {
            assert!((*a - *b).abs() <= epsilon, "Expected {:?}, got {:?}", b, a);
        }
    }
}
