use crate::neural_network::{
    ActivationFunction::{Identity, ReLU},
    FeedForward,
};
use crate::prelude::Value;
use ndarray::array;
use num_traits::FromPrimitive;

/// Creates a simple network with configurable activation functions
///
/// # Architecture
/// - 1 input → 1 hidden layer (3 neurons) → 1 output
///
/// # Parameters
/// - `hidden_activation`: Activation function for the hidden layer
/// - `output_activation`: Activation function for the output layer
///
/// # Returns
/// A `FeedForward<T>` network with predefined weights for testing
pub fn simple_network<T: Value + FromPrimitive + Copy>(
    hidden_activation: crate::neural_network::ActivationFunction,
    output_activation: crate::neural_network::ActivationFunction,
) -> FeedForward<T> {
    // Hidden layer weights: [bias, weight]
    let weights_hidden = array![
        [T::from_f64(0.5).unwrap(), T::from_f64(1.0).unwrap()],
        [T::from_f64(-0.5).unwrap(), T::from_f64(2.0).unwrap()],
        [T::from_f64(0.1).unwrap(), T::from_f64(-1.0).unwrap()]
    ];

    // Output layer weights: [bias, weight_1, weight_2, weight_3]
    let weights_output = array![[
        T::from_f64(0.2).unwrap(),
        T::from_f64(1.0).unwrap(),
        T::from_f64(-1.0).unwrap(),
        T::from_f64(0.5).unwrap()
    ]];

    let weights = vec![weights_hidden, weights_output];
    let activations = vec![hidden_activation, output_activation];

    FeedForward::new_from_weights(weights, activations, T::from_f64(0.01).unwrap())
        .expect("Failed to create simple network")
}

/// Creates a complex network with configurable activation functions
///
/// # Architecture
/// - 2 input neurons → 1 hidden layer (2 neurons) → 2 output neurons
///
/// # Parameters
/// - `hidden_activation`: Activation function for the hidden layer
/// - `output_activation`: Activation function for the output layer
///
/// # Returns
/// A `FeedForward<T>` network with predefined weights for testing
pub fn complex_network<T: Value + FromPrimitive + Copy>(
    hidden_activation: crate::neural_network::ActivationFunction,
    output_activation: crate::neural_network::ActivationFunction,
) -> FeedForward<T> {
    // Hidden layer weights: [bias, weight1, weight2]
    let weights_hidden = array![
        [
            T::from_f64(0.5).unwrap(),
            T::from_f64(1.0).unwrap(),
            T::from_f64(1.0).unwrap()
        ],
        [
            T::from_f64(-0.5).unwrap(),
            T::from_f64(2.0).unwrap(),
            T::from_f64(1.0).unwrap()
        ]
    ];

    // Output layer weights: [bias, weight1, weight2]
    let weights_output = array![
        [
            T::from_f64(1.0).unwrap(),
            T::from_f64(-2.0).unwrap(),
            T::from_f64(3.0).unwrap()
        ],
        [
            T::from_f64(-5.0).unwrap(),
            T::from_f64(4.0).unwrap(),
            T::from_f64(2.0).unwrap()
        ]
    ];

    let weights = vec![weights_hidden, weights_output];
    let activations = vec![hidden_activation, output_activation];

    FeedForward::new_from_weights(weights, activations, T::from_f64(0.01).unwrap())
        .expect("Failed to create complex network")
}

// Convenience functions for common simple network configurations
pub fn simple_network_relu_relu<T: Value + FromPrimitive + Copy>() -> FeedForward<T> {
    simple_network(ReLU, ReLU)
}

pub fn simple_network_identity_identity<T: Value + FromPrimitive + Copy>() -> FeedForward<T> {
    simple_network(Identity, Identity)
}

pub fn simple_network_identity_relu<T: Value + FromPrimitive + Copy>() -> FeedForward<T> {
    simple_network(Identity, ReLU)
}

pub fn simple_network_relu_identity<T: Value + FromPrimitive + Copy>() -> FeedForward<T> {
    simple_network(ReLU, Identity)
}

// Convenience functions for common complex network configurations
pub fn complex_network_relu_relu<T: Value + FromPrimitive + Copy>() -> FeedForward<T> {
    complex_network(ReLU, ReLU)
}

pub fn complex_network_identity_identity<T: Value + FromPrimitive + Copy>() -> FeedForward<T> {
    complex_network(Identity, Identity)
}
