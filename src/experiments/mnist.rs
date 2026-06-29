//! This file implements a FALAFEL run with MNIST data.

use super::common::{audit_clients, compute_max_pedersen_gens};
use crate::auditor::Auditor;
use crate::client::{Client, ClientSchemes};
use crate::crypto_schemes::Generators;
use crate::dataset::Dataset;
use crate::message::{
    InitializeMessage, InitializeMessageSerializable, LocalUpdateMessage,
    LocalUpdateMessageSerializable,
};
use crate::neural_network::{ActivationFunction, FeedForward};
use crate::prelude::*;
use crate::server::Server;
use crate::util::get_iv;
use mnist::{Mnist, MnistBuilder};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::ThreadRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaChaRng;
use std::path::Path;

use crate::proof::LocalTrainingProofConstants;
#[allow(unused_imports)]
use log::info;
use rand::distributions::Uniform;

const MNIST_FILES: [&str; 4] = [
    "train-images-idx3-ubyte",
    "train-labels-idx1-ubyte",
    "t10k-images-idx3-ubyte",
    "t10k-labels-idx1-ubyte",
];

const MNIST_INPUT_SIZE: usize = 784;
const MNIST_OUTPUT_SIZE: usize = 10;

#[derive(Clone, Debug)]
pub struct MnistConfig<T: Value> {
    /// Number of samples for each client
    pub n_samples_vec: Vec<usize>,
    /// Number of hidden layers + how many neurons.
    pub n_hidden: Vec<usize>,
    /// Activation functions for each layer, including output layer.
    pub activation_functions: Vec<ActivationFunction>,
    /// Number of global iterations.
    pub n_global_iter: usize,
    /// Number of local iterations per client.
    pub n_local_iter: usize,
    /// Batch size for local training.
    pub batch_size: usize,
    /// Learning rate for the neural network.
    pub learning_rate: T,
    /// Total number of bits for a number (integral + decimal part)
    pub total_bits: usize,
    /// Number of decimal bits
    pub decimal_bits: usize,
    /// Path to MNIST data files.
    pub mnist_path: String,
    /// Number of training, test and validation samples
    pub n_train: u32,
    pub n_test: u32,
    pub n_valid: u32,
}

impl<T: Value> MnistConfig<T> {
    pub fn validate(&self) -> FalafelResult<()> {
        if self.n_samples_vec.is_empty() {
            return Err(FalafelError::InvalidConfig(
                "n_samples_vec must not be empty".to_string(),
            ));
        }
        if self.n_hidden.is_empty() {
            return Err(FalafelError::InvalidConfig(
                "n_hidden must not be empty".to_string(),
            ));
        }
        if self.activation_functions.len() != self.n_hidden.len() + 1 {
            return Err(FalafelError::InvalidConfig(
                "activation_functions length must be n_hidden length + 1".to_string(),
            ));
        }
        if self.n_global_iter == 0 || self.n_local_iter == 0 {
            return Err(FalafelError::InvalidConfig(
                "n_global_iter and n_local_iter must be greater than 0".to_string(),
            ));
        }
        if self.batch_size == 0 {
            return Err(FalafelError::InvalidConfig(
                "batch_size must be greater than 0".to_string(),
            ));
        }
        if !self.learning_rate.is_positive() {
            return Err(FalafelError::InvalidConfig(
                "learning_rate must be greater than 0".to_string(),
            ));
        }
        if self.decimal_bits >= self.total_bits {
            return Err(FalafelError::InvalidConfig(
                "total_bits must be greater than decimal_bits".to_string(),
            ));
        }
        if self.n_train <= 0 || self.n_test <= 0 || self.n_valid <= 0 {
            return Err(FalafelError::InvalidConfig(
                "Training, test, and validation set sizes must all be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
    /// Returns the total number of clients in this configuration.
    pub fn n_clients(&self) -> usize {
        self.n_samples_vec.len()
    }

    /// Returns the total number of samples across all clients.
    pub fn total_samples(&self) -> usize {
        self.n_samples_vec.iter().sum()
    }
}

/// Load MNIST dataset from disk.
fn load_mnist(path: &str, n_train: u32, n_valid: u32, n_test: u32) -> FalafelResult<Mnist> {
    println!("Loading MNIST dataset from path: {}", path);
    println!(
        "Dataset sizes - Train: {}, Valid: {}, Test: {}",
        n_train, n_valid, n_test
    );

    println!("Checking for required MNIST files...");
    for file in MNIST_FILES {
        let full_path = Path::new(path).join(file);
        if !full_path.exists() {
            println!("❌ File not found: {}", full_path.display());
            return Err(FalafelError::InvalidConfig(format!(
                "Required MNIST file not found: {}",
                full_path.display()
            )));
        } else {
            println!("✅ Found: {}", full_path.display());
        }
    }

    println!("All required files found. Building MNIST dataset...");

    let mnist = MnistBuilder::new()
        .base_path(path)
        .label_format_digit()
        .training_set_length(n_train)
        .validation_set_length(n_valid)
        .test_set_length(n_test)
        .finalize();

    println!("✅ MNIST dataset loaded successfully!");
    Ok(mnist)
}

/// Convert MNIST data to our Dataset format.
fn preprocess_mnist_data<T: Value>(images: &[u8], labels: &[u8], indices: &[usize]) -> Dataset<T> {
    let n_samples = indices.len();
    let n_features = MNIST_INPUT_SIZE;
    let n_labels = MNIST_OUTPUT_SIZE;

    // Create arrays directly in column-major format
    let samples = Array2::from_shape_fn((n_features, n_samples), |(feature, sample)| {
        let idx = indices[sample];
        let pixel = images[idx * n_features + feature];
        T::from_u8(pixel).unwrap()
    });

    let labels = Array2::from_shape_fn((n_labels, n_samples), |(label, sample)| {
        let idx = indices[sample];
        if labels[idx] as usize == label {
            T::one()
        } else {
            T::zero()
        }
    });

    Dataset::new(samples, labels)
}

/// Set up clients with MNIST data.
fn set_up_mnist_clients<'a, T: Value, G: CurveGroup>(
    mnist: &Mnist,
    n_samples_vec: &[usize],
    generators: &'a Generators<G>,
    rng: &mut ThreadRng,
) -> Vec<Client<'a, T, G>> {
    let n_total = mnist.trn_lbl.len();
    let n_clients = n_samples_vec.len();

    // Verify we have enough samples
    let total_requested: usize = n_samples_vec.iter().sum();
    if total_requested > n_total {
        panic!(
            "Requested {} samples but only {} available",
            total_requested, n_total
        );
    }

    let mut indices: Vec<usize> = (0..n_total).collect();
    indices.shuffle(rng);

    let mut start = 0;
    (0..n_clients)
        .map(|i| {
            let n_samples = n_samples_vec[i];
            let end = start + n_samples;

            let client_indices = &indices[start..end];
            let dataset = preprocess_mnist_data(&mnist.trn_img, &mnist.trn_lbl, client_indices);

            start = end; // Move start for next client

            Client::new(i, dataset, generators)
        })
        .collect()
}

pub fn run_falafel_on_mnist<T: Value, G: CurveGroup>(config: MnistConfig<T>) -> FalafelResult<()> {
    #[cfg(feature = "print-trace")]
    {
        info!("Starting FALAFEL protocol with MNIST data");
        info!(
            "Configuration: {} clients, {} total samples",
            config.n_clients(),
            config.total_samples()
        );
    }
    config.validate()?;
    let MnistConfig {
        n_samples_vec,
        n_hidden,
        activation_functions,
        n_global_iter,
        n_local_iter,
        batch_size,
        learning_rate,
        total_bits,
        decimal_bits,
        mnist_path,
        n_train,
        n_test,
        n_valid,
    } = config;

    // SETUP
    let mut crypto_rng = ChaChaRng::from_entropy();
    let mut rng = rand::thread_rng();

    // Load MNIST
    #[cfg(feature = "print-trace")]
    info!("Loading MNIST dataset from {}...", mnist_path);

    let mnist = load_mnist(&mnist_path, n_train, n_valid, n_test)?;

    #[cfg(feature = "print-trace")]
    info!("Loaded {} training samples", mnist.trn_lbl.len());

    let n_input = MNIST_INPUT_SIZE;
    let n_output = MNIST_OUTPUT_SIZE;

    #[cfg(feature = "print-trace")]
    info!("Generating cryptographic parameters...");

    // public parameters
    let n_pedersen_vec_gens = compute_max_pedersen_gens(
        *n_samples_vec.iter().max().unwrap(),
        n_local_iter,
        batch_size,
        n_input,
        &n_hidden,
        n_output,
        total_bits,
    );
    let generators = Generators::<G>::new(n_pedersen_vec_gens);
    #[cfg(feature = "print-trace")]
    info!(
        "   Generated {} Pedersen vector generators",
        n_pedersen_vec_gens
    );

    let client_schemes = ClientSchemes::setup(&generators);

    let auditor_pp = AuditorSignatureScheme::setup(&mut crypto_rng)
        .map_err(|e| FalafelError::ArkworksError(e.to_string()))?;

    // parties
    let auditor = Auditor::new(&auditor_pp, &mut crypto_rng)?;

    let nn = FeedForward::new_with_random_weights(
        n_input,
        n_output,
        n_hidden.clone(),
        activation_functions.clone(),
        learning_rate,
        &Uniform::new_inclusive(-1.0, 1.0),
        decimal_bits,
        &mut rng,
    )?;

    let constants = LocalTrainingProofConstants::new(
        total_bits,
        decimal_bits,
        n_local_iter,
        batch_size,
        n_input,
        n_samples_vec[0],
        &n_hidden,
        n_output,
        &nn,
    );

    let mut server = Server::new(
        &n_samples_vec,
        &constants.cm_w_single_iteration_dimensions,
        constants.clone(),
        &generators,
        &auditor,
        activation_functions.clone(),
        learning_rate,
    );
    let mut clients = set_up_mnist_clients::<T, G>(&mnist, &n_samples_vec, &generators, &mut rng);
    #[cfg(feature = "print-trace")]
    info!("   Auditor, server, and {} clients ready", clients.len());

    // DATASET AUDIT
    #[cfg(feature = "print-trace")]
    info!("Auditing client datasets...");

    audit_clients(&mut clients, &auditor, &mut crypto_rng)?;

    #[cfg(feature = "print-trace")]
    info!("All client datasets successfully audited");

    // PROTOCOL
    // initialization
    #[cfg(feature = "print-trace")]
    info!("Neural Network Initialization");

    let initialize_message = server.initialize_neural_network(nn)?;
    for client in &mut clients {
        client.initialize_neural_network(&initialize_message)?;
        client.shuffle_dataset(&mut rng)?;
    }

    #[cfg(feature = "print-trace")]
    info!("All clients initialized with global model");

    // iterations
    #[cfg(feature = "print-trace")]
    {
        info!("Federated Learning Phase");
        info!(
            "Starting {} global rounds with {} local iterations each",
            n_global_iter, n_local_iter
        );
    }

    println!("entering iterations");
    for global_iter in 0..n_global_iter {
        let mut client_commitments: Vec<Vec<G::Affine>> = Vec::with_capacity(clients.len());

        let initialize_message: InitializeMessageSerializable<T> =
            bitcode::deserialize(&initialize_message)
                .map_err(|e| FalafelError::DeserializationError(e.to_string()))?;
        let initialize_message: InitializeMessage<T, G> = initialize_message.to_regular()?;

        for client in &mut clients {
            let client_message = client.prove_local_training_full(
                n_local_iter,
                batch_size,
                global_iter,
                total_bits,
                decimal_bits,
                &mut crypto_rng,
            )?;

            {
                let deserialized_message_client: LocalUpdateMessageSerializable<T> =
                    bitcode::deserialize(&client_message.0).unwrap();
                let deserialized_message_client: LocalUpdateMessage<G, T> =
                    deserialized_message_client.to_regular()?;

                deserialized_message_client
                    .proof
                    .verify(
                        &get_iv(client.id, global_iter),
                        &auditor_pp,
                        auditor.get_pk(),
                        &initialize_message.cm_initial_weights,
                        &client.schemes,
                    )
                    .unwrap();

                let client_commitment = server
                    .extract_network_commitments_local_proof(
                        &get_iv(client.id, global_iter),
                        deserialized_message_client.proof.clone(),
                        &initialize_message.cm_initial_weights,
                    )?
                    .1
                    .row_coms;
                client_commitments.push(client_commitment);
                server.receive_and_verify_local_update(
                    client_message,
                    &initialize_message.cm_initial_weights,
                )?;
            }
        }

        let (new_weights, aggregation_proof) =
            server.aggregate_and_prove(&initialize_message.cm_initial_weights)?;
        let (aggregation_proof, r_new_weights) = aggregation_proof.unwrap();
        let cm_new_weights = aggregation_proof
            .verify_weight_aggregation(
                &get_iv(usize::MAX, global_iter),
                &client_schemes,
                client_commitments,
                constants.cm_w_single_iteration_dimensions,
            )
            .unwrap();

        for client in &mut clients {
            client.set_new_network(
                cm_new_weights.clone(),
                new_weights.clone(),
                r_new_weights.clone(),
            );
        }
    }

    #[cfg(feature = "print-trace")]
    info!("FALAFEL protocol completed successfully!");

    // FINAL STUFF
    Ok(())
}

#[cfg(test)]
mod test {
    const MNIST_DATA_PATH: &str = "./src/experiments/data_mnist";
    use super::*;
    use crate::neural_network::ActivationFunction;
    use fixed::types::{I16F16, I23F9, I24F8};

    #[test]
    fn test_load_mnist() {
        // Test with valid MNIST path
        let mnist_path = MNIST_DATA_PATH;

        // Test successful loading
        let result = load_mnist(mnist_path, 1000, 100, 100);
        assert!(result.is_ok());

        let mnist = result.unwrap();
        assert_eq!(mnist.trn_lbl.len(), 1000);
        assert_eq!(mnist.val_lbl.len(), 100);
        assert_eq!(mnist.tst_lbl.len(), 100);

        // Test with invalid path
        let result = load_mnist("./invalid/path", 100, 10, 10);
        assert!(result.is_err());
        match result {
            Err(FalafelError::InvalidConfig(msg)) => {
                assert!(msg.contains("Required MNIST file not found"));
            }
            _ => panic!("Expected InvalidConfig error"),
        }
    }

    #[test]
    fn test_preprocess_mnist_data() {
        let mnist_path = MNIST_DATA_PATH;

        // Load actual MNIST data
        let mnist = load_mnist(mnist_path, 100, 10, 10).unwrap();

        // Process first 5 samples
        let indices = vec![0, 1, 2, 3, 4];
        let dataset: Dataset<i32> = preprocess_mnist_data(&mnist.trn_img, &mnist.trn_lbl, &indices);

        // Check dimensions
        assert_eq!(dataset.get_samples().shape(), &[784, 5]);
        assert_eq!(dataset.get_labels().shape(), &[10, 5]);

        // Verify one-hot encoding matches original labels
        for (i, &idx) in indices.iter().enumerate() {
            let label = mnist.trn_lbl[idx] as usize;
            assert_eq!(dataset.get_labels()[[label, i]], 1);

            // Check all other positions are 0
            for j in 0..10 {
                if j != label {
                    assert_eq!(dataset.get_labels()[[j, i]], 0);
                }
            }
        }
    }

    #[test]
    fn mnist_runs_successfully_with_fixed_point_arithmetic() {
        mnist_protocol::<I16F16, ark_curve25519::EdwardsProjective>();
        mnist_protocol::<I23F9, ark_curve25519::EdwardsProjective>();
        mnist_protocol::<I24F8, ark_curve25519::EdwardsProjective>();
    }

    #[test]
    fn mnist_runs_successfully_with_integer_arithmetic() {
        mnist_protocol::<i128, ark_curve25519::EdwardsProjective>();
        mnist_protocol::<i64, ark_curve25519::EdwardsProjective>();
        mnist_protocol::<i32, ark_curve25519::EdwardsProjective>();
    }

    fn mnist_protocol<T: Value, G: CurveGroup>() {
        // PARAMETERS
        let n_global_iter = 1;
        let n_local_iter = 2;
        let batch_size = 10;
        let n_samples_vec = vec![10, 10, 10];
        let n_hidden = vec![8, 8];
        let activation_functions = vec![
            ActivationFunction::ReLU,
            ActivationFunction::ReLU,
            ActivationFunction::ReLU,
        ];
        let total_bits = 20;
        let decimal_bits = 5;

        let n_train = 100;
        let n_test = 10;
        let n_valid = 10;

        // actual protocol run
        let config = MnistConfig {
            n_samples_vec,
            n_hidden,
            activation_functions,
            n_global_iter,
            n_local_iter,
            batch_size,
            learning_rate: T::from_f64(1.0).unwrap(),
            total_bits,
            decimal_bits,
            mnist_path: MNIST_DATA_PATH.to_string(),
            n_train,
            n_test,
            n_valid,
        };

        // Only run if MNIST data exists
        if Path::new(MNIST_DATA_PATH).exists() {
            run_falafel_on_mnist::<T, G>(config).unwrap();
        }
    }
}
