//! This file implements a function to emulate a FALAFEL run on random data.

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
use rand::SeedableRng;
use rand::rngs::ThreadRng;
use rand_chacha::ChaChaRng;

use crate::proof::LocalTrainingProofConstants;
#[allow(unused_imports)]
use log::info;
use num_traits::Pow;
use rand::distributions::Uniform;
// Import the common functions
use super::common::{audit_clients, compute_max_pedersen_gens};

#[derive(Clone, Debug)]
pub struct RandomExperimentConfig<T: Value> {
    /// Number of samples for each client.
    pub n_samples_vec: Vec<usize>,
    /// Number of input features.
    pub n_input: usize,
    /// Number of hidden layers + how many neurons.
    pub n_hidden: Vec<usize>,
    /// Activation functions for each layer, including output layer.
    pub activation_functions: Vec<ActivationFunction>,
    /// Number of output features.
    pub n_output: usize,
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
    /// Number of integer bits for the initial data/weights
    pub initial_integer_bits: usize,
}

impl<T: Value> RandomExperimentConfig<T> {
    pub fn validate(&self) -> FalafelResult<()> {
        if self.n_samples_vec.is_empty() {
            return Err(FalafelError::InvalidConfig(
                "n_samples_vec must not be empty".to_string(),
            ));
        }
        if self.n_input == 0 || self.n_output == 0 {
            return Err(FalafelError::InvalidConfig(
                "n_input and n_output must be greater than 0".to_string(),
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

fn set_up_clients<'a, T: Value, G: CurveGroup>(
    n_samples_vec: &[usize],
    n_input: usize,
    n_output: usize,
    decimal_bits: usize,
    initial_integer_bits: usize,
    generators: &'a Generators<G>,
    rng: &mut ThreadRng,
) -> Vec<Client<'a, T, G>> {
    #[cfg(feature = "print-trace")]
    {
        info!("Generating random datasets...");
        info!(
            "   Dataset dimensions: {} inputs × {} outputs",
            n_input, n_output
        );
        info!("   Sample distribution: {:?}", n_samples_vec);
    }

    // Create clients with random datasets.
    n_samples_vec
        .iter()
        .enumerate()
        .map(|(idx, &n_samples)| {
            #[cfg(feature = "print-trace")]
            info!("   Client {}: generating {} random samples", idx, n_samples);
            let dataset = Dataset::<T>::new_random(
                n_samples,
                n_input,
                n_output,
                &Uniform::new(
                    2.0.pow((initial_integer_bits - 1) as u8) * -1.0,
                    2.0.pow((initial_integer_bits - 1) as u8),
                ),
                decimal_bits,
                rng,
            );
            Client::new(idx, dataset, generators)
        })
        .collect()
}

pub fn run_falafel_on_random_data<T: Value, G: CurveGroup>(
    config: RandomExperimentConfig<T>,
) -> FalafelResult<()> {
    #[cfg(feature = "print-trace")]
    {
        info!("Starting FALAFEL protocol with random data");
        info!(
            "Configuration: {} clients, {} total samples",
            config.n_clients(),
            config.total_samples()
        );
    }
    config.validate()?; // Validate the configuration before proceeding
    let RandomExperimentConfig {
        n_samples_vec,
        n_input,
        n_hidden,
        activation_functions,
        n_output,
        n_global_iter,
        n_local_iter,
        batch_size,
        learning_rate,
        total_bits,
        decimal_bits,
        initial_integer_bits,
    } = config;

    // SETUP
    let mut crypto_rng = ChaChaRng::from_entropy();
    let mut rng = rand::thread_rng();

    // public parameters
    #[cfg(feature = "print-trace")]
    info!("Generating cryptographic parameters...");
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

    // PROTOCOL
    // initialization
    #[cfg(feature = "print-trace")]
    info!("Neural Network Initialization");
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
    let mut clients = set_up_clients::<T, G>(
        &n_samples_vec,
        n_input,
        n_output,
        decimal_bits,
        initial_integer_bits,
        &generators,
        &mut rng,
    );
    #[cfg(feature = "print-trace")]
    info!("   Auditor, server, and {} clients ready", clients.len());

    // DATASET AUDIT
    #[cfg(feature = "print-trace")]
    info!("Auditing client datasets...");

    audit_clients(&mut clients, &auditor, &mut crypto_rng)?;

    #[cfg(feature = "print-trace")]
    info!("All client datasets successfully audited");

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

                assert!(false);

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

    // FINAL STUFF
    #[cfg(feature = "print-trace")]
    info!("FALAFEL protocol completed successfully!");
    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::neural_network::ActivationFunction;
    use crate::util::RandomMatrix;
    use fixed::types::{I23F9, I32F0, I32F32};

    #[test]
    fn protocol_runs_successfully_with_fixed_point_arithmetic() {
        full_protocol::<I32F32, ark_curve25519::EdwardsProjective>();
        full_protocol::<I23F9, ark_curve25519::EdwardsProjective>();
        full_protocol::<I32F0, ark_curve25519::EdwardsProjective>();
    }

    #[test]
    fn protocol_runs_successfully_with_integer_arithmetic() {
        full_protocol::<i128, ark_curve25519::EdwardsProjective>();
        full_protocol::<i64, ark_curve25519::EdwardsProjective>();
        full_protocol::<i32, ark_curve25519::EdwardsProjective>();
    }

    fn full_protocol<T: Value + RandomMatrix, G: CurveGroup>() {
        // PARAMETERS
        let n_parties = 3;
        let n_global_iter = 1;
        let n_local_iter = 1;
        let batch_size = 4;
        let learning_rate = 0.5;
        let sample_sizes = vec![4, 4, 4];
        let n_features = 2;
        let n_labels = 2;
        let n_hidden = vec![2, 2];
        let activation_functions = vec![ActivationFunction::ReLU; n_hidden.len() + 1];
        let total_bits = 15;
        let decimal_bits = 2;
        let initial_integer_bits = 2;

        // actual protocol run
        assert_eq!(sample_sizes.len(), n_parties);
        let config = RandomExperimentConfig {
            n_samples_vec: sample_sizes,
            n_input: n_features,
            n_hidden,
            activation_functions,
            n_output: n_labels,
            n_global_iter,
            n_local_iter,
            batch_size,
            learning_rate: T::from_f64(learning_rate * (1 << decimal_bits) as f64).unwrap(),
            total_bits,
            decimal_bits,
            initial_integer_bits,
        };
        run_falafel_on_random_data::<T, G>(config).unwrap();
    }
}
