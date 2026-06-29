use std::time::Instant;

use crate::auditor::Auditor;
use crate::client::{Client, ClientSchemes};
use crate::crypto_schemes::Generators;
use crate::dataset::Dataset;
use crate::message::{
    InitializeMessage, InitializeMessageSerializable, LocalUpdateMessage,
    LocalUpdateMessageSerializable,
};
use crate::neural_network::FeedForward;
use crate::prelude::*;
use crate::proof::LocalTrainingProofConstants;
use crate::proof::bench::PhaseTiming;
use crate::server::Server;
use crate::util::get_iv;
use ndarray::Array2;
use num_traits::Pow;
use rand::SeedableRng;
use rand::distributions::Uniform;
use rand_chacha::ChaChaRng;

use super::common::{audit_clients, compute_max_pedersen_gens};
use super::random::RandomExperimentConfig;

#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    pub batch_size: usize,
    pub phase: String,
    pub prover_time_s: f64,
    pub verifier_time_s: f64,
    pub proof_size_bytes: usize,
}

#[derive(Clone, Debug)]
pub struct SingleRunResult {
    pub prover_timings: PhaseTiming,
    pub verifier_timings: PhaseTiming,
    pub aggregation_prove_s: f64,
    pub aggregation_verify_s: f64,
    pub local_proof_size_bytes: usize,
    pub aggregation_proof_size_bytes: usize,
}

fn bit_length<T: Value>(v: &T) -> usize {
    let abs_val = v.abs();
    let bytes = abs_val.to_le_bytes();
    let bytes = bytes.as_ref(); // &[u8]
    for (i, &b) in bytes.iter().enumerate().rev() {
        if b != 0 {
            return i * 8 + (8 - b.leading_zeros() as usize);
        }
    }
    0
}

fn print_weights<T: Value>(label: &str, layers: &[Array2<T>]) {
    eprintln!("=== {} ===", label);
    for (i, w) in layers.iter().enumerate() {
        let (rows, cols) = w.dim();
        eprintln!("layer {} ({}x{}):", i, rows, cols);
        let mut max_bits: usize = 0;
        for r in 0..rows {
            let _elems: Vec<String> = (0..cols)
                .map(|c| {
                    let v = &w[[r, c]];
                    let bits = bit_length(v);
                    if bits > max_bits {
                        max_bits = bits;
                    }
                    format!("{:>6?}({}b)", v, bits)
                })
                .collect();
        }
        eprintln!(" layer {} -> max_bits={}", i, max_bits);
    }
    eprintln!();
}

pub fn run_falafel_benchmark<T: Value, G: CurveGroup>(
    config: RandomExperimentConfig<T>,
) -> FalafelResult<Vec<BenchmarkResult>> {
    config.validate()?;
    let RandomExperimentConfig {
        n_samples_vec,
        n_input,
        n_hidden,
        activation_functions,
        n_output,
        n_global_iter: _,
        n_local_iter,
        batch_size,
        learning_rate,
        total_bits,
        decimal_bits,
        initial_integer_bits,
    } = config;

    let mut crypto_rng = ChaChaRng::from_entropy();
    let mut rng = rand::thread_rng();

    let n_pedersen_vec_gens = compute_max_pedersen_gens(
        *n_samples_vec.iter().max().unwrap(),
        n_local_iter,
        batch_size,
        n_input,
        &n_hidden,
        n_output,
        total_bits,
    );
    eprintln!(
        "[batch_size={}] generating {} generators...",
        batch_size, n_pedersen_vec_gens
    );
    let generators = Generators::<G>::new(n_pedersen_vec_gens);
    let client_schemes = ClientSchemes::setup(&generators);
    let repetitions = 10;
    eprintln!(
        "[batch_size={}] setup complete, starting {} repetitions",
        batch_size, repetitions
    );
    let auditor_pp = AuditorSignatureScheme::setup(&mut crypto_rng)
        .map_err(|e| FalafelError::ArkworksError(e.to_string()))?;
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

    let init_weights: Vec<_> = nn.layers.iter().map(|l| l.weights.clone()).collect();
    print_weights("INITIAL WEIGHTS", &init_weights);

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

    let mut raw_runs: Vec<SingleRunResult> = Vec::with_capacity(repetitions);

    for rep in 0..repetitions {
        let mut server = Server::new(
            &n_samples_vec,
            &constants.cm_w_single_iteration_dimensions,
            constants.clone(),
            &generators,
            &auditor,
            activation_functions.clone(),
            learning_rate,
        );

        let mut clients: Vec<Client<T, G>> = n_samples_vec
            .iter()
            .enumerate()
            .map(|(idx, &n_samples)| {
                let dataset = Dataset::<T>::new_random(
                    n_samples,
                    n_input,
                    n_output,
                    &Uniform::new(
                        2.0.pow((initial_integer_bits - 1) as u8) * -1.0,
                        2.0.pow((initial_integer_bits - 1) as u8),
                    ),
                    decimal_bits,
                    &mut rng,
                );
                Client::new(idx, dataset, &generators)
            })
            .collect();

        audit_clients(&mut clients, &auditor, &mut crypto_rng)?;

        let initialize_message = server.initialize_neural_network(nn.clone())?;
        for client in &mut clients {
            client.initialize_neural_network(&initialize_message)?;
            client.shuffle_dataset(&mut rng)?;
        }

        let global_iter = 0usize;
        let n_clients = clients.len();
        let mut client_commitments: Vec<Vec<G::Affine>> = Vec::with_capacity(n_clients);
        let mut all_prover_timings: Vec<PhaseTiming> = Vec::with_capacity(n_clients);
        let mut all_verifier_timings: Vec<PhaseTiming> = Vec::with_capacity(n_clients);
        let mut local_proof_size: usize = 0;

        let initialize_message: InitializeMessageSerializable<T> =
            bitcode::deserialize(&initialize_message)
                .map_err(|e| FalafelError::DeserializationError(e.to_string()))?;
        let initialize_message: InitializeMessage<T, G> = initialize_message.to_regular()?;

        for (client_idx, client) in clients.iter_mut().enumerate() {
            eprintln!(
                "  [batch_size={}] rep {}/{}: proving client {}/{}...",
                batch_size,
                rep + 1,
                repetitions,
                client_idx + 1,
                n_clients
            );
            let timed_result = client.prove_local_training_full_timed(
                n_local_iter,
                batch_size,
                global_iter,
                total_bits,
                decimal_bits,
                &mut rng,
            )?;

            let (message_bytes, weights_bytes, witness_bytes) = timed_result.serialized;

            let client_weights: Vec<Array2<T>> = bitcode::deserialize(&weights_bytes).unwrap();
            print_weights(
                &format!("CLIENT {} WEIGHTS (after local training)", client_idx),
                &client_weights,
            );

            let prover_timings = timed_result.timings;

            let deserialized: LocalUpdateMessageSerializable<T> =
                bitcode::deserialize(&message_bytes)
                    .map_err(|e| FalafelError::DeserializationError(e.to_string()))?;
            let deserialized: LocalUpdateMessage<G, T> = deserialized.to_regular()?;

            local_proof_size = deserialized.proof.proof.len();

            eprintln!(
                "  [batch_size={}] rep {}/{}: verifying client {}/{}...",
                batch_size,
                rep + 1,
                repetitions,
                client_idx + 1,
                n_clients
            );
            let verifier_timings = deserialized
                .proof
                .verify_timed(
                    &get_iv(client.id, global_iter),
                    &auditor_pp,
                    auditor.get_pk(),
                    &initialize_message.cm_initial_weights,
                    &client.schemes,
                )
                .map_err(|e| FalafelError::ProofError(e.to_string()))?;

            all_prover_timings.push(prover_timings);
            all_verifier_timings.push(verifier_timings);

            let commitment = server
                .extract_network_commitments_local_proof(
                    &get_iv(client.id, global_iter),
                    deserialized.proof.clone(),
                    &initialize_message.cm_initial_weights,
                )
                .map_err(|e| FalafelError::ProofError(e.to_string()))?
                .1
                .row_coms;
            client_commitments.push(commitment);
            server
                .receive_and_verify_local_update(
                    (message_bytes, weights_bytes, witness_bytes),
                    &initialize_message.cm_initial_weights,
                )
                .map_err(|e| FalafelError::ProofError(e.to_string()))?;
        }

        eprintln!(
            "  [batch_size={}] rep {}/{}: aggregation...",
            batch_size,
            rep + 1,
            repetitions
        );
        let t_agg_prove = Instant::now();
        let (_new_weights, aggregation_proof) =
            server.aggregate_and_prove(&initialize_message.cm_initial_weights)?;
        let (aggregation_proof, _r_new_weights) =
            aggregation_proof.map_err(|e| FalafelError::ProofError(format!("{:?}", e)))?;
        let aggregation_prove_s = t_agg_prove.elapsed().as_secs_f64();
        let aggregation_proof_size = aggregation_proof.proof.len();

        let t_agg_verify = Instant::now();
        let _cm_new_weights = aggregation_proof
            .verify_weight_aggregation(
                &get_iv(usize::MAX, global_iter),
                &client_schemes,
                client_commitments,
                constants.cm_w_single_iteration_dimensions,
            )
            .map_err(|e| FalafelError::ProofError(format!("{:?}", e)))?;
        let aggregation_verify_s = t_agg_verify.elapsed().as_secs_f64();

        let avg_prover = PhaseTiming::average(&all_prover_timings);
        let avg_verifier = PhaseTiming::average(&all_verifier_timings);

        let run = SingleRunResult {
            prover_timings: avg_prover,
            verifier_timings: avg_verifier,
            aggregation_prove_s,
            aggregation_verify_s,
            local_proof_size_bytes: local_proof_size,
            aggregation_proof_size_bytes: aggregation_proof_size,
        };

        eprintln!(
            "[batch_size={}] rep {}/{}: prove_total={:.2}s, verify_total={:.2}s",
            batch_size,
            rep + 1,
            repetitions,
            run.prover_timings.commitments_s
                + run.prover_timings.batch_correctness_s
                + run.prover_timings.forward_pass_s
                + run.prover_timings.backward_pass_s
                + run.prover_timings.weight_update_s
                + run.prover_timings.poly_evals_s
                + run.aggregation_prove_s,
            run.verifier_timings.commitments_s
                + run.verifier_timings.batch_correctness_s
                + run.verifier_timings.forward_pass_s
                + run.verifier_timings.backward_pass_s
                + run.verifier_timings.weight_update_s
                + run.verifier_timings.poly_evals_s
                + run.aggregation_verify_s,
        );

        raw_runs.push(run);
    }

    let results = build_results(batch_size, &raw_runs);
    Ok(results)
}

fn median(values: &mut Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = values.len();
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}

fn build_results(batch_size: usize, runs: &[SingleRunResult]) -> Vec<BenchmarkResult> {
    let phases = [
        "commitments",
        "batch_correctness",
        "forward_pass",
        "backward_pass",
        "weight_update",
        "poly_evals",
        "aggregation",
    ];

    let proof_size = runs.last().unwrap().local_proof_size_bytes;
    let agg_proof_size = runs.last().unwrap().aggregation_proof_size_bytes;

    phases
        .iter()
        .map(|&phase| {
            let mut prove_times: Vec<f64> = runs
                .iter()
                .map(|r| match phase {
                    "commitments" => r.prover_timings.commitments_s,
                    "batch_correctness" => r.prover_timings.batch_correctness_s,
                    "forward_pass" => r.prover_timings.forward_pass_s,
                    "backward_pass" => r.prover_timings.backward_pass_s,
                    "weight_update" => r.prover_timings.weight_update_s,
                    "poly_evals" => r.prover_timings.poly_evals_s,
                    "aggregation" => r.aggregation_prove_s,
                    _ => unreachable!(),
                })
                .collect();
            let mut verify_times: Vec<f64> = runs
                .iter()
                .map(|r| match phase {
                    "commitments" => r.verifier_timings.commitments_s,
                    "batch_correctness" => r.verifier_timings.batch_correctness_s,
                    "forward_pass" => r.verifier_timings.forward_pass_s,
                    "backward_pass" => r.verifier_timings.backward_pass_s,
                    "weight_update" => r.verifier_timings.weight_update_s,
                    "poly_evals" => r.verifier_timings.poly_evals_s,
                    "aggregation" => r.aggregation_verify_s,
                    _ => unreachable!(),
                })
                .collect();
            BenchmarkResult {
                batch_size,
                phase: phase.to_string(),
                prover_time_s: median(&mut prove_times),
                verifier_time_s: median(&mut verify_times),
                proof_size_bytes: if phase == "aggregation" {
                    agg_proof_size
                } else {
                    proof_size
                },
            }
        })
        .collect()
}
