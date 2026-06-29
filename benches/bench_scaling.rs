use ark_curve25519::EdwardsProjective;
use falafel::experiments::RandomExperimentConfig;
use falafel::experiments::bench::run_falafel_benchmark;
use falafel::neural_network::ActivationFunction;

fn log_ceil(input: usize) -> usize {
    if input <= 1 {
        0
    } else {
        ((input - 1).ilog2() + 1) as usize
    }
}

fn compute_total_bits(
    n_input: usize,
    n_hidden: &[usize],
    n_output: usize,
    decimal_bits: usize,
    initial_integer_bits: usize,
    batch_size: usize,
) -> usize {
    let mut total_bits = 0;
    total_bits += log_ceil(n_input + 1); //Forward pass

    for &width in n_hidden {
        total_bits += log_ceil(width + 1); //Forward pass
        total_bits += log_ceil(width) + 1; //Backward pass
    }

    total_bits += log_ceil(n_output);
    total_bits += decimal_bits + 2 * initial_integer_bits;
    total_bits += 2;
    total_bits -= log_ceil(batch_size);

    total_bits += log_ceil(n_input);

    total_bits += 2; //Safety margin

    total_bits
}

fn main() {
    let decimal_bits = 3;
    let initial_integer_bits = 2;

    println!(
        "experiment,parameter_value,batch_size,iterations,phase,prover_time_s,verifier_time_s,proof_size_bytes"
    );

    // Experiment A: network width scaling
    for batch_size in [4, 8, 16] {
        for width in (2..=128).step_by(2) {
            let n_input = 4;
            let n_output = 4;
            let n_samples = 16;
            let n_layers = 3;
            let n_hidden = vec![width; n_layers];
            let iterations = (n_samples + batch_size - 1) / batch_size;
            let total_bits = compute_total_bits(
                n_input,
                &n_hidden,
                n_output,
                decimal_bits,
                initial_integer_bits,
                batch_size,
            );

            eprintln!(
                "=== Experiment A: width={}, batch_size={}, iterations={}, total_bits={} ===",
                width, batch_size, iterations, total_bits
            );

            let config = RandomExperimentConfig {
                n_samples_vec: vec![n_samples; 3],
                n_input,
                n_hidden,
                activation_functions: vec![ActivationFunction::ReLU; n_layers + 1],
                n_output,
                n_global_iter: 1,
                n_local_iter: 1,
                batch_size,
                learning_rate: 1i64,
                total_bits,
                decimal_bits,
                initial_integer_bits,
            };

            match run_falafel_benchmark::<i64, EdwardsProjective>(config) {
                Ok(results) => {
                    for r in &results {
                        println!(
                            "width,{},{},{},{},{:.6},{:.6},{}",
                            width,
                            batch_size,
                            iterations,
                            r.phase,
                            r.prover_time_s,
                            r.verifier_time_s,
                            r.proof_size_bytes
                        );
                    }
                    eprintln!("--- width={}, batch_size={} done ---", width, batch_size);
                    for r in &results {
                        eprintln!(
                            "  {:<20} prove={:>10.6}s  verify={:>10.6}s  size={:>8} bytes",
                            r.phase, r.prover_time_s, r.verifier_time_s, r.proof_size_bytes
                        );
                    }
                }
                Err(e) => eprintln!("ERROR width={}, batch_size={}: {:?}", width, batch_size, e),
            }
        }
    }

    // Experiment B: sample count scaling
    for batch_size in [4, 8, 16] {
        for n_samples in (2..=128).step_by(2) {
            if n_samples < batch_size {
                continue;
            }
            let n_input = 4;
            let n_output = 4;
            let n_layers = 3;
            let n_hidden = vec![16; n_layers];
            let iterations = (n_samples + batch_size - 1) / batch_size;
            let total_bits = compute_total_bits(
                n_input,
                &n_hidden,
                n_output,
                decimal_bits,
                initial_integer_bits,
                batch_size,
            );

            eprintln!(
                "=== Experiment B: n_samples={}, batch_size={}, iterations={}, total_bits={} ===",
                n_samples, batch_size, iterations, total_bits
            );

            let config = RandomExperimentConfig {
                n_samples_vec: vec![n_samples; 3],
                n_input,
                n_hidden,
                activation_functions: vec![ActivationFunction::ReLU; n_layers + 1],
                n_output,
                n_global_iter: 1,
                n_local_iter: 1,
                batch_size,
                learning_rate: 1i64,
                total_bits,
                decimal_bits,
                initial_integer_bits,
            };

            match run_falafel_benchmark::<i64, EdwardsProjective>(config) {
                Ok(results) => {
                    for r in &results {
                        println!(
                            "samples,{},{},{},{},{:.6},{:.6},{}",
                            n_samples,
                            batch_size,
                            iterations,
                            r.phase,
                            r.prover_time_s,
                            r.verifier_time_s,
                            r.proof_size_bytes
                        );
                    }
                    eprintln!(
                        "--- n_samples={}, batch_size={} done ---",
                        n_samples, batch_size
                    );
                    for r in &results {
                        eprintln!(
                            "  {:<20} prove={:>10.6}s  verify={:>10.6}s  size={:>8} bytes",
                            r.phase, r.prover_time_s, r.verifier_time_s, r.proof_size_bytes
                        );
                    }
                }
                Err(e) => eprintln!(
                    "ERROR n_samples={}, batch_size={}: {:?}",
                    n_samples, batch_size, e
                ),
            }
        }
    }
}
