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

    struct Config {
        label: &'static str,
        n_input: usize,
        n_hidden: Vec<usize>,
        n_output: usize,
        n_samples: usize,
        batch_size: usize,
    }

    let configs = vec![
        Config {
            label: "base",
            n_input: 8,
            n_hidden: vec![8; 3],
            n_output: 8,
            n_samples: 32,
            batch_size: 16,
        },
        Config {
            label: "large_dataset",
            n_input: 8,
            n_hidden: vec![8; 3],
            n_output: 8,
            n_samples: 128,
            batch_size: 64,
        },
        Config {
            label: "many_iters",
            n_input: 8,
            n_hidden: vec![8; 3],
            n_output: 8,
            n_samples: 32,
            batch_size: 2,
        },
        Config {
            label: "wide_network",
            n_input: 32,
            n_hidden: vec![32; 3],
            n_output: 32,
            n_samples: 32,
            batch_size: 16,
        },
        Config {
            label: "deep_network",
            n_input: 8,
            n_hidden: vec![8; 8],
            n_output: 8,
            n_samples: 32,
            batch_size: 16,
        },
    ];

    println!("config,iterations,phase,prover_time_s,verifier_time_s,proof_size_bytes");

    for c in &configs {
        let total_bits = compute_total_bits(
            c.n_input,
            &c.n_hidden,
            c.n_output,
            decimal_bits,
            initial_integer_bits,
            c.batch_size,
        );
        let iterations = (c.n_samples + c.batch_size - 1) / c.batch_size;
        let n_layers = c.n_hidden.len();

        eprintln!(
            "=== Config '{}': n_input={}, n_hidden={:?}, n_output={}, n_samples={}, batch_size={}, iterations={}, total_bits={} ===",
            c.label,
            c.n_input,
            c.n_hidden,
            c.n_output,
            c.n_samples,
            c.batch_size,
            iterations,
            total_bits
        );

        let config = RandomExperimentConfig {
            n_samples_vec: vec![c.n_samples; 3],
            n_input: c.n_input,
            n_hidden: c.n_hidden.clone(),
            activation_functions: vec![ActivationFunction::ReLU; n_layers + 1],
            n_output: c.n_output,
            n_global_iter: 1,
            n_local_iter: 1,
            batch_size: c.batch_size,
            learning_rate: 1i64,
            total_bits,
            decimal_bits,
            initial_integer_bits,
        };

        match run_falafel_benchmark::<i64, EdwardsProjective>(config) {
            Ok(results) => {
                for r in &results {
                    println!(
                        "{},{},{},{:.6},{:.6},{}",
                        c.label,
                        iterations,
                        r.phase,
                        r.prover_time_s,
                        r.verifier_time_s,
                        r.proof_size_bytes
                    );
                }
                eprintln!("--- '{}' done ---", c.label);
                for r in &results {
                    eprintln!(
                        "  {:<20} prove={:>10.6}s  verify={:>10.6}s  size={:>8} bytes",
                        r.phase, r.prover_time_s, r.verifier_time_s, r.proof_size_bytes
                    );
                }
            }
            Err(e) => eprintln!("ERROR '{}': {:?}", c.label, e),
        }
    }
}
