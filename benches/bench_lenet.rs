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
    let batch_sizes = [4, 8, 16];
    // let batch_sizes = [4];
    // let n_input = 8 * 8;
    // let n_hidden = vec![8,8,8];
    //LeNet:
    // let n_input = 32 * 32;
    // let n_hidden = vec![72,72,10];
    // Comparable to LeNet:
    let n_input = 120;
    let n_hidden = vec![120, 120, 120];
    let n_output = 1;
    let decimal_bits = 3;
    let initial_integer_bits = 2;

    println!("batch_size,phase,prover_time_s,verifier_time_s,proof_size_bytes");

    for &bs in &batch_sizes {
        let total_bits = compute_total_bits(
            n_input,
            &n_hidden,
            n_output,
            decimal_bits,
            initial_integer_bits,
            bs,
        );
        let n_samples = bs;
        eprintln!(
            "=== Starting benchmark for batch_size={} with total_bits={} ===",
            bs, total_bits
        );
        let iterations = (n_samples / &bs) as f64;
        let config = RandomExperimentConfig {
            n_samples_vec: vec![n_samples; 3],
            n_input,
            n_hidden: n_hidden.clone(),
            activation_functions: vec![ActivationFunction::ReLU; n_hidden.len() + 1],
            n_output,
            n_global_iter: 1,
            n_local_iter: 1,
            batch_size: bs,
            learning_rate: 1i32,
            total_bits,
            decimal_bits,
            initial_integer_bits,
        };

        match run_falafel_benchmark::<i32, EdwardsProjective>(config) {
            Ok(results) => {
                for r in &results {
                    println!(
                        "{},{},{:.2},{:.2},{}",
                        r.batch_size,
                        r.phase,
                        r.prover_time_s,
                        r.verifier_time_s,
                        r.proof_size_bytes
                    );
                }
                eprintln!("--- batch_size={} results ---", bs);
                for r in &results {
                    eprintln!(
                        "  {:<20} prove={:>10.2}s  verify={:>10.2}s  size={:>8} bytes",
                        r.phase,
                        r.prover_time_s / iterations,
                        r.verifier_time_s / iterations,
                        r.proof_size_bytes
                    );
                }
            }
            Err(e) => {
                eprintln!("ERROR for batch_size={}: {:?}", bs, e);
            }
        }
    }
}
