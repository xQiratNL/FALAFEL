//! Common utilities for all federated learning experiments.

use crate::auditor::Auditor;
use crate::client::Client;
use crate::prelude::*;
use crate::proof::HyraxCmMatricesDimensions;
use ark_std::log2;
use rand_chacha::ChaChaRng;
use std::cmp::max;

/// Compute the maximum number of Pedersen generators needed.
///
pub fn compute_max_pedersen_gens(
    n_samples: usize,
    n_local_iter: usize,
    mut batch_size: usize,
    n_input: usize,
    n_hidden: &[usize],
    n_output: usize,
    total_bits: usize,
) -> usize {
    // Compute the maximum number of Pedersen generators needed.
    // based on the constants computation in src/proof/mod.rs
    let n_batches = (n_samples - 1) / batch_size + 1;
    batch_size = if n_batches == 1 {
        n_samples
    } else {
        batch_size
    };
    let n_epochs = n_local_iter * n_batches;
    let n_matrices_per_layer = 3 + n_hidden.len();
    let n_matrices = n_epochs * n_matrices_per_layer;
    let w_rows = std::iter::once(n_input + 1)
        .chain(n_hidden.iter().map(|layer_weight| layer_weight + 1))
        .chain(std::iter::once(n_output))
        .max()
        .unwrap();
    let w_cols = std::iter::once(1)
        .chain(std::iter::once(n_input + 1))
        .chain(n_hidden.iter().map(|layer_weight| layer_weight + 1))
        .chain(std::iter::once(n_output))
        .max()
        .unwrap();
    let t_rows = w_rows;
    let t_cols = batch_size;

    let g_single_iteration_dims =
        HyraxCmMatricesDimensions::new(n_matrices_per_layer, w_rows, w_cols, true, None);

    let g_dims = HyraxCmMatricesDimensions::new(
        n_matrices,
        w_rows,
        w_cols,
        false,
        Some(log2(g_single_iteration_dims.row_com_dimension) as usize),
    );
    let t_dims = HyraxCmMatricesDimensions::new(n_matrices, t_rows, t_cols, false, None);

    let t_and_r_scaling_generators =
        2 * (t_dims.num_row_coms * t_dims.row_com_dimension * total_bits).next_power_of_two();
    let g_scaling_generators =
        2 * (g_dims.num_row_coms * g_dims.row_com_dimension * total_bits).next_power_of_two();
    max(t_and_r_scaling_generators, g_scaling_generators)
}

/// Audit each client's dataset and store the signature.
///
/// This function:
/// 1. Commits each client's dataset
/// 2. Gets the dataset commitment
/// 3. Has the auditor sign it
/// 4. Stores the signature with the client
pub fn audit_clients<T: Value, G: CurveGroup>(
    clients: &mut [Client<T, G>],
    auditor: &Auditor<G>,
    crypto_rng: &mut ChaChaRng,
) -> FalafelResult<()> {
    for client in clients {
        client.commit_dataset(crypto_rng);
        let cm_dataset = client.get_dataset_cm()?;
        let signature = auditor.sign_dataset(&cm_dataset, crypto_rng)?;
        client.store_dataset_signature(signature);
    }
    Ok(())
}
