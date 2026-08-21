# FALAFEL: Modular Zero-Knowledge Proofs of Training in the Federated Setting

This repository contains

- the proofs for the paper
- the code for the paper

We make our code available open-source under the MIT license.

## Contents

1[Running locally](#running-locally)

## Running Locally

First do the following:

- [Install Rust](https://www.rust-lang.org/tools/install)
- Clone this repository/Download this code

There are 3 experiments to run (besides the tests that are run the regular Rust way). The commands for each experiment is as follows:

1. Configuration study:
   `RUSTFLAGS="-C target-feature=+bmi2,+adx" cargo bench --features asm --bench bench_phases > results_batch_experiment.csv`
2. Scaling:
   `RUSTFLAGS="-C target-feature=+bmi2,+adx" cargo bench --features asm --bench bench_scaling > results_scaling_experiment.csv`
3. Real-world models (if you uncomment/comment different parameters in `benches/bench_scaling.rs` you can switch between lenet/alexnet/vgg-11)
   `RUSTFLAGS="-C target-feature=+bmi2,+adx" cargo bench --features asm --bench bench_lenet > results_lenet_experiment.csv`
