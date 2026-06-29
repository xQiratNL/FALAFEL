pub mod bench;
pub mod common;
pub mod mnist;
pub mod random;

//pub use mnist::{MnistConfig, run_falafel_on_mnist};
pub use bench::{BenchmarkResult, run_falafel_benchmark};
pub use random::{RandomExperimentConfig, run_falafel_on_random_data};
