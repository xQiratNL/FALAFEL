use std::time::Instant;

use crate::auditor::verify_dataset_signature;
use crate::client::ClientSchemes;
use crate::crypto_schemes::{HyraxCommitment, HyraxCommitmentOpening};
use crate::prelude::*;
use crate::proof::{LocalTrainingProof, LocalTrainingProofSeparator, ProofWitnessFull};
use rand::{CryptoRng, Rng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use spongefish::codecs::arkworks_algebra::GroupToUnitDeserialize;
use spongefish::{DefaultHash, DomainSeparator, ProofResult};

#[derive(Clone, Debug)]
pub struct PhaseTiming {
    pub commitments_s: f64,
    pub batch_correctness_s: f64,
    pub forward_pass_s: f64,
    pub backward_pass_s: f64,
    pub weight_update_s: f64,
    pub poly_evals_s: f64,
}

impl PhaseTiming {
    pub fn average(timings: &[PhaseTiming]) -> PhaseTiming {
        let n = timings.len() as f64;
        PhaseTiming {
            commitments_s: timings.iter().map(|t| t.commitments_s).sum::<f64>() / n,
            batch_correctness_s: timings.iter().map(|t| t.batch_correctness_s).sum::<f64>() / n,
            forward_pass_s: timings.iter().map(|t| t.forward_pass_s).sum::<f64>() / n,
            backward_pass_s: timings.iter().map(|t| t.backward_pass_s).sum::<f64>() / n,
            weight_update_s: timings.iter().map(|t| t.weight_update_s).sum::<f64>() / n,
            poly_evals_s: timings.iter().map(|t| t.poly_evals_s).sum::<f64>() / n,
        }
    }
}

pub struct ProveTimedResult<G: CurveGroup, T: Value> {
    pub proof: LocalTrainingProof<G, T>,
    pub opening: HyraxCommitmentOpening<G>,
    pub timings: PhaseTiming,
}

pub struct ClientTimedResult {
    pub serialized: (Vec<u8>, Vec<u8>, Vec<u8>),
    pub timings: PhaseTiming,
}

impl<'a, G: CurveGroup, T: Value> ProofWitnessFull<'a, G, T> {
    pub fn prove_timed(
        &self,
        iv: &str,
        dataset_signature: AuditorSignature<G>,
        client_schemes: &ClientSchemes<G>,
        rng: &mut (impl Rng + CryptoRng),
    ) -> ProofResult<ProveTimedResult<G, T>> {
        let domain_separator: DomainSeparator<DefaultHash> =
            LocalTrainingProofSeparator::<G, T>::new_local_training_proof_separator(
                iv,
                &self.constants,
            );
        let mut prover_state = domain_separator.to_prover_state();

        let t0 = Instant::now();
        let commitments = self.compute_commitments(client_schemes, rng);
        self.add_commitments(&mut prover_state, &commitments)?;
        let commitments_s = t0.elapsed().as_secs_f64();
        eprintln!("    [prove] commitments: {:.2}s", commitments_s);

        prover_state.ratchet()?;

        let t0 = Instant::now();
        self.prove_batch_correctness(&mut prover_state, client_schemes, &commitments)?;
        let batch_correctness_s = t0.elapsed().as_secs_f64();
        eprintln!("    [prove] batch_correctness: {:.2}s", batch_correctness_s);

        prover_state.ratchet()?;

        let t0 = Instant::now();
        let deferred_proofs_fp =
            self.prove_forward_pass(&mut prover_state, client_schemes, &commitments)?;
        let forward_pass_s = t0.elapsed().as_secs_f64();
        eprintln!("    [prove] forward_pass: {:.2}s", forward_pass_s);

        prover_state.ratchet()?;

        let t0 = Instant::now();
        let deferred_proofs_bp =
            self.prove_backward_pass(&mut prover_state, client_schemes, &commitments)?;
        let backward_pass_s = t0.elapsed().as_secs_f64();
        eprintln!("    [prove] backward_pass: {:.2}s", backward_pass_s);

        prover_state.ratchet()?;

        let t0 = Instant::now();
        self.prove_weight_update(&mut prover_state, client_schemes, &commitments)?;
        let weight_update_s = t0.elapsed().as_secs_f64();
        eprintln!("    [prove] weight_update: {:.2}s", weight_update_s);

        prover_state.ratchet()?;

        let t0 = Instant::now();
        self.prove_poly_evals(
            &mut prover_state,
            client_schemes,
            &commitments,
            deferred_proofs_fp,
            deferred_proofs_bp,
        )?;
        let poly_evals_s = t0.elapsed().as_secs_f64();
        eprintln!("    [prove] poly_evals: {:.2}s", poly_evals_s);

        Ok(ProveTimedResult {
            proof: LocalTrainingProof {
                dataset_signature,
                proof: prover_state.narg_string().to_vec(),
                constants: self.constants.clone(),
            },
            opening: HyraxCommitmentOpening {
                input_matrix: commitments
                    .w_final_input_matrix
                    .expect("The prover should have the witness!"),
                randomness: ark_std::cfg_into_iter!(
                    (0..self.opening_initial_weights.randomness.len())
                )
                .map(|idx| {
                    self.opening_initial_weights.randomness[idx]
                        - ark_std::cfg_iter!(
                            commitments
                                .opening_g_hat_all
                                .as_ref()
                                .expect("This should exist")
                        )
                        .map(|opening_g_hat| opening_g_hat.randomness[idx])
                        .sum::<G::ScalarField>()
                })
                .collect::<Vec<_>>(),
            },
            timings: PhaseTiming {
                commitments_s,
                batch_correctness_s,
                forward_pass_s,
                backward_pass_s,
                weight_update_s,
                poly_evals_s,
            },
        })
    }
}

impl<G: CurveGroup, T: Value> LocalTrainingProof<G, T> {
    pub fn verify_timed<'a>(
        &self,
        iv: &str,
        pp_signature: &AuditorSignatureParameters<G>,
        auditor_pk: &AuditorSignaturePublicKey<G>,
        cm_initial_weights: &HyraxCommitment<G>,
        client_schemes: &ClientSchemes<'a, G>,
    ) -> ProofResult<PhaseTiming> {
        let domain_separator: DomainSeparator<DefaultHash> =
            LocalTrainingProofSeparator::<G, T>::new_local_training_proof_separator(
                iv,
                &self.constants,
            );
        let mut verifier_state = domain_separator.to_verifier_state(&self.proof);

        let t0 = Instant::now();
        let [cm_dataset_samples, cm_dataset_labels]: [_; 2] =
            verifier_state.next_points()?.map(|cm: G| cm.into_affine());
        if !verify_dataset_signature(
            pp_signature,
            auditor_pk,
            &(cm_dataset_samples, cm_dataset_labels),
            &self.dataset_signature,
        )
        .unwrap_or(false)
        {
            return Err(spongefish::ProofError::InvalidProof);
        }
        let commitments = self.obtain_commitments(&mut verifier_state)?;
        let commitments_s = t0.elapsed().as_secs_f64();
        eprintln!("    [verify] commitments: {:.2}s", commitments_s);

        verifier_state.ratchet()?;

        let t0 = Instant::now();
        self.verify_batch_correctness(
            &mut verifier_state,
            &commitments,
            client_schemes,
            &cm_dataset_samples,
            &cm_dataset_labels,
        )?;
        let batch_correctness_s = t0.elapsed().as_secs_f64();
        eprintln!(
            "    [verify] batch_correctness: {:.2}s",
            batch_correctness_s
        );

        verifier_state.ratchet()?;

        let t0 = Instant::now();
        let deferred_proofs_fp =
            self.verify_forward_pass(&mut verifier_state, &commitments, client_schemes)?;
        let forward_pass_s = t0.elapsed().as_secs_f64();
        eprintln!("    [verify] forward_pass: {:.2}s", forward_pass_s);

        verifier_state.ratchet()?;

        let t0 = Instant::now();
        let deferred_proofs_bp =
            self.verify_backward_pass(&mut verifier_state, &commitments, client_schemes)?;
        let backward_pass_s = t0.elapsed().as_secs_f64();
        eprintln!("    [verify] backward_pass: {:.2}s", backward_pass_s);

        verifier_state.ratchet()?;

        let t0 = Instant::now();
        self.verify_weight_update(&mut verifier_state, &commitments, client_schemes)?;
        let weight_update_s = t0.elapsed().as_secs_f64();
        eprintln!("    [verify] weight_update: {:.2}s", weight_update_s);

        verifier_state.ratchet()?;

        let t0 = Instant::now();
        self.verify_poly_evals(
            &mut verifier_state,
            &commitments,
            &cm_initial_weights,
            client_schemes,
            deferred_proofs_fp,
            deferred_proofs_bp,
        )?;
        let poly_evals_s = t0.elapsed().as_secs_f64();
        eprintln!("    [verify] poly_evals: {:.2}s", poly_evals_s);

        Ok(PhaseTiming {
            commitments_s,
            batch_correctness_s,
            forward_pass_s,
            backward_pass_s,
            weight_update_s,
            poly_evals_s,
        })
    }
}
