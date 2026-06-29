use crate::client::ClientSchemes;
use crate::prelude::*;
use crate::proof::schnorr::{SchnorrProof, SchnorrSeparator};
use crate::proof::{AggregationProof, HyraxCmMatricesDimensions};
use spongefish::codecs::arkworks_algebra::{
    FieldDomainSeparator, GroupDomainSeparator, GroupToUnitDeserialize, GroupToUnitSerialize,
};
use spongefish::{DefaultHash, DomainSeparator, DuplexSpongeInterface, ProofError, ProofResult};

pub trait AggregationSeparator<G: CurveGroup> {
    fn add_aggregation_proof(
        self,
        commitments: &Vec<Vec<G::Affine>>,
        dimensions: &HyraxCmMatricesDimensions,
    ) -> Self;
    fn new_aggregation_proof_separator(
        iv: &str,
        commitments: &Vec<Vec<G::Affine>>,
        dimensions: &HyraxCmMatricesDimensions,
    ) -> Self;
}

impl<G: CurveGroup, H: DuplexSpongeInterface> AggregationSeparator<G> for DomainSeparator<H>
where
    Self: GroupDomainSeparator<G> + FieldDomainSeparator<G::ScalarField>,
{
    fn add_aggregation_proof(
        self,
        commitments: &Vec<Vec<G::Affine>>,
        dimensions: &HyraxCmMatricesDimensions,
    ) -> Self {
        self.add_points(
            commitments.clone().iter().map(|v| v.len()).sum(),
            "client commitments",
        )
        .add_points(
            dimensions.num_row_coms,
            "aggregated commitment, fresh randomized",
        )
    }

    fn new_aggregation_proof_separator(
        iv: &str,
        commitments: &Vec<Vec<G::Affine>>,
        dimensions: &HyraxCmMatricesDimensions,
    ) -> Self {
        Self::new(iv)
            .add_aggregation_proof(commitments, dimensions)
            .ratchet()
            .add_schnorr_proof(dimensions.num_row_coms)
    }
}

impl AggregationProof {
    //This proof should receive the final commitments of different parties, aggregate them and multiply it with a fresh commitment to 0.
    pub fn prove_weight_aggregation<'a, G: CurveGroup>(
        iv: &str,
        client_schemes: &ClientSchemes<'a, G>,
        client_weight_commitments: Vec<Vec<G::Affine>>,
        client_commitment_witnesses: Vec<Vec<G::ScalarField>>,
        zero_commitments_randomizers: (Vec<G::Affine>, Vec<G::ScalarField>),
        dimensions: &HyraxCmMatricesDimensions,
    ) -> ProofResult<(AggregationProof, Vec<G::ScalarField>)> {
        let zero_commitments = zero_commitments_randomizers.0;
        let zero_commitment_randomizers = zero_commitments_randomizers.1;

        //Construct the domain separator for the aggregation proof
        let domain_separator: DomainSeparator<DefaultHash> =
            AggregationSeparator::<G>::new_aggregation_proof_separator(
                iv,
                &client_weight_commitments,
                dimensions,
            );

        let mut prover_state = domain_separator.to_prover_state();

        //Add the commitments of the clients to the domain separator
        let client_points: Vec<G::Affine> = client_weight_commitments
            .iter()
            .cloned()
            .flatten()
            .collect();
        let client_points_group: Vec<G> =
            client_points.into_iter().map(|x| x.into_group()).collect();

        prover_state.add_points(&client_points_group)?;

        //Aggregate the commitments of the clients in the same way that the server aggregates to networks themselves.
        let mut x = client_weight_commitments.iter();
        let mut aggregated_commitment = x.next().unwrap().clone();
        for row in x {
            for (sum_elem, elem) in aggregated_commitment.iter_mut().zip(row) {
                *sum_elem = (*sum_elem + elem).into_affine();
            }
        }

        let n_clients_scalar = G::ScalarField::from(client_weight_commitments.clone().len() as i32)
            .inverse()
            .unwrap();
        aggregated_commitment = aggregated_commitment
            .into_iter()
            .map(|x| (x * n_clients_scalar).into_affine())
            .collect();

        //Perform the same aggregation on the randomness that the clients used to construct the corresponding commitments
        let mut randomness_iter = client_commitment_witnesses.iter();
        let mut aggregated_randomness = randomness_iter.next().unwrap().clone();
        for row in randomness_iter {
            for (sum_elem, elem) in aggregated_randomness.iter_mut().zip(row) {
                *sum_elem = *sum_elem + elem;
            }
        }

        aggregated_randomness = aggregated_randomness
            .into_iter()
            .map(|x| x * n_clients_scalar)
            .collect();

        //Refresh the commitments so that we can safely share the opening with the clients for the next iteration.
        let fresh_aggregated_commitment: Vec<G::Affine> = aggregated_commitment
            .clone()
            .into_iter()
            .zip(zero_commitments.iter())
            .map(|(x, y)| (x + *y).into_affine())
            .collect();

        let fresh_points_group: Vec<G> = fresh_aggregated_commitment
            .clone()
            .into_iter()
            .map(|x| x.into_group())
            .collect();
        prover_state.add_points(&fresh_points_group)?;

        let fresh_aggregated_randomizer: Vec<G::ScalarField> = aggregated_randomness
            .clone()
            .into_iter()
            .zip(zero_commitment_randomizers.iter())
            .map(|(x, y)| x + *y)
            .collect();

        let cm_x = fresh_aggregated_commitment
            .into_iter()
            .zip(aggregated_commitment.clone().iter())
            .map(|(x, y)| (x - y).into_affine())
            .collect::<Vec<_>>();

        let r_x = fresh_aggregated_randomizer
            .into_iter()
            .zip(aggregated_randomness.iter())
            .map(|(x, y)| x - y)
            .collect::<Vec<_>>();
        prover_state.ratchet()?;
        //proof-of-equality between fresh_aggregated_commitment and aggregation of individual commitments
        let witness = SchnorrProof {
            cm_x: &cm_x,
            r_x: Some(&r_x),
        };

        witness.prove(&mut prover_state, client_schemes)?;

        Ok((
            AggregationProof {
                proof: prover_state.narg_string().to_vec(),
            },
            r_x,
        ))
    }

    pub fn verify_weight_aggregation<'a, G: CurveGroup>(
        &self,
        iv: &str,
        client_schemes: &ClientSchemes<'a, G>,
        client_commitments: Vec<Vec<G::Affine>>,
        dimensions: HyraxCmMatricesDimensions,
    ) -> ProofResult<Vec<G::Affine>> {
        let domain_separator: DomainSeparator<DefaultHash> =
            AggregationSeparator::<G>::new_aggregation_proof_separator(
                iv,
                &client_commitments,
                &dimensions,
            );

        let mut verifier_state = domain_separator.to_verifier_state(&self.proof);

        let mut x = client_commitments.iter();
        let mut aggregated_commitment = x.next().unwrap().clone();
        for row in x {
            for (sum_elem, elem) in aggregated_commitment.iter_mut().zip(row) {
                *sum_elem = (*sum_elem + elem).into_affine();
            }
        }
        let n_clients_scalar = G::ScalarField::from(client_commitments.clone().len() as i32)
            .inverse()
            .unwrap();
        aggregated_commitment = aggregated_commitment
            .into_iter()
            .map(|x| (x * n_clients_scalar).into_affine())
            .collect();

        let client_points: Vec<G::Affine> = client_commitments.iter().cloned().flatten().collect();

        let mut client_points_server = vec![G::ZERO; client_points.len()];
        verifier_state.fill_next_points(&mut client_points_server)?;
        let client_points_server_affine: Vec<G::Affine> = client_points_server
            .into_iter()
            .map(|x| x.into_affine())
            .collect();

        if client_points_server_affine != client_points {
            return Err(ProofError::InvalidProof);
        }

        let mut fresh_commitment_server = vec![G::ZERO; dimensions.num_row_coms];
        verifier_state.fill_next_points(&mut fresh_commitment_server)?;

        let cm_x = fresh_commitment_server
            .into_iter()
            .zip(aggregated_commitment.clone().iter())
            .map(|(x, y)| (x - y).into_affine())
            .collect::<Vec<_>>();

        verifier_state.ratchet()?;

        SchnorrProof::<G> {
            cm_x: &cm_x,
            r_x: None,
        }
        .verify(&mut verifier_state, client_schemes, dimensions.num_row_coms)?;

        Ok(cm_x)
    }
}
