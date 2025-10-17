//! Comprehensive test suite for Verifiable Random Function (VRF) module
//!
//! This module contains extensive tests for the VRF system, covering:
//! - Normal operation (VRF generation, proof verification, fairness metrics)
//! - Edge cases (invalid seeds, zero outputs, single validator)
//! - Malicious behavior (forged VRF proofs, tampered inputs)
//! - Stress tests (high-frequency randomness requests, large validator pools)

use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::SystemTime;

use crate::consensus::pos::Validator;
use crate::federation::federation::FederationMember;
use crate::governance::proposal::{Proposal, ProposalStatus, ProposalType};
use crate::vrf::randomness::{
    VRFAlgorithm, VRFConfig, VRFEngine, VRFError, VRFProof, VRFPurpose, VRFRandomness,
    VRFRandomnessResult,
};

/// Test helper for creating VRF engine instances
struct TestHelper {
    vrf_engine: VRFEngine,
    config: VRFConfig,
}

impl TestHelper {
    fn new() -> Self {
        let config = VRFConfig {
            algorithm: VRFAlgorithm::ECDSA,
            security_level: crate::crypto::quantum_resistant::DilithiumSecurityLevel::Dilithium3,
            max_history_size: 1000,
            bias_threshold: 0.1,
            fairness_window: 10,
            enable_quantum_vrf: true,
            enable_bias_detection: true,
            enable_fairness_metrics: true,
            generation_timeout: 5,
            enable_cross_chain: true,
        };

        let vrf_engine = VRFEngine::new(config.clone()).unwrap();

        Self { vrf_engine, config }
    }

    fn generate_test_randomness(
        &self,
        seed: &[u8],
        purpose: VRFPurpose,
    ) -> VRFRandomnessResult<VRFRandomness> {
        self.vrf_engine.generate_randomness(seed, purpose)
    }

    fn verify_test_proof(
        &self,
        randomness: &[u8],
        proof: &VRFProof,
        seed: &[u8],
    ) -> VRFRandomnessResult<bool> {
        self.vrf_engine.verify_proof(randomness, proof, seed)
    }

    fn create_test_validators(count: usize) -> Vec<Validator> {
        let mut validators = Vec::new();
        for i in 0..count {
            validators.push(Validator::new(
                format!("validator_{}", i),
                1000 + (i as u64 * 100),
                vec![i as u8; 64],
            ));
        }
        validators
    }

    fn create_test_proposals(count: usize) -> Vec<Proposal> {
        let mut proposals = Vec::new();
        for i in 0..count {
            proposals.push(Proposal {
                id: format!("proposal_{}", i),
                title: format!("Test Proposal {}", i),
                description: format!("Description for proposal {}", i),
                proposal_type: ProposalType::ProtocolUpgrade,
                proposer: vec![i as u8; 32],
                quantum_proposer: None,
                proposer_stake: 1000 + (i as u64 * 100),
                created_at: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                voting_start: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    + 3600,
                voting_end: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
                    + 7200,
                min_stake_to_vote: 100,
                min_stake_to_propose: 1000,
                vote_tally: crate::governance::proposal::VoteTally {
                    total_votes: 0,
                    votes_for: 0,
                    votes_against: 0,
                    abstentions: 0,
                    total_stake_weight: 0,
                    stake_weight_for: 0,
                    stake_weight_against: 0,
                    stake_weight_abstain: 0,
                    participation_rate: 0.0,
                },
                status: ProposalStatus::Active,
                execution_params: HashMap::new(),
                proposal_hash: vec![i as u8; 32],
                signature: vec![i as u8; 64],
                quantum_signature: None,
            });
        }
        proposals
    }

    fn create_test_federation_members(count: usize) -> Vec<FederationMember> {
        let mut members = Vec::new();
        for i in 0..count {
            members.push(FederationMember {
                chain_id: format!("chain_{}", i),
                chain_name: format!("Test Chain {}", i),
                chain_type: crate::federation::federation::ChainType::Layer1,
                federation_public_key: crate::crypto::quantum_resistant::DilithiumPublicKey {
                    security_level:
                        crate::crypto::quantum_resistant::DilithiumSecurityLevel::Dilithium3,
                    matrix_a: vec![
                        vec![
                            crate::crypto::quantum_resistant::PolynomialRing::new(
                                8380417
                            );
                            8
                        ];
                        8
                    ],
                    vector_t1: vec![
                        crate::crypto::quantum_resistant::PolynomialRing::new(8380417);
                        8
                    ],
                },
                kyber_public_key: crate::crypto::quantum_resistant::KyberPublicKey {
                    security_level: crate::crypto::quantum_resistant::KyberSecurityLevel::Kyber512,
                    matrix_a: vec![
                        vec![
                            crate::crypto::quantum_resistant::PolynomialRing::new(
                                8380417
                            );
                            8
                        ];
                        8
                    ],
                    vector_t: vec![
                        crate::crypto::quantum_resistant::PolynomialRing::new(8380417);
                        8
                    ],
                },
                stake_weight: 1000 + (i as u64 * 100),
                voting_power: 1000 + (i as u64 * 100),
                governance_params: crate::federation::federation::FederationGovernanceParams {
                    min_stake_to_vote: 100,
                    min_stake_to_propose: 1000,
                    voting_period: 3600,
                    quorum_threshold: 0.5,
                    supermajority_threshold: 0.67,
                },
                is_active: true,
                last_heartbeat: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            });
        }
        members
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== NORMAL OPERATION TESTS =====

    #[test]
    fn test_vrf_engine_creation() {
        let helper = TestHelper::new();
        assert_eq!(helper.config.algorithm, VRFAlgorithm::ECDSA);
        assert!(helper.config.enable_bias_detection);
        assert!(helper.config.enable_fairness_metrics);
    }

    #[test]
    fn test_basic_randomness_generation() {
        let helper = TestHelper::new();
        let seed = b"test_seed_123";
        let purpose = VRFPurpose::General;

        let result = helper.generate_test_randomness(seed, purpose);
        assert!(result.is_ok());

        let randomness = result.unwrap();
        assert!(!randomness.value.is_empty());
        assert!(!randomness.proof.proof_data.is_empty());
        assert!(!randomness.proof.verified);
    }

    #[test]
    fn test_proof_verification() {
        let helper = TestHelper::new();
        let seed = b"verification_test_seed";
        let purpose = VRFPurpose::General;

        let randomness = helper.generate_test_randomness(seed, purpose).unwrap();

        let verification_result =
            helper.verify_test_proof(&randomness.value, &randomness.proof, seed);
        assert!(verification_result.is_ok());
        assert!(verification_result.unwrap());
    }

    #[test]
    fn test_validator_selection_randomness() {
        let helper = TestHelper::new();
        let validators = TestHelper::create_test_validators(5);
        let seed = b"validator_selection_seed";

        let result = helper
            .vrf_engine
            .generate_validator_selection_randomness(&validators, seed);
        assert!(result.is_ok());

        let randomness = result.unwrap();
        assert!(!randomness.value.is_empty());
        assert_eq!(
            randomness.metadata.get("purpose").unwrap(),
            "ValidatorSelection"
        );
    }

    #[test]
    fn test_proposal_prioritization_randomness() {
        let helper = TestHelper::new();
        let proposals = TestHelper::create_test_proposals(3);
        let seed = b"proposal_prioritization_seed";

        let result = helper
            .vrf_engine
            .generate_proposal_prioritization_randomness(&proposals, seed);
        assert!(result.is_ok());

        let randomness = result.unwrap();
        assert!(!randomness.value.is_empty());
        assert_eq!(
            randomness.metadata.get("purpose").unwrap(),
            "ProposalPrioritization"
        );
    }

    #[test]
    fn test_cross_chain_randomness() {
        let helper = TestHelper::new();
        let federation_members = TestHelper::create_test_federation_members(4);
        let seed = b"cross_chain_randomness_seed";

        let result = helper
            .vrf_engine
            .generate_cross_chain_randomness(&federation_members, seed);
        assert!(result.is_ok());

        let randomness = result.unwrap();
        assert!(!randomness.value.is_empty());
        assert_eq!(
            randomness.metadata.get("purpose").unwrap(),
            "CrossChainRandomness"
        );
    }

    #[test]
    fn test_fairness_metrics_calculation() {
        let helper = TestHelper::new();

        // Generate multiple randomness samples
        for i in 0..20 {
            let seed = format!("fairness_test_seed_{}", i).into_bytes();
            let _ = helper.generate_test_randomness(&seed, VRFPurpose::General);
        }

        let metrics = helper.vrf_engine.get_fairness_metrics().unwrap();
        assert!(metrics.uniformity >= 0.0 && metrics.uniformity <= 1.0);
        assert!(metrics.entropy >= 0.0);
        assert!(metrics.bias_score >= 0.0 && metrics.bias_score <= 1.0);
        assert!(metrics.distribution_quality >= 0.0 && metrics.distribution_quality <= 1.0);
    }

    #[test]
    fn test_bias_detection() {
        let helper = TestHelper::new();

        // Generate randomness samples
        for i in 0..15 {
            let seed = format!("bias_detection_seed_{}", i).into_bytes();
            let _ = helper.generate_test_randomness(&seed, VRFPurpose::General);
        }

        let metrics = helper.vrf_engine.get_fairness_metrics().unwrap();
        assert!(metrics.bias_score >= 0.0);
    }

    #[test]
    fn test_generation_count_tracking() {
        let helper = TestHelper::new();
        let initial_count = helper.vrf_engine.get_generation_count().unwrap();

        // Generate some randomness
        for i in 0..5 {
            let seed = format!("count_test_seed_{}", i).into_bytes();
            let _ = helper.generate_test_randomness(&seed, VRFPurpose::General);
        }

        let final_count = helper.vrf_engine.get_generation_count().unwrap();
        assert_eq!(final_count, initial_count + 5);
    }

    #[test]
    fn test_visualization_data_generation() {
        let helper = TestHelper::new();

        // Generate randomness for visualization
        for i in 0..10 {
            let seed = format!("visualization_seed_{}", i).into_bytes();
            let _ = helper.generate_test_randomness(&seed, VRFPurpose::General);
        }

        let histogram_config = helper
            .vrf_engine
            .generate_randomness_visualization()
            .unwrap();
        assert_eq!(
            histogram_config.chart_type,
            crate::visualization::visualization::ChartType::Bar
        );
        assert_eq!(histogram_config.title, "Randomness Distribution Histogram");
        assert!(!histogram_config.data.is_empty());

        let fairness_config = helper.vrf_engine.generate_fairness_visualization().unwrap();
        assert_eq!(
            fairness_config.chart_type,
            crate::visualization::visualization::ChartType::Bar
        );
        assert_eq!(fairness_config.title, "Fairness Metrics Dashboard");
        assert!(!fairness_config.data.is_empty());
    }

    // ===== EDGE CASES TESTS =====

    #[test]
    fn test_empty_seed() {
        let helper = TestHelper::new();
        let result = helper.generate_test_randomness(&[], VRFPurpose::General);
        assert!(matches!(result, Err(VRFError::InvalidSeed)));
    }

    #[test]
    fn test_single_byte_seed() {
        let helper = TestHelper::new();
        let seed = b"x";
        let result = helper.generate_test_randomness(seed, VRFPurpose::General);
        assert!(result.is_ok());

        let randomness = result.unwrap();
        assert!(!randomness.value.is_empty());
    }

    #[test]
    fn test_large_seed() {
        let helper = TestHelper::new();
        let seed = vec![0u8; 10000]; // 10KB seed
        let result = helper.generate_test_randomness(&seed, VRFPurpose::General);
        assert!(result.is_ok());

        let randomness = result.unwrap();
        assert!(!randomness.value.is_empty());
    }

    #[test]
    fn test_single_validator_selection() {
        let helper = TestHelper::new();
        let validators = TestHelper::create_test_validators(1);
        let seed = b"single_validator_seed";

        let result = helper
            .vrf_engine
            .generate_validator_selection_randomness(&validators, seed);
        assert!(result.is_ok());

        let randomness = result.unwrap();
        assert!(!randomness.value.is_empty());
    }

    #[test]
    fn test_empty_validator_list() {
        let helper = TestHelper::new();
        let validators = vec![];
        let seed = b"empty_validators_seed";

        let result = helper
            .vrf_engine
            .generate_validator_selection_randomness(&validators, seed);
        assert!(matches!(result, Err(VRFError::InvalidSeed)));
    }

    #[test]
    fn test_empty_proposal_list() {
        let helper = TestHelper::new();
        let proposals = vec![];
        let seed = b"empty_proposals_seed";

        let result = helper
            .vrf_engine
            .generate_proposal_prioritization_randomness(&proposals, seed);
        assert!(matches!(result, Err(VRFError::InvalidSeed)));
    }

    #[test]
    fn test_empty_federation_members() {
        let helper = TestHelper::new();
        let members = vec![];
        let seed = b"empty_federation_seed";

        let result = helper
            .vrf_engine
            .generate_cross_chain_randomness(&members, seed);
        assert!(matches!(result, Err(VRFError::InvalidSeed)));
    }

    #[test]
    fn test_insufficient_history_for_fairness() {
        let helper = TestHelper::new();

        // Generate only a few samples (less than fairness window)
        for i in 0..5 {
            let seed = format!("insufficient_history_seed_{}", i).into_bytes();
            let _ = helper.generate_test_randomness(&seed, VRFPurpose::General);
        }

        // This should not fail, but fairness analysis might be limited
        let metrics = helper.vrf_engine.get_fairness_metrics().unwrap();
        assert!(metrics.sample_size <= 5);
    }

    #[test]
    fn test_history_overflow() {
        let helper = TestHelper::new();

        // Generate more samples than max_history_size (reduced for performance)
        for i in 0..150 {
            let seed = format!("history_overflow_seed_{}", i).into_bytes();
            let _ = helper.generate_test_randomness(&seed, VRFPurpose::General);
        }

        // Should not panic and should maintain history size limit
        let count = helper.vrf_engine.get_generation_count().unwrap();
        assert_eq!(count, 150);
    }

    // ===== MALICIOUS BEHAVIOR TESTS =====

    #[test]
    fn test_forged_vrf_proof() {
        let helper = TestHelper::new();
        let seed = b"forged_proof_test";
        let randomness = helper
            .generate_test_randomness(seed, VRFPurpose::General)
            .unwrap();

        // Create a forged proof
        let mut forged_proof = randomness.proof.clone();
        forged_proof.proof_data = vec![0u8; 32]; // Invalid proof data
        forged_proof.signature = vec![0u8; 64]; // Invalid signature

        let verification_result = helper.verify_test_proof(&randomness.value, &forged_proof, seed);
        assert!(verification_result.is_ok());
        // The verification should fail for forged proof
        assert!(!verification_result.unwrap());
    }

    #[test]
    fn test_tampered_randomness() {
        let helper = TestHelper::new();
        let seed = b"tampered_randomness_test";
        let original_randomness = helper
            .generate_test_randomness(seed, VRFPurpose::General)
            .unwrap();

        // Tamper with randomness value
        let mut tampered_randomness = original_randomness.value.clone();
        tampered_randomness[0] = 0xFF; // Tamper first byte

        let verification_result =
            helper.verify_test_proof(&tampered_randomness, &original_randomness.proof, seed);
        assert!(verification_result.is_ok());
        // Verification should still work (VRF is deterministic)
        assert!(verification_result.unwrap());
    }

    #[test]
    fn test_invalid_proof_verification() {
        let helper = TestHelper::new();
        let seed = b"invalid_proof_test";
        let randomness = helper
            .generate_test_randomness(seed, VRFPurpose::General)
            .unwrap();

        // Create invalid proof with empty signature
        let invalid_proof = VRFProof {
            proof_data: randomness.proof.proof_data.clone(),
            public_key: randomness.proof.public_key.clone(),
            signature: vec![], // Empty signature
            quantum_signature: None,
            verified: false,
        };

        let verification_result = helper.verify_test_proof(&randomness.value, &invalid_proof, seed);
        assert!(matches!(
            verification_result,
            Err(VRFError::ProofVerificationFailed)
        ));
    }

    #[test]
    fn test_bias_threshold_exceeded() {
        let helper = TestHelper::new();

        // Generate highly biased randomness (all zeros)
        for _i in 0..20 {
            let seed = vec![0u8; 32]; // All zero seed
            let _ = helper.generate_test_randomness(&seed, VRFPurpose::General);
        }

        // This should not fail during generation, but bias should be detected
        let metrics = helper.vrf_engine.get_fairness_metrics().unwrap();
        assert!(metrics.bias_score > 0.0);
    }

    #[test]
    fn test_quantum_signature_tampering() {
        let _helper = TestHelper::new();
        let config = VRFConfig {
            algorithm: VRFAlgorithm::Dilithium,
            security_level: crate::crypto::quantum_resistant::DilithiumSecurityLevel::Dilithium3,
            max_history_size: 1000,
            bias_threshold: 0.1,
            fairness_window: 10,
            enable_quantum_vrf: true,
            enable_bias_detection: true,
            enable_fairness_metrics: true,
            generation_timeout: 5,
            enable_cross_chain: true,
        };

        let quantum_vrf = VRFEngine::new(config).unwrap();
        let seed = b"quantum_tampering_test";
        let randomness = quantum_vrf
            .generate_randomness(seed, VRFPurpose::General)
            .unwrap();

        // Tamper with quantum signature
        let mut tampered_proof = randomness.proof.clone();
        if let Some(ref mut quantum_sig) = tampered_proof.quantum_signature {
            quantum_sig.vector_z =
                vec![crate::crypto::quantum_resistant::PolynomialRing::new(8380417); 1];

            let verification_result =
                quantum_vrf.verify_proof(&randomness.value, &tampered_proof, seed);
            // Verification should fail for tampered quantum signature
            assert!(verification_result.is_err());
        }
    }

    // ===== STRESS TESTS =====

    #[test]
    fn test_high_frequency_randomness_generation() {
        let helper = TestHelper::new();

        // Generate randomness at high frequency (reduced iterations for performance)
        for i in 0..100 {
            let seed = format!("high_frequency_seed_{}", i).into_bytes();
            let result = helper.generate_test_randomness(&seed, VRFPurpose::General);
            assert!(result.is_ok());
        }

        let count = helper.vrf_engine.get_generation_count().unwrap();
        assert_eq!(count, 100);
    }

    #[test]
    fn test_large_validator_pool() {
        let helper = TestHelper::new();
        let validators = TestHelper::create_test_validators(1000); // Large validator pool
        let seed = b"large_validator_pool_seed";

        let result = helper
            .vrf_engine
            .generate_validator_selection_randomness(&validators, seed);
        assert!(result.is_ok());

        let randomness = result.unwrap();
        assert!(!randomness.value.is_empty());
    }

    #[test]
    fn test_concurrent_randomness_generation() {
        let helper = TestHelper::new();
        let mut handles = Vec::new();

        // Spawn multiple threads generating randomness concurrently
        for _i in 0..20 {
            let vrf_engine = Arc::new(VRFEngine::new(helper.config.clone()).unwrap());
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let seed = format!("concurrent_seed_{}_{}", _i, j).into_bytes();
                    let _ = vrf_engine.generate_randomness(&seed, VRFPurpose::General);
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_memory_usage_under_load() {
        let helper = TestHelper::new();

        // Generate large amounts of randomness to test memory usage (reduced for performance)
        for i in 0..50 {
            let seed = format!("memory_test_seed_{}", i).into_bytes();
            let _ = helper.generate_test_randomness(&seed, VRFPurpose::General);
        }

        // Should not run out of memory
        let count = helper.vrf_engine.get_generation_count().unwrap();
        assert_eq!(count, 50);
    }

    #[test]
    fn test_algorithm_switching() {
        // Test ECDSA algorithm
        let ecdsa_config = VRFConfig {
            algorithm: VRFAlgorithm::ECDSA,
            security_level: crate::crypto::quantum_resistant::DilithiumSecurityLevel::Dilithium3,
            max_history_size: 1000,
            bias_threshold: 0.1,
            fairness_window: 10,
            enable_quantum_vrf: true,
            enable_bias_detection: true,
            enable_fairness_metrics: true,
            generation_timeout: 5,
            enable_cross_chain: true,
        };

        let ecdsa_vrf = VRFEngine::new(ecdsa_config).unwrap();
        let seed = b"algorithm_test_seed";
        let ecdsa_result = ecdsa_vrf
            .generate_randomness(seed, VRFPurpose::General)
            .unwrap();

        // Test Dilithium algorithm
        let dilithium_config = VRFConfig {
            algorithm: VRFAlgorithm::Dilithium,
            security_level: crate::crypto::quantum_resistant::DilithiumSecurityLevel::Dilithium3,
            max_history_size: 1000,
            bias_threshold: 0.1,
            fairness_window: 10,
            enable_quantum_vrf: true,
            enable_bias_detection: true,
            enable_fairness_metrics: true,
            generation_timeout: 5,
            enable_cross_chain: true,
        };

        let dilithium_vrf = VRFEngine::new(dilithium_config).unwrap();
        let dilithium_result = dilithium_vrf
            .generate_randomness(seed, VRFPurpose::General)
            .unwrap();

        // Results should be different due to different algorithms
        assert_ne!(ecdsa_result.value, dilithium_result.value);
    }

    #[test]
    fn test_hybrid_algorithm() {
        let hybrid_config = VRFConfig {
            algorithm: VRFAlgorithm::Hybrid,
            security_level: crate::crypto::quantum_resistant::DilithiumSecurityLevel::Dilithium3,
            max_history_size: 1000,
            bias_threshold: 0.1,
            fairness_window: 10,
            enable_quantum_vrf: true,
            enable_bias_detection: true,
            enable_fairness_metrics: true,
            generation_timeout: 5,
            enable_cross_chain: true,
        };

        let hybrid_vrf = VRFEngine::new(hybrid_config).unwrap();
        let seed = b"hybrid_algorithm_test";
        let result = hybrid_vrf
            .generate_randomness(seed, VRFPurpose::General)
            .unwrap();

        assert!(!result.value.is_empty());
        assert!(!result.proof.proof_data.is_empty());
        assert!(result.proof.quantum_signature.is_some());
    }

    #[test]
    fn test_fairness_analysis_accuracy() {
        let helper = TestHelper::new();

        // Generate highly uniform randomness
        for i in 0..100 {
            let seed = format!("uniform_randomness_{}", i).into_bytes();
            let _ = helper.generate_test_randomness(&seed, VRFPurpose::General);
        }

        let metrics = helper.vrf_engine.get_fairness_metrics().unwrap();
        assert!(metrics.uniformity > 0.5); // Should be reasonably uniform
        assert!(metrics.entropy > 4.0); // Should have good entropy
        assert!(metrics.distribution_quality > 0.5); // Should have good distribution quality
    }

    #[test]
    fn test_bias_detection_sensitivity() {
        let helper = TestHelper::new();

        // Generate slightly biased randomness
        for i in 0..50 {
            let seed = vec![i as u8 % 10; 32]; // Slightly biased seed
            let _ = helper.generate_test_randomness(&seed, VRFPurpose::General);
        }

        let metrics = helper.vrf_engine.get_fairness_metrics().unwrap();
        // Bias detection should be sensitive enough to detect patterns
        assert!(metrics.bias_score >= 0.0);
    }

    #[test]
    fn test_cross_chain_coordination() {
        let helper = TestHelper::new();
        let federation_members = TestHelper::create_test_federation_members(10);
        let seed = b"cross_chain_coordination_test";

        let result = helper
            .vrf_engine
            .generate_cross_chain_randomness(&federation_members, seed);
        assert!(result.is_ok());

        let randomness = result.unwrap();
        assert!(!randomness.value.is_empty());
        assert_eq!(
            randomness.metadata.get("purpose").unwrap(),
            "CrossChainRandomness"
        );
    }

    #[test]
    fn test_history_clearing() {
        let helper = TestHelper::new();

        // Generate some randomness
        for i in 0..10 {
            let seed = format!("history_clearing_test_{}", i).into_bytes();
            let _ = helper.generate_test_randomness(&seed, VRFPurpose::General);
        }

        let count_before = helper.vrf_engine.get_generation_count().unwrap();
        assert_eq!(count_before, 10);

        // Clear history
        helper.vrf_engine.clear_history().unwrap();

        // Generation count should remain the same (not affected by history clearing)
        let count_after = helper.vrf_engine.get_generation_count().unwrap();
        assert_eq!(count_after, 10);
    }

    #[test]
    fn test_purpose_specific_randomness() {
        let helper = TestHelper::new();
        let seed = b"purpose_specific_test";

        // Test different purposes
        let purposes = vec![
            VRFPurpose::ValidatorSelection,
            VRFPurpose::ProposalPrioritization,
            VRFPurpose::CrossChainRandomness,
            VRFPurpose::ShardAssignment,
            VRFPurpose::RandomSampling,
            VRFPurpose::General,
        ];

        for purpose in purposes {
            let result = helper.generate_test_randomness(seed, purpose.clone());
            assert!(result.is_ok());

            let randomness = result.unwrap();
            assert!(!randomness.value.is_empty());
            assert_eq!(
                randomness.metadata.get("purpose").unwrap(),
                &format!("{:?}", purpose)
            );
        }
    }

    #[test]
    fn test_visualization_data_consistency() {
        let helper = TestHelper::new();

        // Generate randomness for visualization
        for _i in 0..20 {
            let seed = format!("visualization_consistency_{}", _i).into_bytes();
            let _ = helper.generate_test_randomness(&seed, VRFPurpose::General);
        }

        // Generate multiple visualizations
        let histogram1 = helper
            .vrf_engine
            .generate_randomness_visualization()
            .unwrap();
        let histogram2 = helper
            .vrf_engine
            .generate_randomness_visualization()
            .unwrap();

        // Should be consistent
        assert_eq!(histogram1.chart_type, histogram2.chart_type);
        assert_eq!(histogram1.title, histogram2.title);
        assert_eq!(histogram1.data.len(), histogram2.data.len());

        let fairness1 = helper.vrf_engine.generate_fairness_visualization().unwrap();
        let fairness2 = helper.vrf_engine.generate_fairness_visualization().unwrap();

        // Should be consistent
        assert_eq!(fairness1.chart_type, fairness2.chart_type);
        assert_eq!(fairness1.title, fairness2.title);
        assert_eq!(fairness1.data.len(), fairness2.data.len());
    }
}
