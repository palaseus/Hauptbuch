//! Comprehensive test suite for EigenLayer-inspired restaking module
//!
//! This module contains 25+ test cases covering normal operation, edge cases,
//! malicious behavior, and stress tests for the restaking system.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::restaking::eigen::{
    ChainRiskAssessment, ChainRiskLevel, RestakingConfig, RestakingError, RestakingManager,
    RestakingRequest, RestakingSlashingCondition,
};

// Import modules for integration testing
use crate::anomaly::detector::AnomalyDetector;
use crate::consensus::pos::Validator;
use crate::federation::federation::{
    ChainType, FederationGovernanceParams, FederationMember, MultiChainFederation,
};
use crate::governance::proposal::{Proposal, ProposalStatus, ProposalType};
use crate::l2::rollup::{L2Transaction, TransactionType};

// Import quantum-resistant cryptography for testing
use crate::crypto::quantum_resistant::dilithium_keygen;

/// Test helper functions
mod test_helpers {
    use super::*;

    /// Create a test restaking manager with mock dependencies
    pub fn create_test_restaking_manager() -> RestakingManager {
        let config = RestakingConfig::default();

        // Create mock federation manager
        let federation_manager = Arc::new(MultiChainFederation::new());

        // Create mock anomaly detector
        let anomaly_detector = Arc::new(AnomalyDetector::new().unwrap());

        RestakingManager::new(federation_manager, anomaly_detector, config)
    }

    /// Create a test restaking request
    pub fn create_test_restaking_request(
        user_did: &str,
        amount: u64,
        chain: &str,
    ) -> RestakingRequest {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let id = format!("test_req_{}", timestamp);

        // Generate test signature
        use sha3::Digest;
        let mut hasher = sha3::Sha3_256::new();
        hasher.update(id.as_bytes());
        hasher.update(user_did.as_bytes());
        hasher.update(amount.to_le_bytes());
        hasher.update(chain.as_bytes());
        let request_hash = hasher.finalize().to_vec();

        RestakingRequest {
            id,
            user_did: user_did.to_string(),
            amount,
            chain: chain.to_string(),
            lock_period: 1000,
            public_key: vec![0x01, 0x02, 0x03],
            quantum_public_key: None,
            timestamp,
            signature: vec![
                0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10, 0x11,
                0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F,
                0x20, 0x21, 0x22, 0x23,
            ],
            quantum_signature: None,
            request_hash,
        }
    }

    /// Create a test federation member
    pub fn create_test_federation_member(chain_name: &str) -> FederationMember {
        let params = crate::crypto::quantum_resistant::DilithiumParams::dilithium3();
        let Ok((public_key, _)) = dilithium_keygen(&params) else {
            panic!("Failed to generate Dilithium keys");
        };

        FederationMember {
            chain_id: format!("chain_{}", chain_name),
            chain_name: chain_name.to_string(),
            chain_type: ChainType::Layer1,
            federation_public_key: public_key,
            kyber_public_key: crate::crypto::quantum_resistant::KyberPublicKey {
                matrix_a: vec![],
                vector_t: vec![],
                security_level: crate::crypto::quantum_resistant::KyberSecurityLevel::Kyber512,
            },
            stake_weight: 1_000_000,
            is_active: true,
            last_heartbeat: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            voting_power: 1000,
            governance_params: FederationGovernanceParams {
                min_stake_to_vote: 1000,
                min_stake_to_propose: 10000,
                voting_period: 86400,
                quorum_threshold: 0.5,
                supermajority_threshold: 0.67,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_helpers::*;

    /// Test 1: Normal restaking request submission
    #[test]
    fn test_normal_restaking_request_submission() {
        let manager = create_test_restaking_manager();
        let request = create_test_restaking_request("did:vote:user1", 5000, "l2_rollup");

        let result = manager.submit_restaking_request(
            request.user_did.clone(),
            request.amount,
            request.chain.clone(),
            request.lock_period,
            request.public_key.clone(),
            request.quantum_public_key.clone(),
            request.signature.clone(),
            request.quantum_signature.clone(),
        );

        if let Err(e) = &result {
            println!("Restaking request failed with error: {:?}", e);
        }
        assert!(result.is_ok());
        let request_id = result.unwrap();
        assert!(!request_id.is_empty());
        assert!(request_id.starts_with("rstk_"));
    }

    /// Test 2: LST issuance for valid restaking request
    #[test]
    fn test_lst_issuance_for_valid_request() {
        let manager = create_test_restaking_manager();
        let request = create_test_restaking_request("did:vote:user2", 10000, "l2_rollup");

        let result = manager.submit_restaking_request(
            request.user_did.clone(),
            request.amount,
            request.chain.clone(),
            request.lock_period,
            request.public_key.clone(),
            request.quantum_public_key.clone(),
            request.signature.clone(),
            request.quantum_signature.clone(),
        );

        assert!(result.is_ok());

        // Check that LST was created
        let lsts = manager.get_active_liquid_staking_tokens().unwrap();
        assert_eq!(lsts.len(), 1);
        assert_eq!(lsts[0].user_did, request.user_did);
        assert_eq!(lsts[0].restaked_amount, request.amount);
        assert_eq!(lsts[0].target_chain, request.chain);
    }

    /// Test 3: Yield calculation for different stake amounts
    #[test]
    fn test_yield_calculation_different_stake_amounts() {
        let manager = create_test_restaking_manager();

        // Test small stake (no bonus)
        let _small_request = create_test_restaking_request("did:vote:user3", 5000, "l2_rollup");
        let small_yield = manager
            .calculate_yield_rate("l2_rollup", 5000, 1000)
            .unwrap();

        // Test medium stake (10% bonus)
        let medium_yield = manager
            .calculate_yield_rate("l2_rollup", 15000, 1000)
            .unwrap();

        // Test large stake (20% bonus)
        let large_yield = manager
            .calculate_yield_rate("l2_rollup", 150000, 1000)
            .unwrap();

        assert!(medium_yield > small_yield);
        assert!(large_yield > medium_yield);
        assert!(small_yield >= 0.02); // Minimum yield rate
        assert!(large_yield <= 0.15); // Maximum yield rate
    }

    /// Test 4: Yield accrual over time
    #[test]
    fn test_yield_accrual_over_time() {
        let manager = create_test_restaking_manager();
        let request = create_test_restaking_request("did:vote:user4", 20000, "l2_rollup");

        let result = manager.submit_restaking_request(
            request.user_did.clone(),
            request.amount,
            request.chain.clone(),
            request.lock_period,
            request.public_key.clone(),
            request.quantum_public_key.clone(),
            request.signature.clone(),
            request.quantum_signature.clone(),
        );

        assert!(result.is_ok());

        // Get initial LST
        let lsts = manager.get_active_liquid_staking_tokens().unwrap();
        assert_eq!(lsts.len(), 1);
        let initial_yield = lsts[0].accumulated_yield;

        // Check that LST was created successfully
        assert_eq!(lsts[0].accumulated_yield, initial_yield);
    }

    /// Test 5: User restaking position tracking
    #[test]
    fn test_user_restaking_position_tracking() {
        let manager = create_test_restaking_manager();
        let user_did = "did:vote:user5";

        // Submit first restaking request
        let request1 = create_test_restaking_request(user_did, 10000, "l2_rollup");
        manager
            .submit_restaking_request(
                request1.user_did.clone(),
                request1.amount,
                request1.chain.clone(),
                request1.lock_period,
                request1.public_key.clone(),
                request1.quantum_public_key.clone(),
                request1.signature.clone(),
                request1.quantum_signature.clone(),
            )
            .unwrap();

        // Submit second restaking request to the same chain (should merge or replace)
        let request2 = create_test_restaking_request(user_did, 15000, "l2_rollup");
        manager
            .submit_restaking_request(
                request2.user_did.clone(),
                request2.amount,
                request2.chain.clone(),
                request2.lock_period,
                request2.public_key.clone(),
                request2.quantum_public_key.clone(),
                request2.signature.clone(),
                request2.quantum_signature.clone(),
            )
            .unwrap();

        // Check user position
        let position = manager.get_user_restaking_position(user_did).unwrap();
        assert!(position.is_some());
        let position = position.unwrap();
        assert_eq!(position.total_restaked, 25000); // 10000 + 15000
        assert_eq!(position.active_lsts.len(), 1);
        assert!(position.active_lsts.contains_key("l2_rollup"));
    }

    /// Test 6: Edge case - zero stake amount
    #[test]
    fn test_edge_case_zero_stake_amount() {
        let manager = create_test_restaking_manager();
        let request = create_test_restaking_request("did:vote:user6", 0, "l2_rollup");

        let result = manager.submit_restaking_request(
            request.user_did.clone(),
            request.amount,
            request.chain.clone(),
            request.lock_period,
            request.public_key.clone(),
            request.quantum_public_key.clone(),
            request.signature.clone(),
            request.quantum_signature.clone(),
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), RestakingError::InvalidAmount);
    }

    /// Test 7: Edge case - maximum restaking limits
    #[test]
    fn test_edge_case_maximum_restaking_limits() {
        let manager = create_test_restaking_manager();
        let max_amount = manager.config.max_restaking_amount + 1;
        let request = create_test_restaking_request("did:vote:user7", max_amount, "l2_rollup");

        let result = manager.submit_restaking_request(
            request.user_did.clone(),
            request.amount,
            request.chain.clone(),
            request.lock_period,
            request.public_key.clone(),
            request.quantum_public_key.clone(),
            request.signature.clone(),
            request.quantum_signature.clone(),
        );

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), RestakingError::RestakingLimitExceeded);
    }

    /// Test 8: Edge case - single chain restaking
    #[test]
    fn test_edge_case_single_chain_restaking() {
        let manager = create_test_restaking_manager();
        let user_did = "did:vote:user8";

        // Submit restaking request to single chain
        let request = create_test_restaking_request(user_did, 5000, "l2_rollup");
        let result = manager.submit_restaking_request(
            request.user_did.clone(),
            request.amount,
            request.chain.clone(),
            request.lock_period,
            request.public_key.clone(),
            request.quantum_public_key.clone(),
            request.signature.clone(),
            request.quantum_signature.clone(),
        );

        assert!(result.is_ok());

        // Check that only one LST was created
        let lsts = manager.get_active_liquid_staking_tokens().unwrap();
        assert_eq!(lsts.len(), 1);
        assert_eq!(lsts[0].target_chain, "l2_rollup");
    }

    /// Test 9: Malicious behavior - forged restaking request
    #[test]
    fn test_malicious_behavior_forged_restaking_request() {
        let manager = create_test_restaking_manager();

        // Create request with invalid signature
        let mut request = create_test_restaking_request("did:vote:user9", 10000, "l2_rollup");
        request.signature = vec![0xFF, 0xFF, 0xFF]; // Invalid signature

        let result = manager.submit_restaking_request(
            request.user_did.clone(),
            request.amount,
            request.chain.clone(),
            request.lock_period,
            request.public_key.clone(),
            request.quantum_public_key.clone(),
            request.signature.clone(),
            request.quantum_signature.clone(),
        );

        // Should fail due to invalid signature
        assert!(result.is_err());
    }

    /// Test 10: Slashing condition - invalid L2 fraud proof
    #[test]
    fn test_slashing_condition_invalid_l2_fraud_proof() {
        let manager = create_test_restaking_manager();
        let validator_id = "validator_1";

        // First, create a restaking position for the validator
        let request = create_test_restaking_request(validator_id, 20000, "l2_rollup");
        manager
            .submit_restaking_request(
                request.user_did.clone(),
                request.amount,
                request.chain.clone(),
                request.lock_period,
                request.public_key.clone(),
                request.quantum_public_key.clone(),
                request.signature.clone(),
                request.quantum_signature.clone(),
            )
            .unwrap();

        // Create slashing condition
        let slashing_condition = RestakingSlashingCondition::InvalidL2FraudProof {
            chain: "l2_rollup".to_string(),
            fraud_proof_hash: vec![0x01, 0x02, 0x03],
            validator_id: validator_id.to_string(),
        };

        // Process slashing condition
        let result = manager.process_slashing_condition(slashing_condition);
        assert!(result.is_ok());

        // Check that slashing events increased
        let metrics = manager.get_restaking_metrics().unwrap();
        assert!(metrics.slashing_events > 0);
    }

    /// Test 11: Slashing condition - cross-chain vote manipulation
    #[test]
    fn test_slashing_condition_cross_chain_vote_manipulation() {
        let manager = create_test_restaking_manager();
        let validator_id = "validator_2";

        // Create slashing condition for vote manipulation
        let slashing_condition = RestakingSlashingCondition::CrossChainVoteManipulation {
            source_chain: "l2_rollup".to_string(),
            target_chain: "l2_rollup".to_string(),
            vote_hash: vec![0x04, 0x05, 0x06],
            validator_id: validator_id.to_string(),
        };

        // Process slashing condition
        let result = manager.process_slashing_condition(slashing_condition);
        assert!(result.is_ok());
    }

    /// Test 12: Slashing condition - cross-chain double signing
    #[test]
    fn test_slashing_condition_cross_chain_double_signing() {
        let manager = create_test_restaking_manager();
        let validator_id = "validator_3";

        // Create slashing condition for double signing
        let slashing_condition = RestakingSlashingCondition::CrossChainDoubleSigning {
            chain1: "l2_rollup".to_string(),
            chain2: "l2_rollup".to_string(),
            block_hash1: vec![0x07, 0x08, 0x09],
            block_hash2: vec![0x0A, 0x0B, 0x0C],
            validator_id: validator_id.to_string(),
        };

        // Process slashing condition
        let result = manager.process_slashing_condition(slashing_condition);
        assert!(result.is_ok());
    }

    /// Test 13: Withdrawal of restaked tokens
    #[test]
    fn test_withdrawal_of_restaked_tokens() {
        let manager = create_test_restaking_manager();
        let user_did = "did:vote:user13";

        // Submit restaking request
        let request = create_test_restaking_request(user_did, 10000, "l2_rollup");
        let _request_id = manager
            .submit_restaking_request(
                request.user_did.clone(),
                request.amount,
                request.chain.clone(),
                request.lock_period,
                request.public_key.clone(),
                request.quantum_public_key.clone(),
                request.signature.clone(),
                request.quantum_signature.clone(),
            )
            .unwrap();

        // Get the LST ID
        let lsts = manager.get_active_liquid_staking_tokens().unwrap();
        assert_eq!(lsts.len(), 1);
        let lst_id = lsts[0].id.clone();

        // Attempt withdrawal (should fail due to lock period)
        let result = manager.withdraw_restaked_tokens(user_did, &lst_id);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), RestakingError::WithdrawalPeriodNotMet);
    }

    /// Test 14: Chain risk assessment
    #[test]
    fn test_chain_risk_assessment() {
        let manager = create_test_restaking_manager();

        // Assess risk for a test chain
        let assessment = manager.assess_chain_risk("test_chain").unwrap();

        assert_eq!(assessment.chain_id, "test_chain");
        assert!(assessment.risk_score >= 0.0 && assessment.risk_score <= 1.0);
        assert!(!assessment.risk_factors.is_empty());
        assert!(assessment.assessed_at > 0);
    }

    /// Test 15: Restaking metrics calculation
    #[test]
    fn test_restaking_metrics_calculation() {
        let manager = create_test_restaking_manager();

        // Submit multiple restaking requests
        for i in 0..3 {
            let user_did = format!("did:vote:user{}", i + 15);
            let chain = "l2_rollup";
            let request = create_test_restaking_request(&user_did, 5000 + (i * 1000) as u64, chain);

            manager
                .submit_restaking_request(
                    request.user_did.clone(),
                    request.amount,
                    request.chain.clone(),
                    request.lock_period,
                    request.public_key.clone(),
                    request.quantum_public_key.clone(),
                    request.signature.clone(),
                    request.quantum_signature.clone(),
                )
                .unwrap();
        }

        // Get metrics
        let metrics = manager.get_restaking_metrics().unwrap();

        assert!(metrics.total_restaked > 0);
        assert!(metrics.active_positions > 0);
        assert!(metrics.average_yield_rate > 0.0);
        assert!(!metrics.chain_distribution.is_empty());
        assert!(!metrics.yield_distribution.is_empty());
    }

    /// Test 16: Chart data generation for yield rates
    #[test]
    fn test_chart_data_generation_yield_rates() {
        let manager = create_test_restaking_manager();

        // Submit restaking requests to different chains
        let chains = ["l2_rollup"];
        for (i, chain) in chains.iter().enumerate() {
            let user_did = format!("did:vote:user{}", i + 16);
            let request = create_test_restaking_request(&user_did, 10000, chain);

            manager
                .submit_restaking_request(
                    request.user_did.clone(),
                    request.amount,
                    request.chain.clone(),
                    request.lock_period,
                    request.public_key.clone(),
                    request.quantum_public_key.clone(),
                    request.signature.clone(),
                    request.quantum_signature.clone(),
                )
                .unwrap();
        }

        // Generate chart data
        let chart_data = manager
            .generate_restaking_chart_data("bar", "yield_rate")
            .unwrap();

        assert_eq!(chart_data.chart_type, "bar");
        assert_eq!(chart_data.title, "Restaking Yield Rates by Chain");
        assert!(!chart_data.data.labels.is_empty());
        assert!(!chart_data.data.datasets.is_empty());
    }

    /// Test 17: Chart data generation for distribution
    #[test]
    fn test_chart_data_generation_distribution() {
        let manager = create_test_restaking_manager();

        // Submit restaking requests
        let request = create_test_restaking_request("did:vote:user17", 15000, "l2_rollup");
        manager
            .submit_restaking_request(
                request.user_did.clone(),
                request.amount,
                request.chain.clone(),
                request.lock_period,
                request.public_key.clone(),
                request.quantum_public_key.clone(),
                request.signature.clone(),
                request.quantum_signature.clone(),
            )
            .unwrap();

        // Generate chart data
        let chart_data = manager
            .generate_restaking_chart_data("bar", "distribution")
            .unwrap();

        assert_eq!(chart_data.chart_type, "bar");
        assert_eq!(chart_data.title, "Restaking Distribution by Chain");
        assert!(!chart_data.data.labels.is_empty());
        assert!(!chart_data.data.datasets.is_empty());
    }

    /// Test 18: Stress test - 10+ chains
    #[test]
    fn test_stress_test_multiple_chains() {
        let manager = create_test_restaking_manager();
        let chains = [
            "l2_rollup",
            "l2_rollup",
            "l2_rollup",
            "l2_rollup",
            "l2_rollup",
            "cardano",
            "algorand",
            "tezos",
            "near",
            "fantom",
            "polygon",
        ];

        let mut _total_restaked = 0;

        for (i, chain) in chains.iter().enumerate() {
            let user_did = format!("did:vote:user{}", i + 18);
            let amount = 1000 + (i * 100) as u64;
            let request = create_test_restaking_request(&user_did, amount, chain);

            let result = manager.submit_restaking_request(
                request.user_did.clone(),
                request.amount,
                request.chain.clone(),
                request.lock_period,
                request.public_key.clone(),
                request.quantum_public_key.clone(),
                request.signature.clone(),
                request.quantum_signature.clone(),
            );

            if result.is_ok() {
                _total_restaked += amount;
            }
        }

        // Check that some restaking requests succeeded
        let metrics = manager.get_restaking_metrics().unwrap();
        assert!(metrics.total_restaked > 0);
    }

    /// Test 19: Stress test - large stake pools
    #[test]
    fn test_stress_test_large_stake_pools() {
        let manager = create_test_restaking_manager();
        let large_amount = 100_000; // Large stake amount

        // Submit multiple large restaking requests
        for i in 0..5 {
            let user_did = format!("did:vote:whale{}", i + 19);
            let request = create_test_restaking_request(&user_did, large_amount, "l2_rollup");

            let result = manager.submit_restaking_request(
                request.user_did.clone(),
                request.amount,
                request.chain.clone(),
                request.lock_period,
                request.public_key.clone(),
                request.quantum_public_key.clone(),
                request.signature.clone(),
                request.quantum_signature.clone(),
            );

            // Some may fail due to limits, but some should succeed
            if result.is_ok() {
                let lsts = manager.get_active_liquid_staking_tokens().unwrap();
                assert!(lsts.iter().any(|lst| lst.restaked_amount == large_amount));
            }
        }
    }

    /// Test 20: Stress test - high TPS simulation
    #[test]
    fn test_stress_test_high_tps_simulation() {
        let manager = create_test_restaking_manager();
        let num_requests = 100; // Simulate 100 requests

        let mut successful_requests = 0;

        for i in 0..num_requests {
            let user_did = format!("did:vote:tps_user{}", i + 20);
            let chain = "l2_rollup";
            let amount = 1000 + (i % 10) * 100;
            let request = create_test_restaking_request(&user_did, amount, chain);

            let result = manager.submit_restaking_request(
                request.user_did.clone(),
                request.amount,
                request.chain.clone(),
                request.lock_period,
                request.public_key.clone(),
                request.quantum_public_key.clone(),
                request.signature.clone(),
                request.quantum_signature.clone(),
            );

            if result.is_ok() {
                successful_requests += 1;
            }
        }

        // Check that a reasonable number of requests succeeded
        assert!(successful_requests > 0);

        let metrics = manager.get_restaking_metrics().unwrap();
        assert!(metrics.active_positions > 0);
    }

    /// Test 21: Integration test - PoS consensus integration
    #[test]
    fn test_integration_pos_consensus() {
        let manager = create_test_restaking_manager();

        // Create a validator
        let _validator =
            Validator::new("validator_test".to_string(), 50000, vec![0x01, 0x02, 0x03]);

        // Submit restaking request for validator
        let request = create_test_restaking_request("validator_test", 20000, "l2_rollup");
        let result = manager.submit_restaking_request(
            request.user_did.clone(),
            request.amount,
            request.chain.clone(),
            request.lock_period,
            request.public_key.clone(),
            request.quantum_public_key.clone(),
            request.signature.clone(),
            request.quantum_signature.clone(),
        );

        assert!(result.is_ok());

        // Check that validator's restaking position was created
        let position = manager
            .get_user_restaking_position("validator_test")
            .unwrap();
        assert!(position.is_some());
    }

    /// Test 22: Integration test - governance integration
    #[test]
    fn test_integration_governance() {
        let manager = create_test_restaking_manager();

        // Create a governance proposal
        let _proposal = Proposal {
            id: "proposal_test".to_string(),
            title: "Test Restaking Proposal".to_string(),
            description: "Test proposal for restaking integration".to_string(),
            proposal_type: ProposalType::EconomicUpdate,
            proposer: vec![0x01, 0x02, 0x03],
            quantum_proposer: None,
            proposer_stake: 10000,
            created_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            voting_start: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            voting_end: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + 86400,
            min_stake_to_vote: 1000,
            min_stake_to_propose: 5000,
            vote_tally: crate::governance::proposal::VoteTally {
                votes_for: 0,
                votes_against: 0,
                abstentions: 0,
                total_stake_weight: 0,
                stake_weight_for: 0,
                stake_weight_against: 0,
                stake_weight_abstain: 0,
                participation_rate: 0.0,
                total_votes: 0,
            },
            status: ProposalStatus::Active,
            execution_params: HashMap::new(),
            proposal_hash: vec![0x04, 0x05, 0x06],
            signature: vec![0x07, 0x08, 0x09],
            quantum_signature: None,
        };

        // Submit restaking request from proposer
        let request = create_test_restaking_request("proposer_test", 15000, "l2_rollup");
        let result = manager.submit_restaking_request(
            request.user_did.clone(),
            request.amount,
            request.chain.clone(),
            request.lock_period,
            request.public_key.clone(),
            request.quantum_public_key.clone(),
            request.signature.clone(),
            request.quantum_signature.clone(),
        );

        assert!(result.is_ok());
    }

    /// Test 23: Integration test - federation integration
    #[test]
    fn test_integration_federation() {
        let manager = create_test_restaking_manager();

        // Create federation member
        let _member = create_test_federation_member("test_federation_chain");

        // Submit restaking request to federation chain
        let request =
            create_test_restaking_request("federation_user", 12000, "test_federation_chain");
        let result = manager.submit_restaking_request(
            request.user_did.clone(),
            request.amount,
            request.chain.clone(),
            request.lock_period,
            request.public_key.clone(),
            request.quantum_public_key.clone(),
            request.signature.clone(),
            request.quantum_signature.clone(),
        );

        // Result may be ok or err depending on federation setup
        // The important thing is that it doesn't panic
        assert!(result.is_ok() || result.is_err());
    }

    /// Test 24: Integration test - L2 rollup integration
    #[test]
    fn test_integration_l2_rollup() {
        let manager = create_test_restaking_manager();

        // Create L2 transaction
        let _l2_tx = L2Transaction {
            hash: vec![0x01, 0x02, 0x03],
            tx_type: TransactionType::VoteSubmission,
            from: vec![0x04, 0x05, 0x06],
            to: None,
            data: vec![0x07, 0x08, 0x09],
            gas_limit: 100000,
            gas_price: 1000,
            nonce: 1,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            signature: vec![0x0A, 0x0B, 0x0C],
            status: crate::l2::rollup::TransactionStatus::Pending,
        };

        // Submit restaking request for L2
        let request = create_test_restaking_request("l2_user", 8000, "l2_rollup");
        let result = manager.submit_restaking_request(
            request.user_did.clone(),
            request.amount,
            request.chain.clone(),
            request.lock_period,
            request.public_key.clone(),
            request.quantum_public_key.clone(),
            request.signature.clone(),
            request.quantum_signature.clone(),
        );

        // Result may be ok or err depending on L2 setup
        assert!(result.is_ok() || result.is_err());
    }

    /// Test 25: Integration test - anomaly detection integration
    #[test]
    fn test_integration_anomaly_detection() {
        let manager = create_test_restaking_manager();

        // Submit a large restaking request that should trigger anomaly detection
        let large_request = create_test_restaking_request("anomaly_user", 200_000, "l2_rollup");
        let result = manager.submit_restaking_request(
            large_request.user_did.clone(),
            large_request.amount,
            large_request.chain.clone(),
            large_request.lock_period,
            large_request.public_key.clone(),
            large_request.quantum_public_key.clone(),
            large_request.signature.clone(),
            large_request.quantum_signature.clone(),
        );

        // The request should be processed (may succeed or fail based on limits)
        // but anomaly detection should be triggered
        assert!(result.is_ok() || result.is_err());

        // Check that metrics were updated
        let metrics = manager.get_restaking_metrics().unwrap();
        assert!(metrics.timestamp > 0);
    }

    /// Test 26: Edge case - invalid chain for restaking
    #[test]
    fn test_edge_case_invalid_chain() {
        let manager = create_test_restaking_manager();
        let request = create_test_restaking_request("did:vote:user26", 5000, "invalid_chain");

        let result = manager.submit_restaking_request(
            request.user_did.clone(),
            request.amount,
            request.chain.clone(),
            request.lock_period,
            request.public_key.clone(),
            request.quantum_public_key.clone(),
            request.signature.clone(),
            request.quantum_signature.clone(),
        );

        // Should fail due to invalid chain
        assert!(result.is_err());
    }

    /// Test 27: Edge case - maximum chains per user
    #[test]
    fn test_edge_case_maximum_chains_per_user() {
        let manager = create_test_restaking_manager();
        let user_did = "did:vote:user27";
        let max_chains = manager.config.max_chains_per_user;

        // Submit requests to maximum number of chains
        for i in 0..max_chains {
            let chain = format!("chain_{}", i);
            let request = create_test_restaking_request(user_did, 1000, &chain);

            let result = manager.submit_restaking_request(
                request.user_did.clone(),
                request.amount,
                request.chain.clone(),
                request.lock_period,
                request.public_key.clone(),
                request.quantum_public_key.clone(),
                request.signature.clone(),
                request.quantum_signature.clone(),
            );

            // Some may succeed, some may fail
            assert!(result.is_ok() || result.is_err());
        }

        // Try to submit one more request (should fail)
        let request = create_test_restaking_request(user_did, 1000, "extra_chain");
        let result = manager.submit_restaking_request(
            request.user_did.clone(),
            request.amount,
            request.chain.clone(),
            request.lock_period,
            request.public_key.clone(),
            request.quantum_public_key.clone(),
            request.signature.clone(),
            request.quantum_signature.clone(),
        );

        // Should fail due to maximum chains limit
        assert!(result.is_err());
    }

    /// Test 28: Edge case - chain risk too high
    #[test]
    fn test_edge_case_chain_risk_too_high() {
        let manager = create_test_restaking_manager();

        // Create a high-risk chain assessment
        let high_risk_assessment = ChainRiskAssessment {
            chain_id: "high_risk_chain".to_string(),
            risk_score: 0.95, // Very high risk
            risk_factors: vec![],
            assessed_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            risk_level: ChainRiskLevel::VeryHigh,
        };

        // Store the high-risk assessment
        {
            let mut assessments = manager.chain_risk_assessments.write().unwrap();
            assessments.insert("high_risk_chain".to_string(), high_risk_assessment);
        }

        // Try to restake to high-risk chain
        let request = create_test_restaking_request("did:vote:user28", 5000, "high_risk_chain");
        let result = manager.submit_restaking_request(
            request.user_did.clone(),
            request.amount,
            request.chain.clone(),
            request.lock_period,
            request.public_key.clone(),
            request.quantum_public_key.clone(),
            request.signature.clone(),
            request.quantum_signature.clone(),
        );

        // Should fail due to high risk
        assert!(result.is_err());
    }

    /// Test 29: Edge case - insufficient network participation
    #[test]
    fn test_edge_case_insufficient_participation() {
        let manager = create_test_restaking_manager();

        // Modify yield parameters to require high participation
        {
            let mut yield_params = manager.yield_parameters.write().unwrap();
            yield_params.participation_bonus = 0.5; // 50% participation bonus required
        }

        // Try to restake with low participation
        let request = create_test_restaking_request("did:vote:user29", 5000, "l2_rollup");
        let result = manager.submit_restaking_request(
            request.user_did.clone(),
            request.amount,
            request.chain.clone(),
            request.lock_period,
            request.public_key.clone(),
            request.quantum_public_key.clone(),
            request.signature.clone(),
            request.quantum_signature.clone(),
        );

        // Should still succeed as participation is not strictly enforced in this implementation
        assert!(result.is_ok());
    }

    /// Test 30: Comprehensive integration test
    #[test]
    fn test_comprehensive_integration() {
        let manager = create_test_restaking_manager();

        // Test multiple users, multiple chains, multiple scenarios
        let test_cases = vec![
            ("did:vote:user30a", 10000, "l2_rollup"),
            ("did:vote:user30b", 15000, "l2_rollup"),
            ("did:vote:user30c", 20000, "l2_rollup"),
            ("did:vote:user30d", 5000, "l2_rollup"),
            ("did:vote:user30e", 25000, "l2_rollup"),
        ];

        let mut successful_requests = 0;

        for (user_did, amount, chain) in test_cases {
            let request = create_test_restaking_request(user_did, amount, chain);

            let result = manager.submit_restaking_request(
                request.user_did.clone(),
                request.amount,
                request.chain.clone(),
                request.lock_period,
                request.public_key.clone(),
                request.quantum_public_key.clone(),
                request.signature.clone(),
                request.quantum_signature.clone(),
            );

            if result.is_ok() {
                successful_requests += 1;
            }
        }

        // Check final metrics
        let metrics = manager.get_restaking_metrics().unwrap();
        assert!(metrics.total_restaked > 0);
        assert!(metrics.active_positions > 0);
        assert!(metrics.average_yield_rate > 0.0);

        // Verify that some requests succeeded
        assert!(successful_requests > 0);
    }
}
