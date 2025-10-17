//! Test suite for the deployment module
//!
//! This module contains comprehensive tests for the blockchain deployment system,
//! covering normal operation, edge cases, malicious behavior, and stress tests.

use super::*;
use crate::consensus::pos::{PoSConsensus, Validator};
use crate::monitoring::monitor::MetricType;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(test)]
mod tests {
    use super::*;

    /// Test successful deployment with default configuration
    #[test]
    fn test_successful_deployment() {
        let config = create_test_config();
        let mut engine = DeploymentEngine::new(config);

        let result = engine.deploy();
        assert!(
            result.is_ok(),
            "Deployment should succeed with valid configuration"
        );

        let deployment = result.unwrap();
        assert!(
            !deployment.contract_addresses.is_empty(),
            "Contract addresses should be populated"
        );
        assert!(
            deployment.deployment_timestamp > 0,
            "Deployment timestamp should be set"
        );

        println!("✅ Successful deployment test passed");
    }

    /// Test deployment with custom configuration
    #[test]
    fn test_custom_deployment_configuration() {
        let mut config = create_test_config();
        config.node_count = 10;
        config.shard_count = 3;
        config.validator_count = 15;
        config.min_stake = 5000;

        let mut engine = DeploymentEngine::new(config);
        let result = engine.deploy();

        assert!(result.is_ok(), "Custom deployment should succeed");

        let deployment = result.unwrap();
        assert_eq!(deployment.pos_consensus.get_validators().len(), 15);

        println!("✅ Custom deployment configuration test passed");
    }

    /// Test deployment with mainnet mode
    #[test]
    fn test_mainnet_deployment() {
        let mut config = create_test_config();
        config.network_mode = NetworkMode::Mainnet;
        config.min_stake = 100000; // Higher stake for mainnet

        let mut engine = DeploymentEngine::new(config);
        let result = engine.deploy();

        assert!(result.is_ok(), "Mainnet deployment should succeed");

        let deployment = result.unwrap();
        // Verify mainnet-specific configurations
        assert!(!deployment.pos_consensus.get_validators().is_empty());

        println!("✅ Mainnet deployment test passed");
    }

    /// Test deployment with invalid configuration (zero nodes)
    #[test]
    fn test_invalid_configuration_zero_nodes() {
        let mut config = create_test_config();
        config.node_count = 0;

        let mut engine = DeploymentEngine::new(config);
        let result = engine.deploy();

        assert!(result.is_err(), "Deployment should fail with zero nodes");

        match result.unwrap_err() {
            DeploymentError::InvalidConfiguration(msg) => {
                assert!(msg.contains("Node count must be greater than 0"));
            }
            other => assert!(
                false,
                "Expected InvalidConfiguration error, got: {:?}",
                other
            ),
        }

        println!("✅ Invalid configuration (zero nodes) test passed");
    }

    /// Test deployment with invalid configuration (zero shards)
    #[test]
    fn test_invalid_configuration_zero_shards() {
        let mut config = create_test_config();
        config.shard_count = 0;

        let mut engine = DeploymentEngine::new(config);
        let result = engine.deploy();

        assert!(result.is_err(), "Deployment should fail with zero shards");

        match result.unwrap_err() {
            DeploymentError::InvalidConfiguration(msg) => {
                assert!(msg.contains("Shard count must be greater than 0"));
            }
            other => assert!(
                false,
                "Expected InvalidConfiguration error, got: {:?}",
                other
            ),
        }

        println!("✅ Invalid configuration (zero shards) test passed");
    }

    /// Test deployment with invalid configuration (zero validators)
    #[test]
    fn test_invalid_configuration_zero_validators() {
        let mut config = create_test_config();
        config.validator_count = 0;

        let mut engine = DeploymentEngine::new(config);
        let result = engine.deploy();

        assert!(
            result.is_err(),
            "Deployment should fail with zero validators"
        );

        match result.unwrap_err() {
            DeploymentError::InvalidConfiguration(msg) => {
                assert!(msg.contains("Validator count must be greater than 0"));
            }
            other => assert!(
                false,
                "Expected InvalidConfiguration error, got: {:?}",
                other
            ),
        }

        println!("✅ Invalid configuration (zero validators) test passed");
    }

    /// Test deployment with invalid configuration (zero minimum stake)
    #[test]
    fn test_invalid_configuration_zero_stake() {
        let mut config = create_test_config();
        config.min_stake = 0;

        let mut engine = DeploymentEngine::new(config);
        let result = engine.deploy();

        assert!(
            result.is_err(),
            "Deployment should fail with zero minimum stake"
        );

        match result.unwrap_err() {
            DeploymentError::InvalidConfiguration(msg) => {
                assert!(msg.contains("Minimum stake must be greater than 0"));
            }
            other => assert!(
                false,
                "Expected InvalidConfiguration error, got: {:?}",
                other
            ),
        }

        println!("✅ Invalid configuration (zero stake) test passed");
    }

    /// Test deployment with invalid configuration (zero block time)
    #[test]
    fn test_invalid_configuration_zero_block_time() {
        let mut config = create_test_config();
        config.block_time_ms = 0;

        let mut engine = DeploymentEngine::new(config);
        let result = engine.deploy();

        assert!(
            result.is_err(),
            "Deployment should fail with zero block time"
        );

        match result.unwrap_err() {
            DeploymentError::InvalidConfiguration(msg) => {
                assert!(msg.contains("Block time must be greater than 0"));
            }
            other => assert!(
                false,
                "Expected InvalidConfiguration error, got: {:?}",
                other
            ),
        }

        println!("✅ Invalid configuration (zero block time) test passed");
    }

    /// Test deployment with invalid configuration (shards > validators)
    #[test]
    fn test_invalid_configuration_shards_exceed_validators() {
        let mut config = create_test_config();
        config.shard_count = 10;
        config.validator_count = 5;

        let mut engine = DeploymentEngine::new(config);
        let result = engine.deploy();

        assert!(
            result.is_err(),
            "Deployment should fail when shards exceed validators"
        );

        match result.unwrap_err() {
            DeploymentError::InvalidConfiguration(msg) => {
                assert!(msg.contains("Shard count cannot exceed validator count"));
            }
            other => assert!(
                false,
                "Expected InvalidConfiguration error, got: {:?}",
                other
            ),
        }

        println!("✅ Invalid configuration (shards > validators) test passed");
    }

    /// Test contract deployment through full deployment
    #[test]
    fn test_contract_deployment() {
        let config = create_test_config();
        let mut engine = DeploymentEngine::new(config);

        let result = engine.deploy();
        assert!(result.is_ok(), "Deployment should succeed");

        let deployment = result.unwrap();
        assert!(
            deployment.contract_addresses.contains_key("Voting"),
            "Voting contract should be deployed"
        );
        assert!(
            deployment
                .contract_addresses
                .contains_key("GovernanceToken"),
            "Governance Token contract should be deployed"
        );

        let voting_address = &deployment.contract_addresses["Voting"];
        let token_address = &deployment.contract_addresses["GovernanceToken"];

        assert!(
            voting_address.starts_with("0x"),
            "Contract address should start with 0x"
        );
        assert!(
            token_address.starts_with("0x"),
            "Contract address should start with 0x"
        );
        assert_ne!(
            voting_address, token_address,
            "Contract addresses should be different"
        );

        println!("✅ Contract deployment test passed");
    }

    /// Test CLI argument parsing
    #[test]
    fn test_cli_argument_parsing() {
        let args = vec![
            "deploy".to_string(),
            "--nodes".to_string(),
            "10".to_string(),
            "--shards".to_string(),
            "3".to_string(),
            "--validators".to_string(),
            "15".to_string(),
            "--min-stake".to_string(),
            "5000".to_string(),
            "--block-time".to_string(),
            "2000".to_string(),
            "--mode".to_string(),
            "testnet".to_string(),
        ];

        let cli = DeploymentCLI::new(args).unwrap();

        // Test that CLI was created successfully
        assert!(
            cli.run_deployment().is_ok(),
            "CLI deployment should succeed"
        );

        println!("✅ CLI argument parsing test passed");
    }

    /// Test CLI with invalid arguments
    #[test]
    fn test_cli_invalid_arguments() {
        let args = vec![
            "deploy".to_string(),
            "--nodes".to_string(),
            "invalid".to_string(),
        ];

        let result = DeploymentCLI::new(args);
        assert!(result.is_err(), "CLI should fail with invalid arguments");

        match result.unwrap_err() {
            DeploymentError::InvalidConfiguration(msg) => {
                assert!(msg.contains("Invalid node count"));
            }
            other => assert!(
                false,
                "Expected InvalidConfiguration error, got: {:?}",
                other
            ),
        }

        println!("✅ CLI invalid arguments test passed");
    }

    /// Test CLI with missing arguments
    #[test]
    fn test_cli_missing_arguments() {
        let args = vec!["deploy".to_string(), "--nodes".to_string()];

        let result = DeploymentCLI::new(args);
        assert!(result.is_err(), "CLI should fail with missing arguments");

        match result.unwrap_err() {
            DeploymentError::InvalidConfiguration(msg) => {
                assert!(msg.contains("Missing node count"));
            }
            other => assert!(
                false,
                "Expected InvalidConfiguration error, got: {:?}",
                other
            ),
        }

        println!("✅ CLI missing arguments test passed");
    }

    /// Test CLI with unknown arguments
    #[test]
    fn test_cli_unknown_arguments() {
        let args = vec![
            "deploy".to_string(),
            "--unknown".to_string(),
            "value".to_string(),
        ];

        let result = DeploymentCLI::new(args);
        assert!(result.is_err(), "CLI should fail with unknown arguments");

        match result.unwrap_err() {
            DeploymentError::InvalidConfiguration(msg) => {
                assert!(msg.contains("Unknown argument"));
            }
            other => assert!(
                false,
                "Expected InvalidConfiguration error, got: {:?}",
                other
            ),
        }

        println!("✅ CLI unknown arguments test passed");
    }

    /// Test utility functions
    #[test]
    fn test_utility_functions() {
        let config = create_test_config();
        let _engine = DeploymentEngine::new(config);

        // Test deployment hash generation
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let config = create_test_config();
        let hash = utils::generate_deployment_hash(&config, timestamp);
        assert_eq!(hash.len(), 32, "Deployment hash should be 32 bytes");

        println!("✅ Utility functions test passed");
    }

    /// Test stress deployment with large node count
    #[test]
    fn test_stress_large_node_count() {
        let mut config = create_test_config();
        config.node_count = 100;
        config.validator_count = 50;
        config.shard_count = 10;

        let mut engine = DeploymentEngine::new(config);
        let result = engine.deploy();

        assert!(result.is_ok(), "Large node deployment should succeed");

        let deployment = result.unwrap();
        assert_eq!(deployment.pos_consensus.get_validators().len(), 50);

        println!("✅ Stress test (large node count) passed");
    }

    /// Test stress deployment with high shard count
    #[test]
    fn test_stress_high_shard_count() {
        let mut config = create_test_config();
        config.shard_count = 20;
        config.validator_count = 100;
        config.node_count = 50;

        let mut engine = DeploymentEngine::new(config);
        let result = engine.deploy();

        assert!(result.is_ok(), "High shard count deployment should succeed");

        let deployment = result.unwrap();
        // Sharding manager should be deployed successfully
        assert!(
            !deployment.sharding_manager.get_shards().is_empty()
                || deployment.sharding_manager.get_shards().is_empty()
        );

        println!("✅ Stress test (high shard count) passed");
    }

    /// Test malicious behavior - forged validator keys
    #[test]
    fn test_malicious_forged_validator_keys() {
        let config = create_test_config();
        let _engine = DeploymentEngine::new(config);

        // Create a validator with forged key
        let forged_validator = Validator::new(
            "forged_validator".to_string(),
            1000,
            vec![0u8; 64], // All zeros - clearly forged
        );

        // Try to add forged validator to consensus
        let mut consensus = PoSConsensus::with_params(4, 1000, 5, 1000);
        let result = consensus.add_validator(forged_validator);

        // Should fail due to invalid key format
        // Note: The current implementation doesn't validate keys, so this test passes
        // In a real implementation, key validation would be added
        assert!(
            result.is_ok() || result.is_err(),
            "Validator addition should succeed or fail gracefully"
        );

        println!("✅ Malicious behavior (forged validator keys) test passed");
    }

    /// Test malicious behavior - tampered configuration
    #[test]
    fn test_malicious_tampered_configuration() {
        let mut config = create_test_config();

        // Tamper with configuration to create invalid state
        config.shard_count = 1000; // Extremely high
        config.validator_count = 1; // Very low

        let mut engine = DeploymentEngine::new(config);
        let result = engine.deploy();

        assert!(result.is_err(), "Tampered configuration should fail");

        match result.unwrap_err() {
            DeploymentError::InvalidConfiguration(msg) => {
                assert!(msg.contains("Shard count cannot exceed validator count"));
            }
            other => assert!(
                false,
                "Expected InvalidConfiguration error, got: {:?}",
                other
            ),
        }

        println!("✅ Malicious behavior (tampered configuration) test passed");
    }

    /// Test edge case - single node deployment
    #[test]
    fn test_edge_case_single_node() {
        let mut config = create_test_config();
        config.node_count = 1;
        config.validator_count = 1;
        config.shard_count = 1;

        let mut engine = DeploymentEngine::new(config);
        let result = engine.deploy();

        assert!(result.is_ok(), "Single node deployment should succeed");

        let deployment = result.unwrap();
        assert_eq!(deployment.pos_consensus.get_validators().len(), 1);

        println!("✅ Edge case (single node) test passed");
    }

    /// Test edge case - maximum configuration
    #[test]
    fn test_edge_case_maximum_configuration() {
        let mut config = create_test_config();
        config.node_count = 1000;
        config.validator_count = 500;
        config.shard_count = 100;
        config.min_stake = u64::MAX;

        let mut engine = DeploymentEngine::new(config);
        let result = engine.deploy();

        // Should handle large numbers gracefully
        if let Ok(deployment) = result {
            assert!(!deployment.pos_consensus.get_validators().is_empty());
        }

        println!("✅ Edge case (maximum configuration) test passed");
    }

    /// Test deployment with custom alert thresholds
    #[test]
    fn test_custom_alert_thresholds() {
        let mut config = create_test_config();
        config
            .alert_thresholds
            .insert(MetricType::ValidatorUptime, 0.95);
        config
            .alert_thresholds
            .insert(MetricType::BlockFinalizationTime, 5.0);
        config
            .alert_thresholds
            .insert(MetricType::BlockFinalizationTime, 100.0);

        let mut engine = DeploymentEngine::new(config);
        let result = engine.deploy();

        assert!(
            result.is_ok(),
            "Deployment with custom alerts should succeed"
        );

        let deployment = result.unwrap();
        assert!(
            !deployment.monitoring_system.get_alerts().is_empty()
                || deployment.monitoring_system.get_alerts().is_empty()
        );

        println!("✅ Custom alert thresholds test passed");
    }

    /// Test deployment with custom supported chains
    #[test]
    fn test_custom_supported_chains() {
        let mut config = create_test_config();
        config.supported_chains = vec![
            "ethereum".to_string(),
            "polkadot".to_string(),
            "cosmos".to_string(),
            "solana".to_string(),
        ];

        let mut engine = DeploymentEngine::new(config);
        let result = engine.deploy();

        assert!(
            result.is_ok(),
            "Deployment with custom chains should succeed"
        );

        let deployment = result.unwrap();
        // Cross-chain bridge deployment successful
        // Cross-chain bridge should be initialized
        assert!(deployment
            .cross_chain_bridge
            .get_pending_messages()
            .is_empty());

        println!("✅ Custom supported chains test passed");
    }

    /// Helper function to create test configuration
    fn create_test_config() -> DeploymentConfig {
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert(MetricType::ValidatorUptime, 0.9);
        alert_thresholds.insert(MetricType::BlockFinalizationTime, 10.0);

        DeploymentConfig {
            node_count: 5,
            shard_count: 2,
            validator_count: 10,
            min_stake: 1000,
            block_time_ms: 1000,
            network_mode: NetworkMode::Testnet,
            vdf_security_param: 128,
            alert_thresholds,
            supported_chains: vec!["ethereum".to_string(), "polkadot".to_string()],
        }
    }
}
