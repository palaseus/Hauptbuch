//! Comprehensive test suite for the Cross-Chain Bridge
//!
//! This module contains extensive tests for the cross-chain bridge, covering
//! normal operation, edge cases, malicious behavior, and stress tests to ensure
//! robustness and reliability of the cross-chain interoperability infrastructure.

use super::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to create a test cross-chain bridge
    fn create_test_bridge() -> CrossChainBridge {
        CrossChainBridge::new()
    }

    /// Helper function to create test metadata
    fn create_test_metadata() -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("test".to_string(), "true".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());
        metadata
    }

    /// Helper function to create a test cross-chain message
    fn create_test_message(
        message_type: CrossChainMessageType,
        target_chain: &str,
        payload: Vec<u8>,
    ) -> CrossChainMessage {
        CrossChainMessage {
            id: format!(
                "test_msg_{}",
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs()
            ),
            message_type,
            source_chain: "test_chain".to_string(),
            target_chain: target_chain.to_string(),
            payload,
            proof: vec![0u8; 32], // Dummy proof
            quantum_signature: None,
            encrypted_payload: None,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            expiration: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                + 3600, // 1 hour from now
            status: MessageStatus::Pending,
            priority: 5,
            metadata: create_test_metadata(),
        }
    }

    // ===== NORMAL OPERATION TESTS =====

    #[test]
    fn test_bridge_creation() {
        let bridge = create_test_bridge();
        let status = bridge.get_bridge_status();
        assert!(status.contains("Cross-Chain Bridge Status"));
        assert!(status.contains("Supported Chains"));
        println!("✅ Bridge creation test passed");
    }

    #[test]
    fn test_send_message() {
        let bridge = create_test_bridge();
        let payload = b"test_payload".to_vec();
        let metadata = create_test_metadata();

        let result = bridge.send_message(
            CrossChainMessageType::VoteResult,
            "ethereum",
            payload,
            5,
            metadata,
        );

        assert!(result.is_ok(), "Message sending should succeed");
        let message_id = result.unwrap();
        assert!(!message_id.is_empty(), "Message ID should not be empty");

        println!("✅ Send message test passed");
    }

    #[test]
    fn test_receive_message() {
        // Create bridge with proof verification disabled
        let config = BridgeConfig {
            max_queue_size: 10000,
            message_timeout: 3600,
            max_retries: 3,
            enable_encryption: true,
            enable_proof_verification: false, // Disable for test
            supported_chains: vec!["voting_blockchain".to_string()],
            bridge_fee: 1000,
        };
        let bridge = CrossChainBridge::with_config(config);

        let message = create_test_message(
            CrossChainMessageType::VoteResult,
            "voting_blockchain",
            b"test_payload".to_vec(),
        );

        let result = bridge.receive_message(message);
        assert!(result.is_ok(), "Message receiving should succeed");

        println!("✅ Receive message test passed");
    }

    #[test]
    fn test_lock_asset() {
        let bridge = create_test_bridge();
        let unlock_conditions = b"unlock_conditions".to_vec();

        let result = bridge.lock_asset("asset_123", "token", 1000, "ethereum", unlock_conditions);

        assert!(result.is_ok(), "Asset locking should succeed");
        let tx_hash = result.unwrap();
        assert!(!tx_hash.is_empty(), "Transaction hash should not be empty");

        // Verify asset is locked
        let locked_assets = bridge.get_locked_assets();
        assert!(!locked_assets.is_empty(), "Should have locked assets");

        println!("✅ Lock asset test passed");
    }

    #[test]
    fn test_unlock_asset() {
        let bridge = create_test_bridge();
        let unlock_conditions = b"unlock_conditions".to_vec();

        // First lock an asset
        bridge
            .lock_asset(
                "asset_456",
                "token",
                2000,
                "ethereum", // Use supported chain
                unlock_conditions.clone(),
            )
            .unwrap();

        // Then unlock it
        let unlock_proof = vec![0u8; 32]; // 32-byte proof to satisfy verification
        let result = bridge.unlock_asset("asset_456", unlock_proof);
        if let Err(e) = &result {
            println!("Unlock error: {}", e);
        }
        assert!(result.is_ok(), "Asset unlocking should succeed");

        println!("✅ Unlock asset test passed");
    }

    #[test]
    fn test_merkle_proof_verification() {
        let bridge = create_test_bridge();

        // Create a simple Merkle proof
        let proof = MerkleProof {
            root_hash: vec![1u8; 32],
            path: vec![vec![2u8; 32], vec![3u8; 32]],
            leaf_index: 0,
            total_leaves: 4,
            verified_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        let leaf_data = b"test_leaf_data";
        let result = bridge.verify_merkle_proof(&proof, leaf_data);
        assert!(result.is_ok(), "Merkle proof verification should succeed");

        println!("✅ Merkle proof verification test passed");
    }

    #[test]
    fn test_state_commitment() {
        let bridge = create_test_bridge();
        let state_data = b"test_state_data";

        let result = bridge.create_state_commitment("test_chain", state_data);
        assert!(result.is_ok(), "State commitment creation should succeed");

        let commitment = result.unwrap();
        assert_eq!(commitment.len(), 32, "Commitment should be 32 bytes");

        println!("✅ State commitment test passed");
    }

    #[test]
    fn test_get_pending_messages() {
        let bridge = create_test_bridge();

        // Send some messages
        for i in 0..5 {
            let payload = format!("test_payload_{}", i).into_bytes();
            bridge
                .send_message(
                    CrossChainMessageType::TokenTransfer,
                    "ethereum",
                    payload,
                    5,
                    create_test_metadata(),
                )
                .unwrap();
        }

        let pending_messages = bridge.get_pending_messages();
        assert_eq!(pending_messages.len(), 5, "Should have 5 pending messages");

        println!("✅ Get pending messages test passed");
    }

    #[test]
    fn test_get_locked_assets() {
        let bridge = create_test_bridge();

        // Lock some assets
        for i in 0..3 {
            bridge
                .lock_asset(
                    &format!("asset_{}", i),
                    "token",
                    1000 + i as u64 * 100,
                    "ethereum",
                    b"unlock_conditions".to_vec(),
                )
                .unwrap();
        }

        let locked_assets = bridge.get_locked_assets();
        assert_eq!(locked_assets.len(), 3, "Should have 3 locked assets");

        println!("✅ Get locked assets test passed");
    }

    #[test]
    fn test_clear_expired_messages() {
        let bridge = create_test_bridge();

        // Send a message that will expire
        let payload = b"expiring_payload".to_vec();
        bridge
            .send_message(
                CrossChainMessageType::VoteResult,
                "ethereum",
                payload,
                5,
                create_test_metadata(),
            )
            .unwrap();

        // Clear expired messages (should clear none since message is not expired)
        let cleared = bridge.clear_expired_messages();
        assert_eq!(cleared, 0, "Should not clear any non-expired messages");

        println!("✅ Clear expired messages test passed");
    }

    // ===== EDGE CASE TESTS =====

    #[test]
    fn test_unsupported_target_chain() {
        let bridge = create_test_bridge();
        let payload = b"test_payload".to_vec();

        let result = bridge.send_message(
            CrossChainMessageType::VoteResult,
            "unsupported_chain",
            payload,
            5,
            create_test_metadata(),
        );

        assert!(result.is_err(), "Should fail for unsupported chain");
        assert!(result.unwrap_err().contains("Unsupported target chain"));

        println!("✅ Unsupported target chain test passed");
    }

    #[test]
    fn test_empty_payload() {
        let bridge = create_test_bridge();
        let payload = vec![];

        let result = bridge.send_message(
            CrossChainMessageType::VoteResult,
            "ethereum",
            payload,
            5,
            create_test_metadata(),
        );

        assert!(result.is_ok(), "Empty payload should be allowed");

        println!("✅ Empty payload test passed");
    }

    #[test]
    fn test_high_priority_message() {
        let bridge = create_test_bridge();
        let payload = b"high_priority_payload".to_vec();

        let result = bridge.send_message(
            CrossChainMessageType::ValidatorUpdate,
            "ethereum",
            payload,
            255, // Maximum priority
            create_test_metadata(),
        );

        assert!(result.is_ok(), "High priority message should be sent");

        println!("✅ High priority message test passed");
    }

    #[test]
    fn test_large_payload() {
        let bridge = create_test_bridge();
        let payload = vec![0u8; 10000]; // 10KB payload

        let result = bridge.send_message(
            CrossChainMessageType::ShardSync,
            "ethereum",
            payload,
            5,
            create_test_metadata(),
        );

        assert!(result.is_ok(), "Large payload should be handled");

        println!("✅ Large payload test passed");
    }

    #[test]
    fn test_duplicate_asset_id() {
        let bridge = create_test_bridge();

        // Lock asset with same ID twice
        bridge
            .lock_asset(
                "duplicate_asset",
                "token",
                1000,
                "ethereum",
                b"conditions1".to_vec(),
            )
            .unwrap();

        let result = bridge.lock_asset(
            "duplicate_asset",
            "token",
            2000,
            "polkadot",
            b"conditions2".to_vec(),
        );

        // Should succeed (overwrites previous)
        assert!(result.is_ok(), "Duplicate asset ID should overwrite");

        println!("✅ Duplicate asset ID test passed");
    }

    #[test]
    fn test_zero_amount_asset() {
        let bridge = create_test_bridge();

        let result = bridge.lock_asset(
            "zero_asset",
            "token",
            0, // Zero amount
            "ethereum",
            b"conditions".to_vec(),
        );

        assert!(result.is_ok(), "Zero amount asset should be allowed");

        println!("✅ Zero amount asset test passed");
    }

    // ===== MALICIOUS BEHAVIOR TESTS =====

    #[test]
    fn test_forged_message() {
        let bridge = create_test_bridge();

        // Create message with invalid proof
        let mut message = create_test_message(
            CrossChainMessageType::VoteResult,
            "voting_blockchain",
            b"malicious_payload".to_vec(),
        );
        message.proof = vec![0xFFu8; 32]; // Invalid proof

        let result = bridge.receive_message(message);
        assert!(result.is_err(), "Forged message should be rejected");

        println!("✅ Forged message test passed");
    }

    #[test]
    fn test_tampered_proof() {
        let bridge = create_test_bridge();

        // Create message with tampered proof
        let mut message = create_test_message(
            CrossChainMessageType::TokenTransfer,
            "voting_blockchain",
            b"tampered_payload".to_vec(),
        );
        message.proof = vec![0xAAu8; 32]; // Tampered proof

        let result = bridge.receive_message(message);
        assert!(result.is_err(), "Tampered proof should be rejected");

        println!("✅ Tampered proof test passed");
    }

    #[test]
    fn test_expired_message() {
        let bridge = create_test_bridge();

        // Create expired message
        let mut message = create_test_message(
            CrossChainMessageType::VoteResult,
            "voting_blockchain",
            b"expired_payload".to_vec(),
        );
        message.expiration = 0; // Expired

        let result = bridge.receive_message(message);
        assert!(result.is_err(), "Expired message should be rejected");

        println!("✅ Expired message test passed");
    }

    #[test]
    fn test_invalid_unlock_proof() {
        let bridge = create_test_bridge();

        // Lock an asset
        bridge
            .lock_asset(
                "test_asset",
                "token",
                1000,
                "ethereum",
                b"conditions".to_vec(),
            )
            .unwrap();

        // Try to unlock with invalid proof
        let invalid_proof = vec![0u8; 10]; // Too short
        let result = bridge.unlock_asset("test_asset", invalid_proof);
        assert!(result.is_err(), "Invalid unlock proof should be rejected");

        println!("✅ Invalid unlock proof test passed");
    }

    #[test]
    fn test_nonexistent_asset_unlock() {
        let bridge = create_test_bridge();

        let unlock_proof = b"valid_proof".to_vec();
        let result = bridge.unlock_asset("nonexistent_asset", unlock_proof);
        assert!(result.is_err(), "Unlocking nonexistent asset should fail");

        println!("✅ Nonexistent asset unlock test passed");
    }

    #[test]
    fn test_malicious_merkle_proof() {
        let bridge = create_test_bridge();

        // Create malicious Merkle proof
        let malicious_proof = MerkleProof {
            root_hash: vec![0xFFu8; 32],  // Malicious root
            path: vec![vec![0xAAu8; 32]], // Malicious path
            leaf_index: 0,
            total_leaves: 1,
            verified_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        };

        let leaf_data = b"legitimate_data";
        let result = bridge.verify_merkle_proof(&malicious_proof, leaf_data);
        assert!(
            result.is_ok(),
            "Merkle proof verification should handle malicious proofs"
        );

        // The verification should return false for malicious proofs
        if let Ok(is_valid) = result {
            assert!(!is_valid, "Malicious proof should be invalid");
        }

        println!("✅ Malicious Merkle proof test passed");
    }

    // ===== STRESS TESTS =====

    #[test]
    fn test_high_message_volume() {
        let bridge = create_test_bridge();

        // Send large number of messages
        for i in 0..1000 {
            let payload = format!("stress_payload_{}", i).into_bytes();
            let result = bridge.send_message(
                CrossChainMessageType::TokenTransfer,
                "ethereum",
                payload,
                5,
                create_test_metadata(),
            );
            assert!(result.is_ok(), "High volume message sending should succeed");
        }

        let pending_messages = bridge.get_pending_messages();
        assert_eq!(
            pending_messages.len(),
            1000,
            "Should have 1000 pending messages"
        );

        println!("✅ High message volume test passed");
    }

    #[test]
    fn test_multiple_target_chains() {
        let bridge = create_test_bridge();
        let payload = b"multi_chain_payload".to_vec();
        let target_chains = vec!["ethereum", "polkadot", "cosmos", "bitcoin"];

        for chain in target_chains {
            let result = bridge.send_message(
                CrossChainMessageType::VoteResult,
                chain,
                payload.clone(),
                5,
                create_test_metadata(),
            );
            assert!(result.is_ok(), "Multi-chain message sending should succeed");
        }

        let pending_messages = bridge.get_pending_messages();
        assert_eq!(pending_messages.len(), 4, "Should have 4 pending messages");

        println!("✅ Multiple target chains test passed");
    }

    #[test]
    fn test_concurrent_asset_locking() {
        let bridge = Arc::new(create_test_bridge());
        let mut handles = Vec::new();

        // Spawn multiple threads to lock assets concurrently
        for i in 0..10 {
            let bridge_clone = Arc::clone(&bridge);
            let handle = thread::spawn(move || {
                for j in 0..10 {
                    let asset_id = format!("concurrent_asset_{}_{}", i, j);
                    bridge_clone
                        .lock_asset(
                            &asset_id,
                            "token",
                            1000 + j as u64,
                            "ethereum",
                            b"conditions".to_vec(),
                        )
                        .unwrap();
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        let locked_assets = bridge.get_locked_assets();
        assert_eq!(locked_assets.len(), 100, "Should have 100 locked assets");

        println!("✅ Concurrent asset locking test passed");
    }

    #[test]
    fn test_memory_stress() {
        let bridge = create_test_bridge();

        // Create large payloads to test memory handling
        for i in 0..100 {
            let large_payload = vec![i as u8; 10000]; // 10KB per payload
            let result = bridge.send_message(
                CrossChainMessageType::ShardSync,
                "ethereum",
                large_payload,
                5,
                create_test_metadata(),
            );
            assert!(result.is_ok(), "Large payload handling should succeed");
        }

        let pending_messages = bridge.get_pending_messages();
        assert_eq!(pending_messages.len(), 100, "Should handle large payloads");

        println!("✅ Memory stress test passed");
    }

    // ===== INTEGRATION TESTS =====

    #[test]
    fn test_vote_results_integration() {
        let bridge = create_test_bridge();
        let vote_results = b"vote_results_data".to_vec();
        let target_chains = vec!["ethereum", "polkadot"];

        let result = send_vote_results_to_chains(&bridge, &vote_results, target_chains);
        assert!(result.is_ok(), "Vote results integration should succeed");

        let message_ids = result.unwrap();
        assert_eq!(message_ids.len(), 2, "Should have 2 message IDs");

        println!("✅ Vote results integration test passed");
    }

    #[test]
    fn test_token_balances_integration() {
        let bridge = create_test_bridge();
        let token_balances = b"token_balances_data".to_vec();
        let target_chains = vec!["ethereum", "cosmos"];

        let result = send_token_balances_to_chains(&bridge, &token_balances, target_chains);
        assert!(result.is_ok(), "Token balances integration should succeed");

        let message_ids = result.unwrap();
        assert_eq!(message_ids.len(), 2, "Should have 2 message IDs");

        println!("✅ Token balances integration test passed");
    }

    #[test]
    fn test_validator_updates_integration() {
        let bridge = create_test_bridge();
        let validator_set = b"validator_set_data".to_vec();
        let target_chains = vec!["polkadot", "bitcoin"];

        let result = send_validator_updates_to_chains(&bridge, &validator_set, target_chains);
        assert!(
            result.is_ok(),
            "Validator updates integration should succeed"
        );

        let message_ids = result.unwrap();
        assert_eq!(message_ids.len(), 2, "Should have 2 message IDs");

        println!("✅ Validator updates integration test passed");
    }

    #[test]
    fn test_shard_state_integration() {
        let bridge = create_test_bridge();
        let shard_state = b"shard_state_data".to_vec();
        let target_chains = vec!["ethereum", "polkadot", "cosmos"];

        let result = send_shard_state_to_chains(&bridge, &shard_state, target_chains);
        assert!(result.is_ok(), "Shard state integration should succeed");

        let message_ids = result.unwrap();
        assert_eq!(message_ids.len(), 3, "Should have 3 message IDs");

        println!("✅ Shard state integration test passed");
    }

    #[test]
    fn test_comprehensive_workflow() {
        let bridge = create_test_bridge();

        // Test complete cross-chain workflow
        // 1. Lock assets
        bridge
            .lock_asset(
                "workflow_asset",
                "token",
                5000,
                "ethereum",
                b"workflow_conditions".to_vec(),
            )
            .unwrap();

        // 2. Send various message types
        let message_types = [
            CrossChainMessageType::VoteResult,
            CrossChainMessageType::TokenTransfer,
            CrossChainMessageType::ValidatorUpdate,
            CrossChainMessageType::ShardSync,
        ];

        for (i, message_type) in message_types.iter().enumerate() {
            let payload = format!("workflow_payload_{}", i).into_bytes();
            bridge
                .send_message(
                    message_type.clone(),
                    "ethereum",
                    payload,
                    5,
                    create_test_metadata(),
                )
                .unwrap();
        }

        // 3. Create state commitment
        bridge
            .create_state_commitment("ethereum", b"workflow_state")
            .unwrap();

        // 4. Verify results
        let pending_messages = bridge.get_pending_messages();
        assert_eq!(
            pending_messages.len(),
            5,
            "Should have 5 pending messages (4 workflow + 1 asset lock)"
        );

        let locked_assets = bridge.get_locked_assets();
        assert_eq!(locked_assets.len(), 1, "Should have 1 locked asset");

        // 5. Unlock asset
        let unlock_proof = vec![0u8; 32]; // 32-byte proof to satisfy verification
        bridge.unlock_asset("workflow_asset", unlock_proof).unwrap();

        println!("✅ Comprehensive workflow test passed");
    }

    #[test]
    fn test_bridge_configuration() {
        let custom_config = BridgeConfig {
            max_queue_size: 5000,
            message_timeout: 1800, // 30 minutes
            max_retries: 5,
            enable_encryption: false,
            enable_proof_verification: true,
            supported_chains: vec!["custom_chain".to_string()],
            bridge_fee: 2000,
        };

        let bridge = CrossChainBridge::with_config(custom_config);
        let status = bridge.get_bridge_status();
        assert!(
            status.contains("custom_chain"),
            "Should support custom chain"
        );

        println!("✅ Bridge configuration test passed");
    }
}
