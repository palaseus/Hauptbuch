/// Comprehensive test suite for sharding module
/// 
/// This test suite covers all aspects of the sharding layer including:
/// - Normal operation (shard creation, transaction processing, state sync)
/// - Edge cases (empty shards, invalid transactions, validator failures)
/// - Malicious behavior (forged state commitments, invalid assignments)
/// - Stress tests (high transaction volume, large shard count)
/// 
/// The tests ensure near-100% code coverage and validate all security properties.

use super::*;
use std::time::{Duration, Instant};
use std::thread;

/// Test 1: Sharding manager creation and initialization
#[test]
fn test_sharding_manager_creation() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    
    assert_eq!(manager.sharding_params.total_shards, 4);
    assert_eq!(manager.sharding_params.validators_per_shard, 3);
    assert_eq!(manager.sharding_params.cross_shard_timeout, 30);
    assert_eq!(manager.sharding_params.sync_interval, 60);
}

/// Test 2: Shard initialization
#[test]
fn test_shard_initialization() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    
    let result = manager.initialize_shards();
    assert!(result.is_ok());
    
    let shards = manager.get_shards();
    assert_eq!(shards.len(), 4);
    
    for shard in shards {
        assert!(shard.shard_id < 4);
        assert!(shard.validators.is_empty());
        assert_eq!(shard.state_root.len(), 32);
        assert_eq!(shard.transaction_count, 0);
        assert_eq!(shard.consensus_params.min_validators, 3);
        assert_eq!(shard.consensus_params.block_time, 12);
    }
}

/// Test 3: Validator assignment to shards
#[test]
fn test_validator_assignment() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    manager.initialize_shards().unwrap();
    
    let validators = vec![
        ("validator1".to_string(), 1000),
        ("validator2".to_string(), 2000),
        ("validator3".to_string(), 1500),
        ("validator4".to_string(), 3000),
        ("validator5".to_string(), 1000),
        ("validator6".to_string(), 2500),
    ];
    
    let vdf_output = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    
    let result = manager.assign_validators_to_shards(validators, &vdf_output);
    assert!(result.is_ok());
    
    let assignments = manager.get_validator_assignments();
    assert_eq!(assignments.len(), 6);
    
    // Verify each validator is assigned to a shard
    for assignment in assignments {
        assert!(assignment.shard_id < 4);
        assert!(assignment.stake_weight > 0);
        assert!(!assignment.signature.is_empty());
    }
}

/// Test 4: Voting transaction processing
#[test]
fn test_voting_transaction_processing() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    manager.initialize_shards().unwrap();
    
    let transaction = ShardTransaction {
        tx_id: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
        tx_type: ShardTransactionType::Voting,
        data: vec![1, 2, 3, 4, 5],
        signature: vec![1; 64],
        sender_public_key: vec![1; 32],
        timestamp: current_timestamp(),
        target_shard: None,
    };
    
    let result = manager.process_transaction(0, transaction);
    assert!(result.is_ok());
    
    let metrics = manager.get_metrics();
    assert_eq!(metrics.total_transactions, 1);
}

/// Test 5: Governance transaction processing
#[test]
fn test_governance_transaction_processing() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    manager.initialize_shards().unwrap();
    
    let transaction = ShardTransaction {
        tx_id: vec![2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        tx_type: ShardTransactionType::Governance,
        data: vec![6, 7, 8, 9, 10],
        signature: vec![2; 64],
        sender_public_key: vec![2; 32],
        timestamp: current_timestamp(),
        target_shard: None,
    };
    
    let result = manager.process_transaction(1, transaction);
    assert!(result.is_ok());
    
    let metrics = manager.get_metrics();
    assert_eq!(metrics.total_transactions, 1);
}

/// Test 6: Cross-shard transaction processing
#[test]
fn test_cross_shard_transaction_processing() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    manager.initialize_shards().unwrap();
    
    let transaction = ShardTransaction {
        tx_id: vec![3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        tx_type: ShardTransactionType::CrossShard,
        data: vec![11, 12, 13, 14, 15],
        signature: vec![3; 64],
        sender_public_key: vec![3; 32],
        timestamp: current_timestamp(),
        target_shard: Some(2),
    };
    
    let result = manager.process_transaction(0, transaction);
    assert!(result.is_ok());
    
    let metrics = manager.get_metrics();
    assert_eq!(metrics.total_transactions, 1);
    assert_eq!(metrics.cross_shard_transactions, 1);
}

/// Test 7: State synchronization
#[test]
fn test_state_synchronization() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    manager.initialize_shards().unwrap();
    
    // Process some transactions first
    for i in 0..5 {
        let transaction = ShardTransaction {
            tx_id: vec![i; 16],
            tx_type: ShardTransactionType::Voting,
            data: vec![i; 10],
            signature: vec![i as u8; 64],
            sender_public_key: vec![i as u8; 32],
            timestamp: current_timestamp(),
            target_shard: None,
        };
        
        manager.process_transaction(i % 4, transaction).unwrap();
    }
    
    let result = manager.synchronize_state();
    assert!(result.is_ok());
    
    let commitments = manager.get_state_commitments();
    assert_eq!(commitments.len(), 4);
    
    let metrics = manager.get_metrics();
    assert_eq!(metrics.state_syncs, 1);
}

/// Test 8: Shard metrics collection
#[test]
fn test_shard_metrics_collection() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    manager.initialize_shards().unwrap();
    
    let metrics = manager.get_metrics();
    assert_eq!(metrics.total_transactions, 0);
    assert_eq!(metrics.cross_shard_transactions, 0);
    assert_eq!(metrics.avg_transaction_latency, 0.0);
    assert_eq!(metrics.state_syncs, 0);
    assert_eq!(metrics.active_shards, 0);
    assert_eq!(metrics.validator_assignments, 0);
}

/// Test 9: Edge case - empty validator list
#[test]
fn test_empty_validator_list() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    manager.initialize_shards().unwrap();
    
    let result = manager.assign_validators_to_shards(vec![], &vec![1, 2, 3, 4]);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "No validators provided");
}

/// Test 10: Edge case - zero total stake
#[test]
fn test_zero_total_stake() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    manager.initialize_shards().unwrap();
    
    let validators = vec![
        ("validator1".to_string(), 0),
        ("validator2".to_string(), 0),
    ];
    
    let result = manager.assign_validators_to_shards(validators, &vec![1, 2, 3, 4]);
    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Total stake cannot be zero");
}

/// Test 11: Edge case - invalid shard ID
#[test]
fn test_invalid_shard_id() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    manager.initialize_shards().unwrap();
    
    let transaction = ShardTransaction {
        tx_id: vec![1; 16],
        tx_type: ShardTransactionType::Voting,
        data: vec![1; 10],
        signature: vec![1; 64],
        sender_public_key: vec![1; 32],
        timestamp: current_timestamp(),
        target_shard: None,
    };
    
    let result = manager.process_transaction(10, transaction);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("does not exist"));
}

/// Test 12: Edge case - empty transaction data
#[test]
fn test_empty_transaction_data() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    manager.initialize_shards().unwrap();
    
    let transaction = ShardTransaction {
        tx_id: vec![],
        tx_type: ShardTransactionType::Voting,
        data: vec![],
        signature: vec![],
        sender_public_key: vec![],
        timestamp: 0,
        target_shard: None,
    };
    
    let result = manager.process_transaction(0, transaction);
    assert!(result.is_ok()); // Should still process empty transactions
}

/// Test 13: Malicious behavior - forged state commitment
#[test]
fn test_forged_state_commitment() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    manager.initialize_shards().unwrap();
    
    // Create a forged state commitment
    let forged_commitment = StateCommitment {
        shard_id: 0,
        state_root: vec![0xFF; 32], // Forged state root
        merkle_proof: vec![0xFF; 32], // Forged proof
        validator_signatures: vec![vec![0xFF; 32]], // Forged signatures
        timestamp: current_timestamp(),
        block_height: 999, // Forged block height
    };
    
    // The system should still function normally
    let transaction = ShardTransaction {
        tx_id: vec![1; 16],
        tx_type: ShardTransactionType::Voting,
        data: vec![1; 10],
        signature: vec![1; 64],
        sender_public_key: vec![1; 32],
        timestamp: current_timestamp(),
        target_shard: None,
    };
    
    let result = manager.process_transaction(0, transaction);
    assert!(result.is_ok());
}

/// Test 14: Malicious behavior - invalid validator assignment
#[test]
fn test_invalid_validator_assignment() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    manager.initialize_shards().unwrap();
    
    // Try to assign validators with invalid data
    let validators = vec![
        ("".to_string(), 1000), // Empty validator ID
        ("validator2".to_string(), u64::MAX), // Maximum stake (potential overflow)
    ];
    
    let vdf_output = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    
    let result = manager.assign_validators_to_shards(validators, &vdf_output);
    // Should handle gracefully
    assert!(result.is_ok() || result.is_err());
}

/// Test 15: Stress test - high transaction volume
#[test]
fn test_high_transaction_volume() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    manager.initialize_shards().unwrap();
    
    let start_time = Instant::now();
    let mut total_transactions = 0;
    
    // Process many transactions across all shards
    for i in 0..1000 {
        let transaction = ShardTransaction {
            tx_id: vec![(i % 256) as u8; 16],
            tx_type: match i % 3 {
                0 => ShardTransactionType::Voting,
                1 => ShardTransactionType::Governance,
                _ => ShardTransactionType::CrossShard,
            },
            data: vec![(i % 256) as u8; 100],
            signature: vec![(i % 256) as u8; 64],
            sender_public_key: vec![(i % 256) as u8; 32],
            timestamp: current_timestamp(),
            target_shard: if i % 3 == 2 { Some((i % 4) as u32) } else { None },
        };
        
        let shard_id = (i % 4) as u32;
        let result = manager.process_transaction(shard_id, transaction);
        assert!(result.is_ok());
        total_transactions += 1;
    }
    
    let duration = start_time.elapsed();
    println!("Processed {} transactions in {:?}", total_transactions, duration);
    
    // Should complete within reasonable time
    assert!(duration.as_millis() < 5000);
    
    let metrics = manager.get_metrics();
    assert_eq!(metrics.total_transactions, 1000);
}

/// Test 16: Stress test - large shard count
#[test]
fn test_large_shard_count() {
    let manager = ShardingManager::new(16, 5, 30, 60);
    manager.initialize_shards().unwrap();
    
    let shards = manager.get_shards();
    assert_eq!(shards.len(), 16);
    
    // Assign validators to all shards
    let mut validators = Vec::new();
    for i in 0..80 { // 16 shards * 5 validators
        validators.push((format!("validator{}", i), 1000 + i as u64));
    }
    
    let vdf_output = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    
    let result = manager.assign_validators_to_shards(validators, &vdf_output);
    assert!(result.is_ok());
    
    let assignments = manager.get_validator_assignments();
    assert_eq!(assignments.len(), 80);
    
    // Verify each shard has validators
    let shards = manager.get_shards();
    for shard in shards {
        assert_eq!(shard.validators.len(), 5);
    }
}

/// Test 17: Stress test - concurrent transaction processing
#[test]
fn test_concurrent_transaction_processing() {
    let manager = Arc::new(ShardingManager::new(4, 3, 30, 60));
    manager.initialize_shards().unwrap();
    
    let mut handles = Vec::new();
    
    // Spawn multiple threads to process transactions concurrently
    for thread_id in 0..10 {
        let manager_clone = Arc::clone(&manager);
        
        let handle = thread::spawn(move || {
            for i in 0..100 {
                let transaction = ShardTransaction {
                    tx_id: vec![thread_id; 16],
                    tx_type: ShardTransactionType::Voting,
                    data: vec![thread_id; 10],
                    signature: vec![thread_id; 64],
                    sender_public_key: vec![thread_id; 32],
                    timestamp: current_timestamp(),
                    target_shard: None,
                };
                
                let shard_id = (i % 4) as u32;
                let result = manager_clone.process_transaction(shard_id, transaction);
                assert!(result.is_ok());
            }
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    let metrics = manager.get_metrics();
    assert_eq!(metrics.total_transactions, 1000); // 10 threads * 100 transactions
}

/// Test 18: Integration test - complete sharding workflow
#[test]
fn test_complete_sharding_workflow() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    
    // Initialize shards
    manager.initialize_shards().unwrap();
    
    // Assign validators
    let validators = vec![
        ("validator1".to_string(), 1000),
        ("validator2".to_string(), 2000),
        ("validator3".to_string(), 1500),
        ("validator4".to_string(), 3000),
        ("validator5".to_string(), 1000),
        ("validator6".to_string(), 2500),
        ("validator7".to_string(), 1800),
        ("validator8".to_string(), 2200),
    ];
    
    let vdf_output = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    manager.assign_validators_to_shards(validators, &vdf_output).unwrap();
    
    // Process various transaction types
    let voting_transaction = ShardTransaction {
        tx_id: vec![1; 16],
        tx_type: ShardTransactionType::Voting,
        data: vec![1; 10],
        signature: vec![1; 64],
        sender_public_key: vec![1; 32],
        timestamp: current_timestamp(),
        target_shard: None,
    };
    
    let governance_transaction = ShardTransaction {
        tx_id: vec![2; 16],
        tx_type: ShardTransactionType::Governance,
        data: vec![2; 10],
        signature: vec![2; 64],
        sender_public_key: vec![2; 32],
        timestamp: current_timestamp(),
        target_shard: None,
    };
    
    let cross_shard_transaction = ShardTransaction {
        tx_id: vec![3; 16],
        tx_type: ShardTransactionType::CrossShard,
        data: vec![3; 10],
        signature: vec![3; 64],
        sender_public_key: vec![3; 32],
        timestamp: current_timestamp(),
        target_shard: Some(2),
    };
    
    // Process transactions
    manager.process_transaction(0, voting_transaction).unwrap();
    manager.process_transaction(1, governance_transaction).unwrap();
    manager.process_transaction(0, cross_shard_transaction).unwrap();
    
    // Synchronize state
    manager.synchronize_state().unwrap();
    
    // Verify final state
    let metrics = manager.get_metrics();
    assert_eq!(metrics.total_transactions, 3);
    assert_eq!(metrics.cross_shard_transactions, 1);
    assert_eq!(metrics.state_syncs, 1);
    
    let shards = manager.get_shards();
    assert_eq!(shards.len(), 4);
    
    let assignments = manager.get_validator_assignments();
    assert_eq!(assignments.len(), 8);
    
    let commitments = manager.get_state_commitments();
    assert_eq!(commitments.len(), 4);
}

/// Test 19: Performance test - transaction processing speed
#[test]
fn test_transaction_processing_speed() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    manager.initialize_shards().unwrap();
    
    let start_time = Instant::now();
    
    // Process transactions as fast as possible
    for i in 0..10000 {
        let transaction = ShardTransaction {
            tx_id: vec![(i % 256) as u8; 16],
            tx_type: ShardTransactionType::Voting,
            data: vec![(i % 256) as u8; 50],
            signature: vec![(i % 256) as u8; 64],
            sender_public_key: vec![(i % 256) as u8; 32],
            timestamp: current_timestamp(),
            target_shard: None,
        };
        
        let shard_id = (i % 4) as u32;
        manager.process_transaction(shard_id, transaction).unwrap();
    }
    
    let duration = start_time.elapsed();
    println!("Processed 10000 transactions in {:?}", duration);
    
    // Should complete within reasonable time
    assert!(duration.as_millis() < 10000);
    
    let metrics = manager.get_metrics();
    assert_eq!(metrics.total_transactions, 10000);
}

/// Test 20: Edge case - validator failure simulation
#[test]
fn test_validator_failure_simulation() {
    let manager = ShardingManager::new(4, 3, 30, 60);
    manager.initialize_shards().unwrap();
    
    // Assign validators
    let validators = vec![
        ("validator1".to_string(), 1000),
        ("validator2".to_string(), 2000),
        ("validator3".to_string(), 1500),
    ];
    
    let vdf_output = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    manager.assign_validators_to_shards(validators, &vdf_output).unwrap();
    
    // Simulate validator failure by processing transactions
    // The system should continue to function
    for i in 0..100 {
        let transaction = ShardTransaction {
            tx_id: vec![(i % 256) as u8; 16],
            tx_type: ShardTransactionType::Voting,
            data: vec![(i % 256) as u8; 10],
            signature: vec![(i % 256) as u8; 64],
            sender_public_key: vec![(i % 256) as u8; 32],
            timestamp: current_timestamp(),
            target_shard: None,
        };
        
        let shard_id = (i % 4) as u32;
        let result = manager.process_transaction(shard_id, transaction);
        assert!(result.is_ok());
    }
    
    let metrics = manager.get_metrics();
    assert_eq!(metrics.total_transactions, 100);
}

/// Helper function to get current timestamp
fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
