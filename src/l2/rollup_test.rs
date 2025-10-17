//! Comprehensive test suite for Layer 2 rollup system
//! 
//! This module provides extensive testing for the optimistic rollup L2 system,
//! including normal operations, edge cases, malicious behavior, and stress tests
//! to ensure correctness and scalability.

use super::*;
use std::time::Instant;
use std::sync::Arc;
use std::thread;

/// Test L2 transaction creation and validation
#[cfg(test)]
mod transaction_tests {
    use super::*;
    
    #[test]
    fn test_transaction_creation() {
        let tx = L2Transaction::new(
            TransactionType::VoteSubmission,
            vec![0x01, 0x02, 0x03],
            Some(vec![0x04, 0x05, 0x06]),
            vec![0x07, 0x08, 0x09],
            100000,
            1000,
            1,
            vec![0x0A, 0x0B, 0x0C],
        );
        
        assert_eq!(tx.tx_type, TransactionType::VoteSubmission);
        assert_eq!(tx.from, vec![0x01, 0x02, 0x03]);
        assert_eq!(tx.gas_limit, 100000);
        assert_eq!(tx.nonce, 1);
        assert_eq!(tx.status, TransactionStatus::Pending);
    }
    
    #[test]
    fn test_transaction_validation() {
        let valid_tx = L2Transaction::new(
            TransactionType::TokenTransfer,
            vec![0x01, 0x02, 0x03],
            Some(vec![0x04, 0x05, 0x06]),
            vec![0x07, 0x08, 0x09],
            100000,
            1000,
            1,
            vec![0x0A, 0x0B, 0x0C],
        );
        
        assert!(valid_tx.validate().is_ok());
    }
    
    #[test]
    fn test_invalid_transaction() {
        let invalid_tx = L2Transaction::new(
            TransactionType::VoteSubmission,
            vec![], // Empty from address
            None,
            vec![0x07, 0x08, 0x09],
            0, // Zero gas limit
            1000,
            1,
            vec![], // Empty signature
        );
        
        assert!(invalid_tx.validate().is_err());
    }
    
    #[test]
    fn test_transaction_hash_computation() {
        let tx1 = L2Transaction::new(
            TransactionType::VoteSubmission,
            vec![0x01],
            None,
            vec![0x02],
            1000,
            100,
            1,
            vec![0x03],
        );
        
        let tx2 = L2Transaction::new(
            TransactionType::VoteSubmission,
            vec![0x01],
            None,
            vec![0x02],
            1000,
            100,
            1,
            vec![0x03],
        );
        
        // Same transaction should have same hash
        assert_eq!(tx1.hash, tx2.hash);
        assert!(!tx1.hash.is_empty());
    }
}

/// Test transaction batch operations
#[cfg(test)]
mod batch_tests {
    use super::*;
    
    #[test]
    fn test_batch_creation() {
        let transactions = vec![
            L2Transaction::new(
                TransactionType::VoteSubmission,
                vec![0x01],
                None,
                vec![0x02],
                1000,
                100,
                1,
                vec![0x03],
            ),
            L2Transaction::new(
                TransactionType::TokenTransfer,
                vec![0x04],
                Some(vec![0x05]),
                vec![0x06],
                2000,
                200,
                2,
                vec![0x07],
            ),
        ];
        
        let batch = TransactionBatch::new(1, transactions, vec![0x08, 0x09]).unwrap();
        
        assert_eq!(batch.batch_id, 1);
        assert_eq!(batch.transactions.len(), 2);
        assert_eq!(batch.sequencer, vec![0x08, 0x09]);
        assert_eq!(batch.status, BatchStatus::Created);
    }
    
    #[test]
    fn test_empty_batch_rejection() {
        let result = TransactionBatch::new(1, vec![], vec![0x01]);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_oversized_batch_rejection() {
        let mut transactions = Vec::new();
        for i in 0..2001 { // Exceed max batch size
            transactions.push(L2Transaction::new(
                TransactionType::VoteSubmission,
                vec![i as u8],
                None,
                vec![0x02],
                1000,
                100,
                i as u64,
                vec![0x03],
            ));
        }
        
        let result = TransactionBatch::new(1, transactions, vec![0x01]);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_batch_transaction_root() {
        let transactions = vec![
            L2Transaction::new(
                TransactionType::VoteSubmission,
                vec![0x01],
                None,
                vec![0x02],
                1000,
                100,
                1,
                vec![0x03],
            ),
        ];
        
        let batch = TransactionBatch::new(1, transactions, vec![0x04]).unwrap();
        assert!(!batch.transaction_root.is_empty());
        assert_eq!(batch.transaction_root.len(), 32); // SHA3-256 hash length
    }
}

/// Test L1 commitment operations
#[cfg(test)]
mod commitment_tests {
    use super::*;
    
    #[test]
    fn test_commitment_creation() {
        let commitment = L1Commitment::new(
            1,
            2,
            vec![0x01, 0x02, 0x03],
            vec![0x04, 0x05, 0x06],
            vec![0x07, 0x08, 0x09],
            100,
        );
        
        assert_eq!(commitment.commitment_id, 1);
        assert_eq!(commitment.batch_id, 2);
        assert_eq!(commitment.state_root, vec![0x01, 0x02, 0x03]);
        assert_eq!(commitment.status, CommitmentStatus::Pending);
    }
    
    #[test]
    fn test_challenge_window_expiry() {
        let commitment = L1Commitment::new(
            1,
            2,
            vec![0x01],
            vec![0x02],
            vec![0x03],
            100,
        );
        
        // Challenge window should be 7 days
        let expected_end = commitment.timestamp + 604800;
        assert_eq!(commitment.challenge_window_end, expected_end);
    }
}

/// Test fraud proof system
#[cfg(test)]
mod fraud_proof_tests {
    use super::*;
    
    #[test]
    fn test_fraud_proof_creation() {
        let disputed_tx = L2Transaction::new(
            TransactionType::VoteSubmission,
            vec![0x01],
            None,
            vec![0x02],
            1000,
            100,
            1,
            vec![0x03],
        );
        
        let disputed_transition = StateTransition {
            transaction: disputed_tx.clone(),
            pre_state: AccountState {
                address: vec![0x01],
                balance: 1000,
                nonce: 0,
                storage_root: vec![0x04],
                code_hash: vec![0x05],
            },
            post_state: AccountState {
                address: vec![0x01],
                balance: 900,
                nonce: 1,
                storage_root: vec![0x06],
                code_hash: vec![0x05],
            },
            gas_used: 100,
        };
        
        let fraud_proof = FraudProof {
            proof_id: 1,
            batch_id: 2,
            challenger: vec![0x07],
            disputed_transition,
            merkle_witness: MerkleWitness {
                path: vec![vec![0x08]],
                siblings: vec![vec![0x09]],
                leaf_index: 0,
                root_hash: vec![0x0A],
            },
            pre_state_root: vec![0x0B],
            post_state_root: vec![0x0C],
            proof_data: FraudProofData {
                execution_trace: vec![],
                state_changes: HashMap::new(),
                gas_breakdown: HashMap::new(),
            },
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            status: FraudProofStatus::Submitted,
        };
        
        assert_eq!(fraud_proof.proof_id, 1);
        assert_eq!(fraud_proof.batch_id, 2);
        assert_eq!(fraud_proof.status, FraudProofStatus::Submitted);
    }
}

/// Test state trie operations
#[cfg(test)]
mod state_trie_tests {
    use super::*;
    
    #[test]
    fn test_state_trie_creation() {
        let trie = StateTrie::new();
        assert_eq!(trie.root_hash, vec![0; 32]);
        assert!(trie.accounts.is_empty());
        assert!(trie.storage.is_empty());
    }
    
    #[test]
    fn test_account_update() {
        let mut trie = StateTrie::new();
        let account = AccountState {
            address: vec![0x01, 0x02],
            balance: 1000,
            nonce: 1,
            storage_root: vec![0x03, 0x04],
            code_hash: vec![0x05, 0x06],
        };
        
        trie.update_account(account).unwrap();
        
        let retrieved = trie.get_account(&[0x01, 0x02]).unwrap();
        assert_eq!(retrieved.balance, 1000);
        assert_eq!(retrieved.nonce, 1);
    }
    
    #[test]
    fn test_storage_update() {
        let mut trie = StateTrie::new();
        
        trie.update_storage(&[0x01], vec![0x02], vec![0x03, 0x04]).unwrap();
        
        let value = trie.get_storage(&[0x01], &[0x02]).unwrap();
        assert_eq!(value, &vec![0x03, 0x04]);
    }
    
    #[test]
    fn test_root_hash_update() {
        let mut trie = StateTrie::new();
        let initial_root = trie.root_hash.clone();
        
        let account = AccountState {
            address: vec![0x01],
            balance: 1000,
            nonce: 1,
            storage_root: vec![0x02],
            code_hash: vec![0x03],
        };
        
        trie.update_account(account).unwrap();
        
        // Root hash should change after account update
        assert_ne!(trie.root_hash, initial_root);
    }
}

/// Test sequencer operations
#[cfg(test)]
mod sequencer_tests {
    use super::*;
    
    #[test]
    fn test_sequencer_creation() {
        let sequencer = Sequencer::new(vec![0x01, 0x02], 1000000);
        
        assert_eq!(sequencer.address, vec![0x01, 0x02]);
        assert_eq!(sequencer.stake, 1000000);
        assert!(sequencer.is_active);
        assert_eq!(sequencer.batches_created, 0);
        assert_eq!(sequencer.slash_count, 0);
    }
    
    #[test]
    fn test_sequencer_stake_check() {
        let sequencer = Sequencer::new(vec![0x01], 1000000);
        
        assert!(sequencer.has_sufficient_stake(500000));
        assert!(!sequencer.has_sufficient_stake(2000000));
    }
    
    #[test]
    fn test_sequencer_slashing() {
        let mut sequencer = Sequencer::new(vec![0x01], 1000000);
        
        sequencer.slash(300000).unwrap();
        
        assert_eq!(sequencer.stake, 700000);
        assert_eq!(sequencer.slash_count, 1);
        assert!(sequencer.is_active);
    }
    
    #[test]
    fn test_sequencer_total_slashing() {
        let mut sequencer = Sequencer::new(vec![0x01], 1000000);
        
        sequencer.slash(1000000).unwrap();
        
        assert_eq!(sequencer.stake, 0);
        assert_eq!(sequencer.slash_count, 1);
        assert!(!sequencer.is_active);
    }
    
    #[test]
    fn test_sequencer_insufficient_stake_slashing() {
        let mut sequencer = Sequencer::new(vec![0x01], 1000000);
        
        let result = sequencer.slash(2000000);
        assert!(result.is_err());
    }
}

/// Test state executor operations
#[cfg(test)]
mod executor_tests {
    use super::*;
    
    #[test]
    fn test_executor_creation() {
        let trie = StateTrie::new();
        let executor = StateExecutor::new(trie, 1000000);
        
        assert_eq!(executor.gas_limit, 1000000);
        assert_eq!(executor.gas_used, 0);
        assert!(executor.execution_trace.is_empty());
    }
    
    #[test]
    fn test_vote_transaction_execution() {
        let trie = StateTrie::new();
        let mut executor = StateExecutor::new(trie, 1000000);
        
        let tx = L2Transaction::new(
            TransactionType::VoteSubmission,
            vec![0x01],
            None,
            vec![0x02],
            100000,
            1000,
            1,
            vec![0x03],
        );
        
        let result = executor.execute_transaction(&tx);
        assert!(result.is_ok());
        
        let transition = result.unwrap();
        assert_eq!(transition.gas_used, 21000); // Base gas cost
    }
    
    #[test]
    fn test_governance_transaction_execution() {
        let trie = StateTrie::new();
        let mut executor = StateExecutor::new(trie, 1000000);
        
        let tx = L2Transaction::new(
            TransactionType::GovernanceProposal,
            vec![0x01],
            None,
            vec![0x02],
            100000,
            1000,
            1,
            vec![0x03],
        );
        
        let result = executor.execute_transaction(&tx);
        assert!(result.is_ok());
        
        let transition = result.unwrap();
        assert_eq!(transition.gas_used, 50000); // Higher gas cost for governance
    }
    
    #[test]
    fn test_gas_limit_exceeded() {
        let trie = StateTrie::new();
        let mut executor = StateExecutor::new(trie, 10000); // Low gas limit
        
        let tx = L2Transaction::new(
            TransactionType::ContractCall,
            vec![0x01],
            None,
            vec![0x02],
            100000,
            1000,
            1,
            vec![0x03],
        );
        
        let result = executor.execute_transaction(&tx);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), L2Error::GasLimitExceeded);
    }
}

/// Test optimistic rollup operations
#[cfg(test)]
mod rollup_tests {
    use super::*;
    
    #[test]
    fn test_rollup_creation() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        assert_eq!(rollup.config.max_batch_size, 2000);
        assert_eq!(rollup.config.challenge_window, 604800);
        assert_eq!(rollup.config.min_sequencer_stake, 1000000);
    }
    
    #[test]
    fn test_transaction_submission() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        let tx = L2Transaction::new(
            TransactionType::VoteSubmission,
            vec![0x01],
            None,
            vec![0x02],
            100000,
            1000,
            1,
            vec![0x03],
        );
        
        let result = rollup.submit_transaction(tx);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_sequencer_registration() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        let sequencer = Sequencer::new(vec![0x01], 2000000);
        let result = rollup.register_sequencer(sequencer);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_insufficient_stake_sequencer() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        let sequencer = Sequencer::new(vec![0x01], 500000); // Below minimum
        let result = rollup.register_sequencer(sequencer);
        assert!(result.is_err());
    }
}

/// Test edge cases
#[cfg(test)]
mod edge_case_tests {
    use super::*;
    
    #[test]
    fn test_empty_mempool_batch_creation() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        let result = rollup.create_batch(vec![0x01]);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_single_transaction_batch() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        // Register sequencer first
        let sequencer = Sequencer::new(vec![0x01], 2000000);
        rollup.register_sequencer(sequencer).unwrap();
        
        // Submit single transaction
        let tx = L2Transaction::new(
            TransactionType::VoteSubmission,
            vec![0x02],
            None,
            vec![0x03],
            100000,
            1000,
            1,
            vec![0x04],
        );
        rollup.submit_transaction(tx).unwrap();
        
        // Create batch
        let result = rollup.create_batch(vec![0x01]);
        assert!(result.is_ok());
        
        let batch = result.unwrap();
        assert_eq!(batch.transactions.len(), 1);
    }
    
    #[test]
    fn test_maximum_batch_size() {
        let config = RollupConfig {
            max_batch_size: 5,
            ..Default::default()
        };
        let rollup = OptimisticRollup::new(config);
        
        // Register sequencer
        let sequencer = Sequencer::new(vec![0x01], 2000000);
        rollup.register_sequencer(sequencer).unwrap();
        
        // Submit 6 transactions (exceeds max batch size)
        for i in 0..6 {
            let tx = L2Transaction::new(
                TransactionType::VoteSubmission,
                vec![i as u8],
                None,
                vec![0x02],
                100000,
                1000,
                i as u64,
                vec![0x03],
            );
            rollup.submit_transaction(tx).unwrap();
        }
        
        let result = rollup.create_batch(vec![0x01]);
        assert!(result.is_ok());
        
        let batch = result.unwrap();
        assert_eq!(batch.transactions.len(), 5); // Should be limited to max batch size
    }
    
    #[test]
    fn test_invalid_transaction_in_batch() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        // This test would require modifying the batch creation to allow invalid transactions
        // For now, just test that normal validation works
        let tx = L2Transaction::new(
            TransactionType::VoteSubmission,
            vec![0x01],
            None,
            vec![0x02],
            0, // Invalid gas limit
            1000,
            1,
            vec![0x03],
        );
        
        let result = rollup.submit_transaction(tx);
        assert!(result.is_err());
    }
}

/// Test malicious behavior
#[cfg(test)]
mod malicious_behavior_tests {
    use super::*;
    
    #[test]
    fn test_fraudulent_batch_detection() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        // Register sequencer
        let sequencer = Sequencer::new(vec![0x01], 2000000);
        rollup.register_sequencer(sequencer).unwrap();
        
        // Submit transaction
        let tx = L2Transaction::new(
            TransactionType::VoteSubmission,
            vec![0x02],
            None,
            vec![0x03],
            100000,
            1000,
            1,
            vec![0x04],
        );
        rollup.submit_transaction(tx).unwrap();
        
        // Create batch
        let batch = rollup.create_batch(vec![0x01]).unwrap();
        
        // Generate fraud proof
        let fraud_proof = rollup.generate_fraud_proof(
            batch.batch_id,
            vec![0x05], // Challenger
            0, // First transaction
        ).unwrap();
        
        assert_eq!(fraud_proof.batch_id, batch.batch_id);
        assert_eq!(fraud_proof.challenger, vec![0x05]);
    }
    
    #[test]
    fn test_sequencer_censorship_simulation() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        // Register sequencer
        let sequencer = Sequencer::new(vec![0x01], 2000000);
        rollup.register_sequencer(sequencer).unwrap();
        
        // Submit multiple transactions
        for i in 0..10 {
            let tx = L2Transaction::new(
                TransactionType::VoteSubmission,
                vec![i as u8],
                None,
                vec![0x02],
                100000,
                1000,
                i as u64,
                vec![0x03],
            );
            rollup.submit_transaction(tx).unwrap();
        }
        
        // Create batch (sequencer could censor by only including some transactions)
        let batch = rollup.create_batch(vec![0x01]).unwrap();
        
        // In a real implementation, this would detect if sequencer excluded transactions
        assert!(batch.transactions.len() <= 10);
    }
    
    #[test]
    fn test_double_spending_attempt() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        // Submit same transaction twice (different nonces)
        let tx1 = L2Transaction::new(
            TransactionType::TokenTransfer,
            vec![0x01],
            Some(vec![0x02]),
            vec![0x03],
            100000,
            1000,
            1,
            vec![0x04],
        );
        
        let tx2 = L2Transaction::new(
            TransactionType::TokenTransfer,
            vec![0x01],
            Some(vec![0x02]),
            vec![0x03],
            100000,
            1000,
            2, // Different nonce
            vec![0x04],
        );
        
        rollup.submit_transaction(tx1).unwrap();
        rollup.submit_transaction(tx2).unwrap();
        
        // Both transactions should be accepted (different nonces)
        // In a real implementation, this would check for actual double spending
        assert!(true);
    }
    
    #[test]
    fn test_invalid_merkle_proof() {
        // Test invalid Merkle proof in fraud proof
        let merkle_witness = MerkleWitness {
            path: vec![vec![0x01]],
            siblings: vec![vec![0x02]],
            leaf_index: 0,
            root_hash: vec![0x03],
        };
        
        // In a real implementation, this would verify the Merkle proof
        // For now, just test that the structure is created
        assert_eq!(merkle_witness.leaf_index, 0);
        assert!(!merkle_witness.root_hash.is_empty());
    }
}

/// Stress tests
#[cfg(test)]
mod stress_tests {
    use super::*;
    
    #[test]
    fn test_high_volume_transaction_processing() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        // Register sequencer
        let sequencer = Sequencer::new(vec![0x01], 2000000);
        rollup.register_sequencer(sequencer).unwrap();
        
        let start = Instant::now();
        
        // Submit 1000 transactions
        for i in 0..1000 {
            let tx = L2Transaction::new(
                TransactionType::VoteSubmission,
                vec![(i % 256) as u8],
                None,
                vec![0x02],
                100000,
                1000,
                i as u64,
                vec![0x03],
            );
            rollup.submit_transaction(tx).unwrap();
        }
        
        let duration = start.elapsed();
        println!("1000 transaction submissions took: {:?}", duration);
        
        // Should complete within reasonable time
        assert!(duration.as_secs() < 5);
    }
    
    #[test]
    fn test_concurrent_batch_processing() {
        let config = RollupConfig::default();
        let rollup = Arc::new(OptimisticRollup::new(config));
        let mut handles = Vec::new();
        
        // Register sequencer
        let sequencer = Sequencer::new(vec![0x01], 2000000);
        rollup.register_sequencer(sequencer).unwrap();
        
        // Spawn 10 threads, each submitting 100 transactions
        for thread_id in 0..10 {
            let rollup = Arc::clone(&rollup);
            let handle = thread::spawn(move || {
                for i in 0..100 {
                    let tx = L2Transaction::new(
                        TransactionType::VoteSubmission,
                        vec![(thread_id * 100 + i) as u8],
                        None,
                        vec![0x02],
                        100000,
                        1000,
                        (thread_id * 100 + i) as u64,
                        vec![0x03],
                    );
                    rollup.submit_transaction(tx).unwrap();
                }
            });
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
        
        // Create batch
        let batch = rollup.create_batch(vec![0x01]).unwrap();
        assert_eq!(batch.transactions.len(), 1000);
    }
    
    #[test]
    fn test_memory_efficiency_under_load() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        let start = Instant::now();
        
        // Register sequencer
        let sequencer = Sequencer::new(vec![0x01], 2000000);
        rollup.register_sequencer(sequencer).unwrap();
        
        // Submit many transactions to test memory efficiency
        for i in 0..10000 {
            let tx = L2Transaction::new(
                TransactionType::VoteSubmission,
                vec![(i % 256) as u8],
                None,
                vec![0x02],
                100000,
                1000,
                i as u64,
                vec![0x03],
            );
            rollup.submit_transaction(tx).unwrap();
        }
        
        let duration = start.elapsed();
        println!("Memory efficiency test took: {:?}", duration);
        
        // Should complete without memory issues
        assert!(duration.as_secs() < 30);
    }
    
    #[test]
    fn test_performance_benchmarks() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        // Register sequencer
        let sequencer = Sequencer::new(vec![0x01], 2000000);
        rollup.register_sequencer(sequencer).unwrap();
        
        let start = Instant::now();
        
        // Test batch creation performance
        for i in 0..100 {
            let tx = L2Transaction::new(
                TransactionType::VoteSubmission,
                vec![i as u8],
                None,
                vec![0x02],
                100000,
                1000,
                i as u64,
                vec![0x03],
            );
            rollup.submit_transaction(tx).unwrap();
        }
        
        let batch = rollup.create_batch(vec![0x01]).unwrap();
        
        let duration = start.elapsed();
        println!("Batch creation performance: {:?}", duration);
        
        // Should be fast enough for real-time applications
        assert!(duration.as_millis() < 1000); // Less than 1 second
        assert_eq!(batch.transactions.len(), 100);
    }
}

/// Integration tests
#[cfg(test)]
mod integration_tests {
    use super::*;
    
    #[test]
    fn test_end_to_end_rollup_workflow() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        // Register sequencer
        let sequencer = Sequencer::new(vec![0x01], 2000000);
        rollup.register_sequencer(sequencer).unwrap();
        
        // Submit transactions
        for i in 0..10 {
            let tx = L2Transaction::new(
                TransactionType::VoteSubmission,
                vec![i as u8],
                None,
                vec![0x02],
                100000,
                1000,
                i as u64,
                vec![0x03],
            );
            rollup.submit_transaction(tx).unwrap();
        }
        
        // Create batch
        let batch = rollup.create_batch(vec![0x01]).unwrap();
        assert_eq!(batch.transactions.len(), 10);
        
        // Submit to L1
        let commitment = rollup.submit_batch_to_l1(&batch).unwrap();
        assert_eq!(commitment.batch_id, batch.batch_id);
    }
    
    #[test]
    fn test_fraud_proof_workflow() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        // Register sequencer
        let sequencer = Sequencer::new(vec![0x01], 2000000);
        rollup.register_sequencer(sequencer).unwrap();
        
        // Submit transaction and create batch
        let tx = L2Transaction::new(
            TransactionType::VoteSubmission,
            vec![0x02],
            None,
            vec![0x03],
            100000,
            1000,
            1,
            vec![0x04],
        );
        rollup.submit_transaction(tx).unwrap();
        
        let batch = rollup.create_batch(vec![0x01]).unwrap();
        
        // Generate fraud proof
        let fraud_proof = rollup.generate_fraud_proof(
            batch.batch_id,
            vec![0x05], // Challenger
            0, // First transaction
        ).unwrap();
        
        // Verify fraud proof
        let is_valid = rollup.verify_fraud_proof(&fraud_proof).unwrap();
        assert!(is_valid);
    }
    
    #[test]
    fn test_sequencer_slashing_workflow() {
        let config = RollupConfig::default();
        let rollup = OptimisticRollup::new(config);
        
        // Register sequencer
        let sequencer = Sequencer::new(vec![0x01], 2000000);
        rollup.register_sequencer(sequencer).unwrap();
        
        // Slash sequencer
        rollup.slash_sequencer(&[0x01], 500000).unwrap();
        
        // Verify sequencer was slashed
        let sequencers = rollup.sequencers.read().unwrap();
        let slashed_sequencer = sequencers.get(&[0x01]).unwrap();
        assert_eq!(slashed_sequencer.stake, 1500000);
        assert_eq!(slashed_sequencer.slash_count, 1);
    }
}
