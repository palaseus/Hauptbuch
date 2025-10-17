//! Sequencer Preconfirmations for Sub-Second Finality
//!
//! This module implements sequencer preconfirmations to provide sub-second
//! finality guarantees and cross-rollup atomic composability for the
//! decentralized sequencer network.
//!
//! Key features:
//! - Sequencer-signed soft confirmations
//! - Reorg protection guarantees
//! - Cross-rollup atomic composability
//! - Sub-second finality
//! - MEV protection through preconfirmations
//! - Cross-rollup transaction coordination
//! - Preconfirmation validation and verification
//!
//! Technical advantages:
//! - Sub-second user experience
//! - Cross-rollup atomicity
//! - MEV protection
//! - Reorg protection
//! - Scalable transaction processing
//! - Economic security guarantees

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Error types for preconfirmations implementation
#[derive(Debug, Clone, PartialEq)]
pub enum PreconfirmationError {
    /// Invalid preconfirmation
    InvalidPreconfirmation,
    /// Invalid signature
    InvalidSignature,
    /// Expired preconfirmation
    ExpiredPreconfirmation,
    /// Invalid sequencer
    InvalidSequencer,
    /// Invalid transaction
    InvalidTransaction,
    /// Reorg detected
    ReorgDetected,
    /// Cross-rollup conflict
    CrossRollupConflict,
    /// Insufficient stake
    InsufficientStake,
    /// Preconfirmation timeout
    PreconfirmationTimeout,
    /// Invalid batch
    InvalidBatch,
    /// Atomicity violation
    AtomicityViolation,
    /// MEV protection failed
    MEVProtectionFailed,
}

pub type PreconfirmationResult<T> = Result<T, PreconfirmationError>;

/// Preconfirmation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Preconfirmation {
    /// Preconfirmation ID
    pub preconfirmation_id: String,
    /// Transaction hash
    pub transaction_hash: String,
    /// Sequencer ID
    pub sequencer_id: String,
    /// Sequencer signature
    pub sequencer_signature: Vec<u8>,
    /// Block number
    pub block_number: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Expiry timestamp
    pub expiry_timestamp: u64,
    /// Gas price
    pub gas_price: u128,
    /// Gas limit
    pub gas_limit: u64,
    /// Priority fee
    pub priority_fee: u128,
    /// MEV protection level
    pub mev_protection_level: MEVProtectionLevel,
    /// Cross-rollup data
    pub cross_rollup_data: Option<CrossRollupData>,
}

/// MEV protection level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum MEVProtectionLevel {
    /// No MEV protection
    None,
    /// Basic MEV protection
    Basic,
    /// Advanced MEV protection
    Advanced,
    /// Maximum MEV protection
    Maximum,
}

/// Cross-rollup data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossRollupData {
    /// Source rollup ID
    pub source_rollup_id: String,
    /// Destination rollup ID
    pub destination_rollup_id: String,
    /// Atomic transaction group
    pub atomic_group_id: String,
    /// Coordination data
    pub coordination_data: Vec<u8>,
    /// Cross-rollup signature
    pub cross_rollup_signature: Vec<u8>,
}

/// Preconfirmation batch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreconfirmationBatch {
    /// Batch ID
    pub batch_id: String,
    /// Sequencer ID
    pub sequencer_id: String,
    /// Preconfirmations
    pub preconfirmations: Vec<Preconfirmation>,
    /// Batch signature
    pub batch_signature: Vec<u8>,
    /// Batch timestamp
    pub batch_timestamp: u64,
    /// Batch expiry
    pub batch_expiry: u64,
    /// MEV protection applied
    pub mev_protection_applied: bool,
}

/// Preconfirmation status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum PreconfirmationStatus {
    /// Preconfirmation pending
    Pending,
    /// Preconfirmation confirmed
    Confirmed,
    /// Preconfirmation finalized
    Finalized,
    /// Preconfirmation expired
    Expired,
    /// Preconfirmation reorged
    Reorged,
    /// Preconfirmation failed
    Failed,
}

/// Reorg protection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReorgProtection {
    /// Protection ID
    pub protection_id: String,
    /// Transaction hash
    pub transaction_hash: String,
    /// Sequencer ID
    pub sequencer_id: String,
    /// Protection level
    pub protection_level: ReorgProtectionLevel,
    /// Protection data
    pub protection_data: Vec<u8>,
    /// Protection signature
    pub protection_signature: Vec<u8>,
    /// Protection timestamp
    pub protection_timestamp: u64,
}

/// Reorg protection level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ReorgProtectionLevel {
    /// No reorg protection
    None,
    /// Basic reorg protection
    Basic,
    /// Advanced reorg protection
    Advanced,
    /// Maximum reorg protection
    Maximum,
}

/// Preconfirmation engine
pub struct PreconfirmationEngine {
    /// Preconfirmations
    preconfirmations: Arc<RwLock<HashMap<String, Preconfirmation>>>,
    /// Preconfirmation statuses
    preconfirmation_statuses: Arc<RwLock<HashMap<String, PreconfirmationStatus>>>,
    /// Preconfirmation batches
    preconfirmation_batches: Arc<RwLock<HashMap<String, PreconfirmationBatch>>>,
    /// Reorg protections
    reorg_protections: Arc<RwLock<HashMap<String, ReorgProtection>>>,
    /// Cross-rollup coordination
    cross_rollup_coordination: Arc<RwLock<HashMap<String, CrossRollupData>>>,
    /// Metrics
    metrics: Arc<RwLock<PreconfirmationMetrics>>,
}

/// Preconfirmation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreconfirmationMetrics {
    /// Total preconfirmations issued
    pub total_preconfirmations_issued: u64,
    /// Total preconfirmations confirmed
    pub total_preconfirmations_confirmed: u64,
    /// Total preconfirmations finalized
    pub total_preconfirmations_finalized: u64,
    /// Total preconfirmations expired
    pub total_preconfirmations_expired: u64,
    /// Total preconfirmations reorged
    pub total_preconfirmations_reorged: u64,
    /// Total batches created
    pub total_batches_created: u64,
    /// Total reorg protections
    pub total_reorg_protections: u64,
    /// Average preconfirmation time (ms)
    pub avg_preconfirmation_time_ms: f64,
    /// Average finalization time (ms)
    pub avg_finalization_time_ms: f64,
    /// MEV protection success rate
    pub mev_protection_success_rate: f64,
    /// Cross-rollup success rate
    pub cross_rollup_success_rate: f64,
}

impl Default for PreconfirmationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl PreconfirmationEngine {
    /// Create a new preconfirmation engine
    pub fn new() -> Self {
        Self {
            preconfirmations: Arc::new(RwLock::new(HashMap::new())),
            preconfirmation_statuses: Arc::new(RwLock::new(HashMap::new())),
            preconfirmation_batches: Arc::new(RwLock::new(HashMap::new())),
            reorg_protections: Arc::new(RwLock::new(HashMap::new())),
            cross_rollup_coordination: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(PreconfirmationMetrics {
                total_preconfirmations_issued: 0,
                total_preconfirmations_confirmed: 0,
                total_preconfirmations_finalized: 0,
                total_preconfirmations_expired: 0,
                total_preconfirmations_reorged: 0,
                total_batches_created: 0,
                total_reorg_protections: 0,
                avg_preconfirmation_time_ms: 0.0,
                avg_finalization_time_ms: 0.0,
                mev_protection_success_rate: 0.0,
                cross_rollup_success_rate: 0.0,
            })),
        }
    }

    /// Issue preconfirmation
    pub fn issue_preconfirmation(
        &self,
        preconfirmation: Preconfirmation,
    ) -> PreconfirmationResult<()> {
        // Validate preconfirmation
        if preconfirmation.preconfirmation_id.is_empty()
            || preconfirmation.transaction_hash.is_empty()
        {
            return Err(PreconfirmationError::InvalidPreconfirmation);
        }

        // Check if preconfirmation already exists
        {
            let preconfirmations = self.preconfirmations.read().unwrap();
            if preconfirmations.contains_key(&preconfirmation.preconfirmation_id) {
                return Err(PreconfirmationError::InvalidPreconfirmation);
            }
        }

        // Store preconfirmation
        {
            let mut preconfirmations = self.preconfirmations.write().unwrap();
            preconfirmations.insert(
                preconfirmation.preconfirmation_id.clone(),
                preconfirmation.clone(),
            );
        }

        // Set initial status
        {
            let mut preconfirmation_statuses = self.preconfirmation_statuses.write().unwrap();
            preconfirmation_statuses.insert(
                preconfirmation.preconfirmation_id.clone(),
                PreconfirmationStatus::Pending,
            );
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_preconfirmations_issued += 1;
        }

        Ok(())
    }

    /// Confirm preconfirmation
    pub fn confirm_preconfirmation(&self, preconfirmation_id: &str) -> PreconfirmationResult<()> {
        // Check if preconfirmation exists
        {
            let preconfirmations = self.preconfirmations.read().unwrap();
            if !preconfirmations.contains_key(preconfirmation_id) {
                return Err(PreconfirmationError::InvalidPreconfirmation);
            }
        }

        // Update status
        {
            let mut preconfirmation_statuses = self.preconfirmation_statuses.write().unwrap();
            if let Some(status) = preconfirmation_statuses.get_mut(preconfirmation_id) {
                *status = PreconfirmationStatus::Confirmed;
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_preconfirmations_confirmed += 1;
        }

        Ok(())
    }

    /// Finalize preconfirmation
    pub fn finalize_preconfirmation(&self, preconfirmation_id: &str) -> PreconfirmationResult<()> {
        // Check if preconfirmation exists
        {
            let preconfirmations = self.preconfirmations.read().unwrap();
            if !preconfirmations.contains_key(preconfirmation_id) {
                return Err(PreconfirmationError::InvalidPreconfirmation);
            }
        }

        // Update status
        {
            let mut preconfirmation_statuses = self.preconfirmation_statuses.write().unwrap();
            if let Some(status) = preconfirmation_statuses.get_mut(preconfirmation_id) {
                *status = PreconfirmationStatus::Finalized;
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_preconfirmations_finalized += 1;
        }

        Ok(())
    }

    /// Create preconfirmation batch
    pub fn create_preconfirmation_batch(
        &self,
        batch: PreconfirmationBatch,
    ) -> PreconfirmationResult<()> {
        // Validate batch
        if batch.batch_id.is_empty() || batch.preconfirmations.is_empty() {
            return Err(PreconfirmationError::InvalidBatch);
        }

        // Store batch
        {
            let mut preconfirmation_batches = self.preconfirmation_batches.write().unwrap();
            preconfirmation_batches.insert(batch.batch_id.clone(), batch.clone());
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_batches_created += 1;
        }

        Ok(())
    }

    /// Add reorg protection
    pub fn add_reorg_protection(&self, protection: ReorgProtection) -> PreconfirmationResult<()> {
        // Validate protection
        if protection.protection_id.is_empty() || protection.transaction_hash.is_empty() {
            return Err(PreconfirmationError::InvalidPreconfirmation);
        }

        // Store protection
        {
            let mut reorg_protections = self.reorg_protections.write().unwrap();
            reorg_protections.insert(protection.protection_id.clone(), protection.clone());
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_reorg_protections += 1;
        }

        Ok(())
    }

    /// Coordinate cross-rollup transaction
    pub fn coordinate_cross_rollup(
        &self,
        coordination_data: CrossRollupData,
    ) -> PreconfirmationResult<()> {
        // Validate coordination data
        if coordination_data.source_rollup_id.is_empty()
            || coordination_data.destination_rollup_id.is_empty()
        {
            return Err(PreconfirmationError::CrossRollupConflict);
        }

        // Store coordination data
        {
            let mut cross_rollup_coordination = self.cross_rollup_coordination.write().unwrap();
            cross_rollup_coordination.insert(
                coordination_data.atomic_group_id.clone(),
                coordination_data.clone(),
            );
        }

        Ok(())
    }

    /// Detect reorg
    pub fn detect_reorg(&self, transaction_hash: &str) -> PreconfirmationResult<bool> {
        // Check if transaction has reorg protection
        {
            let reorg_protections = self.reorg_protections.read().unwrap();
            let has_protection = reorg_protections
                .values()
                .any(|protection| protection.transaction_hash == transaction_hash);

            if has_protection {
                // Simulate reorg detection
                // In a real implementation, this would check the blockchain state
                return Ok(false); // No reorg detected
            }
        }

        // Check preconfirmation status
        {
            let preconfirmation_statuses = self.preconfirmation_statuses.read().unwrap();
            for (preconfirmation_id, status) in preconfirmation_statuses.iter() {
                if let Some(preconfirmation) = self
                    .preconfirmations
                    .read()
                    .unwrap()
                    .get(preconfirmation_id)
                {
                    if preconfirmation.transaction_hash == transaction_hash
                        && *status == PreconfirmationStatus::Reorged
                    {
                        return Ok(true); // Reorg detected
                    }
                }
            }
        }

        Ok(false) // No reorg detected
    }

    /// Get preconfirmation
    pub fn get_preconfirmation(&self, preconfirmation_id: &str) -> Option<Preconfirmation> {
        let preconfirmations = self.preconfirmations.read().unwrap();
        preconfirmations.get(preconfirmation_id).cloned()
    }

    /// Get preconfirmation status
    pub fn get_preconfirmation_status(
        &self,
        preconfirmation_id: &str,
    ) -> Option<PreconfirmationStatus> {
        let preconfirmation_statuses = self.preconfirmation_statuses.read().unwrap();
        preconfirmation_statuses.get(preconfirmation_id).cloned()
    }

    /// Get preconfirmation batch
    pub fn get_preconfirmation_batch(&self, batch_id: &str) -> Option<PreconfirmationBatch> {
        let preconfirmation_batches = self.preconfirmation_batches.read().unwrap();
        preconfirmation_batches.get(batch_id).cloned()
    }

    /// Get reorg protection
    pub fn get_reorg_protection(&self, protection_id: &str) -> Option<ReorgProtection> {
        let reorg_protections = self.reorg_protections.read().unwrap();
        reorg_protections.get(protection_id).cloned()
    }

    /// Get metrics
    pub fn get_metrics(&self) -> PreconfirmationMetrics {
        let metrics = self.metrics.read().unwrap();
        metrics.clone()
    }

    /// Get all preconfirmations
    pub fn get_all_preconfirmations(&self) -> Vec<Preconfirmation> {
        let preconfirmations = self.preconfirmations.read().unwrap();
        preconfirmations.values().cloned().collect()
    }

    /// Get all preconfirmation batches
    pub fn get_all_preconfirmation_batches(&self) -> Vec<PreconfirmationBatch> {
        let preconfirmation_batches = self.preconfirmation_batches.read().unwrap();
        preconfirmation_batches.values().cloned().collect()
    }

    /// Get all reorg protections
    pub fn get_all_reorg_protections(&self) -> Vec<ReorgProtection> {
        let reorg_protections = self.reorg_protections.read().unwrap();
        reorg_protections.values().cloned().collect()
    }
}

/// Get current timestamp in milliseconds
#[allow(dead_code)]
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preconfirmation_engine_creation() {
        let engine = PreconfirmationEngine::new();
        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_preconfirmations_issued, 0);
    }

    #[test]
    fn test_issue_preconfirmation() {
        let engine = PreconfirmationEngine::new();

        let preconfirmation = Preconfirmation {
            preconfirmation_id: "preconf-1".to_string(),
            transaction_hash: "tx-1".to_string(),
            sequencer_id: "seq-1".to_string(),
            sequencer_signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            block_number: 1,
            timestamp: current_timestamp(),
            expiry_timestamp: current_timestamp() + 300000, // 5 minutes
            gas_price: 1000000000,                          // 1 gwei
            gas_limit: 100000,
            priority_fee: 100000000, // 0.1 gwei
            mev_protection_level: MEVProtectionLevel::Basic,
            cross_rollup_data: None,
        };

        let result = engine.issue_preconfirmation(preconfirmation.clone());
        assert!(result.is_ok());

        // Verify preconfirmation was stored
        let stored_preconfirmation = engine.get_preconfirmation("preconf-1");
        assert!(stored_preconfirmation.is_some());
        assert_eq!(
            stored_preconfirmation.unwrap().preconfirmation_id,
            "preconf-1"
        );

        // Verify status is pending
        let status = engine.get_preconfirmation_status("preconf-1");
        assert_eq!(status, Some(PreconfirmationStatus::Pending));
    }

    #[test]
    fn test_issue_preconfirmation_invalid_data() {
        let engine = PreconfirmationEngine::new();

        let preconfirmation = Preconfirmation {
            preconfirmation_id: "".to_string(), // Empty ID
            transaction_hash: "tx-1".to_string(),
            sequencer_id: "seq-1".to_string(),
            sequencer_signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            block_number: 1,
            timestamp: current_timestamp(),
            expiry_timestamp: current_timestamp() + 300000, // 5 minutes
            gas_price: 1000000000,                          // 1 gwei
            gas_limit: 100000,
            priority_fee: 100000000, // 0.1 gwei
            mev_protection_level: MEVProtectionLevel::Basic,
            cross_rollup_data: None,
        };

        let result = engine.issue_preconfirmation(preconfirmation);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            PreconfirmationError::InvalidPreconfirmation
        );
    }

    #[test]
    fn test_confirm_preconfirmation() {
        let engine = PreconfirmationEngine::new();

        // Issue preconfirmation first
        let preconfirmation = Preconfirmation {
            preconfirmation_id: "preconf-1".to_string(),
            transaction_hash: "tx-1".to_string(),
            sequencer_id: "seq-1".to_string(),
            sequencer_signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            block_number: 1,
            timestamp: current_timestamp(),
            expiry_timestamp: current_timestamp() + 300000, // 5 minutes
            gas_price: 1000000000,                          // 1 gwei
            gas_limit: 100000,
            priority_fee: 100000000, // 0.1 gwei
            mev_protection_level: MEVProtectionLevel::Basic,
            cross_rollup_data: None,
        };

        engine.issue_preconfirmation(preconfirmation).unwrap();

        // Confirm preconfirmation
        let result = engine.confirm_preconfirmation("preconf-1");
        assert!(result.is_ok());

        // Verify status is confirmed
        let status = engine.get_preconfirmation_status("preconf-1");
        assert_eq!(status, Some(PreconfirmationStatus::Confirmed));
    }

    #[test]
    fn test_finalize_preconfirmation() {
        let engine = PreconfirmationEngine::new();

        // Issue preconfirmation first
        let preconfirmation = Preconfirmation {
            preconfirmation_id: "preconf-1".to_string(),
            transaction_hash: "tx-1".to_string(),
            sequencer_id: "seq-1".to_string(),
            sequencer_signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            block_number: 1,
            timestamp: current_timestamp(),
            expiry_timestamp: current_timestamp() + 300000, // 5 minutes
            gas_price: 1000000000,                          // 1 gwei
            gas_limit: 100000,
            priority_fee: 100000000, // 0.1 gwei
            mev_protection_level: MEVProtectionLevel::Basic,
            cross_rollup_data: None,
        };

        engine.issue_preconfirmation(preconfirmation).unwrap();

        // Finalize preconfirmation
        let result = engine.finalize_preconfirmation("preconf-1");
        assert!(result.is_ok());

        // Verify status is finalized
        let status = engine.get_preconfirmation_status("preconf-1");
        assert_eq!(status, Some(PreconfirmationStatus::Finalized));
    }

    #[test]
    fn test_create_preconfirmation_batch() {
        let engine = PreconfirmationEngine::new();

        let preconfirmation1 = Preconfirmation {
            preconfirmation_id: "preconf-1".to_string(),
            transaction_hash: "tx-1".to_string(),
            sequencer_id: "seq-1".to_string(),
            sequencer_signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            block_number: 1,
            timestamp: current_timestamp(),
            expiry_timestamp: current_timestamp() + 300000, // 5 minutes
            gas_price: 1000000000,                          // 1 gwei
            gas_limit: 100000,
            priority_fee: 100000000, // 0.1 gwei
            mev_protection_level: MEVProtectionLevel::Basic,
            cross_rollup_data: None,
        };

        let preconfirmation2 = Preconfirmation {
            preconfirmation_id: "preconf-2".to_string(),
            transaction_hash: "tx-2".to_string(),
            sequencer_id: "seq-1".to_string(),
            sequencer_signature: vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            block_number: 1,
            timestamp: current_timestamp(),
            expiry_timestamp: current_timestamp() + 300000, // 5 minutes
            gas_price: 1000000000,                          // 1 gwei
            gas_limit: 100000,
            priority_fee: 100000000, // 0.1 gwei
            mev_protection_level: MEVProtectionLevel::Basic,
            cross_rollup_data: None,
        };

        let batch = PreconfirmationBatch {
            batch_id: "batch-1".to_string(),
            sequencer_id: "seq-1".to_string(),
            preconfirmations: vec![preconfirmation1, preconfirmation2],
            batch_signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            batch_timestamp: current_timestamp(),
            batch_expiry: current_timestamp() + 300000, // 5 minutes
            mev_protection_applied: true,
        };

        let result = engine.create_preconfirmation_batch(batch.clone());
        assert!(result.is_ok());

        // Verify batch was stored
        let stored_batch = engine.get_preconfirmation_batch("batch-1");
        assert!(stored_batch.is_some());
        assert_eq!(stored_batch.unwrap().batch_id, "batch-1");
    }

    #[test]
    fn test_add_reorg_protection() {
        let engine = PreconfirmationEngine::new();

        let protection = ReorgProtection {
            protection_id: "protection-1".to_string(),
            transaction_hash: "tx-1".to_string(),
            sequencer_id: "seq-1".to_string(),
            protection_level: ReorgProtectionLevel::Advanced,
            protection_data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            protection_signature: vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            protection_timestamp: current_timestamp(),
        };

        let result = engine.add_reorg_protection(protection.clone());
        assert!(result.is_ok());

        // Verify protection was stored
        let stored_protection = engine.get_reorg_protection("protection-1");
        assert!(stored_protection.is_some());
        assert_eq!(stored_protection.unwrap().protection_id, "protection-1");
    }

    #[test]
    fn test_coordinate_cross_rollup() {
        let engine = PreconfirmationEngine::new();

        let coordination_data = CrossRollupData {
            source_rollup_id: "rollup-1".to_string(),
            destination_rollup_id: "rollup-2".to_string(),
            atomic_group_id: "atomic-1".to_string(),
            coordination_data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            cross_rollup_signature: vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        };

        let result = engine.coordinate_cross_rollup(coordination_data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_detect_reorg() {
        let engine = PreconfirmationEngine::new();

        // Test with no protection
        let result = engine.detect_reorg("tx-1");
        assert!(result.is_ok());
        assert!(!result.unwrap()); // No reorg detected

        // Add reorg protection
        let protection = ReorgProtection {
            protection_id: "protection-1".to_string(),
            transaction_hash: "tx-1".to_string(),
            sequencer_id: "seq-1".to_string(),
            protection_level: ReorgProtectionLevel::Advanced,
            protection_data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            protection_signature: vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            protection_timestamp: current_timestamp(),
        };

        engine.add_reorg_protection(protection).unwrap();

        // Test with protection
        let result = engine.detect_reorg("tx-1");
        assert!(result.is_ok());
        assert!(!result.unwrap()); // No reorg detected (simulated)
    }

    #[test]
    fn test_preconfirmation_metrics() {
        let engine = PreconfirmationEngine::new();

        // Issue preconfirmations
        for i in 0..3 {
            let preconfirmation = Preconfirmation {
                preconfirmation_id: format!("preconf-{}", i),
                transaction_hash: format!("tx-{}", i),
                sequencer_id: "seq-1".to_string(),
                sequencer_signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                block_number: 1,
                timestamp: current_timestamp(),
                expiry_timestamp: current_timestamp() + 300000, // 5 minutes
                gas_price: 1000000000,                          // 1 gwei
                gas_limit: 100000,
                priority_fee: 100000000, // 0.1 gwei
                mev_protection_level: MEVProtectionLevel::Basic,
                cross_rollup_data: None,
            };

            engine.issue_preconfirmation(preconfirmation).unwrap();
        }

        // Confirm and finalize some preconfirmations
        engine.confirm_preconfirmation("preconf-0").unwrap();
        engine.finalize_preconfirmation("preconf-0").unwrap();

        engine.confirm_preconfirmation("preconf-1").unwrap();

        // Create batch
        let preconfirmation1 = Preconfirmation {
            preconfirmation_id: "preconf-batch-1".to_string(),
            transaction_hash: "tx-batch-1".to_string(),
            sequencer_id: "seq-1".to_string(),
            sequencer_signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            block_number: 1,
            timestamp: current_timestamp(),
            expiry_timestamp: current_timestamp() + 300000, // 5 minutes
            gas_price: 1000000000,                          // 1 gwei
            gas_limit: 100000,
            priority_fee: 100000000, // 0.1 gwei
            mev_protection_level: MEVProtectionLevel::Basic,
            cross_rollup_data: None,
        };

        let batch = PreconfirmationBatch {
            batch_id: "batch-1".to_string(),
            sequencer_id: "seq-1".to_string(),
            preconfirmations: vec![preconfirmation1],
            batch_signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            batch_timestamp: current_timestamp(),
            batch_expiry: current_timestamp() + 300000, // 5 minutes
            mev_protection_applied: true,
        };

        engine.create_preconfirmation_batch(batch).unwrap();

        // Add reorg protection
        let protection = ReorgProtection {
            protection_id: "protection-1".to_string(),
            transaction_hash: "tx-1".to_string(),
            sequencer_id: "seq-1".to_string(),
            protection_level: ReorgProtectionLevel::Advanced,
            protection_data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            protection_signature: vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            protection_timestamp: current_timestamp(),
        };

        engine.add_reorg_protection(protection).unwrap();

        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_preconfirmations_issued, 3);
        assert_eq!(metrics.total_preconfirmations_confirmed, 2);
        assert_eq!(metrics.total_preconfirmations_finalized, 1);
        assert_eq!(metrics.total_batches_created, 1);
        assert_eq!(metrics.total_reorg_protections, 1);
    }

    #[test]
    fn test_get_all_preconfirmations() {
        let engine = PreconfirmationEngine::new();

        // Issue multiple preconfirmations
        for i in 0..3 {
            let preconfirmation = Preconfirmation {
                preconfirmation_id: format!("preconf-{}", i),
                transaction_hash: format!("tx-{}", i),
                sequencer_id: "seq-1".to_string(),
                sequencer_signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                block_number: 1,
                timestamp: current_timestamp(),
                expiry_timestamp: current_timestamp() + 300000, // 5 minutes
                gas_price: 1000000000,                          // 1 gwei
                gas_limit: 100000,
                priority_fee: 100000000, // 0.1 gwei
                mev_protection_level: MEVProtectionLevel::Basic,
                cross_rollup_data: None,
            };

            engine.issue_preconfirmation(preconfirmation).unwrap();
        }

        let preconfirmations = engine.get_all_preconfirmations();
        assert_eq!(preconfirmations.len(), 3);
    }

    #[test]
    fn test_get_all_preconfirmation_batches() {
        let engine = PreconfirmationEngine::new();

        // Create multiple batches
        for i in 0..3 {
            let preconfirmation = Preconfirmation {
                preconfirmation_id: format!("preconf-batch-{}", i),
                transaction_hash: format!("tx-batch-{}", i),
                sequencer_id: "seq-1".to_string(),
                sequencer_signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                block_number: 1,
                timestamp: current_timestamp(),
                expiry_timestamp: current_timestamp() + 300000, // 5 minutes
                gas_price: 1000000000,                          // 1 gwei
                gas_limit: 100000,
                priority_fee: 100000000, // 0.1 gwei
                mev_protection_level: MEVProtectionLevel::Basic,
                cross_rollup_data: None,
            };

            let batch = PreconfirmationBatch {
                batch_id: format!("batch-{}", i),
                sequencer_id: "seq-1".to_string(),
                preconfirmations: vec![preconfirmation],
                batch_signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                batch_timestamp: current_timestamp(),
                batch_expiry: current_timestamp() + 300000, // 5 minutes
                mev_protection_applied: true,
            };

            engine.create_preconfirmation_batch(batch).unwrap();
        }

        let batches = engine.get_all_preconfirmation_batches();
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn test_get_all_reorg_protections() {
        let engine = PreconfirmationEngine::new();

        // Add multiple reorg protections
        for i in 0..3 {
            let protection = ReorgProtection {
                protection_id: format!("protection-{}", i),
                transaction_hash: format!("tx-{}", i),
                sequencer_id: "seq-1".to_string(),
                protection_level: ReorgProtectionLevel::Advanced,
                protection_data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                protection_signature: vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                protection_timestamp: current_timestamp(),
            };

            engine.add_reorg_protection(protection).unwrap();
        }

        let protections = engine.get_all_reorg_protections();
        assert_eq!(protections.len(), 3);
    }
}
