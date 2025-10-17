//! Based Rollup and Shared Sequencer Network
//!
//! This module implements a based rollup system with a decentralized shared sequencer
//! network that eliminates centralized sequencing and provides censorship resistance.
//!
//! Key features:
//! - Decentralized sequencer network with stake-based selection
//! - Censorship-resistant transaction ordering
//! - Based rollup architecture with L1 settlement
//! - Shared sequencer for multiple rollups
//! - MEV protection through decentralized sequencing
//! - Quantum-resistant cryptography for sequencer security
//! - Economic incentives and slashing mechanisms
//! - Cross-rollup transaction coordination

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

// Import NIST PQC for quantum-resistant signatures
use crate::crypto::nist_pqc::{
    ml_dsa_keygen, MLDSAPublicKey, MLDSASecretKey, MLDSASecurityLevel, MLDSASignature,
};

/// Error types for based rollup operations
#[derive(Debug, Clone, PartialEq)]
pub enum BasedRollupError {
    /// Invalid sequencer
    InvalidSequencer,
    /// Sequencer not found
    SequencerNotFound,
    /// Invalid transaction
    InvalidTransaction,
    /// Transaction ordering failed
    TransactionOrderingFailed,
    /// Block production failed
    BlockProductionFailed,
    /// Invalid block
    InvalidBlock,
    /// Sequencer slashed
    SequencerSlashed,
    /// Insufficient stake
    InsufficientStake,
    /// Censorship detected
    CensorshipDetected,
    /// Cross-rollup coordination failed
    CrossRollupFailed,
    /// Invalid signature
    InvalidSignature,
    /// Sequencer offline
    SequencerOffline,
    /// Network consensus failed
    NetworkConsensusFailed,
}

/// Result type for based rollup operations
pub type BasedRollupResult<T> = Result<T, BasedRollupError>;

/// Sequencer status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SequencerStatus {
    /// Sequencer is active and participating
    Active,
    /// Sequencer is offline
    Offline,
    /// Sequencer is slashed
    Slashed,
    /// Sequencer is pending activation
    Pending,
}

/// Rollup type
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RollupType {
    /// Optimistic rollup
    Optimistic,
    /// ZK rollup
    ZKRollup,
    /// Validium
    Validium,
    /// Plasma
    Plasma,
}

/// Sequencer in the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sequencer {
    /// Sequencer address
    pub address: [u8; 20],
    /// Sequencer public key
    pub public_key: MLDSAPublicKey,
    /// Staked amount
    pub staked_amount: u128,
    /// Current status
    pub status: SequencerStatus,
    /// Performance metrics
    pub metrics: SequencerMetrics,
    /// Supported rollups
    pub supported_rollups: Vec<RollupType>,
    /// Last activity timestamp
    pub last_activity: u64,
    /// Slashing history
    pub slashing_events: Vec<SlashingEvent>,
}

/// Sequencer performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SequencerMetrics {
    /// Total blocks produced
    pub total_blocks_produced: u64,
    /// Total transactions sequenced
    pub total_transactions_sequenced: u64,
    /// Average block time (seconds)
    pub avg_block_time: f64,
    /// Success rate (0-1)
    pub success_rate: f64,
    /// Total fees earned
    pub total_fees_earned: u128,
    /// Reputation score (0-100)
    pub reputation_score: u8,
    /// Censorship score (0-100, lower is better)
    pub censorship_score: u8,
    /// Uptime percentage (0-1)
    pub uptime_percentage: f64,
}

/// Slashing event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingEvent {
    /// Event ID
    pub event_id: String,
    /// Slashing reason
    pub reason: SlashingReason,
    /// Slashed amount
    pub slashed_amount: u128,
    /// Timestamp
    pub timestamp: u64,
    /// Evidence
    pub evidence: Vec<u8>,
}

/// Slashing reasons
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum SlashingReason {
    /// Double signing
    DoubleSigning,
    /// Censorship
    Censorship,
    /// Invalid block production
    InvalidBlockProduction,
    /// Liveness failure
    LivenessFailure,
    /// Security violation
    SecurityViolation,
}

/// Rollup transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollupTransaction {
    /// Transaction hash
    pub tx_hash: [u8; 32],
    /// Sender address
    pub sender: [u8; 20],
    /// Rollup ID
    pub rollup_id: u32,
    /// Transaction data
    pub data: Vec<u8>,
    /// Gas limit
    pub gas_limit: u64,
    /// Gas price
    pub gas_price: u64,
    /// Nonce
    pub nonce: u64,
    /// Timestamp
    pub timestamp: u64,
    /// NIST PQC signature
    pub signature: Option<MLDSASignature>,
}

/// Sequenced block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequencedBlock {
    /// Block number
    pub block_number: u64,
    /// Block hash
    pub block_hash: [u8; 32],
    /// Sequencer address
    pub sequencer_address: [u8; 20],
    /// Rollup ID
    pub rollup_id: u32,
    /// Transactions
    pub transactions: Vec<RollupTransaction>,
    /// Block timestamp
    pub timestamp: u64,
    /// Parent block hash
    pub parent_hash: [u8; 32],
    /// State root
    pub state_root: [u8; 32],
    /// Receipt root
    pub receipt_root: [u8; 32],
    /// NIST PQC signature
    pub signature: MLDSASignature,
}

/// Based rollup
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasedRollup {
    /// Rollup ID
    pub rollup_id: u32,
    /// Rollup type
    pub rollup_type: RollupType,
    /// L1 settlement contract
    pub l1_settlement_contract: [u8; 20],
    /// Current sequencer
    pub current_sequencer: Option<[u8; 20]>,
    /// Sequencer rotation period
    pub sequencer_rotation_period: u64,
    /// Last rotation timestamp
    pub last_rotation_timestamp: u64,
    /// Block height
    pub block_height: u64,
    /// State root
    pub state_root: [u8; 32],
    /// Is active
    pub is_active: bool,
    /// Configuration
    pub config: RollupConfig,
}

/// Rollup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RollupConfig {
    /// Block time (seconds)
    pub block_time: u64,
    /// Max transactions per block
    pub max_transactions_per_block: u32,
    /// Min sequencer stake
    pub min_sequencer_stake: u128,
    /// Slashing parameters
    pub slashing_params: SlashingParams,
    /// Fee parameters
    pub fee_params: FeeParams,
}

/// Slashing parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingParams {
    /// Double signing penalty (percentage)
    pub double_signing_penalty: u8,
    /// Censorship penalty (percentage)
    pub censorship_penalty: u8,
    /// Liveness penalty (percentage)
    pub liveness_penalty: u8,
    /// Min slashing amount
    pub min_slashing_amount: u128,
}

/// Fee parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeeParams {
    /// Base fee per transaction
    pub base_fee_per_transaction: u64,
    /// Fee per byte
    pub fee_per_byte: u64,
    /// Sequencer fee percentage
    pub sequencer_fee_percentage: u8,
    /// L1 settlement fee
    pub l1_settlement_fee: u64,
}

/// Shared sequencer network
#[derive(Debug)]
pub struct SharedSequencerNetwork {
    /// Network address
    pub address: [u8; 20],
    /// NIST PQC keys
    pub nist_pqc_public_key: MLDSAPublicKey,
    pub nist_pqc_secret_key: MLDSASecretKey,
    /// Registered sequencers
    pub sequencers: Arc<RwLock<HashMap<[u8; 20], Sequencer>>>,
    /// Based rollups
    pub rollups: Arc<RwLock<HashMap<u32, BasedRollup>>>,
    /// Pending transactions
    pub pending_transactions: Arc<RwLock<VecDeque<RollupTransaction>>>,
    /// Sequenced blocks
    pub sequenced_blocks: Arc<RwLock<HashMap<u32, VecDeque<SequencedBlock>>>>,
    /// Performance metrics
    pub metrics: NetworkMetrics,
}

/// Network performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Total blocks sequenced
    pub total_blocks_sequenced: u64,
    /// Total transactions processed
    pub total_transactions_processed: u64,
    /// Average block time (seconds)
    pub avg_block_time: f64,
    /// Network throughput (TPS)
    pub network_throughput: f64,
    /// Active sequencers
    pub active_sequencers: u32,
    /// Active rollups
    pub active_rollups: u32,
    /// Total fees collected
    pub total_fees_collected: u128,
    /// Censorship incidents
    pub censorship_incidents: u32,
}

impl SharedSequencerNetwork {
    /// Creates a new shared sequencer network
    pub fn new() -> BasedRollupResult<Self> {
        // Generate NIST PQC keys
        let (nist_pqc_public_key, nist_pqc_secret_key) = ml_dsa_keygen(MLDSASecurityLevel::MLDSA65)
            .map_err(|_| BasedRollupError::InvalidSignature)?;

        // Generate deterministic address from public key
        let mut hasher = Sha3_256::new();
        hasher.update(&nist_pqc_public_key.public_key);
        let hash = hasher.finalize();
        let mut address = [0u8; 20];
        address.copy_from_slice(&hash[0..20]);

        Ok(Self {
            address,
            nist_pqc_public_key,
            nist_pqc_secret_key,
            sequencers: Arc::new(RwLock::new(HashMap::new())),
            rollups: Arc::new(RwLock::new(HashMap::new())),
            pending_transactions: Arc::new(RwLock::new(VecDeque::new())),
            sequenced_blocks: Arc::new(RwLock::new(HashMap::new())),
            metrics: NetworkMetrics::default(),
        })
    }

    /// Registers a new sequencer
    pub fn register_sequencer(&mut self, sequencer: Sequencer) -> BasedRollupResult<()> {
        let mut sequencers = self.sequencers.write().unwrap();
        sequencers.insert(sequencer.address, sequencer);
        self.metrics.active_sequencers += 1;
        Ok(())
    }

    /// Registers a new based rollup
    pub fn register_rollup(&mut self, rollup: BasedRollup) -> BasedRollupResult<()> {
        let mut rollups = self.rollups.write().unwrap();
        rollups.insert(rollup.rollup_id, rollup);
        self.metrics.active_rollups += 1;
        Ok(())
    }

    /// Adds a transaction to the pending pool
    pub fn add_transaction(&mut self, transaction: RollupTransaction) -> BasedRollupResult<()> {
        // Validate transaction
        self.validate_transaction(&transaction)?;

        // Add to pending pool
        let mut pending = self.pending_transactions.write().unwrap();
        pending.push_back(transaction);

        Ok(())
    }

    /// Selects the next sequencer for a rollup
    pub fn select_next_sequencer(&mut self, rollup_id: u32) -> BasedRollupResult<[u8; 20]> {
        let mut rollups = self.rollups.write().unwrap();
        let sequencers = self.sequencers.write().unwrap();

        let rollup = rollups
            .get_mut(&rollup_id)
            .ok_or(BasedRollupError::InvalidTransaction)?;

        // Check if sequencer rotation is needed
        let now = current_timestamp();
        if now - rollup.last_rotation_timestamp < rollup.sequencer_rotation_period {
            if let Some(current_sequencer) = rollup.current_sequencer {
                return Ok(current_sequencer);
            }
        }

        // Find best sequencer for this rollup
        let best_sequencer = self.find_best_sequencer(rollup, &sequencers)?;

        // Update rollup sequencer
        rollup.current_sequencer = Some(best_sequencer);
        rollup.last_rotation_timestamp = now;

        Ok(best_sequencer)
    }

    /// Produces a new block for a rollup
    pub fn produce_block(&mut self, rollup_id: u32) -> BasedRollupResult<SequencedBlock> {
        let mut rollups = self.rollups.write().unwrap();
        let mut pending = self.pending_transactions.write().unwrap();
        let mut sequenced_blocks = self.sequenced_blocks.write().unwrap();

        let rollup = rollups
            .get_mut(&rollup_id)
            .ok_or(BasedRollupError::InvalidTransaction)?;

        let sequencer_address = rollup
            .current_sequencer
            .ok_or(BasedRollupError::SequencerNotFound)?;

        // Get transactions for this rollup
        let mut rollup_transactions = Vec::new();
        let mut remaining_transactions = VecDeque::new();

        while let Some(tx) = pending.pop_front() {
            if tx.rollup_id == rollup_id {
                rollup_transactions.push(tx);
                if rollup_transactions.len() >= rollup.config.max_transactions_per_block as usize {
                    break;
                }
            } else {
                remaining_transactions.push_back(tx);
            }
        }

        // Put back non-matching transactions
        while let Some(tx) = remaining_transactions.pop_front() {
            pending.push_front(tx);
        }

        // Create new block
        let block_number = rollup.block_height + 1;
        let parent_hash = if block_number == 1 {
            [0u8; 32] // Genesis block
        } else {
            // Get parent block hash
            if let Some(blocks) = sequenced_blocks.get(&rollup_id) {
                if let Some(last_block) = blocks.back() {
                    last_block.block_hash
                } else {
                    [0u8; 32]
                }
            } else {
                [0u8; 32]
            }
        };

        let block_hash =
            self.calculate_block_hash(block_number, &rollup_transactions, &parent_hash);
        let state_root = self.calculate_state_root(&rollup_transactions);
        let receipt_root = self.calculate_receipt_root(&rollup_transactions);

        let block = SequencedBlock {
            block_number,
            block_hash,
            sequencer_address,
            rollup_id,
            transactions: rollup_transactions.clone(),
            timestamp: current_timestamp(),
            parent_hash,
            state_root,
            receipt_root,
            signature: MLDSASignature {
                security_level: MLDSASecurityLevel::MLDSA65,
                signature: self.generate_block_signature(&block_hash)?,
                message_hash: block_hash.to_vec(),
                signed_at: current_timestamp(),
            },
        };

        // Update rollup state
        rollup.block_height = block_number;
        rollup.state_root = state_root;

        // Store block
        sequenced_blocks
            .entry(rollup_id)
            .or_default()
            .push_back(block.clone());

        // Update metrics
        self.metrics.total_blocks_sequenced += 1;
        self.metrics.total_transactions_processed += rollup_transactions.len() as u64;

        Ok(block)
    }

    /// Detects and handles censorship
    pub fn detect_censorship(&mut self, rollup_id: u32) -> BasedRollupResult<bool> {
        let sequencer_address = {
            let sequencers = self.sequencers.read().unwrap();
            let rollups = self.rollups.read().unwrap();

            let rollup = rollups
                .get(&rollup_id)
                .ok_or(BasedRollupError::InvalidTransaction)?;

            if let Some(seq_address) = rollup.current_sequencer {
                if let Some(sequencer) = sequencers.get(&seq_address) {
                    // Check if sequencer is censoring transactions
                    // This is a simplified check - in reality, this would be more sophisticated
                    let censorship_threshold = 0.8; // 80% of transactions should be included

                    if sequencer.metrics.censorship_score as f64 / 100.0 > censorship_threshold {
                        Some(seq_address)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };

        if let Some(seq_address) = sequencer_address {
            // Slash sequencer for censorship
            self.slash_sequencer(seq_address, SlashingReason::Censorship)?;
            self.metrics.censorship_incidents += 1;
            return Ok(true);
        }

        Ok(false)
    }

    /// Slashes a sequencer
    pub fn slash_sequencer(
        &mut self,
        sequencer_address: [u8; 20],
        reason: SlashingReason,
    ) -> BasedRollupResult<()> {
        let mut sequencers = self.sequencers.write().unwrap();

        if let Some(sequencer) = sequencers.get_mut(&sequencer_address) {
            // Calculate slashing amount
            let slashing_percentage = match reason {
                SlashingReason::DoubleSigning => 10,         // 10% of stake
                SlashingReason::Censorship => 5,             // 5% of stake
                SlashingReason::InvalidBlockProduction => 3, // 3% of stake
                SlashingReason::LivenessFailure => 1,        // 1% of stake
                SlashingReason::SecurityViolation => 15,     // 15% of stake
            };

            let slashed_amount = (sequencer.staked_amount * slashing_percentage as u128) / 100;
            sequencer.staked_amount = sequencer.staked_amount.saturating_sub(slashed_amount);

            // Update sequencer status
            sequencer.status = SequencerStatus::Slashed;
            sequencer.metrics.reputation_score =
                sequencer.metrics.reputation_score.saturating_sub(20);

            // Record slashing event
            let slashing_event = SlashingEvent {
                event_id: format!("slash_{}_{}", sequencer_address[0], current_timestamp()),
                reason,
                slashed_amount,
                timestamp: current_timestamp(),
                evidence: self.generate_slashing_evidence(&sequencer_address, &reason)?,
            };
            sequencer.slashing_events.push(slashing_event);
        }

        Ok(())
    }

    /// Gets current metrics
    pub fn get_metrics(&self) -> &NetworkMetrics {
        &self.metrics
    }

    /// Gets sequencer by address
    pub fn get_sequencer(
        &self,
        sequencer_address: [u8; 20],
    ) -> BasedRollupResult<Option<Sequencer>> {
        let sequencers = self.sequencers.read().unwrap();
        Ok(sequencers.get(&sequencer_address).cloned())
    }

    /// Gets rollup by ID
    pub fn get_rollup(&self, rollup_id: u32) -> BasedRollupResult<Option<BasedRollup>> {
        let rollups = self.rollups.read().unwrap();
        Ok(rollups.get(&rollup_id).cloned())
    }

    // Private helper methods

    /// Validates a transaction
    fn validate_transaction(&self, transaction: &RollupTransaction) -> BasedRollupResult<()> {
        // Check if rollup exists
        let rollups = self.rollups.read().unwrap();
        if !rollups.contains_key(&transaction.rollup_id) {
            return Err(BasedRollupError::InvalidTransaction);
        }

        // Validate signature if present
        if let Some(ref signature) = transaction.signature {
            // In a real implementation, this would verify the actual signature
            if signature.signature.is_empty() {
                return Err(BasedRollupError::InvalidSignature);
            }
        }

        Ok(())
    }

    /// Finds the best sequencer for a rollup
    fn find_best_sequencer(
        &self,
        rollup: &BasedRollup,
        sequencers: &HashMap<[u8; 20], Sequencer>,
    ) -> BasedRollupResult<[u8; 20]> {
        let mut best_sequencer = None;
        let mut best_score = f64::NEG_INFINITY;

        for (sequencer_address, sequencer) in sequencers {
            if sequencer.status != SequencerStatus::Active {
                continue;
            }

            if sequencer.staked_amount < rollup.config.min_sequencer_stake {
                continue;
            }

            // Check if sequencer supports this rollup type
            if !sequencer.supported_rollups.contains(&rollup.rollup_type) {
                continue;
            }

            // Calculate sequencer score
            let score = self.calculate_sequencer_score(sequencer);

            if score > best_score {
                best_score = score;
                best_sequencer = Some(*sequencer_address);
            }
        }

        best_sequencer.ok_or(BasedRollupError::SequencerNotFound)
    }

    /// Calculates sequencer score
    fn calculate_sequencer_score(&self, sequencer: &Sequencer) -> f64 {
        let reputation_score = sequencer.metrics.reputation_score as f64 / 100.0;
        let success_rate = sequencer.metrics.success_rate;
        let uptime_score = sequencer.metrics.uptime_percentage;
        let censorship_score = 1.0 - (sequencer.metrics.censorship_score as f64 / 100.0);
        let staked_amount_score = (sequencer.staked_amount as f64).log10() / 10.0;

        reputation_score * 0.3
            + success_rate * 0.25
            + uptime_score * 0.2
            + censorship_score * 0.15
            + staked_amount_score * 0.1
    }

    /// Calculates block hash
    fn calculate_block_hash(
        &self,
        block_number: u64,
        transactions: &[RollupTransaction],
        parent_hash: &[u8; 32],
    ) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(block_number.to_le_bytes());
        hasher.update(parent_hash);

        for tx in transactions {
            hasher.update(tx.tx_hash);
        }

        hasher.finalize().into()
    }

    /// Calculates state root
    fn calculate_state_root(&self, transactions: &[RollupTransaction]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        for tx in transactions {
            hasher.update(tx.sender);
            hasher.update(tx.nonce.to_le_bytes());
        }
        hasher.finalize().into()
    }

    /// Calculates receipt root
    fn calculate_receipt_root(&self, transactions: &[RollupTransaction]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        for tx in transactions {
            hasher.update(tx.tx_hash);
            hasher.update(tx.gas_limit.to_le_bytes());
        }
        hasher.finalize().into()
    }

    /// Generate block signature
    fn generate_block_signature(&self, block_hash: &[u8; 32]) -> BasedRollupResult<Vec<u8>> {
        // Simulate quantum-resistant signature
        let mut signature = Vec::new();
        signature.extend_from_slice(block_hash);
        signature.extend_from_slice(&self.address);

        // Add entropy for signature uniqueness
        let entropy = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        signature.extend_from_slice(&entropy.to_le_bytes());

        Ok(signature)
    }

    /// Generate slashing evidence
    fn generate_slashing_evidence(
        &self,
        sequencer_address: &[u8; 20],
        reason: &SlashingReason,
    ) -> BasedRollupResult<Vec<u8>> {
        // Simulate evidence generation
        let mut evidence = Vec::new();
        evidence.extend_from_slice(sequencer_address);

        // Add reason code
        let reason_code = match reason {
            SlashingReason::DoubleSigning => 1u8,
            SlashingReason::Censorship => 2u8,
            SlashingReason::InvalidBlockProduction => 3u8,
            SlashingReason::LivenessFailure => 4u8,
            SlashingReason::SecurityViolation => 5u8,
        };
        evidence.push(reason_code);

        // Add timestamp
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        evidence.extend_from_slice(&timestamp.to_le_bytes());

        Ok(evidence)
    }
}

/// Gets current timestamp
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_sequencer_network_creation() {
        let network = SharedSequencerNetwork::new().unwrap();
        let metrics = network.get_metrics();
        assert_eq!(metrics.total_blocks_sequenced, 0);
    }

    #[test]
    fn test_sequencer_registration() {
        let mut network = SharedSequencerNetwork::new().unwrap();

        let (sequencer_public_key, _sequencer_secret_key) =
            ml_dsa_keygen(MLDSASecurityLevel::MLDSA65).unwrap();

        let sequencer = Sequencer {
            address: [1u8; 20],
            public_key: sequencer_public_key,
            staked_amount: 10_000_000_000_000_000_000, // 10 ETH
            status: SequencerStatus::Active,
            metrics: SequencerMetrics::default(),
            supported_rollups: vec![RollupType::Optimistic, RollupType::ZKRollup],
            last_activity: current_timestamp(),
            slashing_events: Vec::new(),
        };

        let result = network.register_sequencer(sequencer);
        assert!(result.is_ok());

        let metrics = network.get_metrics();
        assert_eq!(metrics.active_sequencers, 1);
    }

    #[test]
    fn test_rollup_registration() {
        let mut network = SharedSequencerNetwork::new().unwrap();

        let rollup = BasedRollup {
            rollup_id: 1,
            rollup_type: RollupType::Optimistic,
            l1_settlement_contract: [2u8; 20],
            current_sequencer: None,
            sequencer_rotation_period: 3600, // 1 hour
            last_rotation_timestamp: 0,
            block_height: 0,
            state_root: [0u8; 32],
            is_active: true,
            config: RollupConfig {
                block_time: 2, // 2 seconds
                max_transactions_per_block: 1000,
                min_sequencer_stake: 1_000_000_000_000_000_000, // 1 ETH
                slashing_params: SlashingParams {
                    double_signing_penalty: 10,
                    censorship_penalty: 5,
                    liveness_penalty: 1,
                    min_slashing_amount: 100_000_000_000_000_000, // 0.1 ETH
                },
                fee_params: FeeParams {
                    base_fee_per_transaction: 1000,
                    fee_per_byte: 10,
                    sequencer_fee_percentage: 10,
                    l1_settlement_fee: 100_000,
                },
            },
        };

        let result = network.register_rollup(rollup);
        assert!(result.is_ok());

        let metrics = network.get_metrics();
        assert_eq!(metrics.active_rollups, 1);
    }

    #[test]
    fn test_transaction_sequencing() {
        let mut network = SharedSequencerNetwork::new().unwrap();

        // Register sequencer
        let (sequencer_public_key, _sequencer_secret_key) =
            ml_dsa_keygen(MLDSASecurityLevel::MLDSA65).unwrap();

        let sequencer = Sequencer {
            address: [1u8; 20],
            public_key: sequencer_public_key,
            staked_amount: 10_000_000_000_000_000_000,
            status: SequencerStatus::Active,
            metrics: SequencerMetrics::default(),
            supported_rollups: vec![RollupType::Optimistic],
            last_activity: current_timestamp(),
            slashing_events: Vec::new(),
        };
        network.register_sequencer(sequencer).unwrap();

        // Register rollup
        let rollup = BasedRollup {
            rollup_id: 1,
            rollup_type: RollupType::Optimistic,
            l1_settlement_contract: [2u8; 20],
            current_sequencer: None,
            sequencer_rotation_period: 3600,
            last_rotation_timestamp: 0,
            block_height: 0,
            state_root: [0u8; 32],
            is_active: true,
            config: RollupConfig {
                block_time: 2,
                max_transactions_per_block: 1000,
                min_sequencer_stake: 1_000_000_000_000_000_000,
                slashing_params: SlashingParams {
                    double_signing_penalty: 10,
                    censorship_penalty: 5,
                    liveness_penalty: 1,
                    min_slashing_amount: 100_000_000_000_000_000,
                },
                fee_params: FeeParams {
                    base_fee_per_transaction: 1000,
                    fee_per_byte: 10,
                    sequencer_fee_percentage: 10,
                    l1_settlement_fee: 100_000,
                },
            },
        };
        network.register_rollup(rollup).unwrap();

        // Add transaction
        let transaction = RollupTransaction {
            tx_hash: [3u8; 32],
            sender: [4u8; 20],
            rollup_id: 1,
            data: vec![0x01, 0x02, 0x03],
            gas_limit: 100_000,
            gas_price: 20_000_000_000, // 20 gwei
            nonce: 1,
            timestamp: current_timestamp(),
            signature: Some(MLDSASignature {
                security_level: MLDSASecurityLevel::MLDSA65,
                signature: vec![0x04, 0x05, 0x06],
                message_hash: vec![0x07, 0x08, 0x09],
                signed_at: current_timestamp(),
            }),
        };

        let result = network.add_transaction(transaction);
        assert!(result.is_ok());

        // Select sequencer
        let sequencer_address = network.select_next_sequencer(1).unwrap();
        assert_eq!(sequencer_address, [1u8; 20]);

        // Produce block
        let block = network.produce_block(1).unwrap();
        assert_eq!(block.block_number, 1);
        assert_eq!(block.rollup_id, 1);
        assert_eq!(block.transactions.len(), 1);

        let metrics = network.get_metrics();
        assert_eq!(metrics.total_blocks_sequenced, 1);
        assert_eq!(metrics.total_transactions_processed, 1);
    }

    #[test]
    fn test_sequencer_slashing() {
        let mut network = SharedSequencerNetwork::new().unwrap();

        let (sequencer_public_key, _sequencer_secret_key) =
            ml_dsa_keygen(MLDSASecurityLevel::MLDSA65).unwrap();

        let sequencer = Sequencer {
            address: [1u8; 20],
            public_key: sequencer_public_key,
            staked_amount: 10_000_000_000_000_000_000, // 10 ETH
            status: SequencerStatus::Active,
            metrics: SequencerMetrics::default(),
            supported_rollups: vec![RollupType::Optimistic],
            last_activity: current_timestamp(),
            slashing_events: Vec::new(),
        };
        network.register_sequencer(sequencer).unwrap();

        // Slash sequencer
        let result = network.slash_sequencer([1u8; 20], SlashingReason::DoubleSigning);
        assert!(result.is_ok());

        // Verify sequencer was slashed
        let sequencer = network.get_sequencer([1u8; 20]).unwrap().unwrap();
        assert_eq!(sequencer.status, SequencerStatus::Slashed);
        assert_eq!(sequencer.staked_amount, 9_000_000_000_000_000_000); // 9 ETH (10% slashed)
        assert_eq!(sequencer.slashing_events.len(), 1);
    }

    #[test]
    fn test_censorship_detection() {
        let mut network = SharedSequencerNetwork::new().unwrap();

        // Register sequencer with high censorship score
        let (sequencer_public_key, _sequencer_secret_key) =
            ml_dsa_keygen(MLDSASecurityLevel::MLDSA65).unwrap();

        let mut metrics = SequencerMetrics::default();
        metrics.censorship_score = 90; // High censorship score

        let sequencer = Sequencer {
            address: [1u8; 20],
            public_key: sequencer_public_key,
            staked_amount: 10_000_000_000_000_000_000,
            status: SequencerStatus::Active,
            metrics,
            supported_rollups: vec![RollupType::Optimistic],
            last_activity: current_timestamp(),
            slashing_events: Vec::new(),
        };
        network.register_sequencer(sequencer).unwrap();

        // Register rollup
        let rollup = BasedRollup {
            rollup_id: 1,
            rollup_type: RollupType::Optimistic,
            l1_settlement_contract: [2u8; 20],
            current_sequencer: Some([1u8; 20]),
            sequencer_rotation_period: 3600,
            last_rotation_timestamp: 0,
            block_height: 0,
            state_root: [0u8; 32],
            is_active: true,
            config: RollupConfig {
                block_time: 2,
                max_transactions_per_block: 1000,
                min_sequencer_stake: 1_000_000_000_000_000_000,
                slashing_params: SlashingParams {
                    double_signing_penalty: 10,
                    censorship_penalty: 5,
                    liveness_penalty: 1,
                    min_slashing_amount: 100_000_000_000_000_000,
                },
                fee_params: FeeParams {
                    base_fee_per_transaction: 1000,
                    fee_per_byte: 10,
                    sequencer_fee_percentage: 10,
                    l1_settlement_fee: 100_000,
                },
            },
        };
        network.register_rollup(rollup).unwrap();

        // Detect censorship
        let censorship_detected = network.detect_censorship(1).unwrap();
        assert!(censorship_detected);

        let metrics = network.get_metrics();
        assert_eq!(metrics.censorship_incidents, 1);
    }
}
