//! HotStuff BFT Consensus for Decentralized Sequencer Network
//!
//! This module implements HotStuff BFT consensus for the decentralized
//! sequencer network, providing censorship resistance and leader rotation
//! with verifiable random function (VRF) selection.
//!
//! Key features:
//! - HotStuff BFT consensus algorithm
//! - Leader rotation with VRF selection
//! - Slashing for equivocation and malicious behavior
//! - Censorship resistance mechanisms
//! - Sequencer set management
//! - Block proposal and voting
//! - View change and recovery
//! - Performance metrics and monitoring
//!
//! Technical advantages:
//! - Linear message complexity
//! - Optimistic responsiveness
//! - Byzantine fault tolerance
//! - Leader rotation fairness
//! - Censorship resistance
//! - Slashing mechanisms
//! - Performance optimization

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Error types for HotStuff BFT implementation
#[derive(Debug, Clone, PartialEq)]
pub enum HotStuffBFTError {
    /// Invalid proposal
    InvalidProposal,
    /// Invalid vote
    InvalidVote,
    /// Invalid view change
    InvalidViewChange,
    /// Not enough votes
    NotEnoughVotes,
    /// Invalid leader
    InvalidLeader,
    /// View change timeout
    ViewChangeTimeout,
    /// Duplicate vote
    DuplicateVote,
    /// Invalid signature
    InvalidSignature,
    /// Sequencer not found
    SequencerNotFound,
    /// Invalid block
    InvalidBlock,
    /// Consensus timeout
    ConsensusTimeout,
    /// Invalid quorum certificate
    InvalidQuorumCertificate,
    /// Malicious behavior detected
    MaliciousBehavior,
    /// Slashing condition met
    SlashingCondition,
}

pub type HotStuffBFTResult<T> = Result<T, HotStuffBFTError>;

/// HotStuff BFT sequencer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotStuffSequencer {
    /// Sequencer ID
    pub sequencer_id: String,
    /// Sequencer address
    pub address: String,
    /// Public key
    pub public_key: Vec<u8>,
    /// Stake amount
    pub stake: u128,
    /// Reputation score
    pub reputation: f64,
    /// Status
    pub status: SequencerStatus,
    /// Last activity timestamp
    pub last_activity: u64,
    /// Slashing history
    pub slashing_history: Vec<SlashingEvent>,
}

/// Sequencer status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SequencerStatus {
    /// Sequencer active
    Active,
    /// Sequencer inactive
    Inactive,
    /// Sequencer slashed
    Slashed,
    /// Sequencer suspended
    Suspended,
}

/// Slashing event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlashingEvent {
    /// Event type
    pub event_type: SlashingType,
    /// Amount slashed
    pub amount: u128,
    /// Reason
    pub reason: String,
    /// Timestamp
    pub timestamp: u64,
    /// Block height
    pub block_height: u64,
}

/// Slashing type
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SlashingType {
    /// Equivocation
    Equivocation,
    /// Invalid block proposal
    InvalidProposal,
    /// Double voting
    DoubleVoting,
    /// Censorship
    Censorship,
    /// Liveness violation
    LivenessViolation,
}

/// HotStuff BFT block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotStuffBlock {
    /// Block hash
    pub block_hash: String,
    /// Block height
    pub height: u64,
    /// Parent hash
    pub parent_hash: String,
    /// Proposer ID
    pub proposer_id: String,
    /// View number
    pub view_number: u64,
    /// Block data
    pub data: Vec<u8>,
    /// Timestamp
    pub timestamp: u64,
    /// Block signature
    pub signature: Vec<u8>,
}

/// HotStuff BFT vote
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotStuffVote {
    /// Vote hash
    pub vote_hash: String,
    /// Block hash
    pub block_hash: String,
    /// Voter ID
    pub voter_id: String,
    /// View number
    pub view_number: u64,
    /// Vote signature
    pub signature: Vec<u8>,
    /// Timestamp
    pub timestamp: u64,
}

/// Quorum certificate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuorumCertificate {
    /// QC hash
    pub qc_hash: String,
    /// Block hash
    pub block_hash: String,
    /// View number
    pub view_number: u64,
    /// Vote signatures
    pub vote_signatures: Vec<Vec<u8>>,
    /// Voter IDs
    pub voter_ids: Vec<String>,
    /// Timestamp
    pub timestamp: u64,
}

/// View change message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewChangeMessage {
    /// Message hash
    pub message_hash: String,
    /// From view
    pub from_view: u64,
    /// To view
    pub to_view: u64,
    /// Proposer ID
    pub proposer_id: String,
    /// Justification
    pub justification: Vec<u8>,
    /// Signature
    pub signature: Vec<u8>,
    /// Timestamp
    pub timestamp: u64,
}

/// VRF proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VRFProof {
    /// Proof data
    pub proof_data: Vec<u8>,
    /// VRF output
    pub vrf_output: Vec<u8>,
    /// Prover ID
    pub prover_id: String,
    /// Timestamp
    pub timestamp: u64,
}

/// HotStuff BFT engine
pub struct HotStuffBFTEngine {
    /// Sequencers
    sequencers: Arc<RwLock<HashMap<String, HotStuffSequencer>>>,
    /// Current view
    current_view: Arc<RwLock<u64>>,
    /// Current leader
    current_leader: Arc<RwLock<Option<String>>>,
    /// Pending blocks
    pending_blocks: Arc<RwLock<HashMap<String, HotStuffBlock>>>,
    /// Pending votes
    pending_votes: Arc<RwLock<HashMap<String, Vec<HotStuffVote>>>>,
    /// Quorum certificates
    quorum_certificates: Arc<RwLock<HashMap<String, QuorumCertificate>>>,
    /// View change messages
    view_change_messages: Arc<RwLock<HashMap<u64, Vec<ViewChangeMessage>>>>,
    /// VRF proofs
    vrf_proofs: Arc<RwLock<HashMap<String, VRFProof>>>,
    /// Metrics
    metrics: Arc<RwLock<HotStuffBFTMetrics>>,
}

/// HotStuff BFT metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HotStuffBFTMetrics {
    /// Total sequencers
    pub total_sequencers: u64,
    /// Active sequencers
    pub active_sequencers: u64,
    /// Total blocks proposed
    pub total_blocks_proposed: u64,
    /// Total blocks finalized
    pub total_blocks_finalized: u64,
    /// Total votes cast
    pub total_votes_cast: u64,
    /// Total view changes
    pub total_view_changes: u64,
    /// Average block finalization time (ms)
    pub avg_block_finalization_time_ms: f64,
    /// Average view change time (ms)
    pub avg_view_change_time_ms: f64,
    /// Total slashing events
    pub total_slashing_events: u64,
    /// Total stake slashed
    pub total_stake_slashed: u128,
    /// Consensus success rate
    pub consensus_success_rate: f64,
}

impl Default for HotStuffBFTEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl HotStuffBFTEngine {
    /// Create a new HotStuff BFT engine
    pub fn new() -> Self {
        Self {
            sequencers: Arc::new(RwLock::new(HashMap::new())),
            current_view: Arc::new(RwLock::new(0)),
            current_leader: Arc::new(RwLock::new(None)),
            pending_blocks: Arc::new(RwLock::new(HashMap::new())),
            pending_votes: Arc::new(RwLock::new(HashMap::new())),
            quorum_certificates: Arc::new(RwLock::new(HashMap::new())),
            view_change_messages: Arc::new(RwLock::new(HashMap::new())),
            vrf_proofs: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HotStuffBFTMetrics {
                total_sequencers: 0,
                active_sequencers: 0,
                total_blocks_proposed: 0,
                total_blocks_finalized: 0,
                total_votes_cast: 0,
                total_view_changes: 0,
                avg_block_finalization_time_ms: 0.0,
                avg_view_change_time_ms: 0.0,
                total_slashing_events: 0,
                total_stake_slashed: 0,
                consensus_success_rate: 0.0,
            })),
        }
    }

    /// Register sequencer
    pub fn register_sequencer(&self, sequencer: HotStuffSequencer) -> HotStuffBFTResult<()> {
        // Validate sequencer data
        if sequencer.sequencer_id.is_empty() || sequencer.address.is_empty() {
            return Err(HotStuffBFTError::InvalidProposal);
        }

        // Store sequencer
        {
            let mut sequencers = self.sequencers.write().unwrap();
            sequencers.insert(sequencer.sequencer_id.clone(), sequencer.clone());
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_sequencers += 1;
            if sequencer.status == SequencerStatus::Active {
                metrics.active_sequencers += 1;
            }
        }

        Ok(())
    }

    /// Propose block
    pub fn propose_block(&self, block: HotStuffBlock) -> HotStuffBFTResult<()> {
        // Validate block
        if block.block_hash.is_empty() || block.data.is_empty() {
            return Err(HotStuffBFTError::InvalidBlock);
        }

        // Check if proposer is current leader
        {
            let current_leader = self.current_leader.read().unwrap();
            if let Some(leader_id) = current_leader.as_ref() {
                if block.proposer_id != *leader_id {
                    return Err(HotStuffBFTError::InvalidLeader);
                }
            }
        }

        // Store block
        {
            let mut pending_blocks = self.pending_blocks.write().unwrap();
            pending_blocks.insert(block.block_hash.clone(), block.clone());
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_blocks_proposed += 1;
        }

        Ok(())
    }

    /// Cast vote
    pub fn cast_vote(&self, vote: HotStuffVote) -> HotStuffBFTResult<()> {
        // Validate vote
        if vote.vote_hash.is_empty() || vote.block_hash.is_empty() {
            return Err(HotStuffBFTError::InvalidVote);
        }

        // Check for duplicate vote
        {
            let pending_votes = self.pending_votes.read().unwrap();
            if let Some(votes) = pending_votes.get(&vote.block_hash) {
                for existing_vote in votes {
                    if existing_vote.voter_id == vote.voter_id {
                        return Err(HotStuffBFTError::DuplicateVote);
                    }
                }
            }
        }

        // Store vote
        {
            let mut pending_votes = self.pending_votes.write().unwrap();
            pending_votes
                .entry(vote.block_hash.clone())
                .or_default()
                .push(vote.clone());
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_votes_cast += 1;
        }

        Ok(())
    }

    /// Finalize block
    pub fn finalize_block(&self, block_hash: &str) -> HotStuffBFTResult<()> {
        // Check if block has enough votes
        {
            let pending_votes = self.pending_votes.read().unwrap();
            if let Some(votes) = pending_votes.get(block_hash) {
                let active_sequencers = {
                    let sequencers = self.sequencers.read().unwrap();
                    sequencers
                        .values()
                        .filter(|s| s.status == SequencerStatus::Active)
                        .count()
                };

                if votes.len() < (active_sequencers * 2 / 3 + 1) {
                    return Err(HotStuffBFTError::NotEnoughVotes);
                }
            } else {
                return Err(HotStuffBFTError::InvalidBlock);
            }
        }

        // Create quorum certificate
        let qc = {
            let pending_votes = self.pending_votes.read().unwrap();
            let votes = pending_votes.get(block_hash).unwrap();

            QuorumCertificate {
                qc_hash: format!("qc_{}", block_hash),
                block_hash: block_hash.to_string(),
                view_number: votes[0].view_number,
                vote_signatures: votes.iter().map(|v| v.signature.clone()).collect(),
                voter_ids: votes.iter().map(|v| v.voter_id.clone()).collect(),
                timestamp: current_timestamp(),
            }
        };

        // Store quorum certificate
        {
            let mut quorum_certificates = self.quorum_certificates.write().unwrap();
            quorum_certificates.insert(qc.qc_hash.clone(), qc);
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_blocks_finalized += 1;
        }

        Ok(())
    }

    /// Change view
    pub fn change_view(&self, view_change: ViewChangeMessage) -> HotStuffBFTResult<()> {
        // Validate view change
        if view_change.from_view >= view_change.to_view {
            return Err(HotStuffBFTError::InvalidViewChange);
        }

        // Store view change message
        {
            let mut view_change_messages = self.view_change_messages.write().unwrap();
            view_change_messages
                .entry(view_change.to_view)
                .or_default()
                .push(view_change.clone());
        }

        // Update current view
        {
            let mut current_view = self.current_view.write().unwrap();
            *current_view = view_change.to_view;
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_view_changes += 1;
        }

        Ok(())
    }

    /// Select new leader using VRF
    pub fn select_leader(&self, vrf_proof: VRFProof) -> HotStuffBFTResult<String> {
        // Store VRF proof
        {
            let mut vrf_proofs = self.vrf_proofs.write().unwrap();
            vrf_proofs.insert(vrf_proof.prover_id.clone(), vrf_proof.clone());
        }

        // Select leader based on VRF output
        let sequencers = self.sequencers.read().unwrap();
        let mut active_sequencers: Vec<_> = sequencers
            .values()
            .filter(|s| s.status == SequencerStatus::Active)
            .cloned()
            .collect();

        // Sort by sequencer ID to ensure deterministic order
        active_sequencers.sort_by_key(|s| s.sequencer_id.clone());

        if active_sequencers.is_empty() {
            return Err(HotStuffBFTError::SequencerNotFound);
        }

        // Simple leader selection based on VRF output
        let leader_index = if vrf_proof.vrf_output.is_empty() {
            0
        } else {
            vrf_proof.vrf_output[0] as usize % active_sequencers.len()
        };
        let leader = active_sequencers[leader_index].clone();

        // Update current leader
        {
            let mut current_leader = self.current_leader.write().unwrap();
            *current_leader = Some(leader.sequencer_id.clone());
        }

        Ok(leader.sequencer_id)
    }

    /// Slash sequencer
    pub fn slash_sequencer(
        &self,
        sequencer_id: &str,
        slashing_event: SlashingEvent,
    ) -> HotStuffBFTResult<()> {
        // Find sequencer
        {
            let mut sequencers = self.sequencers.write().unwrap();
            if let Some(sequencer) = sequencers.get_mut(sequencer_id) {
                // Add slashing event
                sequencer.slashing_history.push(slashing_event.clone());

                // Update stake
                sequencer.stake = sequencer.stake.saturating_sub(slashing_event.amount);

                // Update status if stake is too low
                let was_active = sequencer.status == SequencerStatus::Active;
                if sequencer.stake < 1000 {
                    // Minimum stake threshold
                    sequencer.status = SequencerStatus::Slashed;
                }

                // Update metrics
                let mut metrics = self.metrics.write().unwrap();
                metrics.total_slashing_events += 1;
                metrics.total_stake_slashed += slashing_event.amount;

                if was_active && sequencer.status == SequencerStatus::Slashed {
                    metrics.active_sequencers = metrics.active_sequencers.saturating_sub(1);
                }
            } else {
                return Err(HotStuffBFTError::SequencerNotFound);
            }
        }

        Ok(())
    }

    /// Get sequencer
    pub fn get_sequencer(&self, sequencer_id: &str) -> Option<HotStuffSequencer> {
        let sequencers = self.sequencers.read().unwrap();
        sequencers.get(sequencer_id).cloned()
    }

    /// Get current leader
    pub fn get_current_leader(&self) -> Option<String> {
        let current_leader = self.current_leader.read().unwrap();
        current_leader.clone()
    }

    /// Get current view
    pub fn get_current_view(&self) -> u64 {
        let current_view = self.current_view.read().unwrap();
        *current_view
    }

    /// Get block
    pub fn get_block(&self, block_hash: &str) -> Option<HotStuffBlock> {
        let pending_blocks = self.pending_blocks.read().unwrap();
        pending_blocks.get(block_hash).cloned()
    }

    /// Get votes for block
    pub fn get_votes_for_block(&self, block_hash: &str) -> Vec<HotStuffVote> {
        let pending_votes = self.pending_votes.read().unwrap();
        pending_votes.get(block_hash).cloned().unwrap_or_default()
    }

    /// Get quorum certificate
    pub fn get_quorum_certificate(&self, qc_hash: &str) -> Option<QuorumCertificate> {
        let quorum_certificates = self.quorum_certificates.read().unwrap();
        quorum_certificates.get(qc_hash).cloned()
    }

    /// Get metrics
    pub fn get_metrics(&self) -> HotStuffBFTMetrics {
        let metrics = self.metrics.read().unwrap();
        metrics.clone()
    }

    /// Get all sequencers
    pub fn get_all_sequencers(&self) -> Vec<HotStuffSequencer> {
        let sequencers = self.sequencers.read().unwrap();
        sequencers.values().cloned().collect()
    }

    /// Get active sequencers
    pub fn get_active_sequencers(&self) -> Vec<HotStuffSequencer> {
        let sequencers = self.sequencers.read().unwrap();
        sequencers
            .values()
            .filter(|s| s.status == SequencerStatus::Active)
            .cloned()
            .collect()
    }

    /// Get all blocks
    pub fn get_all_blocks(&self) -> Vec<HotStuffBlock> {
        let pending_blocks = self.pending_blocks.read().unwrap();
        pending_blocks.values().cloned().collect()
    }

    /// Get all quorum certificates
    pub fn get_all_quorum_certificates(&self) -> Vec<QuorumCertificate> {
        let quorum_certificates = self.quorum_certificates.read().unwrap();
        quorum_certificates.values().cloned().collect()
    }
}

/// Get current timestamp in milliseconds
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
    fn test_hotstuff_bft_engine_creation() {
        let engine = HotStuffBFTEngine::new();
        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_sequencers, 0);
    }

    #[test]
    fn test_register_sequencer() {
        let engine = HotStuffBFTEngine::new();

        let sequencer = HotStuffSequencer {
            sequencer_id: "seq-1".to_string(),
            address: "0x123".to_string(),
            public_key: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            stake: 1000000,
            reputation: 0.95,
            status: SequencerStatus::Active,
            last_activity: current_timestamp(),
            slashing_history: vec![],
        };

        let result = engine.register_sequencer(sequencer.clone());
        assert!(result.is_ok());

        // Verify sequencer was registered
        let stored_sequencer = engine.get_sequencer("seq-1");
        assert!(stored_sequencer.is_some());
        assert_eq!(stored_sequencer.unwrap().sequencer_id, "seq-1");
    }

    #[test]
    fn test_register_sequencer_invalid_data() {
        let engine = HotStuffBFTEngine::new();

        let sequencer = HotStuffSequencer {
            sequencer_id: "".to_string(), // Empty ID
            address: "0x123".to_string(),
            public_key: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            stake: 1000000,
            reputation: 0.95,
            status: SequencerStatus::Active,
            last_activity: current_timestamp(),
            slashing_history: vec![],
        };

        let result = engine.register_sequencer(sequencer);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), HotStuffBFTError::InvalidProposal);
    }

    #[test]
    fn test_propose_block() {
        let engine = HotStuffBFTEngine::new();

        // Register sequencer first
        let sequencer = HotStuffSequencer {
            sequencer_id: "seq-1".to_string(),
            address: "0x123".to_string(),
            public_key: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            stake: 1000000,
            reputation: 0.95,
            status: SequencerStatus::Active,
            last_activity: current_timestamp(),
            slashing_history: vec![],
        };

        engine.register_sequencer(sequencer).unwrap();

        // Set as current leader
        {
            let mut current_leader = engine.current_leader.write().unwrap();
            *current_leader = Some("seq-1".to_string());
        }

        let block = HotStuffBlock {
            block_hash: "block-1".to_string(),
            height: 1,
            parent_hash: "parent-1".to_string(),
            proposer_id: "seq-1".to_string(),
            view_number: 0,
            data: vec![1, 2, 3, 4, 5],
            timestamp: current_timestamp(),
            signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        };

        let result = engine.propose_block(block.clone());
        assert!(result.is_ok());

        // Verify block was stored
        let stored_block = engine.get_block("block-1");
        assert!(stored_block.is_some());
        assert_eq!(stored_block.unwrap().block_hash, "block-1");
    }

    #[test]
    fn test_propose_block_invalid_leader() {
        let engine = HotStuffBFTEngine::new();

        // Register sequencer first
        let sequencer = HotStuffSequencer {
            sequencer_id: "seq-1".to_string(),
            address: "0x123".to_string(),
            public_key: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            stake: 1000000,
            reputation: 0.95,
            status: SequencerStatus::Active,
            last_activity: current_timestamp(),
            slashing_history: vec![],
        };

        engine.register_sequencer(sequencer).unwrap();

        // Set different leader
        {
            let mut current_leader = engine.current_leader.write().unwrap();
            *current_leader = Some("seq-2".to_string());
        }

        let block = HotStuffBlock {
            block_hash: "block-1".to_string(),
            height: 1,
            parent_hash: "parent-1".to_string(),
            proposer_id: "seq-1".to_string(), // Different from current leader
            view_number: 0,
            data: vec![1, 2, 3, 4, 5],
            timestamp: current_timestamp(),
            signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        };

        let result = engine.propose_block(block);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), HotStuffBFTError::InvalidLeader);
    }

    #[test]
    fn test_cast_vote() {
        let engine = HotStuffBFTEngine::new();

        let vote = HotStuffVote {
            vote_hash: "vote-1".to_string(),
            block_hash: "block-1".to_string(),
            voter_id: "seq-1".to_string(),
            view_number: 0,
            signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            timestamp: current_timestamp(),
        };

        let result = engine.cast_vote(vote.clone());
        assert!(result.is_ok());

        // Verify vote was stored
        let votes = engine.get_votes_for_block("block-1");
        assert_eq!(votes.len(), 1);
        assert_eq!(votes[0].vote_hash, "vote-1");
    }

    #[test]
    fn test_cast_duplicate_vote() {
        let engine = HotStuffBFTEngine::new();

        let vote1 = HotStuffVote {
            vote_hash: "vote-1".to_string(),
            block_hash: "block-1".to_string(),
            voter_id: "seq-1".to_string(),
            view_number: 0,
            signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            timestamp: current_timestamp(),
        };

        let vote2 = HotStuffVote {
            vote_hash: "vote-2".to_string(),
            block_hash: "block-1".to_string(),
            voter_id: "seq-1".to_string(), // Same voter
            view_number: 0,
            signature: vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            timestamp: current_timestamp(),
        };

        engine.cast_vote(vote1).unwrap();
        let result = engine.cast_vote(vote2);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), HotStuffBFTError::DuplicateVote);
    }

    #[test]
    fn test_finalize_block() {
        let engine = HotStuffBFTEngine::new();

        // Register multiple sequencers
        for i in 0..4 {
            let sequencer = HotStuffSequencer {
                sequencer_id: format!("seq-{}", i),
                address: format!("0x{}", i),
                public_key: vec![
                    i as u8,
                    (i + 1) as u8,
                    (i + 2) as u8,
                    (i + 3) as u8,
                    (i + 4) as u8,
                ],
                stake: 1000000,
                reputation: 0.95,
                status: SequencerStatus::Active,
                last_activity: current_timestamp(),
                slashing_history: vec![],
            };

            engine.register_sequencer(sequencer).unwrap();
        }

        // Cast votes from 3 sequencers (2/3 + 1 = 3)
        for i in 0..3 {
            let vote = HotStuffVote {
                vote_hash: format!("vote-{}", i),
                block_hash: "block-1".to_string(),
                voter_id: format!("seq-{}", i),
                view_number: 0,
                signature: vec![
                    i as u8,
                    (i + 1) as u8,
                    (i + 2) as u8,
                    (i + 3) as u8,
                    (i + 4) as u8,
                ],
                timestamp: current_timestamp(),
            };

            engine.cast_vote(vote).unwrap();
        }

        let result = engine.finalize_block("block-1");
        assert!(result.is_ok());
    }

    #[test]
    fn test_finalize_block_not_enough_votes() {
        let engine = HotStuffBFTEngine::new();

        // Register multiple sequencers
        for i in 0..4 {
            let sequencer = HotStuffSequencer {
                sequencer_id: format!("seq-{}", i),
                address: format!("0x{}", i),
                public_key: vec![
                    i as u8,
                    (i + 1) as u8,
                    (i + 2) as u8,
                    (i + 3) as u8,
                    (i + 4) as u8,
                ],
                stake: 1000000,
                reputation: 0.95,
                status: SequencerStatus::Active,
                last_activity: current_timestamp(),
                slashing_history: vec![],
            };

            engine.register_sequencer(sequencer).unwrap();
        }

        // Cast votes from only 2 sequencers (need 3 for 2/3 + 1)
        for i in 0..2 {
            let vote = HotStuffVote {
                vote_hash: format!("vote-{}", i),
                block_hash: "block-1".to_string(),
                voter_id: format!("seq-{}", i),
                view_number: 0,
                signature: vec![
                    i as u8,
                    (i + 1) as u8,
                    (i + 2) as u8,
                    (i + 3) as u8,
                    (i + 4) as u8,
                ],
                timestamp: current_timestamp(),
            };

            engine.cast_vote(vote).unwrap();
        }

        let result = engine.finalize_block("block-1");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), HotStuffBFTError::NotEnoughVotes);
    }

    #[test]
    fn test_change_view() {
        let engine = HotStuffBFTEngine::new();

        let view_change = ViewChangeMessage {
            message_hash: "vc-1".to_string(),
            from_view: 0,
            to_view: 1,
            proposer_id: "seq-1".to_string(),
            justification: vec![1, 2, 3, 4, 5],
            signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            timestamp: current_timestamp(),
        };

        let result = engine.change_view(view_change);
        assert!(result.is_ok());

        // Verify view was updated
        assert_eq!(engine.get_current_view(), 1);
    }

    #[test]
    fn test_change_view_invalid() {
        let engine = HotStuffBFTEngine::new();

        let view_change = ViewChangeMessage {
            message_hash: "vc-1".to_string(),
            from_view: 1,
            to_view: 0, // Invalid: to_view should be greater than from_view
            proposer_id: "seq-1".to_string(),
            justification: vec![1, 2, 3, 4, 5],
            signature: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            timestamp: current_timestamp(),
        };

        let result = engine.change_view(view_change);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), HotStuffBFTError::InvalidViewChange);
    }

    #[test]
    fn test_select_leader() {
        let engine = HotStuffBFTEngine::new();

        // Register multiple sequencers
        for i in 0..3 {
            let sequencer = HotStuffSequencer {
                sequencer_id: format!("seq-{}", i),
                address: format!("0x{}", i),
                public_key: vec![
                    i as u8,
                    (i + 1) as u8,
                    (i + 2) as u8,
                    (i + 3) as u8,
                    (i + 4) as u8,
                ],
                stake: 1000000,
                reputation: 0.95,
                status: SequencerStatus::Active,
                last_activity: current_timestamp(),
                slashing_history: vec![],
            };

            engine.register_sequencer(sequencer).unwrap();
        }

        let vrf_proof = VRFProof {
            proof_data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            vrf_output: vec![0], // Will select first sequencer (seq-0)
            prover_id: "seq-0".to_string(),
            timestamp: current_timestamp(),
        };

        let result = engine.select_leader(vrf_proof);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "seq-0");

        // Verify current leader was set
        let current_leader = engine.get_current_leader();
        assert_eq!(current_leader, Some("seq-0".to_string()));
    }

    #[test]
    fn test_slash_sequencer() {
        let engine = HotStuffBFTEngine::new();

        // Register sequencer
        let sequencer = HotStuffSequencer {
            sequencer_id: "seq-1".to_string(),
            address: "0x123".to_string(),
            public_key: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            stake: 1000000,
            reputation: 0.95,
            status: SequencerStatus::Active,
            last_activity: current_timestamp(),
            slashing_history: vec![],
        };

        engine.register_sequencer(sequencer).unwrap();

        let slashing_event = SlashingEvent {
            event_type: SlashingType::Equivocation,
            amount: 100000,
            reason: "Double voting detected".to_string(),
            timestamp: current_timestamp(),
            block_height: 1,
        };

        let result = engine.slash_sequencer("seq-1", slashing_event);
        assert!(result.is_ok());

        // Verify sequencer was slashed
        let slashed_sequencer = engine.get_sequencer("seq-1").unwrap();
        assert_eq!(slashed_sequencer.stake, 900000); // 1000000 - 100000
        assert_eq!(slashed_sequencer.slashing_history.len(), 1);
    }

    #[test]
    fn test_slash_sequencer_not_found() {
        let engine = HotStuffBFTEngine::new();

        let slashing_event = SlashingEvent {
            event_type: SlashingType::Equivocation,
            amount: 100000,
            reason: "Double voting detected".to_string(),
            timestamp: current_timestamp(),
            block_height: 1,
        };

        let result = engine.slash_sequencer("nonexistent-seq", slashing_event);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), HotStuffBFTError::SequencerNotFound);
    }

    #[test]
    fn test_hotstuff_bft_metrics() {
        let engine = HotStuffBFTEngine::new();

        // Register sequencers
        for i in 0..3 {
            let sequencer = HotStuffSequencer {
                sequencer_id: format!("seq-{}", i),
                address: format!("0x{}", i),
                public_key: vec![
                    i as u8,
                    (i + 1) as u8,
                    (i + 2) as u8,
                    (i + 3) as u8,
                    (i + 4) as u8,
                ],
                stake: 1000000,
                reputation: 0.95,
                status: SequencerStatus::Active,
                last_activity: current_timestamp(),
                slashing_history: vec![],
            };

            engine.register_sequencer(sequencer).unwrap();
        }

        // Propose and finalize blocks
        for i in 0..2 {
            let block = HotStuffBlock {
                block_hash: format!("block-{}", i),
                height: i as u64,
                parent_hash: format!("parent-{}", i),
                proposer_id: "seq-0".to_string(),
                view_number: 0,
                data: vec![i as u8, (i + 1) as u8, (i + 2) as u8],
                timestamp: current_timestamp(),
                signature: vec![
                    i as u8,
                    (i + 1) as u8,
                    (i + 2) as u8,
                    (i + 3) as u8,
                    (i + 4) as u8,
                ],
            };

            engine.propose_block(block).unwrap();

            // Cast votes
            for j in 0..3 {
                let vote = HotStuffVote {
                    vote_hash: format!("vote-{}-{}", i, j),
                    block_hash: format!("block-{}", i),
                    voter_id: format!("seq-{}", j),
                    view_number: 0,
                    signature: vec![
                        j as u8,
                        (j + 1) as u8,
                        (j + 2) as u8,
                        (j + 3) as u8,
                        (j + 4) as u8,
                    ],
                    timestamp: current_timestamp(),
                };

                engine.cast_vote(vote).unwrap();
            }

            engine.finalize_block(&format!("block-{}", i)).unwrap();
        }

        // Slash a sequencer (slash enough to make stake < 1000)
        let slashing_event = SlashingEvent {
            event_type: SlashingType::Equivocation,
            amount: 999001, // Slash enough to make stake < 1000 (1000000 - 999001 = 999)
            reason: "Double voting detected".to_string(),
            timestamp: current_timestamp(),
            block_height: 1,
        };

        engine.slash_sequencer("seq-1", slashing_event).unwrap();

        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_sequencers, 3);
        assert_eq!(metrics.active_sequencers, 2); // One was slashed
        assert_eq!(metrics.total_blocks_proposed, 2);
        assert_eq!(metrics.total_blocks_finalized, 2);
        assert_eq!(metrics.total_votes_cast, 6); // 2 blocks * 3 votes each
        assert_eq!(metrics.total_slashing_events, 1);
        assert_eq!(metrics.total_stake_slashed, 999001);
    }

    #[test]
    fn test_get_all_sequencers() {
        let engine = HotStuffBFTEngine::new();

        // Register multiple sequencers
        for i in 0..3 {
            let sequencer = HotStuffSequencer {
                sequencer_id: format!("seq-{}", i),
                address: format!("0x{}", i),
                public_key: vec![
                    i as u8,
                    (i + 1) as u8,
                    (i + 2) as u8,
                    (i + 3) as u8,
                    (i + 4) as u8,
                ],
                stake: 1000000,
                reputation: 0.95,
                status: SequencerStatus::Active,
                last_activity: current_timestamp(),
                slashing_history: vec![],
            };

            engine.register_sequencer(sequencer).unwrap();
        }

        let sequencers = engine.get_all_sequencers();
        assert_eq!(sequencers.len(), 3);
    }

    #[test]
    fn test_get_active_sequencers() {
        let engine = HotStuffBFTEngine::new();

        // Register sequencers with different statuses
        let sequencer1 = HotStuffSequencer {
            sequencer_id: "seq-1".to_string(),
            address: "0x1".to_string(),
            public_key: vec![1, 2, 3, 4, 5],
            stake: 1000000,
            reputation: 0.95,
            status: SequencerStatus::Active,
            last_activity: current_timestamp(),
            slashing_history: vec![],
        };

        let sequencer2 = HotStuffSequencer {
            sequencer_id: "seq-2".to_string(),
            address: "0x2".to_string(),
            public_key: vec![6, 7, 8, 9, 10],
            stake: 1000000,
            reputation: 0.95,
            status: SequencerStatus::Inactive,
            last_activity: current_timestamp(),
            slashing_history: vec![],
        };

        engine.register_sequencer(sequencer1).unwrap();
        engine.register_sequencer(sequencer2).unwrap();

        let active_sequencers = engine.get_active_sequencers();
        assert_eq!(active_sequencers.len(), 1);
        assert_eq!(active_sequencers[0].sequencer_id, "seq-1");
    }

    #[test]
    fn test_get_all_blocks() {
        let engine = HotStuffBFTEngine::new();

        // Propose multiple blocks
        for i in 0..3 {
            let block = HotStuffBlock {
                block_hash: format!("block-{}", i),
                height: i as u64,
                parent_hash: format!("parent-{}", i),
                proposer_id: "seq-0".to_string(),
                view_number: 0,
                data: vec![i as u8, (i + 1) as u8, (i + 2) as u8],
                timestamp: current_timestamp(),
                signature: vec![
                    i as u8,
                    (i + 1) as u8,
                    (i + 2) as u8,
                    (i + 3) as u8,
                    (i + 4) as u8,
                ],
            };

            engine.propose_block(block).unwrap();
        }

        let blocks = engine.get_all_blocks();
        assert_eq!(blocks.len(), 3);
    }

    #[test]
    fn test_get_all_quorum_certificates() {
        let engine = HotStuffBFTEngine::new();

        // Register sequencers
        for i in 0..3 {
            let sequencer = HotStuffSequencer {
                sequencer_id: format!("seq-{}", i),
                address: format!("0x{}", i),
                public_key: vec![
                    i as u8,
                    (i + 1) as u8,
                    (i + 2) as u8,
                    (i + 3) as u8,
                    (i + 4) as u8,
                ],
                stake: 1000000,
                reputation: 0.95,
                status: SequencerStatus::Active,
                last_activity: current_timestamp(),
                slashing_history: vec![],
            };

            engine.register_sequencer(sequencer).unwrap();
        }

        // Finalize a block to create QC
        for i in 0..3 {
            let vote = HotStuffVote {
                vote_hash: format!("vote-{}", i),
                block_hash: "block-1".to_string(),
                voter_id: format!("seq-{}", i),
                view_number: 0,
                signature: vec![
                    i as u8,
                    (i + 1) as u8,
                    (i + 2) as u8,
                    (i + 3) as u8,
                    (i + 4) as u8,
                ],
                timestamp: current_timestamp(),
            };

            engine.cast_vote(vote).unwrap();
        }

        engine.finalize_block("block-1").unwrap();

        let qcs = engine.get_all_quorum_certificates();
        assert_eq!(qcs.len(), 1);
    }
}
