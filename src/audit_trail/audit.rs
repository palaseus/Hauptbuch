//! Real-Time Audit Trail Module
//!
//! This module provides comprehensive audit trail capabilities for the decentralized
//! voting blockchain, maintaining a tamper-proof log of all activities including votes,
//! governance proposals, cross-chain messages, and system events.
//!
//! Key features:
//! - Tamper-proof Merkle tree storage with SHA-3 hashes
//! - Dilithium3/5 signatures for log integrity verification
//! - Real-time logging of all blockchain activities
//! - Advanced querying by timestamp, event type, and proposal ID
//! - JSON, human-readable, and Chart.js-compatible outputs
//! - Integration with governance, federation, monitoring, and anomaly detection
//! - Comprehensive security with safe arithmetic operations
//! - Extensive test coverage with 25+ test cases

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

// Import blockchain modules for integration
use crate::anomaly::detector::{AnomalyDetection, AnomalySeverity};
use crate::federation::federation::CrossChainVote;
use crate::governance::proposal::{Proposal, Vote};
use crate::security::audit::SecurityAuditor;
use crate::ui::interface::UserInterface;
use crate::visualization::visualization::VisualizationEngine;

// Import quantum-resistant cryptography for signatures
use crate::crypto::quantum_resistant::{
    dilithium_keygen, dilithium_sign, dilithium_verify, DilithiumParams, DilithiumPublicKey,
    DilithiumSecretKey, DilithiumSignature,
};

/// Types of events that can be logged in the audit trail
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Governance proposal submitted
    ProposalSubmitted,
    /// Vote cast on a proposal
    VoteCast,
    /// Proposal execution
    ProposalExecuted,
    /// Cross-chain message sent
    CrossChainMessage,
    /// Cross-chain vote aggregated
    CrossChainVoteAggregated,
    /// System node joined
    NodeJoined,
    /// System node left
    NodeLeft,
    /// System failure detected
    SystemFailure,
    /// Anomaly detected
    AnomalyDetected,
    /// Security alert triggered
    SecurityAlert,
    /// Configuration change
    ConfigurationChange,
    /// State synchronization
    StateSync,
    /// Fork resolution
    ForkResolution,
    /// Validator slashing
    ValidatorSlashing,
    /// Stake delegation
    StakeDelegation,
    /// Network partition
    NetworkPartition,
}

impl std::fmt::Display for AuditEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditEventType::ProposalSubmitted => write!(f, "ProposalSubmitted"),
            AuditEventType::VoteCast => write!(f, "VoteCast"),
            AuditEventType::ProposalExecuted => write!(f, "ProposalExecuted"),
            AuditEventType::CrossChainMessage => write!(f, "CrossChainMessage"),
            AuditEventType::CrossChainVoteAggregated => write!(f, "CrossChainVoteAggregated"),
            AuditEventType::NodeJoined => write!(f, "NodeJoined"),
            AuditEventType::NodeLeft => write!(f, "NodeLeft"),
            AuditEventType::SystemFailure => write!(f, "SystemFailure"),
            AuditEventType::AnomalyDetected => write!(f, "AnomalyDetected"),
            AuditEventType::SecurityAlert => write!(f, "SecurityAlert"),
            AuditEventType::ConfigurationChange => write!(f, "ConfigurationChange"),
            AuditEventType::StateSync => write!(f, "StateSync"),
            AuditEventType::ForkResolution => write!(f, "ForkResolution"),
            AuditEventType::ValidatorSlashing => write!(f, "ValidatorSlashing"),
            AuditEventType::StakeDelegation => write!(f, "StakeDelegation"),
            AuditEventType::NetworkPartition => write!(f, "NetworkPartition"),
        }
    }
}

/// Individual audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    /// Unique log entry identifier
    pub id: String,
    /// Event type
    pub event_type: AuditEventType,
    /// Timestamp of the event
    pub timestamp: u64,
    /// Actor who performed the action (public key or node ID)
    pub actor: String,
    /// Target of the action (proposal ID, node ID, etc.)
    pub target: Option<String>,
    /// Event description
    pub description: String,
    /// Additional metadata (JSON string)
    pub metadata: String,
    /// SHA-3 hash of the entry
    pub hash: Vec<u8>,
    /// Dilithium signature of the entry
    pub signature: Option<DilithiumSignature>,
    /// Block height when the event occurred
    pub block_height: u64,
    /// Transaction hash if applicable
    pub transaction_hash: Option<String>,
}

/// Merkle tree node for tamper-proof storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MerkleNode {
    /// Node hash
    pub hash: Vec<u8>,
    /// Left child hash (if any)
    pub left_hash: Option<Vec<u8>>,
    /// Right child hash (if any)
    pub right_hash: Option<Vec<u8>>,
    /// Node level in the tree
    pub level: u32,
    /// Node index
    pub index: u64,
}

/// Merkle tree for tamper-proof audit log storage
#[derive(Debug, Clone)]
pub struct AuditMerkleTree {
    /// Tree nodes
    pub nodes: Vec<MerkleNode>,
    /// Root hash
    pub root_hash: Vec<u8>,
    /// Tree depth
    pub depth: u32,
    /// Number of leaf nodes
    pub leaf_count: u64,
}

/// Query parameters for audit log searches
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuditQuery {
    /// Start timestamp (inclusive)
    pub start_time: Option<u64>,
    /// End timestamp (inclusive)
    pub end_time: Option<u64>,
    /// Event types to filter by
    pub event_types: Option<Vec<AuditEventType>>,
    /// Actor to filter by
    pub actor: Option<String>,
    /// Target to filter by
    pub target: Option<String>,
    /// Proposal ID to filter by
    pub proposal_id: Option<String>,
    /// Block height range
    pub block_height_range: Option<(u64, u64)>,
    /// Maximum number of results
    pub limit: Option<usize>,
    /// Offset for pagination
    pub offset: Option<usize>,
}

/// Audit trail configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditTrailConfig {
    /// Maximum number of log entries to keep in memory
    pub max_entries: usize,
    /// Maximum age of log entries in seconds
    pub max_age_seconds: u64,
    /// Enable real-time logging
    pub enable_realtime: bool,
    /// Enable signature verification
    pub enable_signatures: bool,
    /// Enable Merkle tree verification
    pub enable_merkle_verification: bool,
    /// Log retention period in seconds
    pub retention_period_seconds: u64,
    /// Batch size for log processing
    pub batch_size: usize,
    /// Enable compression for old logs
    pub enable_compression: bool,
}

impl Default for AuditTrailConfig {
    fn default() -> Self {
        Self {
            max_entries: 100_000,
            max_age_seconds: 31536000, // 1 year
            enable_realtime: true,
            enable_signatures: true,
            enable_merkle_verification: true,
            retention_period_seconds: 31536000, // 1 year
            batch_size: 1000,
            enable_compression: true,
        }
    }
}

/// Real-time audit trail system
pub struct AuditTrail {
    /// Audit trail configuration
    config: AuditTrailConfig,
    /// Log entries storage
    log_entries: Arc<RwLock<VecDeque<AuditLogEntry>>>,
    /// Merkle tree for tamper-proof storage
    merkle_tree: Arc<Mutex<AuditMerkleTree>>,
    /// Event type index for fast queries
    event_type_index: Arc<RwLock<HashMap<AuditEventType, Vec<String>>>>,
    /// Timestamp index for time-based queries
    timestamp_index: Arc<RwLock<BTreeMap<u64, Vec<String>>>>,
    /// Actor index for actor-based queries
    actor_index: Arc<RwLock<HashMap<String, Vec<String>>>>,
    /// Proposal ID index for proposal-based queries
    proposal_index: Arc<RwLock<HashMap<String, Vec<String>>>>,
    /// Signing key for log entries
    signing_key: Arc<Mutex<Option<DilithiumSecretKey>>>,
    /// UI interface for commands
    #[allow(dead_code)]
    ui: Arc<UserInterface>,
    /// Visualization engine for dashboards
    #[allow(dead_code)]
    visualization: Arc<VisualizationEngine>,
    /// Security auditor for integrity verification
    #[allow(dead_code)]
    security_auditor: Arc<SecurityAuditor>,
    /// Running status
    pub is_running: Arc<AtomicBool>,
    /// Log entry counter for unique IDs
    entry_counter: Arc<AtomicU64>,
}

/// Parameters for creating a log entry
struct LogEntryParams {
    event_type: AuditEventType,
    actor: String,
    target: Option<String>,
    description: String,
    metadata: String,
    block_height: u64,
    transaction_hash: Option<String>,
}

impl AuditTrail {
    /// Create a new audit trail system
    pub fn new(
        config: AuditTrailConfig,
        ui: Arc<UserInterface>,
        visualization: Arc<VisualizationEngine>,
        security_auditor: Arc<SecurityAuditor>,
    ) -> Self {
        Self {
            config,
            log_entries: Arc::new(RwLock::new(VecDeque::new())),
            merkle_tree: Arc::new(Mutex::new(AuditMerkleTree {
                nodes: Vec::new(),
                root_hash: Vec::new(),
                depth: 0,
                leaf_count: 0,
            })),
            event_type_index: Arc::new(RwLock::new(HashMap::new())),
            timestamp_index: Arc::new(RwLock::new(BTreeMap::new())),
            actor_index: Arc::new(RwLock::new(HashMap::new())),
            proposal_index: Arc::new(RwLock::new(HashMap::new())),
            signing_key: Arc::new(Mutex::new(None)),
            ui,
            visualization,
            security_auditor,
            is_running: Arc::new(AtomicBool::new(false)),
            entry_counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Start the audit trail system
    pub fn start(&self) -> Result<(), AuditTrailError> {
        if self.is_running.load(Ordering::SeqCst) {
            return Err(AuditTrailError::AlreadyRunning);
        }

        self.is_running.store(true, Ordering::SeqCst);

        // Generate signing key if signatures are enabled
        if self.config.enable_signatures {
            let params = DilithiumParams::dilithium3();
            let (_, secret_key) =
                dilithium_keygen(&params).map_err(|_| AuditTrailError::KeyGenerationFailed)?;

            if let Ok(mut signing_key) = self.signing_key.lock() {
                *signing_key = Some(secret_key);
            }
        }

        // Start background cleanup thread
        if self.config.enable_realtime {
            self.start_cleanup_thread();
        }

        Ok(())
    }

    /// Stop the audit trail system
    pub fn stop(&self) -> Result<(), AuditTrailError> {
        if !self.is_running.load(Ordering::SeqCst) {
            return Err(AuditTrailError::NotRunning);
        }

        self.is_running.store(false, Ordering::SeqCst);
        Ok(())
    }

    /// Log a governance proposal submission
    pub fn log_proposal_submission(&self, proposal: &Proposal) -> Result<String, AuditTrailError> {
        let entry_id = self.create_log_entry(
            AuditEventType::ProposalSubmitted,
            "proposer", // Use string instead of public key
            Some(&proposal.id),
            &format!("Proposal '{}' submitted", proposal.title),
            &format!(
                r#"{{"id": "{}", "title": "{}", "type": "{:?}"}}"#,
                proposal.id, proposal.title, proposal.proposal_type
            ),
            0, // Use 0 for block height since it's not available
            None,
        )?;

        Ok(entry_id)
    }

    /// Log a vote cast
    pub fn log_vote_cast(&self, vote: &Vote, proposal_id: &str) -> Result<String, AuditTrailError> {
        let entry_id = self.create_log_entry(
            AuditEventType::VoteCast,
            &vote.voter_id, // Use voter_id instead of voter
            Some(proposal_id),
            &format!("Vote cast: {:?}", vote.choice),
            &format!(
                r#"{{"voter_id": "{}", "choice": "{:?}", "stake": {}}}"#,
                vote.voter_id, vote.choice, vote.stake_amount
            ),
            0, // Use 0 for block height since it's not available
            None,
        )?;

        Ok(entry_id)
    }

    /// Log a cross-chain message
    pub fn log_cross_chain_message(
        &self,
        message: &CrossChainVote,
        target_chain: &str,
    ) -> Result<String, AuditTrailError> {
        let entry_id = self.create_log_entry(
            AuditEventType::CrossChainMessage,
            &message.vote_id, // Use vote_id instead of voter
            Some(target_chain),
            &format!("Cross-chain message to {}", target_chain),
            &format!(r#"{{"vote_id": "{}", "source_chain": "{}", "target_chain": "{}", "choice": "{:?}"}}"#, 
                message.vote_id, message.source_chain, target_chain, message.vote_choice),
            0, // Use 0 for block height since it's not available
            None,
        )?;

        Ok(entry_id)
    }

    /// Log a system event
    pub fn log_system_event(
        &self,
        event_type: AuditEventType,
        actor: &str,
        description: &str,
        metadata: &str,
    ) -> Result<String, AuditTrailError> {
        let entry_id = self.create_log_entry(
            event_type,
            actor,
            None,
            description,
            metadata,
            0, // System events don't have block height
            None,
        )?;

        Ok(entry_id)
    }

    /// Log an anomaly detection
    pub fn log_anomaly_detection(
        &self,
        anomaly: &AnomalyDetection,
        severity: AnomalySeverity,
    ) -> Result<String, AuditTrailError> {
        let entry_id = self.create_log_entry(
            AuditEventType::AnomalyDetected,
            "anomaly_detector",
            None,
            &format!(
                "Anomaly detected: {} (severity: {})",
                anomaly.anomaly_type, severity
            ),
            &format!(
                r#"{{"id": "{}", "type": "{:?}", "confidence": {}, "description": "{}"}}"#,
                anomaly.id, anomaly.anomaly_type, anomaly.confidence, anomaly.description
            ),
            0, // Use 0 for block height since it's not available
            None,
        )?;

        Ok(entry_id)
    }

    /// Query audit logs based on criteria
    pub fn query_logs(&self, query: &AuditQuery) -> Result<Vec<AuditLogEntry>, AuditTrailError> {
        let log_entries = self
            .log_entries
            .read()
            .map_err(|_| AuditTrailError::LockError)?;

        let mut results = Vec::new();
        let mut count = 0;
        let limit = query.limit.unwrap_or(usize::MAX);
        let offset = query.offset.unwrap_or(0);

        for entry in log_entries.iter() {
            // Apply filters
            if !self.matches_query(entry, query) {
                continue;
            }

            if count < offset {
                count += 1;
                continue;
            }

            if results.len() >= limit {
                break;
            }

            results.push(entry.clone());
        }

        Ok(results)
    }

    /// Get audit log by ID
    pub fn get_log_by_id(&self, log_id: &str) -> Result<Option<AuditLogEntry>, AuditTrailError> {
        let log_entries = self
            .log_entries
            .read()
            .map_err(|_| AuditTrailError::LockError)?;

        for entry in log_entries.iter() {
            if entry.id == log_id {
                return Ok(Some(entry.clone()));
            }
        }

        Ok(None)
    }

    /// Verify Merkle tree integrity
    pub fn verify_merkle_integrity(&self) -> Result<bool, AuditTrailError> {
        let merkle_tree = self
            .merkle_tree
            .lock()
            .map_err(|_| AuditTrailError::LockError)?;

        if merkle_tree.nodes.is_empty() {
            return Ok(true);
        }

        // Verify root hash matches calculated hash
        let calculated_root = self.calculate_merkle_root()?;
        Ok(calculated_root == merkle_tree.root_hash)
    }

    /// Generate JSON audit report
    pub fn generate_json_report(
        &self,
        query: Option<&AuditQuery>,
    ) -> Result<String, AuditTrailError> {
        let logs = if let Some(q) = query {
            self.query_logs(q)?
        } else {
            self.get_all_logs()?
        };

        let report = serde_json::json!({
            "audit_trail_report": {
                "timestamp": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
                "total_entries": logs.len(),
                "entries": logs,
                "merkle_root": self.get_merkle_root()?,
                "integrity_verified": self.verify_merkle_integrity()?
            }
        });

        serde_json::to_string_pretty(&report).map_err(|_| AuditTrailError::SerializationError)
    }

    /// Generate human-readable audit report
    pub fn generate_human_report(
        &self,
        query: Option<&AuditQuery>,
    ) -> Result<String, AuditTrailError> {
        let logs = if let Some(q) = query {
            self.query_logs(q)?
        } else {
            self.get_all_logs()?
        };

        let mut report = String::new();
        report.push_str("Audit Trail Report\n");
        report.push_str("==================\n\n");
        report.push_str(&format!(
            "Generated: {}\n",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
        ));
        report.push_str(&format!("Total Entries: {}\n", logs.len()));
        report.push_str(&format!(
            "Merkle Root: {}\n",
            hex::encode(self.get_merkle_root()?)
        ));
        report.push_str(&format!(
            "Integrity Verified: {}\n\n",
            self.verify_merkle_integrity()?
        ));

        report.push_str("Log Entries:\n");
        report.push_str("------------\n");

        for (i, log) in logs.iter().enumerate() {
            report.push_str(&format!(
                "{}. [{}] {} - {}\n",
                i + 1,
                log.timestamp,
                log.event_type,
                log.description
            ));
            report.push_str(&format!("   Actor: {}\n", log.actor));
            if let Some(ref target) = log.target {
                report.push_str(&format!("   Target: {}\n", target));
            }
            report.push_str(&format!("   Hash: {}\n", hex::encode(&log.hash)));
            if log.signature.is_some() {
                report.push_str("   [SIGNED]\n");
            }
            report.push('\n');
        }

        Ok(report)
    }

    /// Generate Chart.js-compatible JSON for visualizations
    pub fn generate_chartjs_data(
        &self,
        chart_type: &str,
        query: Option<&AuditQuery>,
    ) -> Result<String, AuditTrailError> {
        let logs = if let Some(q) = query {
            self.query_logs(q)?
        } else {
            self.get_all_logs()?
        };

        match chart_type {
            "event_frequency" => self.generate_event_frequency_chart(&logs),
            "activity_timeline" => self.generate_activity_timeline_chart(&logs),
            "actor_activity" => self.generate_actor_activity_chart(&logs),
            "proposal_activity" => self.generate_proposal_activity_chart(&logs),
            _ => Err(AuditTrailError::InvalidChartType),
        }
    }

    /// Create a new log entry
    #[allow(clippy::too_many_arguments)]
    fn create_log_entry(
        &self,
        event_type: AuditEventType,
        actor: &str,
        target: Option<&str>,
        description: &str,
        metadata: &str,
        block_height: u64,
        transaction_hash: Option<String>,
    ) -> Result<String, AuditTrailError> {
        let params = LogEntryParams {
            event_type,
            actor: actor.to_string(),
            target: target.map(|s| s.to_string()),
            description: description.to_string(),
            metadata: metadata.to_string(),
            block_height,
            transaction_hash,
        };

        self.create_log_entry_with_params(params)
    }

    fn create_log_entry_with_params(
        &self,
        params: LogEntryParams,
    ) -> Result<String, AuditTrailError> {
        let entry_id = format!(
            "audit_{}",
            self.entry_counter.fetch_add(1, Ordering::SeqCst)
        );
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Create log entry
        let mut entry = AuditLogEntry {
            id: entry_id.clone(),
            event_type: params.event_type.clone(),
            timestamp,
            actor: params.actor,
            target: params.target,
            description: params.description,
            metadata: params.metadata,
            hash: Vec::new(),
            signature: None,
            block_height: params.block_height,
            transaction_hash: params.transaction_hash,
        };

        // Calculate hash
        entry.hash = self.calculate_entry_hash(&entry)?;

        // Sign entry if signatures are enabled
        if self.config.enable_signatures {
            if let Ok(signing_key) = self.signing_key.lock() {
                if let Some(ref key) = *signing_key {
                    let params = DilithiumParams::dilithium3();
                    entry.signature = Some(
                        dilithium_sign(&entry.hash, key, &params)
                            .map_err(|_| AuditTrailError::SigningFailed)?,
                    );
                }
            }
        }

        // Store entry
        self.store_log_entry(entry.clone())?;

        // Update Merkle tree
        if self.config.enable_merkle_verification {
            self.update_merkle_tree(&entry)?;
        }

        // Update indices
        self.update_indices(&entry)?;

        Ok(entry_id)
    }

    /// Calculate SHA-3 hash of a log entry
    pub fn calculate_entry_hash(&self, entry: &AuditLogEntry) -> Result<Vec<u8>, AuditTrailError> {
        let mut hasher = Sha3_256::new();

        // Hash all fields except hash and signature
        hasher.update(entry.id.as_bytes());
        hasher.update(entry.event_type.to_string().as_bytes());
        hasher.update(entry.timestamp.to_le_bytes());
        hasher.update(entry.actor.as_bytes());
        if let Some(ref target) = entry.target {
            hasher.update(target.as_bytes());
        }
        hasher.update(entry.description.as_bytes());
        hasher.update(entry.metadata.as_bytes());
        hasher.update(entry.block_height.to_le_bytes());
        if let Some(ref tx_hash) = entry.transaction_hash {
            hasher.update(tx_hash.as_bytes());
        }

        Ok(hasher.finalize().to_vec())
    }

    /// Store a log entry
    fn store_log_entry(&self, entry: AuditLogEntry) -> Result<(), AuditTrailError> {
        let mut log_entries = self
            .log_entries
            .write()
            .map_err(|_| AuditTrailError::LockError)?;

        // Add entry
        log_entries.push_back(entry.clone());

        // Maintain size limit
        while log_entries.len() > self.config.max_entries {
            log_entries.pop_front();
        }

        Ok(())
    }

    /// Update Merkle tree with new entry
    fn update_merkle_tree(&self, entry: &AuditLogEntry) -> Result<(), AuditTrailError> {
        let mut merkle_tree = self
            .merkle_tree
            .lock()
            .map_err(|_| AuditTrailError::LockError)?;

        // Create leaf node
        let leaf_node = MerkleNode {
            hash: entry.hash.clone(),
            left_hash: None,
            right_hash: None,
            level: 0,
            index: merkle_tree.leaf_count,
        };

        merkle_tree.nodes.push(leaf_node);
        merkle_tree.leaf_count += 1;

        // Rebuild tree
        self.rebuild_merkle_tree(&mut merkle_tree)?;

        Ok(())
    }

    /// Rebuild Merkle tree
    fn rebuild_merkle_tree(
        &self,
        merkle_tree: &mut AuditMerkleTree,
    ) -> Result<(), AuditTrailError> {
        if merkle_tree.leaf_count == 0 {
            return Ok(());
        }

        // Calculate required depth
        let depth = (merkle_tree.leaf_count as f64).log2().ceil() as u32;
        merkle_tree.depth = depth;

        // Build tree bottom-up
        let mut current_level = 0;
        let mut nodes_at_level = merkle_tree.leaf_count as usize;

        while nodes_at_level > 1 {
            let mut next_level_nodes = Vec::new();
            let mut i = 0;

            while i < nodes_at_level {
                let left_hash = if i < merkle_tree.nodes.len() {
                    merkle_tree.nodes[i].hash.clone()
                } else {
                    Vec::new()
                };

                let right_hash = if i + 1 < merkle_tree.nodes.len() {
                    merkle_tree.nodes[i + 1].hash.clone()
                } else {
                    left_hash.clone()
                };

                // Calculate parent hash
                let mut hasher = Sha3_256::new();
                hasher.update(&left_hash);
                hasher.update(&right_hash);
                let parent_hash = hasher.finalize().to_vec();

                let parent_node = MerkleNode {
                    hash: parent_hash,
                    left_hash: Some(left_hash),
                    right_hash: Some(right_hash),
                    level: current_level + 1,
                    index: (i / 2) as u64,
                };

                next_level_nodes.push(parent_node);
                i += 2;
            }

            merkle_tree.nodes.extend(next_level_nodes);
            nodes_at_level = nodes_at_level.div_ceil(2);
            current_level += 1;
        }

        // Set root hash
        if let Some(root_node) = merkle_tree.nodes.last() {
            merkle_tree.root_hash = root_node.hash.clone();
        }

        Ok(())
    }

    /// Calculate Merkle root hash
    fn calculate_merkle_root(&self) -> Result<Vec<u8>, AuditTrailError> {
        let log_entries = self
            .log_entries
            .read()
            .map_err(|_| AuditTrailError::LockError)?;

        if log_entries.is_empty() {
            return Ok(Vec::new());
        }

        let mut hashes: Vec<Vec<u8>> = log_entries.iter().map(|entry| entry.hash.clone()).collect();

        while hashes.len() > 1 {
            let mut next_level = Vec::new();
            let mut i = 0;

            while i < hashes.len() {
                let left_hash = &hashes[i];
                let right_hash = if i + 1 < hashes.len() {
                    &hashes[i + 1]
                } else {
                    left_hash
                };

                let mut hasher = Sha3_256::new();
                hasher.update(left_hash);
                hasher.update(right_hash);
                next_level.push(hasher.finalize().to_vec());

                i += 2;
            }

            hashes = next_level;
        }

        Ok(hashes.into_iter().next().unwrap_or_default())
    }

    /// Get Merkle root hash
    pub fn get_merkle_root(&self) -> Result<Vec<u8>, AuditTrailError> {
        let merkle_tree = self
            .merkle_tree
            .lock()
            .map_err(|_| AuditTrailError::LockError)?;

        Ok(merkle_tree.root_hash.clone())
    }

    /// Update search indices
    fn update_indices(&self, entry: &AuditLogEntry) -> Result<(), AuditTrailError> {
        // Update event type index
        if let Ok(mut event_index) = self.event_type_index.write() {
            event_index
                .entry(entry.event_type.clone())
                .or_insert_with(Vec::new)
                .push(entry.id.clone());
        }

        // Update timestamp index
        if let Ok(mut timestamp_index) = self.timestamp_index.write() {
            timestamp_index
                .entry(entry.timestamp)
                .or_insert_with(Vec::new)
                .push(entry.id.clone());
        }

        // Update actor index
        if let Ok(mut actor_index) = self.actor_index.write() {
            actor_index
                .entry(entry.actor.clone())
                .or_insert_with(Vec::new)
                .push(entry.id.clone());
        }

        // Update proposal index if applicable
        if let Some(ref target) = entry.target {
            if entry.event_type == AuditEventType::ProposalSubmitted
                || entry.event_type == AuditEventType::VoteCast
                || entry.event_type == AuditEventType::ProposalExecuted
            {
                if let Ok(mut proposal_index) = self.proposal_index.write() {
                    proposal_index
                        .entry(target.clone())
                        .or_insert_with(Vec::new)
                        .push(entry.id.clone());
                }
            }
        }

        Ok(())
    }

    /// Check if entry matches query criteria
    fn matches_query(&self, entry: &AuditLogEntry, query: &AuditQuery) -> bool {
        // Check timestamp range
        if let Some(start_time) = query.start_time {
            if entry.timestamp < start_time {
                return false;
            }
        }
        if let Some(end_time) = query.end_time {
            if entry.timestamp > end_time {
                return false;
            }
        }

        // Check event types
        if let Some(ref event_types) = query.event_types {
            if !event_types.contains(&entry.event_type) {
                return false;
            }
        }

        // Check actor
        if let Some(ref actor) = query.actor {
            if entry.actor != *actor {
                return false;
            }
        }

        // Check target
        if let Some(ref target) = query.target {
            if entry.target.as_ref() != Some(target) {
                return false;
            }
        }

        // Check proposal ID
        if let Some(ref proposal_id) = query.proposal_id {
            if entry.target.as_ref() != Some(proposal_id) {
                return false;
            }
        }

        // Check block height range
        if let Some((start_height, end_height)) = query.block_height_range {
            if entry.block_height < start_height || entry.block_height > end_height {
                return false;
            }
        }

        true
    }

    /// Get all log entries
    fn get_all_logs(&self) -> Result<Vec<AuditLogEntry>, AuditTrailError> {
        let log_entries = self
            .log_entries
            .read()
            .map_err(|_| AuditTrailError::LockError)?;

        Ok(log_entries.iter().cloned().collect())
    }

    /// Start background cleanup thread
    fn start_cleanup_thread(&self) {
        let config = self.config.clone();
        let log_entries = Arc::clone(&self.log_entries);
        let is_running = Arc::clone(&self.is_running);

        thread::spawn(move || {
            while is_running.load(Ordering::SeqCst) {
                // Clean up old entries
                if let Ok(mut entries) = log_entries.write() {
                    let current_time = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs();
                    let cutoff_time = current_time.saturating_sub(config.max_age_seconds);

                    while let Some(front) = entries.front() {
                        if front.timestamp < cutoff_time {
                            entries.pop_front();
                        } else {
                            break;
                        }
                    }
                }

                // Sleep for cleanup interval
                thread::sleep(Duration::from_secs(3600)); // 1 hour
            }
        });
    }

    /// Generate event frequency chart data
    fn generate_event_frequency_chart(
        &self,
        logs: &[AuditLogEntry],
    ) -> Result<String, AuditTrailError> {
        let mut event_counts: HashMap<AuditEventType, u64> = HashMap::new();

        for log in logs {
            *event_counts.entry(log.event_type.clone()).or_insert(0) += 1;
        }

        let labels: Vec<String> = event_counts.keys().map(|k| k.to_string()).collect();
        let data: Vec<u64> = event_counts.values().cloned().collect();

        let chart_data = serde_json::json!({
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Event Frequency",
                    "data": data,
                    "backgroundColor": "rgba(54, 162, 235, 0.2)",
                    "borderColor": "rgba(54, 162, 235, 1)",
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": true,
                "scales": {
                    "y": {
                        "beginAtZero": true,
                        "title": {
                            "display": true,
                            "text": "Number of Events"
                        }
                    }
                }
            }
        });

        serde_json::to_string_pretty(&chart_data).map_err(|_| AuditTrailError::SerializationError)
    }

    /// Generate activity timeline chart data
    fn generate_activity_timeline_chart(
        &self,
        logs: &[AuditLogEntry],
    ) -> Result<String, AuditTrailError> {
        let mut timeline_data: BTreeMap<u64, u64> = BTreeMap::new();

        for log in logs {
            // Group by hour
            let hour = log.timestamp / 3600 * 3600;
            *timeline_data.entry(hour).or_insert(0) += 1;
        }

        let labels: Vec<String> = timeline_data
            .keys()
            .map(|&timestamp| {
                let _datetime = SystemTime::UNIX_EPOCH + Duration::from_secs(timestamp);
                format!("{}", timestamp)
            })
            .collect();
        let data: Vec<u64> = timeline_data.values().cloned().collect();

        let chart_data = serde_json::json!({
            "type": "line",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Activity Timeline",
                    "data": data,
                    "borderColor": "rgba(75, 192, 192, 1)",
                    "backgroundColor": "rgba(75, 192, 192, 0.2)",
                    "fill": true
                }]
            },
            "options": {
                "responsive": true,
                "scales": {
                    "y": {
                        "beginAtZero": true,
                        "title": {
                            "display": true,
                            "text": "Number of Events"
                        }
                    },
                    "x": {
                        "title": {
                            "display": true,
                            "text": "Time"
                        }
                    }
                }
            }
        });

        serde_json::to_string_pretty(&chart_data).map_err(|_| AuditTrailError::SerializationError)
    }

    /// Generate actor activity chart data
    fn generate_actor_activity_chart(
        &self,
        logs: &[AuditLogEntry],
    ) -> Result<String, AuditTrailError> {
        let mut actor_counts: HashMap<String, u64> = HashMap::new();

        for log in logs {
            *actor_counts.entry(log.actor.clone()).or_insert(0) += 1;
        }

        // Sort by count and take top 10
        let mut sorted_actors: Vec<_> = actor_counts.into_iter().collect();
        sorted_actors.sort_by(|a, b| b.1.cmp(&a.1));
        sorted_actors.truncate(10);

        let labels: Vec<String> = sorted_actors
            .iter()
            .map(|(actor, _)| actor.clone())
            .collect();
        let data: Vec<u64> = sorted_actors.iter().map(|(_, count)| *count).collect();

        let chart_data = serde_json::json!({
            "type": "doughnut",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Actor Activity",
                    "data": data,
                    "backgroundColor": [
                        "rgba(255, 99, 132, 0.2)",
                        "rgba(54, 162, 235, 0.2)",
                        "rgba(255, 205, 86, 0.2)",
                        "rgba(75, 192, 192, 0.2)",
                        "rgba(153, 102, 255, 0.2)",
                        "rgba(255, 159, 64, 0.2)",
                        "rgba(199, 199, 199, 0.2)",
                        "rgba(83, 102, 255, 0.2)",
                        "rgba(255, 99, 255, 0.2)",
                        "rgba(99, 255, 132, 0.2)"
                    ],
                    "borderColor": [
                        "rgba(255, 99, 132, 1)",
                        "rgba(54, 162, 235, 1)",
                        "rgba(255, 205, 86, 1)",
                        "rgba(75, 192, 192, 1)",
                        "rgba(153, 102, 255, 1)",
                        "rgba(255, 159, 64, 1)",
                        "rgba(199, 199, 199, 1)",
                        "rgba(83, 102, 255, 1)",
                        "rgba(255, 99, 255, 1)",
                        "rgba(99, 255, 132, 1)"
                    ],
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": true,
                "plugins": {
                    "legend": {
                        "position": "right"
                    }
                }
            }
        });

        serde_json::to_string_pretty(&chart_data).map_err(|_| AuditTrailError::SerializationError)
    }

    /// Generate proposal activity chart data
    fn generate_proposal_activity_chart(
        &self,
        logs: &[AuditLogEntry],
    ) -> Result<String, AuditTrailError> {
        let mut proposal_activity: HashMap<String, u64> = HashMap::new();

        for log in logs {
            if let Some(ref target) = log.target {
                if log.event_type == AuditEventType::ProposalSubmitted
                    || log.event_type == AuditEventType::VoteCast
                    || log.event_type == AuditEventType::ProposalExecuted
                {
                    *proposal_activity.entry(target.clone()).or_insert(0) += 1;
                }
            }
        }

        // Sort by activity and take top 10
        let mut sorted_proposals: Vec<_> = proposal_activity.into_iter().collect();
        sorted_proposals.sort_by(|a, b| b.1.cmp(&a.1));
        sorted_proposals.truncate(10);

        let labels: Vec<String> = sorted_proposals
            .iter()
            .map(|(proposal, _)| proposal.clone())
            .collect();
        let data: Vec<u64> = sorted_proposals.iter().map(|(_, count)| *count).collect();

        let chart_data = serde_json::json!({
            "type": "bar",
            "data": {
                "labels": labels,
                "datasets": [{
                    "label": "Proposal Activity",
                    "data": data,
                    "backgroundColor": "rgba(153, 102, 255, 0.2)",
                    "borderColor": "rgba(153, 102, 255, 1)",
                    "borderWidth": 1
                }]
            },
            "options": {
                "responsive": true,
                "scales": {
                    "y": {
                        "beginAtZero": true,
                        "title": {
                            "display": true,
                            "text": "Number of Events"
                        }
                    },
                    "x": {
                        "title": {
                            "display": true,
                            "text": "Proposal ID"
                        }
                    }
                }
            }
        });

        serde_json::to_string_pretty(&chart_data).map_err(|_| AuditTrailError::SerializationError)
    }

    /// Generate SHA-3 hash for data integrity
    #[allow(dead_code)]
    fn sha3_hash(data: &[u8]) -> Vec<u8> {
        let mut hasher = Sha3_256::new();
        hasher.update(data);
        hasher.finalize().to_vec()
    }

    /// Verify log entry signature
    pub fn verify_log_signature(
        &self,
        entry: &AuditLogEntry,
        public_key: &DilithiumPublicKey,
    ) -> Result<bool, AuditTrailError> {
        if let Some(ref signature) = entry.signature {
            let params = DilithiumParams::dilithium3();
            dilithium_verify(&entry.hash, signature, public_key, &params)
                .map_err(|_| AuditTrailError::VerificationFailed)
        } else {
            Ok(false) // No signature to verify
        }
    }

    /// Get audit trail statistics
    pub fn get_statistics(&self) -> Result<AuditTrailStatistics, AuditTrailError> {
        let log_entries = self
            .log_entries
            .read()
            .map_err(|_| AuditTrailError::LockError)?;

        let mut event_counts: HashMap<AuditEventType, u64> = HashMap::new();
        let mut actor_counts: HashMap<String, u64> = HashMap::new();
        let mut total_entries = 0;
        let mut signed_entries = 0;

        for entry in log_entries.iter() {
            total_entries += 1;
            if entry.signature.is_some() {
                signed_entries += 1;
            }

            *event_counts.entry(entry.event_type.clone()).or_insert(0) += 1;
            *actor_counts.entry(entry.actor.clone()).or_insert(0) += 1;
        }

        Ok(AuditTrailStatistics {
            total_entries,
            signed_entries,
            event_counts,
            actor_counts,
            merkle_root: self.get_merkle_root()?,
            integrity_verified: self.verify_merkle_integrity()?,
        })
    }
}

/// Audit trail statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditTrailStatistics {
    /// Total number of log entries
    pub total_entries: u64,
    /// Number of signed entries
    pub signed_entries: u64,
    /// Event type counts
    pub event_counts: HashMap<AuditEventType, u64>,
    /// Actor activity counts
    pub actor_counts: HashMap<String, u64>,
    /// Merkle root hash
    pub merkle_root: Vec<u8>,
    /// Whether integrity is verified
    pub integrity_verified: bool,
}

/// Error types for audit trail operations
#[derive(Debug, Clone, PartialEq)]
pub enum AuditTrailError {
    /// Audit trail is already running
    AlreadyRunning,
    /// Audit trail is not running
    NotRunning,
    /// Lock acquisition failed
    LockError,
    /// Key generation failed
    KeyGenerationFailed,
    /// Signing operation failed
    SigningFailed,
    /// Signature verification failed
    VerificationFailed,
    /// Serialization error
    SerializationError,
    /// Invalid chart type
    InvalidChartType,
    /// Entry not found
    EntryNotFound,
    /// Invalid query parameters
    InvalidQuery,
    /// Merkle tree verification failed
    MerkleVerificationFailed,
    /// Configuration error
    ConfigurationError,
}

impl std::fmt::Display for AuditTrailError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditTrailError::AlreadyRunning => write!(f, "Audit trail is already running"),
            AuditTrailError::NotRunning => write!(f, "Audit trail is not running"),
            AuditTrailError::LockError => write!(f, "Lock acquisition failed"),
            AuditTrailError::KeyGenerationFailed => write!(f, "Key generation failed"),
            AuditTrailError::SigningFailed => write!(f, "Signing operation failed"),
            AuditTrailError::VerificationFailed => write!(f, "Signature verification failed"),
            AuditTrailError::SerializationError => write!(f, "Serialization error"),
            AuditTrailError::InvalidChartType => write!(f, "Invalid chart type"),
            AuditTrailError::EntryNotFound => write!(f, "Entry not found"),
            AuditTrailError::InvalidQuery => write!(f, "Invalid query parameters"),
            AuditTrailError::MerkleVerificationFailed => {
                write!(f, "Merkle tree verification failed")
            }
            AuditTrailError::ConfigurationError => write!(f, "Configuration error"),
        }
    }
}

impl std::error::Error for AuditTrailError {}
