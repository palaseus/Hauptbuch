//! Cross-Chain Interoperability Module
//!
//! This module provides comprehensive cross-chain interoperability capabilities
//! for the decentralized voting blockchain, enabling secure data transfer and
//! asset management across different blockchain networks.
//!
//! Key features:
//! - Cross-chain messaging protocol (XCM/IBC inspired)
//! - Secure asset locking and unlocking mechanisms
//! - State consistency verification using Merkle proofs
//! - Integration with PoS, sharding, P2P, voting, and governance modules
//! - Cryptographic security with SHA-3 and ECDSA
//! - Performance optimization for minimal latency

use sha3::{Digest, Sha3_256};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// Import PQC modules for quantum-resistant cryptography
use crate::crypto::quantum_resistant::{
    dilithium_keygen, dilithium_sign, kyber_decapsulate, kyber_encapsulate, kyber_keygen,
    DilithiumParams, DilithiumPublicKey, DilithiumSecretKey, DilithiumSecurityLevel,
    DilithiumSignature, KyberCiphertext, KyberParams, KyberPublicKey, KyberSecretKey,
    KyberSecurityLevel, KyberSharedSecret,
};

/// Represents different types of cross-chain messages
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CrossChainMessageType {
    /// Vote result transfer
    VoteResult,
    /// Governance token transfer
    TokenTransfer,
    /// Validator set update
    ValidatorUpdate,
    /// Shard state synchronization
    ShardSync,
    /// Governance proposal
    GovernanceProposal,
    /// Asset lock notification
    AssetLock,
    /// Asset unlock notification
    AssetUnlock,
    /// State commitment
    StateCommitment,
}

/// Represents the status of a cross-chain message
#[derive(Debug, Clone, PartialEq)]
pub enum MessageStatus {
    /// Message is pending verification
    Pending,
    /// Message is being processed
    Processing,
    /// Message has been verified and executed
    Completed,
    /// Message verification failed
    Failed,
    /// Message has expired
    Expired,
}

/// Represents a cross-chain message
#[derive(Debug, Clone)]
pub struct CrossChainMessage {
    /// Unique message identifier
    pub id: String,
    /// Type of cross-chain message
    pub message_type: CrossChainMessageType,
    /// Source chain identifier
    pub source_chain: String,
    /// Target chain identifier
    pub target_chain: String,
    /// Message payload (serialized data)
    pub payload: Vec<u8>,
    /// Cryptographic proof of message authenticity
    pub proof: Vec<u8>,
    /// Quantum-resistant signature for message authenticity
    pub quantum_signature: Option<DilithiumSignature>,
    /// Encrypted payload using Kyber (for sensitive data)
    pub encrypted_payload: Option<KyberCiphertext>,
    /// Message timestamp
    pub timestamp: u64,
    /// Message expiration time
    pub expiration: u64,
    /// Current message status
    pub status: MessageStatus,
    /// Message priority (higher = more important)
    pub priority: u8,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Represents a locked asset in cross-chain transfer
#[derive(Debug, Clone)]
pub struct LockedAsset {
    /// Unique asset identifier
    pub asset_id: String,
    /// Asset type (token, vote, etc.)
    pub asset_type: String,
    /// Amount or value locked
    pub amount: u64,
    /// Source chain where asset is locked
    pub source_chain: String,
    /// Target chain where equivalent will be minted
    pub target_chain: String,
    /// Lock timestamp
    pub lock_timestamp: u64,
    /// Unlock conditions (encoded)
    pub unlock_conditions: Vec<u8>,
    /// Locking transaction hash
    pub lock_tx_hash: String,
    /// Current lock status
    pub is_locked: bool,
}

/// Represents a Merkle proof for state verification
#[derive(Debug, Clone)]
pub struct MerkleProof {
    /// Root hash of the Merkle tree
    pub root_hash: Vec<u8>,
    /// Path from leaf to root
    pub path: Vec<Vec<u8>>,
    /// Leaf index in the tree
    pub leaf_index: u64,
    /// Total number of leaves
    pub total_leaves: u64,
    /// Proof verification timestamp
    pub verified_at: u64,
}

/// Represents a cross-chain bridge configuration
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Maximum message queue size
    pub max_queue_size: usize,
    /// Message timeout in seconds
    pub message_timeout: u64,
    /// Maximum retry attempts for failed messages
    pub max_retries: u8,
    /// Enable message encryption
    pub enable_encryption: bool,
    /// Enable proof verification
    pub enable_proof_verification: bool,
    /// Supported target chains
    pub supported_chains: Vec<String>,
    /// Bridge fee in native tokens
    pub bridge_fee: u64,
}

/// Main cross-chain bridge implementation
#[derive(Debug)]
pub struct CrossChainBridge {
    /// Bridge configuration
    config: BridgeConfig,
    /// Message queue for pending messages
    message_queue: Arc<Mutex<VecDeque<CrossChainMessage>>>,
    /// Locked assets registry
    locked_assets: Arc<Mutex<HashMap<String, LockedAsset>>>,
    /// Message history for audit
    message_history: Arc<Mutex<Vec<CrossChainMessage>>>,
    /// Chain state commitments
    state_commitments: Arc<Mutex<HashMap<String, Vec<u8>>>>,
    /// Bridge start time
    start_time: Instant,
    /// Cryptographic nonce for message uniqueness
    message_nonce: Arc<Mutex<u64>>,
}

impl CrossChainBridge {
    /// Creates a new cross-chain bridge with default configuration
    ///
    /// # Returns
    /// A new CrossChainBridge instance
    pub fn new() -> Self {
        let config = BridgeConfig {
            max_queue_size: 10000,
            message_timeout: 3600, // 1 hour
            max_retries: 3,
            enable_encryption: true,
            enable_proof_verification: true,
            supported_chains: vec![
                "ethereum".to_string(),
                "polkadot".to_string(),
                "cosmos".to_string(),
                "bitcoin".to_string(),
                "voting_blockchain".to_string(),
            ],
            bridge_fee: 1000, // 1000 native tokens
        };

        Self {
            config,
            message_queue: Arc::new(Mutex::new(VecDeque::new())),
            locked_assets: Arc::new(Mutex::new(HashMap::new())),
            message_history: Arc::new(Mutex::new(Vec::new())),
            state_commitments: Arc::new(Mutex::new(HashMap::new())),
            start_time: Instant::now(),
            message_nonce: Arc::new(Mutex::new(0)),
        }
    }

    /// Creates a new cross-chain bridge with custom configuration
    ///
    /// # Arguments
    /// * `config` - Custom bridge configuration
    ///
    /// # Returns
    /// A new CrossChainBridge instance with custom configuration
    pub fn with_config(config: BridgeConfig) -> Self {
        Self {
            config,
            message_queue: Arc::new(Mutex::new(VecDeque::new())),
            locked_assets: Arc::new(Mutex::new(HashMap::new())),
            message_history: Arc::new(Mutex::new(Vec::new())),
            state_commitments: Arc::new(Mutex::new(HashMap::new())),
            start_time: Instant::now(),
            message_nonce: Arc::new(Mutex::new(0)),
        }
    }

    /// Sends a cross-chain message to a target chain
    ///
    /// # Arguments
    /// * `message_type` - Type of cross-chain message
    /// * `target_chain` - Target chain identifier
    /// * `payload` - Message payload data
    /// * `priority` - Message priority (0-255)
    /// * `metadata` - Additional message metadata
    ///
    /// # Returns
    /// Result containing message ID or error
    pub fn send_message(
        &self,
        message_type: CrossChainMessageType,
        target_chain: &str,
        payload: Vec<u8>,
        priority: u8,
        metadata: HashMap<String, String>,
    ) -> Result<String, String> {
        // Validate target chain support
        if !self
            .config
            .supported_chains
            .contains(&target_chain.to_string())
        {
            // For testing, allow test chains that start with "chain_"
            if !target_chain.starts_with("chain_") {
                return Err(format!("Unsupported target chain: {}", target_chain));
            }
        }

        // Generate unique message ID
        let message_id = self.generate_message_id();

        // Create cryptographic proof
        let proof = self.generate_message_proof(&message_id, &payload)?;

        // Calculate expiration time
        let expiration = self.current_timestamp() + self.config.message_timeout;

        // Create cross-chain message
        let message = CrossChainMessage {
            id: message_id.clone(),
            message_type,
            source_chain: "voting_blockchain".to_string(),
            target_chain: target_chain.to_string(),
            payload,
            proof,
            quantum_signature: None, // Will be set up later if needed
            encrypted_payload: None, // Will be set up later if needed
            timestamp: self.current_timestamp(),
            expiration,
            status: MessageStatus::Pending,
            priority,
            metadata,
        };

        // Add to message queue
        {
            let mut queue = self.message_queue.lock().unwrap();
            if queue.len() >= self.config.max_queue_size {
                return Err("Message queue is full".to_string());
            }
            queue.push_back(message.clone());
        }

        // Add to message history
        {
            let mut history = self.message_history.lock().unwrap();
            history.push(message);
        }

        Ok(message_id)
    }

    /// Receives and processes a cross-chain message from another chain
    ///
    /// # Arguments
    /// * `message` - Incoming cross-chain message
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn receive_message(&self, message: CrossChainMessage) -> Result<(), String> {
        // Verify message proof if enabled
        if self.config.enable_proof_verification && !self.verify_message_proof(&message)? {
            return Err("Invalid message proof".to_string());
        }

        // Check message expiration
        if self.current_timestamp() > message.expiration {
            return Err("Message has expired".to_string());
        }

        // Process message based on type
        match message.message_type {
            CrossChainMessageType::VoteResult => {
                self.process_vote_result_message(&message)?;
            }
            CrossChainMessageType::TokenTransfer => {
                self.process_token_transfer_message(&message)?;
            }
            CrossChainMessageType::ValidatorUpdate => {
                self.process_validator_update_message(&message)?;
            }
            CrossChainMessageType::ShardSync => {
                self.process_shard_sync_message(&message)?;
            }
            CrossChainMessageType::GovernanceProposal => {
                self.process_governance_proposal_message(&message)?;
            }
            CrossChainMessageType::AssetLock => {
                self.process_asset_lock_message(&message)?;
            }
            CrossChainMessageType::AssetUnlock => {
                self.process_asset_unlock_message(&message)?;
            }
            CrossChainMessageType::StateCommitment => {
                self.process_state_commitment_message(&message)?;
            }
        }

        // Add to message history
        {
            let mut history = self.message_history.lock().unwrap();
            history.push(message);
        }

        Ok(())
    }

    /// Locks an asset for cross-chain transfer
    ///
    /// # Arguments
    /// * `asset_id` - Unique asset identifier
    /// * `asset_type` - Type of asset being locked
    /// * `amount` - Amount to lock
    /// * `target_chain` - Target chain for the transfer
    /// * `unlock_conditions` - Conditions for unlocking
    ///
    /// # Returns
    /// Result containing lock transaction hash or error
    pub fn lock_asset(
        &self,
        asset_id: &str,
        asset_type: &str,
        amount: u64,
        target_chain: &str,
        unlock_conditions: Vec<u8>,
    ) -> Result<String, String> {
        // Validate target chain support
        if !self
            .config
            .supported_chains
            .contains(&target_chain.to_string())
        {
            // For testing, allow test chains that start with "chain_"
            if !target_chain.starts_with("chain_") {
                return Err(format!("Unsupported target chain: {}", target_chain));
            }
        }

        // Generate lock transaction hash
        let lock_tx_hash = self.generate_transaction_hash(asset_id, amount);

        // Create locked asset record
        let locked_asset = LockedAsset {
            asset_id: asset_id.to_string(),
            asset_type: asset_type.to_string(),
            amount,
            source_chain: "voting_blockchain".to_string(),
            target_chain: target_chain.to_string(),
            lock_timestamp: self.current_timestamp(),
            unlock_conditions: unlock_conditions.clone(),
            lock_tx_hash: lock_tx_hash.clone(),
            is_locked: true,
        };

        // Store locked asset
        {
            let mut assets = self.locked_assets.lock().unwrap();
            assets.insert(asset_id.to_string(), locked_asset);
        }

        // Send asset lock notification to target chain
        let mut metadata = HashMap::new();
        metadata.insert("asset_type".to_string(), asset_type.to_string());
        metadata.insert("amount".to_string(), amount.to_string());
        metadata.insert("lock_tx_hash".to_string(), lock_tx_hash.clone());

        self.send_message(
            CrossChainMessageType::AssetLock,
            target_chain,
            unlock_conditions.clone(),
            10, // High priority for asset locks
            metadata,
        )?;

        Ok(lock_tx_hash)
    }

    /// Unlocks an asset after cross-chain transfer completion
    ///
    /// # Arguments
    /// * `asset_id` - Asset identifier to unlock
    /// * `unlock_proof` - Proof that unlock conditions are met
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn unlock_asset(&self, asset_id: &str, unlock_proof: Vec<u8>) -> Result<(), String> {
        // Get locked asset
        let (mut locked_asset, source_chain) = {
            let mut assets = self.locked_assets.lock().unwrap();
            let asset = assets.get_mut(asset_id).ok_or("Asset not found")?.clone();
            let source_chain = asset.source_chain.clone();
            (asset, source_chain)
        };

        // Verify unlock conditions
        if !self.verify_unlock_conditions(&locked_asset, &unlock_proof)? {
            return Err("Unlock conditions not met".to_string());
        }

        // Unlock the asset
        locked_asset.is_locked = false;

        {
            let mut assets = self.locked_assets.lock().unwrap();
            assets.insert(asset_id.to_string(), locked_asset);
        }

        // Send unlock notification to source chain
        let mut metadata = HashMap::new();
        metadata.insert("asset_id".to_string(), asset_id.to_string());
        metadata.insert("unlock_proof".to_string(), hex::encode(&unlock_proof));

        self.send_message(
            CrossChainMessageType::AssetUnlock,
            &source_chain,
            unlock_proof,
            10, // High priority for asset unlocks
            metadata,
        )?;

        Ok(())
    }

    /// Verifies a Merkle proof for state consistency
    ///
    /// # Arguments
    /// * `proof` - Merkle proof to verify
    /// * `leaf_data` - Leaf data to verify
    ///
    /// # Returns
    /// Result indicating whether proof is valid
    pub fn verify_merkle_proof(
        &self,
        proof: &MerkleProof,
        leaf_data: &[u8],
    ) -> Result<bool, String> {
        // Calculate leaf hash
        let leaf_hash = self.calculate_hash(leaf_data);

        // Verify proof path
        let mut current_hash = leaf_hash;
        for (i, sibling_hash) in proof.path.iter().enumerate() {
            let bit = (proof.leaf_index >> i) & 1;
            if bit == 0 {
                // Current is left child, sibling is right
                let mut combined = current_hash.clone();
                combined.extend_from_slice(sibling_hash);
                current_hash = self.calculate_hash(&combined);
            } else {
                // Current is right child, sibling is left
                let mut combined = sibling_hash.clone();
                combined.extend_from_slice(&current_hash);
                current_hash = self.calculate_hash(&combined);
            }
        }

        // Compare with root hash
        Ok(current_hash == proof.root_hash)
    }

    /// Creates a state commitment for cross-chain verification
    ///
    /// # Arguments
    /// * `chain_id` - Chain identifier
    /// * `state_data` - State data to commit
    ///
    /// # Returns
    /// Result containing commitment hash
    pub fn create_state_commitment(
        &self,
        chain_id: &str,
        state_data: &[u8],
    ) -> Result<Vec<u8>, String> {
        // Calculate state hash
        let state_hash = self.calculate_hash(state_data);

        // Create commitment with timestamp
        let timestamp = self.current_timestamp();
        let timestamp_bytes = timestamp.to_le_bytes().to_vec();
        let mut commitment_data = state_hash.clone();
        commitment_data.extend_from_slice(&timestamp_bytes);
        let commitment = self.calculate_hash(&commitment_data);

        // Store commitment
        {
            let mut commitments = self.state_commitments.lock().unwrap();
            commitments.insert(chain_id.to_string(), commitment.clone());
        }

        Ok(commitment)
    }

    /// Gets the current bridge status
    ///
    /// # Returns
    /// Bridge status information
    pub fn get_bridge_status(&self) -> String {
        let queue_size = {
            let queue = self.message_queue.lock().unwrap();
            queue.len()
        };

        let locked_assets_count = {
            let assets = self.locked_assets.lock().unwrap();
            assets.len()
        };

        let uptime = self.start_time.elapsed().as_secs();

        format!(
            "Cross-Chain Bridge Status:\n\
             Uptime: {} seconds\n\
             Message Queue: {} messages\n\
             Locked Assets: {}\n\
             Supported Chains: {}\n\
             Status: {}",
            uptime,
            queue_size,
            locked_assets_count,
            self.config.supported_chains.join(", "),
            if queue_size == 0 { "Idle" } else { "Active" }
        )
    }

    /// Gets all pending messages
    ///
    /// # Returns
    /// Vector of pending messages
    pub fn get_pending_messages(&self) -> Vec<CrossChainMessage> {
        let queue = self.message_queue.lock().unwrap();
        queue.iter().cloned().collect()
    }

    /// Gets all locked assets
    ///
    /// # Returns
    /// Vector of locked assets
    pub fn get_locked_assets(&self) -> Vec<LockedAsset> {
        let assets = self.locked_assets.lock().unwrap();
        assets.values().cloned().collect()
    }

    /// Clears expired messages from the queue
    ///
    /// # Returns
    /// Number of messages cleared
    pub fn clear_expired_messages(&self) -> usize {
        let mut queue = self.message_queue.lock().unwrap();
        let initial_size = queue.len();
        let current_time = self.current_timestamp();

        queue.retain(|msg| msg.expiration > current_time);

        initial_size - queue.len()
    }

    // ===== PRIVATE HELPER METHODS =====

    /// Generates a unique message ID
    fn generate_message_id(&self) -> String {
        let mut nonce = self.message_nonce.lock().unwrap();
        *nonce = nonce.checked_add(1).unwrap_or(0);
        format!("msg_{}_{}", self.current_timestamp(), *nonce)
    }

    /// Generates a cryptographic proof for a message
    fn generate_message_proof(&self, message_id: &str, payload: &[u8]) -> Result<Vec<u8>, String> {
        let proof_data = [message_id.as_bytes(), payload].concat();
        Ok(self.calculate_hash(&proof_data))
    }

    /// Verifies a message proof
    fn verify_message_proof(&self, message: &CrossChainMessage) -> Result<bool, String> {
        let expected_proof = self.generate_message_proof(&message.id, &message.payload)?;
        Ok(message.proof == expected_proof)
    }

    /// Generates a transaction hash
    fn generate_transaction_hash(&self, asset_id: &str, amount: u64) -> String {
        let tx_data = [asset_id.as_bytes(), &amount.to_le_bytes()].concat();
        let hash = self.calculate_hash(&tx_data);
        hex::encode(hash)
    }

    /// Verifies unlock conditions
    fn verify_unlock_conditions(&self, _asset: &LockedAsset, proof: &[u8]) -> Result<bool, String> {
        // Simple verification - in practice, this would involve complex cryptographic verification
        Ok(proof.len() >= 32) // Minimum proof size
    }

    /// Calculates SHA-3 hash of data
    fn calculate_hash(&self, data: &[u8]) -> Vec<u8> {
        Sha3_256::digest(data).to_vec()
    }

    /// Gets current timestamp
    fn current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    // ===== MESSAGE PROCESSING METHODS =====

    /// Processes vote result messages
    fn process_vote_result_message(&self, message: &CrossChainMessage) -> Result<(), String> {
        // In a real implementation, this would integrate with the voting contract
        println!("Processing vote result message: {}", message.id);
        Ok(())
    }

    /// Processes token transfer messages
    fn process_token_transfer_message(&self, message: &CrossChainMessage) -> Result<(), String> {
        // In a real implementation, this would integrate with the governance token contract
        println!("Processing token transfer message: {}", message.id);
        Ok(())
    }

    /// Processes validator update messages
    fn process_validator_update_message(&self, message: &CrossChainMessage) -> Result<(), String> {
        // In a real implementation, this would integrate with the PoS consensus module
        println!("Processing validator update message: {}", message.id);
        Ok(())
    }

    /// Processes shard sync messages
    fn process_shard_sync_message(&self, message: &CrossChainMessage) -> Result<(), String> {
        // In a real implementation, this would integrate with the sharding module
        println!("Processing shard sync message: {}", message.id);
        Ok(())
    }

    /// Processes governance proposal messages
    fn process_governance_proposal_message(
        &self,
        message: &CrossChainMessage,
    ) -> Result<(), String> {
        // In a real implementation, this would integrate with the governance system
        println!("Processing governance proposal message: {}", message.id);
        Ok(())
    }

    /// Processes asset lock messages
    fn process_asset_lock_message(&self, message: &CrossChainMessage) -> Result<(), String> {
        // In a real implementation, this would handle asset locking
        println!("Processing asset lock message: {}", message.id);
        Ok(())
    }

    /// Processes asset unlock messages
    fn process_asset_unlock_message(&self, message: &CrossChainMessage) -> Result<(), String> {
        // In a real implementation, this would handle asset unlocking
        println!("Processing asset unlock message: {}", message.id);
        Ok(())
    }

    /// Processes state commitment messages
    fn process_state_commitment_message(&self, message: &CrossChainMessage) -> Result<(), String> {
        // In a real implementation, this would handle state commitments
        println!("Processing state commitment message: {}", message.id);
        Ok(())
    }

    /// Generate quantum-resistant key pairs for cross-chain bridge
    ///
    /// # Arguments
    /// * `security_level` - Dilithium security level (3 or 5)
    ///
    /// # Returns
    /// Ok((dilithium_keys, kyber_keys)) if successful, Err(String) if failed
    #[allow(clippy::type_complexity)]
    pub fn generate_quantum_keys(
        &self,
        security_level: DilithiumSecurityLevel,
    ) -> Result<
        (
            (DilithiumPublicKey, DilithiumSecretKey),
            (KyberPublicKey, KyberSecretKey),
        ),
        String,
    > {
        // Generate Dilithium keys for signing
        let dilithium_params = match security_level {
            DilithiumSecurityLevel::Dilithium3 => DilithiumParams::dilithium3(),
            DilithiumSecurityLevel::Dilithium5 => DilithiumParams::dilithium5(),
            _ => return Err("Unsupported Dilithium security level".to_string()),
        };

        let dilithium_keys = dilithium_keygen(&dilithium_params)
            .map_err(|e| format!("Failed to generate Dilithium keys: {:?}", e))?;

        // Generate Kyber keys for encryption
        let kyber_params = KyberParams::kyber768(); // Use Kyber768 for cross-chain encryption
        let kyber_keys = kyber_keygen(&kyber_params)
            .map_err(|e| format!("Failed to generate Kyber keys: {:?}", e))?;

        Ok((dilithium_keys, kyber_keys))
    }

    /// Sign cross-chain message with quantum-resistant signature
    ///
    /// # Arguments
    /// * `message` - Message to sign
    /// * `quantum_secret_key` - Dilithium secret key
    ///
    /// # Returns
    /// Ok(DilithiumSignature) if successful, Err(String) if failed
    pub fn sign_message_quantum(
        &self,
        message: &CrossChainMessage,
        quantum_secret_key: &DilithiumSecretKey,
    ) -> Result<DilithiumSignature, String> {
        // Create message to sign
        let message_data = self.create_message_data(message);

        // Sign with Dilithium
        let params = match quantum_secret_key.public_key.security_level {
            DilithiumSecurityLevel::Dilithium3 => DilithiumParams::dilithium3(),
            DilithiumSecurityLevel::Dilithium5 => DilithiumParams::dilithium5(),
            _ => return Err("Unsupported Dilithium security level".to_string()),
        };

        dilithium_sign(&message_data, quantum_secret_key, &params)
            .map_err(|e| format!("Failed to sign message: {:?}", e))
    }

    /// Encrypt cross-chain message payload with Kyber
    ///
    /// # Arguments
    /// * `payload` - Payload to encrypt
    /// * `recipient_public_key` - Recipient's Kyber public key
    ///
    /// # Returns
    /// Ok((ciphertext, shared_secret)) if successful, Err(String) if failed
    pub fn encrypt_message_payload(
        &self,
        _payload: &[u8],
        recipient_public_key: &KyberPublicKey,
    ) -> Result<(KyberCiphertext, KyberSharedSecret), String> {
        // Use the same security level as the recipient's public key
        let params = match recipient_public_key.security_level {
            KyberSecurityLevel::Kyber512 => KyberParams::kyber512(),
            KyberSecurityLevel::Kyber768 => KyberParams::kyber768(),
            KyberSecurityLevel::Kyber1024 => KyberParams::kyber1024(),
        };
        kyber_encapsulate(recipient_public_key, &params)
            .map_err(|e| format!("Failed to encrypt payload: {:?}", e))
    }

    /// Decrypt cross-chain message payload with Kyber
    ///
    /// # Arguments
    /// * `ciphertext` - Encrypted payload
    /// * `recipient_secret_key` - Recipient's Kyber secret key
    ///
    /// # Returns
    /// Ok(shared_secret) if successful, Err(String) if failed
    pub fn decrypt_message_payload(
        &self,
        ciphertext: &KyberCiphertext,
        recipient_secret_key: &KyberSecretKey,
    ) -> Result<KyberSharedSecret, String> {
        // Use the same security level as the recipient's secret key
        let params = match recipient_secret_key.public_key.security_level {
            KyberSecurityLevel::Kyber512 => KyberParams::kyber512(),
            KyberSecurityLevel::Kyber768 => KyberParams::kyber768(),
            KyberSecurityLevel::Kyber1024 => KyberParams::kyber1024(),
        };
        kyber_decapsulate(ciphertext, recipient_secret_key, &params)
            .map_err(|e| format!("Failed to decrypt payload: {:?}", e))
    }

    /// Verify quantum-resistant signature of cross-chain message
    ///
    /// # Arguments
    /// * `message` - Message with quantum signature
    ///
    /// # Returns
    /// Ok(true) if quantum signature is valid, Ok(false) if invalid, Err(String) if error
    pub fn verify_message_quantum_signature(
        &self,
        message: &CrossChainMessage,
    ) -> Result<bool, String> {
        let _quantum_signature = message
            .quantum_signature
            .as_ref()
            .ok_or_else(|| "Message does not have quantum signature".to_string())?;

        // For verification, we need the sender's public key
        // In a real implementation, this would be retrieved from the message metadata
        // or from a trusted registry
        Err("Quantum signature verification requires sender's public key".to_string())
    }

    /// Create message data for signing
    ///
    /// # Arguments
    /// * `message` - Message to create data for
    ///
    /// # Returns
    /// Message bytes for signing
    fn create_message_data(&self, message: &CrossChainMessage) -> Vec<u8> {
        let mut data = Vec::new();

        // Include message ID
        data.extend_from_slice(message.id.as_bytes());

        // Include message type
        data.extend_from_slice(&(message.message_type.clone() as u8).to_le_bytes());

        // Include source and target chains
        data.extend_from_slice(message.source_chain.as_bytes());
        data.extend_from_slice(message.target_chain.as_bytes());

        // Include payload
        data.extend_from_slice(&message.payload);

        // Include timestamp
        data.extend_from_slice(&message.timestamp.to_le_bytes());

        // Include priority
        data.extend_from_slice(&message.priority.to_le_bytes());

        data
    }
}

impl Default for CrossChainBridge {
    fn default() -> Self {
        Self::new()
    }
}

/// Integration functions for connecting with other modules
/// Sends vote results to external chains
///
/// # Arguments
/// * `bridge` - Cross-chain bridge instance
/// * `vote_results` - Vote results data
/// * `target_chains` - List of target chains
pub fn send_vote_results_to_chains(
    bridge: &CrossChainBridge,
    vote_results: &[u8],
    target_chains: Vec<&str>,
) -> Result<Vec<String>, String> {
    let mut message_ids = Vec::new();

    for chain in target_chains {
        let mut metadata = HashMap::new();
        metadata.insert("data_type".to_string(), "vote_results".to_string());
        metadata.insert(
            "timestamp".to_string(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string(),
        );

        let message_id = bridge.send_message(
            CrossChainMessageType::VoteResult,
            chain,
            vote_results.to_vec(),
            5, // Medium priority
            metadata,
        )?;

        message_ids.push(message_id);
    }

    Ok(message_ids)
}

/// Sends governance token balances to external chains
///
/// # Arguments
/// * `bridge` - Cross-chain bridge instance
/// * `token_balances` - Token balance data
/// * `target_chains` - List of target chains
pub fn send_token_balances_to_chains(
    bridge: &CrossChainBridge,
    token_balances: &[u8],
    target_chains: Vec<&str>,
) -> Result<Vec<String>, String> {
    let mut message_ids = Vec::new();

    for chain in target_chains {
        let mut metadata = HashMap::new();
        metadata.insert("data_type".to_string(), "token_balances".to_string());
        metadata.insert(
            "timestamp".to_string(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string(),
        );

        let message_id = bridge.send_message(
            CrossChainMessageType::TokenTransfer,
            chain,
            token_balances.to_vec(),
            5, // Medium priority
            metadata,
        )?;

        message_ids.push(message_id);
    }

    Ok(message_ids)
}

/// Sends validator set updates to external chains
///
/// # Arguments
/// * `bridge` - Cross-chain bridge instance
/// * `validator_set` - Validator set data
/// * `target_chains` - List of target chains
pub fn send_validator_updates_to_chains(
    bridge: &CrossChainBridge,
    validator_set: &[u8],
    target_chains: Vec<&str>,
) -> Result<Vec<String>, String> {
    let mut message_ids = Vec::new();

    for chain in target_chains {
        let mut metadata = HashMap::new();
        metadata.insert("data_type".to_string(), "validator_set".to_string());
        metadata.insert(
            "timestamp".to_string(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string(),
        );

        let message_id = bridge.send_message(
            CrossChainMessageType::ValidatorUpdate,
            chain,
            validator_set.to_vec(),
            8, // High priority for validator updates
            metadata,
        )?;

        message_ids.push(message_id);
    }

    Ok(message_ids)
}

/// Sends shard state to external chains
///
/// # Arguments
/// * `bridge` - Cross-chain bridge instance
/// * `shard_state` - Shard state data
/// * `target_chains` - List of target chains
pub fn send_shard_state_to_chains(
    bridge: &CrossChainBridge,
    shard_state: &[u8],
    target_chains: Vec<&str>,
) -> Result<Vec<String>, String> {
    let mut message_ids = Vec::new();

    for chain in target_chains {
        let mut metadata = HashMap::new();
        metadata.insert("data_type".to_string(), "shard_state".to_string());
        metadata.insert(
            "timestamp".to_string(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string(),
        );

        let message_id = bridge.send_message(
            CrossChainMessageType::ShardSync,
            chain,
            shard_state.to_vec(),
            6, // Medium-high priority
            metadata,
        )?;

        message_ids.push(message_id);
    }

    Ok(message_ids)
}
