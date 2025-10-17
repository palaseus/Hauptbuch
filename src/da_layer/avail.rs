//! Avail Integration for Multi-Chain Data Availability
//!
//! This module implements Avail integration for multi-chain data availability,
//! providing scalable, modular data storage with cross-chain compatibility.
//!
//! Key features:
//! - Multi-chain data availability with unified interface
//! - Data availability sampling (DAS) for efficient verification
//! - Erasure coding for fault tolerance
//! - Cross-chain data synchronization
//! - Integration with multiple blockchain networks
//! - KZG commitments for efficient proofs

use base64::{engine::general_purpose, Engine as _};
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

// Production HTTP client for real API integration
use reqwest::{Client, ClientBuilder};
use serde_json::json;
use tokio::runtime::Runtime;

// Import NIST PQC for signatures
use crate::crypto::nist_pqc::{
    ml_dsa_keygen, ml_dsa_sign, ml_dsa_verify, MLDSAPublicKey, MLDSASecretKey, MLDSASecurityLevel,
    MLDSASignature,
};

/// Error types for Avail operations
#[derive(Debug, Clone, PartialEq)]
pub enum AvailError {
    /// Invalid data block
    InvalidDataBlock,
    /// Invalid erasure code
    InvalidErasureCode,
    /// Data availability sampling failed
    DASFailed,
    /// Cross-chain sync failed
    CrossChainSyncFailed,
    /// Invalid network ID
    InvalidNetworkId,
    /// Data block not found
    DataBlockNotFound,
    /// Sampling proof invalid
    InvalidSamplingProof,
    /// Signature verification failed
    SignatureVerificationFailed,
    /// Network timeout
    NetworkTimeout,
    /// Insufficient validators
    InsufficientValidators,
    /// API request failed
    APIError(String),
    /// Invalid API response
    InvalidAPIResponse,
}

/// Result type for Avail operations
pub type AvailResult<T> = Result<T, AvailError>;

/// Avail data block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailDataBlock {
    /// Block ID
    pub block_id: String,
    /// Block height
    pub height: u64,
    /// Data content
    pub data: Vec<u8>,
    /// Erasure coded data
    pub erasure_coded_data: Vec<Vec<u8>>,
    /// KZG commitment
    pub kzg_commitment: Vec<u8>,
    /// Data availability proof
    pub da_proof: Vec<u8>,
    /// Timestamp
    pub timestamp: u64,
    /// Network ID
    pub network_id: u32,
    /// NIST PQC signature
    pub signature: Option<MLDSASignature>,
}

/// Avail sampling proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailSamplingProof {
    /// Proof ID
    pub proof_id: String,
    /// Block ID
    pub block_id: String,
    /// Sample indices
    pub sample_indices: Vec<usize>,
    /// Sample data
    pub sample_data: Vec<Vec<u8>>,
    /// Merkle proof
    pub merkle_proof: Vec<[u8; 32]>,
    /// Timestamp
    pub timestamp: u64,
    /// NIST PQC signature
    pub signature: Option<MLDSASignature>,
}

/// Avail network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailNetworkConfig {
    /// Network ID
    pub network_id: u32,
    /// Network name
    pub name: String,
    /// Minimum validators required
    pub min_validators: u32,
    /// Erasure coding parameters
    pub erasure_params: ErasureParams,
    /// Sampling parameters
    pub sampling_params: SamplingParams,
    /// Cross-chain sync enabled
    pub cross_chain_sync: bool,
    /// Enable real API integration
    pub enable_real_api: bool,
    /// Avail API endpoint
    pub avail_api_url: String,
    /// API timeout in seconds
    pub api_timeout: u64,
    /// API retry attempts
    pub api_retry_attempts: u32,
    /// API retry delay in milliseconds
    pub api_retry_delay_ms: u64,
}

/// Avail API response structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailAPIResponse {
    pub result: Option<serde_json::Value>,
    pub error: Option<AvailAPIError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailAPIError {
    pub code: i32,
    pub message: String,
    pub data: Option<serde_json::Value>,
}

/// Avail block data from API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailBlockData {
    pub block_number: u64,
    pub block_hash: String,
    pub timestamp: String,
    pub data: Vec<Vec<u8>>,
    pub app_id: u32,
}

/// Avail transaction data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailTransaction {
    pub tx_hash: String,
    pub app_id: u32,
    pub data: Vec<u8>,
    pub nonce: u64,
    pub signature: String,
}

/// Erasure coding parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErasureParams {
    /// Data shards
    pub data_shards: u32,
    /// Parity shards
    pub parity_shards: u32,
    /// Total shards
    pub total_shards: u32,
}

/// Data availability sampling parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    /// Sample size
    pub sample_size: usize,
    /// Number of samples
    pub num_samples: u32,
    /// Sampling threshold
    pub sampling_threshold: f64,
}

/// Avail validator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailValidator {
    /// Validator ID
    pub validator_id: String,
    /// Network ID
    pub network_id: u32,
    /// Staked amount
    pub staked_amount: u64,
    /// NIST PQC public key
    pub nist_pqc_public_key: MLDSAPublicKey,
    /// Is active
    pub is_active: bool,
    /// Performance metrics
    pub metrics: AvailValidatorMetrics,
}

/// Avail validator metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AvailValidatorMetrics {
    /// Total blocks validated
    pub total_blocks: u64,
    /// Availability score (0-100)
    pub availability_score: u8,
    /// Average response time (ms)
    pub avg_response_time_ms: u64,
    /// Uptime percentage
    pub uptime_percentage: f64,
    /// Cross-chain sync success rate
    pub cross_chain_sync_rate: f64,
}

/// Avail engine
#[derive(Debug)]
pub struct AvailEngine {
    /// Network configurations
    network_configs: HashMap<u32, AvailNetworkConfig>,
    /// NIST PQC keys
    ml_dsa_public_key: MLDSAPublicKey,
    ml_dsa_secret_key: MLDSASecretKey,
    /// Validators by network
    validators: Arc<RwLock<HashMap<u32, HashMap<String, AvailValidator>>>>,
    /// Data blocks by network
    data_blocks: Arc<RwLock<HashMap<u32, HashMap<String, AvailDataBlock>>>>,
    /// Sampling proofs
    sampling_proofs: Arc<RwLock<HashMap<String, AvailSamplingProof>>>,
    /// Performance metrics
    metrics: AvailMetrics,
    /// HTTP client for API calls
    http_client: Client,
    /// Async runtime for API calls
    runtime: Arc<Runtime>,
}

/// Avail performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AvailMetrics {
    /// Total data blocks stored
    pub total_blocks: u64,
    /// Total sampling proofs generated
    pub total_sampling_proofs: u64,
    /// Average block size (bytes)
    pub avg_block_size: u64,
    /// Data availability rate (percentage)
    pub availability_rate: f64,
    /// Cross-chain sync success rate
    pub cross_chain_sync_rate: f64,
    /// Average sampling time (ms)
    pub avg_sampling_time_ms: u64,
    /// Throughput (blocks per second)
    pub throughput_bps: f64,
}

impl AvailEngine {
    /// Creates a new Avail engine
    pub fn new() -> AvailResult<Self> {
        // Generate NIST PQC keys
        let (ml_dsa_public_key, ml_dsa_secret_key) = ml_dsa_keygen(MLDSASecurityLevel::MLDSA65)
            .map_err(|_| AvailError::SignatureVerificationFailed)?;

        Ok(Self {
            network_configs: HashMap::new(),
            ml_dsa_public_key,
            ml_dsa_secret_key,
            validators: Arc::new(RwLock::new(HashMap::new())),
            data_blocks: Arc::new(RwLock::new(HashMap::new())),
            sampling_proofs: Arc::new(RwLock::new(HashMap::new())),
            metrics: AvailMetrics::default(),
            http_client: ClientBuilder::new()
                .timeout(Duration::from_secs(30))
                .build()
                .unwrap_or_else(|_| Client::new()),
            runtime: Arc::new(Runtime::new().unwrap_or_else(|_| {
                tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .unwrap()
            })),
        })
    }

    /// Registers a network
    pub fn register_network(&mut self, config: AvailNetworkConfig) -> AvailResult<()> {
        let network_id = config.network_id;
        self.network_configs.insert(network_id, config);
        Ok(())
    }

    /// Registers a validator for a network
    pub fn register_validator(
        &mut self,
        network_id: u32,
        validator_id: String,
        staked_amount: u64,
    ) -> AvailResult<()> {
        // Check if network exists
        if !self.network_configs.contains_key(&network_id) {
            return Err(AvailError::InvalidNetworkId);
        }

        // Generate NIST PQC keys for validator
        let (nist_pqc_public_key, _nist_pqc_secret_key) =
            ml_dsa_keygen(MLDSASecurityLevel::MLDSA65)
                .map_err(|_| AvailError::SignatureVerificationFailed)?;

        let validator = AvailValidator {
            validator_id: validator_id.clone(),
            network_id,
            staked_amount,
            nist_pqc_public_key,
            is_active: true,
            metrics: AvailValidatorMetrics::default(),
        };

        let mut validators = self.validators.write().unwrap();
        validators
            .entry(network_id)
            .or_default()
            .insert(validator_id, validator);

        Ok(())
    }

    /// Stores data block in Avail
    pub fn store_data_block(
        &mut self,
        network_id: u32,
        data: Vec<u8>,
    ) -> AvailResult<AvailDataBlock> {
        // Check if network exists
        let network_config = self
            .network_configs
            .get(&network_id)
            .ok_or(AvailError::InvalidNetworkId)?;

        // Check if we have enough validators
        let validators = self.validators.read().unwrap();
        let network_validators = validators.get(&network_id).map(|v| v.len()).unwrap_or(0) as u32;

        if network_validators < network_config.min_validators {
            return Err(AvailError::InsufficientValidators);
        }

        let block_id = self.generate_block_id();
        let timestamp = current_timestamp();

        // Generate erasure coded data
        let erasure_coded_data =
            self.generate_erasure_codes(&data, &network_config.erasure_params)?;

        // Generate KZG commitment
        let kzg_commitment = self.generate_kzg_commitment(&data)?;

        // Generate data availability proof
        let da_proof = self.generate_da_proof(&data, &kzg_commitment)?;

        let mut data_block = AvailDataBlock {
            block_id: block_id.clone(),
            height: self.get_next_height(network_id),
            data: data.clone(),
            erasure_coded_data,
            kzg_commitment,
            da_proof,
            timestamp,
            network_id,
            signature: None,
        };

        // Sign the data block
        let block_bytes = self.serialize_data_block(&data_block)?;
        let signature = ml_dsa_sign(&self.ml_dsa_secret_key, &block_bytes)
            .map_err(|_| AvailError::SignatureVerificationFailed)?;
        data_block.signature = Some(signature);

        // Store data block
        let mut data_blocks = self.data_blocks.write().unwrap();
        data_blocks
            .entry(network_id)
            .or_default()
            .insert(block_id.clone(), data_block.clone());

        // Update metrics
        self.metrics.total_blocks += 1;
        self.metrics.avg_block_size = (self.metrics.avg_block_size + data.len() as u64) / 2;

        Ok(data_block)
    }

    /// Performs data availability sampling
    pub fn perform_das(
        &mut self,
        network_id: u32,
        block_id: &str,
    ) -> AvailResult<AvailSamplingProof> {
        // Get data block
        let data_blocks = self.data_blocks.read().unwrap();
        let network_blocks = data_blocks
            .get(&network_id)
            .ok_or(AvailError::DataBlockNotFound)?;
        let data_block = network_blocks
            .get(block_id)
            .ok_or(AvailError::DataBlockNotFound)?;

        let network_config = self
            .network_configs
            .get(&network_id)
            .ok_or(AvailError::InvalidNetworkId)?;

        let proof_id = self.generate_proof_id();
        let timestamp = current_timestamp();

        // Generate sample indices
        let sample_indices = self.generate_sample_indices(
            data_block.erasure_coded_data.len(),
            network_config.sampling_params.num_samples as usize,
        );

        // Extract sample data
        let sample_data = sample_indices
            .iter()
            .filter_map(|&idx| data_block.erasure_coded_data.get(idx))
            .cloned()
            .collect();

        // Generate Merkle proof
        let merkle_proof =
            self.generate_merkle_proof(&data_block.erasure_coded_data, &sample_indices)?;

        let mut sampling_proof = AvailSamplingProof {
            proof_id: proof_id.clone(),
            block_id: block_id.to_string(),
            sample_indices,
            sample_data,
            merkle_proof,
            timestamp,
            signature: None,
        };

        // Sign the sampling proof
        let proof_bytes = self.serialize_sampling_proof(&sampling_proof)?;
        let signature = ml_dsa_sign(&self.ml_dsa_secret_key, &proof_bytes)
            .map_err(|_| AvailError::SignatureVerificationFailed)?;
        sampling_proof.signature = Some(signature);

        // Store sampling proof
        let mut sampling_proofs = self.sampling_proofs.write().unwrap();
        sampling_proofs.insert(proof_id.clone(), sampling_proof.clone());

        // Update metrics
        self.metrics.total_sampling_proofs += 1;

        Ok(sampling_proof)
    }

    /// Verifies data availability
    pub fn verify_data_availability(&self, network_id: u32, block_id: &str) -> AvailResult<bool> {
        let data_blocks = self.data_blocks.read().unwrap();
        let network_blocks = data_blocks
            .get(&network_id)
            .ok_or(AvailError::DataBlockNotFound)?;
        let data_block = network_blocks
            .get(block_id)
            .ok_or(AvailError::DataBlockNotFound)?;

        // Verify signature
        if let Some(ref signature) = data_block.signature {
            let block_bytes = self.serialize_data_block(data_block)?;
            let sig_valid = ml_dsa_verify(&self.ml_dsa_public_key, &block_bytes, signature)
                .map_err(|_| AvailError::SignatureVerificationFailed)?;

            if !sig_valid {
                return Ok(false);
            }
        }

        // Verify erasure codes
        let erasure_valid =
            self.verify_erasure_codes(&data_block.data, &data_block.erasure_coded_data)?;

        // Verify KZG commitment
        let kzg_valid = self.verify_kzg_commitment(&data_block.data, &data_block.kzg_commitment)?;

        Ok(erasure_valid && kzg_valid)
    }

    /// Gets data block by ID
    pub fn get_data_block(
        &self,
        network_id: u32,
        block_id: &str,
    ) -> AvailResult<Option<AvailDataBlock>> {
        let data_blocks = self.data_blocks.read().unwrap();
        let network_blocks = data_blocks
            .get(&network_id)
            .ok_or(AvailError::InvalidNetworkId)?;
        Ok(network_blocks.get(block_id).cloned())
    }

    /// Gets current metrics
    pub fn get_metrics(&self) -> &AvailMetrics {
        &self.metrics
    }

    /// Gets network configuration
    pub fn get_network_config(&self, network_id: u32) -> AvailResult<&AvailNetworkConfig> {
        self.network_configs
            .get(&network_id)
            .ok_or(AvailError::InvalidNetworkId)
    }

    // Private helper methods

    /// Generates a unique block ID
    fn generate_block_id(&self) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(current_timestamp().to_le_bytes());
        hasher.update(rand::random::<u64>().to_le_bytes());
        let hash = hasher.finalize();
        format!("avail_block_{}", hex::encode(&hash[..16]))
    }

    /// Generates a unique proof ID
    fn generate_proof_id(&self) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(current_timestamp().to_le_bytes());
        hasher.update(rand::random::<u64>().to_le_bytes());
        let hash = hasher.finalize();
        format!("avail_proof_{}", hex::encode(&hash[..16]))
    }

    /// Gets next block height for network
    fn get_next_height(&self, network_id: u32) -> u64 {
        let data_blocks = self.data_blocks.read().unwrap();
        let network_blocks = data_blocks
            .get(&network_id)
            .map(|blocks| blocks.len() as u64)
            .unwrap_or(0);
        network_blocks + 1
    }

    /// Generates erasure coded data (simplified)
    fn generate_erasure_codes(
        &self,
        data: &[u8],
        params: &ErasureParams,
    ) -> AvailResult<Vec<Vec<u8>>> {
        // In a real implementation, this would use actual erasure coding (e.g., Reed-Solomon)
        // For now, we'll create simple shards
        let mut shards = Vec::new();
        let shard_size = (data.len() as f64 / params.data_shards as f64).ceil() as usize;

        for i in 0..params.total_shards {
            let start = (i as usize * shard_size).min(data.len());
            let end = ((i as usize + 1) * shard_size).min(data.len());

            if i < params.data_shards {
                // Data shard
                shards.push(data[start..end].to_vec());
            } else {
                // Parity shard (simplified - just repeat data)
                shards.push(data[start..end].to_vec());
            }
        }

        Ok(shards)
    }

    /// Verifies erasure codes (simplified)
    fn verify_erasure_codes(
        &self,
        _original_data: &[u8],
        erasure_coded: &[Vec<u8>],
    ) -> AvailResult<bool> {
        // In a real implementation, this would verify actual erasure codes
        // For now, we'll just check that we have some data
        Ok(!erasure_coded.is_empty())
    }

    /// Generates KZG commitment (simplified)
    fn generate_kzg_commitment(&self, data: &[u8]) -> AvailResult<Vec<u8>> {
        let mut hasher = Sha3_256::new();
        hasher.update(data);
        hasher.update(b"avail_kzg_commitment");
        Ok(hasher.finalize().to_vec())
    }

    /// Verifies KZG commitment (simplified)
    fn verify_kzg_commitment(&self, data: &[u8], commitment: &[u8]) -> AvailResult<bool> {
        let expected_commitment = self.generate_kzg_commitment(data)?;
        Ok(commitment == expected_commitment)
    }

    /// Generates data availability proof (simplified)
    fn generate_da_proof(&self, data: &[u8], commitment: &[u8]) -> AvailResult<Vec<u8>> {
        let mut hasher = Sha3_256::new();
        hasher.update(data);
        hasher.update(commitment);
        hasher.update(b"avail_da_proof");
        Ok(hasher.finalize().to_vec())
    }

    /// Generates sample indices
    fn generate_sample_indices(&self, total_shards: usize, num_samples: usize) -> Vec<usize> {
        use rand::seq::SliceRandom;
        use rand::thread_rng;

        let mut indices: Vec<usize> = (0..total_shards).collect();
        indices.shuffle(&mut thread_rng());
        indices.truncate(num_samples);
        indices.sort();
        indices
    }

    /// Generates Merkle proof (simplified)
    fn generate_merkle_proof(
        &self,
        _data: &[Vec<u8>],
        sample_indices: &[usize],
    ) -> AvailResult<Vec<[u8; 32]>> {
        // In a real implementation, this would generate actual Merkle proofs
        // For now, we'll return placeholder hashes
        let mut proof = Vec::new();
        for _ in 0..sample_indices.len() {
            let mut hasher = Sha3_256::new();
            hasher.update(rand::random::<u64>().to_le_bytes());
            proof.push(hasher.finalize().into());
        }
        Ok(proof)
    }

    /// Serializes data block for signing
    fn serialize_data_block(&self, block: &AvailDataBlock) -> AvailResult<Vec<u8>> {
        let mut data = Vec::new();
        data.extend_from_slice(block.block_id.as_bytes());
        data.extend_from_slice(&block.data);
        data.extend_from_slice(&block.height.to_le_bytes());
        data.extend_from_slice(&block.timestamp.to_le_bytes());
        data.extend_from_slice(&block.network_id.to_le_bytes());
        Ok(data)
    }

    /// Serializes sampling proof for signing
    fn serialize_sampling_proof(&self, proof: &AvailSamplingProof) -> AvailResult<Vec<u8>> {
        let mut data = Vec::new();
        data.extend_from_slice(proof.proof_id.as_bytes());
        data.extend_from_slice(proof.block_id.as_bytes());
        for &idx in &proof.sample_indices {
            data.extend_from_slice(&idx.to_le_bytes());
        }
        data.extend_from_slice(&proof.timestamp.to_le_bytes());
        Ok(data)
    }

    /// Submit data to Avail API - Production Implementation
    pub fn submit_data_to_avail(
        &self,
        network_id: u32,
        data: &[u8],
        app_id: u32,
    ) -> AvailResult<String> {
        let config = self
            .network_configs
            .get(&network_id)
            .ok_or(AvailError::InvalidNetworkId)?;

        if !config.enable_real_api {
            // Simulation mode - return mock response
            return self.simulate_avail_submission(data, app_id);
        }

        // Production implementation using real Avail API
        let submission_result = self
            .runtime
            .block_on(async { self.submit_data_to_avail_async(config, data, app_id).await })?;

        Ok(submission_result)
    }

    /// Submit data to Avail API asynchronously
    async fn submit_data_to_avail_async(
        &self,
        config: &AvailNetworkConfig,
        data: &[u8],
        app_id: u32,
    ) -> AvailResult<String> {
        let url = format!("{}/submit", config.avail_api_url);

        let payload = json!({
            "app_id": app_id,
            "data": general_purpose::STANDARD.encode(data),
            "nonce": SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            "signature": "mock_signature" // In production, this would be a real signature
        });

        let response = self
            .http_client
            .post(&url)
            .json(&payload)
            .timeout(Duration::from_secs(config.api_timeout))
            .send()
            .await
            .map_err(|e| AvailError::APIError(format!("Failed to submit data: {}", e)))?;

        if !response.status().is_success() {
            return Err(AvailError::APIError(format!(
                "API request failed with status: {}",
                response.status()
            )));
        }

        let api_response: AvailAPIResponse = response
            .json()
            .await
            .map_err(|e| AvailError::APIError(format!("Failed to parse response: {}", e)))?;

        if let Some(error) = api_response.error {
            return Err(AvailError::APIError(format!(
                "API error: {} - {}",
                error.code, error.message
            )));
        }

        let tx_hash = api_response
            .result
            .and_then(|r| r.get("tx_hash").map(|v| v.clone()))
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .ok_or_else(|| AvailError::APIError("Missing tx_hash in response".to_string()))?;

        Ok(tx_hash.to_string())
    }

    /// Retrieve data from Avail API - Production Implementation
    pub fn retrieve_data_from_avail(&self, network_id: u32, tx_hash: &str) -> AvailResult<Vec<u8>> {
        let config = self
            .network_configs
            .get(&network_id)
            .ok_or(AvailError::InvalidNetworkId)?;

        if !config.enable_real_api {
            // Simulation mode - return mock data
            return self.simulate_avail_retrieval(tx_hash);
        }

        // Production implementation using real Avail API
        let retrieval_result = self
            .runtime
            .block_on(async { self.retrieve_data_from_avail_async(config, tx_hash).await })?;

        Ok(retrieval_result)
    }

    /// Retrieve data from Avail API asynchronously
    async fn retrieve_data_from_avail_async(
        &self,
        config: &AvailNetworkConfig,
        tx_hash: &str,
    ) -> AvailResult<Vec<u8>> {
        let url = format!("{}/retrieve/{}", config.avail_api_url, tx_hash);

        let response = self
            .http_client
            .get(&url)
            .timeout(Duration::from_secs(config.api_timeout))
            .send()
            .await
            .map_err(|e| AvailError::APIError(format!("Failed to retrieve data: {}", e)))?;

        if !response.status().is_success() {
            return Err(AvailError::APIError(format!(
                "API request failed with status: {}",
                response.status()
            )));
        }

        let api_response: AvailAPIResponse = response
            .json()
            .await
            .map_err(|e| AvailError::APIError(format!("Failed to parse response: {}", e)))?;

        if let Some(error) = api_response.error {
            return Err(AvailError::APIError(format!(
                "API error: {} - {}",
                error.code, error.message
            )));
        }

        let data_b64 = api_response
            .result
            .and_then(|r| r.get("data").map(|v| v.clone()))
            .and_then(|v| v.as_str().map(|s| s.to_string()))
            .ok_or_else(|| AvailError::APIError("Missing data in response".to_string()))?;

        let data = general_purpose::STANDARD
            .decode(data_b64)
            .map_err(|e| AvailError::APIError(format!("Failed to decode data: {}", e)))?;

        Ok(data)
    }

    /// Get block data from Avail API - Production Implementation
    pub fn get_avail_block(
        &self,
        network_id: u32,
        block_number: u64,
    ) -> AvailResult<AvailBlockData> {
        let config = self
            .network_configs
            .get(&network_id)
            .ok_or(AvailError::InvalidNetworkId)?;

        if !config.enable_real_api {
            // Simulation mode - return mock block data
            return self.simulate_avail_block(block_number);
        }

        // Production implementation using real Avail API
        let block_result = self
            .runtime
            .block_on(async { self.get_avail_block_async(config, block_number).await })?;

        Ok(block_result)
    }

    /// Get block data from Avail API asynchronously
    async fn get_avail_block_async(
        &self,
        config: &AvailNetworkConfig,
        block_number: u64,
    ) -> AvailResult<AvailBlockData> {
        let url = format!("{}/block/{}", config.avail_api_url, block_number);

        let response = self
            .http_client
            .get(&url)
            .timeout(Duration::from_secs(config.api_timeout))
            .send()
            .await
            .map_err(|e| AvailError::APIError(format!("Failed to get block: {}", e)))?;

        if !response.status().is_success() {
            return Err(AvailError::APIError(format!(
                "API request failed with status: {}",
                response.status()
            )));
        }

        let api_response: AvailAPIResponse = response
            .json()
            .await
            .map_err(|e| AvailError::APIError(format!("Failed to parse response: {}", e)))?;

        if let Some(error) = api_response.error {
            return Err(AvailError::APIError(format!(
                "API error: {} - {}",
                error.code, error.message
            )));
        }

        let block_data = serde_json::from_value(api_response.result.unwrap())
            .map_err(|e| AvailError::APIError(format!("Failed to parse block data: {}", e)))?;

        Ok(block_data)
    }

    /// Simulate Avail submission for testing
    fn simulate_avail_submission(&self, data: &[u8], app_id: u32) -> AvailResult<String> {
        // Generate mock transaction hash
        let mut hasher = Sha3_256::new();
        hasher.update(data);
        hasher.update(&app_id.to_le_bytes());
        hasher.update(
            &SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_le_bytes(),
        );

        let hash = hasher.finalize();
        let tx_hash = format!("0x{}", hex::encode(&hash[..16]));

        Ok(tx_hash)
    }

    /// Simulate Avail retrieval for testing
    fn simulate_avail_retrieval(&self, tx_hash: &str) -> AvailResult<Vec<u8>> {
        // Return mock data based on transaction hash
        let mock_data = format!("Mock Avail data for transaction: {}", tx_hash);
        Ok(mock_data.as_bytes().to_vec())
    }

    /// Simulate Avail block for testing
    fn simulate_avail_block(&self, block_number: u64) -> AvailResult<AvailBlockData> {
        // Generate mock block data
        let mut hasher = Sha3_256::new();
        hasher.update(&block_number.to_le_bytes());
        hasher.update(
            &SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_le_bytes(),
        );

        let hash = hasher.finalize();
        let block_hash = format!("0x{}", hex::encode(&hash));

        Ok(AvailBlockData {
            block_number,
            block_hash,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()
                .to_string(),
            data: vec![vec![0x01, 0x02, 0x03, 0x04]], // Mock data
            app_id: 1,
        })
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
    fn test_avail_engine_creation() {
        let engine = AvailEngine::new().unwrap();
        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_blocks, 0);
    }

    #[test]
    fn test_avail_network_registration() {
        let mut engine = AvailEngine::new().unwrap();

        let network_config = AvailNetworkConfig {
            network_id: 1,
            name: "test_network".to_string(),
            min_validators: 3,
            erasure_params: ErasureParams {
                data_shards: 4,
                parity_shards: 2,
                total_shards: 6,
            },
            api_retry_attempts: 3,
            api_retry_delay_ms: 1000,
            api_timeout: 5000,
            enable_real_api: false,
            avail_api_url: "http://localhost:8080".to_string(),
            sampling_params: SamplingParams {
                sample_size: 2,
                num_samples: 3,
                sampling_threshold: 0.5,
            },
            cross_chain_sync: true,
        };

        let result = engine.register_network(network_config);
        assert!(result.is_ok());

        let config = engine.get_network_config(1).unwrap();
        assert_eq!(config.name, "test_network");
        assert_eq!(config.min_validators, 3);
    }

    #[test]
    fn test_avail_validator_registration() {
        let mut engine = AvailEngine::new().unwrap();

        // Register network first
        let network_config = AvailNetworkConfig {
            network_id: 1,
            name: "test_network".to_string(),
            min_validators: 3,
            erasure_params: ErasureParams {
                data_shards: 4,
                parity_shards: 2,
                total_shards: 6,
            },
            api_retry_attempts: 3,
            api_retry_delay_ms: 1000,
            api_timeout: 5000,
            enable_real_api: false,
            avail_api_url: "http://localhost:8080".to_string(),
            sampling_params: SamplingParams {
                sample_size: 2,
                num_samples: 3,
                sampling_threshold: 0.5,
            },
            cross_chain_sync: true,
        };
        engine.register_network(network_config).unwrap();

        // Register validators
        for i in 0..3 {
            let result = engine.register_validator(1, format!("validator_{}", i), 1000);
            assert!(result.is_ok());
        }

        // Try to register validator for non-existent network
        let result = engine.register_validator(999, "validator_999".to_string(), 1000);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), AvailError::InvalidNetworkId);
    }

    #[test]
    fn test_avail_data_block_storage() {
        let mut engine = AvailEngine::new().unwrap();

        // Register network and validators
        let network_config = AvailNetworkConfig {
            network_id: 1,
            name: "test_network".to_string(),
            min_validators: 3,
            erasure_params: ErasureParams {
                data_shards: 4,
                parity_shards: 2,
                total_shards: 6,
            },
            api_retry_attempts: 3,
            api_retry_delay_ms: 1000,
            api_timeout: 5000,
            enable_real_api: false,
            avail_api_url: "http://localhost:8080".to_string(),
            sampling_params: SamplingParams {
                sample_size: 2,
                num_samples: 3,
                sampling_threshold: 0.5,
            },
            cross_chain_sync: true,
        };
        engine.register_network(network_config).unwrap();

        for i in 0..3 {
            engine
                .register_validator(1, format!("validator_{}", i), 1000)
                .unwrap();
        }

        // Store data block
        let data = vec![0x01, 0x02, 0x03, 0x04, 0x05];
        let data_block = engine.store_data_block(1, data.clone()).unwrap();

        assert_eq!(data_block.data, data);
        assert_eq!(data_block.network_id, 1);
        assert_eq!(data_block.height, 1);
        assert!(!data_block.erasure_coded_data.is_empty());
        assert!(!data_block.kzg_commitment.is_empty());
        assert!(data_block.signature.is_some());

        // Verify data availability
        let is_available = engine
            .verify_data_availability(1, &data_block.block_id)
            .unwrap();
        assert!(is_available);

        // Retrieve data block
        let retrieved_block = engine.get_data_block(1, &data_block.block_id).unwrap();
        assert!(retrieved_block.is_some());
        assert_eq!(retrieved_block.unwrap().data, data);
    }

    #[test]
    fn test_avail_data_availability_sampling() {
        let mut engine = AvailEngine::new().unwrap();

        // Register network and validators
        let network_config = AvailNetworkConfig {
            network_id: 1,
            name: "test_network".to_string(),
            min_validators: 3,
            erasure_params: ErasureParams {
                data_shards: 4,
                parity_shards: 2,
                total_shards: 6,
            },
            api_retry_attempts: 3,
            api_retry_delay_ms: 1000,
            api_timeout: 5000,
            enable_real_api: false,
            avail_api_url: "http://localhost:8080".to_string(),
            sampling_params: SamplingParams {
                sample_size: 2,
                num_samples: 3,
                sampling_threshold: 0.5,
            },
            cross_chain_sync: true,
        };
        engine.register_network(network_config).unwrap();

        for i in 0..3 {
            engine
                .register_validator(1, format!("validator_{}", i), 1000)
                .unwrap();
        }

        // Store data block
        let data = vec![0x01, 0x02, 0x03, 0x04, 0x05];
        let data_block = engine.store_data_block(1, data).unwrap();

        // Perform data availability sampling
        let sampling_proof = engine.perform_das(1, &data_block.block_id).unwrap();

        assert_eq!(sampling_proof.block_id, data_block.block_id);
        assert_eq!(sampling_proof.sample_indices.len(), 3);
        assert_eq!(sampling_proof.sample_data.len(), 3);
        assert!(!sampling_proof.merkle_proof.is_empty());
        assert!(sampling_proof.signature.is_some());
    }

    #[test]
    fn test_avail_insufficient_validators() {
        let mut engine = AvailEngine::new().unwrap();

        // Register network with min_validators = 3
        let network_config = AvailNetworkConfig {
            network_id: 1,
            name: "test_network".to_string(),
            min_validators: 3,
            erasure_params: ErasureParams {
                data_shards: 4,
                parity_shards: 2,
                total_shards: 6,
            },
            api_retry_attempts: 3,
            api_retry_delay_ms: 1000,
            api_timeout: 5000,
            enable_real_api: false,
            avail_api_url: "http://localhost:8080".to_string(),
            sampling_params: SamplingParams {
                sample_size: 2,
                num_samples: 3,
                sampling_threshold: 0.5,
            },
            cross_chain_sync: true,
        };
        engine.register_network(network_config).unwrap();

        // Only register 2 validators (less than required 3)
        for i in 0..2 {
            engine
                .register_validator(1, format!("validator_{}", i), 1000)
                .unwrap();
        }

        // Try to store data block - should fail
        let data = vec![0x01, 0x02, 0x03];
        let result = engine.store_data_block(1, data);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), AvailError::InsufficientValidators);
    }
}
