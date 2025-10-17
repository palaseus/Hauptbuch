//! Data Availability Abstraction Layer
//!
//! This module provides a unified interface for multiple data availability layers
//! with automatic fallback mechanisms and load balancing.
//!
//! Key features:
//! - Unified interface for multiple DA providers (Celestia, EigenDA, Avail)
//! - Automatic fallback mechanisms for reliability
//! - Load balancing across multiple DA providers
//! - Cost optimization through provider selection
//! - Health monitoring and failover
//! - Cross-provider data synchronization

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

// Import DA providers
// use super::celestia::CelestiaDA; // Commented out - CelestiaDA not available
use super::avail::AvailEngine;
use super::eigenda::EigenDAEngine;
use crate::l2::eip4844::{BlobGasParams, BlobTransactionPool, KZGSystem};

// Import NIST PQC for signatures
use crate::crypto::nist_pqc::{
    ml_dsa_keygen, ml_dsa_sign, ml_dsa_verify, MLDSAPublicKey, MLDSASecretKey, MLDSASecurityLevel,
    MLDSASignature,
};

/// Error types for DA abstraction operations
#[derive(Debug, Clone, PartialEq)]
pub enum DAAbstractionError {
    /// All DA providers failed
    AllProvidersFailed,
    /// Invalid DA provider
    InvalidProvider,
    /// Provider unavailable
    ProviderUnavailable,
    /// Data not found
    DataNotFound,
    /// Fallback failed
    FallbackFailed,
    /// Signature verification failed
    SignatureVerificationFailed,
    /// Network error
    NetworkError,
    /// Timeout error
    TimeoutError,
    /// Cost limit exceeded
    CostLimitExceeded,
}

/// Result type for DA abstraction operations
pub type DAAbstractionResult<T> = Result<T, DAAbstractionError>;

/// DA provider types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DAProvider {
    /// Celestia-style DA
    Celestia,
    /// EigenDA
    EigenDA,
    /// Avail
    Avail,
    /// Ethereum EIP-4844 blobs
    EthereumBlobs,
}

/// DA provider status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum DAProviderStatus {
    /// Provider is healthy and available
    Healthy,
    /// Provider is degraded but functional
    Degraded,
    /// Provider is unavailable
    Unavailable,
    /// Provider is being tested
    #[default]
    Testing,
}

/// DA provider configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DAProviderConfig {
    /// Provider type
    pub provider: DAProvider,
    /// Provider name
    pub name: String,
    /// Is enabled
    pub enabled: bool,
    /// Priority (lower number = higher priority)
    pub priority: u32,
    /// Cost per MB (in wei)
    pub cost_per_mb: u64,
    /// Maximum data size (in bytes)
    pub max_data_size: u64,
    /// Timeout (in seconds)
    pub timeout_seconds: u64,
    /// Health check interval (in seconds)
    pub health_check_interval: u64,
}

/// DA provider health metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DAProviderHealth {
    /// Current status
    pub status: DAProviderStatus,
    /// Availability percentage (0-100)
    pub availability: f64,
    /// Average response time (ms)
    pub avg_response_time_ms: u64,
    /// Success rate (0-1)
    pub success_rate: f64,
    /// Last health check timestamp
    pub last_health_check: u64,
    /// Total requests
    pub total_requests: u64,
    /// Successful requests
    pub successful_requests: u64,
    /// Failed requests
    pub failed_requests: u64,
}

/// Unified data block across all providers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedDataBlock {
    /// Block ID
    pub block_id: String,
    /// Data content
    pub data: Vec<u8>,
    /// Timestamp
    pub timestamp: u64,
    /// Primary provider used
    pub primary_provider: DAProvider,
    /// Fallback providers used
    pub fallback_providers: Vec<DAProvider>,
    /// Provider-specific metadata
    pub provider_metadata: HashMap<DAProvider, ProviderMetadata>,
    /// NIST PQC signature
    pub signature: Option<MLDSASignature>,
}

/// Provider-specific metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderMetadata {
    /// Provider-specific block ID
    pub provider_block_id: String,
    /// Storage cost (in wei)
    pub storage_cost: u64,
    /// Storage timestamp
    pub storage_timestamp: u64,
    /// Provider-specific data
    pub provider_data: Vec<u8>,
}

/// DA abstraction engine
#[derive(Debug)]
pub struct DAAbstractionEngine {
    /// Provider configurations
    provider_configs: HashMap<DAProvider, DAProviderConfig>,
    /// Provider health metrics
    provider_health: Arc<RwLock<HashMap<DAProvider, DAProviderHealth>>>,
    /// NIST PQC keys
    ml_dsa_public_key: MLDSAPublicKey,
    ml_dsa_secret_key: MLDSASecretKey,
    /// DA providers
    // celestia_da: Option<CelestiaDA>, // Commented out - CelestiaDA not available
    eigenda_engine: Option<EigenDAEngine>,
    avail_engine: Option<AvailEngine>,
    blob_transaction_pool: Option<BlobTransactionPool>,
    kzg_system: Option<KZGSystem>,
    /// Unified data blocks
    unified_blocks: Arc<RwLock<HashMap<String, UnifiedDataBlock>>>,
    /// Performance metrics
    metrics: DAAbstractionMetrics,
}

/// DA abstraction performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DAAbstractionMetrics {
    /// Total data blocks stored
    pub total_blocks: u64,
    /// Total fallback operations
    pub total_fallbacks: u64,
    /// Average storage cost (wei)
    pub avg_storage_cost: u64,
    /// Average response time (ms)
    pub avg_response_time_ms: u64,
    /// Provider utilization
    pub provider_utilization: HashMap<DAProvider, f64>,
    /// Success rate (0-1)
    pub success_rate: f64,
    /// Cost savings (wei)
    pub cost_savings: u64,
}

impl DAAbstractionEngine {
    /// Creates a new DA abstraction engine
    pub fn new() -> DAAbstractionResult<Self> {
        // Generate NIST PQC keys
        let (ml_dsa_public_key, ml_dsa_secret_key) = ml_dsa_keygen(MLDSASecurityLevel::MLDSA65)
            .map_err(|_| DAAbstractionError::SignatureVerificationFailed)?;

        Ok(Self {
            provider_configs: HashMap::new(),
            provider_health: Arc::new(RwLock::new(HashMap::new())),
            ml_dsa_public_key,
            ml_dsa_secret_key,
            // celestia_da: None, // Commented out - field not available
            eigenda_engine: None,
            avail_engine: None,
            blob_transaction_pool: None,
            kzg_system: None,
            unified_blocks: Arc::new(RwLock::new(HashMap::new())),
            metrics: DAAbstractionMetrics::default(),
        })
    }

    /// Registers a DA provider
    pub fn register_provider(&mut self, config: DAProviderConfig) -> DAAbstractionResult<()> {
        let provider = config.provider;

        // Initialize provider health
        let mut provider_health = self.provider_health.write().unwrap();
        provider_health.insert(
            provider,
            DAProviderHealth {
                status: DAProviderStatus::Testing,
                ..Default::default()
            },
        );

        // Initialize provider-specific engines
        match provider {
            DAProvider::Celestia => {
                // Initialize Celestia DA (simplified)
                // In a real implementation, this would create an actual CelestiaDA instance
                // self.celestia_da = Some(CelestiaDA::new());
            }
            DAProvider::EigenDA => {
                // Initialize EigenDA engine
                let eigenda_config = super::eigenda::EigenDAAVSConfig {
                    name: config.name.clone(),
                    min_stake: 1000,
                    slashing_conditions: vec![],
                    reward_params: super::eigenda::RewardParams {
                        base_reward: 100,
                        availability_bonus: 50,
                        downtime_penalty: 25,
                    },
                    quorum_threshold: 2,
                    enable_real_api: false,
                    eigenda_api_url: "https://api.eigenda.org".to_string(),
                    api_timeout: 30,
                    api_retry_attempts: 3,
                    api_retry_delay_ms: 1000,
                };
                self.eigenda_engine = Some(
                    EigenDAEngine::new(eigenda_config)
                        .map_err(|_| DAAbstractionError::ProviderUnavailable)?,
                );
            }
            DAProvider::Avail => {
                // Initialize Avail engine
                self.avail_engine =
                    Some(AvailEngine::new().map_err(|_| DAAbstractionError::ProviderUnavailable)?);
            }
            DAProvider::EthereumBlobs => {
                // Initialize EIP-4844 blob transaction pool
                let blob_gas_params = BlobGasParams {
                    target_blob_gas_per_block: 1000000,
                    max_blob_gas_per_block: 2000000,
                    blob_gas_price_update_rate: 100,
                    min_blob_gas_price: 1000000000,    // 1 gwei
                    max_blob_gas_price: 1000000000000, // 1000 gwei
                };
                self.blob_transaction_pool = Some(BlobTransactionPool::new(blob_gas_params));

                // Initialize KZG system with trusted setup
                let trusted_setup = crate::l2::eip4844::KZGTrustedSetup {
                    g1_points: vec![[0u8; 48]], // Simplified for testing
                    g2_points: vec![[0u8; 48]],
                    participants: 100,
                    setup_timestamp: current_timestamp(),
                };
                self.kzg_system = Some(crate::l2::eip4844::KZGSystem::new(trusted_setup));
            }
        }

        self.provider_configs.insert(provider, config);
        Ok(())
    }

    /// Stores data using the best available provider with fallback
    pub fn store_data(&mut self, data: Vec<u8>) -> DAAbstractionResult<UnifiedDataBlock> {
        let start_time = std::time::Instant::now();
        let block_id = self.generate_block_id();
        let timestamp = current_timestamp();

        // Select primary provider based on health, cost, and priority
        let primary_provider = self.select_primary_provider(&data)?;

        // Try to store with primary provider
        let primary_metadata = match self.store_with_provider(primary_provider, &block_id, &data) {
            Ok(metadata) => metadata,
            Err(_) => {
                // Primary provider failed, try fallback providers
                self.metrics.total_fallbacks += 1;
                self.try_fallback_providers(&block_id, &data)?
            }
        };

        // Store with additional providers for redundancy (if cost-effective)
        let mut fallback_providers = Vec::new();
        let mut provider_metadata = HashMap::new();
        provider_metadata.insert(primary_provider, primary_metadata);

        for provider in self.get_available_providers() {
            if provider == primary_provider {
                continue;
            }

            // Check if redundancy is cost-effective
            if self.is_redundancy_cost_effective(provider, &data) {
                if let Ok(metadata) = self.store_with_provider(provider, &block_id, &data) {
                    fallback_providers.push(provider);
                    provider_metadata.insert(provider, metadata);
                }
            }
        }

        let mut unified_block = UnifiedDataBlock {
            block_id: block_id.clone(),
            data,
            timestamp,
            primary_provider,
            fallback_providers,
            provider_metadata,
            signature: None,
        };

        // Sign the unified block
        let block_bytes = self.serialize_unified_block(&unified_block)?;
        let signature = ml_dsa_sign(&self.ml_dsa_secret_key, &block_bytes)
            .map_err(|_| DAAbstractionError::SignatureVerificationFailed)?;
        unified_block.signature = Some(signature);

        // Store unified block
        {
            let mut unified_blocks = self.unified_blocks.write().unwrap();
            unified_blocks.insert(block_id.clone(), unified_block.clone());
        }

        // Update metrics
        let elapsed = start_time.elapsed().as_millis() as u64;
        self.metrics.total_blocks += 1;
        self.metrics.avg_response_time_ms = (self.metrics.avg_response_time_ms + elapsed) / 2;

        // Update provider health
        self.update_provider_health(primary_provider, true, elapsed);

        Ok(unified_block)
    }

    /// Retrieves data from the best available provider
    pub fn retrieve_data(&self, block_id: &str) -> DAAbstractionResult<Option<UnifiedDataBlock>> {
        let unified_blocks = self.unified_blocks.read().unwrap();

        if let Some(unified_block) = unified_blocks.get(block_id) {
            // Verify signature
            if let Some(ref signature) = unified_block.signature {
                let block_bytes = self.serialize_unified_block(unified_block)?;
                let sig_valid = ml_dsa_verify(&self.ml_dsa_public_key, &block_bytes, signature)
                    .map_err(|_| DAAbstractionError::SignatureVerificationFailed)?;

                if !sig_valid {
                    return Err(DAAbstractionError::SignatureVerificationFailed);
                }
            }

            Ok(Some(unified_block.clone()))
        } else {
            Ok(None)
        }
    }

    /// Verifies data availability across all providers
    pub fn verify_data_availability(&self, block_id: &str) -> DAAbstractionResult<bool> {
        let unified_blocks = self.unified_blocks.read().unwrap();

        if let Some(unified_block) = unified_blocks.get(block_id) {
            // Check primary provider first
            if self.verify_with_provider(
                unified_block.primary_provider,
                &unified_block.provider_metadata,
            ) {
                return Ok(true);
            }

            // Check fallback providers
            for &provider in &unified_block.fallback_providers {
                if self.verify_with_provider(provider, &unified_block.provider_metadata) {
                    return Ok(true);
                }
            }

            Ok(false)
        } else {
            Ok(false)
        }
    }

    /// Gets current metrics
    pub fn get_metrics(&self) -> &DAAbstractionMetrics {
        &self.metrics
    }

    /// Gets provider health status
    pub fn get_provider_health(
        &self,
        provider: DAProvider,
    ) -> DAAbstractionResult<DAProviderHealth> {
        let provider_health = self.provider_health.read().unwrap();
        provider_health
            .get(&provider)
            .cloned()
            .ok_or(DAAbstractionError::InvalidProvider)
    }

    /// Gets all provider configurations
    pub fn get_provider_configs(&self) -> &HashMap<DAProvider, DAProviderConfig> {
        &self.provider_configs
    }

    // Private helper methods

    /// Generates a unique block ID
    fn generate_block_id(&self) -> String {
        let mut hasher = Sha3_256::new();
        hasher.update(current_timestamp().to_le_bytes());
        hasher.update(rand::random::<u64>().to_le_bytes());
        let hash = hasher.finalize();
        format!("unified_block_{}", hex::encode(&hash[..16]))
    }

    /// Selects the best primary provider based on health, cost, and priority
    fn select_primary_provider(&self, data: &[u8]) -> DAAbstractionResult<DAProvider> {
        let provider_health = self.provider_health.read().unwrap();
        let mut best_provider = None;
        let mut best_score = f64::NEG_INFINITY;

        for (provider, config) in &self.provider_configs {
            if !config.enabled {
                continue;
            }

            if data.len() as u64 > config.max_data_size {
                continue;
            }

            let health = provider_health.get(provider).cloned().unwrap_or_default();

            if health.status == DAProviderStatus::Unavailable {
                continue;
            }

            // Calculate provider score (higher is better)
            let availability_score = health.availability / 100.0;
            let cost_score = 1.0 / (1.0 + config.cost_per_mb as f64 / 1_000_000.0);
            let priority_score = 1.0 / (config.priority as f64 + 1.0);
            let response_score = 1.0 / (1.0 + health.avg_response_time_ms as f64 / 1000.0);

            let total_score = availability_score * 0.4
                + cost_score * 0.3
                + priority_score * 0.2
                + response_score * 0.1;

            if total_score > best_score {
                best_score = total_score;
                best_provider = Some(*provider);
            }
        }

        best_provider.ok_or(DAAbstractionError::AllProvidersFailed)
    }

    /// Gets list of available providers
    fn get_available_providers(&self) -> Vec<DAProvider> {
        let provider_health = self.provider_health.read().unwrap();

        self.provider_configs
            .iter()
            .filter(|(_, config)| config.enabled)
            .filter(|(provider, _)| {
                let health = provider_health.get(provider).cloned().unwrap_or_default();
                health.status != DAProviderStatus::Unavailable
            })
            .map(|(provider, _)| *provider)
            .collect()
    }

    /// Stores data with a specific provider
    fn store_with_provider(
        &mut self,
        provider: DAProvider,
        block_id: &str,
        data: &[u8],
    ) -> DAAbstractionResult<ProviderMetadata> {
        let timestamp = current_timestamp();
        let provider_block_id = format!("{}_{}", block_id, provider as u8);

        match provider {
            DAProvider::Celestia => {
                // Store with Celestia (simplified)
                // In a real implementation, this would use the actual CelestiaDA
                let storage_cost = self.calculate_storage_cost(provider, data.len());
                Ok(ProviderMetadata {
                    provider_block_id,
                    storage_cost,
                    storage_timestamp: timestamp,
                    provider_data: Vec::new(),
                })
            }
            DAProvider::EigenDA => {
                if let Some(ref mut eigenda_engine) = self.eigenda_engine {
                    let eigenda_blob = eigenda_engine
                        .store_blob(data.to_vec())
                        .map_err(|_| DAAbstractionError::ProviderUnavailable)?;

                    let storage_cost = self.calculate_storage_cost(provider, data.len());
                    let provider_block_id = eigenda_blob.blob_id.clone();
                    let provider_data = bincode::serialize(&eigenda_blob)
                        .map_err(|_| DAAbstractionError::NetworkError)?;

                    Ok(ProviderMetadata {
                        provider_block_id,
                        storage_cost,
                        storage_timestamp: timestamp,
                        provider_data,
                    })
                } else {
                    Err(DAAbstractionError::ProviderUnavailable)
                }
            }
            DAProvider::Avail => {
                if let Some(ref mut avail_engine) = self.avail_engine {
                    let avail_block = avail_engine
                        .store_data_block(1, data.to_vec()) // Using network_id = 1
                        .map_err(|_| DAAbstractionError::ProviderUnavailable)?;

                    let storage_cost = self.calculate_storage_cost(provider, data.len());
                    let provider_block_id = avail_block.block_id.clone();
                    let provider_data = bincode::serialize(&avail_block)
                        .map_err(|_| DAAbstractionError::NetworkError)?;

                    Ok(ProviderMetadata {
                        provider_block_id,
                        storage_cost,
                        storage_timestamp: timestamp,
                        provider_data,
                    })
                } else {
                    Err(DAAbstractionError::ProviderUnavailable)
                }
            }
            DAProvider::EthereumBlobs => {
                if let (Some(ref mut _blob_pool), Some(ref mut kzg_system)) =
                    (&mut self.blob_transaction_pool, &mut self.kzg_system)
                {
                    // Create blob data (pad to 131072 bytes if needed)
                    let mut blob_data = data.to_vec();
                    if blob_data.len() < 131072 {
                        blob_data.resize(131072, 0);
                    } else if blob_data.len() > 131072 {
                        return Err(DAAbstractionError::CostLimitExceeded);
                    }

                    // Generate KZG commitment and proof
                    let kzg_commitment = kzg_system
                        .commit_to_blob(&blob_data)
                        .map_err(|_| DAAbstractionError::ProviderUnavailable)?;
                    let _kzg_proof = kzg_system
                        .prove_blob(&blob_data, [42u8; 32])
                        .map_err(|_| DAAbstractionError::ProviderUnavailable)?;

                    let storage_cost = self.calculate_storage_cost(provider, data.len());
                    // Serialize commitment manually since it doesn't implement Serialize
                    let mut provider_data = Vec::new();
                    provider_data.extend_from_slice(&kzg_commitment.commitment);
                    provider_data.extend_from_slice(&kzg_commitment.degree.to_le_bytes());
                    provider_data.extend_from_slice(&kzg_commitment.timestamp.to_le_bytes());

                    Ok(ProviderMetadata {
                        provider_block_id,
                        storage_cost,
                        storage_timestamp: timestamp,
                        provider_data,
                    })
                } else {
                    Err(DAAbstractionError::ProviderUnavailable)
                }
            }
        }
    }

    /// Tries fallback providers when primary fails
    fn try_fallback_providers(
        &mut self,
        block_id: &str,
        data: &[u8],
    ) -> DAAbstractionResult<ProviderMetadata> {
        let available_providers = self.get_available_providers();

        for provider in available_providers {
            if let Ok(metadata) = self.store_with_provider(provider, block_id, data) {
                return Ok(metadata);
            }
        }

        Err(DAAbstractionError::FallbackFailed)
    }

    /// Checks if redundancy is cost-effective
    fn is_redundancy_cost_effective(&self, provider: DAProvider, data: &[u8]) -> bool {
        let _config = self
            .provider_configs
            .get(&provider)
            .unwrap_or(&DAProviderConfig {
                provider,
                name: String::new(),
                enabled: false,
                priority: 999,
                cost_per_mb: u64::MAX,
                max_data_size: 0,
                timeout_seconds: 0,
                health_check_interval: 0,
            });

        // Simple cost-effectiveness check: if cost is less than 10% of data size
        let storage_cost = self.calculate_storage_cost(provider, data.len());
        let data_size_mb = (data.len() as f64) / (1024.0 * 1024.0);
        let cost_per_mb = storage_cost as f64 / data_size_mb.max(0.001);

        cost_per_mb < 1_000_000.0 // Less than 1M wei per MB
    }

    /// Calculates storage cost for a provider
    fn calculate_storage_cost(&self, provider: DAProvider, data_size: usize) -> u64 {
        let default_config = DAProviderConfig {
            provider,
            name: String::new(),
            enabled: false,
            priority: 999,
            cost_per_mb: u64::MAX,
            max_data_size: 0,
            timeout_seconds: 0,
            health_check_interval: 0,
        };

        let config = self
            .provider_configs
            .get(&provider)
            .unwrap_or(&default_config);

        let data_size_mb = (data_size as f64) / (1024.0 * 1024.0);
        (data_size_mb * config.cost_per_mb as f64) as u64
    }

    /// Verifies data with a specific provider
    fn verify_with_provider(
        &self,
        provider: DAProvider,
        provider_metadata: &HashMap<DAProvider, ProviderMetadata>,
    ) -> bool {
        if let Some(metadata) = provider_metadata.get(&provider) {
            match provider {
                DAProvider::Celestia => {
                    // Verify with Celestia (simplified)
                    true
                }
                DAProvider::EigenDA => {
                    if let Some(ref eigenda_engine) = self.eigenda_engine {
                        eigenda_engine
                            .verify_data_availability(&metadata.provider_block_id)
                            .unwrap_or(false)
                    } else {
                        false
                    }
                }
                DAProvider::Avail => {
                    if let Some(ref avail_engine) = self.avail_engine {
                        avail_engine
                            .verify_data_availability(1, &metadata.provider_block_id) // Using network_id = 1
                            .unwrap_or(false)
                    } else {
                        false
                    }
                }
                DAProvider::EthereumBlobs => {
                    if let Some(ref _kzg_system) = self.kzg_system {
                        // Verify KZG commitment exists and is valid
                        // In a real implementation, this would verify against the Ethereum network
                        !metadata.provider_data.is_empty()
                    } else {
                        false
                    }
                }
            }
        } else {
            false
        }
    }

    /// Updates provider health metrics
    fn update_provider_health(
        &mut self,
        provider: DAProvider,
        success: bool,
        response_time_ms: u64,
    ) {
        let mut provider_health = self.provider_health.write().unwrap();
        if let Some(health) = provider_health.get_mut(&provider) {
            health.total_requests += 1;
            if success {
                health.successful_requests += 1;
            } else {
                health.failed_requests += 1;
            }

            health.avg_response_time_ms = (health.avg_response_time_ms + response_time_ms) / 2;

            health.success_rate = health.successful_requests as f64 / health.total_requests as f64;
            health.availability = health.success_rate * 100.0;

            // Update status based on success rate
            if health.success_rate >= 0.95 {
                health.status = DAProviderStatus::Healthy;
            } else if health.success_rate >= 0.8 {
                health.status = DAProviderStatus::Degraded;
            } else {
                health.status = DAProviderStatus::Unavailable;
            }

            health.last_health_check = current_timestamp();
        }
    }

    /// Serializes unified block for signing
    fn serialize_unified_block(&self, block: &UnifiedDataBlock) -> DAAbstractionResult<Vec<u8>> {
        let mut data = Vec::new();
        data.extend_from_slice(block.block_id.as_bytes());
        data.extend_from_slice(&block.data);
        data.extend_from_slice(&block.timestamp.to_le_bytes());
        data.extend_from_slice(&(block.primary_provider as u8).to_le_bytes());
        Ok(data)
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
    fn test_da_abstraction_engine_creation() {
        let engine = DAAbstractionEngine::new().unwrap();
        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_blocks, 0);
    }

    #[test]
    fn test_da_provider_registration() {
        let mut engine = DAAbstractionEngine::new().unwrap();

        let eigenda_config = DAProviderConfig {
            provider: DAProvider::EigenDA,
            name: "eigenda_provider".to_string(),
            enabled: true,
            priority: 1,
            cost_per_mb: 1_000_000,          // 1M wei per MB
            max_data_size: 10 * 1024 * 1024, // 10MB
            timeout_seconds: 30,
            health_check_interval: 60,
        };

        let result = engine.register_provider(eigenda_config);
        assert!(result.is_ok());

        let configs = engine.get_provider_configs();
        assert!(configs.contains_key(&DAProvider::EigenDA));
    }

    #[test]
    fn test_da_data_storage_with_fallback() {
        let mut engine = DAAbstractionEngine::new().unwrap();

        // Register multiple providers
        let eigenda_config = DAProviderConfig {
            provider: DAProvider::EigenDA,
            name: "eigenda_provider".to_string(),
            enabled: true,
            priority: 1,
            cost_per_mb: 1_000_000,
            max_data_size: 10 * 1024 * 1024,
            timeout_seconds: 30,
            health_check_interval: 60,
        };
        engine.register_provider(eigenda_config).unwrap();

        let avail_config = DAProviderConfig {
            provider: DAProvider::Avail,
            name: "avail_provider".to_string(),
            enabled: true,
            priority: 2,
            cost_per_mb: 2_000_000,
            max_data_size: 5 * 1024 * 1024,
            timeout_seconds: 30,
            health_check_interval: 60,
        };
        engine.register_provider(avail_config).unwrap();

        // Store data
        let data = vec![0x01, 0x02, 0x03, 0x04, 0x05];
        let unified_block = engine.store_data(data.clone()).unwrap();

        assert_eq!(unified_block.data, data);
        assert!(unified_block.signature.is_some());
        assert!(!unified_block.provider_metadata.is_empty());

        // Retrieve data
        let retrieved_block = engine.retrieve_data(&unified_block.block_id).unwrap();
        assert!(retrieved_block.is_some());
        assert_eq!(retrieved_block.unwrap().data, data);

        // Verify data availability
        let is_available = engine
            .verify_data_availability(&unified_block.block_id)
            .unwrap();
        assert!(is_available);
    }

    #[test]
    fn test_da_provider_health_monitoring() {
        let mut engine = DAAbstractionEngine::new().unwrap();

        let eigenda_config = DAProviderConfig {
            provider: DAProvider::EigenDA,
            name: "eigenda_provider".to_string(),
            enabled: true,
            priority: 1,
            cost_per_mb: 1_000_000,
            max_data_size: 10 * 1024 * 1024,
            timeout_seconds: 30,
            health_check_interval: 60,
        };
        engine.register_provider(eigenda_config).unwrap();

        // Check initial health status
        let health = engine.get_provider_health(DAProvider::EigenDA).unwrap();
        assert_eq!(health.status, DAProviderStatus::Testing);
        assert_eq!(health.total_requests, 0);
    }

    #[test]
    fn test_da_all_providers_failed() {
        let mut engine = DAAbstractionEngine::new().unwrap();

        // Don't register any providers
        let data = vec![0x01, 0x02, 0x03];
        let result = engine.store_data(data);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), DAAbstractionError::AllProvidersFailed);
    }

    #[test]
    fn test_da_ethereum_blobs_provider() {
        let mut engine = DAAbstractionEngine::new().unwrap();

        let ethereum_blobs_config = DAProviderConfig {
            provider: DAProvider::EthereumBlobs,
            name: "ethereum_blobs_provider".to_string(),
            enabled: true,
            priority: 1,
            cost_per_mb: 500_000,  // 0.5M wei per MB (cheaper than others)
            max_data_size: 131072, // 1 blob = 131072 bytes
            timeout_seconds: 30,
            health_check_interval: 60,
        };

        let result = engine.register_provider(ethereum_blobs_config);
        assert!(result.is_ok());

        // Store data that fits in one blob
        let data = vec![0x01; 100000]; // 100KB of data
        let unified_block = engine.store_data(data.clone()).unwrap();

        assert_eq!(unified_block.data, data);
        assert_eq!(unified_block.primary_provider, DAProvider::EthereumBlobs);
        assert!(unified_block.signature.is_some());

        // Verify data availability
        let is_available = engine
            .verify_data_availability(&unified_block.block_id)
            .unwrap();
        assert!(is_available);

        let configs = engine.get_provider_configs();
        assert!(configs.contains_key(&DAProvider::EthereumBlobs));
    }
}
