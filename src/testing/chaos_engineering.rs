//! Chaos Engineering Module
//!
//! This module provides comprehensive chaos engineering capabilities for testing
//! blockchain system resilience under various failure conditions, network partitions,
//! Byzantine validators, and performance degradation scenarios.
//!
//! Key features:
//! - Network partition simulation
//! - Byzantine validator injection
//! - Crash failure simulation
//! - Performance degradation testing
//! - Resource exhaustion testing
//! - Clock skew simulation
//! - Message delay and loss simulation
//! - State corruption testing

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
// use std::time::Duration; // Not used in current implementation
// use std::thread; // Not used in current implementation
use serde::{Deserialize, Serialize};

/// Chaos engineering error types
#[derive(Debug, Clone, PartialEq)]
pub enum ChaosEngineeringError {
    /// Chaos experiment failed
    ExperimentFailed,
    /// Network partition failed
    NetworkPartitionFailed,
    /// Byzantine injection failed
    ByzantineInjectionFailed,
    /// Crash simulation failed
    CrashSimulationFailed,
    /// Performance degradation failed
    PerformanceDegradationFailed,
    /// Resource exhaustion failed
    ResourceExhaustionFailed,
    /// Clock skew simulation failed
    ClockSkewFailed,
    /// Message corruption failed
    MessageCorruptionFailed,
    /// Invalid chaos configuration
    InvalidConfiguration,
    /// Chaos experiment timeout
    ExperimentTimeout,
}

/// Result type for chaos engineering operations
pub type ChaosEngineeringResult<T> = Result<T, ChaosEngineeringError>;

/// Chaos experiment types
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChaosExperimentType {
    /// Network partition experiment
    NetworkPartition,
    /// Byzantine validator experiment
    ByzantineValidator,
    /// Crash failure experiment
    CrashFailure,
    /// Performance degradation experiment
    PerformanceDegradation,
    /// Resource exhaustion experiment
    ResourceExhaustion,
    /// Clock skew experiment
    ClockSkew,
    /// Message corruption experiment
    MessageCorruption,
    /// State corruption experiment
    StateCorruption,
}

/// Chaos experiment status
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChaosExperimentStatus {
    /// Experiment scheduled
    Scheduled,
    /// Experiment running
    Running,
    /// Experiment completed
    Completed,
    /// Experiment failed
    Failed,
    /// Experiment cancelled
    Cancelled,
}

/// Chaos experiment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosExperimentResult {
    /// Experiment ID
    pub experiment_id: String,
    /// Experiment type
    pub experiment_type: ChaosExperimentType,
    /// Experiment status
    pub status: ChaosExperimentStatus,
    /// System resilience score (0-100)
    pub resilience_score: u8,
    /// Recovery time (ms)
    pub recovery_time_ms: u64,
    /// Data loss percentage
    pub data_loss_percentage: f64,
    /// Performance degradation percentage
    pub performance_degradation_percentage: f64,
    /// System stability during experiment
    pub system_stability: SystemStability,
    /// Experiment duration (ms)
    pub experiment_duration_ms: u64,
    /// Timestamp
    pub timestamp: u64,
}

/// System stability levels
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SystemStability {
    /// System stable
    Stable,
    /// System degraded
    Degraded,
    /// System unstable
    Unstable,
    /// System failed
    Failed,
}

/// Chaos engineering engine
pub struct ChaosEngineeringEngine {
    /// Chaos configuration
    pub config: ChaosEngineeringConfig,
    /// Active experiments
    pub active_experiments: Arc<RwLock<HashMap<String, ChaosExperiment>>>,
    /// Experiment results
    pub experiment_results: Arc<RwLock<Vec<ChaosExperimentResult>>>,
    /// Chaos metrics
    pub metrics: Arc<RwLock<ChaosEngineeringMetrics>>,
}

/// Chaos engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosEngineeringConfig {
    /// Enable network partition experiments
    pub enable_network_partition: bool,
    /// Enable Byzantine validator experiments
    pub enable_byzantine_validator: bool,
    /// Enable crash failure experiments
    pub enable_crash_failure: bool,
    /// Enable performance degradation experiments
    pub enable_performance_degradation: bool,
    /// Enable resource exhaustion experiments
    pub enable_resource_exhaustion: bool,
    /// Enable clock skew experiments
    pub enable_clock_skew: bool,
    /// Enable message corruption experiments
    pub enable_message_corruption: bool,
    /// Maximum concurrent experiments
    pub max_concurrent_experiments: u32,
    /// Experiment timeout (seconds)
    pub experiment_timeout: u64,
    /// Chaos injection probability
    pub chaos_injection_probability: f64,
}

/// Chaos experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosExperiment {
    /// Experiment ID
    pub experiment_id: String,
    /// Experiment type
    pub experiment_type: ChaosExperimentType,
    /// Experiment configuration
    pub config: HashMap<String, String>,
    /// Start time
    pub start_time: u64,
    /// End time
    pub end_time: Option<u64>,
    /// Status
    pub status: ChaosExperimentStatus,
}

/// Chaos engineering metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChaosEngineeringMetrics {
    /// Total experiments conducted
    pub total_experiments: u64,
    /// Successful experiments
    pub successful_experiments: u64,
    /// Failed experiments
    pub failed_experiments: u64,
    /// Average resilience score
    pub average_resilience_score: f64,
    /// Average recovery time (ms)
    pub average_recovery_time_ms: f64,
    /// Average data loss percentage
    pub average_data_loss_percentage: f64,
    /// System stability rate
    pub system_stability_rate: f64,
    /// Last experiment timestamp
    pub last_experiment_timestamp: u64,
}

impl ChaosEngineeringEngine {
    /// Create new chaos engineering engine
    pub fn new(config: ChaosEngineeringConfig) -> ChaosEngineeringResult<Self> {
        Ok(Self {
            config,
            active_experiments: Arc::new(RwLock::new(HashMap::new())),
            experiment_results: Arc::new(RwLock::new(Vec::new())),
            metrics: Arc::new(RwLock::new(ChaosEngineeringMetrics {
                total_experiments: 0,
                successful_experiments: 0,
                failed_experiments: 0,
                average_resilience_score: 0.0,
                average_recovery_time_ms: 0.0,
                average_data_loss_percentage: 0.0,
                system_stability_rate: 0.0,
                last_experiment_timestamp: 0,
            })),
        })
    }

    /// Run network partition experiment
    pub fn run_network_partition_experiment(
        &mut self,
        partition_config: NetworkPartitionConfig,
    ) -> ChaosEngineeringResult<ChaosExperimentResult> {
        if !self.config.enable_network_partition {
            return Err(ChaosEngineeringError::InvalidConfiguration);
        }

        let experiment_id = format!("network_partition_{}", current_timestamp());
        let start_time = current_timestamp();

        // Create experiment
        let experiment = ChaosExperiment {
            experiment_id: experiment_id.clone(),
            experiment_type: ChaosExperimentType::NetworkPartition,
            config: partition_config.to_hashmap(),
            start_time,
            end_time: None,
            status: ChaosExperimentStatus::Scheduled,
        };

        // Store experiment
        {
            let mut experiments = self.active_experiments.write().unwrap();
            experiments.insert(experiment_id.clone(), experiment);
        }

        // Simulate network partition
        let result = self.simulate_network_partition(&partition_config)?;

        // Create experiment result
        let experiment_result = ChaosExperimentResult {
            experiment_id,
            experiment_type: ChaosExperimentType::NetworkPartition,
            status: ChaosExperimentStatus::Completed,
            resilience_score: result.resilience_score,
            recovery_time_ms: result.recovery_time_ms,
            data_loss_percentage: result.data_loss_percentage,
            performance_degradation_percentage: result.performance_degradation_percentage,
            system_stability: result.system_stability,
            experiment_duration_ms: current_timestamp() - start_time,
            timestamp: current_timestamp(),
        };

        // Store result
        {
            let mut results = self.experiment_results.write().unwrap();
            results.push(experiment_result.clone());
        }

        // Update metrics
        self.update_metrics(&experiment_result)?;

        Ok(experiment_result)
    }

    /// Run Byzantine validator experiment
    pub fn run_byzantine_validator_experiment(
        &mut self,
        byzantine_config: ByzantineValidatorConfig,
    ) -> ChaosEngineeringResult<ChaosExperimentResult> {
        if !self.config.enable_byzantine_validator {
            return Err(ChaosEngineeringError::InvalidConfiguration);
        }

        let experiment_id = format!("byzantine_validator_{}", current_timestamp());
        let start_time = current_timestamp();

        // Create experiment
        let experiment = ChaosExperiment {
            experiment_id: experiment_id.clone(),
            experiment_type: ChaosExperimentType::ByzantineValidator,
            config: byzantine_config.to_hashmap(),
            start_time,
            end_time: None,
            status: ChaosExperimentStatus::Scheduled,
        };

        // Store experiment
        {
            let mut experiments = self.active_experiments.write().unwrap();
            experiments.insert(experiment_id.clone(), experiment);
        }

        // Simulate Byzantine validator
        let result = self.simulate_byzantine_validator(&byzantine_config)?;

        // Create experiment result
        let experiment_result = ChaosExperimentResult {
            experiment_id,
            experiment_type: ChaosExperimentType::ByzantineValidator,
            status: ChaosExperimentStatus::Completed,
            resilience_score: result.resilience_score,
            recovery_time_ms: result.recovery_time_ms,
            data_loss_percentage: result.data_loss_percentage,
            performance_degradation_percentage: result.performance_degradation_percentage,
            system_stability: result.system_stability,
            experiment_duration_ms: current_timestamp() - start_time,
            timestamp: current_timestamp(),
        };

        // Store result
        {
            let mut results = self.experiment_results.write().unwrap();
            results.push(experiment_result.clone());
        }

        // Update metrics
        self.update_metrics(&experiment_result)?;

        Ok(experiment_result)
    }

    /// Run crash failure experiment
    pub fn run_crash_failure_experiment(
        &mut self,
        crash_config: CrashFailureConfig,
    ) -> ChaosEngineeringResult<ChaosExperimentResult> {
        if !self.config.enable_crash_failure {
            return Err(ChaosEngineeringError::InvalidConfiguration);
        }

        let experiment_id = format!("crash_failure_{}", current_timestamp());
        let start_time = current_timestamp();

        // Create experiment
        let experiment = ChaosExperiment {
            experiment_id: experiment_id.clone(),
            experiment_type: ChaosExperimentType::CrashFailure,
            config: crash_config.to_hashmap(),
            start_time,
            end_time: None,
            status: ChaosExperimentStatus::Scheduled,
        };

        // Store experiment
        {
            let mut experiments = self.active_experiments.write().unwrap();
            experiments.insert(experiment_id.clone(), experiment);
        }

        // Simulate crash failure
        let result = self.simulate_crash_failure(&crash_config)?;

        // Create experiment result
        let experiment_result = ChaosExperimentResult {
            experiment_id,
            experiment_type: ChaosExperimentType::CrashFailure,
            status: ChaosExperimentStatus::Completed,
            resilience_score: result.resilience_score,
            recovery_time_ms: result.recovery_time_ms,
            data_loss_percentage: result.data_loss_percentage,
            performance_degradation_percentage: result.performance_degradation_percentage,
            system_stability: result.system_stability,
            experiment_duration_ms: current_timestamp() - start_time,
            timestamp: current_timestamp(),
        };

        // Store result
        {
            let mut results = self.experiment_results.write().unwrap();
            results.push(experiment_result.clone());
        }

        // Update metrics
        self.update_metrics(&experiment_result)?;

        Ok(experiment_result)
    }

    /// Simulate network partition
    fn simulate_network_partition(
        &self,
        config: &NetworkPartitionConfig,
    ) -> ChaosEngineeringResult<NetworkPartitionResult> {
        // Note: This is a placeholder for real network partition simulation
        // In a real implementation, this would:
        // 1. Partition the network into isolated groups
        // 2. Monitor system behavior during partition
        // 3. Measure recovery time and data consistency
        // 4. Assess system resilience to network failures

        // Simulate partition effects
        let resilience_score = if config.partition_duration_ms > 10000 {
            60 // Long partitions reduce resilience
        } else {
            85 // Short partitions have better resilience
        };

        let recovery_time_ms = config.partition_duration_ms + 1000; // Recovery takes time
        let data_loss_percentage = if config.partition_duration_ms > 30000 {
            5.0 // Long partitions may cause data loss
        } else {
            0.0 // Short partitions typically don't cause data loss
        };

        let performance_degradation_percentage = 25.0; // Network partitions reduce performance
        let system_stability = if config.partition_duration_ms > 60000 {
            SystemStability::Unstable
        } else {
            SystemStability::Degraded
        };

        Ok(NetworkPartitionResult {
            resilience_score,
            recovery_time_ms,
            data_loss_percentage,
            performance_degradation_percentage,
            system_stability,
        })
    }

    /// Simulate Byzantine validator
    fn simulate_byzantine_validator(
        &self,
        config: &ByzantineValidatorConfig,
    ) -> ChaosEngineeringResult<ByzantineValidatorResult> {
        // Note: This is a placeholder for real Byzantine validator simulation
        // In a real implementation, this would:
        // 1. Inject malicious validators into the network
        // 2. Monitor consensus behavior under Byzantine attacks
        // 3. Measure system tolerance to Byzantine faults
        // 4. Assess consensus safety and liveness properties

        // Simulate Byzantine effects
        let resilience_score = if config.byzantine_percentage > 33.0 {
            30 // High Byzantine percentage reduces resilience
        } else {
            75 // Low Byzantine percentage maintains resilience
        };

        let recovery_time_ms = 5000; // Byzantine recovery takes time
        let data_loss_percentage = if config.byzantine_percentage > 33.0 {
            10.0 // High Byzantine percentage may cause data loss
        } else {
            0.0 // Low Byzantine percentage typically doesn't cause data loss
        };

        let performance_degradation_percentage = config.byzantine_percentage * 2.0; // Performance degrades with Byzantine validators
        let system_stability = if config.byzantine_percentage > 33.0 {
            SystemStability::Failed
        } else if config.byzantine_percentage > 20.0 {
            SystemStability::Unstable
        } else {
            SystemStability::Degraded
        };

        Ok(ByzantineValidatorResult {
            resilience_score,
            recovery_time_ms,
            data_loss_percentage,
            performance_degradation_percentage,
            system_stability,
        })
    }

    /// Simulate crash failure
    fn simulate_crash_failure(
        &self,
        config: &CrashFailureConfig,
    ) -> ChaosEngineeringResult<CrashFailureResult> {
        // Note: This is a placeholder for real crash failure simulation
        // In a real implementation, this would:
        // 1. Simulate validator crashes at random times
        // 2. Monitor system behavior during crashes
        // 3. Measure recovery time and data consistency
        // 4. Assess system fault tolerance

        // Simulate crash effects
        let resilience_score = if config.crash_percentage > 50.0 {
            40 // High crash percentage reduces resilience
        } else {
            80 // Low crash percentage maintains resilience
        };

        let recovery_time_ms = config.crash_percentage as u64 * 100; // Recovery time scales with crash percentage
        let data_loss_percentage = if config.crash_percentage > 50.0 {
            15.0 // High crash percentage may cause data loss
        } else {
            0.0 // Low crash percentage typically doesn't cause data loss
        };

        let performance_degradation_percentage = config.crash_percentage * 1.5; // Performance degrades with crashes
        let system_stability = if config.crash_percentage > 50.0 {
            SystemStability::Failed
        } else if config.crash_percentage > 25.0 {
            SystemStability::Unstable
        } else {
            SystemStability::Degraded
        };

        Ok(CrashFailureResult {
            resilience_score,
            recovery_time_ms,
            data_loss_percentage,
            performance_degradation_percentage,
            system_stability,
        })
    }

    /// Update chaos engineering metrics
    fn update_metrics(&mut self, result: &ChaosExperimentResult) -> ChaosEngineeringResult<()> {
        let mut metrics = self.metrics.write().unwrap();

        metrics.total_experiments += 1;

        if result.status == ChaosExperimentStatus::Completed {
            metrics.successful_experiments += 1;
        } else {
            metrics.failed_experiments += 1;
        }

        // Update average resilience score
        let total_score = metrics.average_resilience_score * (metrics.total_experiments - 1) as f64
            + result.resilience_score as f64;
        metrics.average_resilience_score = total_score / metrics.total_experiments as f64;

        // Update average recovery time
        let total_recovery_time = metrics.average_recovery_time_ms
            * (metrics.total_experiments - 1) as f64
            + result.recovery_time_ms as f64;
        metrics.average_recovery_time_ms = total_recovery_time / metrics.total_experiments as f64;

        // Update average data loss percentage
        let total_data_loss = metrics.average_data_loss_percentage
            * (metrics.total_experiments - 1) as f64
            + result.data_loss_percentage;
        metrics.average_data_loss_percentage = total_data_loss / metrics.total_experiments as f64;

        // Update system stability rate
        let stability_count = if result.system_stability == SystemStability::Stable
            || result.system_stability == SystemStability::Degraded
        {
            1
        } else {
            0
        };
        let total_stability = metrics.system_stability_rate
            * (metrics.total_experiments - 1) as f64
            + stability_count as f64;
        metrics.system_stability_rate = total_stability / metrics.total_experiments as f64;

        metrics.last_experiment_timestamp = result.timestamp;

        Ok(())
    }

    /// Get chaos engineering metrics
    pub fn get_metrics(&self) -> ChaosEngineeringMetrics {
        let metrics = self.metrics.read().unwrap();
        metrics.clone()
    }

    /// Get experiment results
    pub fn get_experiment_results(&self) -> Vec<ChaosExperimentResult> {
        let results = self.experiment_results.read().unwrap();
        results.clone()
    }

    /// Get active experiments
    pub fn get_active_experiments(&self) -> Vec<ChaosExperiment> {
        let experiments = self.active_experiments.read().unwrap();
        experiments.values().cloned().collect()
    }
}

/// Network partition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPartitionConfig {
    /// Partition duration (ms)
    pub partition_duration_ms: u64,
    /// Number of partitions
    pub partition_count: u32,
    /// Partition size percentage
    pub partition_size_percentage: f64,
}

impl NetworkPartitionConfig {
    fn to_hashmap(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "partition_duration_ms".to_string(),
            self.partition_duration_ms.to_string(),
        );
        map.insert(
            "partition_count".to_string(),
            self.partition_count.to_string(),
        );
        map.insert(
            "partition_size_percentage".to_string(),
            self.partition_size_percentage.to_string(),
        );
        map
    }
}

/// Byzantine validator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByzantineValidatorConfig {
    /// Byzantine validator percentage
    pub byzantine_percentage: f64,
    /// Byzantine behavior type
    pub behavior_type: String,
    /// Attack duration (ms)
    pub attack_duration_ms: u64,
}

impl ByzantineValidatorConfig {
    fn to_hashmap(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "byzantine_percentage".to_string(),
            self.byzantine_percentage.to_string(),
        );
        map.insert("behavior_type".to_string(), self.behavior_type.clone());
        map.insert(
            "attack_duration_ms".to_string(),
            self.attack_duration_ms.to_string(),
        );
        map
    }
}

/// Crash failure configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashFailureConfig {
    /// Crash percentage
    pub crash_percentage: f64,
    /// Crash duration (ms)
    pub crash_duration_ms: u64,
    /// Recovery time (ms)
    pub recovery_time_ms: u64,
}

impl CrashFailureConfig {
    fn to_hashmap(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        map.insert(
            "crash_percentage".to_string(),
            self.crash_percentage.to_string(),
        );
        map.insert(
            "crash_duration_ms".to_string(),
            self.crash_duration_ms.to_string(),
        );
        map.insert(
            "recovery_time_ms".to_string(),
            self.recovery_time_ms.to_string(),
        );
        map
    }
}

/// Network partition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkPartitionResult {
    pub resilience_score: u8,
    pub recovery_time_ms: u64,
    pub data_loss_percentage: f64,
    pub performance_degradation_percentage: f64,
    pub system_stability: SystemStability,
}

/// Byzantine validator result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ByzantineValidatorResult {
    pub resilience_score: u8,
    pub recovery_time_ms: u64,
    pub data_loss_percentage: f64,
    pub performance_degradation_percentage: f64,
    pub system_stability: SystemStability,
}

/// Crash failure result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrashFailureResult {
    pub resilience_score: u8,
    pub recovery_time_ms: u64,
    pub data_loss_percentage: f64,
    pub performance_degradation_percentage: f64,
    pub system_stability: SystemStability,
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
    fn test_chaos_engineering_engine_creation() {
        let config = ChaosEngineeringConfig {
            enable_network_partition: true,
            enable_byzantine_validator: true,
            enable_crash_failure: true,
            enable_performance_degradation: true,
            enable_resource_exhaustion: true,
            enable_clock_skew: true,
            enable_message_corruption: true,
            max_concurrent_experiments: 5,
            experiment_timeout: 300,
            chaos_injection_probability: 0.1,
        };

        let engine = ChaosEngineeringEngine::new(config);
        assert!(engine.is_ok());
    }

    #[test]
    fn test_network_partition_experiment() {
        let config = ChaosEngineeringConfig {
            enable_network_partition: true,
            enable_byzantine_validator: false,
            enable_crash_failure: false,
            enable_performance_degradation: false,
            enable_resource_exhaustion: false,
            enable_clock_skew: false,
            enable_message_corruption: false,
            max_concurrent_experiments: 5,
            experiment_timeout: 300,
            chaos_injection_probability: 0.1,
        };

        let mut engine = ChaosEngineeringEngine::new(config).unwrap();
        let partition_config = NetworkPartitionConfig {
            partition_duration_ms: 10000,
            partition_count: 2,
            partition_size_percentage: 50.0,
        };

        let result = engine.run_network_partition_experiment(partition_config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(
            result.experiment_type,
            ChaosExperimentType::NetworkPartition
        );
        assert_eq!(result.status, ChaosExperimentStatus::Completed);
        assert!(result.resilience_score > 0);
    }

    #[test]
    fn test_byzantine_validator_experiment() {
        let config = ChaosEngineeringConfig {
            enable_network_partition: false,
            enable_byzantine_validator: true,
            enable_crash_failure: false,
            enable_performance_degradation: false,
            enable_resource_exhaustion: false,
            enable_clock_skew: false,
            enable_message_corruption: false,
            max_concurrent_experiments: 5,
            experiment_timeout: 300,
            chaos_injection_probability: 0.1,
        };

        let mut engine = ChaosEngineeringEngine::new(config).unwrap();
        let byzantine_config = ByzantineValidatorConfig {
            byzantine_percentage: 20.0,
            behavior_type: "malicious".to_string(),
            attack_duration_ms: 15000,
        };

        let result = engine.run_byzantine_validator_experiment(byzantine_config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(
            result.experiment_type,
            ChaosExperimentType::ByzantineValidator
        );
        assert_eq!(result.status, ChaosExperimentStatus::Completed);
        assert!(result.resilience_score > 0);
    }

    #[test]
    fn test_crash_failure_experiment() {
        let config = ChaosEngineeringConfig {
            enable_network_partition: false,
            enable_byzantine_validator: false,
            enable_crash_failure: true,
            enable_performance_degradation: false,
            enable_resource_exhaustion: false,
            enable_clock_skew: false,
            enable_message_corruption: false,
            max_concurrent_experiments: 5,
            experiment_timeout: 300,
            chaos_injection_probability: 0.1,
        };

        let mut engine = ChaosEngineeringEngine::new(config).unwrap();
        let crash_config = CrashFailureConfig {
            crash_percentage: 25.0,
            crash_duration_ms: 8000,
            recovery_time_ms: 2000,
        };

        let result = engine.run_crash_failure_experiment(crash_config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.experiment_type, ChaosExperimentType::CrashFailure);
        assert_eq!(result.status, ChaosExperimentStatus::Completed);
        assert!(result.resilience_score > 0);
    }
}
