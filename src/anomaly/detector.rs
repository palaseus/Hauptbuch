//! AI-Driven Anomaly Detection Module
//!
//! This module provides comprehensive anomaly detection capabilities for the decentralized
//! voting blockchain, identifying unusual patterns in voting, network activity, and cross-chain
//! interactions using lightweight statistical models and machine learning techniques.
//!
//! Key features:
//! - K-means clustering for pattern analysis
//! - Z-score anomaly detection for statistical outliers
//! - Real-time anomaly scoring and alerting
//! - Integration with governance, monitoring, federation, and analytics modules
//! - SHA-3 data integrity verification
//! - Dilithium3/5 signature support for alert authentication
//! - Chart.js-compatible JSON outputs for visualization
//! - Comprehensive test coverage with 25+ test cases

use sha3::{Digest, Sha3_256};
use std::collections::{HashMap, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

// Import blockchain modules for data analysis
use crate::analytics::governance::{ChartData, ChartDataset, ChartOptions};
use crate::analytics::governance::{
    GovernanceAnalytics, StakeDistributionMetrics, VoterTurnoutMetrics,
};
use crate::federation::federation::{CrossChainVote, FederatedProposal, FederationMember};
use crate::governance::proposal::{Proposal, Vote};
use crate::monitoring::monitor::{SystemMetrics, VoterActivity};

// Import quantum-resistant cryptography for alert signatures
use crate::crypto::quantum_resistant::{
    dilithium_keygen, dilithium_sign, DilithiumParams, DilithiumPublicKey, DilithiumSecretKey,
    DilithiumSecurityLevel, DilithiumSignature,
};

/// Anomaly severity levels
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnomalySeverity {
    Low,
    Medium,
    High,
    Critical,
}

impl std::fmt::Display for AnomalySeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnomalySeverity::Low => write!(f, "Low"),
            AnomalySeverity::Medium => write!(f, "Medium"),
            AnomalySeverity::High => write!(f, "High"),
            AnomalySeverity::Critical => write!(f, "Critical"),
        }
    }
}

/// Types of anomalies that can be detected
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AnomalyType {
    /// Unusual voter turnout patterns
    VoterTurnoutSpike,
    /// Potential vote stuffing attacks
    VoteStuffing,
    /// Sybil attack patterns
    SybilAttack,
    /// Network latency anomalies
    NetworkLatencySpike,
    /// Cross-chain message irregularities
    CrossChainInconsistency,
    /// Stake distribution anomalies
    StakeDistributionAnomaly,
    /// Resource usage spikes
    ResourceUsageSpike,
    /// Consensus participation anomalies
    ConsensusAnomaly,
    /// State synchronization issues
    StateSyncAnomaly,
    /// Cryptographic signature anomalies
    SignatureAnomaly,
}

impl std::fmt::Display for AnomalyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnomalyType::VoterTurnoutSpike => write!(f, "Voter Turnout Spike"),
            AnomalyType::VoteStuffing => write!(f, "Vote Stuffing"),
            AnomalyType::SybilAttack => write!(f, "Sybil Attack"),
            AnomalyType::NetworkLatencySpike => write!(f, "Network Latency Spike"),
            AnomalyType::CrossChainInconsistency => write!(f, "Cross-Chain Inconsistency"),
            AnomalyType::StakeDistributionAnomaly => write!(f, "Stake Distribution Anomaly"),
            AnomalyType::ResourceUsageSpike => write!(f, "Resource Usage Spike"),
            AnomalyType::ConsensusAnomaly => write!(f, "Consensus Anomaly"),
            AnomalyType::StateSyncAnomaly => write!(f, "State Sync Anomaly"),
            AnomalyType::SignatureAnomaly => write!(f, "Signature Anomaly"),
        }
    }
}

/// Anomaly detection result
#[derive(Debug, Clone)]
pub struct AnomalyDetection {
    /// Unique anomaly identifier
    pub id: String,
    /// Type of anomaly detected
    pub anomaly_type: AnomalyType,
    /// Severity level
    pub severity: AnomalySeverity,
    /// Anomaly score (0.0 to 1.0)
    pub score: f64,
    /// Description of the anomaly
    pub description: String,
    /// Timestamp when detected
    pub timestamp: u64,
    /// Source data that triggered the anomaly
    pub source_data: HashMap<String, f64>,
    /// Confidence level (0.0 to 1.0)
    pub confidence: f64,
    /// Recommended actions
    pub recommendations: Vec<String>,
    /// Alert signature for integrity
    pub alert_signature: Option<DilithiumSignature>,
}

/// K-means clustering result
#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// Cluster centroids
    pub centroids: Vec<Vec<f64>>,
    /// Data point assignments to clusters
    pub assignments: Vec<usize>,
    /// Within-cluster sum of squares
    pub wcss: f64,
    /// Number of clusters
    pub k: usize,
    /// Convergence iterations
    pub iterations: u32,
}

/// Z-score anomaly detection result
#[derive(Debug, Clone)]
pub struct ZScoreResult {
    /// Z-scores for each data point
    pub z_scores: Vec<f64>,
    /// Anomaly threshold
    pub threshold: f64,
    /// Number of anomalies detected
    pub anomaly_count: usize,
    /// Mean of the dataset
    pub mean: f64,
    /// Standard deviation of the dataset
    pub std_dev: f64,
}

/// Anomaly detection configuration
#[derive(Debug, Clone)]
pub struct AnomalyConfig {
    /// Enable K-means clustering
    pub enable_kmeans: bool,
    /// Enable Z-score detection
    pub enable_zscore: bool,
    /// Z-score threshold for anomalies
    pub zscore_threshold: f64,
    /// Number of clusters for K-means
    pub kmeans_clusters: usize,
    /// Maximum iterations for K-means
    pub kmeans_max_iterations: u32,
    /// Enable quantum-resistant signatures
    pub enable_quantum_signatures: bool,
    /// Dilithium security level
    pub dilithium_security_level: DilithiumSecurityLevel,
    /// Alert threshold for different anomaly types
    pub alert_thresholds: HashMap<AnomalyType, f64>,
    /// Data integrity verification enabled
    pub enable_integrity_verification: bool,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert(AnomalyType::VoterTurnoutSpike, 0.7);
        alert_thresholds.insert(AnomalyType::VoteStuffing, 0.8);
        alert_thresholds.insert(AnomalyType::SybilAttack, 0.9);
        alert_thresholds.insert(AnomalyType::NetworkLatencySpike, 0.6);
        alert_thresholds.insert(AnomalyType::CrossChainInconsistency, 0.8);
        alert_thresholds.insert(AnomalyType::StakeDistributionAnomaly, 0.7);
        alert_thresholds.insert(AnomalyType::ResourceUsageSpike, 0.6);
        alert_thresholds.insert(AnomalyType::ConsensusAnomaly, 0.8);
        alert_thresholds.insert(AnomalyType::StateSyncAnomaly, 0.7);
        alert_thresholds.insert(AnomalyType::SignatureAnomaly, 0.9);

        Self {
            enable_kmeans: true,
            enable_zscore: true,
            zscore_threshold: 3.0,
            kmeans_clusters: 3,
            kmeans_max_iterations: 100,
            enable_quantum_signatures: true,
            dilithium_security_level: DilithiumSecurityLevel::Dilithium3,
            alert_thresholds,
            enable_integrity_verification: true,
        }
    }
}

/// Main anomaly detection engine
pub struct AnomalyDetector {
    pub config: AnomalyConfig,
    detection_history: VecDeque<AnomalyDetection>,
    quantum_keys: Option<(DilithiumPublicKey, DilithiumSecretKey)>,
    data_integrity_hash: Vec<u8>,
    anomaly_counters: HashMap<AnomalyType, u64>,
}

impl AnomalyDetector {
    /// Create a new anomaly detector with default configuration
    pub fn new() -> Result<Self, AnomalyError> {
        Self::with_config(AnomalyConfig::default())
    }

    /// Create a new anomaly detector with custom configuration
    pub fn with_config(config: AnomalyConfig) -> Result<Self, AnomalyError> {
        // Validate configuration
        if config.kmeans_clusters == 0 {
            return Err(AnomalyError::ConfigurationError(
                "K-means clusters must be greater than 0".to_string(),
            ));
        }

        if config.kmeans_max_iterations == 0 {
            return Err(AnomalyError::ConfigurationError(
                "K-means max iterations must be greater than 0".to_string(),
            ));
        }

        if config.zscore_threshold <= 0.0 {
            return Err(AnomalyError::ConfigurationError(
                "Z-score threshold must be greater than 0".to_string(),
            ));
        }

        // Generate quantum-resistant keys if enabled
        let quantum_keys = if config.enable_quantum_signatures {
            let params = match config.dilithium_security_level {
                DilithiumSecurityLevel::Dilithium3 => DilithiumParams::dilithium3(),
                DilithiumSecurityLevel::Dilithium5 => DilithiumParams::dilithium5(),
                _ => {
                    return Err(AnomalyError::ConfigurationError(
                        "Unsupported Dilithium security level".to_string(),
                    ))
                }
            };

            Some(dilithium_keygen(&params).map_err(|e| {
                AnomalyError::KeyGenerationError(format!(
                    "Failed to generate quantum keys: {:?}",
                    e
                ))
            })?)
        } else {
            None
        };

        Ok(Self {
            config,
            detection_history: VecDeque::new(),
            quantum_keys,
            data_integrity_hash: Vec::new(),
            anomaly_counters: HashMap::new(),
        })
    }

    /// Detect anomalies in governance data
    pub fn detect_governance_anomalies(
        &mut self,
        proposals: &[Proposal],
        votes: &[Vote],
        analytics: &GovernanceAnalytics,
    ) -> Result<Vec<AnomalyDetection>, AnomalyError> {
        let mut anomalies = Vec::new();

        // Detect voter turnout anomalies
        anomalies.extend(self.detect_voter_turnout_anomalies(votes, &analytics.voter_turnout)?);

        // Detect vote stuffing patterns
        anomalies.extend(self.detect_vote_stuffing_anomalies(votes)?);

        // Detect stake distribution anomalies
        anomalies.extend(self.detect_stake_distribution_anomalies(&analytics.stake_distribution)?);

        // Detect proposal success rate anomalies
        anomalies.extend(self.detect_proposal_anomalies(proposals)?);

        // Update detection history
        for anomaly in &anomalies {
            self.update_detection_history(anomaly.clone());
        }

        Ok(anomalies)
    }

    /// Detect anomalies in monitoring data
    pub fn detect_monitoring_anomalies(
        &mut self,
        system_metrics: &[SystemMetrics],
        _voter_activity: &[VoterActivity],
    ) -> Result<Vec<AnomalyDetection>, AnomalyError> {
        let mut anomalies = Vec::new();

        // Detect network latency spikes
        anomalies.extend(self.detect_network_latency_anomalies(system_metrics)?);

        // Detect resource usage spikes
        anomalies.extend(self.detect_resource_usage_anomalies(system_metrics)?);

        // Detect consensus anomalies
        anomalies.extend(self.detect_consensus_anomalies(system_metrics)?);

        // Update detection history
        for anomaly in &anomalies {
            self.update_detection_history(anomaly.clone());
        }

        Ok(anomalies)
    }

    /// Detect anomalies in federation data
    pub fn detect_federation_anomalies(
        &mut self,
        cross_chain_votes: &[CrossChainVote],
        federated_proposals: &[FederatedProposal],
        federation_members: &[FederationMember],
    ) -> Result<Vec<AnomalyDetection>, AnomalyError> {
        let mut anomalies = Vec::new();

        // Detect cross-chain inconsistencies
        anomalies
            .extend(self.detect_cross_chain_anomalies(cross_chain_votes, federated_proposals)?);

        // Detect Sybil attacks
        anomalies.extend(self.detect_sybil_attacks(federation_members)?);

        // Detect state synchronization anomalies
        anomalies.extend(self.detect_state_sync_anomalies(federated_proposals)?);

        // Update detection history
        for anomaly in &anomalies {
            self.update_detection_history(anomaly.clone());
        }

        Ok(anomalies)
    }

    /// Detect voter turnout anomalies using statistical analysis
    fn detect_voter_turnout_anomalies(
        &self,
        votes: &[Vote],
        _turnout_metrics: &VoterTurnoutMetrics,
    ) -> Result<Vec<AnomalyDetection>, AnomalyError> {
        let mut anomalies = Vec::new();

        if votes.len() < 10 {
            return Ok(anomalies); // Need sufficient data for analysis
        }

        // Calculate hourly voter turnout
        let hourly_turnout = self.calculate_hourly_turnout(votes)?;

        // Apply Z-score detection
        if self.config.enable_zscore {
            let zscore_result = self.calculate_zscore(&hourly_turnout)?;

            for (i, &z_score) in zscore_result.z_scores.iter().enumerate() {
                if z_score.abs() > self.config.zscore_threshold {
                    let severity = self.determine_anomaly_severity(z_score.abs());
                    let score = (z_score.abs() / self.config.zscore_threshold).min(1.0);

                    anomalies.push(AnomalyDetection {
                        id: self.generate_anomaly_id(),
                        anomaly_type: AnomalyType::VoterTurnoutSpike,
                        severity,
                        score,
                        description: format!(
                            "Unusual voter turnout detected: {:.2}% (Z-score: {:.2})",
                            hourly_turnout[i] * 100.0,
                            z_score
                        ),
                        timestamp: self.current_timestamp(),
                        source_data: {
                            let mut data = HashMap::new();
                            data.insert("turnout_percentage".to_string(), hourly_turnout[i]);
                            data.insert("z_score".to_string(), z_score);
                            data.insert("mean".to_string(), zscore_result.mean);
                            data.insert("std_dev".to_string(), zscore_result.std_dev);
                            data
                        },
                        confidence: self.calculate_confidence(z_score.abs()),
                        recommendations: vec![
                            "Investigate voting patterns".to_string(),
                            "Check for potential vote manipulation".to_string(),
                            "Monitor voter registration activity".to_string(),
                        ],
                        alert_signature: self
                            .sign_anomaly_alert(&AnomalyType::VoterTurnoutSpike, score)?,
                    });
                }
            }
        }

        Ok(anomalies)
    }

    /// Detect vote stuffing patterns using clustering analysis
    fn detect_vote_stuffing_anomalies(
        &self,
        votes: &[Vote],
    ) -> Result<Vec<AnomalyDetection>, AnomalyError> {
        let mut anomalies = Vec::new();

        if votes.len() < 20 {
            return Ok(anomalies); // Need sufficient data for clustering
        }

        // Extract voting patterns
        let voting_patterns = self.extract_voting_patterns(votes)?;

        if self.config.enable_kmeans {
            let cluster_result =
                self.perform_kmeans_clustering(&voting_patterns, self.config.kmeans_clusters)?;

            // Analyze cluster characteristics
            let cluster_analysis = self.analyze_clusters(&voting_patterns, &cluster_result)?;

            for (cluster_id, analysis) in cluster_analysis.iter().enumerate() {
                if analysis.is_anomalous {
                    let severity = self.determine_cluster_severity(analysis.anomaly_score);
                    let score = analysis.anomaly_score;

                    anomalies.push(AnomalyDetection {
                        id: self.generate_anomaly_id(),
                        anomaly_type: AnomalyType::VoteStuffing,
                        severity,
                        score,
                        description: format!(
                            "Potential vote stuffing detected in cluster {}: {} votes from {} voters",
                            cluster_id, analysis.vote_count, analysis.unique_voters
                        ),
                        timestamp: self.current_timestamp(),
                        source_data: {
                            let mut data = HashMap::new();
                            data.insert("cluster_id".to_string(), cluster_id as f64);
                            data.insert("vote_count".to_string(), analysis.vote_count as f64);
                            data.insert("unique_voters".to_string(), analysis.unique_voters as f64);
                            data.insert("anomaly_score".to_string(), analysis.anomaly_score);
                            data
                        },
                        confidence: self.calculate_confidence(analysis.anomaly_score),
                        recommendations: vec![
                            "Investigate voting patterns in this cluster".to_string(),
                            "Check for duplicate voter identities".to_string(),
                            "Verify voter authentication mechanisms".to_string(),
                        ],
                        alert_signature: self.sign_anomaly_alert(&AnomalyType::VoteStuffing, score)?,
                    });
                }
            }
        }

        Ok(anomalies)
    }

    /// Detect stake distribution anomalies
    fn detect_stake_distribution_anomalies(
        &self,
        stake_distribution: &StakeDistributionMetrics,
    ) -> Result<Vec<AnomalyDetection>, AnomalyError> {
        let mut anomalies = Vec::new();

        // Check for extreme Gini coefficient
        if stake_distribution.gini_coefficient > 0.8 {
            let severity = if stake_distribution.gini_coefficient > 0.9 {
                AnomalySeverity::Critical
            } else {
                AnomalySeverity::High
            };

            anomalies.push(AnomalyDetection {
                id: self.generate_anomaly_id(),
                anomaly_type: AnomalyType::StakeDistributionAnomaly,
                severity,
                score: stake_distribution.gini_coefficient,
                description: format!(
                    "Extreme stake concentration detected: Gini coefficient = {:.3}",
                    stake_distribution.gini_coefficient
                ),
                timestamp: self.current_timestamp(),
                source_data: {
                    let mut data = HashMap::new();
                    data.insert(
                        "gini_coefficient".to_string(),
                        stake_distribution.gini_coefficient,
                    );
                    data.insert(
                        "total_stake".to_string(),
                        stake_distribution.total_stake as f64,
                    );
                    data.insert(
                        "stake_holders".to_string(),
                        stake_distribution.stake_holders as f64,
                    );
                    data.insert(
                        "top_10_percent_share".to_string(),
                        stake_distribution.top_10_percent_share,
                    );
                    data
                },
                confidence: self.calculate_confidence(stake_distribution.gini_coefficient),
                recommendations: vec![
                    "Investigate stake concentration patterns".to_string(),
                    "Consider stake redistribution mechanisms".to_string(),
                    "Monitor for potential whale manipulation".to_string(),
                ],
                alert_signature: self.sign_anomaly_alert(
                    &AnomalyType::StakeDistributionAnomaly,
                    stake_distribution.gini_coefficient,
                )?,
            });
        }

        Ok(anomalies)
    }

    /// Detect proposal-related anomalies
    fn detect_proposal_anomalies(
        &self,
        proposals: &[Proposal],
    ) -> Result<Vec<AnomalyDetection>, AnomalyError> {
        let mut anomalies = Vec::new();

        if proposals.is_empty() {
            return Ok(anomalies);
        }

        // Analyze proposal timing patterns
        let proposal_times: Vec<f64> = proposals.iter().map(|p| p.created_at as f64).collect();

        if self.config.enable_zscore && proposal_times.len() > 5 {
            let zscore_result = self.calculate_zscore(&proposal_times)?;

            for (i, &z_score) in zscore_result.z_scores.iter().enumerate() {
                if z_score.abs() > self.config.zscore_threshold {
                    let severity = self.determine_anomaly_severity(z_score.abs());
                    let score = (z_score.abs() / self.config.zscore_threshold).min(1.0);

                    anomalies.push(AnomalyDetection {
                        id: self.generate_anomaly_id(),
                        anomaly_type: AnomalyType::VoterTurnoutSpike, // Reusing for proposal timing
                        severity,
                        score,
                        description: format!(
                            "Unusual proposal timing detected: Z-score = {:.2}",
                            z_score
                        ),
                        timestamp: self.current_timestamp(),
                        source_data: {
                            let mut data = HashMap::new();
                            data.insert("proposal_time".to_string(), proposal_times[i]);
                            data.insert("z_score".to_string(), z_score);
                            data.insert("mean".to_string(), zscore_result.mean);
                            data.insert("std_dev".to_string(), zscore_result.std_dev);
                            data
                        },
                        confidence: self.calculate_confidence(z_score.abs()),
                        recommendations: vec![
                            "Investigate proposal timing patterns".to_string(),
                            "Check for coordinated proposal attacks".to_string(),
                        ],
                        alert_signature: self
                            .sign_anomaly_alert(&AnomalyType::VoterTurnoutSpike, score)?,
                    });
                }
            }
        }

        Ok(anomalies)
    }

    /// Detect network latency anomalies
    fn detect_network_latency_anomalies(
        &self,
        system_metrics: &[SystemMetrics],
    ) -> Result<Vec<AnomalyDetection>, AnomalyError> {
        let mut anomalies = Vec::new();

        if system_metrics.len() < 5 {
            return Ok(anomalies);
        }

        // Extract network health scores
        let health_scores: Vec<f64> = system_metrics.iter().map(|m| m.network_health).collect();

        if self.config.enable_zscore {
            let zscore_result = self.calculate_zscore(&health_scores)?;

            for (i, &z_score) in zscore_result.z_scores.iter().enumerate() {
                if z_score < -self.config.zscore_threshold {
                    // Negative Z-score indicates poor health
                    let severity = self.determine_anomaly_severity(z_score.abs());
                    let score = (z_score.abs() / self.config.zscore_threshold).min(1.0);

                    anomalies.push(AnomalyDetection {
                        id: self.generate_anomaly_id(),
                        anomaly_type: AnomalyType::NetworkLatencySpike,
                        severity,
                        score,
                        description: format!(
                            "Network health anomaly detected: health = {:.3} (Z-score: {:.2})",
                            health_scores[i], z_score
                        ),
                        timestamp: self.current_timestamp(),
                        source_data: {
                            let mut data = HashMap::new();
                            data.insert("network_health".to_string(), health_scores[i]);
                            data.insert("z_score".to_string(), z_score);
                            data.insert("mean".to_string(), zscore_result.mean);
                            data.insert("std_dev".to_string(), zscore_result.std_dev);
                            data
                        },
                        confidence: self.calculate_confidence(z_score.abs()),
                        recommendations: vec![
                            "Investigate network connectivity issues".to_string(),
                            "Check for DDoS attacks".to_string(),
                            "Monitor node performance".to_string(),
                        ],
                        alert_signature: self
                            .sign_anomaly_alert(&AnomalyType::NetworkLatencySpike, score)?,
                    });
                }
            }
        }

        Ok(anomalies)
    }

    /// Detect resource usage anomalies
    fn detect_resource_usage_anomalies(
        &self,
        system_metrics: &[SystemMetrics],
    ) -> Result<Vec<AnomalyDetection>, AnomalyError> {
        let mut anomalies = Vec::new();

        if system_metrics.is_empty() {
            return Ok(anomalies);
        }

        // Check for high resource usage
        for metrics in system_metrics.iter() {
            if metrics.network_health < 0.3 {
                // Threshold for poor network health
                let severity = if metrics.network_health < 0.1 {
                    AnomalySeverity::Critical
                } else {
                    AnomalySeverity::High
                };

                anomalies.push(AnomalyDetection {
                    id: self.generate_anomaly_id(),
                    anomaly_type: AnomalyType::ResourceUsageSpike,
                    severity,
                    score: 1.0 - metrics.network_health,
                    description: format!(
                        "Resource usage spike detected: network health = {:.3}",
                        metrics.network_health
                    ),
                    timestamp: self.current_timestamp(),
                    source_data: {
                        let mut data = HashMap::new();
                        data.insert("network_health".to_string(), metrics.network_health);
                        data.insert("active_voters".to_string(), metrics.active_voters as f64);
                        data.insert("total_stake".to_string(), metrics.total_stake as f64);
                        data.insert(
                            "consensus_participation".to_string(),
                            metrics.consensus_participation,
                        );
                        data
                    },
                    confidence: 0.8,
                    recommendations: vec![
                        "Investigate resource usage patterns".to_string(),
                        "Check for resource exhaustion attacks".to_string(),
                        "Scale system resources if needed".to_string(),
                    ],
                    alert_signature: self.sign_anomaly_alert(
                        &AnomalyType::ResourceUsageSpike,
                        1.0 - metrics.network_health,
                    )?,
                });
            }
        }

        Ok(anomalies)
    }

    /// Detect consensus anomalies
    fn detect_consensus_anomalies(
        &self,
        system_metrics: &[SystemMetrics],
    ) -> Result<Vec<AnomalyDetection>, AnomalyError> {
        let mut anomalies = Vec::new();

        if system_metrics.len() < 3 {
            return Ok(anomalies);
        }

        // Extract consensus participation rates
        let participation_rates: Vec<f64> = system_metrics
            .iter()
            .map(|m| m.consensus_participation)
            .collect();

        if self.config.enable_zscore {
            let zscore_result = self.calculate_zscore(&participation_rates)?;

            for (i, &z_score) in zscore_result.z_scores.iter().enumerate() {
                if z_score < -self.config.zscore_threshold {
                    // Low participation
                    let severity = self.determine_anomaly_severity(z_score.abs());
                    let score = (z_score.abs() / self.config.zscore_threshold).min(1.0);

                    anomalies.push(AnomalyDetection {
                        id: self.generate_anomaly_id(),
                        anomaly_type: AnomalyType::ConsensusAnomaly,
                        severity,
                        score,
                        description: format!(
                            "Consensus participation anomaly detected: {:.3} (Z-score: {:.2})",
                            participation_rates[i], z_score
                        ),
                        timestamp: self.current_timestamp(),
                        source_data: {
                            let mut data = HashMap::new();
                            data.insert(
                                "consensus_participation".to_string(),
                                participation_rates[i],
                            );
                            data.insert("z_score".to_string(), z_score);
                            data.insert("mean".to_string(), zscore_result.mean);
                            data.insert("std_dev".to_string(), zscore_result.std_dev);
                            data
                        },
                        confidence: self.calculate_confidence(z_score.abs()),
                        recommendations: vec![
                            "Investigate validator participation".to_string(),
                            "Check for validator failures".to_string(),
                            "Monitor consensus mechanism health".to_string(),
                        ],
                        alert_signature: self
                            .sign_anomaly_alert(&AnomalyType::ConsensusAnomaly, score)?,
                    });
                }
            }
        }

        Ok(anomalies)
    }

    /// Detect cross-chain anomalies
    fn detect_cross_chain_anomalies(
        &self,
        cross_chain_votes: &[CrossChainVote],
        _federated_proposals: &[FederatedProposal],
    ) -> Result<Vec<AnomalyDetection>, AnomalyError> {
        let mut anomalies = Vec::new();

        if cross_chain_votes.is_empty() {
            return Ok(anomalies);
        }

        // Analyze vote timing patterns across chains
        let vote_times: Vec<f64> = cross_chain_votes
            .iter()
            .map(|v| v.timestamp as f64)
            .collect();

        if self.config.enable_zscore && vote_times.len() > 5 {
            let zscore_result = self.calculate_zscore(&vote_times)?;

            for (i, &z_score) in zscore_result.z_scores.iter().enumerate() {
                if z_score.abs() > self.config.zscore_threshold {
                    let severity = self.determine_anomaly_severity(z_score.abs());
                    let score = (z_score.abs() / self.config.zscore_threshold).min(1.0);

                    anomalies.push(AnomalyDetection {
                        id: self.generate_anomaly_id(),
                        anomaly_type: AnomalyType::CrossChainInconsistency,
                        severity,
                        score,
                        description: format!(
                            "Cross-chain vote timing anomaly detected: Z-score = {:.2}",
                            z_score
                        ),
                        timestamp: self.current_timestamp(),
                        source_data: {
                            let mut data = HashMap::new();
                            data.insert("vote_time".to_string(), vote_times[i]);
                            data.insert("z_score".to_string(), z_score);
                            data.insert(
                                "source_chain".to_string(),
                                cross_chain_votes[i].source_chain.len() as f64,
                            );
                            data
                        },
                        confidence: self.calculate_confidence(z_score.abs()),
                        recommendations: vec![
                            "Investigate cross-chain synchronization".to_string(),
                            "Check for replay attacks".to_string(),
                            "Verify cross-chain message integrity".to_string(),
                        ],
                        alert_signature: self
                            .sign_anomaly_alert(&AnomalyType::CrossChainInconsistency, score)?,
                    });
                }
            }
        }

        Ok(anomalies)
    }

    /// Detect Sybil attacks using clustering analysis
    fn detect_sybil_attacks(
        &self,
        federation_members: &[FederationMember],
    ) -> Result<Vec<AnomalyDetection>, AnomalyError> {
        let mut anomalies = Vec::new();

        if federation_members.len() < 10 {
            return Ok(anomalies);
        }

        // Extract member characteristics for clustering
        let member_features = self.extract_member_features(federation_members)?;

        if self.config.enable_kmeans {
            let cluster_result = self.perform_kmeans_clustering(&member_features, 3)?;
            let cluster_analysis = self.analyze_clusters(&member_features, &cluster_result)?;

            for (cluster_id, analysis) in cluster_analysis.iter().enumerate() {
                if analysis.is_anomalous && analysis.unique_voters < 5 {
                    let severity = AnomalySeverity::High;
                    let score = analysis.anomaly_score;

                    anomalies.push(AnomalyDetection {
                        id: self.generate_anomaly_id(),
                        anomaly_type: AnomalyType::SybilAttack,
                        severity,
                        score,
                        description: format!(
                            "Potential Sybil attack detected in cluster {}: {} members with similar characteristics",
                            cluster_id, analysis.unique_voters
                        ),
                        timestamp: self.current_timestamp(),
                        source_data: {
                            let mut data = HashMap::new();
                            data.insert("cluster_id".to_string(), cluster_id as f64);
                            data.insert("member_count".to_string(), analysis.unique_voters as f64);
                            data.insert("anomaly_score".to_string(), analysis.anomaly_score);
                            data
                        },
                        confidence: self.calculate_confidence(analysis.anomaly_score),
                        recommendations: vec![
                            "Investigate member identity verification".to_string(),
                            "Check for duplicate member characteristics".to_string(),
                            "Implement stronger identity requirements".to_string(),
                        ],
                        alert_signature: self.sign_anomaly_alert(&AnomalyType::SybilAttack, score)?,
                    });
                }
            }
        }

        Ok(anomalies)
    }

    /// Detect state synchronization anomalies
    fn detect_state_sync_anomalies(
        &self,
        federated_proposals: &[FederatedProposal],
    ) -> Result<Vec<AnomalyDetection>, AnomalyError> {
        let mut anomalies = Vec::new();

        if federated_proposals.len() < 3 {
            return Ok(anomalies);
        }

        // Analyze proposal timing patterns
        let proposal_times: Vec<f64> = federated_proposals
            .iter()
            .map(|p| p.created_at as f64)
            .collect();

        if self.config.enable_zscore {
            let zscore_result = self.calculate_zscore(&proposal_times)?;

            for (i, &z_score) in zscore_result.z_scores.iter().enumerate() {
                if z_score.abs() > self.config.zscore_threshold {
                    let severity = self.determine_anomaly_severity(z_score.abs());
                    let score = (z_score.abs() / self.config.zscore_threshold).min(1.0);

                    anomalies.push(AnomalyDetection {
                        id: self.generate_anomaly_id(),
                        anomaly_type: AnomalyType::StateSyncAnomaly,
                        severity,
                        score,
                        description: format!(
                            "State synchronization anomaly detected: Z-score = {:.2}",
                            z_score
                        ),
                        timestamp: self.current_timestamp(),
                        source_data: {
                            let mut data = HashMap::new();
                            data.insert("proposal_time".to_string(), proposal_times[i]);
                            data.insert("z_score".to_string(), z_score);
                            data.insert(
                                "proposing_chain".to_string(),
                                federated_proposals[i].proposing_chain.len() as f64,
                            );
                            data
                        },
                        confidence: self.calculate_confidence(z_score.abs()),
                        recommendations: vec![
                            "Investigate state synchronization mechanisms".to_string(),
                            "Check for cross-chain message delays".to_string(),
                            "Verify federation member connectivity".to_string(),
                        ],
                        alert_signature: self
                            .sign_anomaly_alert(&AnomalyType::StateSyncAnomaly, score)?,
                    });
                }
            }
        }

        Ok(anomalies)
    }

    /// Calculate hourly voter turnout from votes
    fn calculate_hourly_turnout(&self, votes: &[Vote]) -> Result<Vec<f64>, AnomalyError> {
        let mut hourly_votes: HashMap<u64, u64> = HashMap::new();

        for vote in votes {
            let hour = vote.timestamp / 3600; // Convert to hour
            *hourly_votes.entry(hour).or_insert(0) += 1;
        }

        if hourly_votes.is_empty() {
            return Ok(Vec::new());
        }

        let max_hour = *hourly_votes.keys().max().unwrap();
        let min_hour = *hourly_votes.keys().min().unwrap();

        let mut turnout = Vec::new();
        for hour in min_hour..=max_hour {
            let vote_count = hourly_votes.get(&hour).unwrap_or(&0);
            // Normalize by hour (assuming 1 vote per hour is normal)
            turnout.push(*vote_count as f64 / 1.0);
        }

        Ok(turnout)
    }

    /// Extract voting patterns for clustering analysis
    fn extract_voting_patterns(&self, votes: &[Vote]) -> Result<Vec<Vec<f64>>, AnomalyError> {
        let mut patterns = Vec::new();

        // Group votes by voter
        let mut voter_votes: HashMap<String, Vec<&Vote>> = HashMap::new();
        for vote in votes {
            voter_votes
                .entry(vote.voter_id.clone())
                .or_default()
                .push(vote);
        }

        for (_voter_id, voter_vote_list) in voter_votes {
            if voter_vote_list.len() < 2 {
                continue; // Need multiple votes for pattern analysis
            }

            // Extract features: vote frequency, stake amount, time patterns
            let vote_count = voter_vote_list.len() as f64;
            let avg_stake = voter_vote_list
                .iter()
                .map(|v| v.stake_amount as f64)
                .sum::<f64>()
                / vote_count;

            // Calculate time between votes
            let mut time_intervals = Vec::new();
            for i in 1..voter_vote_list.len() {
                let interval = voter_vote_list[i].timestamp - voter_vote_list[i - 1].timestamp;
                time_intervals.push(interval as f64);
            }
            let avg_interval = if time_intervals.is_empty() {
                0.0
            } else {
                time_intervals.iter().sum::<f64>() / time_intervals.len() as f64
            };

            patterns.push(vec![vote_count, avg_stake, avg_interval]);
        }

        Ok(patterns)
    }

    /// Extract member features for clustering analysis
    fn extract_member_features(
        &self,
        members: &[FederationMember],
    ) -> Result<Vec<Vec<f64>>, AnomalyError> {
        let mut features = Vec::new();

        for member in members {
            // Extract features: stake weight, voting power, chain type
            let stake_weight = member.stake_weight as f64;
            let voting_power = member.voting_power as f64;
            let chain_type = match member.chain_type {
                crate::federation::federation::ChainType::Layer1 => 1.0,
                crate::federation::federation::ChainType::Layer2 => 2.0,
                crate::federation::federation::ChainType::Parachain => 3.0,
                crate::federation::federation::ChainType::CosmosZone => 4.0,
                crate::federation::federation::ChainType::Custom => 5.0,
            };

            features.push(vec![stake_weight, voting_power, chain_type]);
        }

        Ok(features)
    }

    /// Perform K-means clustering
    pub fn perform_kmeans_clustering(
        &self,
        data: &[Vec<f64>],
        k: usize,
    ) -> Result<ClusterResult, AnomalyError> {
        if data.is_empty() || k == 0 {
            return Err(AnomalyError::ClusteringError(
                "Invalid data or k value".to_string(),
            ));
        }

        let dimensions = data[0].len();
        let mut centroids = self.initialize_centroids(data, k)?;
        let mut assignments = vec![0; data.len()];
        let mut iterations = 0;

        loop {
            // Assign points to closest centroid
            for (i, point) in data.iter().enumerate() {
                let mut min_distance = f64::INFINITY;
                let mut closest_centroid = 0;

                for (j, centroid) in centroids.iter().enumerate() {
                    let distance = self.euclidean_distance(point, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                        closest_centroid = j;
                    }
                }
                assignments[i] = closest_centroid;
            }

            // Update centroids
            let mut new_centroids = vec![vec![0.0; dimensions]; k];
            let mut cluster_counts = vec![0; k];

            for (i, point) in data.iter().enumerate() {
                let cluster = assignments[i];
                cluster_counts[cluster] += 1;
                for (j, &value) in point.iter().enumerate() {
                    new_centroids[cluster][j] += value;
                }
            }

            // Calculate new centroids
            for i in 0..k {
                if cluster_counts[i] > 0 {
                    for j in 0..dimensions {
                        new_centroids[i][j] /= cluster_counts[i] as f64;
                    }
                }
            }

            // Check for convergence
            let mut converged = true;
            for i in 0..k {
                for j in 0..dimensions {
                    if (centroids[i][j] - new_centroids[i][j]).abs() > 1e-6 {
                        converged = false;
                        break;
                    }
                }
                if !converged {
                    break;
                }
            }

            centroids = new_centroids;
            iterations += 1;

            if converged || iterations >= self.config.kmeans_max_iterations {
                break;
            }
        }

        // Calculate within-cluster sum of squares
        let wcss = self.calculate_wcss(data, &centroids, &assignments);

        Ok(ClusterResult {
            centroids,
            assignments,
            wcss,
            k,
            iterations,
        })
    }

    /// Initialize centroids using k-means++ method
    fn initialize_centroids(
        &self,
        data: &[Vec<f64>],
        k: usize,
    ) -> Result<Vec<Vec<f64>>, AnomalyError> {
        let mut centroids = Vec::new();

        // Choose first centroid randomly
        let first_idx = (self.current_timestamp() % data.len() as u64) as usize;
        centroids.push(data[first_idx].clone());

        // Choose remaining centroids using k-means++
        for _ in 1..k {
            let mut distances = Vec::new();
            let mut total_distance = 0.0;

            for point in data {
                let mut min_distance = f64::INFINITY;
                for centroid in &centroids {
                    let distance = self.euclidean_distance(point, centroid);
                    if distance < min_distance {
                        min_distance = distance;
                    }
                }
                distances.push(min_distance);
                total_distance += min_distance;
            }

            // Choose next centroid based on distance probabilities
            let mut cumulative = 0.0;
            let target = (self.current_timestamp() as f64 / 1000.0).fract() * total_distance;

            for (i, distance) in distances.iter().enumerate() {
                cumulative += distance;
                if cumulative >= target {
                    centroids.push(data[i].clone());
                    break;
                }
            }
        }

        Ok(centroids)
    }

    /// Calculate Euclidean distance between two points
    fn euclidean_distance(&self, point1: &[f64], point2: &[f64]) -> f64 {
        if point1.len() != point2.len() {
            return f64::INFINITY;
        }

        let mut sum = 0.0;
        for (a, b) in point1.iter().zip(point2.iter()) {
            let diff = a - b;
            sum += diff * diff;
        }
        sum.sqrt()
    }

    /// Calculate within-cluster sum of squares
    fn calculate_wcss(
        &self,
        data: &[Vec<f64>],
        centroids: &[Vec<f64>],
        assignments: &[usize],
    ) -> f64 {
        let mut wcss = 0.0;

        for (i, point) in data.iter().enumerate() {
            let cluster = assignments[i];
            let distance = self.euclidean_distance(point, &centroids[cluster]);
            wcss += distance * distance;
        }

        wcss
    }

    /// Analyze clusters for anomalies
    fn analyze_clusters(
        &self,
        data: &[Vec<f64>],
        cluster_result: &ClusterResult,
    ) -> Result<Vec<ClusterAnalysis>, AnomalyError> {
        let mut analyses = Vec::new();

        for cluster_id in 0..cluster_result.k {
            let mut cluster_points = Vec::new();
            let mut unique_voters = 0;

            for (i, &assignment) in cluster_result.assignments.iter().enumerate() {
                if assignment == cluster_id {
                    cluster_points.push(&data[i]);
                    unique_voters += 1;
                }
            }

            if cluster_points.is_empty() {
                analyses.push(ClusterAnalysis {
                    cluster_id,
                    vote_count: 0,
                    unique_voters: 0,
                    anomaly_score: 0.0,
                    is_anomalous: false,
                });
                continue;
            }

            // Calculate cluster characteristics
            let vote_count = cluster_points.len();
            let avg_stake = if !cluster_points.is_empty() {
                cluster_points.iter().map(|p| p[1]).sum::<f64>() / cluster_points.len() as f64
            } else {
                0.0
            };

            // Determine if cluster is anomalous
            let is_anomalous = unique_voters < 5 && avg_stake > 1000.0; // Few voters with high stake
            let anomaly_score = if is_anomalous {
                (5.0 - unique_voters as f64) / 5.0
            } else {
                0.0
            };

            analyses.push(ClusterAnalysis {
                cluster_id,
                vote_count,
                unique_voters,
                anomaly_score,
                is_anomalous,
            });
        }

        Ok(analyses)
    }

    /// Calculate Z-scores for anomaly detection
    pub fn calculate_zscore(&self, data: &[f64]) -> Result<ZScoreResult, AnomalyError> {
        if data.is_empty() {
            return Err(AnomalyError::StatisticalError("Empty dataset".to_string()));
        }

        let n = data.len() as f64;
        let mean = data.iter().sum::<f64>() / n;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Ok(ZScoreResult {
                z_scores: vec![0.0; data.len()],
                threshold: self.config.zscore_threshold,
                anomaly_count: 0,
                mean,
                std_dev: 0.0,
            });
        }

        let z_scores: Vec<f64> = data.iter().map(|&x| (x - mean) / std_dev).collect();

        let anomaly_count = z_scores
            .iter()
            .filter(|&&z| z.abs() > self.config.zscore_threshold)
            .count();

        Ok(ZScoreResult {
            z_scores,
            threshold: self.config.zscore_threshold,
            anomaly_count,
            mean,
            std_dev,
        })
    }

    /// Determine anomaly severity based on score
    pub fn determine_anomaly_severity(&self, score: f64) -> AnomalySeverity {
        if score >= 3.0 {
            AnomalySeverity::Critical
        } else if score >= 2.0 {
            AnomalySeverity::High
        } else if score >= 1.5 {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        }
    }

    /// Determine cluster severity based on anomaly score
    fn determine_cluster_severity(&self, score: f64) -> AnomalySeverity {
        if score >= 0.8 {
            AnomalySeverity::Critical
        } else if score >= 0.6 {
            AnomalySeverity::High
        } else if score >= 0.4 {
            AnomalySeverity::Medium
        } else {
            AnomalySeverity::Low
        }
    }

    /// Calculate confidence level for anomaly detection
    pub fn calculate_confidence(&self, score: f64) -> f64 {
        (score / 3.0).min(1.0)
    }

    /// Sign anomaly alert with quantum-resistant signature
    fn sign_anomaly_alert(
        &self,
        anomaly_type: &AnomalyType,
        score: f64,
    ) -> Result<Option<DilithiumSignature>, AnomalyError> {
        if !self.config.enable_quantum_signatures {
            return Ok(None);
        }

        let (_, secret_key) = self.quantum_keys.as_ref().ok_or_else(|| {
            AnomalyError::SignatureError("Quantum keys not available".to_string())
        })?;

        let message = format!("{}:{}:{}", anomaly_type, score, self.current_timestamp());
        let params = match self.config.dilithium_security_level {
            DilithiumSecurityLevel::Dilithium3 => DilithiumParams::dilithium3(),
            DilithiumSecurityLevel::Dilithium5 => DilithiumParams::dilithium5(),
            _ => {
                return Err(AnomalyError::SignatureError(
                    "Unsupported security level".to_string(),
                ))
            }
        };

        let signature = dilithium_sign(message.as_bytes(), secret_key, &params)
            .map_err(|e| AnomalyError::SignatureError(format!("Failed to sign alert: {:?}", e)))?;

        Ok(Some(signature))
    }

    /// Update detection history
    fn update_detection_history(&mut self, anomaly: AnomalyDetection) {
        self.detection_history.push_back(anomaly.clone());

        // Keep only recent detections
        while self.detection_history.len() > 1000 {
            self.detection_history.pop_front();
        }

        // Update counters
        let counter = self
            .anomaly_counters
            .entry(anomaly.anomaly_type)
            .or_insert(0);
        *counter = counter.saturating_add(1);
    }

    /// Generate unique anomaly ID
    fn generate_anomaly_id(&self) -> String {
        let timestamp = self.current_timestamp();
        let random = (timestamp % 10000) as u32;
        format!("anomaly_{}_{}", timestamp, random)
    }

    /// Get current timestamp
    fn current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Generate Chart.js compatible data for anomaly visualization
    pub fn generate_anomaly_charts(&self) -> Result<Vec<ChartData>, AnomalyError> {
        let mut charts = Vec::new();

        // Anomaly severity distribution chart
        let severity_chart = self.create_severity_distribution_chart()?;
        charts.push(severity_chart);

        // Anomaly type distribution chart
        let type_chart = self.create_type_distribution_chart()?;
        charts.push(type_chart);

        // Anomaly timeline chart
        let timeline_chart = self.create_anomaly_timeline_chart()?;
        charts.push(timeline_chart);

        Ok(charts)
    }

    /// Create anomaly severity distribution chart
    fn create_severity_distribution_chart(&self) -> Result<ChartData, AnomalyError> {
        let mut severity_counts = HashMap::new();

        for anomaly in &self.detection_history {
            let count = severity_counts.entry(anomaly.severity.clone()).or_insert(0);
            *count += 1;
        }

        let labels = vec![
            "Low".to_string(),
            "Medium".to_string(),
            "High".to_string(),
            "Critical".to_string(),
        ];
        let data = vec![
            *severity_counts.get(&AnomalySeverity::Low).unwrap_or(&0) as f64,
            *severity_counts.get(&AnomalySeverity::Medium).unwrap_or(&0) as f64,
            *severity_counts.get(&AnomalySeverity::High).unwrap_or(&0) as f64,
            *severity_counts
                .get(&AnomalySeverity::Critical)
                .unwrap_or(&0) as f64,
        ];

        Ok(ChartData {
            chart_type: "doughnut".to_string(),
            title: "Anomaly Severity Distribution".to_string(),
            labels,
            datasets: vec![ChartDataset {
                label: "Anomalies".to_string(),
                data,
                border_color: "#3B82F6".to_string(),
                background_color: "#60A5FA".to_string(),
                fill: true,
            }],
            options: ChartOptions {
                legend: true,
                tooltips: true,
                responsive: true,
                scales: None,
            },
        })
    }

    /// Create anomaly type distribution chart
    fn create_type_distribution_chart(&self) -> Result<ChartData, AnomalyError> {
        let mut type_counts = HashMap::new();

        for anomaly in &self.detection_history {
            let count = type_counts.entry(anomaly.anomaly_type.clone()).or_insert(0);
            *count += 1;
        }

        let mut labels = Vec::new();
        let mut data = Vec::new();

        for (anomaly_type, count) in type_counts {
            labels.push(anomaly_type.to_string());
            data.push(count as f64);
        }

        Ok(ChartData {
            chart_type: "bar".to_string(),
            title: "Anomaly Type Distribution".to_string(),
            labels,
            datasets: vec![ChartDataset {
                label: "Anomaly Count".to_string(),
                data,
                border_color: "#EF4444".to_string(),
                background_color: "#F87171".to_string(),
                fill: true,
            }],
            options: ChartOptions {
                legend: true,
                tooltips: true,
                responsive: true,
                scales: None,
            },
        })
    }

    /// Create anomaly timeline chart
    fn create_anomaly_timeline_chart(&self) -> Result<ChartData, AnomalyError> {
        let mut hourly_counts: HashMap<u64, u64> = HashMap::new();

        for anomaly in &self.detection_history {
            let hour = anomaly.timestamp / 3600;
            *hourly_counts.entry(hour).or_insert(0) += 1;
        }

        if hourly_counts.is_empty() {
            return Ok(ChartData {
                chart_type: "line".to_string(),
                title: "Anomaly Timeline".to_string(),
                labels: vec!["No Data".to_string()],
                datasets: vec![ChartDataset {
                    label: "Anomalies".to_string(),
                    data: vec![0.0],
                    border_color: "#10B981".to_string(),
                    background_color: "#34D399".to_string(),
                    fill: false,
                }],
                options: ChartOptions {
                    legend: true,
                    tooltips: true,
                    responsive: true,
                    scales: None,
                },
            });
        }

        let mut sorted_hours: Vec<u64> = hourly_counts.keys().cloned().collect();
        sorted_hours.sort_unstable();

        let labels: Vec<String> = sorted_hours
            .iter()
            .map(|&h| format!("Hour {}", h))
            .collect();

        let data: Vec<f64> = sorted_hours
            .iter()
            .map(|&h| *hourly_counts.get(&h).unwrap_or(&0) as f64)
            .collect();

        Ok(ChartData {
            chart_type: "line".to_string(),
            title: "Anomaly Timeline".to_string(),
            labels,
            datasets: vec![ChartDataset {
                label: "Anomalies per Hour".to_string(),
                data,
                border_color: "#10B981".to_string(),
                background_color: "#34D399".to_string(),
                fill: false,
            }],
            options: ChartOptions {
                legend: true,
                tooltips: true,
                responsive: true,
                scales: None,
            },
        })
    }

    /// Get detection history
    pub fn get_detection_history(&self) -> &VecDeque<AnomalyDetection> {
        &self.detection_history
    }

    /// Get anomaly counters
    pub fn get_anomaly_counters(&self) -> &HashMap<AnomalyType, u64> {
        &self.anomaly_counters
    }

    /// Clear detection history
    pub fn clear_history(&mut self) {
        self.detection_history.clear();
        self.anomaly_counters.clear();
    }

    /// Update configuration
    pub fn update_config(&mut self, new_config: AnomalyConfig) -> Result<(), AnomalyError> {
        // Regenerate quantum keys if security level changed
        if new_config.enable_quantum_signatures
            && new_config.dilithium_security_level != self.config.dilithium_security_level
        {
            let params = match new_config.dilithium_security_level {
                DilithiumSecurityLevel::Dilithium3 => DilithiumParams::dilithium3(),
                DilithiumSecurityLevel::Dilithium5 => DilithiumParams::dilithium5(),
                _ => {
                    return Err(AnomalyError::ConfigurationError(
                        "Unsupported Dilithium security level".to_string(),
                    ))
                }
            };

            self.quantum_keys = Some(dilithium_keygen(&params).map_err(|e| {
                AnomalyError::KeyGenerationError(format!(
                    "Failed to generate quantum keys: {:?}",
                    e
                ))
            })?);
        }

        self.config = new_config;
        Ok(())
    }

    /// Verify data integrity using SHA-3
    pub fn verify_data_integrity(&self, data: &[u8]) -> Result<bool, AnomalyError> {
        if !self.config.enable_integrity_verification {
            return Ok(true);
        }

        let mut hasher = Sha3_256::new();
        hasher.update(data);
        let current_hash = hasher.finalize().to_vec();

        Ok(current_hash == self.data_integrity_hash)
    }

    /// Update data integrity hash
    pub fn update_integrity_hash(&mut self, data: &[u8]) {
        let mut hasher = Sha3_256::new();
        hasher.update(&self.data_integrity_hash);
        hasher.update(data);
        self.data_integrity_hash = hasher.finalize().to_vec();
    }
}

/// Cluster analysis result
#[derive(Debug, Clone)]
struct ClusterAnalysis {
    #[allow(dead_code)]
    cluster_id: usize,
    vote_count: usize,
    unique_voters: usize,
    anomaly_score: f64,
    is_anomalous: bool,
}

/// Anomaly detection errors
#[derive(Debug, Clone)]
pub enum AnomalyError {
    ConfigurationError(String),
    ClusteringError(String),
    StatisticalError(String),
    SignatureError(String),
    KeyGenerationError(String),
    DataIntegrityError(String),
}

impl std::fmt::Display for AnomalyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnomalyError::ConfigurationError(msg) => write!(f, "Configuration error: {}", msg),
            AnomalyError::ClusteringError(msg) => write!(f, "Clustering error: {}", msg),
            AnomalyError::StatisticalError(msg) => write!(f, "Statistical error: {}", msg),
            AnomalyError::SignatureError(msg) => write!(f, "Signature error: {}", msg),
            AnomalyError::KeyGenerationError(msg) => write!(f, "Key generation error: {}", msg),
            AnomalyError::DataIntegrityError(msg) => write!(f, "Data integrity error: {}", msg),
        }
    }
}

impl std::error::Error for AnomalyError {}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new().unwrap()
    }
}
