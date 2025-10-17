//! Monitoring System for Network Health and Performance
//!
//! This module provides comprehensive monitoring capabilities for the decentralized
//! voting blockchain, tracking health and performance metrics across all system
//! components including PoS consensus, sharding, P2P networking, voting, and governance.
//!
//! Key features:
//! - Real-time metric collection and aggregation
//! - Anomaly detection and alerting
//! - JSON logging interface for external analysis
//! - P2P integration for alert broadcasting
//! - Low-overhead asynchronous monitoring
//! - Cryptographic integrity verification

use sha3::{Digest, Sha3_256};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Represents different types of metrics that can be collected
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Validator uptime percentage
    ValidatorUptime,
    /// Block finalization time in milliseconds
    BlockFinalizationTime,
    /// Number of slashing events
    SlashingEvents,
    /// Shard throughput (transactions per second)
    ShardThroughput,
    /// Cross-shard latency in milliseconds
    CrossShardLatency,
    /// State synchronization success rate
    StateSyncSuccessRate,
    /// Node connectivity percentage
    NodeConnectivity,
    /// Message propagation delay in milliseconds
    MessagePropagationDelay,
    /// Bandwidth usage in bytes per second
    BandwidthUsage,
    /// Vote submission rate (votes per second)
    VoteSubmissionRate,
    /// zk-SNARK verification time in milliseconds
    ZkSnarkVerificationTime,
    /// Staking activity (stakes per hour)
    StakingActivity,
    /// Token transfer rate (transfers per second)
    TokenTransferRate,
    /// Voting weight update rate
    VotingWeightUpdateRate,
}

/// Represents a single metric measurement
#[derive(Debug, Clone)]
pub struct Metric {
    /// Type of metric
    pub metric_type: MetricType,
    /// Value of the metric
    pub value: f64,
    /// Timestamp when metric was collected
    pub timestamp: u64,
    /// Source module that generated the metric
    pub source: String,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Represents aggregated statistics for a metric type
#[derive(Debug, Clone)]
pub struct MetricStats {
    /// Current value
    pub current: f64,
    /// Average value over time window
    pub average: f64,
    /// Minimum value in time window
    pub minimum: f64,
    /// Maximum value in time window
    pub maximum: f64,
    /// Standard deviation
    pub standard_deviation: f64,
    /// Number of samples
    pub sample_count: usize,
    /// Time window duration in seconds
    pub window_duration: u64,
}

/// Represents an alert condition
#[derive(Debug, Clone, PartialEq)]
pub enum AlertLevel {
    /// Informational alert
    Info,
    /// Warning alert
    Warning,
    /// Critical alert
    Critical,
    /// Emergency alert
    Emergency,
}

/// Represents an alert that can be triggered by metric conditions
#[derive(Debug, Clone)]
pub struct Alert {
    /// Unique alert identifier
    pub id: String,
    /// Alert level
    pub level: AlertLevel,
    /// Alert message
    pub message: String,
    /// Metric type that triggered the alert
    pub metric_type: MetricType,
    /// Threshold value that was exceeded
    pub threshold: f64,
    /// Actual value that triggered the alert
    pub actual_value: f64,
    /// Timestamp when alert was triggered
    pub timestamp: u64,
    /// Whether alert has been acknowledged
    pub acknowledged: bool,
}

/// System metrics for monitoring
#[derive(Debug, Clone)]
pub struct SystemMetrics {
    /// Timestamp of the metrics
    pub timestamp: u64,
    /// Number of active voters
    pub active_voters: u64,
    /// Total stake in the system
    pub total_stake: u64,
    /// Current block height
    pub block_height: u64,
    /// Total transaction count
    pub transaction_count: u64,
    /// Network health score (0.0 to 1.0)
    pub network_health: f64,
    /// Consensus participation rate (0.0 to 1.0)
    pub consensus_participation: f64,
}

/// Voter activity tracking
#[derive(Debug, Clone)]
pub struct VoterActivity {
    /// Voter's unique identifier
    pub voter_id: String,
    /// Timestamp of the activity
    pub timestamp: u64,
    /// Stake amount involved in the activity
    pub stake_amount: u64,
    /// Type of activity (vote, stake, unstake, etc.)
    pub activity_type: String,
    /// Associated proposal ID (if applicable)
    pub proposal_id: Option<String>,
}

/// Represents a monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Collection interval in milliseconds
    pub collection_interval: u64,
    /// Aggregation window in seconds
    pub aggregation_window: u64,
    /// Maximum number of metrics to keep in memory
    pub max_metrics: usize,
    /// Alert thresholds for different metric types
    pub alert_thresholds: HashMap<MetricType, f64>,
    /// Whether to enable anomaly detection
    pub enable_anomaly_detection: bool,
    /// Whether to enable P2P alert broadcasting
    pub enable_p2p_alerts: bool,
    /// Logging configuration
    pub log_format: LogFormat,
}

/// Represents different logging formats
#[derive(Debug, Clone, PartialEq)]
pub enum LogFormat {
    /// JSON format for external analysis
    Json,
    /// Human-readable format
    Human,
    /// CSV format for data analysis
    Csv,
}

/// Main monitoring system that collects and analyzes metrics
#[derive(Debug, Clone)]
pub struct MonitoringSystem {
    /// Configuration for the monitoring system
    config: MonitoringConfig,
    /// Collected metrics storage
    metrics: Arc<Mutex<HashMap<MetricType, VecDeque<Metric>>>>,
    /// Aggregated statistics
    stats: Arc<Mutex<HashMap<MetricType, MetricStats>>>,
    /// Active alerts
    alerts: Arc<Mutex<Vec<Alert>>>,
    /// Anomaly detection state
    anomaly_state: Arc<Mutex<HashMap<MetricType, VecDeque<f64>>>>,
    /// System start time
    start_time: Instant,
    /// Cryptographic hash for integrity verification
    integrity_hash: Arc<Mutex<Vec<u8>>>,
}

impl MonitoringSystem {
    /// Creates a new monitoring system with default configuration
    ///
    /// # Returns
    /// A new MonitoringSystem instance
    pub fn new() -> Self {
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert(MetricType::ValidatorUptime, 50.0); // 50% uptime threshold
        alert_thresholds.insert(MetricType::BlockFinalizationTime, 5000.0); // 5 second threshold
        alert_thresholds.insert(MetricType::SlashingEvents, 10.0); // 10 slashing events
        alert_thresholds.insert(MetricType::ShardThroughput, 100.0); // 100 TPS threshold
        alert_thresholds.insert(MetricType::CrossShardLatency, 1000.0); // 1 second threshold
        alert_thresholds.insert(MetricType::StateSyncSuccessRate, 80.0); // 80% success rate
        alert_thresholds.insert(MetricType::NodeConnectivity, 70.0); // 70% connectivity
        alert_thresholds.insert(MetricType::MessagePropagationDelay, 2000.0); // 2 second threshold
        alert_thresholds.insert(MetricType::BandwidthUsage, 1000000.0); // 1MB/s threshold
        alert_thresholds.insert(MetricType::VoteSubmissionRate, 50.0); // 50 votes/second
        alert_thresholds.insert(MetricType::ZkSnarkVerificationTime, 10000.0); // 10 second threshold
        alert_thresholds.insert(MetricType::StakingActivity, 100.0); // 100 stakes/hour
        alert_thresholds.insert(MetricType::TokenTransferRate, 200.0); // 200 transfers/second
        alert_thresholds.insert(MetricType::VotingWeightUpdateRate, 10.0); // 10 updates/second

        Self {
            config: MonitoringConfig {
                collection_interval: 1000, // 1 second
                aggregation_window: 300,   // 5 minutes
                max_metrics: 10000,
                alert_thresholds,
                enable_anomaly_detection: true,
                enable_p2p_alerts: true,
                log_format: LogFormat::Json,
            },
            metrics: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(HashMap::new())),
            alerts: Arc::new(Mutex::new(Vec::new())),
            anomaly_state: Arc::new(Mutex::new(HashMap::new())),
            start_time: Instant::now(),
            integrity_hash: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Creates a new monitoring system with custom configuration
    ///
    /// # Arguments
    /// * `config` - Custom monitoring configuration
    ///
    /// # Returns
    /// A new MonitoringSystem instance with custom configuration
    pub fn with_config(config: MonitoringConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(Mutex::new(HashMap::new())),
            stats: Arc::new(Mutex::new(HashMap::new())),
            alerts: Arc::new(Mutex::new(Vec::new())),
            anomaly_state: Arc::new(Mutex::new(HashMap::new())),
            start_time: Instant::now(),
            integrity_hash: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Collects a metric from a source module
    ///
    /// # Arguments
    /// * `metric_type` - Type of metric being collected
    /// * `value` - Metric value
    /// * `source` - Source module name
    /// * `metadata` - Additional metadata
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn collect_metric(
        &self,
        metric_type: MetricType,
        value: f64,
        source: &str,
        metadata: HashMap<String, String>,
    ) -> Result<(), String> {
        let timestamp = self.current_timestamp();
        let metric = Metric {
            metric_type: metric_type.clone(),
            value,
            timestamp,
            source: source.to_string(),
            metadata,
        };

        // Store metric
        {
            let mut metrics = self.metrics.lock().unwrap();
            let metric_queue = metrics.entry(metric_type.clone()).or_default();

            // Remove old metrics if we exceed the limit
            while metric_queue.len() >= self.config.max_metrics {
                metric_queue.pop_front();
            }

            metric_queue.push_back(metric.clone());
        }

        // Update statistics
        self.update_statistics(&metric_type)?;

        // Check for alerts
        self.check_alerts(&metric_type, value)?;

        // Update anomaly detection
        if self.config.enable_anomaly_detection {
            self.update_anomaly_detection(&metric_type, value)?;
        }

        // Update integrity hash
        self.update_integrity_hash(&metric)?;

        Ok(())
    }

    /// Updates aggregated statistics for a metric type
    ///
    /// # Arguments
    /// * `metric_type` - Type of metric to update statistics for
    ///
    /// # Returns
    /// Result indicating success or failure
    fn update_statistics(&self, metric_type: &MetricType) -> Result<(), String> {
        let metrics = self.metrics.lock().unwrap();
        let metric_queue = metrics.get(metric_type).ok_or("Metric type not found")?;

        if metric_queue.is_empty() {
            return Ok(());
        }

        let window_start = self.current_timestamp() - self.config.aggregation_window;
        let recent_metrics: Vec<&Metric> = metric_queue
            .iter()
            .filter(|m| m.timestamp >= window_start)
            .collect();

        if recent_metrics.is_empty() {
            return Ok(());
        }

        let values: Vec<f64> = recent_metrics.iter().map(|m| m.value).collect();
        let current = values.last().copied().unwrap_or(0.0);
        let average = values.iter().sum::<f64>() / values.len() as f64;
        let minimum = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let maximum = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate standard deviation
        let variance =
            values.iter().map(|&x| (x - average).powi(2)).sum::<f64>() / values.len() as f64;
        let standard_deviation = variance.sqrt();

        let stats = MetricStats {
            current,
            average,
            minimum,
            maximum,
            standard_deviation,
            sample_count: values.len(),
            window_duration: self.config.aggregation_window,
        };

        let mut stats_map = self.stats.lock().unwrap();
        stats_map.insert(metric_type.clone(), stats);

        Ok(())
    }

    /// Checks if a metric value triggers any alerts
    ///
    /// # Arguments
    /// * `metric_type` - Type of metric to check
    /// * `value` - Current metric value
    ///
    /// # Returns
    /// Result indicating success or failure
    fn check_alerts(&self, metric_type: &MetricType, value: f64) -> Result<(), String> {
        if let Some(threshold) = self.config.alert_thresholds.get(metric_type) {
            let should_alert = match metric_type {
                MetricType::ValidatorUptime
                | MetricType::StateSyncSuccessRate
                | MetricType::NodeConnectivity => {
                    value < *threshold // Lower is worse for these metrics
                }
                _ => value > *threshold, // Higher is worse for other metrics
            };

            if should_alert {
                let alert_level = self.determine_alert_level(metric_type, value, *threshold);
                let alert = Alert {
                    id: format!("{:?}_{}", metric_type, self.current_timestamp()),
                    level: alert_level,
                    message: format!(
                        "{} exceeded threshold: {} (actual: {})",
                        self.format_metric_name(metric_type),
                        threshold,
                        value
                    ),
                    metric_type: metric_type.clone(),
                    threshold: *threshold,
                    actual_value: value,
                    timestamp: self.current_timestamp(),
                    acknowledged: false,
                };

                let mut alerts = self.alerts.lock().unwrap();
                alerts.push(alert);
            }
        }

        Ok(())
    }

    /// Determines the alert level based on metric severity
    ///
    /// # Arguments
    /// * `metric_type` - Type of metric
    /// * `value` - Current value
    /// * `threshold` - Alert threshold
    ///
    /// # Returns
    /// Alert level
    fn determine_alert_level(
        &self,
        metric_type: &MetricType,
        value: f64,
        threshold: f64,
    ) -> AlertLevel {
        let severity = match metric_type {
            MetricType::ValidatorUptime => {
                if value < threshold * 0.5 {
                    4.0
                }
                // Emergency if < 25% uptime
                else if value < threshold * 0.7 {
                    3.0
                }
                // Critical if < 35% uptime
                else if value < threshold {
                    2.0
                }
                // Warning if < 50% uptime
                else {
                    1.0
                } // Info
            }
            MetricType::BlockFinalizationTime => {
                if value > threshold * 2.0 {
                    4.0
                }
                // Emergency if > 10 seconds
                else if value > threshold * 1.5 {
                    3.0
                }
                // Critical if > 7.5 seconds
                else if value > threshold {
                    2.0
                }
                // Warning if > 5 seconds
                else {
                    1.0
                } // Info
            }
            _ => {
                if value > threshold * 2.0 {
                    4.0
                }
                // Emergency
                else if value > threshold * 1.5 {
                    3.0
                }
                // Critical
                else if value > threshold {
                    2.0
                }
                // Warning
                else {
                    1.0
                } // Info
            }
        };

        match severity {
            s if s >= 4.0 => AlertLevel::Emergency,
            s if s >= 3.0 => AlertLevel::Critical,
            s if s >= 2.0 => AlertLevel::Warning,
            _ => AlertLevel::Info,
        }
    }

    /// Updates anomaly detection state
    ///
    /// # Arguments
    /// * `metric_type` - Type of metric
    /// * `value` - Current value
    ///
    /// # Returns
    /// Result indicating success or failure
    fn update_anomaly_detection(&self, metric_type: &MetricType, value: f64) -> Result<(), String> {
        let mut anomaly_state = self.anomaly_state.lock().unwrap();
        let values = anomaly_state.entry(metric_type.clone()).or_default();

        values.push_back(value);

        // Keep only recent values for anomaly detection
        while values.len() > 100 {
            values.pop_front();
        }

        // Detect anomalies using statistical methods
        if values.len() >= 10 {
            let recent_values: Vec<f64> = values.iter().cloned().collect();
            let mean = recent_values.iter().sum::<f64>() / recent_values.len() as f64;
            let variance = recent_values
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / recent_values.len() as f64;
            let std_dev = variance.sqrt();

            // Anomaly if value is more than 3 standard deviations from mean
            if (value - mean).abs() > 3.0 * std_dev {
                let alert = Alert {
                    id: format!("anomaly_{:?}_{}", metric_type, self.current_timestamp()),
                    level: AlertLevel::Warning,
                    message: format!(
                        "Anomaly detected in {}: {} (mean: {}, std_dev: {})",
                        self.format_metric_name(metric_type),
                        value,
                        mean,
                        std_dev
                    ),
                    metric_type: metric_type.clone(),
                    threshold: mean + 3.0 * std_dev,
                    actual_value: value,
                    timestamp: self.current_timestamp(),
                    acknowledged: false,
                };

                let mut alerts = self.alerts.lock().unwrap();
                alerts.push(alert);
            }
        }

        Ok(())
    }

    /// Updates the integrity hash for metric verification
    ///
    /// # Arguments
    /// * `metric` - Metric to include in hash
    ///
    /// # Returns
    /// Result indicating success or failure
    fn update_integrity_hash(&self, metric: &Metric) -> Result<(), String> {
        let mut hasher = Sha3_256::new();
        hasher.update(metric.timestamp.to_le_bytes());
        hasher.update(metric.value.to_le_bytes());
        hasher.update(metric.source.as_bytes());

        let mut integrity_hash = self.integrity_hash.lock().unwrap();
        hasher.update(&*integrity_hash);
        *integrity_hash = hasher.finalize().to_vec();

        Ok(())
    }

    /// Gets current statistics for a metric type
    ///
    /// # Arguments
    /// * `metric_type` - Type of metric to get statistics for
    ///
    /// # Returns
    /// Option containing statistics if available
    pub fn get_statistics(&self, metric_type: &MetricType) -> Option<MetricStats> {
        let stats = self.stats.lock().unwrap();
        stats.get(metric_type).cloned()
    }

    /// Gets all active alerts
    ///
    /// # Returns
    /// Vector of active alerts
    pub fn get_alerts(&self) -> Vec<Alert> {
        let alerts = self.alerts.lock().unwrap();
        alerts.clone()
    }

    /// Acknowledges an alert
    ///
    /// # Arguments
    /// * `alert_id` - ID of alert to acknowledge
    ///
    /// # Returns
    /// Result indicating success or failure
    pub fn acknowledge_alert(&self, alert_id: &str) -> Result<(), String> {
        let mut alerts = self.alerts.lock().unwrap();
        for alert in alerts.iter_mut() {
            if alert.id == alert_id {
                alert.acknowledged = true;
                return Ok(());
            }
        }
        Err("Alert not found".to_string())
    }

    /// Generates a JSON log entry for metrics
    ///
    /// # Arguments
    /// * `metric_type` - Type of metric to log
    ///
    /// # Returns
    /// JSON string containing metric data
    pub fn generate_json_log(&self, metric_type: &MetricType) -> Result<String, String> {
        let stats = self.get_statistics(metric_type);
        let alerts = self.get_alerts();
        let metric_alerts: Vec<&Alert> = alerts
            .iter()
            .filter(|a| a.metric_type == *metric_type && !a.acknowledged)
            .collect();

        let mut json = String::new();
        json.push_str("{\n");
        json.push_str(&format!("  \"timestamp\": {},\n", self.current_timestamp()));
        json.push_str(&format!(
            "  \"metric_type\": \"{}\",\n",
            self.format_metric_name(metric_type)
        ));

        if let Some(stats) = stats {
            json.push_str("  \"statistics\": {\n");
            json.push_str(&format!("    \"current\": {:.2},\n", stats.current));
            json.push_str(&format!("    \"average\": {:.2},\n", stats.average));
            json.push_str(&format!("    \"minimum\": {:.2},\n", stats.minimum));
            json.push_str(&format!("    \"maximum\": {:.2},\n", stats.maximum));
            json.push_str(&format!(
                "    \"standard_deviation\": {:.2},\n",
                stats.standard_deviation
            ));
            json.push_str(&format!("    \"sample_count\": {},\n", stats.sample_count));
            json.push_str(&format!(
                "    \"window_duration\": {}\n",
                stats.window_duration
            ));
            json.push_str("  },\n");
        } else {
            json.push_str("  \"statistics\": null,\n");
        }

        json.push_str("  \"alerts\": [\n");
        for (i, alert) in metric_alerts.iter().enumerate() {
            if i > 0 {
                json.push_str(",\n");
            }
            json.push_str("    {\n");
            json.push_str(&format!("      \"id\": \"{}\",\n", alert.id));
            json.push_str(&format!("      \"level\": \"{:?}\",\n", alert.level));
            json.push_str(&format!("      \"message\": \"{}\",\n", alert.message));
            json.push_str(&format!("      \"threshold\": {:.2},\n", alert.threshold));
            json.push_str(&format!(
                "      \"actual_value\": {:.2}\n",
                alert.actual_value
            ));
            json.push_str("    }");
        }
        json.push_str("\n  ],\n");

        let integrity_hash = self.integrity_hash.lock().unwrap();
        json.push_str(&format!(
            "  \"integrity_hash\": \"{}\"\n",
            integrity_hash
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect::<String>()
        ));
        json.push('}');

        Ok(json)
    }

    /// Generates a human-readable log entry
    ///
    /// # Arguments
    /// * `metric_type` - Type of metric to log
    ///
    /// # Returns
    /// Human-readable string containing metric data
    pub fn generate_human_log(&self, metric_type: &MetricType) -> Result<String, String> {
        let stats = self.get_statistics(metric_type);
        let alerts = self.get_alerts();
        let metric_alerts: Vec<&Alert> = alerts
            .iter()
            .filter(|a| a.metric_type == *metric_type && !a.acknowledged)
            .collect();

        let mut log = format!("=== {} Metrics ===\n", self.format_metric_name(metric_type));
        log.push_str(&format!("Timestamp: {}\n", self.current_timestamp()));

        if let Some(stats) = stats {
            log.push_str(&format!("Current: {:.2}\n", stats.current));
            log.push_str(&format!("Average: {:.2}\n", stats.average));
            log.push_str(&format!("Min: {:.2}\n", stats.minimum));
            log.push_str(&format!("Max: {:.2}\n", stats.maximum));
            log.push_str(&format!("Std Dev: {:.2}\n", stats.standard_deviation));
            log.push_str(&format!("Samples: {}\n", stats.sample_count));
        }

        if !metric_alerts.is_empty() {
            log.push_str("\n=== Active Alerts ===\n");
            for alert in metric_alerts {
                log.push_str(&format!(
                    "[{:?}] {}: {}\n",
                    alert.level, alert.id, alert.message
                ));
            }
        }

        Ok(log)
    }

    /// Gets system uptime in seconds
    ///
    /// # Returns
    /// System uptime in seconds
    pub fn get_uptime(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Gets current timestamp
    ///
    /// # Returns
    /// Current timestamp in seconds
    fn current_timestamp(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs()
    }

    /// Formats metric name for display
    ///
    /// # Arguments
    /// * `metric_type` - Type of metric
    ///
    /// # Returns
    /// Formatted metric name
    fn format_metric_name(&self, metric_type: &MetricType) -> String {
        match metric_type {
            MetricType::ValidatorUptime => "Validator Uptime",
            MetricType::BlockFinalizationTime => "Block Finalization Time",
            MetricType::SlashingEvents => "Slashing Events",
            MetricType::ShardThroughput => "Shard Throughput",
            MetricType::CrossShardLatency => "Cross-Shard Latency",
            MetricType::StateSyncSuccessRate => "State Sync Success Rate",
            MetricType::NodeConnectivity => "Node Connectivity",
            MetricType::MessagePropagationDelay => "Message Propagation Delay",
            MetricType::BandwidthUsage => "Bandwidth Usage",
            MetricType::VoteSubmissionRate => "Vote Submission Rate",
            MetricType::ZkSnarkVerificationTime => "zk-SNARK Verification Time",
            MetricType::StakingActivity => "Staking Activity",
            MetricType::TokenTransferRate => "Token Transfer Rate",
            MetricType::VotingWeightUpdateRate => "Voting Weight Update Rate",
        }
        .to_string()
    }

    /// Updates monitoring configuration
    ///
    /// # Arguments
    /// * `new_config` - New configuration
    pub fn update_config(&mut self, new_config: MonitoringConfig) {
        self.config = new_config;
    }

    /// Gets current configuration
    ///
    /// # Returns
    /// Current configuration
    pub fn get_config(&self) -> &MonitoringConfig {
        &self.config
    }

    /// Clears all metrics and statistics
    pub fn clear_metrics(&self) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.clear();

        let mut stats = self.stats.lock().unwrap();
        stats.clear();

        let mut alerts = self.alerts.lock().unwrap();
        alerts.clear();

        let mut anomaly_state = self.anomaly_state.lock().unwrap();
        anomaly_state.clear();
    }

    /// Gets monitoring system health status
    ///
    /// # Returns
    /// Health status as a string
    pub fn get_health_status(&self) -> String {
        let metrics_count = {
            let metrics = self.metrics.lock().unwrap();
            metrics.values().map(|v| v.len()).sum::<usize>()
        };

        let alerts_count = {
            let alerts = self.alerts.lock().unwrap();
            alerts.iter().filter(|a| !a.acknowledged).count()
        };

        let uptime = self.get_uptime();

        format!(
            "Monitoring System Health:\n\
             Uptime: {} seconds\n\
             Metrics Collected: {}\n\
             Active Alerts: {}\n\
             Status: {}",
            uptime,
            metrics_count,
            alerts_count,
            if alerts_count == 0 {
                "Healthy"
            } else {
                "Issues Detected"
            }
        )
    }

    /// Gets the current integrity hash for testing purposes
    ///
    /// # Returns
    /// Current integrity hash as byte vector
    pub fn get_integrity_hash(&self) -> Vec<u8> {
        self.integrity_hash.lock().unwrap().clone()
    }
}

impl Default for MonitoringSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Integration functions for connecting monitoring with other modules
/// Collects PoS consensus metrics
///
/// # Arguments
/// * `monitor` - Monitoring system instance
/// * `validator_uptime` - Validator uptime percentage
/// * `finalization_time` - Block finalization time in milliseconds
/// * `slashing_events` - Number of slashing events
pub fn collect_pos_metrics(
    monitor: &MonitoringSystem,
    validator_uptime: f64,
    finalization_time: f64,
    slashing_events: f64,
) -> Result<(), String> {
    let mut metadata = HashMap::new();
    metadata.insert("module".to_string(), "pos_consensus".to_string());

    monitor.collect_metric(
        MetricType::ValidatorUptime,
        validator_uptime,
        "PoS",
        metadata.clone(),
    )?;
    monitor.collect_metric(
        MetricType::BlockFinalizationTime,
        finalization_time,
        "PoS",
        metadata.clone(),
    )?;
    monitor.collect_metric(MetricType::SlashingEvents, slashing_events, "PoS", metadata)?;

    Ok(())
}

/// Collects sharding metrics
///
/// # Arguments
/// * `monitor` - Monitoring system instance
/// * `throughput` - Shard throughput in TPS
/// * `latency` - Cross-shard latency in milliseconds
/// * `sync_success_rate` - State synchronization success rate
pub fn collect_sharding_metrics(
    monitor: &MonitoringSystem,
    throughput: f64,
    latency: f64,
    sync_success_rate: f64,
) -> Result<(), String> {
    let mut metadata = HashMap::new();
    metadata.insert("module".to_string(), "sharding".to_string());

    monitor.collect_metric(
        MetricType::ShardThroughput,
        throughput,
        "Sharding",
        metadata.clone(),
    )?;
    monitor.collect_metric(
        MetricType::CrossShardLatency,
        latency,
        "Sharding",
        metadata.clone(),
    )?;
    monitor.collect_metric(
        MetricType::StateSyncSuccessRate,
        sync_success_rate,
        "Sharding",
        metadata,
    )?;

    Ok(())
}

/// Collects P2P networking metrics
///
/// # Arguments
/// * `monitor` - Monitoring system instance
/// * `connectivity` - Node connectivity percentage
/// * `propagation_delay` - Message propagation delay in milliseconds
/// * `bandwidth` - Bandwidth usage in bytes per second
pub fn collect_p2p_metrics(
    monitor: &MonitoringSystem,
    connectivity: f64,
    propagation_delay: f64,
    bandwidth: f64,
) -> Result<(), String> {
    let mut metadata = HashMap::new();
    metadata.insert("module".to_string(), "p2p_networking".to_string());

    monitor.collect_metric(
        MetricType::NodeConnectivity,
        connectivity,
        "P2P",
        metadata.clone(),
    )?;
    monitor.collect_metric(
        MetricType::MessagePropagationDelay,
        propagation_delay,
        "P2P",
        metadata.clone(),
    )?;
    monitor.collect_metric(MetricType::BandwidthUsage, bandwidth, "P2P", metadata)?;

    Ok(())
}

/// Collects voting contract metrics
///
/// # Arguments
/// * `monitor` - Monitoring system instance
/// * `submission_rate` - Vote submission rate in votes per second
/// * `verification_time` - zk-SNARK verification time in milliseconds
pub fn collect_voting_metrics(
    monitor: &MonitoringSystem,
    submission_rate: f64,
    verification_time: f64,
) -> Result<(), String> {
    let mut metadata = HashMap::new();
    metadata.insert("module".to_string(), "voting_contract".to_string());

    monitor.collect_metric(
        MetricType::VoteSubmissionRate,
        submission_rate,
        "Voting",
        metadata.clone(),
    )?;
    monitor.collect_metric(
        MetricType::ZkSnarkVerificationTime,
        verification_time,
        "Voting",
        metadata,
    )?;

    Ok(())
}

/// Collects governance token metrics
///
/// # Arguments
/// * `monitor` - Monitoring system instance
/// * `staking_activity` - Staking activity in stakes per hour
/// * `transfer_rate` - Token transfer rate in transfers per second
/// * `weight_update_rate` - Voting weight update rate in updates per second
pub fn collect_governance_metrics(
    monitor: &MonitoringSystem,
    staking_activity: f64,
    transfer_rate: f64,
    weight_update_rate: f64,
) -> Result<(), String> {
    let mut metadata = HashMap::new();
    metadata.insert("module".to_string(), "governance_token".to_string());

    monitor.collect_metric(
        MetricType::StakingActivity,
        staking_activity,
        "Governance",
        metadata.clone(),
    )?;
    monitor.collect_metric(
        MetricType::TokenTransferRate,
        transfer_rate,
        "Governance",
        metadata.clone(),
    )?;
    monitor.collect_metric(
        MetricType::VotingWeightUpdateRate,
        weight_update_rate,
        "Governance",
        metadata,
    )?;

    Ok(())
}
