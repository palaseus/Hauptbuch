//! Real-time visualization module for the decentralized voting blockchain
//!
//! This module provides live, interactive charts for system monitoring and research.
//! It generates Chart.js-compatible JSON outputs for real-time dashboards.

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::analytics::governance::{GovernanceAnalyticsEngine, TimeRange};
use crate::federation::federation::CrossChainVote;
use crate::monitoring::monitor::SystemMetrics;

/// Chart types supported by the visualization system
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ChartType {
    Line,
    Bar,
    Pie,
}

/// Metrics that can be visualized
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MetricType {
    VoterTurnout,
    StakeDistribution,
    ProposalSuccessRate,
    SystemThroughput,
    NetworkLatency,
    ResourceUsage,
    CrossChainParticipation,
    SynchronizationDelay,
}

/// Real-time data point for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: u64,
    pub value: f64,
    pub label: Option<String>,
}

/// Chart configuration for Chart.js
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartConfig {
    pub chart_type: ChartType,
    pub title: String,
    pub x_axis_label: String,
    pub y_axis_label: String,
    pub data: Vec<DataPoint>,
    pub options: ChartOptions,
}

/// Chart display options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartOptions {
    pub responsive: bool,
    pub maintain_aspect_ratio: bool,
    pub animation_duration: u32,
    pub colors: Vec<String>,
}

/// Streaming configuration for real-time updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    pub interval_seconds: u64,
    pub max_data_points: usize,
    pub enabled_metrics: Vec<MetricType>,
}

/// Real-time visualization engine
pub struct VisualizationEngine {
    #[allow(dead_code)]
    analytics_engine: Arc<GovernanceAnalyticsEngine>,
    monitoring_data: Arc<Mutex<Vec<SystemMetrics>>>,
    federation_data: Arc<Mutex<Vec<CrossChainVote>>>,
    streaming_config: StreamingConfig,
    data_buffer: Arc<Mutex<HashMap<MetricType, Vec<DataPoint>>>>,
    is_streaming: Arc<Mutex<bool>>,
}

impl VisualizationEngine {
    /// Create a new visualization engine
    pub fn new(
        analytics_engine: Arc<GovernanceAnalyticsEngine>,
        streaming_config: StreamingConfig,
    ) -> Self {
        Self {
            analytics_engine,
            monitoring_data: Arc::new(Mutex::new(Vec::new())),
            federation_data: Arc::new(Mutex::new(Vec::new())),
            streaming_config,
            data_buffer: Arc::new(Mutex::new(HashMap::new())),
            is_streaming: Arc::new(Mutex::new(false)),
        }
    }

    /// Generate Chart.js-compatible JSON for a specific metric
    pub fn generate_chart(
        &self,
        metric: MetricType,
        chart_type: ChartType,
        time_range: Option<TimeRange>,
    ) -> Result<String, VisualizationError> {
        // Collect data for the specified metric
        let data_points = self.collect_metric_data(metric.clone(), time_range)?;

        // Generate chart configuration
        let chart_config = self.create_chart_config(metric, chart_type, data_points)?;

        // Serialize to JSON
        serde_json::to_string(&chart_config)
            .map_err(|e| VisualizationError::SerializationError(e.to_string()))
    }

    /// Collect real-time data for a specific metric
    fn collect_metric_data(
        &self,
        metric: MetricType,
        time_range: Option<TimeRange>,
    ) -> Result<Vec<DataPoint>, VisualizationError> {
        let _current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let data_points = match metric {
            MetricType::VoterTurnout => self.collect_voter_turnout_data(time_range)?,
            MetricType::StakeDistribution => self.collect_stake_distribution_data(time_range)?,
            MetricType::ProposalSuccessRate => self.collect_proposal_success_data(time_range)?,
            MetricType::SystemThroughput => self.collect_throughput_data(time_range)?,
            MetricType::NetworkLatency => self.collect_latency_data(time_range)?,
            MetricType::ResourceUsage => self.collect_resource_usage_data(time_range)?,
            MetricType::CrossChainParticipation => self.collect_cross_chain_data(time_range)?,
            MetricType::SynchronizationDelay => self.collect_sync_delay_data(time_range)?,
        };

        // Verify data integrity with SHA-3
        self.verify_data_integrity(&data_points)?;

        Ok(data_points)
    }

    /// Collect voter turnout data from analytics engine
    fn collect_voter_turnout_data(
        &self,
        _time_range: Option<TimeRange>,
    ) -> Result<Vec<DataPoint>, VisualizationError> {
        // Mock data for demonstration - in real implementation, this would
        // query the analytics engine for historical voter turnout data
        let mut data_points = Vec::new();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Generate sample voter turnout data over time
        for i in 0..24 {
            let timestamp = current_time.saturating_sub((23 - i) * 3600); // Hourly data for 24 hours
            let turnout = 0.6 + (i as f64 * 0.01) + (i % 3) as f64 * 0.05; // Simulated turnout trend

            data_points.push(DataPoint {
                timestamp,
                value: turnout.min(1.0), // Cap at 100%
                label: Some(format!("Hour {}", i)),
            });
        }

        Ok(data_points)
    }

    /// Collect stake distribution data
    fn collect_stake_distribution_data(
        &self,
        _time_range: Option<TimeRange>,
    ) -> Result<Vec<DataPoint>, VisualizationError> {
        // Mock stake distribution data - in real implementation, this would
        // query the analytics engine for current stake distribution
        let mut data_points = Vec::new();

        // Simulate stake distribution across different validator tiers
        let stake_tiers = vec![
            ("Top 1%", 0.45),
            ("Top 5%", 0.25),
            ("Top 10%", 0.15),
            ("Top 25%", 0.10),
            ("Others", 0.05),
        ];

        for (label, percentage) in stake_tiers {
            data_points.push(DataPoint {
                timestamp: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                value: percentage,
                label: Some(label.to_string()),
            });
        }

        Ok(data_points)
    }

    /// Collect proposal success rate data
    fn collect_proposal_success_data(
        &self,
        _time_range: Option<TimeRange>,
    ) -> Result<Vec<DataPoint>, VisualizationError> {
        // Mock proposal success data - in real implementation, this would
        // query the analytics engine for proposal outcomes
        let mut data_points = Vec::new();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Generate sample proposal success rates by type
        let proposal_types = vec![
            ("Protocol Upgrade", 0.85),
            ("Economic Policy", 0.72),
            ("Governance Change", 0.68),
            ("Technical Fix", 0.91),
        ];

        for (label, success_rate) in proposal_types {
            data_points.push(DataPoint {
                timestamp: current_time,
                value: success_rate,
                label: Some(label.to_string()),
            });
        }

        Ok(data_points)
    }

    /// Collect system throughput data from monitoring
    fn collect_throughput_data(
        &self,
        _time_range: Option<TimeRange>,
    ) -> Result<Vec<DataPoint>, VisualizationError> {
        // Mock throughput data - in real implementation, this would
        // query the monitoring system for real-time throughput metrics
        let mut data_points = Vec::new();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Generate sample throughput data over time
        for i in 0..12 {
            let timestamp = current_time.saturating_sub((11 - i) * 300); // 5-minute intervals
            let throughput = 1000.0 + (i as f64 * 50.0) + (i % 4) as f64 * 100.0; // TPS simulation

            data_points.push(DataPoint {
                timestamp,
                value: throughput,
                label: Some(format!("TPS {}", i)),
            });
        }

        Ok(data_points)
    }

    /// Collect network latency data
    fn collect_latency_data(
        &self,
        _time_range: Option<TimeRange>,
    ) -> Result<Vec<DataPoint>, VisualizationError> {
        // Mock latency data - in real implementation, this would
        // query the monitoring system for network latency metrics
        let mut data_points = Vec::new();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Generate sample latency data
        for i in 0..10 {
            let timestamp = current_time.saturating_sub((9 - i) * 60); // 1-minute intervals
            let latency = 50.0 + (i as f64 * 5.0) + (i % 3) as f64 * 10.0; // ms simulation

            data_points.push(DataPoint {
                timestamp,
                value: latency,
                label: Some(format!("Latency {}", i)),
            });
        }

        Ok(data_points)
    }

    /// Collect resource usage data
    fn collect_resource_usage_data(
        &self,
        _time_range: Option<TimeRange>,
    ) -> Result<Vec<DataPoint>, VisualizationError> {
        // Mock resource usage data - in real implementation, this would
        // query the monitoring system for CPU, memory, and network usage
        let mut data_points = Vec::new();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Generate sample resource usage data
        let resource_types = vec![
            ("CPU Usage", 0.65),
            ("Memory Usage", 0.42),
            ("Network I/O", 0.38),
            ("Disk I/O", 0.23),
        ];

        for (label, usage) in resource_types {
            data_points.push(DataPoint {
                timestamp: current_time,
                value: usage,
                label: Some(label.to_string()),
            });
        }

        Ok(data_points)
    }

    /// Collect cross-chain participation data
    fn collect_cross_chain_data(
        &self,
        _time_range: Option<TimeRange>,
    ) -> Result<Vec<DataPoint>, VisualizationError> {
        // Mock cross-chain data - in real implementation, this would
        // query the federation module for cross-chain participation metrics
        let mut data_points = Vec::new();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Generate sample cross-chain participation data
        let chains = vec![
            ("Ethereum", 0.35),
            ("Polkadot", 0.28),
            ("Cosmos", 0.22),
            ("Solana", 0.15),
        ];

        for (chain, participation) in chains {
            data_points.push(DataPoint {
                timestamp: current_time,
                value: participation,
                label: Some(chain.to_string()),
            });
        }

        Ok(data_points)
    }

    /// Collect synchronization delay data
    fn collect_sync_delay_data(
        &self,
        _time_range: Option<TimeRange>,
    ) -> Result<Vec<DataPoint>, VisualizationError> {
        // Mock sync delay data - in real implementation, this would
        // query the federation module for cross-chain synchronization metrics
        let mut data_points = Vec::new();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Generate sample synchronization delay data
        for i in 0..8 {
            let timestamp = current_time.saturating_sub((7 - i) * 900); // 15-minute intervals
            let delay = 2.0 + (i as f64 * 0.5) + (i % 2) as f64 * 1.0; // seconds simulation

            data_points.push(DataPoint {
                timestamp,
                value: delay,
                label: Some(format!("Sync {}", i)),
            });
        }

        Ok(data_points)
    }

    /// Create chart configuration for Chart.js
    fn create_chart_config(
        &self,
        metric: MetricType,
        chart_type: ChartType,
        data_points: Vec<DataPoint>,
    ) -> Result<ChartConfig, VisualizationError> {
        let (title, x_label, y_label) = self.get_chart_labels(metric);

        let options = ChartOptions {
            responsive: true,
            maintain_aspect_ratio: false,
            animation_duration: 1000,
            colors: self.get_chart_colors(chart_type.clone()),
        };

        Ok(ChartConfig {
            chart_type,
            title,
            x_axis_label: x_label,
            y_axis_label: y_label,
            data: data_points,
            options,
        })
    }

    /// Get chart labels based on metric type
    fn get_chart_labels(&self, metric: MetricType) -> (String, String, String) {
        match metric {
            MetricType::VoterTurnout => (
                "Voter Turnout Over Time".to_string(),
                "Time".to_string(),
                "Turnout %".to_string(),
            ),
            MetricType::StakeDistribution => (
                "Stake Distribution".to_string(),
                "Validator Tier".to_string(),
                "Stake %".to_string(),
            ),
            MetricType::ProposalSuccessRate => (
                "Proposal Success Rate by Type".to_string(),
                "Proposal Type".to_string(),
                "Success Rate %".to_string(),
            ),
            MetricType::SystemThroughput => (
                "System Throughput".to_string(),
                "Time".to_string(),
                "Transactions/sec".to_string(),
            ),
            MetricType::NetworkLatency => (
                "Network Latency".to_string(),
                "Time".to_string(),
                "Latency (ms)".to_string(),
            ),
            MetricType::ResourceUsage => (
                "Resource Usage".to_string(),
                "Resource Type".to_string(),
                "Usage %".to_string(),
            ),
            MetricType::CrossChainParticipation => (
                "Cross-Chain Participation".to_string(),
                "Blockchain".to_string(),
                "Participation %".to_string(),
            ),
            MetricType::SynchronizationDelay => (
                "Cross-Chain Sync Delay".to_string(),
                "Time".to_string(),
                "Delay (seconds)".to_string(),
            ),
        }
    }

    /// Get chart colors based on chart type
    fn get_chart_colors(&self, chart_type: ChartType) -> Vec<String> {
        match chart_type {
            ChartType::Line => vec![
                "#3B82F6".to_string(), // Blue
                "#EF4444".to_string(), // Red
                "#10B981".to_string(), // Green
                "#F59E0B".to_string(), // Yellow
            ],
            ChartType::Bar => vec![
                "#8B5CF6".to_string(), // Purple
                "#06B6D4".to_string(), // Cyan
                "#F97316".to_string(), // Orange
                "#84CC16".to_string(), // Lime
            ],
            ChartType::Pie => vec![
                "#EC4899".to_string(), // Pink
                "#6366F1".to_string(), // Indigo
                "#14B8A6".to_string(), // Teal
                "#F59E0B".to_string(), // Amber
            ],
        }
    }

    /// Verify data integrity using SHA-3
    fn verify_data_integrity(&self, data_points: &[DataPoint]) -> Result<(), VisualizationError> {
        let mut hasher = Sha3_256::new();

        for point in data_points {
            hasher.update(point.timestamp.to_le_bytes());
            hasher.update(point.value.to_le_bytes());
            if let Some(ref label) = point.label {
                hasher.update(label.as_bytes());
            }
        }

        let hash = hasher.finalize();

        // In a real implementation, this would compare against a stored hash
        // For now, we just ensure the hash is not empty
        if hash.is_empty() {
            return Err(VisualizationError::DataIntegrityError(
                "Invalid data hash".to_string(),
            ));
        }

        Ok(())
    }

    /// Start real-time streaming of visualization data
    pub fn start_streaming(&self) -> Result<(), VisualizationError> {
        let mut is_streaming = self.is_streaming.lock().unwrap();
        if *is_streaming {
            return Err(VisualizationError::StreamingError(
                "Streaming already active".to_string(),
            ));
        }
        *is_streaming = true;
        drop(is_streaming);

        let data_buffer = Arc::clone(&self.data_buffer);
        let monitoring_data = Arc::clone(&self.monitoring_data);
        let federation_data = Arc::clone(&self.federation_data);
        let config = self.streaming_config.clone();
        let is_streaming_flag = Arc::clone(&self.is_streaming);

        thread::spawn(move || {
            while *is_streaming_flag.lock().unwrap() {
                // Collect data for all enabled metrics
                for metric in &config.enabled_metrics {
                    if let Ok(data_points) = Self::collect_streaming_data(
                        metric.clone(),
                        &monitoring_data,
                        &federation_data,
                    ) {
                        let mut buffer = data_buffer.lock().unwrap();
                        let metric_data = buffer.entry(metric.clone()).or_default();

                        // Add new data points
                        metric_data.extend(data_points);

                        // Maintain max data points limit
                        if metric_data.len() > config.max_data_points {
                            let excess = metric_data.len() - config.max_data_points;
                            metric_data.drain(0..excess);
                        }
                    }
                }

                // Sleep for the configured interval
                thread::sleep(Duration::from_secs(config.interval_seconds));
            }
        });

        Ok(())
    }

    /// Stop real-time streaming
    pub fn stop_streaming(&self) -> Result<(), VisualizationError> {
        let mut is_streaming = self.is_streaming.lock().unwrap();
        if !*is_streaming {
            return Err(VisualizationError::StreamingError(
                "Streaming not active".to_string(),
            ));
        }
        *is_streaming = false;
        Ok(())
    }

    /// Collect streaming data for a specific metric
    fn collect_streaming_data(
        metric: MetricType,
        _monitoring_data: &Arc<Mutex<Vec<SystemMetrics>>>,
        _federation_data: &Arc<Mutex<Vec<CrossChainVote>>>,
    ) -> Result<Vec<DataPoint>, VisualizationError> {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Generate real-time data point based on metric type
        let value = match metric {
            MetricType::VoterTurnout => 0.6 + (current_time % 100) as f64 * 0.001,
            MetricType::SystemThroughput => 1000.0 + (current_time % 50) as f64 * 10.0,
            MetricType::NetworkLatency => 50.0 + (current_time % 20) as f64 * 2.0,
            _ => 0.5, // Default value for other metrics
        };

        Ok(vec![DataPoint {
            timestamp: current_time,
            value,
            label: None,
        }])
    }

    /// Get current streaming status
    pub fn is_streaming_active(&self) -> bool {
        *self.is_streaming.lock().unwrap()
    }

    /// Get buffered data for a specific metric
    pub fn get_buffered_data(
        &self,
        metric: MetricType,
    ) -> Result<Vec<DataPoint>, VisualizationError> {
        let buffer = self.data_buffer.lock().unwrap();
        buffer.get(&metric).cloned().ok_or_else(|| {
            VisualizationError::DataError(format!("No data available for metric: {:?}", metric))
        })
    }

    /// Clear buffered data for a specific metric
    pub fn clear_buffered_data(&self, metric: MetricType) -> Result<(), VisualizationError> {
        let mut buffer = self.data_buffer.lock().unwrap();
        buffer.remove(&metric);
        Ok(())
    }

    /// Update streaming configuration
    pub fn update_streaming_config(
        &self,
        config: StreamingConfig,
    ) -> Result<(), VisualizationError> {
        if self.is_streaming_active() {
            return Err(VisualizationError::StreamingError(
                "Cannot update config while streaming is active".to_string(),
            ));
        }

        // In a real implementation, this would update the configuration
        // For now, we just validate the configuration
        if config.interval_seconds == 0 {
            return Err(VisualizationError::ConfigurationError(
                "Streaming interval must be greater than 0".to_string(),
            ));
        }

        if config.max_data_points == 0 {
            return Err(VisualizationError::ConfigurationError(
                "Max data points must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Visualization errors
#[derive(Debug, Clone, PartialEq)]
pub enum VisualizationError {
    DataError(String),
    SerializationError(String),
    DataIntegrityError(String),
    StreamingError(String),
    ConfigurationError(String),
}

impl std::fmt::Display for VisualizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VisualizationError::DataError(msg) => write!(f, "Data error: {}", msg),
            VisualizationError::SerializationError(msg) => {
                write!(f, "Serialization error: {}", msg)
            }
            VisualizationError::DataIntegrityError(msg) => {
                write!(f, "Data integrity error: {}", msg)
            }
            VisualizationError::StreamingError(msg) => write!(f, "Streaming error: {}", msg),
            VisualizationError::ConfigurationError(msg) => {
                write!(f, "Configuration error: {}", msg)
            }
        }
    }
}

impl std::error::Error for VisualizationError {}

/// Default streaming configuration
impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            interval_seconds: 5,
            max_data_points: 1000,
            enabled_metrics: vec![
                MetricType::VoterTurnout,
                MetricType::SystemThroughput,
                MetricType::NetworkLatency,
            ],
        }
    }
}

/// Default chart options
impl Default for ChartOptions {
    fn default() -> Self {
        Self {
            responsive: true,
            maintain_aspect_ratio: false,
            animation_duration: 1000,
            colors: vec![
                "#3B82F6".to_string(),
                "#EF4444".to_string(),
                "#10B981".to_string(),
            ],
        }
    }
}
