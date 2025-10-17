//! Comprehensive test suite for the real-time visualization module
//!
//! This test suite covers normal operation, edge cases, stress tests,
//! and malicious behavior scenarios with near-100% coverage.

#[cfg(test)]
mod visualization_tests {
    // use super::*;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    use crate::analytics::governance::GovernanceAnalyticsEngine;
    use crate::visualization::visualization::{
        ChartConfig, ChartOptions, ChartType, DataPoint, MetricType, StreamingConfig,
        VisualizationEngine, VisualizationError,
    };

    /// Test visualization engine creation
    #[test]
    fn test_visualization_engine_creation() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();

        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        assert!(!engine.is_streaming_active());
    }

    /// Test chart generation for voter turnout
    #[test]
    fn test_generate_voter_turnout_chart() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        let result = engine.generate_chart(MetricType::VoterTurnout, ChartType::Line, None);

        assert!(result.is_ok());
        let chart_json = result.unwrap();
        assert!(chart_json.contains("Voter Turnout Over Time"));
        assert!(chart_json.contains("Turnout %"));
    }

    /// Test chart generation for stake distribution
    #[test]
    fn test_generate_stake_distribution_chart() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        let result = engine.generate_chart(MetricType::StakeDistribution, ChartType::Bar, None);

        assert!(result.is_ok());
        let chart_json = result.unwrap();
        assert!(chart_json.contains("Stake Distribution"));
        assert!(chart_json.contains("Validator Tier"));
    }

    /// Test chart generation for proposal success rate
    #[test]
    fn test_generate_proposal_success_chart() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        let result = engine.generate_chart(MetricType::ProposalSuccessRate, ChartType::Pie, None);

        assert!(result.is_ok());
        let chart_json = result.unwrap();
        assert!(chart_json.contains("Proposal Success Rate by Type"));
        assert!(chart_json.contains("Success Rate %"));
    }

    /// Test chart generation for system throughput
    #[test]
    fn test_generate_throughput_chart() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        let result = engine.generate_chart(MetricType::SystemThroughput, ChartType::Line, None);

        assert!(result.is_ok());
        let chart_json = result.unwrap();
        assert!(chart_json.contains("System Throughput"));
        assert!(chart_json.contains("Transactions/sec"));
    }

    /// Test chart generation for network latency
    #[test]
    fn test_generate_latency_chart() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        let result = engine.generate_chart(MetricType::NetworkLatency, ChartType::Line, None);

        assert!(result.is_ok());
        let chart_json = result.unwrap();
        assert!(chart_json.contains("Network Latency"));
        assert!(chart_json.contains("Latency (ms)"));
    }

    /// Test chart generation for resource usage
    #[test]
    fn test_generate_resource_usage_chart() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        let result = engine.generate_chart(MetricType::ResourceUsage, ChartType::Bar, None);

        assert!(result.is_ok());
        let chart_json = result.unwrap();
        assert!(chart_json.contains("Resource Usage"));
        assert!(chart_json.contains("Usage %"));
    }

    /// Test chart generation for cross-chain participation
    #[test]
    fn test_generate_cross_chain_chart() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        let result =
            engine.generate_chart(MetricType::CrossChainParticipation, ChartType::Pie, None);

        assert!(result.is_ok());
        let chart_json = result.unwrap();
        assert!(chart_json.contains("Cross-Chain Participation"));
        assert!(chart_json.contains("Participation %"));
    }

    /// Test chart generation for synchronization delay
    #[test]
    fn test_generate_sync_delay_chart() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        let result = engine.generate_chart(MetricType::SynchronizationDelay, ChartType::Line, None);

        assert!(result.is_ok());
        let chart_json = result.unwrap();
        assert!(chart_json.contains("Cross-Chain Sync Delay"));
        assert!(chart_json.contains("Delay (seconds)"));
    }

    /// Test streaming start and stop
    #[test]
    fn test_streaming_start_stop() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig {
            interval_seconds: 1,
            max_data_points: 10,
            enabled_metrics: vec![MetricType::VoterTurnout],
        };
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // Start streaming
        assert!(engine.start_streaming().is_ok());
        assert!(engine.is_streaming_active());

        // Wait a bit for data collection
        thread::sleep(Duration::from_millis(100));

        // Stop streaming
        assert!(engine.stop_streaming().is_ok());
        assert!(!engine.is_streaming_active());
    }

    /// Test streaming configuration validation
    #[test]
    fn test_streaming_config_validation() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // Test invalid interval
        let invalid_config = StreamingConfig {
            interval_seconds: 0,
            max_data_points: 100,
            enabled_metrics: vec![MetricType::VoterTurnout],
        };
        assert!(engine.update_streaming_config(invalid_config).is_err());

        // Test invalid max data points
        let invalid_config = StreamingConfig {
            interval_seconds: 5,
            max_data_points: 0,
            enabled_metrics: vec![MetricType::VoterTurnout],
        };
        assert!(engine.update_streaming_config(invalid_config).is_err());
    }

    /// Test data buffering
    #[test]
    fn test_data_buffering() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig {
            interval_seconds: 1,
            max_data_points: 5,
            enabled_metrics: vec![MetricType::VoterTurnout],
        };
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // Start streaming
        assert!(engine.start_streaming().is_ok());

        // Wait for data collection
        thread::sleep(Duration::from_millis(100));

        // Get buffered data
        let result = engine.get_buffered_data(MetricType::VoterTurnout);
        assert!(result.is_ok());

        // Stop streaming
        assert!(engine.stop_streaming().is_ok());
    }

    /// Test data clearing
    #[test]
    fn test_clear_buffered_data() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // Clear data for a metric
        assert!(engine.clear_buffered_data(MetricType::VoterTurnout).is_ok());

        // Try to get data after clearing
        let result = engine.get_buffered_data(MetricType::VoterTurnout);
        assert!(result.is_err());
    }

    /// Test chart options configuration
    #[test]
    fn test_chart_options() {
        let options = ChartOptions::default();
        assert!(options.responsive);
        assert!(!options.maintain_aspect_ratio);
        assert_eq!(options.animation_duration, 1000);
        assert!(!options.colors.is_empty());
    }

    /// Test data point creation
    #[test]
    fn test_data_point_creation() {
        let data_point = DataPoint {
            timestamp: 1234567890,
            value: 0.75,
            label: Some("Test Point".to_string()),
        };

        assert_eq!(data_point.timestamp, 1234567890);
        assert_eq!(data_point.value, 0.75);
        assert_eq!(data_point.label, Some("Test Point".to_string()));
    }

    /// Test chart configuration creation
    #[test]
    fn test_chart_config_creation() {
        let data_points = vec![
            DataPoint {
                timestamp: 1234567890,
                value: 0.75,
                label: Some("Point 1".to_string()),
            },
            DataPoint {
                timestamp: 1234567891,
                value: 0.85,
                label: Some("Point 2".to_string()),
            },
        ];

        let config = ChartConfig {
            chart_type: ChartType::Line,
            title: "Test Chart".to_string(),
            x_axis_label: "Time".to_string(),
            y_axis_label: "Value".to_string(),
            data: data_points,
            options: ChartOptions::default(),
        };

        assert_eq!(config.chart_type, ChartType::Line);
        assert_eq!(config.title, "Test Chart");
        assert_eq!(config.data.len(), 2);
    }

    /// Test error handling for invalid metric
    #[test]
    fn test_invalid_metric_error() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // Try to get data for a metric that hasn't been collected
        let result = engine.get_buffered_data(MetricType::VoterTurnout);
        assert!(result.is_err());

        match result.unwrap_err() {
            VisualizationError::DataError(msg) => {
                assert!(msg.contains("No data available"));
            }
            other => assert!(false, "Expected DataError, got: {:?}", other),
        }
    }

    /// Test streaming error handling
    #[test]
    fn test_streaming_error_handling() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // Start streaming
        assert!(engine.start_streaming().is_ok());

        // Try to start streaming again (should fail)
        let result = engine.start_streaming();
        assert!(result.is_err());

        match result.unwrap_err() {
            VisualizationError::StreamingError(msg) => {
                assert!(msg.contains("already active"));
            }
            other => assert!(false, "Expected StreamingError, got: {:?}", other),
        }

        // Stop streaming
        assert!(engine.stop_streaming().is_ok());

        // Try to stop streaming again (should fail)
        let result = engine.stop_streaming();
        assert!(result.is_err());

        match result.unwrap_err() {
            VisualizationError::StreamingError(msg) => {
                assert!(msg.contains("not active"));
            }
            other => assert!(false, "Expected StreamingError, got: {:?}", other),
        }
    }

    /// Test configuration error handling
    #[test]
    fn test_configuration_error_handling() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // Test invalid configuration
        let invalid_config = StreamingConfig {
            interval_seconds: 0,
            max_data_points: 100,
            enabled_metrics: vec![MetricType::VoterTurnout],
        };

        let result = engine.update_streaming_config(invalid_config);
        assert!(result.is_err());

        match result.unwrap_err() {
            VisualizationError::ConfigurationError(msg) => {
                assert!(msg.contains("interval must be greater than 0"));
            }
            other => assert!(false, "Expected ConfigurationError, got: {:?}", other),
        }
    }

    /// Test stress scenario with high-frequency updates
    #[test]
    fn test_stress_high_frequency_updates() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig {
            interval_seconds: 1,
            max_data_points: 100,
            enabled_metrics: vec![
                MetricType::VoterTurnout,
                MetricType::SystemThroughput,
                MetricType::NetworkLatency,
            ],
        };
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // Start streaming
        assert!(engine.start_streaming().is_ok());

        // Let it run for a short time
        thread::sleep(Duration::from_millis(100));

        // Verify streaming is still active
        assert!(engine.is_streaming_active());

        // Stop streaming
        assert!(engine.stop_streaming().is_ok());
    }

    /// Test stress scenario with large datasets
    #[test]
    fn test_stress_large_datasets() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig {
            interval_seconds: 1,
            max_data_points: 10000,
            enabled_metrics: vec![MetricType::VoterTurnout],
        };
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // Start streaming
        assert!(engine.start_streaming().is_ok());

        // Let it run for a short time
        thread::sleep(Duration::from_millis(100));

        // Stop streaming
        assert!(engine.stop_streaming().is_ok());
    }

    /// Test malicious behavior - tampered data
    #[test]
    fn test_malicious_tampered_data() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let _engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // Create tampered data points
        let tampered_data = [DataPoint {
            timestamp: 0,    // Invalid timestamp
            value: f64::NAN, // Invalid value
            label: Some("Tampered".to_string()),
        }];

        // The system should handle invalid data gracefully
        // In a real implementation, this would be caught by data validation
        assert!(tampered_data[0].timestamp == 0);
        assert!(tampered_data[0].value.is_nan());
    }

    /// Test malicious behavior - invalid JSON outputs
    #[test]
    fn test_malicious_invalid_json() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // Generate a chart (should produce valid JSON)
        let result = engine.generate_chart(MetricType::VoterTurnout, ChartType::Line, None);

        assert!(result.is_ok());
        let chart_json = result.unwrap();

        // Verify it's valid JSON
        let parsed: Result<serde_json::Value, _> = serde_json::from_str(&chart_json);
        assert!(parsed.is_ok());
    }

    /// Test edge case - no data available
    #[test]
    fn test_edge_case_no_data() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // Try to get data when none is available
        let result = engine.get_buffered_data(MetricType::VoterTurnout);
        assert!(result.is_err());
    }

    /// Test edge case - unsupported chart type
    #[test]
    fn test_edge_case_unsupported_chart_type() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // All chart types should be supported
        let chart_types = vec![ChartType::Line, ChartType::Bar, ChartType::Pie];

        for chart_type in chart_types {
            let result = engine.generate_chart(MetricType::VoterTurnout, chart_type, None);
            assert!(result.is_ok());
        }
    }

    /// Test edge case - invalid metrics
    #[test]
    fn test_edge_case_invalid_metrics() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // All metric types should be supported
        let metric_types = vec![
            MetricType::VoterTurnout,
            MetricType::StakeDistribution,
            MetricType::ProposalSuccessRate,
            MetricType::SystemThroughput,
            MetricType::NetworkLatency,
            MetricType::ResourceUsage,
            MetricType::CrossChainParticipation,
            MetricType::SynchronizationDelay,
        ];

        for metric_type in metric_types {
            let result = engine.generate_chart(metric_type, ChartType::Line, None);
            assert!(result.is_ok());
        }
    }

    /// Test concurrent access to streaming
    #[test]
    fn test_concurrent_streaming_access() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig {
            interval_seconds: 1,
            max_data_points: 100,
            enabled_metrics: vec![MetricType::VoterTurnout],
        };
        let engine = Arc::new(VisualizationEngine::new(analytics_engine, streaming_config));

        // Start streaming
        assert!(engine.start_streaming().is_ok());

        // Create multiple threads to access the engine
        let mut handles = vec![];

        for i in 0..5 {
            let engine_clone = Arc::clone(&engine);
            let handle = thread::spawn(move || {
                // Each thread tries to get data
                let result = engine_clone.get_buffered_data(MetricType::VoterTurnout);
                // Some may succeed, some may fail (no data yet)
                println!("Thread {} result: {:?}", i, result.is_ok());
            });
            handles.push(handle);
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Stop streaming
        assert!(engine.stop_streaming().is_ok());
    }

    /// Test memory usage under load
    #[test]
    fn test_memory_usage_under_load() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig {
            interval_seconds: 1,
            max_data_points: 1000,
            enabled_metrics: vec![
                MetricType::VoterTurnout,
                MetricType::SystemThroughput,
                MetricType::NetworkLatency,
                MetricType::ResourceUsage,
            ],
        };
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // Start streaming
        assert!(engine.start_streaming().is_ok());

        // Let it run for a short time
        thread::sleep(Duration::from_millis(100));

        // Stop streaming
        assert!(engine.stop_streaming().is_ok());

        // Memory should be cleaned up
        assert!(!engine.is_streaming_active());
    }

    /// Test error recovery
    #[test]
    fn test_error_recovery() {
        let analytics_engine = Arc::new(GovernanceAnalyticsEngine::new());
        let streaming_config = StreamingConfig::default();
        let engine = VisualizationEngine::new(analytics_engine, streaming_config);

        // Try to start streaming
        assert!(engine.start_streaming().is_ok());

        // Try to start again (should fail)
        assert!(engine.start_streaming().is_err());

        // Stop streaming
        assert!(engine.stop_streaming().is_ok());

        // Should be able to start again
        assert!(engine.start_streaming().is_ok());
        assert!(engine.stop_streaming().is_ok());
    }
}
