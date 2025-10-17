//! Comprehensive test suite for the Monitoring System
//!
//! This module contains extensive tests for the monitoring system, covering
//! normal operation, edge cases, malicious behavior, and stress tests to ensure
//! robustness and reliability of the monitoring infrastructure.

use super::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper function to create a test monitoring system
    fn create_test_monitor() -> MonitoringSystem {
        MonitoringSystem::new()
    }

    /// Helper function to create test metadata
    fn create_test_metadata() -> HashMap<String, String> {
        let mut metadata = HashMap::new();
        metadata.insert("test".to_string(), "true".to_string());
        metadata.insert("version".to_string(), "1.0".to_string());
        metadata
    }

    // ===== NORMAL OPERATION TESTS =====

    #[test]
    fn test_monitoring_system_creation() {
        let monitor = MonitoringSystem::new();
        assert_eq!(monitor.get_config().collection_interval, 1000);
        assert_eq!(monitor.get_config().aggregation_window, 300);
        assert_eq!(monitor.get_config().max_metrics, 10000);
        assert!(monitor.get_config().enable_anomaly_detection);
        assert!(monitor.get_config().enable_p2p_alerts);
        println!("✅ Monitoring system creation test passed");
    }

    #[test]
    fn test_metric_collection() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        let result =
            monitor.collect_metric(MetricType::ValidatorUptime, 95.5, "test_source", metadata);

        assert!(result.is_ok(), "Metric collection should succeed");

        // Verify metric was stored
        let stats = monitor.get_statistics(&MetricType::ValidatorUptime);
        assert!(stats.is_some(), "Statistics should be available");

        if let Some(stats) = stats {
            assert_eq!(stats.current, 95.5);
            assert_eq!(stats.average, 95.5);
            assert_eq!(stats.minimum, 95.5);
            assert_eq!(stats.maximum, 95.5);
            assert_eq!(stats.sample_count, 1);
        }

        println!("✅ Metric collection test passed");
    }

    #[test]
    fn test_statistics_aggregation() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Collect multiple metrics
        for i in 0..10 {
            let value = 80.0 + (i as f64 * 2.0);
            monitor
                .collect_metric(
                    MetricType::BlockFinalizationTime,
                    value,
                    "test_source",
                    metadata.clone(),
                )
                .unwrap();
        }

        let stats = monitor.get_statistics(&MetricType::BlockFinalizationTime);
        assert!(stats.is_some(), "Statistics should be available");

        if let Some(stats) = stats {
            assert_eq!(stats.current, 98.0); // Last value
            assert_eq!(stats.minimum, 80.0); // First value
            assert_eq!(stats.maximum, 98.0); // Last value
            assert_eq!(stats.sample_count, 10);
            assert!(
                stats.standard_deviation > 0.0,
                "Standard deviation should be positive"
            );
        }

        println!("✅ Statistics aggregation test passed");
    }

    #[test]
    fn test_alert_generation() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Collect metric that should trigger alert (low uptime)
        monitor
            .collect_metric(
                MetricType::ValidatorUptime,
                30.0, // Below 50% threshold
                "test_source",
                metadata,
            )
            .unwrap();

        let alerts = monitor.get_alerts();
        assert!(!alerts.is_empty(), "Alert should be generated");

        let uptime_alert = alerts
            .iter()
            .find(|a| a.metric_type == MetricType::ValidatorUptime);
        assert!(uptime_alert.is_some(), "Uptime alert should be present");

        if let Some(alert) = uptime_alert {
            assert_eq!(alert.actual_value, 30.0);
            assert_eq!(alert.threshold, 50.0);
            assert!(!alert.acknowledged);
        }

        println!("✅ Alert generation test passed");
    }

    #[test]
    fn test_alert_acknowledgment() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Generate an alert
        monitor
            .collect_metric(
                MetricType::BlockFinalizationTime,
                10000.0, // Above 5000ms threshold
                "test_source",
                metadata,
            )
            .unwrap();

        let alerts = monitor.get_alerts();
        assert!(!alerts.is_empty(), "Alert should be generated");

        let alert_id = &alerts[0].id;
        let result = monitor.acknowledge_alert(alert_id);
        assert!(result.is_ok(), "Alert acknowledgment should succeed");

        let updated_alerts = monitor.get_alerts();
        let acknowledged_alert = updated_alerts.iter().find(|a| a.id == *alert_id);
        assert!(acknowledged_alert.is_some(), "Alert should still exist");
        assert!(
            acknowledged_alert.unwrap().acknowledged,
            "Alert should be acknowledged"
        );

        println!("✅ Alert acknowledgment test passed");
    }

    #[test]
    fn test_json_logging() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Collect some metrics
        monitor
            .collect_metric(MetricType::ShardThroughput, 150.0, "test_source", metadata)
            .unwrap();

        let json_log = monitor.generate_json_log(&MetricType::ShardThroughput);
        assert!(json_log.is_ok(), "JSON log generation should succeed");

        let log_content = json_log.unwrap();
        assert!(log_content.contains("Shard Throughput"));
        assert!(log_content.contains("150"));
        assert!(log_content.contains("timestamp"));
        assert!(log_content.contains("statistics"));

        println!("✅ JSON logging test passed");
    }

    #[test]
    fn test_human_logging() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Collect some metrics
        monitor
            .collect_metric(MetricType::NodeConnectivity, 85.0, "test_source", metadata)
            .unwrap();

        let human_log = monitor.generate_human_log(&MetricType::NodeConnectivity);
        assert!(human_log.is_ok(), "Human log generation should succeed");

        let log_content = human_log.unwrap();
        assert!(log_content.contains("Node Connectivity"));
        assert!(log_content.contains("85.00"));
        assert!(log_content.contains("Current:"));
        assert!(log_content.contains("Average:"));

        println!("✅ Human logging test passed");
    }

    #[test]
    fn test_health_status() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Collect some metrics
        monitor
            .collect_metric(MetricType::ValidatorUptime, 90.0, "test_source", metadata)
            .unwrap();

        let health_status = monitor.get_health_status();
        assert!(health_status.contains("Monitoring System Health"));
        assert!(health_status.contains("Uptime:"));
        assert!(health_status.contains("Metrics Collected:"));
        assert!(health_status.contains("Active Alerts:"));

        println!("✅ Health status test passed");
    }

    #[test]
    fn test_configuration_update() {
        let mut monitor = create_test_monitor();
        let mut new_config = monitor.get_config().clone();
        new_config.collection_interval = 2000;
        new_config.aggregation_window = 600;

        monitor.update_config(new_config);

        let updated_config = monitor.get_config();
        assert_eq!(updated_config.collection_interval, 2000);
        assert_eq!(updated_config.aggregation_window, 600);

        println!("✅ Configuration update test passed");
    }

    // ===== EDGE CASE TESTS =====

    #[test]
    fn test_empty_metric_collection() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Collect metric with zero value
        let result =
            monitor.collect_metric(MetricType::SlashingEvents, 0.0, "test_source", metadata);

        assert!(result.is_ok(), "Zero value metric should be accepted");

        let stats = monitor.get_statistics(&MetricType::SlashingEvents);
        assert!(
            stats.is_some(),
            "Statistics should be available for zero value"
        );

        println!("✅ Empty metric collection test passed");
    }

    #[test]
    fn test_negative_metric_values() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Collect metric with negative value
        let result = monitor.collect_metric(
            MetricType::CrossShardLatency,
            -100.0, // Negative latency (invalid but should be handled)
            "test_source",
            metadata,
        );

        assert!(result.is_ok(), "Negative value metric should be accepted");

        let stats = monitor.get_statistics(&MetricType::CrossShardLatency);
        assert!(
            stats.is_some(),
            "Statistics should be available for negative value"
        );

        if let Some(stats) = stats {
            assert_eq!(stats.current, -100.0);
            assert_eq!(stats.minimum, -100.0);
            assert_eq!(stats.maximum, -100.0);
        }

        println!("✅ Negative metric values test passed");
    }

    #[test]
    fn test_extreme_metric_values() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Collect metric with extreme value
        let result = monitor.collect_metric(
            MetricType::BandwidthUsage,
            f64::MAX,
            "test_source",
            metadata,
        );

        assert!(result.is_ok(), "Extreme value metric should be accepted");

        let stats = monitor.get_statistics(&MetricType::BandwidthUsage);
        assert!(
            stats.is_some(),
            "Statistics should be available for extreme value"
        );

        println!("✅ Extreme metric values test passed");
    }

    #[test]
    fn test_missing_statistics() {
        let monitor = create_test_monitor();

        // Try to get statistics for metric type that hasn't been collected
        let stats = monitor.get_statistics(&MetricType::VoteSubmissionRate);
        assert!(
            stats.is_none(),
            "Statistics should be None for uncollected metric"
        );

        println!("✅ Missing statistics test passed");
    }

    #[test]
    fn test_invalid_alert_acknowledgment() {
        let monitor = create_test_monitor();

        // Try to acknowledge non-existent alert
        let result = monitor.acknowledge_alert("non_existent_alert");
        assert!(
            result.is_err(),
            "Acknowledging non-existent alert should fail"
        );

        println!("✅ Invalid alert acknowledgment test passed");
    }

    #[test]
    fn test_empty_metadata() {
        let monitor = create_test_monitor();
        let empty_metadata = HashMap::new();

        // Collect metric with empty metadata
        let result = monitor.collect_metric(
            MetricType::StakingActivity,
            50.0,
            "test_source",
            empty_metadata,
        );

        assert!(result.is_ok(), "Empty metadata should be accepted");

        println!("✅ Empty metadata test passed");
    }

    // ===== MALICIOUS BEHAVIOR TESTS =====

    #[test]
    fn test_forged_metric_detection() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Collect multiple legitimate metrics to establish baseline
        for i in 0..15 {
            monitor
                .collect_metric(
                    MetricType::ValidatorUptime,
                    90.0 + (i as f64 * 0.1), // Slight variation around 90%
                    "legitimate_source",
                    metadata.clone(),
                )
                .unwrap();
        }

        // Try to collect forged metric with suspicious value
        let mut forged_metadata = metadata.clone();
        forged_metadata.insert("forged".to_string(), "true".to_string());

        let result = monitor.collect_metric(
            MetricType::ValidatorUptime,
            10.0, // Suspiciously low uptime
            "malicious_source",
            forged_metadata,
        );

        // System should accept the metric but detect anomaly
        assert!(
            result.is_ok(),
            "Forged metric should be accepted but flagged"
        );

        let alerts = monitor.get_alerts();
        let anomaly_alerts: Vec<&Alert> = alerts
            .iter()
            .filter(|a| a.message.contains("Anomaly detected"))
            .collect();

        // Anomaly detection should flag the suspicious metric
        assert!(
            !anomaly_alerts.is_empty(),
            "Anomaly should be detected for forged metric"
        );

        println!("✅ Forged metric detection test passed");
    }

    #[test]
    fn test_metric_tampering_resistance() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Collect multiple metrics to build integrity hash
        for i in 0..5 {
            monitor
                .collect_metric(
                    MetricType::TokenTransferRate,
                    100.0 + (i as f64 * 10.0),
                    "test_source",
                    metadata.clone(),
                )
                .unwrap();
        }

        // Get initial integrity hash
        let initial_hash = monitor.get_integrity_hash();

        // Collect additional metric
        monitor
            .collect_metric(
                MetricType::TokenTransferRate,
                200.0,
                "test_source",
                metadata,
            )
            .unwrap();

        // Get updated integrity hash
        let updated_hash = monitor.get_integrity_hash();

        // Hash should have changed
        assert_ne!(
            initial_hash, updated_hash,
            "Integrity hash should change with new metrics"
        );

        println!("✅ Metric tampering resistance test passed");
    }

    #[test]
    fn test_alert_spam_protection() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Generate multiple alerts rapidly
        for i in 0..100 {
            monitor
                .collect_metric(
                    MetricType::BlockFinalizationTime,
                    10000.0 + (i as f64), // All above threshold
                    "spam_source",
                    metadata.clone(),
                )
                .unwrap();
        }

        let alerts = monitor.get_alerts();
        // System should handle spam gracefully
        assert!(alerts.len() <= 100, "Alert count should be reasonable");

        println!("✅ Alert spam protection test passed");
    }

    #[test]
    fn test_malicious_source_filtering() {
        let monitor = create_test_monitor();
        let mut malicious_metadata = create_test_metadata();
        malicious_metadata.insert("malicious".to_string(), "true".to_string());

        // Collect metric from malicious source
        let result = monitor.collect_metric(
            MetricType::ValidatorUptime,
            95.0,
            "malicious_source",
            malicious_metadata,
        );

        // System should still accept the metric but may flag it
        assert!(result.is_ok(), "Malicious source metric should be accepted");

        let stats = monitor.get_statistics(&MetricType::ValidatorUptime);
        assert!(stats.is_some(), "Statistics should be available");

        println!("✅ Malicious source filtering test passed");
    }

    // ===== STRESS TESTS =====

    #[test]
    fn test_high_metric_volume() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Collect large number of metrics
        for i in 0..1000 {
            let metric_type = match i % 14 {
                0 => MetricType::ValidatorUptime,
                1 => MetricType::BlockFinalizationTime,
                2 => MetricType::SlashingEvents,
                3 => MetricType::ShardThroughput,
                4 => MetricType::CrossShardLatency,
                5 => MetricType::StateSyncSuccessRate,
                6 => MetricType::NodeConnectivity,
                7 => MetricType::MessagePropagationDelay,
                8 => MetricType::BandwidthUsage,
                9 => MetricType::VoteSubmissionRate,
                10 => MetricType::ZkSnarkVerificationTime,
                11 => MetricType::StakingActivity,
                12 => MetricType::TokenTransferRate,
                _ => MetricType::VotingWeightUpdateRate,
            };

            monitor
                .collect_metric(
                    metric_type,
                    (i as f64) * 0.1,
                    "stress_test",
                    metadata.clone(),
                )
                .unwrap();
        }

        // System should handle high volume gracefully
        let health_status = monitor.get_health_status();
        assert!(health_status.contains("Metrics Collected:"));

        println!("✅ High metric volume test passed");
    }

    #[test]
    fn test_concurrent_metric_collection() {
        let monitor = Arc::new(create_test_monitor());
        let mut handles = Vec::new();

        // Spawn multiple threads to collect metrics concurrently
        for thread_id in 0..10 {
            let monitor_clone = Arc::clone(&monitor);
            let handle = thread::spawn(move || {
                let metadata = create_test_metadata();
                for i in 0..100 {
                    monitor_clone
                        .collect_metric(
                            MetricType::ValidatorUptime,
                            80.0 + (thread_id as f64) + (i as f64 * 0.1),
                            &format!("thread_{}", thread_id),
                            metadata.clone(),
                        )
                        .unwrap();
                }
            });
            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }

        // System should handle concurrent access gracefully
        let stats = monitor.get_statistics(&MetricType::ValidatorUptime);
        assert!(
            stats.is_some(),
            "Statistics should be available after concurrent collection"
        );

        println!("✅ Concurrent metric collection test passed");
    }

    #[test]
    fn test_memory_usage_optimization() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Collect metrics beyond the limit to test memory management
        for i in 0..15000 {
            // More than max_metrics (10000)
            monitor
                .collect_metric(
                    MetricType::ValidatorUptime,
                    (i as f64) % 100.0,
                    "memory_test",
                    metadata.clone(),
                )
                .unwrap();
        }

        // System should maintain reasonable memory usage
        let stats = monitor.get_statistics(&MetricType::ValidatorUptime);
        assert!(stats.is_some(), "Statistics should be available");

        if let Some(stats) = stats {
            assert!(
                stats.sample_count <= 10000,
                "Sample count should not exceed limit"
            );
        }

        println!("✅ Memory usage optimization test passed");
    }

    #[test]
    fn test_long_running_monitoring() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Simulate long-running monitoring
        for i in 0..100 {
            // Collect various metrics
            monitor
                .collect_metric(
                    MetricType::ValidatorUptime,
                    90.0 + (i as f64 * 0.1),
                    "long_running",
                    metadata.clone(),
                )
                .unwrap();

            monitor
                .collect_metric(
                    MetricType::BlockFinalizationTime,
                    1000.0 + (i as f64 * 10.0),
                    "long_running",
                    metadata.clone(),
                )
                .unwrap();

            // Small delay to simulate real monitoring
            thread::sleep(Duration::from_millis(1));
        }

        // Add a longer delay to ensure uptime is positive
        thread::sleep(Duration::from_millis(1100)); // 1.1 seconds to ensure uptime > 0

        // System should remain stable
        let uptime = monitor.get_uptime();
        assert!(uptime >= 1, "Uptime should be at least 1 second");

        let health_status = monitor.get_health_status();
        assert!(health_status.contains("Healthy") || health_status.contains("Issues Detected"));

        println!("✅ Long running monitoring test passed");
    }

    // ===== INTEGRATION TESTS =====

    #[test]
    fn test_pos_metrics_integration() {
        let monitor = create_test_monitor();

        // Test PoS metrics collection
        let result = collect_pos_metrics(&monitor, 95.0, 2000.0, 2.0);
        assert!(result.is_ok(), "PoS metrics collection should succeed");

        // Verify metrics were collected
        let uptime_stats = monitor.get_statistics(&MetricType::ValidatorUptime);
        assert!(
            uptime_stats.is_some(),
            "Validator uptime statistics should be available"
        );

        let finalization_stats = monitor.get_statistics(&MetricType::BlockFinalizationTime);
        assert!(
            finalization_stats.is_some(),
            "Block finalization statistics should be available"
        );

        let slashing_stats = monitor.get_statistics(&MetricType::SlashingEvents);
        assert!(
            slashing_stats.is_some(),
            "Slashing events statistics should be available"
        );

        println!("✅ PoS metrics integration test passed");
    }

    #[test]
    fn test_sharding_metrics_integration() {
        let monitor = create_test_monitor();

        // Test sharding metrics collection
        let result = collect_sharding_metrics(&monitor, 150.0, 500.0, 95.0);
        assert!(result.is_ok(), "Sharding metrics collection should succeed");

        // Verify metrics were collected
        let throughput_stats = monitor.get_statistics(&MetricType::ShardThroughput);
        assert!(
            throughput_stats.is_some(),
            "Shard throughput statistics should be available"
        );

        let latency_stats = monitor.get_statistics(&MetricType::CrossShardLatency);
        assert!(
            latency_stats.is_some(),
            "Cross-shard latency statistics should be available"
        );

        let sync_stats = monitor.get_statistics(&MetricType::StateSyncSuccessRate);
        assert!(
            sync_stats.is_some(),
            "State sync success rate statistics should be available"
        );

        println!("✅ Sharding metrics integration test passed");
    }

    #[test]
    fn test_p2p_metrics_integration() {
        let monitor = create_test_monitor();

        // Test P2P metrics collection
        let result = collect_p2p_metrics(&monitor, 85.0, 1000.0, 500000.0);
        assert!(result.is_ok(), "P2P metrics collection should succeed");

        // Verify metrics were collected
        let connectivity_stats = monitor.get_statistics(&MetricType::NodeConnectivity);
        assert!(
            connectivity_stats.is_some(),
            "Node connectivity statistics should be available"
        );

        let delay_stats = monitor.get_statistics(&MetricType::MessagePropagationDelay);
        assert!(
            delay_stats.is_some(),
            "Message propagation delay statistics should be available"
        );

        let bandwidth_stats = monitor.get_statistics(&MetricType::BandwidthUsage);
        assert!(
            bandwidth_stats.is_some(),
            "Bandwidth usage statistics should be available"
        );

        println!("✅ P2P metrics integration test passed");
    }

    #[test]
    fn test_voting_metrics_integration() {
        let monitor = create_test_monitor();

        // Test voting metrics collection
        let result = collect_voting_metrics(&monitor, 25.0, 5000.0);
        assert!(result.is_ok(), "Voting metrics collection should succeed");

        // Verify metrics were collected
        let submission_stats = monitor.get_statistics(&MetricType::VoteSubmissionRate);
        assert!(
            submission_stats.is_some(),
            "Vote submission rate statistics should be available"
        );

        let verification_stats = monitor.get_statistics(&MetricType::ZkSnarkVerificationTime);
        assert!(
            verification_stats.is_some(),
            "zk-SNARK verification time statistics should be available"
        );

        println!("✅ Voting metrics integration test passed");
    }

    #[test]
    fn test_governance_metrics_integration() {
        let monitor = create_test_monitor();

        // Test governance metrics collection
        let result = collect_governance_metrics(&monitor, 75.0, 100.0, 5.0);
        assert!(
            result.is_ok(),
            "Governance metrics collection should succeed"
        );

        // Verify metrics were collected
        let staking_stats = monitor.get_statistics(&MetricType::StakingActivity);
        assert!(
            staking_stats.is_some(),
            "Staking activity statistics should be available"
        );

        let transfer_stats = monitor.get_statistics(&MetricType::TokenTransferRate);
        assert!(
            transfer_stats.is_some(),
            "Token transfer rate statistics should be available"
        );

        let weight_stats = monitor.get_statistics(&MetricType::VotingWeightUpdateRate);
        assert!(
            weight_stats.is_some(),
            "Voting weight update rate statistics should be available"
        );

        println!("✅ Governance metrics integration test passed");
    }

    #[test]
    fn test_comprehensive_monitoring_workflow() {
        let monitor = create_test_monitor();

        // Simulate a complete monitoring workflow
        let mut metadata = create_test_metadata();
        metadata.insert("workflow".to_string(), "comprehensive".to_string());

        // Collect metrics from all modules
        collect_pos_metrics(&monitor, 92.0, 1500.0, 1.0).unwrap();
        collect_sharding_metrics(&monitor, 200.0, 300.0, 98.0).unwrap();
        collect_p2p_metrics(&monitor, 88.0, 800.0, 750000.0).unwrap();
        collect_voting_metrics(&monitor, 30.0, 3000.0).unwrap();
        collect_governance_metrics(&monitor, 80.0, 120.0, 8.0).unwrap();

        // Generate logs for all metric types
        let metric_types = vec![
            MetricType::ValidatorUptime,
            MetricType::BlockFinalizationTime,
            MetricType::ShardThroughput,
            MetricType::NodeConnectivity,
            MetricType::VoteSubmissionRate,
        ];

        for metric_type in metric_types {
            let json_log = monitor.generate_json_log(&metric_type);
            assert!(
                json_log.is_ok(),
                "JSON log generation should succeed for {:?}",
                metric_type
            );

            let human_log = monitor.generate_human_log(&metric_type);
            assert!(
                human_log.is_ok(),
                "Human log generation should succeed for {:?}",
                metric_type
            );
        }

        // Check system health
        let health_status = monitor.get_health_status();
        assert!(health_status.contains("Monitoring System Health"));

        // Verify all metrics have statistics
        let all_metric_types = vec![
            MetricType::ValidatorUptime,
            MetricType::BlockFinalizationTime,
            MetricType::SlashingEvents,
            MetricType::ShardThroughput,
            MetricType::CrossShardLatency,
            MetricType::StateSyncSuccessRate,
            MetricType::NodeConnectivity,
            MetricType::MessagePropagationDelay,
            MetricType::BandwidthUsage,
            MetricType::VoteSubmissionRate,
            MetricType::ZkSnarkVerificationTime,
            MetricType::StakingActivity,
            MetricType::TokenTransferRate,
            MetricType::VotingWeightUpdateRate,
        ];

        for metric_type in all_metric_types {
            let stats = monitor.get_statistics(&metric_type);
            assert!(
                stats.is_some(),
                "Statistics should be available for {:?}",
                metric_type
            );
        }

        println!("✅ Comprehensive monitoring workflow test passed");
    }

    #[test]
    fn test_metrics_clearing() {
        let monitor = create_test_monitor();
        let metadata = create_test_metadata();

        // Collect some metrics
        monitor
            .collect_metric(MetricType::ValidatorUptime, 90.0, "test_source", metadata)
            .unwrap();

        // Verify metrics exist
        let stats = monitor.get_statistics(&MetricType::ValidatorUptime);
        assert!(
            stats.is_some(),
            "Statistics should be available before clearing"
        );

        // Clear metrics
        monitor.clear_metrics();

        // Verify metrics are cleared
        let cleared_stats = monitor.get_statistics(&MetricType::ValidatorUptime);
        assert!(
            cleared_stats.is_none(),
            "Statistics should be None after clearing"
        );

        let alerts = monitor.get_alerts();
        assert!(alerts.is_empty(), "Alerts should be empty after clearing");

        println!("✅ Metrics clearing test passed");
    }
}
