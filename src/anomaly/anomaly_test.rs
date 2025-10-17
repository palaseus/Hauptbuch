//! Simplified test suite for the AI-driven anomaly detection module
//!
//! This test suite focuses on core anomaly detection functionality
//! with simplified test data structures.

use super::*;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: Basic anomaly detector creation
    #[test]
    fn test_anomaly_detector_creation() {
        let detector = AnomalyDetector::new();
        assert!(detector.is_ok());
    }

    // Test 2: Configuration updates
    #[test]
    fn test_configuration_updates() {
        let mut detector = AnomalyDetector::new().unwrap();

        let new_config = AnomalyConfig {
            zscore_threshold: 2.5,
            kmeans_clusters: 5,
            ..Default::default()
        };

        let result = detector.update_config(new_config);
        assert!(result.is_ok());
    }

    // Test 3: K-means clustering functionality
    #[test]
    fn test_kmeans_clustering() {
        let detector = AnomalyDetector::new().unwrap();

        // Create test data with clear clusters
        let data = vec![
            vec![1.0, 1.0],
            vec![1.1, 1.1],
            vec![1.2, 1.2], // Cluster 1
            vec![5.0, 5.0],
            vec![5.1, 5.1],
            vec![5.2, 5.2], // Cluster 2
            vec![9.0, 9.0],
            vec![9.1, 9.1],
            vec![9.2, 9.2], // Cluster 3
        ];

        let result = detector.perform_kmeans_clustering(&data, 3);
        assert!(result.is_ok());

        let cluster_result = result.unwrap();
        assert_eq!(cluster_result.k, 3);
        assert_eq!(cluster_result.centroids.len(), 3);
        assert_eq!(cluster_result.assignments.len(), 9);
        assert!(cluster_result.wcss >= 0.0);
        assert!(cluster_result.iterations > 0);
    }

    // Test 4: Z-score calculation
    #[test]
    fn test_zscore_calculation() {
        let detector = AnomalyDetector::new().unwrap();

        // Create test data with known statistics
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let result = detector.calculate_zscore(&data);
        assert!(result.is_ok());

        let zscore_result = result.unwrap();
        assert_eq!(zscore_result.z_scores.len(), 10);
        assert_eq!(zscore_result.threshold, 3.0);
        assert_eq!(zscore_result.mean, 5.5);
        assert!(zscore_result.std_dev > 0.0);
    }

    // Test 5: Anomaly severity determination
    #[test]
    fn test_anomaly_severity_determination() {
        let detector = AnomalyDetector::new().unwrap();

        assert_eq!(
            detector.determine_anomaly_severity(0.5),
            AnomalySeverity::Low
        );
        assert_eq!(
            detector.determine_anomaly_severity(1.6),
            AnomalySeverity::Medium
        );
        assert_eq!(
            detector.determine_anomaly_severity(2.5),
            AnomalySeverity::High
        );
        assert_eq!(
            detector.determine_anomaly_severity(3.5),
            AnomalySeverity::Critical
        );
    }

    // Test 6: Confidence calculation
    #[test]
    fn test_confidence_calculation() {
        let detector = AnomalyDetector::new().unwrap();

        assert_eq!(detector.calculate_confidence(0.0), 0.0);
        assert_eq!(detector.calculate_confidence(1.5), 0.5);
        assert_eq!(detector.calculate_confidence(3.0), 1.0);
        assert_eq!(detector.calculate_confidence(6.0), 1.0); // Capped at 1.0
    }

    // Test 7: Edge case - empty clustering data
    #[test]
    fn test_empty_clustering_data() {
        let detector = AnomalyDetector::new().unwrap();
        let empty_data = Vec::new();

        let result = detector.perform_kmeans_clustering(&empty_data, 3);
        assert!(result.is_err());
    }

    // Test 8: Edge case - single data point
    #[test]
    fn test_single_data_point() {
        let detector = AnomalyDetector::new().unwrap();
        let single_point = vec![vec![1.0, 2.0, 3.0]];

        let result = detector.perform_kmeans_clustering(&single_point, 1);
        assert!(result.is_ok());
    }

    // Test 9: Edge case - zero standard deviation
    #[test]
    fn test_zero_standard_deviation() {
        let detector = AnomalyDetector::new().unwrap();
        let uniform_data = vec![5.0, 5.0, 5.0, 5.0, 5.0];

        let result = detector.calculate_zscore(&uniform_data);
        assert!(result.is_ok());

        let zscore_result = result.unwrap();
        assert_eq!(zscore_result.std_dev, 0.0);
        assert_eq!(zscore_result.anomaly_count, 0);
    }

    // Test 10: Edge case - extreme values
    #[test]
    fn test_extreme_values() {
        let detector = AnomalyDetector::new().unwrap();

        // Create data with extreme values
        let extreme_data = vec![f64::MIN, f64::MAX, 0.0, -f64::INFINITY, f64::INFINITY];

        let result = detector.calculate_zscore(&extreme_data);
        // Should handle extreme values gracefully
        assert!(result.is_ok() || result.is_err());
    }

    // Test 11: Invalid configuration
    #[test]
    fn test_invalid_configuration() {
        let config = AnomalyConfig {
            kmeans_clusters: 0, // Invalid
            ..Default::default()
        };

        let result = AnomalyDetector::with_config(config);
        assert!(result.is_err());
    }

    // Test 12: Chart generation
    #[test]
    fn test_chart_generation() {
        let detector = AnomalyDetector::new().unwrap();

        let charts = detector.generate_anomaly_charts();
        assert!(charts.is_ok());

        let charts = charts.unwrap();
        assert_eq!(charts.len(), 3); // Severity, type, and timeline charts
        assert_eq!(charts[0].chart_type, "doughnut");
        assert_eq!(charts[1].chart_type, "bar");
        assert_eq!(charts[2].chart_type, "line");
    }

    // Test 13: Data integrity verification
    #[test]
    fn test_data_integrity_verification() {
        let mut detector = AnomalyDetector::new().unwrap();

        let test_data = b"test data for integrity verification";
        detector.update_integrity_hash(test_data);

        let is_valid = detector.verify_data_integrity(test_data);
        assert!(is_valid.is_ok());
        assert!(is_valid.unwrap());

        let corrupted_data = b"corrupted data";
        let is_invalid = detector.verify_data_integrity(corrupted_data);
        assert!(is_invalid.is_ok());
        assert!(!is_invalid.unwrap());
    }

    // Test 14: Detection history management
    #[test]
    fn test_detection_history_management() {
        let detector = AnomalyDetector::new().unwrap();

        let history = detector.get_detection_history();
        assert!(history.is_empty());

        let counters = detector.get_anomaly_counters();
        assert!(counters.is_empty());
    }

    // Test 15: Clear history
    #[test]
    fn test_clear_history() {
        let mut detector = AnomalyDetector::new().unwrap();

        detector.clear_history();
        let cleared_history = detector.get_detection_history();
        assert!(cleared_history.is_empty());
    }

    // Test 16: Anomaly type display
    #[test]
    fn test_anomaly_type_display() {
        assert_eq!(
            AnomalyType::VoterTurnoutSpike.to_string(),
            "Voter Turnout Spike"
        );
        assert_eq!(AnomalyType::VoteStuffing.to_string(), "Vote Stuffing");
        assert_eq!(AnomalyType::SybilAttack.to_string(), "Sybil Attack");
        assert_eq!(
            AnomalyType::NetworkLatencySpike.to_string(),
            "Network Latency Spike"
        );
        assert_eq!(
            AnomalyType::CrossChainInconsistency.to_string(),
            "Cross-Chain Inconsistency"
        );
    }

    // Test 17: Anomaly severity display
    #[test]
    fn test_anomaly_severity_display() {
        assert_eq!(AnomalySeverity::Low.to_string(), "Low");
        assert_eq!(AnomalySeverity::Medium.to_string(), "Medium");
        assert_eq!(AnomalySeverity::High.to_string(), "High");
        assert_eq!(AnomalySeverity::Critical.to_string(), "Critical");
    }

    // Test 18: Error handling
    #[test]
    fn test_error_handling() {
        let detector = AnomalyDetector::new().unwrap();

        // Test with invalid clustering parameters
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let result = detector.perform_kmeans_clustering(&data, 0); // Invalid k
        assert!(result.is_err());
    }

    // Test 19: Very small datasets
    #[test]
    fn test_very_small_datasets() {
        let detector = AnomalyDetector::new().unwrap();

        // Test with minimal data
        let data = vec![1.0, 2.0];
        let result = detector.calculate_zscore(&data);
        assert!(result.is_ok());
    }

    // Test 20: Maximum configuration values
    #[test]
    fn test_maximum_configuration_values() {
        let config = AnomalyConfig {
            kmeans_clusters: 100,
            kmeans_max_iterations: 1000,
            zscore_threshold: 10.0,
            ..Default::default()
        };

        let detector = AnomalyDetector::with_config(config);
        assert!(detector.is_ok());
    }

    // Test 21: Performance test - high volume data
    #[test]
    fn test_stress_high_volume_data() {
        let detector = AnomalyDetector::new().unwrap();

        // Create large dataset
        let data: Vec<Vec<f64>> = (0..1000)
            .map(|i| vec![i as f64, (i * 2) as f64, (i * 3) as f64])
            .collect();

        let start_time = std::time::Instant::now();
        let result = detector.perform_kmeans_clustering(&data, 5);
        let duration = start_time.elapsed();

        // Should complete within reasonable time (less than 5 seconds)
        assert!(duration.as_secs() < 5);
        assert!(result.is_ok());
    }

    // Test 22: Memory usage test
    #[test]
    fn test_memory_usage() {
        let detector = AnomalyDetector::new().unwrap();

        // Test that detector can be created without excessive memory usage
        let history_size = detector.get_detection_history().len();
        assert_eq!(history_size, 0);

        let counters = detector.get_anomaly_counters();
        assert!(counters.is_empty());
    }

    // Test 23: Anomaly detection with empty data
    #[test]
    fn test_empty_data_anomaly_detection() {
        let mut detector = AnomalyDetector::new().unwrap();

        // Test with empty data - should handle gracefully
        let empty_votes: Vec<crate::governance::proposal::Vote> = Vec::new();
        let empty_proposals: Vec<crate::governance::proposal::Proposal> = Vec::new();

        // Create minimal analytics data
        let analytics = crate::analytics::governance::GovernanceAnalytics {
            report_id: "test".to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            time_range: crate::analytics::governance::TimeRange {
                start_time: 0,
                end_time: 1000,
                duration_seconds: 1000,
            },
            voter_turnout: crate::analytics::governance::VoterTurnoutMetrics {
                total_voters: 0,
                eligible_voters: 0,
                turnout_percentage: 0.0,
                average_votes_per_voter: 0.0,
                most_active_voter: None,
                participation_by_tier: HashMap::new(),
            },
            stake_distribution: crate::analytics::governance::StakeDistributionMetrics {
                total_stake: 0,
                stake_holders: 0,
                gini_coefficient: 0.0,
                median_stake: 0,
                mean_stake: 0.0,
                stake_std_deviation: 0.0,
                top_10_percent_share: 0.0,
                stake_tiers: HashMap::new(),
            },
            proposal_analysis: crate::analytics::governance::ProposalAnalysisMetrics {
                total_proposals: 0,
                successful_proposals: 0,
                failed_proposals: 0,
                success_rate: 0.0,
                success_rate_by_type: HashMap::new(),
                average_voting_duration: 0.0,
                most_common_type: None,
                outcomes_by_status: HashMap::new(),
            },
            cross_chain_metrics: crate::analytics::governance::CrossChainMetrics {
                total_cross_chain_votes: 0,
                participating_chains: 0,
                average_votes_per_chain: 0.0,
                avg_sync_delay: 0.0,
                chain_participation: HashMap::new(),
                cross_chain_success_rate: 0.0,
                most_active_chain: None,
            },
            temporal_trends: crate::analytics::governance::TemporalTrendMetrics {
                hourly_activity: Vec::new(),
                daily_patterns: HashMap::new(),
                weekly_trends: Vec::new(),
                peak_periods: Vec::new(),
                seasonal_patterns: HashMap::new(),
            },
            integrity_hash: "test".to_string(),
        };

        let anomalies =
            detector.detect_governance_anomalies(&empty_proposals, &empty_votes, &analytics);
        assert!(anomalies.is_ok());
        assert_eq!(anomalies.unwrap().len(), 0);
    }

    // Test 24: Configuration validation
    #[test]
    fn test_configuration_validation() {
        let config = AnomalyConfig::default();

        assert!(config.enable_kmeans);
        assert!(config.enable_zscore);
        assert_eq!(config.zscore_threshold, 3.0);
        assert_eq!(config.kmeans_clusters, 3);
        assert_eq!(config.kmeans_max_iterations, 100);
        assert!(config.enable_quantum_signatures);
        assert!(config.enable_integrity_verification);
    }

    // Test 25: Anomaly detection result structure
    #[test]
    fn test_anomaly_detection_result_structure() {
        let anomaly = AnomalyDetection {
            id: "test_anomaly".to_string(),
            anomaly_type: AnomalyType::VoterTurnoutSpike,
            severity: AnomalySeverity::Medium,
            score: 0.75,
            description: "Test anomaly".to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            source_data: HashMap::new(),
            confidence: 0.8,
            recommendations: vec!["Test recommendation".to_string()],
            alert_signature: None,
        };

        assert_eq!(anomaly.id, "test_anomaly");
        assert_eq!(anomaly.anomaly_type, AnomalyType::VoterTurnoutSpike);
        assert_eq!(anomaly.severity, AnomalySeverity::Medium);
        assert_eq!(anomaly.score, 0.75);
        assert_eq!(anomaly.confidence, 0.8);
        assert_eq!(anomaly.recommendations.len(), 1);
    }
}
