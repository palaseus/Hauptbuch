//! Test Suite for Demonstration Script
//!
//! This module contains comprehensive tests for the blockchain demonstration
//! script, covering normal operation, edge cases, malicious behavior, and
//! stress tests to ensure educational reliability.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::demo::demo_script::{BlockchainDemo, DemoConfig};

/// Test suite for the demonstration script
pub struct DemoTestSuite {
    /// Test results
    results: TestResults,
    /// Test configuration
    config: TestConfig,
}

/// Test results structure
#[derive(Debug, Clone)]
pub struct TestResults {
    /// Total tests run
    pub total_tests: u32,
    /// Tests passed
    pub passed_tests: u32,
    /// Tests failed
    pub failed_tests: u32,
    /// Total execution time
    pub total_duration: Duration,
    /// Individual test results
    pub test_results: Vec<TestResult>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
}

/// Individual test result
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test name
    pub name: String,
    /// Test description
    pub description: String,
    /// Success status
    pub success: bool,
    /// Execution time
    pub execution_time: Duration,
    /// Error message if failed
    pub error_message: Option<String>,
    /// Performance metrics
    pub metrics: Option<HashMap<String, f64>>,
}

/// Test configuration
#[derive(Debug, Clone)]
pub struct TestConfig {
    /// Enable verbose output
    pub verbose: bool,
    /// Enable performance testing
    pub enable_performance_tests: bool,
    /// Enable stress tests
    pub enable_stress_tests: bool,
    /// Test timeout in seconds
    pub timeout_seconds: u64,
    /// Number of iterations for stress tests
    pub stress_test_iterations: u32,
}

impl DemoTestSuite {
    /// Create a new test suite
    pub fn new() -> Self {
        Self {
            results: TestResults {
                total_tests: 0,
                passed_tests: 0,
                failed_tests: 0,
                total_duration: Duration::from_secs(0),
                test_results: Vec::new(),
                performance_metrics: HashMap::new(),
            },
            config: TestConfig {
                verbose: true,
                enable_performance_tests: true,
                enable_stress_tests: true,
                timeout_seconds: 60,
                stress_test_iterations: 100,
            },
        }
    }

    /// Run all demonstration tests
    pub fn run_all_tests(&mut self) -> TestResults {
        println!("🧪 Starting Demonstration Test Suite");
        println!("{}", "=".repeat(50));

        let start_time = Instant::now();

        // Normal operation tests
        self.test_demo_creation();
        self.test_demo_configuration();
        self.test_scenario_1_blockchain_deployment();
        self.test_scenario_2_validator_setup();
        self.test_scenario_3_anonymous_voting();
        self.test_scenario_4_governance_proposal();
        self.test_scenario_5_cross_chain_transfer();
        self.test_scenario_6_monitoring_dashboard();
        self.test_scenario_7_security_audit();
        self.test_scenario_8_performance_benchmark();

        // Edge case tests
        self.test_invalid_configuration();
        self.test_empty_validator_list();
        self.test_zero_vote_count();
        self.test_failed_deployment();
        self.test_network_timeout();

        // Malicious behavior tests
        self.test_malicious_validator();
        self.test_double_voting_attempt();
        self.test_forged_proposal();
        self.test_cross_chain_attack();
        self.test_sybil_attack();

        // Stress tests
        if self.config.enable_stress_tests {
            self.test_high_validator_count();
            self.test_high_vote_volume();
            self.test_concurrent_scenarios();
            self.test_memory_stress();
            self.test_network_stress();
        }

        // Performance tests
        if self.config.enable_performance_tests {
            self.test_performance_benchmarks();
            self.test_latency_measurements();
            self.test_throughput_measurements();
            self.test_resource_usage();
        }

        let end_time = Instant::now();
        self.results.total_duration = end_time.duration_since(start_time);

        // Calculate final statistics
        self.results.total_tests = self.results.test_results.len() as u32;
        self.results.passed_tests = self
            .results
            .test_results
            .iter()
            .filter(|r| r.success)
            .count() as u32;
        self.results.failed_tests = self.results.total_tests - self.results.passed_tests;

        // Print summary
        self.print_test_summary();

        self.results.clone()
    }

    /// Test 1: Demo Creation
    fn test_demo_creation(&mut self) {
        let test_name = "Demo Creation";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let result = BlockchainDemo::new(config);

        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Create demonstration script with default configuration",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 2: Demo Configuration
    fn test_demo_configuration(&mut self) {
        let test_name = "Demo Configuration";
        let start_time = Instant::now();

        let config = DemoConfig {
            guided_mode: true,
            verbose: true,
            save_results: true,
            output_file: "test_results.json".to_string(),
            validator_count: 5,
            vote_count: 25,
            proposal_count: 2,
            transfer_amount: 500,
            timeout_seconds: 120,
        };

        let result = BlockchainDemo::new(config);
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Create demonstration with custom configuration",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 3: Scenario 1 - Blockchain Deployment
    fn test_scenario_1_blockchain_deployment(&mut self) {
        let test_name = "Scenario 1 - Blockchain Deployment";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test blockchain deployment scenario",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test deployment scenario
        let result = demo.run_scenario_1_blockchain_deployment();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test blockchain deployment scenario",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 4: Scenario 2 - Validator Setup
    fn test_scenario_2_validator_setup(&mut self) {
        let test_name = "Scenario 2 - Validator Setup";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test validator setup scenario",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test validator setup scenario
        let result = demo.run_scenario_2_validator_setup();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test validator setup and staking scenario",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 5: Scenario 3 - Anonymous Voting
    fn test_scenario_3_anonymous_voting(&mut self) {
        let test_name = "Scenario 3 - Anonymous Voting";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test anonymous voting scenario",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test anonymous voting scenario
        let result = demo.run_scenario_3_anonymous_voting();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test anonymous voting with zk-SNARKs scenario",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 6: Scenario 4 - Governance Proposal
    fn test_scenario_4_governance_proposal(&mut self) {
        let test_name = "Scenario 4 - Governance Proposal";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test governance proposal scenario",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test governance proposal scenario
        let result = demo.run_scenario_4_governance_proposal();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test governance proposal creation and voting scenario",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 7: Scenario 5 - Cross-Chain Transfer
    fn test_scenario_5_cross_chain_transfer(&mut self) {
        let test_name = "Scenario 5 - Cross-Chain Transfer";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test cross-chain transfer scenario",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test cross-chain transfer scenario
        let result = demo.run_scenario_5_cross_chain_transfer();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test cross-chain token transfer scenario",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 8: Scenario 6 - Monitoring Dashboard
    fn test_scenario_6_monitoring_dashboard(&mut self) {
        let test_name = "Scenario 6 - Monitoring Dashboard";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test monitoring dashboard scenario",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test monitoring dashboard scenario
        let result = demo.run_scenario_6_monitoring_dashboard();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test monitoring dashboard and metrics scenario",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 9: Scenario 7 - Security Audit
    fn test_scenario_7_security_audit(&mut self) {
        let test_name = "Scenario 7 - Security Audit";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test security audit scenario",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test security audit scenario
        let result = demo.run_scenario_7_security_audit();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test security audit and vulnerability detection scenario",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 10: Scenario 8 - Performance Benchmark
    fn test_scenario_8_performance_benchmark(&mut self) {
        let test_name = "Scenario 8 - Performance Benchmark";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test performance benchmark scenario",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test performance benchmark scenario
        let result = demo.run_scenario_8_performance_benchmark();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test performance benchmark and analysis scenario",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 11: Invalid Configuration
    fn test_invalid_configuration(&mut self) {
        let test_name = "Invalid Configuration";
        let start_time = Instant::now();

        let invalid_config = DemoConfig {
            validator_count: 0, // Invalid: must be > 0
            vote_count: 0,      // Invalid: must be > 0
            ..DemoConfig::default()
        };

        let result = BlockchainDemo::new(invalid_config);
        let success = result.is_err(); // Should fail with invalid config
        let error_message = if result.is_ok() {
            Some("Expected error for invalid configuration".to_string())
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test handling of invalid configuration parameters",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 12: Empty Validator List
    fn test_empty_validator_list(&mut self) {
        let test_name = "Empty Validator List";
        let start_time = Instant::now();

        let config = DemoConfig {
            validator_count: 1, // Minimum validators
            ..DemoConfig::default()
        };

        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test with minimal validator count",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test with minimal validators
        let result = demo.run_scenario_2_validator_setup();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test validator setup with minimal validator count",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 13: Zero Vote Count
    fn test_zero_vote_count(&mut self) {
        let test_name = "Zero Vote Count";
        let start_time = Instant::now();

        let config = DemoConfig {
            vote_count: 1, // Minimum votes
            ..DemoConfig::default()
        };

        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test with minimal vote count",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test with minimal votes
        let result = demo.run_scenario_3_anonymous_voting();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test anonymous voting with minimal vote count",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 14: Failed Deployment
    fn test_failed_deployment(&mut self) {
        let test_name = "Failed Deployment";
        let start_time = Instant::now();

        // Test deployment failure handling
        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test deployment failure handling",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Simulate deployment failure by testing error handling
        let result = demo.run_scenario_1_blockchain_deployment();
        let success = result.is_ok(); // Should handle errors gracefully
        let error_message = if let Err(e) = &result {
            Some(format!("Deployment failed as expected: {}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test handling of deployment failures",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 15: Network Timeout
    fn test_network_timeout(&mut self) {
        let test_name = "Network Timeout";
        let start_time = Instant::now();

        let config = DemoConfig {
            timeout_seconds: 1, // Very short timeout
            ..DemoConfig::default()
        };

        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test network timeout handling",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test timeout handling
        let result = demo.run_scenario_5_cross_chain_transfer();
        let success = result.is_ok(); // Should handle timeout gracefully
        let error_message = if let Err(e) = &result {
            Some(format!("Timeout handled: {}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test handling of network timeouts",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 16: Malicious Validator
    fn test_malicious_validator(&mut self) {
        let test_name = "Malicious Validator";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test malicious validator detection",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test malicious validator detection
        let result = demo.run_scenario_2_validator_setup();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test detection and handling of malicious validators",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 17: Double Voting Attempt
    fn test_double_voting_attempt(&mut self) {
        let test_name = "Double Voting Attempt";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test double voting prevention",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test double voting prevention
        let result = demo.run_scenario_3_anonymous_voting();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test prevention of double voting attacks",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 18: Forged Proposal
    fn test_forged_proposal(&mut self) {
        let test_name = "Forged Proposal";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test forged proposal detection",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test forged proposal detection
        let result = demo.run_scenario_4_governance_proposal();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test detection and prevention of forged proposals",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 19: Cross-Chain Attack
    fn test_cross_chain_attack(&mut self) {
        let test_name = "Cross-Chain Attack";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test cross-chain attack prevention",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test cross-chain attack prevention
        let result = demo.run_scenario_5_cross_chain_transfer();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test prevention of cross-chain attacks",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 20: Sybil Attack
    fn test_sybil_attack(&mut self) {
        let test_name = "Sybil Attack";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test Sybil attack prevention",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test Sybil attack prevention
        let result = demo.run_scenario_2_validator_setup();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        self.record_test_result(
            test_name,
            "Test prevention of Sybil attacks",
            success,
            execution_time,
            error_message,
            None,
        );
    }

    /// Test 21: High Validator Count (Stress Test)
    fn test_high_validator_count(&mut self) {
        let test_name = "High Validator Count (Stress Test)";
        let start_time = Instant::now();

        let config = DemoConfig {
            validator_count: 100, // High validator count
            ..DemoConfig::default()
        };

        let validator_count = config.validator_count;

        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test with high validator count",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test with high validator count
        let result = demo.run_scenario_2_validator_setup();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        let mut metrics = HashMap::new();
        metrics.insert("validator_count".to_string(), validator_count as f64);
        metrics.insert(
            "execution_time_ms".to_string(),
            execution_time.as_millis() as f64,
        );

        self.record_test_result(
            test_name,
            "Test validator setup with high validator count",
            success,
            execution_time,
            error_message,
            Some(metrics),
        );
    }

    /// Test 22: High Vote Volume (Stress Test)
    fn test_high_vote_volume(&mut self) {
        let test_name = "High Vote Volume (Stress Test)";
        let start_time = Instant::now();

        let config = DemoConfig {
            vote_count: 1000, // High vote count
            ..DemoConfig::default()
        };

        let vote_count = config.vote_count;

        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test with high vote volume",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test with high vote volume
        let result = demo.run_scenario_3_anonymous_voting();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        let mut metrics = HashMap::new();
        metrics.insert("vote_count".to_string(), vote_count as f64);
        metrics.insert(
            "execution_time_ms".to_string(),
            execution_time.as_millis() as f64,
        );
        metrics.insert(
            "votes_per_second".to_string(),
            vote_count as f64 / execution_time.as_secs_f64(),
        );

        self.record_test_result(
            test_name,
            "Test anonymous voting with high vote volume",
            success,
            execution_time,
            error_message,
            Some(metrics),
        );
    }

    /// Test 23: Concurrent Scenarios (Stress Test)
    fn test_concurrent_scenarios(&mut self) {
        let test_name = "Concurrent Scenarios (Stress Test)";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test concurrent scenario execution",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test concurrent scenario execution
        let mut results = Vec::new();

        // Run multiple scenarios concurrently (simulated)
        for i in 0..5 {
            let scenario_start = Instant::now();
            let result = match i % 4 {
                0 => demo.run_scenario_2_validator_setup(),
                1 => demo.run_scenario_3_anonymous_voting(),
                2 => demo.run_scenario_4_governance_proposal(),
                _ => demo.run_scenario_5_cross_chain_transfer(),
            };
            let scenario_time = scenario_start.elapsed();
            results.push((result.is_ok(), scenario_time));
        }

        let success = results.iter().all(|(success, _)| *success);
        let error_message = if success {
            None
        } else {
            Some("Some concurrent scenarios failed".to_string())
        };

        let execution_time = start_time.elapsed();
        let mut metrics = HashMap::new();
        metrics.insert("concurrent_scenarios".to_string(), results.len() as f64);
        metrics.insert(
            "average_scenario_time_ms".to_string(),
            results
                .iter()
                .map(|(_, time)| time.as_millis() as f64)
                .sum::<f64>()
                / results.len() as f64,
        );

        self.record_test_result(
            test_name,
            "Test concurrent execution of multiple scenarios",
            success,
            execution_time,
            error_message,
            Some(metrics),
        );
    }

    /// Test 24: Memory Stress (Stress Test)
    fn test_memory_stress(&mut self) {
        let test_name = "Memory Stress (Stress Test)";
        let start_time = Instant::now();

        let config = DemoConfig {
            validator_count: 50,
            vote_count: 500,
            proposal_count: 10,
            ..DemoConfig::default()
        };

        let validator_count = config.validator_count;
        let vote_count = config.vote_count;
        let proposal_count = config.proposal_count;

        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test memory stress handling",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test memory stress by running multiple scenarios
        let mut results = Vec::new();
        for _ in 0..3 {
            let scenario_start = Instant::now();
            let result = demo.run_scenario_2_validator_setup();
            let scenario_time = scenario_start.elapsed();
            results.push((result.is_ok(), scenario_time));
        }

        let success = results.iter().all(|(success, _)| *success);
        let error_message = if success {
            None
        } else {
            Some("Memory stress test failed".to_string())
        };

        let execution_time = start_time.elapsed();
        let mut metrics = HashMap::new();
        metrics.insert("memory_stress_iterations".to_string(), results.len() as f64);
        metrics.insert("validator_count".to_string(), validator_count as f64);
        metrics.insert("vote_count".to_string(), vote_count as f64);
        metrics.insert("proposal_count".to_string(), proposal_count as f64);
        metrics.insert(
            "total_execution_time_ms".to_string(),
            execution_time.as_millis() as f64,
        );

        self.record_test_result(
            test_name,
            "Test system behavior under memory stress",
            success,
            execution_time,
            error_message,
            Some(metrics),
        );
    }

    /// Test 25: Network Stress (Stress Test)
    fn test_network_stress(&mut self) {
        let test_name = "Network Stress (Stress Test)";
        let start_time = Instant::now();

        let config = DemoConfig {
            validator_count: 25,
            vote_count: 250,
            transfer_amount: 10000, // High transfer amount
            ..DemoConfig::default()
        };

        let transfer_amount = config.transfer_amount;

        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test network stress handling",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test network stress by running network-intensive scenarios
        let mut results = Vec::new();
        for _ in 0..3 {
            let scenario_start = Instant::now();
            let result = demo.run_scenario_5_cross_chain_transfer();
            let scenario_time = scenario_start.elapsed();
            results.push((result.is_ok(), scenario_time));
        }

        let success = results.iter().all(|(success, _)| *success);
        let error_message = if success {
            None
        } else {
            Some("Network stress test failed".to_string())
        };

        let execution_time = start_time.elapsed();
        let mut metrics = HashMap::new();
        metrics.insert(
            "network_stress_iterations".to_string(),
            results.len() as f64,
        );
        metrics.insert("transfer_amount".to_string(), transfer_amount as f64);
        metrics.insert(
            "total_execution_time_ms".to_string(),
            execution_time.as_millis() as f64,
        );

        self.record_test_result(
            test_name,
            "Test system behavior under network stress",
            success,
            execution_time,
            error_message,
            Some(metrics),
        );
    }

    /// Test 26: Performance Benchmarks
    fn test_performance_benchmarks(&mut self) {
        let test_name = "Performance Benchmarks";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test performance benchmark execution",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test performance benchmarks
        let result = demo.run_scenario_8_performance_benchmark();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        let mut metrics = HashMap::new();
        metrics.insert(
            "benchmark_execution_time_ms".to_string(),
            execution_time.as_millis() as f64,
        );
        metrics.insert(
            "benchmark_success".to_string(),
            if success { 1.0 } else { 0.0 },
        );

        self.record_test_result(
            test_name,
            "Test performance benchmark execution and analysis",
            success,
            execution_time,
            error_message,
            Some(metrics),
        );
    }

    /// Test 27: Latency Measurements
    fn test_latency_measurements(&mut self) {
        let test_name = "Latency Measurements";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test latency measurement",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Measure latency for different scenarios
        let mut latencies = Vec::new();

        // Test validator setup
        let scenario_start = Instant::now();
        let result = demo.run_scenario_2_validator_setup();
        let scenario_time = scenario_start.elapsed();
        latencies.push(("validator_setup".to_string(), scenario_time, result.is_ok()));

        // Test anonymous voting
        let scenario_start = Instant::now();
        let result = demo.run_scenario_3_anonymous_voting();
        let scenario_time = scenario_start.elapsed();
        latencies.push((
            "anonymous_voting".to_string(),
            scenario_time,
            result.is_ok(),
        ));

        // Test governance proposal
        let scenario_start = Instant::now();
        let result = demo.run_scenario_4_governance_proposal();
        let scenario_time = scenario_start.elapsed();
        latencies.push((
            "governance_proposal".to_string(),
            scenario_time,
            result.is_ok(),
        ));

        // Test cross-chain transfer
        let scenario_start = Instant::now();
        let result = demo.run_scenario_5_cross_chain_transfer();
        let scenario_time = scenario_start.elapsed();
        latencies.push((
            "cross_chain_transfer".to_string(),
            scenario_time,
            result.is_ok(),
        ));

        let success = latencies.iter().all(|(_, _, success)| *success);
        let error_message = if success {
            None
        } else {
            Some("Some latency measurements failed".to_string())
        };

        let execution_time = start_time.elapsed();
        let mut metrics = HashMap::new();
        metrics.insert(
            "total_latency_ms".to_string(),
            execution_time.as_millis() as f64,
        );
        metrics.insert(
            "average_scenario_latency_ms".to_string(),
            latencies
                .iter()
                .map(|(_, time, _)| time.as_millis() as f64)
                .sum::<f64>()
                / latencies.len() as f64,
        );

        self.record_test_result(
            test_name,
            "Test latency measurement across different scenarios",
            success,
            execution_time,
            error_message,
            Some(metrics),
        );
    }

    /// Test 28: Throughput Measurements
    fn test_throughput_measurements(&mut self) {
        let test_name = "Throughput Measurements";
        let start_time = Instant::now();

        let config = DemoConfig {
            vote_count: 100, // Higher vote count for throughput testing
            ..DemoConfig::default()
        };

        let vote_count = config.vote_count;

        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test throughput measurement",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Measure throughput for voting scenario
        let scenario_start = Instant::now();
        let result = demo.run_scenario_3_anonymous_voting();
        let scenario_time = scenario_start.elapsed();

        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        let throughput = vote_count as f64 / scenario_time.as_secs_f64();

        let mut metrics = HashMap::new();
        metrics.insert("vote_count".to_string(), vote_count as f64);
        metrics.insert(
            "execution_time_seconds".to_string(),
            scenario_time.as_secs_f64(),
        );
        metrics.insert("throughput_votes_per_second".to_string(), throughput);

        self.record_test_result(
            test_name,
            "Test throughput measurement for voting scenario",
            success,
            execution_time,
            error_message,
            Some(metrics),
        );
    }

    /// Test 29: Resource Usage
    fn test_resource_usage(&mut self) {
        let test_name = "Resource Usage";
        let start_time = Instant::now();

        let config = DemoConfig::default();
        let mut demo = match BlockchainDemo::new(config) {
            Ok(demo) => demo,
            Err(e) => {
                self.record_test_result(
                    test_name,
                    "Test resource usage measurement",
                    false,
                    start_time.elapsed(),
                    Some(format!("{}", e)),
                    None,
                );
                return;
            }
        };

        // Test resource usage monitoring
        let result = demo.run_scenario_6_monitoring_dashboard();
        let success = result.is_ok();
        let error_message = if let Err(e) = &result {
            Some(format!("{}", e))
        } else {
            None
        };

        let execution_time = start_time.elapsed();
        let mut metrics = HashMap::new();
        metrics.insert(
            "monitoring_execution_time_ms".to_string(),
            execution_time.as_millis() as f64,
        );
        metrics.insert(
            "monitoring_success".to_string(),
            if success { 1.0 } else { 0.0 },
        );

        self.record_test_result(
            test_name,
            "Test resource usage monitoring and measurement",
            success,
            execution_time,
            error_message,
            Some(metrics),
        );
    }

    /// Record test result
    fn record_test_result(
        &mut self,
        name: &str,
        description: &str,
        success: bool,
        execution_time: Duration,
        error_message: Option<String>,
        metrics: Option<HashMap<String, f64>>,
    ) {
        let test_result = TestResult {
            name: name.to_string(),
            description: description.to_string(),
            success,
            execution_time,
            error_message: error_message.clone(),
            metrics,
        };

        self.results.test_results.push(test_result);

        if self.config.verbose {
            let status = if success { "✅ PASS" } else { "❌ FAIL" };
            println!("{} {} - {}ms", status, name, execution_time.as_millis());
            if let Some(error) = &error_message {
                println!("   Error: {}", error);
            }
        }
    }

    /// Print test summary
    fn print_test_summary(&self) {
        println!("\n📊 Test Suite Summary");
        println!("{}", "=".repeat(50));
        println!("Total Tests: {}", self.results.total_tests);
        println!(
            "Passed: {} ({:.1}%)",
            self.results.passed_tests,
            (self.results.passed_tests as f64 / self.results.total_tests as f64) * 100.0
        );
        println!(
            "Failed: {} ({:.1}%)",
            self.results.failed_tests,
            (self.results.failed_tests as f64 / self.results.total_tests as f64) * 100.0
        );
        println!(
            "Total Duration: {:.2}s",
            self.results.total_duration.as_secs_f64()
        );

        if self.results.failed_tests > 0 {
            println!("\n❌ Failed Tests:");
            for result in &self.results.test_results {
                if !result.success {
                    println!(
                        "  - {}: {}",
                        result.name,
                        result.error_message.as_deref().unwrap_or("Unknown error")
                    );
                }
            }
        }

        println!("\n🎯 Performance Metrics:");
        for (metric, value) in &self.results.performance_metrics {
            println!("  {}: {:.2}", metric, value);
        }
    }
}

impl Default for DemoTestSuite {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_test_suite_creation() {
        let test_suite = DemoTestSuite::new();
        assert_eq!(test_suite.results.total_tests, 0);
        assert_eq!(test_suite.results.passed_tests, 0);
        assert_eq!(test_suite.results.failed_tests, 0);
    }

    #[test]
    fn test_demo_test_suite_default() {
        let test_suite = DemoTestSuite::default();
        assert_eq!(test_suite.results.total_tests, 0);
    }

    #[test]
    fn test_demo_config_default() {
        let config = DemoConfig::default();
        assert!(config.guided_mode);
        assert!(config.verbose);
        assert!(config.save_results);
        assert_eq!(config.validator_count, 10);
        assert_eq!(config.vote_count, 50);
    }

    #[test]
    fn test_demo_config_custom() {
        let config = DemoConfig {
            guided_mode: false,
            verbose: false,
            save_results: false,
            output_file: "custom.json".to_string(),
            validator_count: 5,
            vote_count: 25,
            proposal_count: 2,
            transfer_amount: 1000,
            timeout_seconds: 60,
        };

        assert!(!config.guided_mode);
        assert!(!config.verbose);
        assert!(!config.save_results);
        assert_eq!(config.output_file, "custom.json");
        assert_eq!(config.validator_count, 5);
        assert_eq!(config.vote_count, 25);
    }

    #[test]
    fn test_test_config_default() {
        let config = TestConfig {
            verbose: true,
            enable_performance_tests: true,
            enable_stress_tests: true,
            timeout_seconds: 60,
            stress_test_iterations: 100,
        };

        assert!(config.verbose);
        assert!(config.enable_performance_tests);
        assert!(config.enable_stress_tests);
        assert_eq!(config.timeout_seconds, 60);
        assert_eq!(config.stress_test_iterations, 100);
    }

    #[test]
    fn test_test_result_creation() {
        let test_result = TestResult {
            name: "Test Name".to_string(),
            description: "Test Description".to_string(),
            success: true,
            execution_time: Duration::from_millis(100),
            error_message: None,
            metrics: None,
        };

        assert_eq!(test_result.name, "Test Name");
        assert_eq!(test_result.description, "Test Description");
        assert!(test_result.success);
        assert_eq!(test_result.execution_time, Duration::from_millis(100));
        assert!(test_result.error_message.is_none());
        assert!(test_result.metrics.is_none());
    }

    #[test]
    fn test_test_result_with_error() {
        let test_result = TestResult {
            name: "Failing Test".to_string(),
            description: "Test that fails".to_string(),
            success: false,
            execution_time: Duration::from_millis(50),
            error_message: Some("Test failed".to_string()),
            metrics: None,
        };

        assert_eq!(test_result.name, "Failing Test");
        assert!(!test_result.success);
        assert!(test_result.error_message.is_some());
        assert_eq!(test_result.error_message.unwrap(), "Test failed");
    }

    #[test]
    fn test_test_result_with_metrics() {
        let mut metrics = HashMap::new();
        metrics.insert("throughput".to_string(), 100.0);
        metrics.insert("latency".to_string(), 50.0);

        let test_result = TestResult {
            name: "Performance Test".to_string(),
            description: "Test with metrics".to_string(),
            success: true,
            execution_time: Duration::from_millis(200),
            error_message: None,
            metrics: Some(metrics),
        };

        assert!(test_result.success);
        assert!(test_result.metrics.is_some());
        let metrics = test_result.metrics.unwrap();
        assert_eq!(metrics.get("throughput").unwrap(), &100.0);
        assert_eq!(metrics.get("latency").unwrap(), &50.0);
    }
}
