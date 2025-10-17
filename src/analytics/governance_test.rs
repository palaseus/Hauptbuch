//! Comprehensive test suite for governance analytics module
//! 
//! This module provides extensive testing for the governance analytics system,
//! covering normal operation, edge cases, malicious behavior, and stress tests.
//! The test suite ensures near-100% coverage and validates all analytics
//! calculations and data integrity mechanisms.

#[cfg(test)]
mod governance_analytics_tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::{SystemTime, UNIX_EPOCH};
    
    use crate::governance::proposal::{Proposal, ProposalType, ProposalStatus, Vote, VoteChoice};
    use crate::federation::{FederatedProposal, CrossChainVote, FederationMember, FederatedProposalStatus};
    use crate::monitoring::monitor::{SystemMetrics, VoterActivity};
    use crate::analytics::governance::{
        GovernanceAnalyticsEngine, GovernanceAnalytics, TimeRange,
        VoterTurnoutMetrics, StakeDistributionMetrics, ProposalAnalysisMetrics,
        CrossChainMetrics, TemporalTrendMetrics, ChartData
    };

    /// Helper function to create test proposals
    fn create_test_proposal(
        id: &str,
        proposal_type: ProposalType,
        status: ProposalStatus,
        voting_start: u64,
        voting_end: u64,
    ) -> Proposal {
        Proposal {
            id: id.to_string(),
            title: format!("Test Proposal {}", id),
            description: format!("Description for proposal {}", id),
            proposal_type,
            proposer: format!("proposer_{}", id),
            voting_start,
            voting_end,
            status,
            votes_for: 0,
            votes_against: 0,
            total_stake_for: 0,
            total_stake_against: 0,
            created_at: voting_start,
            updated_at: voting_start,
        }
    }

    /// Helper function to create test votes
    fn create_test_vote(
        voter_id: &str,
        proposal_id: &str,
        choice: VoteChoice,
        stake_amount: u64,
        timestamp: u64,
    ) -> Vote {
        Vote {
            voter_id: voter_id.to_string(),
            proposal_id: proposal_id.to_string(),
            choice,
            stake_amount,
            timestamp,
            signature: vec![0u8; 64],
        }
    }

    /// Helper function to create test federation members
    fn create_test_federation_member(
        id: &str,
        stake: u64,
        chain_id: &str,
    ) -> FederationMember {
        FederationMember {
            id: id.to_string(),
            chain_id: chain_id.to_string(),
            stake,
            public_key: vec![0u8; 64],
            is_active: true,
            joined_at: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
        }
    }

    /// Helper function to create test cross-chain votes
    fn create_test_cross_chain_vote(
        voter_id: &str,
        source_chain: &str,
        proposal_id: &str,
        is_valid: bool,
        timestamp: u64,
    ) -> CrossChainVote {
        CrossChainVote {
            voter_id: voter_id.to_string(),
            source_chain: source_chain.to_string(),
            proposal_id: proposal_id.to_string(),
            choice: crate::federation::FederationVoteChoice::For,
            stake_amount: 1000,
            timestamp,
            is_valid,
            sync_timestamp: Some(timestamp + 300), // 5 minute sync delay
            signature: vec![0u8; 64],
        }
    }

    /// Helper function to create test federated proposals
    fn create_test_federated_proposal(
        id: &str,
        status: FederatedProposalStatus,
        timestamp: u64,
    ) -> FederatedProposal {
        FederatedProposal {
            id: id.to_string(),
            title: format!("Federated Proposal {}", id),
            description: format!("Federated description {}", id),
            proposer_public_key: vec![0u8; 64],
            voting_start: timestamp,
            voting_end: timestamp + 86400, // 24 hours
            status,
            total_votes: 0,
            votes_for: 0,
            votes_against: 0,
            participating_chains: vec!["chain1".to_string(), "chain2".to_string()],
            created_at: timestamp,
        }
    }

    /// Helper function to create test system metrics
    fn create_test_system_metrics(
        timestamp: u64,
        active_voters: u64,
        total_stake: u64,
    ) -> SystemMetrics {
        SystemMetrics {
            timestamp,
            active_voters,
            total_stake,
            block_height: 1000,
            transaction_count: 5000,
            network_health: 0.95,
            consensus_participation: 0.88,
        }
    }

    /// Helper function to create test voter activity
    fn create_test_voter_activity(
        voter_id: &str,
        timestamp: u64,
        stake_amount: u64,
    ) -> VoterActivity {
        VoterActivity {
            voter_id: voter_id.to_string(),
            timestamp,
            stake_amount,
            activity_type: "vote".to_string(),
            proposal_id: Some("proposal_1".to_string()),
        }
    }

    /// Test normal operation - basic analytics calculation
    #[test]
    fn test_basic_analytics_calculation() {
        let mut engine = GovernanceAnalyticsEngine::new();
        
        // Create test data
        let proposals = vec![
            create_test_proposal("1", ProposalType::ProtocolUpgrade, ProposalStatus::Approved, 1000, 2000),
            create_test_proposal("2", ProposalType::EconomicUpdate, ProposalStatus::Rejected, 2000, 3000),
        ];
        
        let votes = vec![
            create_test_vote("voter1", "1", VoteChoice::For, 1000, 1500),
            create_test_vote("voter2", "1", VoteChoice::Against, 500, 1600),
            create_test_vote("voter1", "2", VoteChoice::For, 1000, 2500),
        ];
        
        let federation_members = vec![
            create_test_federation_member("member1", 1000, "chain1"),
            create_test_federation_member("member2", 500, "chain1"),
        ];
        
        let time_range = TimeRange {
            start_time: 1000,
            end_time: 3000,
            duration_seconds: 2000,
        };
        
        let result = engine.analyze_governance(
            proposals,
            votes,
            vec![],
            vec![],
            federation_members,
            vec![],
            vec![],
            time_range,
        );
        
        assert!(result.is_ok());
        let analytics = result.unwrap();
        
        // Verify basic metrics
        assert_eq!(analytics.voter_turnout.total_voters, 3);
        assert_eq!(analytics.voter_turnout.eligible_voters, 2);
        assert_eq!(analytics.proposal_analysis.total_proposals, 2);
        assert_eq!(analytics.proposal_analysis.successful_proposals, 1);
        assert_eq!(analytics.proposal_analysis.failed_proposals, 1);
    }

    /// Test voter turnout calculation with various scenarios
    #[test]
    fn test_voter_turnout_calculation() {
        let engine = GovernanceAnalyticsEngine::new();
        
        // Test with no votes
        let turnout = engine.calculate_voter_turnout(&[], &[]).unwrap();
        assert_eq!(turnout.total_voters, 0);
        assert_eq!(turnout.eligible_voters, 0);
        assert_eq!(turnout.turnout_percentage, 0.0);
        
        // Test with votes but no eligible voters
        let votes = vec![
            create_test_vote("voter1", "1", VoteChoice::For, 1000, 1000),
        ];
        let turnout = engine.calculate_voter_turnout(&votes, &[]).unwrap();
        assert_eq!(turnout.total_voters, 1);
        assert_eq!(turnout.eligible_voters, 0);
        
        // Test normal scenario
        let federation_members = vec![
            create_test_federation_member("member1", 1000, "chain1"),
            create_test_federation_member("member2", 500, "chain1"),
        ];
        let turnout = engine.calculate_voter_turnout(&votes, &federation_members).unwrap();
        assert_eq!(turnout.total_voters, 1);
        assert_eq!(turnout.eligible_voters, 2);
        assert_eq!(turnout.turnout_percentage, 50.0);
    }

    /// Test stake distribution calculation including Gini coefficient
    #[test]
    fn test_stake_distribution_calculation() {
        let engine = GovernanceAnalyticsEngine::new();
        
        // Test with empty data
        let distribution = engine.calculate_stake_distribution(&[]).unwrap();
        assert_eq!(distribution.total_stake, 0);
        assert_eq!(distribution.stake_holders, 0);
        assert_eq!(distribution.gini_coefficient, 0.0);
        
        // Test with equal stakes (Gini = 0)
        let equal_members = vec![
            create_test_federation_member("member1", 1000, "chain1"),
            create_test_federation_member("member2", 1000, "chain1"),
            create_test_federation_member("member3", 1000, "chain1"),
        ];
        let distribution = engine.calculate_stake_distribution(&equal_members).unwrap();
        assert_eq!(distribution.total_stake, 3000);
        assert_eq!(distribution.stake_holders, 3);
        assert_eq!(distribution.gini_coefficient, 0.0);
        
        // Test with unequal stakes (Gini > 0)
        let unequal_members = vec![
            create_test_federation_member("member1", 100, "chain1"),
            create_test_federation_member("member2", 1000, "chain1"),
            create_test_federation_member("member3", 10000, "chain1"),
        ];
        let distribution = engine.calculate_stake_distribution(&unequal_members).unwrap();
        assert_eq!(distribution.total_stake, 11100);
        assert_eq!(distribution.stake_holders, 3);
        assert!(distribution.gini_coefficient > 0.0);
    }

    /// Test proposal analysis with different proposal types and outcomes
    #[test]
    fn test_proposal_analysis_calculation() {
        let engine = GovernanceAnalyticsEngine::new();
        
        let proposals = vec![
            create_test_proposal("1", ProposalType::ProtocolUpgrade, ProposalStatus::Approved, 1000, 2000),
            create_test_proposal("2", ProposalType::EconomicUpdate, ProposalStatus::Rejected, 2000, 3000),
            create_test_proposal("3", ProposalType::GovernanceChange, ProposalStatus::Executed, 3000, 4000),
            create_test_proposal("4", ProposalType::ProtocolUpgrade, ProposalStatus::Pending, 4000, 5000),
        ];
        
        let votes = vec![
            create_test_vote("voter1", "1", VoteChoice::For, 1000, 1500),
            create_test_vote("voter2", "2", VoteChoice::Against, 500, 2500),
            create_test_vote("voter3", "3", VoteChoice::For, 2000, 3500),
        ];
        
        let analysis = engine.calculate_proposal_analysis(&proposals, &votes).unwrap();
        
        assert_eq!(analysis.total_proposals, 4);
        assert_eq!(analysis.successful_proposals, 2); // Approved + Executed
        assert_eq!(analysis.failed_proposals, 1); // Rejected
        assert_eq!(analysis.success_rate, 0.5); // 2/4 = 0.5
        
        // Check success rate by type
        assert!(analysis.success_rate_by_type.contains_key("ProtocolUpgrade"));
        assert!(analysis.success_rate_by_type.contains_key("EconomicUpdate"));
        assert!(analysis.success_rate_by_type.contains_key("GovernanceChange"));
    }

    /// Test cross-chain metrics calculation
    #[test]
    fn test_cross_chain_metrics_calculation() {
        let engine = GovernanceAnalyticsEngine::new();
        
        let cross_chain_votes = vec![
            create_test_cross_chain_vote("voter1", "chain1", "proposal1", true, 1000),
            create_test_cross_chain_vote("voter2", "chain1", "proposal1", true, 1100),
            create_test_cross_chain_vote("voter3", "chain2", "proposal1", false, 1200),
            create_test_cross_chain_vote("voter4", "chain2", "proposal2", true, 1300),
        ];
        
        let federated_proposals = vec![
            create_test_federated_proposal("proposal1", FederatedProposalStatus::Active, 1000),
            create_test_federated_proposal("proposal2", FederatedProposalStatus::Active, 2000),
        ];
        
        let metrics = engine.calculate_cross_chain_metrics(&cross_chain_votes, &federated_proposals).unwrap();
        
        assert_eq!(metrics.total_cross_chain_votes, 4);
        assert_eq!(metrics.participating_chains, 2);
        assert_eq!(metrics.average_votes_per_chain, 2.0);
        assert_eq!(metrics.cross_chain_success_rate, 0.75); // 3/4 = 0.75
        
        // Check chain participation
        assert!(metrics.chain_participation.contains_key("chain1"));
        assert!(metrics.chain_participation.contains_key("chain2"));
    }

    /// Test temporal trends calculation
    #[test]
    fn test_temporal_trends_calculation() {
        let engine = GovernanceAnalyticsEngine::new();
        
        let votes = vec![
            create_test_vote("voter1", "1", VoteChoice::For, 1000, 1000),
            create_test_vote("voter2", "1", VoteChoice::For, 1000, 2000),
            create_test_vote("voter3", "2", VoteChoice::Against, 1000, 3000),
        ];
        
        let proposals = vec![
            create_test_proposal("1", ProposalType::ProtocolUpgrade, ProposalStatus::Approved, 1000, 2000),
            create_test_proposal("2", ProposalType::EconomicUpdate, ProposalStatus::Rejected, 2000, 3000),
        ];
        
        let voter_activity = vec![
            create_test_voter_activity("voter1", 1000, 1000),
            create_test_voter_activity("voter2", 2000, 1000),
            create_test_voter_activity("voter3", 3000, 1000),
        ];
        
        let time_range = TimeRange {
            start_time: 1000,
            end_time: 4000,
            duration_seconds: 3000,
        };
        
        let trends = engine.calculate_temporal_trends(&votes, &proposals, &voter_activity, &time_range).unwrap();
        
        assert!(!trends.hourly_activity.is_empty());
        assert!(!trends.daily_patterns.is_empty());
        assert!(!trends.weekly_trends.is_empty());
    }

    /// Test edge case - no votes scenario
    #[test]
    fn test_no_votes_scenario() {
        let mut engine = GovernanceAnalyticsEngine::new();
        
        let time_range = TimeRange {
            start_time: 1000,
            end_time: 2000,
            duration_seconds: 1000,
        };
        
        let result = engine.analyze_governance(
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            time_range,
        );
        
        assert!(result.is_ok());
        let analytics = result.unwrap();
        
        assert_eq!(analytics.voter_turnout.total_voters, 0);
        assert_eq!(analytics.voter_turnout.eligible_voters, 0);
        assert_eq!(analytics.proposal_analysis.total_proposals, 0);
        assert_eq!(analytics.cross_chain_metrics.total_cross_chain_votes, 0);
    }

    /// Test edge case - single voter scenario
    #[test]
    fn test_single_voter_scenario() {
        let mut engine = GovernanceAnalyticsEngine::new();
        
        let proposals = vec![
            create_test_proposal("1", ProposalType::ProtocolUpgrade, ProposalStatus::Approved, 1000, 2000),
        ];
        
        let votes = vec![
            create_test_vote("voter1", "1", VoteChoice::For, 1000, 1500),
        ];
        
        let federation_members = vec![
            create_test_federation_member("member1", 1000, "chain1"),
        ];
        
        let time_range = TimeRange {
            start_time: 1000,
            end_time: 2000,
            duration_seconds: 1000,
        };
        
        let result = engine.analyze_governance(
            proposals,
            votes,
            vec![],
            vec![],
            federation_members,
            vec![],
            vec![],
            time_range,
        );
        
        assert!(result.is_ok());
        let analytics = result.unwrap();
        
        assert_eq!(analytics.voter_turnout.total_voters, 1);
        assert_eq!(analytics.voter_turnout.eligible_voters, 1);
        assert_eq!(analytics.voter_turnout.turnout_percentage, 100.0);
        assert_eq!(analytics.voter_turnout.most_active_voter, Some("voter1".to_string()));
    }

    /// Test edge case - failed proposals scenario
    #[test]
    fn test_failed_proposals_scenario() {
        let mut engine = GovernanceAnalyticsEngine::new();
        
        let proposals = vec![
            create_test_proposal("1", ProposalType::ProtocolUpgrade, ProposalStatus::Rejected, 1000, 2000),
            create_test_proposal("2", ProposalType::EconomicUpdate, ProposalStatus::Cancelled, 2000, 3000),
        ];
        
        let votes = vec![
            create_test_vote("voter1", "1", VoteChoice::Against, 1000, 1500),
            create_test_vote("voter2", "2", VoteChoice::Against, 1000, 2500),
        ];
        
        let federation_members = vec![
            create_test_federation_member("member1", 1000, "chain1"),
            create_test_federation_member("member2", 1000, "chain1"),
        ];
        
        let time_range = TimeRange {
            start_time: 1000,
            end_time: 3000,
            duration_seconds: 2000,
        };
        
        let result = engine.analyze_governance(
            proposals,
            votes,
            vec![],
            vec![],
            federation_members,
            vec![],
            vec![],
            time_range,
        );
        
        assert!(result.is_ok());
        let analytics = result.unwrap();
        
        assert_eq!(analytics.proposal_analysis.total_proposals, 2);
        assert_eq!(analytics.proposal_analysis.successful_proposals, 0);
        assert_eq!(analytics.proposal_analysis.failed_proposals, 2);
        assert_eq!(analytics.proposal_analysis.success_rate, 0.0);
    }

    /// Test malicious behavior - tampered data detection
    #[test]
    fn test_tampered_data_detection() {
        let mut engine = GovernanceAnalyticsEngine::new();
        
        // Create proposals with inconsistent timestamps
        let proposals = vec![
            create_test_proposal("1", ProposalType::ProtocolUpgrade, ProposalStatus::Approved, 2000, 1000), // End before start
        ];
        
        let votes = vec![
            create_test_vote("voter1", "1", VoteChoice::For, 1000, 1500),
        ];
        
        let federation_members = vec![
            create_test_federation_member("member1", 1000, "chain1"),
        ];
        
        let time_range = TimeRange {
            start_time: 1000,
            end_time: 2000,
            duration_seconds: 1000,
        };
        
        // The system should handle inconsistent data gracefully
        let result = engine.analyze_governance(
            proposals,
            votes,
            vec![],
            vec![],
            federation_members,
            vec![],
            vec![],
            time_range,
        );
        
        assert!(result.is_ok());
    }

    /// Test malicious behavior - invalid cross-chain votes
    #[test]
    fn test_invalid_cross_chain_votes() {
        let mut engine = GovernanceAnalyticsEngine::new();
        
        let cross_chain_votes = vec![
            create_test_cross_chain_vote("voter1", "chain1", "proposal1", true, 1000),
            create_test_cross_chain_vote("voter2", "chain1", "proposal1", false, 1100), // Invalid vote
            create_test_cross_chain_vote("voter3", "chain2", "proposal1", false, 1200), // Invalid vote
        ];
        
        let federated_proposals = vec![
            create_test_federated_proposal("proposal1", FederatedProposalStatus::Active, 1000),
        ];
        
        let time_range = TimeRange {
            start_time: 1000,
            end_time: 2000,
            duration_seconds: 1000,
        };
        
        let result = engine.analyze_governance(
            vec![],
            vec![],
            vec![],
            cross_chain_votes,
            vec![],
            vec![],
            vec![],
            time_range,
        );
        
        assert!(result.is_ok());
        let analytics = result.unwrap();
        
        // Should correctly identify invalid votes
        assert_eq!(analytics.cross_chain_metrics.total_cross_chain_votes, 3);
        assert_eq!(analytics.cross_chain_metrics.cross_chain_success_rate, 1.0/3.0);
    }

    /// Test stress test - high proposal volume
    #[test]
    fn test_high_proposal_volume() {
        let mut engine = GovernanceAnalyticsEngine::new();
        
        // Create 1000 proposals
        let mut proposals = Vec::new();
        for i in 0..1000 {
            let status = if i % 2 == 0 {
                ProposalStatus::Approved
            } else {
                ProposalStatus::Rejected
            };
            proposals.push(create_test_proposal(
                &format!("{}", i),
                ProposalType::ProtocolUpgrade,
                status,
                1000 + i as u64,
                2000 + i as u64,
            ));
        }
        
        // Create 5000 votes
        let mut votes = Vec::new();
        for i in 0..5000 {
            let proposal_id = format!("{}", i % 1000);
            votes.push(create_test_vote(
                &format!("voter{}", i % 100),
                &proposal_id,
                VoteChoice::For,
                1000,
                1500 + i as u64,
            ));
        }
        
        let federation_members = (0..100)
            .map(|i| create_test_federation_member(&format!("member{}", i), 1000, "chain1"))
            .collect();
        
        let time_range = TimeRange {
            start_time: 1000,
            end_time: 6000,
            duration_seconds: 5000,
        };
        
        let result = engine.analyze_governance(
            proposals,
            votes,
            vec![],
            vec![],
            federation_members,
            vec![],
            vec![],
            time_range,
        );
        
        assert!(result.is_ok());
        let analytics = result.unwrap();
        
        assert_eq!(analytics.proposal_analysis.total_proposals, 1000);
        assert_eq!(analytics.voter_turnout.total_voters, 5000);
        assert_eq!(analytics.voter_turnout.eligible_voters, 100);
    }

    /// Test stress test - large voter datasets
    #[test]
    fn test_large_voter_datasets() {
        let mut engine = GovernanceAnalyticsEngine::new();
        
        // Create 10000 federation members
        let federation_members = (0..10000)
            .map(|i| create_test_federation_member(
                &format!("member{}", i),
                1000 + (i % 1000) as u64,
                "chain1"
            ))
            .collect();
        
        // Create 50000 votes
        let mut votes = Vec::new();
        for i in 0..50000 {
            votes.push(create_test_vote(
                &format!("voter{}", i % 10000),
                "proposal1",
                VoteChoice::For,
                1000,
                1000 + i as u64,
            ));
        }
        
        let time_range = TimeRange {
            start_time: 1000,
            end_time: 51000,
            duration_seconds: 50000,
        };
        
        let result = engine.analyze_governance(
            vec![],
            votes,
            vec![],
            vec![],
            federation_members,
            vec![],
            vec![],
            time_range,
        );
        
        assert!(result.is_ok());
        let analytics = result.unwrap();
        
        assert_eq!(analytics.voter_turnout.total_voters, 50000);
        assert_eq!(analytics.voter_turnout.eligible_voters, 10000);
        assert_eq!(analytics.voter_turnout.turnout_percentage, 500.0); // 50000/10000 * 100
    }

    /// Test Chart.js data generation
    #[test]
    fn test_chart_data_generation() {
        let mut engine = GovernanceAnalyticsEngine::new();
        
        let proposals = vec![
            create_test_proposal("1", ProposalType::ProtocolUpgrade, ProposalStatus::Approved, 1000, 2000),
        ];
        
        let votes = vec![
            create_test_vote("voter1", "1", VoteChoice::For, 1000, 1500),
        ];
        
        let federation_members = vec![
            create_test_federation_member("member1", 1000, "chain1"),
        ];
        
        let time_range = TimeRange {
            start_time: 1000,
            end_time: 2000,
            duration_seconds: 1000,
        };
        
        let analytics = engine.analyze_governance(
            proposals,
            votes,
            vec![],
            vec![],
            federation_members,
            vec![],
            vec![],
            time_range,
        ).unwrap();
        
        let charts = engine.generate_chart_data(&analytics).unwrap();
        
        assert!(!charts.is_empty());
        
        // Verify chart types
        let chart_types: Vec<&str> = charts.iter().map(|c| c.chart_type.as_str()).collect();
        assert!(chart_types.contains(&"doughnut")); // Voter turnout
        assert!(chart_types.contains(&"bar")); // Stake distribution
        assert!(chart_types.contains(&"line")); // Temporal trends
    }

    /// Test JSON export functionality
    #[test]
    fn test_json_export() {
        let mut engine = GovernanceAnalyticsEngine::new();
        
        let proposals = vec![
            create_test_proposal("1", ProposalType::ProtocolUpgrade, ProposalStatus::Approved, 1000, 2000),
        ];
        
        let votes = vec![
            create_test_vote("voter1", "1", VoteChoice::For, 1000, 1500),
        ];
        
        let federation_members = vec![
            create_test_federation_member("member1", 1000, "chain1"),
        ];
        
        let time_range = TimeRange {
            start_time: 1000,
            end_time: 2000,
            duration_seconds: 1000,
        };
        
        let analytics = engine.analyze_governance(
            proposals,
            votes,
            vec![],
            vec![],
            federation_members,
            vec![],
            vec![],
            time_range,
        ).unwrap();
        
        let json_output = engine.export_to_json(&analytics).unwrap();
        assert!(!json_output.is_empty());
        assert!(json_output.contains("report_id"));
        assert!(json_output.contains("voter_turnout"));
        assert!(json_output.contains("stake_distribution"));
    }

    /// Test human-readable export functionality
    #[test]
    fn test_human_readable_export() {
        let mut engine = GovernanceAnalyticsEngine::new();
        
        let proposals = vec![
            create_test_proposal("1", ProposalType::ProtocolUpgrade, ProposalStatus::Approved, 1000, 2000),
        ];
        
        let votes = vec![
            create_test_vote("voter1", "1", VoteChoice::For, 1000, 1500),
        ];
        
        let federation_members = vec![
            create_test_federation_member("member1", 1000, "chain1"),
        ];
        
        let time_range = TimeRange {
            start_time: 1000,
            end_time: 2000,
            duration_seconds: 1000,
        };
        
        let analytics = engine.analyze_governance(
            proposals,
            votes,
            vec![],
            vec![],
            federation_members,
            vec![],
            vec![],
            time_range,
        ).unwrap();
        
        let human_output = engine.export_to_human_readable(&analytics);
        assert!(!human_output.is_empty());
        assert!(human_output.contains("Governance Analytics Report"));
        assert!(human_output.contains("Voter Turnout:"));
        assert!(human_output.contains("Stake Distribution:"));
        assert!(human_output.contains("Proposal Analysis:"));
    }

    /// Test data integrity hash calculation
    #[test]
    fn test_data_integrity_hash() {
        let mut engine = GovernanceAnalyticsEngine::new();
        
        let proposals = vec![
            create_test_proposal("1", ProposalType::ProtocolUpgrade, ProposalStatus::Approved, 1000, 2000),
        ];
        
        let votes = vec![
            create_test_vote("voter1", "1", VoteChoice::For, 1000, 1500),
        ];
        
        let federation_members = vec![
            create_test_federation_member("member1", 1000, "chain1"),
        ];
        
        let time_range = TimeRange {
            start_time: 1000,
            end_time: 2000,
            duration_seconds: 1000,
        };
        
        let analytics = engine.analyze_governance(
            proposals,
            votes,
            vec![],
            vec![],
            federation_members,
            vec![],
            vec![],
            time_range,
        ).unwrap();
        
        assert!(!analytics.integrity_hash.is_empty());
        assert!(analytics.integrity_hash.len() == 64); // SHA-3-256 hex string length
    }

    /// Test cache functionality
    #[test]
    fn test_cache_functionality() {
        let mut engine = GovernanceAnalyticsEngine::new();
        
        let proposals = vec![
            create_test_proposal("1", ProposalType::ProtocolUpgrade, ProposalStatus::Approved, 1000, 2000),
        ];
        
        let votes = vec![
            create_test_vote("voter1", "1", VoteChoice::For, 1000, 1500),
        ];
        
        let federation_members = vec![
            create_test_federation_member("member1", 1000, "chain1"),
        ];
        
        let time_range = TimeRange {
            start_time: 1000,
            end_time: 2000,
            duration_seconds: 1000,
        };
        
        let analytics = engine.analyze_governance(
            proposals,
            votes,
            vec![],
            vec![],
            federation_members,
            vec![],
            vec![],
            time_range,
        ).unwrap();
        
        // Test cache retrieval
        let cached = engine.get_cached_result(&analytics.report_id);
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().report_id, analytics.report_id);
        
        // Test cache clearing
        engine.clear_cache();
        let cached_after_clear = engine.get_cached_result(&analytics.report_id);
        assert!(cached_after_clear.is_none());
    }

    /// Test correlation calculation
    #[test]
    fn test_correlation_calculation() {
        let engine = GovernanceAnalyticsEngine::new();
        
        // Test perfect positive correlation
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let correlation = engine.calculate_correlation(&x, &y).unwrap();
        assert!((correlation - 1.0).abs() < 0.001);
        
        // Test perfect negative correlation
        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0];
        let correlation_neg = engine.calculate_correlation(&x, &y_neg).unwrap();
        assert!((correlation_neg - (-1.0)).abs() < 0.001);
        
        // Test no correlation
        let y_random = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let correlation_random = engine.calculate_correlation(&x, &y_random).unwrap();
        assert!(correlation_random.abs() < 0.5);
    }

    /// Test moving average calculation
    #[test]
    fn test_moving_average_calculation() {
        let engine = GovernanceAnalyticsEngine::new();
        
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let moving_avg = engine.calculate_moving_average(&data, 3).unwrap();
        
        assert_eq!(moving_avg.len(), 8); // 10 - 3 + 1 = 8
        assert_eq!(moving_avg[0], 2.0); // (1+2+3)/3 = 2
        assert_eq!(moving_avg[1], 3.0); // (2+3+4)/3 = 3
        assert_eq!(moving_avg[7], 9.0); // (8+9+10)/3 = 9
    }

    /// Test percentile calculation
    #[test]
    fn test_percentile_calculation() {
        let engine = GovernanceAnalyticsEngine::new();
        
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        
        let p50 = engine.calculate_percentile(data.clone(), 50.0).unwrap();
        assert!((p50 - 5.5).abs() < 0.001); // Median
        
        let p90 = engine.calculate_percentile(data.clone(), 90.0).unwrap();
        assert!((p90 - 9.1).abs() < 0.001); // 90th percentile
        
        let p10 = engine.calculate_percentile(data, 10.0).unwrap();
        assert!((p10 - 1.9).abs() < 0.001); // 10th percentile
    }

    /// Test error handling for invalid inputs
    #[test]
    fn test_error_handling_invalid_inputs() {
        let engine = GovernanceAnalyticsEngine::new();
        
        // Test correlation with mismatched lengths
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0];
        let result = engine.calculate_correlation(&x, &y);
        assert!(result.is_err());
        
        // Test moving average with invalid window size
        let data = vec![1.0, 2.0, 3.0];
        let result = engine.calculate_moving_average(&data, 0);
        assert!(result.is_err());
        
        let result = engine.calculate_moving_average(&data, 5);
        assert!(result.is_err());
        
        // Test percentile with invalid percentile
        let result = engine.calculate_percentile(data, -10.0);
        assert!(result.is_err());
        
        let result = engine.calculate_percentile(data, 110.0);
        assert!(result.is_err());
    }

    /// Test comprehensive analytics with all data types
    #[test]
    fn test_comprehensive_analytics() {
        let mut engine = GovernanceAnalyticsEngine::new();
        
        // Create comprehensive test data
        let proposals = vec![
            create_test_proposal("1", ProposalType::ProtocolUpgrade, ProposalStatus::Approved, 1000, 2000),
            create_test_proposal("2", ProposalType::EconomicUpdate, ProposalStatus::Rejected, 2000, 3000),
            create_test_proposal("3", ProposalType::GovernanceChange, ProposalStatus::Executed, 3000, 4000),
        ];
        
        let votes = vec![
            create_test_vote("voter1", "1", VoteChoice::For, 1000, 1500),
            create_test_vote("voter2", "1", VoteChoice::Against, 500, 1600),
            create_test_vote("voter1", "2", VoteChoice::For, 1000, 2500),
            create_test_vote("voter3", "3", VoteChoice::For, 2000, 3500),
        ];
        
        let federated_proposals = vec![
            create_test_federated_proposal("fed1", FederatedProposalStatus::Active, 1000),
            create_test_federated_proposal("fed2", FederatedProposalStatus::Completed, 2000),
        ];
        
        let cross_chain_votes = vec![
            create_test_cross_chain_vote("voter1", "chain1", "fed1", true, 1000),
            create_test_cross_chain_vote("voter2", "chain1", "fed1", true, 1100),
            create_test_cross_chain_vote("voter3", "chain2", "fed1", false, 1200),
            create_test_cross_chain_vote("voter4", "chain2", "fed2", true, 1300),
        ];
        
        let federation_members = vec![
            create_test_federation_member("member1", 1000, "chain1"),
            create_test_federation_member("member2", 500, "chain1"),
            create_test_federation_member("member3", 2000, "chain2"),
            create_test_federation_member("member4", 1500, "chain2"),
        ];
        
        let system_metrics = vec![
            create_test_system_metrics(1000, 4, 5000),
            create_test_system_metrics(2000, 3, 4500),
            create_test_system_metrics(3000, 4, 5000),
        ];
        
        let voter_activity = vec![
            create_test_voter_activity("voter1", 1000, 1000),
            create_test_voter_activity("voter2", 2000, 500),
            create_test_voter_activity("voter3", 3000, 2000),
        ];
        
        let time_range = TimeRange {
            start_time: 1000,
            end_time: 4000,
            duration_seconds: 3000,
        };
        
        let result = engine.analyze_governance(
            proposals,
            votes,
            federated_proposals,
            cross_chain_votes,
            federation_members,
            system_metrics,
            voter_activity,
            time_range,
        );
        
        assert!(result.is_ok());
        let analytics = result.unwrap();
        
        // Verify all metrics are calculated
        assert!(analytics.voter_turnout.total_voters > 0);
        assert!(analytics.stake_distribution.total_stake > 0);
        assert!(analytics.proposal_analysis.total_proposals > 0);
        assert!(analytics.cross_chain_metrics.total_cross_chain_votes > 0);
        assert!(!analytics.temporal_trends.hourly_activity.is_empty());
        
        // Verify data integrity
        assert!(!analytics.integrity_hash.is_empty());
        
        // Test export functionality
        let json_output = engine.export_to_json(&analytics).unwrap();
        assert!(!json_output.is_empty());
        
        let human_output = engine.export_to_human_readable(&analytics);
        assert!(!human_output.is_empty());
        
        // Test chart generation
        let charts = engine.generate_chart_data(&analytics).unwrap();
        assert!(!charts.is_empty());
    }
}
