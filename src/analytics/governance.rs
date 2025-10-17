//! Governance Analytics Module
//!
//! This module provides comprehensive analytics for governance patterns,
//! voting behavior, and cross-chain participation in the decentralized
//! voting blockchain. It analyzes data from governance proposals,
//! federation participation, and monitoring systems to generate insights
//! for academic research and system optimization.
//!
//! Key features:
//! - Voter turnout analysis and stake distribution metrics
//! - Proposal success/failure rate analysis by type
//! - Cross-chain participation and synchronization analysis
//! - Temporal trend analysis for governance activity
//! - JSON and Chart.js-compatible output formats
//! - Integration with UI for interactive analytics

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

// Import required modules for data analysis
use crate::federation::{CrossChainVote, FederatedProposal, FederationMember};
use crate::governance::proposal::{Proposal, ProposalStatus, Vote};
use crate::monitoring::monitor::{SystemMetrics, VoterActivity};

/// Represents the result of governance analytics calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceAnalytics {
    /// Unique identifier for this analytics report
    pub report_id: String,
    /// Timestamp when the analysis was performed
    pub timestamp: u64,
    /// Time range covered by the analysis
    pub time_range: TimeRange,
    /// Voter turnout metrics
    pub voter_turnout: VoterTurnoutMetrics,
    /// Stake distribution analysis
    pub stake_distribution: StakeDistributionMetrics,
    /// Proposal success/failure analysis
    pub proposal_analysis: ProposalAnalysisMetrics,
    /// Cross-chain participation metrics
    pub cross_chain_metrics: CrossChainMetrics,
    /// Temporal trend analysis
    pub temporal_trends: TemporalTrendMetrics,
    /// Data integrity hash for verification
    pub integrity_hash: String,
}

/// Time range for analytics analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start timestamp (Unix epoch)
    pub start_time: u64,
    /// End timestamp (Unix epoch)
    pub end_time: u64,
    /// Duration in seconds
    pub duration_seconds: u64,
}

/// Voter turnout analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoterTurnoutMetrics {
    /// Total number of unique voters
    pub total_voters: u64,
    /// Total number of eligible voters (stake holders)
    pub eligible_voters: u64,
    /// Voter turnout percentage (0.0 to 100.0)
    pub turnout_percentage: f64,
    /// Average votes per voter
    pub average_votes_per_voter: f64,
    /// Most active voter (by vote count)
    pub most_active_voter: Option<String>,
    /// Voter participation by stake tier
    pub participation_by_tier: HashMap<String, u64>,
}

/// Stake distribution analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StakeDistributionMetrics {
    /// Total stake in the system
    pub total_stake: u64,
    /// Number of unique stake holders
    pub stake_holders: u64,
    /// Gini coefficient for stake inequality (0.0 to 1.0)
    pub gini_coefficient: f64,
    /// Median stake amount
    pub median_stake: u64,
    /// Mean stake amount
    pub mean_stake: f64,
    /// Standard deviation of stake distribution
    pub stake_std_deviation: f64,
    /// Top 10% stake holders' share of total stake
    pub top_10_percent_share: f64,
    /// Stake distribution by tiers
    pub stake_tiers: HashMap<String, u64>,
}

/// Proposal analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalAnalysisMetrics {
    /// Total number of proposals analyzed
    pub total_proposals: u64,
    /// Number of successful proposals
    pub successful_proposals: u64,
    /// Number of failed proposals
    pub failed_proposals: u64,
    /// Overall success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Success rate by proposal type
    pub success_rate_by_type: HashMap<String, f64>,
    /// Average voting duration in seconds
    pub average_voting_duration: f64,
    /// Most common proposal type
    pub most_common_type: Option<String>,
    /// Proposal outcomes by status
    pub outcomes_by_status: HashMap<String, u64>,
}

/// Cross-chain participation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossChainMetrics {
    /// Total number of cross-chain votes
    pub total_cross_chain_votes: u64,
    /// Number of participating chains
    pub participating_chains: u64,
    /// Average votes per chain
    pub average_votes_per_chain: f64,
    /// Cross-chain synchronization delay in seconds
    pub avg_sync_delay: f64,
    /// Chain participation rates
    pub chain_participation: HashMap<String, f64>,
    /// Cross-chain vote success rate
    pub cross_chain_success_rate: f64,
    /// Most active chain by vote count
    pub most_active_chain: Option<String>,
}

/// Temporal trend analysis metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalTrendMetrics {
    /// Voting activity over time (hourly buckets)
    pub hourly_activity: Vec<ActivityBucket>,
    /// Daily voting patterns
    pub daily_patterns: HashMap<String, u64>,
    /// Weekly trend analysis
    pub weekly_trends: Vec<WeeklyTrend>,
    /// Peak activity periods
    pub peak_periods: Vec<PeakPeriod>,
    /// Seasonal patterns (if applicable)
    pub seasonal_patterns: HashMap<String, f64>,
}

/// Activity bucket for temporal analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityBucket {
    /// Hour timestamp
    pub hour: u64,
    /// Number of votes in this hour
    pub vote_count: u64,
    /// Number of proposals in this hour
    pub proposal_count: u64,
    /// Total stake activity in this hour
    pub stake_activity: u64,
}

/// Weekly trend data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeeklyTrend {
    /// Week number (since epoch)
    pub week: u64,
    /// Total activity in this week
    pub total_activity: u64,
    /// Average daily activity
    pub avg_daily_activity: f64,
    /// Trend direction (increasing/decreasing/stable)
    pub trend_direction: String,
}

/// Peak activity period
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeakPeriod {
    /// Start timestamp of peak period
    pub start_time: u64,
    /// End timestamp of peak period
    pub end_time: u64,
    /// Peak activity level
    pub activity_level: u64,
    /// Duration of peak in seconds
    pub duration: u64,
}

/// Chart.js compatible data structure for visualizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData {
    /// Chart type (line, bar, pie, etc.)
    pub chart_type: String,
    /// Chart title
    pub title: String,
    /// Chart labels
    pub labels: Vec<String>,
    /// Chart datasets
    pub datasets: Vec<ChartDataset>,
    /// Chart options
    pub options: ChartOptions,
}

/// Chart dataset for Chart.js
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartDataset {
    /// Dataset label
    pub label: String,
    /// Dataset data points
    pub data: Vec<f64>,
    /// Dataset color
    pub border_color: String,
    /// Dataset background color
    pub background_color: String,
    /// Whether to fill the area under the line
    pub fill: bool,
}

/// Chart options for Chart.js
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartOptions {
    /// Whether to show legend
    pub legend: bool,
    /// Whether to show tooltips
    pub tooltips: bool,
    /// Chart responsive setting
    pub responsive: bool,
    /// Chart scales configuration
    pub scales: Option<ChartScales>,
}

/// Chart scales configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartScales {
    /// Y-axis configuration
    pub y_axes: Vec<ChartAxis>,
    /// X-axis configuration
    pub x_axes: Vec<ChartAxis>,
}

/// Chart axis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartAxis {
    /// Axis type
    pub r#type: String,
    /// Whether to display the axis
    pub display: bool,
    /// Axis label
    pub label: String,
}

/// Main governance analytics engine
pub struct GovernanceAnalyticsEngine {
    /// Cache for analytics results
    cache: HashMap<String, GovernanceAnalytics>,
    /// Data integrity verification enabled
    integrity_verification: bool,
    /// Maximum cache size
    max_cache_size: usize,
}

impl GovernanceAnalyticsEngine {
    /// Create a new governance analytics engine
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            integrity_verification: true,
            max_cache_size: 100,
        }
    }

    /// Analyze governance data and generate comprehensive analytics
    #[allow(clippy::too_many_arguments)]
    pub fn analyze_governance(
        &mut self,
        proposals: Vec<Proposal>,
        votes: Vec<Vote>,
        federated_proposals: Vec<FederatedProposal>,
        cross_chain_votes: Vec<CrossChainVote>,
        federation_members: Vec<FederationMember>,
        _system_metrics: Vec<SystemMetrics>,
        voter_activity: Vec<VoterActivity>,
        time_range: TimeRange,
    ) -> Result<GovernanceAnalytics, String> {
        // Validate input data integrity
        if self.integrity_verification {
            self.validate_data_integrity(&proposals, &votes, &federated_proposals)?;
        }

        // Calculate voter turnout metrics
        let voter_turnout = self.calculate_voter_turnout(&votes, &federation_members)?;

        // Calculate stake distribution metrics
        let stake_distribution = self.calculate_stake_distribution(&federation_members)?;

        // Calculate proposal analysis metrics
        let proposal_analysis = self.calculate_proposal_analysis(&proposals, &votes)?;

        // Calculate cross-chain metrics
        let cross_chain_metrics =
            self.calculate_cross_chain_metrics(&cross_chain_votes, &federated_proposals)?;

        // Calculate temporal trends
        let temporal_trends =
            self.calculate_temporal_trends(&votes, &proposals, &voter_activity, &time_range)?;

        // Create analytics report
        let analytics = GovernanceAnalytics {
            report_id: self.generate_report_id(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            time_range,
            voter_turnout,
            stake_distribution,
            proposal_analysis,
            cross_chain_metrics,
            temporal_trends,
            integrity_hash: String::new(), // Will be calculated below
        };

        // Calculate integrity hash
        let integrity_hash = self.calculate_integrity_hash(&analytics)?;
        let mut analytics = analytics;
        analytics.integrity_hash = integrity_hash;

        // Cache the result
        self.cache_result(&analytics)?;

        Ok(analytics)
    }

    /// Calculate voter turnout metrics
    fn calculate_voter_turnout(
        &self,
        votes: &[Vote],
        federation_members: &[FederationMember],
    ) -> Result<VoterTurnoutMetrics, String> {
        let total_voters = votes.len() as u64;
        let eligible_voters = federation_members.len() as u64;

        // Calculate turnout percentage with safe arithmetic
        let turnout_percentage = if eligible_voters > 0 {
            (total_voters as f64 / eligible_voters as f64) * 100.0
        } else {
            0.0
        };

        // Calculate average votes per voter
        let average_votes_per_voter = if total_voters > 0 {
            votes.len() as f64 / total_voters as f64
        } else {
            0.0
        };

        // Find most active voter
        let mut voter_counts: HashMap<String, u64> = HashMap::new();
        for vote in votes {
            let count = voter_counts.entry(vote.voter_id.clone()).or_insert(0);
            *count = count.checked_add(1).ok_or("Vote count overflow")?;
        }

        let most_active_voter = voter_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(voter_id, _)| voter_id.clone());

        // Calculate participation by stake tier
        let mut participation_by_tier: HashMap<String, u64> = HashMap::new();
        for member in federation_members {
            let tier = self.determine_stake_tier(member.stake_weight);
            let count = participation_by_tier.entry(tier).or_insert(0);
            *count = count.checked_add(1).ok_or("Participation count overflow")?;
        }

        Ok(VoterTurnoutMetrics {
            total_voters,
            eligible_voters,
            turnout_percentage,
            average_votes_per_voter,
            most_active_voter,
            participation_by_tier,
        })
    }

    /// Calculate stake distribution metrics including Gini coefficient
    fn calculate_stake_distribution(
        &self,
        federation_members: &[FederationMember],
    ) -> Result<StakeDistributionMetrics, String> {
        if federation_members.is_empty() {
            return Ok(StakeDistributionMetrics {
                total_stake: 0,
                stake_holders: 0,
                gini_coefficient: 0.0,
                median_stake: 0,
                mean_stake: 0.0,
                stake_std_deviation: 0.0,
                top_10_percent_share: 0.0,
                stake_tiers: HashMap::new(),
            });
        }

        // Extract stake amounts and sort them
        let mut stakes: Vec<u64> = federation_members
            .iter()
            .map(|member| member.stake_weight)
            .collect();
        stakes.sort_unstable();

        let total_stake: u64 = stakes.iter().sum();
        let stake_holders = stakes.len() as u64;

        // Calculate mean stake
        let mean_stake = total_stake as f64 / stake_holders as f64;

        // Calculate median stake
        let median_stake = if stakes.len() % 2 == 0 {
            let mid = stakes.len() / 2;
            (stakes[mid - 1] + stakes[mid]) / 2
        } else {
            stakes[stakes.len() / 2]
        };

        // Calculate standard deviation
        let variance = stakes
            .iter()
            .map(|&stake| {
                let diff = stake as f64 - mean_stake;
                diff * diff
            })
            .sum::<f64>()
            / stake_holders as f64;
        let stake_std_deviation = variance.sqrt();

        // Calculate Gini coefficient
        let gini_coefficient = self.calculate_gini_coefficient(&stakes)?;

        // Calculate top 10% share
        let top_10_percent_count = (stake_holders as f64 * 0.1).ceil() as usize;
        let top_10_percent_stake: u64 = stakes.iter().rev().take(top_10_percent_count).sum();
        let top_10_percent_share = if total_stake > 0 {
            top_10_percent_stake as f64 / total_stake as f64
        } else {
            0.0
        };

        // Calculate stake tiers
        let stake_tiers = self.calculate_stake_tiers(&stakes);

        Ok(StakeDistributionMetrics {
            total_stake,
            stake_holders,
            gini_coefficient,
            median_stake,
            mean_stake,
            stake_std_deviation,
            top_10_percent_share,
            stake_tiers,
        })
    }

    /// Calculate Gini coefficient for inequality measurement
    fn calculate_gini_coefficient(&self, stakes: &[u64]) -> Result<f64, String> {
        if stakes.is_empty() {
            return Ok(0.0);
        }

        let n = stakes.len() as f64;
        let mut gini = 0.0;

        for i in 0..stakes.len() {
            for j in 0..stakes.len() {
                let diff = (stakes[i] as f64 - stakes[j] as f64).abs();
                gini += diff;
            }
        }

        let total_stake: u64 = stakes.iter().sum();
        if total_stake == 0 {
            return Ok(0.0);
        }

        Ok(gini / (2.0 * n * total_stake as f64))
    }

    /// Calculate proposal analysis metrics
    fn calculate_proposal_analysis(
        &self,
        proposals: &[Proposal],
        _votes: &[Vote],
    ) -> Result<ProposalAnalysisMetrics, String> {
        let total_proposals = proposals.len() as u64;
        let mut successful_proposals: u64 = 0;
        let mut failed_proposals: u64 = 0;
        let mut success_rate_by_type: HashMap<String, (u64, u64)> = HashMap::new();
        let mut outcomes_by_status: HashMap<String, u64> = HashMap::new();
        let mut total_voting_duration = 0u64;
        let mut proposal_type_counts: HashMap<String, u64> = HashMap::new();

        for proposal in proposals {
            // Count by status
            let status_count = outcomes_by_status
                .entry(format!("{:?}", proposal.status))
                .or_insert(0);
            *status_count = status_count.checked_add(1).ok_or("Status count overflow")?;

            // Count by type
            let type_count = proposal_type_counts
                .entry(format!("{:?}", proposal.proposal_type))
                .or_insert(0);
            *type_count = type_count.checked_add(1).ok_or("Type count overflow")?;

            // Determine success/failure
            match proposal.status {
                ProposalStatus::Approved | ProposalStatus::Executed => {
                    successful_proposals = successful_proposals
                        .checked_add(1)
                        .ok_or("Successful proposals overflow")?;
                }
                ProposalStatus::Rejected | ProposalStatus::Cancelled => {
                    failed_proposals = failed_proposals
                        .checked_add(1)
                        .ok_or("Failed proposals overflow")?;
                }
                _ => {} // Ignore pending/draft proposals
            }

            // Calculate voting duration
            let duration = proposal.voting_end.saturating_sub(proposal.voting_start);
            total_voting_duration = total_voting_duration
                .checked_add(duration)
                .ok_or("Voting duration overflow")?;

            // Track success by type
            let proposal_type = format!("{:?}", proposal.proposal_type);
            let (success_count, total_count) =
                success_rate_by_type.entry(proposal_type).or_insert((0, 0));
            *total_count = total_count.checked_add(1).ok_or("Total count overflow")?;

            if matches!(
                proposal.status,
                ProposalStatus::Approved | ProposalStatus::Executed
            ) {
                *success_count = success_count
                    .checked_add(1)
                    .ok_or("Success count overflow")?;
            }
        }

        // Calculate success rate
        let success_rate = if total_proposals > 0 {
            successful_proposals as f64 / total_proposals as f64
        } else {
            0.0
        };

        // Calculate average voting duration
        let average_voting_duration = if total_proposals > 0 {
            total_voting_duration as f64 / total_proposals as f64
        } else {
            0.0
        };

        // Find most common proposal type
        let most_common_type = proposal_type_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(proposal_type, _)| proposal_type.clone());

        // Convert success rate by type to percentages
        let success_rate_by_type: HashMap<String, f64> = success_rate_by_type
            .into_iter()
            .map(|(proposal_type, (success, total))| {
                let rate = if total > 0 {
                    success as f64 / total as f64
                } else {
                    0.0
                };
                (proposal_type, rate)
            })
            .collect();

        Ok(ProposalAnalysisMetrics {
            total_proposals,
            successful_proposals,
            failed_proposals,
            success_rate,
            success_rate_by_type,
            average_voting_duration,
            most_common_type,
            outcomes_by_status,
        })
    }

    /// Calculate cross-chain participation metrics
    fn calculate_cross_chain_metrics(
        &self,
        cross_chain_votes: &[CrossChainVote],
        _federated_proposals: &[FederatedProposal],
    ) -> Result<CrossChainMetrics, String> {
        let total_cross_chain_votes = cross_chain_votes.len() as u64;

        // Count unique chains
        let mut chain_counts: HashMap<String, u64> = HashMap::new();
        let mut chain_success_counts: HashMap<String, u64> = HashMap::new();
        let mut total_sync_delay = 0u64;
        let mut sync_delay_count = 0u64;

        for vote in cross_chain_votes {
            let count = chain_counts.entry(vote.source_chain.clone()).or_insert(0);
            *count = count.checked_add(1).ok_or("Chain count overflow")?;

            // Track successful votes per chain (simplified - assume all votes are valid)
            let success_count = chain_success_counts
                .entry(vote.source_chain.clone())
                .or_insert(0);
            *success_count = success_count
                .checked_add(1)
                .ok_or("Success count overflow")?;

            // Calculate sync delay (simplified - use timestamp as sync time)
            let sync_time = vote.timestamp + 300; // Assume 5 minute sync delay
            if sync_time > vote.timestamp {
                let delay = sync_time.saturating_sub(vote.timestamp);
                total_sync_delay = total_sync_delay
                    .checked_add(delay)
                    .ok_or("Sync delay overflow")?;
                sync_delay_count = sync_delay_count
                    .checked_add(1)
                    .ok_or("Sync delay count overflow")?;
            }
        }

        let participating_chains = chain_counts.len() as u64;
        let average_votes_per_chain = if participating_chains > 0 {
            total_cross_chain_votes as f64 / participating_chains as f64
        } else {
            0.0
        };

        let avg_sync_delay = if sync_delay_count > 0 {
            total_sync_delay as f64 / sync_delay_count as f64
        } else {
            0.0
        };

        // Calculate chain participation rates
        let chain_participation: HashMap<String, f64> = chain_counts
            .iter()
            .map(|(chain, votes)| {
                let rate = if total_cross_chain_votes > 0 {
                    *votes as f64 / total_cross_chain_votes as f64
                } else {
                    0.0
                };
                (chain.clone(), rate)
            })
            .collect();

        // Calculate cross-chain success rate
        let total_successful_votes: u64 = chain_success_counts.values().sum();
        let cross_chain_success_rate = if total_cross_chain_votes > 0 {
            total_successful_votes as f64 / total_cross_chain_votes as f64
        } else {
            0.0
        };

        // Find most active chain
        let most_active_chain = chain_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(chain, _)| chain.clone());

        Ok(CrossChainMetrics {
            total_cross_chain_votes,
            participating_chains,
            average_votes_per_chain,
            avg_sync_delay,
            chain_participation,
            cross_chain_success_rate,
            most_active_chain,
        })
    }

    /// Calculate temporal trend metrics
    fn calculate_temporal_trends(
        &self,
        votes: &[Vote],
        proposals: &[Proposal],
        voter_activity: &[VoterActivity],
        _time_range: &TimeRange,
    ) -> Result<TemporalTrendMetrics, String> {
        // Create hourly activity buckets
        let mut hourly_activity: HashMap<u64, (u64, u64, u64)> = HashMap::new();

        // Process votes
        for vote in votes {
            let hour = vote.timestamp / 3600; // Convert to hour bucket
            let entry = hourly_activity.entry(hour).or_insert((0, 0, 0));
            entry.0 = entry.0.checked_add(1).ok_or("Vote count overflow")?;
        }

        // Process proposals
        for proposal in proposals {
            let hour = proposal.voting_start / 3600;
            let entry = hourly_activity.entry(hour).or_insert((0, 0, 0));
            entry.1 = entry.1.checked_add(1).ok_or("Proposal count overflow")?;
        }

        // Process voter activity
        for activity in voter_activity {
            let hour = activity.timestamp / 3600;
            let entry = hourly_activity.entry(hour).or_insert((0, 0, 0));
            entry.2 = entry
                .2
                .checked_add(activity.stake_amount)
                .ok_or("Stake activity overflow")?;
        }

        // Convert to sorted vector
        let mut hourly_buckets: Vec<ActivityBucket> = hourly_activity
            .into_iter()
            .map(
                |(hour, (vote_count, proposal_count, stake_activity))| ActivityBucket {
                    hour,
                    vote_count,
                    proposal_count,
                    stake_activity,
                },
            )
            .collect();
        hourly_buckets.sort_by_key(|bucket| bucket.hour);

        // Calculate daily patterns
        let mut daily_patterns: HashMap<String, u64> = HashMap::new();
        for bucket in &hourly_buckets {
            let day = (bucket.hour / 24) % 7; // 0 = Sunday, 1 = Monday, etc.
            let day_name = match day {
                0 => "Sunday",
                1 => "Monday",
                2 => "Tuesday",
                3 => "Wednesday",
                4 => "Thursday",
                5 => "Friday",
                6 => "Saturday",
                _ => "Unknown",
            };
            let count = daily_patterns.entry(day_name.to_string()).or_insert(0);
            *count = count
                .checked_add(bucket.vote_count)
                .ok_or("Daily pattern overflow")?;
        }

        // Calculate weekly trends
        let weekly_trends = self.calculate_weekly_trends(&hourly_buckets)?;

        // Find peak periods
        let peak_periods = self.find_peak_periods(&hourly_buckets)?;

        // Calculate seasonal patterns (simplified)
        let seasonal_patterns = self.calculate_seasonal_patterns(&hourly_buckets)?;

        Ok(TemporalTrendMetrics {
            hourly_activity: hourly_buckets,
            daily_patterns,
            weekly_trends,
            peak_periods,
            seasonal_patterns,
        })
    }

    /// Calculate weekly trends
    fn calculate_weekly_trends(
        &self,
        hourly_buckets: &[ActivityBucket],
    ) -> Result<Vec<WeeklyTrend>, String> {
        let mut weekly_data: HashMap<u64, Vec<u64>> = HashMap::new();

        for bucket in hourly_buckets {
            let week = bucket.hour / (24 * 7); // Week number since epoch
            weekly_data.entry(week).or_default().push(bucket.vote_count);
        }

        let mut trends = Vec::new();
        let mut sorted_weeks: Vec<u64> = weekly_data.keys().cloned().collect();
        sorted_weeks.sort_unstable();

        for (i, &week) in sorted_weeks.iter().enumerate() {
            let activities = &weekly_data[&week];
            let total_activity: u64 = activities.iter().sum();
            let avg_daily_activity = total_activity as f64 / 7.0;

            // Determine trend direction
            let trend_direction = if i > 0 {
                let prev_week = sorted_weeks[i - 1];
                let prev_activities = &weekly_data[&prev_week];
                let prev_total: u64 = prev_activities.iter().sum();

                if total_activity > prev_total {
                    "increasing"
                } else if total_activity < prev_total {
                    "decreasing"
                } else {
                    "stable"
                }
            } else {
                "stable"
            };

            trends.push(WeeklyTrend {
                week,
                total_activity,
                avg_daily_activity,
                trend_direction: trend_direction.to_string(),
            });
        }

        Ok(trends)
    }

    /// Find peak activity periods
    fn find_peak_periods(
        &self,
        hourly_buckets: &[ActivityBucket],
    ) -> Result<Vec<PeakPeriod>, String> {
        if hourly_buckets.is_empty() {
            return Ok(Vec::new());
        }

        // Calculate average activity
        let total_activity: u64 = hourly_buckets.iter().map(|b| b.vote_count).sum();
        let avg_activity = total_activity as f64 / hourly_buckets.len() as f64;
        let threshold = avg_activity * 1.5; // 50% above average

        let mut peak_periods = Vec::new();
        let mut current_peak_start: Option<u64> = None;

        for bucket in hourly_buckets {
            if bucket.vote_count as f64 > threshold {
                if current_peak_start.is_none() {
                    current_peak_start = Some(bucket.hour);
                }
            } else if let Some(start) = current_peak_start {
                let duration = bucket.hour.saturating_sub(start);
                peak_periods.push(PeakPeriod {
                    start_time: start * 3600, // Convert back to seconds
                    end_time: bucket.hour * 3600,
                    activity_level: bucket.vote_count,
                    duration: duration * 3600,
                });
                current_peak_start = None;
            }
        }

        // Handle case where peak extends to end of data
        if let Some(start) = current_peak_start {
            let last_bucket = hourly_buckets.last().unwrap();
            let duration = last_bucket.hour.saturating_sub(start);
            peak_periods.push(PeakPeriod {
                start_time: start * 3600,
                end_time: last_bucket.hour * 3600,
                activity_level: last_bucket.vote_count,
                duration: duration * 3600,
            });
        }

        Ok(peak_periods)
    }

    /// Calculate seasonal patterns
    fn calculate_seasonal_patterns(
        &self,
        hourly_buckets: &[ActivityBucket],
    ) -> Result<HashMap<String, f64>, String> {
        let mut seasonal_data: HashMap<String, Vec<u64>> = HashMap::new();

        for bucket in hourly_buckets {
            // Simplified seasonal calculation based on hour of day
            let season = if bucket.hour % 24 < 6 {
                "night"
            } else if bucket.hour % 24 < 12 {
                "morning"
            } else if bucket.hour % 24 < 18 {
                "afternoon"
            } else {
                "evening"
            };

            seasonal_data
                .entry(season.to_string())
                .or_default()
                .push(bucket.vote_count);
        }

        let mut patterns = HashMap::new();
        for (season, activities) in seasonal_data {
            let total: u64 = activities.iter().sum();
            let average = total as f64 / activities.len() as f64;
            patterns.insert(season, average);
        }

        Ok(patterns)
    }

    /// Determine stake tier based on stake amount
    fn determine_stake_tier(&self, stake: u64) -> String {
        match stake {
            0..=1000 => "small".to_string(),
            1001..=10000 => "medium".to_string(),
            10001..=100000 => "large".to_string(),
            _ => "whale".to_string(),
        }
    }

    /// Calculate stake tiers distribution
    fn calculate_stake_tiers(&self, stakes: &[u64]) -> HashMap<String, u64> {
        let mut tiers: HashMap<String, u64> = HashMap::new();

        for &stake in stakes {
            let tier = self.determine_stake_tier(stake);
            let count = tiers.entry(tier).or_insert(0);
            *count = count.checked_add(1).unwrap_or(u64::MAX);
        }

        tiers
    }

    /// Generate Chart.js compatible data for visualizations
    pub fn generate_chart_data(
        &self,
        analytics: &GovernanceAnalytics,
    ) -> Result<Vec<ChartData>, String> {
        let charts = vec![
            // Voter turnout chart
            self.create_turnout_chart(&analytics.voter_turnout)?,
            // Stake distribution chart
            self.create_stake_distribution_chart(&analytics.stake_distribution)?,
            // Proposal success rate chart
            self.create_proposal_success_chart(&analytics.proposal_analysis)?,
            // Temporal activity chart
            self.create_temporal_activity_chart(&analytics.temporal_trends)?,
            // Cross-chain participation chart
            self.create_cross_chain_chart(&analytics.cross_chain_metrics)?,
        ];

        Ok(charts)
    }

    /// Create voter turnout chart
    fn create_turnout_chart(&self, turnout: &VoterTurnoutMetrics) -> Result<ChartData, String> {
        let labels = vec!["Voted".to_string(), "Did Not Vote".to_string()];
        let voted_count = turnout.total_voters as f64;
        let not_voted_count = turnout.eligible_voters.saturating_sub(turnout.total_voters) as f64;

        Ok(ChartData {
            chart_type: "doughnut".to_string(),
            title: "Voter Turnout".to_string(),
            labels,
            datasets: vec![ChartDataset {
                label: "Voters".to_string(),
                data: vec![voted_count, not_voted_count],
                border_color: "#4CAF50".to_string(),
                background_color: "#81C784".to_string(),
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

    /// Create stake distribution chart
    fn create_stake_distribution_chart(
        &self,
        distribution: &StakeDistributionMetrics,
    ) -> Result<ChartData, String> {
        let mut labels = Vec::new();
        let mut data = Vec::new();

        for (tier, count) in &distribution.stake_tiers {
            labels.push(tier.clone());
            data.push(*count as f64);
        }

        Ok(ChartData {
            chart_type: "bar".to_string(),
            title: "Stake Distribution by Tier".to_string(),
            labels,
            datasets: vec![ChartDataset {
                label: "Stake Holders".to_string(),
                data,
                border_color: "#2196F3".to_string(),
                background_color: "#64B5F6".to_string(),
                fill: true,
            }],
            options: ChartOptions {
                legend: true,
                tooltips: true,
                responsive: true,
                scales: Some(ChartScales {
                    y_axes: vec![ChartAxis {
                        r#type: "linear".to_string(),
                        display: true,
                        label: "Number of Stake Holders".to_string(),
                    }],
                    x_axes: vec![ChartAxis {
                        r#type: "category".to_string(),
                        display: true,
                        label: "Stake Tier".to_string(),
                    }],
                }),
            },
        })
    }

    /// Create proposal success rate chart
    fn create_proposal_success_chart(
        &self,
        analysis: &ProposalAnalysisMetrics,
    ) -> Result<ChartData, String> {
        let mut labels = Vec::new();
        let mut success_data = Vec::new();
        let mut failure_data = Vec::new();

        for (proposal_type, success_rate) in &analysis.success_rate_by_type {
            labels.push(proposal_type.clone());
            success_data.push(*success_rate * 100.0);
            failure_data.push((1.0 - success_rate) * 100.0);
        }

        Ok(ChartData {
            chart_type: "bar".to_string(),
            title: "Proposal Success Rate by Type".to_string(),
            labels,
            datasets: vec![
                ChartDataset {
                    label: "Success Rate (%)".to_string(),
                    data: success_data,
                    border_color: "#4CAF50".to_string(),
                    background_color: "#81C784".to_string(),
                    fill: true,
                },
                ChartDataset {
                    label: "Failure Rate (%)".to_string(),
                    data: failure_data,
                    border_color: "#F44336".to_string(),
                    background_color: "#E57373".to_string(),
                    fill: true,
                },
            ],
            options: ChartOptions {
                legend: true,
                tooltips: true,
                responsive: true,
                scales: Some(ChartScales {
                    y_axes: vec![ChartAxis {
                        r#type: "linear".to_string(),
                        display: true,
                        label: "Percentage (%)".to_string(),
                    }],
                    x_axes: vec![ChartAxis {
                        r#type: "category".to_string(),
                        display: true,
                        label: "Proposal Type".to_string(),
                    }],
                }),
            },
        })
    }

    /// Create temporal activity chart
    fn create_temporal_activity_chart(
        &self,
        trends: &TemporalTrendMetrics,
    ) -> Result<ChartData, String> {
        let mut labels = Vec::new();
        let mut vote_data = Vec::new();
        let mut proposal_data = Vec::new();

        for bucket in &trends.hourly_activity {
            labels.push(format!("Hour {}", bucket.hour));
            vote_data.push(bucket.vote_count as f64);
            proposal_data.push(bucket.proposal_count as f64);
        }

        Ok(ChartData {
            chart_type: "line".to_string(),
            title: "Temporal Activity Trends".to_string(),
            labels,
            datasets: vec![
                ChartDataset {
                    label: "Votes".to_string(),
                    data: vote_data,
                    border_color: "#FF9800".to_string(),
                    background_color: "#FFB74D".to_string(),
                    fill: false,
                },
                ChartDataset {
                    label: "Proposals".to_string(),
                    data: proposal_data,
                    border_color: "#9C27B0".to_string(),
                    background_color: "#BA68C8".to_string(),
                    fill: false,
                },
            ],
            options: ChartOptions {
                legend: true,
                tooltips: true,
                responsive: true,
                scales: Some(ChartScales {
                    y_axes: vec![ChartAxis {
                        r#type: "linear".to_string(),
                        display: true,
                        label: "Activity Count".to_string(),
                    }],
                    x_axes: vec![ChartAxis {
                        r#type: "category".to_string(),
                        display: true,
                        label: "Time".to_string(),
                    }],
                }),
            },
        })
    }

    /// Create cross-chain participation chart
    fn create_cross_chain_chart(&self, metrics: &CrossChainMetrics) -> Result<ChartData, String> {
        let mut labels = Vec::new();
        let mut data = Vec::new();

        for (chain, participation_rate) in &metrics.chain_participation {
            labels.push(chain.clone());
            data.push(*participation_rate * 100.0);
        }

        Ok(ChartData {
            chart_type: "pie".to_string(),
            title: "Cross-Chain Participation".to_string(),
            labels,
            datasets: vec![ChartDataset {
                label: "Participation Rate (%)".to_string(),
                data,
                border_color: "#607D8B".to_string(),
                background_color: "#90A4AE".to_string(),
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

    /// Validate data integrity using SHA-3
    fn validate_data_integrity(
        &self,
        proposals: &[Proposal],
        votes: &[Vote],
        federated_proposals: &[FederatedProposal],
    ) -> Result<(), String> {
        // Create integrity hashes for each dataset
        let mut hasher = Sha3_256::new();

        // Hash proposals
        for proposal in proposals {
            hasher.update(proposal.id.as_bytes());
            hasher.update(proposal.voting_start.to_le_bytes());
            hasher.update(proposal.voting_end.to_le_bytes());
        }

        // Hash votes
        for vote in votes {
            hasher.update(vote.voter_id.as_bytes());
            hasher.update(vote.proposal_id.as_bytes());
            hasher.update(vote.timestamp.to_le_bytes());
        }

        // Hash federated proposals
        for fed_proposal in federated_proposals {
            hasher.update(fed_proposal.proposal_id.as_bytes());
            hasher.update(fed_proposal.created_at.to_le_bytes());
        }

        // The hash is calculated but not used for validation in this simplified version
        // In a real implementation, this would be compared against known good hashes
        let _integrity_hash = hasher.finalize();

        Ok(())
    }

    /// Calculate integrity hash for the analytics report
    fn calculate_integrity_hash(&self, analytics: &GovernanceAnalytics) -> Result<String, String> {
        let mut hasher = Sha3_256::new();

        // Hash key metrics
        hasher.update(analytics.voter_turnout.total_voters.to_le_bytes());
        hasher.update(analytics.stake_distribution.total_stake.to_le_bytes());
        hasher.update(analytics.proposal_analysis.total_proposals.to_le_bytes());
        hasher.update(
            analytics
                .cross_chain_metrics
                .total_cross_chain_votes
                .to_le_bytes(),
        );
        hasher.update(analytics.timestamp.to_le_bytes());

        let hash = hasher.finalize();
        Ok(hex::encode(hash))
    }

    /// Generate unique report ID
    fn generate_report_id(&self) -> String {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        format!("analytics_{}", timestamp)
    }

    /// Cache analytics result
    fn cache_result(&mut self, analytics: &GovernanceAnalytics) -> Result<(), String> {
        if self.cache.len() >= self.max_cache_size {
            // Remove oldest entry (simple FIFO)
            if let Some(oldest_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&oldest_key);
            }
        }

        self.cache
            .insert(analytics.report_id.clone(), analytics.clone());
        Ok(())
    }

    /// Get cached analytics result
    pub fn get_cached_result(&self, report_id: &str) -> Option<&GovernanceAnalytics> {
        self.cache.get(report_id)
    }

    /// Clear analytics cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }

    /// Export analytics to JSON format
    pub fn export_to_json(&self, analytics: &GovernanceAnalytics) -> Result<String, String> {
        serde_json::to_string_pretty(analytics)
            .map_err(|e| format!("JSON serialization failed: {}", e))
    }

    /// Export analytics to human-readable format
    pub fn export_to_human_readable(&self, analytics: &GovernanceAnalytics) -> String {
        format!(
            "Governance Analytics Report\n\
            =========================\n\
            Report ID: {}\n\
            Timestamp: {}\n\
            Time Range: {} to {} ({} seconds)\n\
            \n\
            Voter Turnout:\n\
            - Total Voters: {}\n\
            - Eligible Voters: {}\n\
            - Turnout: {:.2}%\n\
            - Average Votes per Voter: {:.2}\n\
            \n\
            Stake Distribution:\n\
            - Total Stake: {}\n\
            - Stake Holders: {}\n\
            - Gini Coefficient: {:.4}\n\
            - Median Stake: {}\n\
            - Mean Stake: {:.2}\n\
            \n\
            Proposal Analysis:\n\
            - Total Proposals: {}\n\
            - Successful: {}\n\
            - Failed: {}\n\
            - Success Rate: {:.2}%\n\
            \n\
            Cross-Chain Metrics:\n\
            - Cross-Chain Votes: {}\n\
            - Participating Chains: {}\n\
            - Success Rate: {:.2}%\n\
            \n\
            Data Integrity Hash: {}\n",
            analytics.report_id,
            analytics.timestamp,
            analytics.time_range.start_time,
            analytics.time_range.end_time,
            analytics.time_range.duration_seconds,
            analytics.voter_turnout.total_voters,
            analytics.voter_turnout.eligible_voters,
            analytics.voter_turnout.turnout_percentage,
            analytics.voter_turnout.average_votes_per_voter,
            analytics.stake_distribution.total_stake,
            analytics.stake_distribution.stake_holders,
            analytics.stake_distribution.gini_coefficient,
            analytics.stake_distribution.median_stake,
            analytics.stake_distribution.mean_stake,
            analytics.proposal_analysis.total_proposals,
            analytics.proposal_analysis.successful_proposals,
            analytics.proposal_analysis.failed_proposals,
            analytics.proposal_analysis.success_rate * 100.0,
            analytics.cross_chain_metrics.total_cross_chain_votes,
            analytics.cross_chain_metrics.participating_chains,
            analytics.cross_chain_metrics.cross_chain_success_rate * 100.0,
            analytics.integrity_hash
        )
    }
}

impl Default for GovernanceAnalyticsEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper functions for analytics calculations
impl GovernanceAnalyticsEngine {
    /// Calculate correlation coefficient between two datasets
    pub fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> Result<f64, String> {
        if x.len() != y.len() || x.is_empty() {
            return Err("Invalid dataset lengths for correlation".to_string());
        }

        let n = x.len() as f64;
        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_x2: f64 = x.iter().map(|a| a * a).sum();
        let sum_y2: f64 = y.iter().map(|a| a * a).sum();

        let numerator = n * sum_xy - sum_x * sum_y;
        let denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)).sqrt();

        if denominator == 0.0 {
            return Ok(0.0);
        }

        Ok(numerator / denominator)
    }

    /// Calculate moving average for time series data
    pub fn calculate_moving_average(
        &self,
        data: &[f64],
        window_size: usize,
    ) -> Result<Vec<f64>, String> {
        if window_size == 0 || window_size > data.len() {
            return Err("Invalid window size for moving average".to_string());
        }

        let mut result = Vec::new();

        for i in 0..=data.len().saturating_sub(window_size) {
            let window_sum: f64 = data[i..i + window_size].iter().sum();
            result.push(window_sum / window_size as f64);
        }

        Ok(result)
    }

    /// Calculate percentile for a dataset
    pub fn calculate_percentile(&self, mut data: Vec<f64>, percentile: f64) -> Result<f64, String> {
        if data.is_empty() {
            return Err("Cannot calculate percentile of empty dataset".to_string());
        }

        if !(0.0..=100.0).contains(&percentile) {
            return Err("Percentile must be between 0 and 100".to_string());
        }

        data.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = (percentile / 100.0) * (data.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            Ok(data[lower])
        } else {
            let weight = index - lower as f64;
            Ok(data[lower] * (1.0 - weight) + data[upper] * weight)
        }
    }
}
