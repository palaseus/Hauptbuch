//! Game Theory Analysis Module
//!
//! This module provides comprehensive game-theoretic analysis for cross-chain governance
//! scenarios, modeling voter and validator behaviors, incentives, and outcomes in
//! federated governance systems.
//!
//! Key features:
//! - Nash equilibrium analysis for voting strategies
//! - Collusion detection and prevention mechanisms
//! - Fairness metrics and participation equity analysis
//! - Cross-chain incentive modeling
//! - JSON reports and Chart.js-compatible visualizations
//! - Integration with governance simulator, federation, and analytics modules

pub mod analyzer;
#[cfg(test)]
pub mod analyzer_test;

// Re-export main types for external use
pub use analyzer::{
    CollusionAnalysis, CrossChainIncentive, FairnessMetrics, GameScenario, GameTheoryAnalyzer,
    GameTheoryError, GameTheoryReport, IncentiveStructure, NashEquilibrium, ParticipationEquity,
    PayoffMatrix, StrategyOutcome, ValidatorStrategy, VoterStrategy,
};
