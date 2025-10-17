# Governance Examples

## Overview

This document provides comprehensive examples of governance functionality in the Hauptbuch blockchain platform. Learn how to create proposals, vote on them, execute governance decisions, and manage the governance process.

## Table of Contents

- [Getting Started](#getting-started)
- [Proposal Management](#proposal-management)
- [Voting System](#voting-system)
- [Governance Execution](#governance-execution)
- [Governance Analytics](#governance-analytics)
- [Advanced Governance](#advanced-governance)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Getting Started

### Basic Governance Setup

```rust
use hauptbuch_governance::{GovernanceEngine, GovernanceConfig, Proposal, Vote};

#[tokio::main]
async fn main() -> Result<(), GovernanceError> {
    // Create governance engine
    let mut governance = GovernanceEngine::new();
    
    // Configure governance parameters
    let config = GovernanceConfig {
        proposal_threshold: 1000,
        voting_period: 7 * 24 * 60 * 60, // 7 days
        execution_delay: 24 * 60 * 60,    // 1 day
        quorum_threshold: 0.1,            // 10%
        supermajority_threshold: 0.67,     // 67%
    };
    
    governance.set_config(config);
    
    println!("Governance engine initialized");
    Ok(())
}
```

### Governance Client Setup

```rust
use hauptbuch_governance::{GovernanceClient, GovernanceConfig};

async fn setup_governance_client() -> Result<(), GovernanceError> {
    let client = GovernanceClient::new("http://localhost:8080")?;
    
    // Get governance configuration
    let config = client.get_governance_config().await?;
    println!("Proposal threshold: {}", config.proposal_threshold);
    println!("Voting period: {} seconds", config.voting_period);
    println!("Quorum threshold: {}%", config.quorum_threshold * 100.0);
    
    Ok(())
}
```

## Proposal Management

### Creating a Proposal

```rust
use hauptbuch_governance::{GovernanceClient, Proposal, ProposalType, ProposalStatus};

async fn create_proposal(
    client: &GovernanceClient,
    author: &str,
    title: &str,
    description: &str,
) -> Result<u64, GovernanceError> {
    // Create proposal
    let proposal = Proposal::new()
        .set_title(title)
        .set_description(description)
        .set_author(author)
        .set_proposal_type(ProposalType::ParameterChange)
        .set_parameters(vec![
            ("block_time", "5000".to_string()),
            ("gas_limit", "10000000".to_string()),
        ])
        .build()?;
    
    // Submit proposal
    let proposal_id = client.submit_proposal(&proposal).await?;
    
    println!("Proposal created with ID: {}", proposal_id);
    Ok(proposal_id)
}
```

### Parameter Change Proposal

```rust
use hauptbuch_governance::{GovernanceClient, Proposal, ProposalType, ParameterChange};

async fn create_parameter_change_proposal(
    client: &GovernanceClient,
    author: &str,
) -> Result<u64, GovernanceError> {
    let parameter_changes = vec![
        ParameterChange {
            parameter: "block_time".to_string(),
            old_value: "5000".to_string(),
            new_value: "3000".to_string(),
            description: "Reduce block time to 3 seconds for faster finality".to_string(),
        },
        ParameterChange {
            parameter: "gas_limit".to_string(),
            old_value: "10000000".to_string(),
            new_value: "15000000".to_string(),
            description: "Increase gas limit to 15M for higher throughput".to_string(),
        },
    ];
    
    let proposal = Proposal::new()
        .set_title("Optimize Network Parameters")
        .set_description("Reduce block time and increase gas limit for better performance")
        .set_author(author)
        .set_proposal_type(ProposalType::ParameterChange)
        .set_parameter_changes(parameter_changes)
        .build()?;
    
    let proposal_id = client.submit_proposal(&proposal).await?;
    println!("Parameter change proposal created: {}", proposal_id);
    Ok(proposal_id)
}
```

### Treasury Proposal

```rust
use hauptbuch_governance::{GovernanceClient, Proposal, ProposalType, TreasurySpending};

async fn create_treasury_proposal(
    client: &GovernanceClient,
    author: &str,
    recipient: &str,
    amount: u64,
    purpose: &str,
) -> Result<u64, GovernanceError> {
    let treasury_spending = TreasurySpending {
        recipient: recipient.to_string(),
        amount,
        purpose: purpose.to_string(),
        category: "Development".to_string(),
    };
    
    let proposal = Proposal::new()
        .set_title("Treasury Spending Proposal")
        .set_description(format!("Spend {} HBK for {}", amount, purpose))
        .set_author(author)
        .set_proposal_type(ProposalType::TreasurySpending)
        .set_treasury_spending(treasury_spending)
        .build()?;
    
    let proposal_id = client.submit_proposal(&proposal).await?;
    println!("Treasury proposal created: {}", proposal_id);
    Ok(proposal_id)
}
```

### Upgrade Proposal

```rust
use hauptbuch_governance::{GovernanceClient, Proposal, ProposalType, UpgradePlan};

async fn create_upgrade_proposal(
    client: &GovernanceClient,
    author: &str,
    new_version: &str,
    upgrade_height: u64,
) -> Result<u64, GovernanceError> {
    let upgrade_plan = UpgradePlan {
        new_version: new_version.to_string(),
        upgrade_height,
        description: format!("Upgrade to version {}", new_version),
        migration_script: "migration_v2.sh".to_string(),
        rollback_plan: "rollback_v1.sh".to_string(),
    };
    
    let proposal = Proposal::new()
        .set_title("Network Upgrade")
        .set_description(format!("Upgrade to version {} at height {}", new_version, upgrade_height))
        .set_author(author)
        .set_proposal_type(ProposalType::NetworkUpgrade)
        .set_upgrade_plan(upgrade_plan)
        .build()?;
    
    let proposal_id = client.submit_proposal(&proposal).await?;
    println!("Upgrade proposal created: {}", proposal_id);
    Ok(proposal_id)
}
```

## Voting System

### Basic Voting

```rust
use hauptbuch_governance::{GovernanceClient, Vote, VoteChoice};

async fn vote_on_proposal(
    client: &GovernanceClient,
    voter: &str,
    proposal_id: u64,
    choice: VoteChoice,
    voting_power: u64,
) -> Result<(), GovernanceError> {
    let vote = Vote::new()
        .set_voter(voter)
        .set_proposal_id(proposal_id)
        .set_choice(choice)
        .set_voting_power(voting_power)
        .set_reason("I support this proposal")
        .build()?;
    
    client.vote(&vote).await?;
    println!("Vote cast for proposal {}", proposal_id);
    Ok(())
}
```

### Delegated Voting

```rust
use hauptbuch_governance::{GovernanceClient, Vote, VoteChoice, Delegation};

async fn delegate_voting_power(
    client: &GovernanceClient,
    delegator: &str,
    delegate: &str,
    amount: u64,
) -> Result<(), GovernanceError> {
    let delegation = Delegation {
        delegator: delegator.to_string(),
        delegate: delegate.to_string(),
        amount,
        start_time: chrono::Utc::now().timestamp(),
        end_time: None, // No expiration
    };
    
    client.delegate_voting_power(&delegation).await?;
    println!("Delegated {} voting power from {} to {}", amount, delegator, delegate);
    Ok(())
}
```

### Voting with Reason

```rust
use hauptbuch_governance::{GovernanceClient, Vote, VoteChoice};

async fn vote_with_reason(
    client: &GovernanceClient,
    voter: &str,
    proposal_id: u64,
    choice: VoteChoice,
    reason: &str,
) -> Result<(), GovernanceError> {
    let vote = Vote::new()
        .set_voter(voter)
        .set_proposal_id(proposal_id)
        .set_choice(choice)
        .set_reason(reason)
        .set_voting_power(1000)
        .build()?;
    
    client.vote(&vote).await?;
    println!("Vote cast with reason: {}", reason);
    Ok(())
}
```

### Voting Power Calculation

```rust
use hauptbuch_governance::{GovernanceClient, VotingPower};

async fn calculate_voting_power(
    client: &GovernanceClient,
    voter: &str,
) -> Result<VotingPower, GovernanceError> {
    let voting_power = client.get_voting_power(voter).await?;
    
    println!("Voter: {}", voter);
    println!("Total voting power: {}", voting_power.total);
    println!("Delegated voting power: {}", voting_power.delegated);
    println!("Own voting power: {}", voting_power.own);
    println!("Available for delegation: {}", voting_power.available_for_delegation);
    
    Ok(voting_power)
}
```

## Governance Execution

### Executing Proposals

```rust
use hauptbuch_governance::{GovernanceClient, ProposalStatus};

async fn execute_proposal(
    client: &GovernanceClient,
    proposal_id: u64,
) -> Result<(), GovernanceError> {
    // Check if proposal is ready for execution
    let status = client.get_proposal_status(proposal_id).await?;
    
    match status {
        ProposalStatus::Passed => {
            // Execute the proposal
            client.execute_proposal(proposal_id).await?;
            println!("Proposal {} executed successfully", proposal_id);
        }
        ProposalStatus::Active => {
            println!("Proposal {} is still active", proposal_id);
        }
        ProposalStatus::Failed => {
            println!("Proposal {} failed and cannot be executed", proposal_id);
        }
        _ => {
            println!("Proposal {} is not ready for execution", proposal_id);
        }
    }
    
    Ok(())
}
```

### Batch Execution

```rust
use hauptbuch_governance::{GovernanceClient, ProposalStatus};

async fn execute_passed_proposals(
    client: &GovernanceClient,
) -> Result<(), GovernanceError> {
    // Get all proposals
    let proposals = client.get_proposals(None, None, None).await?;
    
    // Filter passed proposals
    let passed_proposals: Vec<_> = proposals
        .into_iter()
        .filter(|p| p.status == ProposalStatus::Passed)
        .collect();
    
    println!("Found {} passed proposals", passed_proposals.len());
    
    // Execute each passed proposal
    for proposal in passed_proposals {
        match client.execute_proposal(proposal.id).await {
            Ok(_) => println!("Proposal {} executed", proposal.id),
            Err(e) => println!("Failed to execute proposal {}: {}", proposal.id, e),
        }
    }
    
    Ok(())
}
```

### Emergency Execution

```rust
use hauptbuch_governance::{GovernanceClient, EmergencyProposal};

async fn create_emergency_proposal(
    client: &GovernanceClient,
    author: &str,
    reason: &str,
) -> Result<u64, GovernanceError> {
    let emergency_proposal = EmergencyProposal {
        author: author.to_string(),
        reason: reason.to_string(),
        emergency_type: "Security".to_string(),
        immediate_execution: true,
        bypass_quorum: true,
    };
    
    let proposal_id = client.submit_emergency_proposal(&emergency_proposal).await?;
    println!("Emergency proposal created: {}", proposal_id);
    Ok(proposal_id)
}
```

## Governance Analytics

### Proposal Statistics

```rust
use hauptbuch_governance::{GovernanceClient, ProposalStats};

async fn get_proposal_statistics(
    client: &GovernanceClient,
) -> Result<(), GovernanceError> {
    let stats = client.get_proposal_statistics().await?;
    
    println!("Total proposals: {}", stats.total_proposals);
    println!("Active proposals: {}", stats.active_proposals);
    println!("Passed proposals: {}", stats.passed_proposals);
    println!("Failed proposals: {}", stats.failed_proposals);
    println!("Average voting participation: {:.2}%", stats.average_participation * 100.0);
    println!("Average proposal duration: {} days", stats.average_duration / (24 * 60 * 60));
    
    Ok(())
}
```

### Voting Analytics

```rust
use hauptbuch_governance::{GovernanceClient, VotingAnalytics};

async fn analyze_voting_patterns(
    client: &GovernanceClient,
    proposal_id: u64,
) -> Result<(), GovernanceError> {
    let analytics = client.get_voting_analytics(proposal_id).await?;
    
    println!("Proposal {} Voting Analytics:", proposal_id);
    println!("Total votes: {}", analytics.total_votes);
    println!("Yes votes: {}", analytics.yes_votes);
    println!("No votes: {}", analytics.no_votes);
    println!("Abstain votes: {}", analytics.abstain_votes);
    println!("Participation rate: {:.2}%", analytics.participation_rate * 100.0);
    println!("Quorum reached: {}", analytics.quorum_reached);
    println!("Supermajority reached: {}", analytics.supermajority_reached);
    
    Ok(())
}
```

### Governance Metrics

```rust
use hauptbuch_governance::{GovernanceClient, GovernanceMetrics};

async fn get_governance_metrics(
    client: &GovernanceClient,
) -> Result<(), GovernanceError> {
    let metrics = client.get_governance_metrics().await?;
    
    println!("Governance Metrics:");
    println!("Active voters: {}", metrics.active_voters);
    println!("Total voting power: {}", metrics.total_voting_power);
    println!("Average proposal success rate: {:.2}%", metrics.success_rate * 100.0);
    println!("Average voting participation: {:.2}%", metrics.participation_rate * 100.0);
    println!("Most active proposers: {:?}", metrics.top_proposers);
    println!("Most active voters: {:?}", metrics.top_voters);
    
    Ok(())
}
```

## Advanced Governance

### Multi-Signature Governance

```rust
use hauptbuch_governance::{GovernanceClient, MultiSigProposal, MultiSigVote};

async fn create_multisig_proposal(
    client: &GovernanceClient,
    proposer: &str,
    signers: Vec<String>,
    threshold: u32,
) -> Result<u64, GovernanceError> {
    let multisig_proposal = MultiSigProposal {
        proposer: proposer.to_string(),
        signers,
        threshold,
        title: "Multi-sig Proposal".to_string(),
        description: "A proposal requiring multiple signatures".to_string(),
    };
    
    let proposal_id = client.submit_multisig_proposal(&multisig_proposal).await?;
    println!("Multi-sig proposal created: {}", proposal_id);
    Ok(proposal_id)
}
```

### Time-Locked Governance

```rust
use hauptbuch_governance::{GovernanceClient, TimeLockedProposal};

async fn create_timelocked_proposal(
    client: &GovernanceClient,
    author: &str,
    delay: u64,
) -> Result<u64, GovernanceError> {
    let timelocked_proposal = TimeLockedProposal {
        author: author.to_string(),
        title: "Time-locked Proposal".to_string(),
        description: "A proposal with execution delay".to_string(),
        delay,
        execution_time: chrono::Utc::now().timestamp() + delay as i64,
    };
    
    let proposal_id = client.submit_timelocked_proposal(&timelocked_proposal).await?;
    println!("Time-locked proposal created: {}", proposal_id);
    Ok(proposal_id)
}
```

### Conditional Governance

```rust
use hauptbuch_governance::{GovernanceClient, ConditionalProposal, Condition};

async fn create_conditional_proposal(
    client: &GovernanceClient,
    author: &str,
    condition: Condition,
) -> Result<u64, GovernanceError> {
    let conditional_proposal = ConditionalProposal {
        author: author.to_string(),
        title: "Conditional Proposal".to_string(),
        description: "A proposal with execution conditions".to_string(),
        condition,
        execution_script: "execute_conditionally.sh".to_string(),
    };
    
    let proposal_id = client.submit_conditional_proposal(&conditional_proposal).await?;
    println!("Conditional proposal created: {}", proposal_id);
    Ok(proposal_id)
}
```

## Error Handling

### Governance Error Handling

```rust
use hauptbuch_governance::{GovernanceClient, GovernanceError};

async fn handle_governance_errors(
    client: &GovernanceClient,
    proposal_id: u64,
) -> Result<(), GovernanceError> {
    match client.get_proposal(proposal_id).await {
        Ok(proposal) => {
            println!("Proposal found: {}", proposal.title);
        }
        Err(GovernanceError::ProposalNotFound) => {
            println!("Proposal {} not found", proposal_id);
        }
        Err(GovernanceError::InsufficientVotingPower) => {
            println!("Insufficient voting power to vote on proposal {}", proposal_id);
        }
        Err(GovernanceError::ProposalNotActive) => {
            println!("Proposal {} is not active", proposal_id);
        }
        Err(GovernanceError::AlreadyVoted) => {
            println!("Already voted on proposal {}", proposal_id);
        }
        Err(e) => {
            println!("Other governance error: {}", e);
        }
    }
    
    Ok(())
}
```

### Retry Logic for Governance

```rust
use hauptbuch_governance::{GovernanceClient, GovernanceError};
use tokio::time::{sleep, Duration};

async fn retry_governance_operation(
    client: &GovernanceClient,
    proposal_id: u64,
    max_retries: u32,
) -> Result<(), GovernanceError> {
    let mut retries = 0;
    
    loop {
        match client.get_proposal(proposal_id).await {
            Ok(proposal) => {
                println!("Proposal retrieved: {}", proposal.title);
                return Ok(());
            }
            Err(GovernanceError::NetworkError(_)) if retries < max_retries => {
                retries += 1;
                println!("Retry {} of {}", retries, max_retries);
                sleep(Duration::from_secs(1)).await;
            }
            Err(e) => return Err(e),
        }
    }
}
```

## Best Practices

### Governance Security

```rust
use hauptbuch_governance::{GovernanceClient, Proposal, SecurityConfig};

async fn secure_governance_operations(
    client: &GovernanceClient,
) -> Result<(), GovernanceError> {
    // Use secure voting
    let security_config = SecurityConfig {
        require_identity_verification: true,
        enable_audit_trail: true,
        require_multisig: true,
        minimum_voting_power: 1000,
    };
    
    client.set_security_config(security_config).await?;
    
    // Verify proposal before voting
    let proposal_id = 1;
    let proposal = client.get_proposal(proposal_id).await?;
    
    // Check proposal integrity
    if !proposal.verify_integrity() {
        return Err(GovernanceError::InvalidProposal);
    }
    
    // Vote with verification
    let vote = Vote::new()
        .set_voter("0x1234...")
        .set_proposal_id(proposal_id)
        .set_choice(VoteChoice::Yes)
        .set_voting_power(1000)
        .verify_signature(true)
        .build()?;
    
    client.vote(&vote).await?;
    Ok(())
}
```

### Performance Optimization

```rust
use hauptbuch_governance::{GovernanceClient, Proposal, Vote};
use tokio::task;

async fn optimize_governance_performance(
    client: &GovernanceClient,
) -> Result<(), GovernanceError> {
    // Batch operations
    let proposal_ids = vec![1, 2, 3, 4, 5];
    let mut tasks = Vec::new();
    
    for proposal_id in proposal_ids {
        let client = client.clone();
        let task = task::spawn(async move {
            client.get_proposal(proposal_id).await
        });
        tasks.push(task);
    }
    
    // Wait for all tasks
    let results = futures::future::join_all(tasks).await;
    
    for result in results {
        match result {
            Ok(Ok(proposal)) => println!("Proposal: {}", proposal.title),
            Ok(Err(e)) => println!("Error: {}", e),
            Err(e) => println!("Task error: {}", e),
        }
    }
    
    Ok(())
}
```

### Governance Monitoring

```rust
use hauptbuch_governance::{GovernanceClient, GovernanceEvent, EventType};

async fn monitor_governance_events(
    client: &GovernanceClient,
) -> Result<(), GovernanceError> {
    let mut event_stream = client.subscribe_to_governance_events().await?;
    
    while let Some(event) = event_stream.next().await {
        match event.event_type {
            EventType::ProposalCreated => {
                println!("New proposal created: {}", event.proposal_id);
            }
            EventType::VoteCast => {
                println!("Vote cast on proposal {}: {}", event.proposal_id, event.vote_choice);
            }
            EventType::ProposalExecuted => {
                println!("Proposal {} executed", event.proposal_id);
            }
            EventType::ProposalFailed => {
                println!("Proposal {} failed", event.proposal_id);
            }
        }
    }
    
    Ok(())
}
```

## Conclusion

These governance examples provide comprehensive guidance for implementing and managing governance on the Hauptbuch blockchain platform. Follow the best practices and security considerations to ensure effective and secure governance operations.
