# Voting Contract

## Overview

The Voting Contract is a comprehensive smart contract implementation for governance and voting on the Hauptbuch blockchain platform. It provides secure, transparent, and efficient voting mechanisms with support for various voting types and advanced features.

## Table of Contents

- [Contract Overview](#contract-overview)
- [Voting Types](#voting-types)
- [Core Functions](#core-functions)
- [Advanced Features](#advanced-features)
- [Security Considerations](#security-considerations)
- [Usage Examples](#usage-examples)
- [Integration Guide](#integration-guide)
- [Best Practices](#best-practices)

## Contract Overview

### Contract Address

```
Mainnet: 0x1234567890123456789012345678901234567890
Testnet: 0xabcdefabcdefabcdefabcdefabcdefabcdefabcd
```

### Contract Interface

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

interface IVotingContract {
    // Core voting functions
    function createProposal(
        string memory title,
        string memory description,
        uint256 votingPeriod,
        uint256 quorumThreshold,
        uint256 supermajorityThreshold
    ) external returns (uint256);
    
    function vote(uint256 proposalId, uint8 choice, string memory reason) external;
    function executeProposal(uint256 proposalId) external;
    
    // Proposal management
    function getProposal(uint256 proposalId) external view returns (Proposal memory);
    function getProposalStatus(uint256 proposalId) external view returns (ProposalStatus);
    function getProposalVotes(uint256 proposalId) external view returns (Vote[] memory);
    
    // Voting power management
    function getVotingPower(address voter) external view returns (uint256);
    function delegateVotingPower(address delegate) external;
    function undelegateVotingPower() external;
    
    // Emergency functions
    function emergencyPause() external;
    function emergencyUnpause() external;
    function emergencyExecute(uint256 proposalId) external;
}
```

### Data Structures

```solidity
struct Proposal {
    uint256 id;
    string title;
    string description;
    address proposer;
    uint256 startTime;
    uint256 endTime;
    uint256 quorumThreshold;
    uint256 supermajorityThreshold;
    uint256 totalVotes;
    uint256 yesVotes;
    uint256 noVotes;
    uint256 abstainVotes;
    ProposalStatus status;
    bool executed;
    mapping(address => bool) hasVoted;
}

struct Vote {
    address voter;
    uint8 choice; // 0 = abstain, 1 = no, 2 = yes
    uint256 votingPower;
    string reason;
    uint256 timestamp;
}

enum ProposalStatus {
    Pending,
    Active,
    Passed,
    Failed,
    Executed,
    Cancelled
}
```

## Voting Types

### Standard Voting

```solidity
function createStandardProposal(
    string memory title,
    string memory description,
    uint256 votingPeriod
) external returns (uint256) {
    return createProposal(
        title,
        description,
        votingPeriod,
        quorumThreshold,
        supermajorityThreshold
    );
}
```

### Weighted Voting

```solidity
function createWeightedProposal(
    string memory title,
    string memory description,
    uint256 votingPeriod,
    mapping(address => uint256) weights
) external returns (uint256) {
    // Implementation for weighted voting
}
```

### Quadratic Voting

```solidity
function createQuadraticProposal(
    string memory title,
    string memory description,
    uint256 votingPeriod,
    uint256 maxVotesPerOption
) external returns (uint256) {
    // Implementation for quadratic voting
}
```

### Ranked Choice Voting

```solidity
function createRankedChoiceProposal(
    string memory title,
    string memory description,
    uint256 votingPeriod,
    string[] memory options
) external returns (uint256) {
    // Implementation for ranked choice voting
}
```

## Core Functions

### Proposal Creation

```solidity
function createProposal(
    string memory title,
    string memory description,
    uint256 votingPeriod,
    uint256 quorumThreshold,
    uint256 supermajorityThreshold
) external returns (uint256) {
    require(bytes(title).length > 0, "Title cannot be empty");
    require(bytes(description).length > 0, "Description cannot be empty");
    require(votingPeriod > 0, "Voting period must be positive");
    require(quorumThreshold <= 100, "Quorum threshold cannot exceed 100%");
    require(supermajorityThreshold <= 100, "Supermajority threshold cannot exceed 100%");
    
    uint256 proposalId = proposalCount++;
    
    proposals[proposalId] = Proposal({
        id: proposalId,
        title: title,
        description: description,
        proposer: msg.sender,
        startTime: block.timestamp,
        endTime: block.timestamp + votingPeriod,
        quorumThreshold: quorumThreshold,
        supermajorityThreshold: supermajorityThreshold,
        totalVotes: 0,
        yesVotes: 0,
        noVotes: 0,
        abstainVotes: 0,
        status: ProposalStatus.Active,
        executed: false
    });
    
    emit ProposalCreated(proposalId, msg.sender, title, votingPeriod);
    
    return proposalId;
}
```

### Voting

```solidity
function vote(
    uint256 proposalId,
    uint8 choice,
    string memory reason
) external {
    require(proposalId < proposalCount, "Proposal does not exist");
    require(proposals[proposalId].status == ProposalStatus.Active, "Proposal is not active");
    require(block.timestamp <= proposals[proposalId].endTime, "Voting period has ended");
    require(!proposals[proposalId].hasVoted[msg.sender], "Already voted");
    require(choice <= 2, "Invalid vote choice");
    
    uint256 votingPower = getVotingPower(msg.sender);
    require(votingPower > 0, "No voting power");
    
    proposals[proposalId].hasVoted[msg.sender] = true;
    proposals[proposalId].totalVotes += votingPower;
    
    if (choice == 2) {
        proposals[proposalId].yesVotes += votingPower;
    } else if (choice == 1) {
        proposals[proposalId].noVotes += votingPower;
    } else {
        proposals[proposalId].abstainVotes += votingPower;
    }
    
    votes[proposalId].push(Vote({
        voter: msg.sender,
        choice: choice,
        votingPower: votingPower,
        reason: reason,
        timestamp: block.timestamp
    }));
    
    emit VoteCast(proposalId, msg.sender, choice, votingPower, reason);
}
```

### Proposal Execution

```solidity
function executeProposal(uint256 proposalId) external {
    require(proposalId < proposalCount, "Proposal does not exist");
    require(proposals[proposalId].status == ProposalStatus.Passed, "Proposal has not passed");
    require(!proposals[proposalId].executed, "Proposal already executed");
    require(block.timestamp >= proposals[proposalId].endTime, "Voting period has not ended");
    
    proposals[proposalId].executed = true;
    proposals[proposalId].status = ProposalStatus.Executed;
    
    // Execute proposal logic here
    _executeProposal(proposalId);
    
    emit ProposalExecuted(proposalId);
}
```

### Voting Power Management

```solidity
function getVotingPower(address voter) public view returns (uint256) {
    uint256 balance = token.balanceOf(voter);
    uint256 delegated = delegatedVotingPower[voter];
    return balance + delegated;
}

function delegateVotingPower(address delegate) external {
    require(delegate != msg.sender, "Cannot delegate to self");
    require(delegate != address(0), "Invalid delegate address");
    
    uint256 votingPower = token.balanceOf(msg.sender);
    require(votingPower > 0, "No voting power to delegate");
    
    // Undelegate previous delegate if any
    if (delegates[msg.sender] != address(0)) {
        undelegateVotingPower();
    }
    
    delegates[msg.sender] = delegate;
    delegatedVotingPower[delegate] += votingPower;
    
    emit VotingPowerDelegated(msg.sender, delegate, votingPower);
}

function undelegateVotingPower() public {
    address delegate = delegates[msg.sender];
    require(delegate != address(0), "No delegate to undelegate");
    
    uint256 votingPower = token.balanceOf(msg.sender);
    delegatedVotingPower[delegate] -= votingPower;
    delegates[msg.sender] = address(0);
    
    emit VotingPowerUndelegated(msg.sender, delegate, votingPower);
}
```

## Advanced Features

### Time-Locked Execution

```solidity
function createTimeLockedProposal(
    string memory title,
    string memory description,
    uint256 votingPeriod,
    uint256 executionDelay
) external returns (uint256) {
    uint256 proposalId = createProposal(
        title,
        description,
        votingPeriod,
        quorumThreshold,
        supermajorityThreshold
    );
    
    timeLockDelays[proposalId] = executionDelay;
    
    return proposalId;
}

function executeTimeLockedProposal(uint256 proposalId) external {
    require(proposalId < proposalCount, "Proposal does not exist");
    require(proposals[proposalId].status == ProposalStatus.Passed, "Proposal has not passed");
    require(!proposals[proposalId].executed, "Proposal already executed");
    
    uint256 executionDelay = timeLockDelays[proposalId];
    require(block.timestamp >= proposals[proposalId].endTime + executionDelay, "Execution delay not met");
    
    proposals[proposalId].executed = true;
    proposals[proposalId].status = ProposalStatus.Executed;
    
    _executeProposal(proposalId);
    
    emit ProposalExecuted(proposalId);
}
```

### Multi-Signature Proposals

```solidity
function createMultiSigProposal(
    string memory title,
    string memory description,
    address[] memory signers,
    uint256 threshold
) external returns (uint256) {
    require(signers.length > 0, "No signers provided");
    require(threshold > 0 && threshold <= signers.length, "Invalid threshold");
    
    uint256 proposalId = proposalCount++;
    
    proposals[proposalId] = Proposal({
        id: proposalId,
        title: title,
        description: description,
        proposer: msg.sender,
        startTime: block.timestamp,
        endTime: block.timestamp + 86400, // 24 hours
        quorumThreshold: 0,
        supermajorityThreshold: 0,
        totalVotes: 0,
        yesVotes: 0,
        noVotes: 0,
        abstainVotes: 0,
        status: ProposalStatus.Active,
        executed: false
    });
    
    multiSigSigners[proposalId] = signers;
    multiSigThresholds[proposalId] = threshold;
    
    emit MultiSigProposalCreated(proposalId, msg.sender, signers, threshold);
    
    return proposalId;
}
```

### Emergency Proposals

```solidity
function createEmergencyProposal(
    string memory title,
    string memory description,
    string memory emergencyReason
) external returns (uint256) {
    require(hasRole(EMERGENCY_ROLE, msg.sender), "Not authorized for emergency proposals");
    
    uint256 proposalId = createProposal(
        title,
        description,
        3600, // 1 hour voting period
        0, // No quorum requirement
        51 // Simple majority
    );
    
    emergencyProposals[proposalId] = true;
    emergencyReasons[proposalId] = emergencyReason;
    
    emit EmergencyProposalCreated(proposalId, msg.sender, emergencyReason);
    
    return proposalId;
}
```

## Security Considerations

### Access Control

```solidity
modifier onlyAuthorized() {
    require(hasRole(AUTHORIZED_ROLE, msg.sender), "Not authorized");
    _;
}

modifier onlyEmergency() {
    require(hasRole(EMERGENCY_ROLE, msg.sender), "Not authorized for emergency");
    _;
}

modifier onlyAdmin() {
    require(hasRole(ADMIN_ROLE, msg.sender), "Not admin");
    _;
}
```

### Reentrancy Protection

```solidity
modifier nonReentrant() {
    require(!locked, "Reentrant call");
    locked = true;
    _;
    locked = false;
}
```

### Input Validation

```solidity
modifier validProposal(uint256 proposalId) {
    require(proposalId < proposalCount, "Proposal does not exist");
    _;
}

modifier validVoteChoice(uint8 choice) {
    require(choice <= 2, "Invalid vote choice");
    _;
}
```

## Usage Examples

### Basic Voting

```solidity
// Create a proposal
uint256 proposalId = votingContract.createProposal(
    "Increase Gas Limit",
    "Proposal to increase the gas limit from 10M to 15M",
    7 days,
    10, // 10% quorum
    67  // 67% supermajority
);

// Vote on the proposal
votingContract.vote(proposalId, 2, "I support this proposal");

// Execute the proposal after voting period
votingContract.executeProposal(proposalId);
```

### Delegated Voting

```solidity
// Delegate voting power
votingContract.delegateVotingPower(delegateAddress);

// Vote with delegated power
votingContract.vote(proposalId, 2, "Voting with delegated power");

// Undelegate voting power
votingContract.undelegateVotingPower();
```

### Emergency Voting

```solidity
// Create emergency proposal
uint256 emergencyProposalId = votingContract.createEmergencyProposal(
    "Emergency Security Update",
    "Critical security update required",
    "Security vulnerability discovered"
);

// Vote on emergency proposal
votingContract.vote(emergencyProposalId, 2, "Critical security issue");

// Execute emergency proposal
votingContract.executeProposal(emergencyProposalId);
```

## Integration Guide

### Solidity Integration

```solidity
import "@hauptbuch/voting-contract/contracts/VotingContract.sol";

contract MyGovernanceContract {
    VotingContract public votingContract;
    
    constructor(address _votingContract) {
        votingContract = VotingContract(_votingContract);
    }
    
    function proposeChange(string memory title, string memory description) external {
        uint256 proposalId = votingContract.createProposal(
            title,
            description,
            7 days,
            10,
            67
        );
        
        emit ProposalCreated(proposalId);
    }
}
```

### Rust Integration

```rust
use hauptbuch_smart_contracts::{VotingContract, Proposal, Vote};

#[tokio::main]
async fn main() -> Result<(), ContractError> {
    let voting_contract = VotingContract::new("0x1234...")?;
    
    // Create proposal
    let proposal_id = voting_contract.create_proposal(
        "Test Proposal",
        "This is a test proposal",
        7 * 24 * 60 * 60, // 7 days
        10, // 10% quorum
        67, // 67% supermajority
    ).await?;
    
    // Vote on proposal
    voting_contract.vote(proposal_id, 2, "I support this proposal").await?;
    
    // Execute proposal
    voting_contract.execute_proposal(proposal_id).await?;
    
    Ok(())
}
```

### JavaScript Integration

```javascript
import { VotingContract } from '@hauptbuch/voting-contract';

const votingContract = new VotingContract('0x1234...');

// Create proposal
const proposalId = await votingContract.createProposal(
    'Test Proposal',
    'This is a test proposal',
    7 * 24 * 60 * 60, // 7 days
    10, // 10% quorum
    67  // 67% supermajority
);

// Vote on proposal
await votingContract.vote(proposalId, 2, 'I support this proposal');

// Execute proposal
await votingContract.executeProposal(proposalId);
```

## Best Practices

### Security Best Practices

1. **Input Validation**: Always validate inputs to prevent malicious data
2. **Access Control**: Implement proper role-based access control
3. **Reentrancy Protection**: Use reentrancy guards for state-changing functions
4. **Integer Overflow**: Use SafeMath for arithmetic operations
5. **Event Logging**: Log all important events for transparency

### Gas Optimization

1. **Batch Operations**: Group multiple operations to reduce gas costs
2. **Storage Optimization**: Use packed structs to reduce storage costs
3. **Function Optimization**: Optimize functions to reduce gas consumption
4. **Event Optimization**: Use indexed parameters for events

### Governance Best Practices

1. **Transparency**: Make all proposals and votes publicly visible
2. **Participation**: Encourage community participation in governance
3. **Education**: Provide clear information about proposals
4. **Fairness**: Ensure fair representation in voting
5. **Accountability**: Hold proposers accountable for their proposals

## Conclusion

The Voting Contract provides a comprehensive and secure foundation for governance on the Hauptbuch blockchain platform. By following the best practices and security considerations outlined in this document, you can implement effective governance mechanisms for your decentralized applications.
