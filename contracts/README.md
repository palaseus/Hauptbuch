# Voting Smart Contract

A secure, privacy-preserving vote tallying smart contract built with Solidity, featuring zk-SNARKs for anonymous voting and comprehensive security measures.

## 🎯 Features

### Core Functionality
- **Anonymous Vote Submission**: Uses zk-SNARKs to prove voter eligibility without revealing identity
- **Double Voting Prevention**: Nullifier mechanism prevents duplicate votes
- **Merkle Tree Verification**: Efficient voter commitment verification
- **Gas Optimization**: Storage packing and minimal state updates
- **Access Control**: Restricted vote submission and result finalization

### Security Features
- **Reentrancy Protection**: Guard against reentrancy attacks
- **Front-running Prevention**: Nullifier mechanism prevents vote manipulation
- **Input Validation**: Comprehensive validation of all inputs
- **Cryptographic Security**: SHA-3 hashing and ECDSA-style signatures
- **Access Control**: Role-based permissions for administrative functions

## 📁 Contract Structure

```
contracts/
├── Voting.sol          # Main voting contract
├── VotingTest.sol      # Comprehensive test suite
├── compile.js          # Verification script
└── README.md           # This documentation
```

## 🔧 Contract Architecture

### Voting.sol - Main Contract

#### State Variables
- `_locked`: Reentrancy guard
- `owner`: Contract administrator
- `VotingSession`: Current voting session configuration
- `registeredVoters`: Mapping of registered voter addresses
- `usedNullifiers`: Prevents double voting
- `voterCommitments`: Merkle tree commitments
- `zkParams`: zk-SNARKs verification parameters

#### Key Functions

**Voter Registration**
```solidity
function registerVoter(address voterAddress, bytes32 commitment) external onlyOwner
function batchRegisterVoters(address[] calldata voterAddresses, bytes32[] calldata commitments) external onlyOwner
```

**Voting Session Management**
```solidity
function createVotingSession(string memory _title, string[] memory _options, uint256 _startTime, uint256 _endTime) external onlyOwner
function finalizeVotingSession() external onlyOwner
```

**Vote Submission**
```solidity
function submitVote(bytes32 nullifier, uint256 optionIndex, bytes32[] memory merkleProof, uint256[8] memory zkProof) external onlyActiveSession nonReentrant
```

**Result Verification**
```solidity
function getVoteCount(uint256 optionIndex) external view onlyFinalizedSession returns (uint256)
function getAllVoteResults() external view onlyFinalizedSession returns (uint256[] memory)
```

### VotingTest.sol - Test Suite

#### Test Categories

**Normal Operation Tests (5 tests)**
- `testVoterRegistration()`: Voter registration functionality
- `testBatchVoterRegistration()`: Batch voter registration
- `testVotingSessionCreation()`: Voting session creation
- `testValidVoteSubmission()`: Valid vote submission
- `testVoteTallying()`: Vote tallying and result verification

**Edge Case Tests (5 tests)**
- `testInvalidZKProof()`: Invalid zk-SNARKs proof handling
- `testDuplicateVotePrevention()`: Duplicate vote prevention
- `testUnauthorizedAccess()`: Unauthorized access prevention
- `testInvalidVoteOption()`: Invalid vote option handling
- `testVotingSessionExpiration()`: Voting session expiration

**Malicious Behavior Tests (3 tests)**
- `testReentrancyPrevention()`: Reentrancy attack prevention
- `testFrontRunningPrevention()`: Front-running attack prevention
- `testInvalidMerkleProof()`: Invalid Merkle proof handling

**Stress Tests (7 tests)**
- `testLargeVoterCount()`: Large voter count handling
- `testMaximumVoteOptions()`: Maximum vote options
- `testGasEfficiency()`: Gas efficiency under load
- `testMerkleTreeDepth()`: Merkle tree depth handling
- `testZKParametersValidation()`: ZK parameters validation
- `testEmergencyPause()`: Emergency pause functionality
- `testComprehensiveIntegration()`: Full integration test

## 🚀 Usage

### Deployment

1. **Initialize zk-SNARKs Parameters**
```solidity
Voting.ZKParams memory zkParams = Voting.ZKParams({
    alpha: [uint256(1), uint256(2)],
    beta: [[uint256(3), uint256(4)], [uint256(5), uint256(6)]],
    gamma: [uint256(7), uint256(8)],
    delta: [uint256(9), uint256(10)],
    ic: new uint256[2][](2)
};

Voting voting = new Voting(zkParams, 3); // 3 = Merkle tree depth
```

2. **Register Voters**
```solidity
bytes32 commitment = keccak256(abi.encodePacked("voter_secret"));
voting.registerVoter(voterAddress, commitment);
```

3. **Create Voting Session**
```solidity
string[] memory options = ["Option A", "Option B", "Option C"];
voting.createVotingSession("Election", options, startTime, endTime);
```

4. **Submit Votes**
```solidity
bytes32 nullifier = keccak256(abi.encodePacked("unique_nullifier"));
bytes32[] memory merkleProof = new bytes32[](0);
uint256[8] memory zkProof = [1, 2, 3, 4, 5, 6, 7, 8];
voting.submitVote(nullifier, optionIndex, merkleProof, zkProof);
```

5. **Finalize and Get Results**
```solidity
voting.finalizeVotingSession();
uint256[] memory results = voting.getAllVoteResults();
```

### Testing

Run the comprehensive test suite:

```bash
node compile.js
```

The test suite includes:
- **20 test functions** covering all scenarios
- **Near-100% code coverage**
- **Automatic test execution**
- **Descriptive error messages**

## 🔒 Security Considerations

### zk-SNARKs Privacy
- Voters prove eligibility without revealing identity
- Nullifier mechanism prevents double voting
- Merkle tree enables efficient commitment verification

### Attack Prevention
- **Reentrancy**: Guard prevents multiple simultaneous calls
- **Front-running**: Nullifier uniqueness prevents manipulation
- **Double Voting**: Used nullifiers tracking
- **Unauthorized Access**: Role-based access control

### Gas Optimization
- Storage packing for efficient data storage
- Minimal state updates
- Batch operations for multiple voters
- Optimized Merkle tree calculations

## 📊 Gas Efficiency

The contract is optimized for gas efficiency:

- **Storage Packing**: Efficient struct packing
- **Batch Operations**: Multiple voter registration in single transaction
- **Minimal State Updates**: Only necessary state changes
- **Optimized Algorithms**: Efficient Merkle tree and zk-SNARKs verification

## 🧪 Testing Coverage

The test suite achieves comprehensive coverage:

- **Normal Operation**: 5 tests
- **Edge Cases**: 5 tests  
- **Malicious Behavior**: 3 tests
- **Stress Tests**: 7 tests
- **Total**: 20 test functions

## 🔧 Customization

### zk-SNARKs Parameters
Update verification parameters for different proving systems:

```solidity
voting.updateZKParams(newZKParams);
```

### Merkle Tree Depth
Adjust tree depth based on expected voter count:

```solidity
voting.updateMerkleTreeDepth(newDepth);
```

### Emergency Controls
Pause/resume voting in emergency situations:

```solidity
voting.pauseVoting();
voting.resumeVoting();
```

## 📝 Events

The contract emits events for transparency:

- `VoterRegistered`: Voter registration
- `VoteSubmitted`: Vote submission
- `VotingSessionCreated`: Session creation
- `VotingSessionFinalized`: Results finalization
- `ZKParamsUpdated`: Parameter updates

## ⚠️ Important Notes

1. **zk-SNARKs Implementation**: The current implementation uses simplified zk-SNARKs verification. For production, integrate with proper zk-SNARKs libraries like snarkjs.

2. **Trusted Setup**: Ensure proper trusted setup for zk-SNARKs parameters in production.

3. **Gas Limits**: Consider gas limits when registering large numbers of voters.

4. **Security Audit**: Conduct thorough security audit before mainnet deployment.

## 🚀 Deployment Checklist

- [ ] Verify zk-SNARKs parameters
- [ ] Test with expected voter count
- [ ] Validate Merkle tree depth
- [ ] Run comprehensive test suite
- [ ] Conduct security audit
- [ ] Deploy to testnet first
- [ ] Monitor gas usage
- [ ] Verify all functionality

## 📚 References

- [zk-SNARKs Documentation](https://z.cash/technology/zksnarks/)
- [Merkle Trees in Blockchain](https://en.wikipedia.org/wiki/Merkle_tree)
- [Solidity Security Best Practices](https://consensys.github.io/smart-contract-best-practices/)
- [Gas Optimization Techniques](https://docs.soliditylang.org/en/v0.8.19/gas-optimization.html)

---

**⚠️ Disclaimer**: This contract is for educational and research purposes. For production use, conduct thorough security audits and integrate with proper zk-SNARKs libraries.
