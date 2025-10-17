# Governance Token Smart Contract

A comprehensive Solidity smart contract for a native governance token with staking, slashing, and governance functionality for the decentralized voting blockchain.

## 🎯 **Contract Overview**

The `GovernanceToken` contract implements a native governance token with the following key features:

- **ERC-20-like functionality** with minting, burning, and transfers
- **Staking mechanism** for PoS validator participation
- **Slashing mechanism** for malicious behavior detection
- **Governance voting weight** proportional to staked tokens
- **Gas-optimized storage** and operations
- **Comprehensive security** features and emergency controls

## 📁 **Contract Structure**

```
contracts/
├── GovernanceToken.sol          # Main governance token contract
├── GovernanceTokenTest.sol      # Comprehensive test suite
├── verify_governance.js         # Verification script
└── README_GOVERNANCE.md         # This documentation
```

## 🔧 **Core Features**

### **1. ERC-20-like Token Functionality**

#### **Basic Token Operations**
```solidity
// Token information
string public constant name = "Hauptbuch Governance Token";
string public constant symbol = "HGT";
uint8 public constant decimals = 18;
uint256 public totalSupply;
uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**18; // 1 billion tokens

// Standard ERC-20 functions
function transfer(address to, uint256 amount) external returns (bool)
function transferFrom(address from, address to, uint256 amount) external returns (bool)
function approve(address spender, uint256 amount) external returns (bool)
function balanceOf(address account) external view returns (uint256)
function allowance(address owner, address spender) external view returns (uint256)
```

#### **Minting and Burning**
```solidity
// Minting (owner only)
function mint(address to, uint256 amount) external onlyOwner
function batchMint(address[] calldata recipients, uint256[] calldata amounts) external onlyOwner

// Burning
function burn(uint256 amount) external
function burnFrom(address from, uint256 amount) external
```

### **2. Staking Mechanism**

#### **Staking Functions**
```solidity
// Stake tokens for PoS validator participation
function stake(uint256 amount, uint256 lockPeriod) external whenNotPaused whenStakingEnabled

// Unstake tokens after lock period
function unstake(uint256 amount) external whenNotPaused

// Emergency unstake with penalty
function emergencyUnstake(uint256 amount) external whenNotPaused
```

#### **Staking Parameters**
- **Minimum Stake**: 1,000 tokens
- **Maximum Lock Period**: 365 days
- **Emergency Penalty**: 50% of staked amount

#### **Gas-Optimized Storage**
```solidity
// Packed staking info for gas efficiency
mapping(address => uint256) public stakingInfo; // lockPeriod (32 bits) + startTime (32 bits)

// Helper functions for packing/unpacking
function _packStakingInfo(uint256 lockPeriod, uint256 startTime) internal pure returns (uint256)
function _unpackStakingInfo(address user) internal view returns (uint256 lockPeriod, uint256 startTime)
```

### **3. Slashing Mechanism**

#### **Slashing Functions**
```solidity
// Slash specific amount (PoS consensus only)
function slash(address user, uint256 amount, string calldata reason) external onlyPoSConsensus

// Slash percentage of staked tokens
function slashPercentage(address user, uint256 percentage, string calldata reason) external onlyPoSConsensus
```

#### **Slashing Parameters**
- **Slash Penalty**: 5% (configurable)
- **Authorization**: Only PoS consensus module can trigger slashing
- **Reasons**: Malicious behavior, double signing, etc.

### **4. Governance Voting Weight**

#### **Voting Weight System**
```solidity
// Voting weight proportional to staked tokens
mapping(address => uint256) public votingWeights;
uint256 public totalVotingWeight;

// Get voting weight for address
function getVotingWeight(address user) external view returns (uint256)
function getTotalVotingWeight() external view returns (uint256)

// Update voting weight (voting contract only)
function updateVotingWeight(address user, uint256 amount, bool isAddition) external onlyVotingContract
```

### **5. Security Features**

#### **Access Control**
```solidity
modifier onlyOwner()           // Contract owner
modifier onlyPoSConsensus()    // PoS consensus module
modifier onlyVotingContract()  // Voting contract
```

#### **Reentrancy Protection**
```solidity
bool private _locked;
modifier nonReentrant() {
    require(!_locked, "ReentrancyGuard: reentrant call");
    _locked = true;
    _;
    _locked = false;
}
```

#### **Emergency Controls**
```solidity
bool public emergencyPaused;    // Emergency pause
bool public stakingEnabled;     // Staking toggle
bool public transfersEnabled;   // Transfers toggle

function toggleEmergencyPause() external onlyOwner
function toggleStaking() external onlyOwner
function toggleTransfers() external onlyOwner
```

### **6. Gas Optimization Features**

#### **Storage Packing**
- **Staking Info**: Packed into single storage slot (lockPeriod + startTime)
- **Efficient Data Structures**: Optimized mappings and arrays
- **Batch Operations**: Multiple operations in single transaction

#### **Batch Functions**
```solidity
// Batch transfer for gas efficiency
function batchTransfer(address[] calldata recipients, uint256[] calldata amounts) external

// Batch mint for gas efficiency
function batchMint(address[] calldata recipients, uint256[] calldata amounts) external onlyOwner
```

## 🧪 **Comprehensive Test Suite**

### **Test Categories (32 Total Tests)**

#### **Normal Operation Tests (6 tests)**
1. `testTokenMinting()` - Token minting functionality
2. `testTokenTransfer()` - Token transfer functionality
3. `testTokenApproval()` - Token approval and transferFrom
4. `testTokenBurning()` - Token burning functionality
5. `testStaking()` - Staking functionality
6. `testVotingWeight()` - Voting weight calculation

#### **Edge Case Tests (6 tests)**
7. `testZeroBalanceTransfer()` - Zero balance transfers
8. `testMaximumTokenSupply()` - Maximum token supply limits
9. `testInvalidAddresses()` - Invalid address handling
10. `testMinimumStaking()` - Minimum staking requirements
11. `testMaximumStakingPeriod()` - Maximum staking period limits
12. `testZeroAmountOperations()` - Zero amount operations

#### **Malicious Behavior Tests (4 tests)**
13. `testReentrancyPrevention()` - Reentrancy attack prevention
14. `testUnauthorizedSlashing()` - Unauthorized slashing attempts
15. `testFrontRunningPrevention()` - Front-running prevention
16. `testOverflowProtection()` - Overflow protection

#### **Stress Tests (8 tests)**
17. `testLargeTokenTransfers()` - Large token transfers
18. `testHighStakingVolumes()` - High staking volumes
19. `testEmergencyPause()` - Emergency pause functionality
20. `testStakingToggle()` - Staking toggle functionality
21. `testTransfersToggle()` - Transfers toggle functionality
22. `testSlashing()` - Slashing functionality
23. `testPercentageSlashing()` - Percentage slashing
24. `testUnstakingAfterLockPeriod()` - Unstaking after lock period

#### **Advanced Tests (8 tests)**
25. `testEmergencyUnstaking()` - Emergency unstaking with penalty
26. `testContractStatistics()` - Contract statistics
27. `testStakingInformation()` - Staking information queries
28. `testCanStakeUnstakeChecks()` - Can stake/unstake checks
29. `testBatchTransfer()` - Batch transfer functionality
30. `testBatchMint()` - Batch mint functionality
31. `testGasOptimization()` - Gas optimization with packed storage
32. `testBatchOverflowProtection()` - Overflow protection with batch operations

### **Test Coverage**
- **Total Tests**: 32 comprehensive test cases
- **Normal Operation**: 6 tests
- **Edge Cases**: 6 tests
- **Malicious Behavior**: 4 tests
- **Stress Tests**: 8 tests
- **Advanced Tests**: 8 tests
- **Coverage**: Near-100% code coverage

## 🔒 **Security Features**

### **1. Reentrancy Protection**
- **Reentrancy Guard**: Prevents reentrancy attacks
- **Lock Mechanism**: `_locked` flag with `nonReentrant` modifier
- **Safe Operations**: All external calls protected

### **2. Access Control**
- **Owner Functions**: Minting, burning, administrative functions
- **PoS Consensus**: Slashing functions only
- **Voting Contract**: Voting weight updates only
- **Role-based Permissions**: Clear separation of concerns

### **3. Input Validation**
- **Address Validation**: Zero address checks
- **Amount Validation**: Positive amount requirements
- **Range Validation**: Minimum/maximum limits
- **Array Validation**: Length and content validation

### **4. Overflow Protection**
- **Safe Arithmetic**: Checked operations where needed
- **Balance Validation**: Sufficient balance checks
- **Supply Limits**: Maximum supply enforcement
- **Batch Validation**: Array length and total amount checks

### **5. Emergency Controls**
- **Emergency Pause**: Global pause functionality
- **Feature Toggles**: Staking and transfers can be disabled
- **Administrative Functions**: Owner can update critical parameters
- **Graceful Degradation**: Safe operation during emergencies

## ⛽ **Gas Optimization**

### **1. Storage Optimization**
- **Packed Storage**: Staking info packed into single slot
- **Efficient Mappings**: Optimized data structures
- **Minimal State Updates**: Only necessary state changes
- **Batch Operations**: Multiple operations in single transaction

### **2. Memory Optimization**
- **Stack Variables**: Local variables where possible
- **Efficient Loops**: Optimized iteration patterns
- **Reduced Allocations**: Minimize dynamic memory usage
- **Cache-friendly Access**: Sequential memory patterns

### **3. Function Optimization**
- **Early Returns**: Fast failure for invalid inputs
- **Batch Processing**: Multiple operations efficiently
- **Gas-efficient Operations**: Optimized arithmetic and logic
- **Minimal External Calls**: Reduced cross-contract calls

## 🚀 **Deployment and Usage**

### **1. Contract Deployment**
```solidity
// Deploy with initial supply and owner
GovernanceToken token = new GovernanceToken(INITIAL_SUPPLY, owner);

// Set up contract addresses
token.setPoSConsensus(posConsensusAddress);
token.setVotingContract(votingContractAddress);
```

### **2. Basic Usage**
```solidity
// Transfer tokens
token.transfer(recipient, amount);

// Approve and transferFrom
token.approve(spender, amount);
token.transferFrom(owner, recipient, amount);

// Mint new tokens (owner only)
token.mint(recipient, amount);
```

### **3. Staking Operations**
```solidity
// Stake tokens
token.stake(amount, lockPeriod);

// Unstake after lock period
token.unstake(amount);

// Emergency unstake (with penalty)
token.emergencyUnstake(amount);
```

### **4. Governance Operations**
```solidity
// Get voting weight
uint256 weight = token.getVotingWeight(user);

// Update voting weight (voting contract only)
token.updateVotingWeight(user, amount, true);
```

### **5. Administrative Operations**
```solidity
// Emergency controls
token.toggleEmergencyPause();
token.toggleStaking();
token.toggleTransfers();

// Slashing (PoS consensus only)
token.slash(user, amount, "Malicious behavior");
```

## 📊 **Integration with Other Contracts**

### **1. PoS Consensus Integration**
- **Slashing Triggers**: PoS consensus can slash malicious validators
- **Validator Status**: Staking status affects validator selection
- **Validator Selection**: Staked tokens influence selection probability

### **2. Voting Contract Integration**
- **Voting Weight**: Proportional to staked tokens
- **Weight Updates**: Automatic updates on stake/unstake
- **Governance Participation**: Staked tokens determine voting power

### **3. Token Economics**
- **Staking Rewards**: Incentivize long-term participation
- **Slashing Penalties**: Deter malicious behavior
- **Governance Power**: Voting weight proportional to stake
- **Economic Security**: Stake provides economic security for network

## 🔧 **Configuration Parameters**

### **Token Parameters**
- **Name**: "Hauptbuch Governance Token"
- **Symbol**: "HGT"
- **Decimals**: 18
- **Max Supply**: 1,000,000,000 tokens

### **Staking Parameters**
- **Minimum Stake**: 1,000 tokens
- **Maximum Lock Period**: 365 days
- **Emergency Penalty**: 50%

### **Security Parameters**
- **Slash Penalty**: 5%
- **Batch Limit**: 100 recipients
- **Reentrancy Protection**: Enabled
- **Overflow Protection**: Enabled

## 📝 **Events and Monitoring**

### **Token Events**
```solidity
event Transfer(address indexed from, address indexed to, uint256 value);
event Approval(address indexed owner, address indexed spender, uint256 value);
```

### **Staking Events**
```solidity
event Staked(address indexed user, uint256 amount, uint256 lockPeriod);
event Unstaked(address indexed user, uint256 amount);
```

### **Slashing Events**
```solidity
event Slashed(address indexed user, uint256 amount, string reason);
```

### **Governance Events**
```solidity
event VotingWeightUpdated(address indexed user, uint256 oldWeight, uint256 newWeight);
```

### **Administrative Events**
```solidity
event EmergencyPauseToggled(bool paused);
event StakingToggled(bool enabled);
event TransfersToggled(bool enabled);
event PoSConsensusUpdated(address indexed oldAddress, address indexed newAddress);
event VotingContractUpdated(address indexed oldAddress, address indexed newAddress);
```

## ⚠️ **Important Considerations**

### **1. Security Considerations**
- **Private Key Security**: Secure key management for owner
- **Contract Upgrades**: Consider upgradeability if needed
- **Audit Requirements**: Thorough security audit before mainnet
- **Emergency Procedures**: Clear emergency response procedures

### **2. Economic Considerations**
- **Token Distribution**: Fair initial distribution
- **Staking Incentives**: Appropriate rewards for staking
- **Slashing Parameters**: Balanced penalty system
- **Governance Participation**: Encourage active participation

### **3. Technical Considerations**
- **Gas Costs**: Monitor gas usage for operations
- **Scalability**: Consider scaling solutions
- **Integration**: Ensure compatibility with other contracts
- **Upgrades**: Plan for future improvements

## 🎯 **Best Practices**

### **1. Development**
- **Test Coverage**: Maintain high test coverage
- **Code Review**: Regular code reviews
- **Documentation**: Keep documentation updated
- **Version Control**: Proper version management

### **2. Deployment**
- **Testnet Testing**: Thorough testing on testnets
- **Gradual Rollout**: Phased deployment approach
- **Monitoring**: Continuous monitoring post-deployment
- **Emergency Procedures**: Clear emergency response

### **3. Operations**
- **Regular Audits**: Periodic security audits
- **Parameter Updates**: Careful parameter adjustments
- **Community Engagement**: Active community involvement
- **Governance Participation**: Encourage token holder participation

## 📚 **References**

- [ERC-20 Token Standard](https://eips.ethereum.org/EIPS/eip-20)
- [Solidity Security Best Practices](https://consensys.github.io/smart-contract-best-practices/)
- [Gas Optimization Techniques](https://docs.soliditylang.org/en/v0.8.19/gas-optimization.html)
- [Staking Mechanisms](https://ethereum.org/en/developers/docs/consensus-mechanisms/pos/)
- [Governance Token Design](https://compound.finance/docs/governance)

---

**⚠️ Disclaimer**: This contract is for educational and research purposes. For production use, conduct thorough security audits and consider additional security measures.
