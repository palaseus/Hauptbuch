// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title GovernanceToken
 * @dev Native governance token for decentralized voting blockchain
 * @notice Implements ERC-20-like functionality with staking, slashing, and governance features
 * @author Hauptbuch Blockchain Team
 */
contract GovernanceToken {
    // ============ STATE VARIABLES ============
    
    /// @notice Token name
    string public constant name = "Hauptbuch Governance Token";
    
    /// @notice Token symbol
    string public constant symbol = "HGT";
    
    /// @notice Token decimals
    uint8 public constant decimals = 18;
    
    /// @notice Total token supply
    uint256 public totalSupply;
    
    /// @notice Maximum token supply (1 billion tokens)
    uint256 public constant MAX_SUPPLY = 1_000_000_000 * 10**18;
    
    /// @notice Contract owner (minting authority)
    address public owner;
    
    /// @notice Reentrancy guard
    bool private _locked;
    
    /// @notice Token balances mapping
    mapping(address => uint256) public balances;
    
    /// @notice Allowance mapping for ERC-20 transfers
    mapping(address => mapping(address => uint256)) public allowances;
    
    /// @notice Staked balances mapping
    mapping(address => uint256) public stakedBalances;
    
    /// @notice Total staked tokens
    uint256 public totalStaked;
    
    /// @notice Staking information packed into single storage slot
    /// @dev Packed struct: lockPeriod (32 bits) + startTime (32 bits) + unused (192 bits)
    mapping(address => uint256) public stakingInfo;
    
    /// @notice Minimum staking amount
    uint256 public constant MIN_STAKE = 1000 * 10**18; // 1000 tokens
    
    /// @notice Maximum staking period (1 year)
    uint256 public constant MAX_STAKING_PERIOD = 365 days;
    
    /// @notice Slashing penalty percentage (5%)
    uint256 public constant SLASH_PENALTY = 5; // 5%
    
    /// @notice Governance voting weights (proportional to staked tokens)
    mapping(address => uint256) public votingWeights;
    
    /// @notice Total voting weight
    uint256 public totalVotingWeight;
    
    /// @notice PoS consensus module address (for slashing triggers)
    address public posConsensus;
    
    /// @notice Voting contract address (for weight updates)
    address public votingContract;
    
    /// @notice Emergency pause flag
    bool public emergencyPaused;
    
    /// @notice Staking enabled flag
    bool public stakingEnabled = true;
    
    /// @notice Transfer enabled flag
    bool public transfersEnabled = true;
    
    // ============ EVENTS ============
    
    /// @notice Emitted when tokens are transferred
    event Transfer(address indexed from, address indexed to, uint256 value);
    
    /// @notice Emitted when allowance is set
    event Approval(address indexed owner, address indexed spender, uint256 value);
    
    /// @notice Emitted when tokens are staked
    event Staked(address indexed user, uint256 amount, uint256 lockPeriod);
    
    /// @notice Emitted when tokens are unstaked
    event Unstaked(address indexed user, uint256 amount);
    
    /// @notice Emitted when tokens are slashed
    event Slashed(address indexed user, uint256 amount, string reason);
    
    /// @notice Emitted when voting weight is updated
    event VotingWeightUpdated(address indexed user, uint256 oldWeight, uint256 newWeight);
    
    /// @notice Emitted when emergency pause is toggled
    event EmergencyPauseToggled(bool paused);
    
    /// @notice Emitted when staking is toggled
    event StakingToggled(bool enabled);
    
    /// @notice Emitted when transfers are toggled
    event TransfersToggled(bool enabled);
    
    /// @notice Emitted when PoS consensus address is updated
    event PoSConsensusUpdated(address indexed oldAddress, address indexed newAddress);
    
    /// @notice Emitted when voting contract address is updated
    event VotingContractUpdated(address indexed oldAddress, address indexed newAddress);
    
    // ============ MODIFIERS ============
    
    /// @notice Prevents reentrancy attacks
    modifier nonReentrant() {
        require(!_locked, "ReentrancyGuard: reentrant call");
        _locked = true;
        _;
        _locked = false;
    }
    
    /// @notice Restricts access to contract owner
    modifier onlyOwner() {
        require(msg.sender == owner, "Ownable: caller is not the owner");
        _;
    }
    
    /// @notice Restricts access to PoS consensus module
    modifier onlyPoSConsensus() {
        require(msg.sender == posConsensus, "GovernanceToken: caller is not PoS consensus");
        _;
    }
    
    /// @notice Restricts access to voting contract
    modifier onlyVotingContract() {
        require(msg.sender == votingContract, "GovernanceToken: caller is not voting contract");
        _;
    }
    
    /// @notice Prevents operations when contract is paused
    modifier whenNotPaused() {
        require(!emergencyPaused, "GovernanceToken: contract is paused");
        _;
    }
    
    /// @notice Prevents operations when staking is disabled
    modifier whenStakingEnabled() {
        require(stakingEnabled, "GovernanceToken: staking is disabled");
        _;
    }
    
    /// @notice Prevents operations when transfers are disabled
    modifier whenTransfersEnabled() {
        require(transfersEnabled, "GovernanceToken: transfers are disabled");
        _;
    }
    
    // ============ CONSTRUCTOR ============
    
    /// @notice Initializes the governance token contract
    /// @param _initialSupply Initial token supply to mint
    /// @param _owner Contract owner address
    constructor(uint256 _initialSupply, address _owner) {
        require(_owner != address(0), "GovernanceToken: owner cannot be zero address");
        require(_initialSupply <= MAX_SUPPLY, "GovernanceToken: initial supply exceeds maximum");
        
        owner = _owner;
        totalSupply = _initialSupply;
        balances[_owner] = _initialSupply;
        
        emit Transfer(address(0), _owner, _initialSupply);
    }
    
    // ============ ERC-20 FUNCTIONS ============
    
    /// @notice Returns the token balance of an account
    /// @param account Address to query balance for
    /// @return Balance of the account
    function balanceOf(address account) external view returns (uint256) {
        return balances[account];
    }
    
    /// @notice Returns the allowance of a spender for an owner
    /// @param owner Address of the token owner
    /// @param spender Address of the spender
    /// @return Allowance amount
    function allowance(address owner, address spender) external view returns (uint256) {
        return allowances[owner][spender];
    }
    
    /// @notice Transfers tokens from caller to recipient
    /// @param to Recipient address
    /// @param amount Amount to transfer
    /// @return Success status
    function transfer(address to, uint256 amount) external whenNotPaused whenTransfersEnabled returns (bool) {
        require(to != address(0), "GovernanceToken: transfer to zero address");
        require(amount > 0, "GovernanceToken: transfer amount must be positive");
        require(balances[msg.sender] >= amount, "GovernanceToken: insufficient balance");
        
        _transfer(msg.sender, to, amount);
        return true;
    }
    
    /// @notice Transfers tokens from sender to recipient (with allowance)
    /// @param from Sender address
    /// @param to Recipient address
    /// @param amount Amount to transfer
    /// @return Success status
    function transferFrom(address from, address to, uint256 amount) external whenNotPaused whenTransfersEnabled returns (bool) {
        require(to != address(0), "GovernanceToken: transfer to zero address");
        require(amount > 0, "GovernanceToken: transfer amount must be positive");
        require(balances[from] >= amount, "GovernanceToken: insufficient balance");
        require(allowances[from][msg.sender] >= amount, "GovernanceToken: insufficient allowance");
        
        allowances[from][msg.sender] -= amount;
        _transfer(from, to, amount);
        return true;
    }
    
    /// @notice Approves spender to spend tokens
    /// @param spender Address to approve
    /// @param amount Amount to approve
    /// @return Success status
    function approve(address spender, uint256 amount) external whenNotPaused returns (bool) {
        require(spender != address(0), "GovernanceToken: approve to zero address");
        
        allowances[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }
    
    /// @notice Internal transfer function with overflow protection
    /// @param from Sender address
    /// @param to Recipient address
    /// @param amount Amount to transfer
    function _transfer(address from, address to, uint256 amount) internal {
        require(balances[from] >= amount, "GovernanceToken: insufficient balance");
        require(balances[to] + amount >= balances[to], "GovernanceToken: overflow protection");
        
        balances[from] -= amount;
        balances[to] += amount;
        
        emit Transfer(from, to, amount);
    }
    
    // ============ MINTING FUNCTIONS ============
    
    /// @notice Mints new tokens (owner only)
    /// @param to Address to mint tokens to
    /// @param amount Amount to mint
    function mint(address to, uint256 amount) external onlyOwner whenNotPaused {
        require(to != address(0), "GovernanceToken: mint to zero address");
        require(amount > 0, "GovernanceToken: mint amount must be positive");
        require(totalSupply + amount <= MAX_SUPPLY, "GovernanceToken: would exceed maximum supply");
        
        totalSupply += amount;
        balances[to] += amount;
        
        emit Transfer(address(0), to, amount);
    }
    
    /// @notice Burns tokens from caller's balance
    /// @param amount Amount to burn
    function burn(uint256 amount) external whenNotPaused {
        require(amount > 0, "GovernanceToken: burn amount must be positive");
        require(balances[msg.sender] >= amount, "GovernanceToken: insufficient balance");
        
        totalSupply -= amount;
        balances[msg.sender] -= amount;
        
        emit Transfer(msg.sender, address(0), amount);
    }
    
    /// @notice Burns tokens from specified address (with allowance)
    /// @param from Address to burn tokens from
    /// @param amount Amount to burn
    function burnFrom(address from, uint256 amount) external whenNotPaused {
        require(amount > 0, "GovernanceToken: burn amount must be positive");
        require(balances[from] >= amount, "GovernanceToken: insufficient balance");
        require(allowances[from][msg.sender] >= amount, "GovernanceToken: insufficient allowance");
        
        allowances[from][msg.sender] -= amount;
        totalSupply -= amount;
        balances[from] -= amount;
        
        emit Transfer(from, address(0), amount);
    }
    
    // ============ STAKING FUNCTIONS ============
    
    /// @notice Stakes tokens for PoS validator participation
    /// @param amount Amount to stake
    /// @param lockPeriod Lock period in seconds
    function stake(uint256 amount, uint256 lockPeriod) external whenNotPaused whenStakingEnabled nonReentrant {
        require(amount >= MIN_STAKE, "GovernanceToken: stake amount below minimum");
        require(lockPeriod <= MAX_STAKING_PERIOD, "GovernanceToken: lock period too long");
        require(balances[msg.sender] >= amount, "GovernanceToken: insufficient balance");
        require(stakedBalances[msg.sender] == 0, "GovernanceToken: already staked");
        
        // Transfer tokens to contract
        balances[msg.sender] -= amount;
        stakedBalances[msg.sender] = amount;
        
        // Pack staking info into single storage slot for gas efficiency
        uint256 packedInfo = (lockPeriod << 224) | (block.timestamp & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF);
        stakingInfo[msg.sender] = packedInfo;
        
        totalStaked += amount;
        
        // Update voting weight
        _updateVotingWeight(msg.sender, amount, true);
        
        emit Staked(msg.sender, amount, lockPeriod);
    }
    
    /// @notice Unstakes tokens after lock period
    /// @param amount Amount to unstake
    function unstake(uint256 amount) external whenNotPaused nonReentrant {
        require(amount > 0, "GovernanceToken: unstake amount must be positive");
        require(stakedBalances[msg.sender] >= amount, "GovernanceToken: insufficient staked balance");
        
        // Unpack staking info
        (uint256 lockPeriod, uint256 startTime) = _unpackStakingInfo(msg.sender);
        require(block.timestamp >= startTime + lockPeriod, "GovernanceToken: lock period not expired");
        
        stakedBalances[msg.sender] -= amount;
        balances[msg.sender] += amount;
        totalStaked -= amount;
        
        // Update voting weight
        _updateVotingWeight(msg.sender, amount, false);
        
        emit Unstaked(msg.sender, amount);
    }
    
    /// @notice Emergency unstake (with penalty)
    /// @param amount Amount to unstake
    function emergencyUnstake(uint256 amount) external whenNotPaused nonReentrant {
        require(amount > 0, "GovernanceToken: unstake amount must be positive");
        require(stakedBalances[msg.sender] >= amount, "GovernanceToken: insufficient staked balance");
        
        // Calculate penalty (50% of staked amount)
        uint256 penalty = amount / 2;
        uint256 actualAmount = amount - penalty;
        
        stakedBalances[msg.sender] -= amount;
        balances[msg.sender] += actualAmount;
        totalStaked -= amount;
        
        // Update voting weight
        _updateVotingWeight(msg.sender, amount, false);
        
        emit Unstaked(msg.sender, actualAmount);
        emit Slashed(msg.sender, penalty, "Emergency unstake penalty");
    }
    
    // ============ SLASHING FUNCTIONS ============
    
    /// @notice Slashes staked tokens for malicious behavior (PoS consensus only)
    /// @param user Address to slash
    /// @param amount Amount to slash
    /// @param reason Reason for slashing
    function slash(address user, uint256 amount, string calldata reason) external onlyPoSConsensus whenNotPaused {
        require(user != address(0), "GovernanceToken: slash zero address");
        require(amount > 0, "GovernanceToken: slash amount must be positive");
        require(stakedBalances[user] >= amount, "GovernanceToken: insufficient staked balance");
        
        stakedBalances[user] -= amount;
        totalStaked -= amount;
        
        // Update voting weight
        _updateVotingWeight(user, amount, false);
        
        emit Slashed(user, amount, reason);
    }
    
    /// @notice Slashes percentage of staked tokens
    /// @param user Address to slash
    /// @param percentage Percentage to slash (1-100)
    /// @param reason Reason for slashing
    function slashPercentage(address user, uint256 percentage, string calldata reason) external onlyPoSConsensus whenNotPaused {
        require(user != address(0), "GovernanceToken: slash zero address");
        require(percentage > 0 && percentage <= 100, "GovernanceToken: invalid percentage");
        
        uint256 slashAmount = (stakedBalances[user] * percentage) / 100;
        require(slashAmount > 0, "GovernanceToken: no tokens to slash");
        
        stakedBalances[user] -= slashAmount;
        totalStaked -= slashAmount;
        
        // Update voting weight
        _updateVotingWeight(user, slashAmount, false);
        
        emit Slashed(user, slashAmount, reason);
    }
    
    // ============ GOVERNANCE FUNCTIONS ============
    
    /// @notice Returns voting weight for an address
    /// @param user Address to query voting weight for
    /// @return Voting weight
    function getVotingWeight(address user) external view returns (uint256) {
        return votingWeights[user];
    }
    
    /// @notice Returns total voting weight
    /// @return Total voting weight
    function getTotalVotingWeight() external view returns (uint256) {
        return totalVotingWeight;
    }
    
    /// @notice Updates voting weight for an address
    /// @param user Address to update
    /// @param amount Amount to add/subtract
    /// @param isAddition Whether to add or subtract
    function _updateVotingWeight(address user, uint256 amount, bool isAddition) internal {
        uint256 oldWeight = votingWeights[user];
        uint256 newWeight;
        
        if (isAddition) {
            newWeight = oldWeight + amount;
            totalVotingWeight += amount;
        } else {
            newWeight = oldWeight > amount ? oldWeight - amount : 0;
            totalVotingWeight = totalVotingWeight > amount ? totalVotingWeight - amount : 0;
        }
        
        votingWeights[user] = newWeight;
        emit VotingWeightUpdated(user, oldWeight, newWeight);
    }
    
    /// @notice Packs staking info into single storage slot for gas efficiency
    /// @param lockPeriod Lock period in seconds
    /// @param startTime Staking start time
    /// @return Packed staking info
    function _packStakingInfo(uint256 lockPeriod, uint256 startTime) internal pure returns (uint256) {
        return (lockPeriod << 224) | (startTime & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF);
    }
    
    /// @notice Unpacks staking info from storage slot
    /// @param user Address to unpack info for
    /// @return lockPeriod Lock period in seconds
    /// @return startTime Staking start time
    function _unpackStakingInfo(address user) internal view returns (uint256 lockPeriod, uint256 startTime) {
        uint256 packed = stakingInfo[user];
        lockPeriod = packed >> 224;
        startTime = packed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF;
    }
    
    /// @notice Updates voting weight (voting contract only)
    /// @param user Address to update
    /// @param amount Amount to add/subtract
    /// @param isAddition Whether to add or subtract
    function updateVotingWeight(address user, uint256 amount, bool isAddition) external onlyVotingContract {
        _updateVotingWeight(user, amount, isAddition);
    }
    
    /// @notice Batch transfer function for gas efficiency
    /// @param recipients Array of recipient addresses
    /// @param amounts Array of amounts to transfer
    function batchTransfer(address[] calldata recipients, uint256[] calldata amounts) external whenNotPaused whenTransfersEnabled {
        require(recipients.length == amounts.length, "GovernanceToken: arrays length mismatch");
        require(recipients.length > 0, "GovernanceToken: empty arrays");
        require(recipients.length <= 100, "GovernanceToken: too many recipients");
        
        uint256 totalAmount = 0;
        for (uint256 i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }
        
        require(balances[msg.sender] >= totalAmount, "GovernanceToken: insufficient balance");
        
        for (uint256 i = 0; i < recipients.length; i++) {
            require(recipients[i] != address(0), "GovernanceToken: transfer to zero address");
            require(amounts[i] > 0, "GovernanceToken: transfer amount must be positive");
            
            balances[msg.sender] -= amounts[i];
            balances[recipients[i]] += amounts[i];
            
            emit Transfer(msg.sender, recipients[i], amounts[i]);
        }
    }
    
    /// @notice Batch mint function for gas efficiency
    /// @param recipients Array of recipient addresses
    /// @param amounts Array of amounts to mint
    function batchMint(address[] calldata recipients, uint256[] calldata amounts) external onlyOwner whenNotPaused {
        require(recipients.length == amounts.length, "GovernanceToken: arrays length mismatch");
        require(recipients.length > 0, "GovernanceToken: empty arrays");
        require(recipients.length <= 100, "GovernanceToken: too many recipients");
        
        uint256 totalAmount = 0;
        for (uint256 i = 0; i < amounts.length; i++) {
            totalAmount += amounts[i];
        }
        
        require(totalSupply + totalAmount <= MAX_SUPPLY, "GovernanceToken: would exceed maximum supply");
        
        for (uint256 i = 0; i < recipients.length; i++) {
            require(recipients[i] != address(0), "GovernanceToken: mint to zero address");
            require(amounts[i] > 0, "GovernanceToken: mint amount must be positive");
            
            totalSupply += amounts[i];
            balances[recipients[i]] += amounts[i];
            
            emit Transfer(address(0), recipients[i], amounts[i]);
        }
    }
    
    // ============ ADMIN FUNCTIONS ============
    
    /// @notice Sets PoS consensus module address
    /// @param _posConsensus New PoS consensus address
    function setPoSConsensus(address _posConsensus) external onlyOwner {
        require(_posConsensus != address(0), "GovernanceToken: PoS consensus cannot be zero address");
        
        address oldAddress = posConsensus;
        posConsensus = _posConsensus;
        emit PoSConsensusUpdated(oldAddress, _posConsensus);
    }
    
    /// @notice Sets voting contract address
    /// @param _votingContract New voting contract address
    function setVotingContract(address _votingContract) external onlyOwner {
        require(_votingContract != address(0), "GovernanceToken: voting contract cannot be zero address");
        
        address oldAddress = votingContract;
        votingContract = _votingContract;
        emit VotingContractUpdated(oldAddress, _votingContract);
    }
    
    /// @notice Toggles emergency pause
    function toggleEmergencyPause() external onlyOwner {
        emergencyPaused = !emergencyPaused;
        emit EmergencyPauseToggled(emergencyPaused);
    }
    
    /// @notice Toggles staking functionality
    function toggleStaking() external onlyOwner {
        stakingEnabled = !stakingEnabled;
        emit StakingToggled(stakingEnabled);
    }
    
    /// @notice Toggles transfers
    function toggleTransfers() external onlyOwner {
        transfersEnabled = !transfersEnabled;
        emit TransfersToggled(transfersEnabled);
    }
    
    // ============ VIEW FUNCTIONS ============
    
    /// @notice Returns staking information for an address
    /// @param user Address to query
    /// @return stakedAmount Staked amount
    /// @return lockPeriod Lock period
    /// @return startTime Staking start time
    /// @return canUnstake Whether user can unstake
    function getStakingInfo(address user) external view returns (
        uint256 stakedAmount,
        uint256 lockPeriod,
        uint256 startTime,
        bool canUnstake
    ) {
        stakedAmount = stakedBalances[user];
        (lockPeriod, startTime) = _unpackStakingInfo(user);
        canUnstake = block.timestamp >= startTime + lockPeriod;
    }
    
    /// @notice Returns contract statistics
    /// @return _totalSupply Total token supply
    /// @return _totalStaked Total staked tokens
    /// @return _totalVotingWeight Total voting weight
    /// @return _stakingEnabled Whether staking is enabled
    /// @return _transfersEnabled Whether transfers are enabled
    function getContractStats() external view returns (
        uint256 _totalSupply,
        uint256 _totalStaked,
        uint256 _totalVotingWeight,
        bool _stakingEnabled,
        bool _transfersEnabled
    ) {
        _totalSupply = totalSupply;
        _totalStaked = totalStaked;
        _totalVotingWeight = totalVotingWeight;
        _stakingEnabled = stakingEnabled;
        _transfersEnabled = transfersEnabled;
    }
    
    /// @notice Returns whether an address can stake
    /// @param user Address to check
    /// @return Whether user can stake
    function canStake(address user) external view returns (bool) {
        return stakingEnabled && !emergencyPaused && stakedBalances[user] == 0;
    }
    
    /// @notice Returns whether an address can unstake
    /// @param user Address to check
    /// @return Whether user can unstake
    function canUnstake(address user) external view returns (bool) {
        if (stakedBalances[user] == 0) return false;
        (uint256 lockPeriod, uint256 startTime) = _unpackStakingInfo(user);
        return block.timestamp >= startTime + lockPeriod;
    }
}
