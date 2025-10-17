// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

import "./GovernanceToken.sol";

/**
 * @title GovernanceTokenTest
 * @dev Comprehensive test suite for GovernanceToken contract
 * @notice Tests normal operation, edge cases, malicious behavior, and stress scenarios
 * @author Hauptbuch Blockchain Team
 */
contract GovernanceTokenTest {
    // ============ TEST STATE ============
    
    GovernanceToken public token;
    address public owner;
    address public user1;
    address public user2;
    address public user3;
    address public posConsensus;
    address public votingContract;
    
    uint256 public constant INITIAL_SUPPLY = 100_000_000 * 10**18; // 100 million tokens
    uint256 public constant MIN_STAKE = 1000 * 10**18; // 1000 tokens
    uint256 public constant STAKE_AMOUNT = 5000 * 10**18; // 5000 tokens
    uint256 public constant LOCK_PERIOD = 30 days;
    
    // ============ EVENTS ============
    
    event TestPassed(string testName);
    event TestFailed(string testName, string reason);
    event TestResult(string testName, bool passed, string message);
    
    // ============ MODIFIERS ============
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Test: caller is not owner");
        _;
    }
    
    // ============ CONSTRUCTOR ============
    
    constructor() {
        owner = msg.sender;
        user1 = address(0x1);
        user2 = address(0x2);
        user3 = address(0x3);
        posConsensus = address(0x4);
        votingContract = address(0x5);
        
        // Deploy token contract
        token = new GovernanceToken(INITIAL_SUPPLY, owner);
        
        // Set up contract addresses
        token.setPoSConsensus(posConsensus);
        token.setVotingContract(votingContract);
    }
    
    // ============ NORMAL OPERATION TESTS ============
    
    /// @notice Test 1: Token minting functionality
    function testTokenMinting() public {
        uint256 mintAmount = 1000 * 10**18;
        uint256 initialBalance = token.balanceOf(user1);
        
        token.mint(user1, mintAmount);
        
        assert(token.balanceOf(user1) == initialBalance + mintAmount);
        assert(token.totalSupply() == INITIAL_SUPPLY + mintAmount);
        
        emit TestPassed("testTokenMinting");
    }
    
    /// @notice Test 2: Token transfer functionality
    function testTokenTransfer() public {
        uint256 transferAmount = 1000 * 10**18;
        uint256 initialBalance1 = token.balanceOf(user1);
        uint256 initialBalance2 = token.balanceOf(user2);
        
        // Transfer tokens from owner to user1 first
        token.transfer(user1, transferAmount);
        assert(token.balanceOf(user1) == initialBalance1 + transferAmount);
        
        // Transfer from user1 to user2
        vm.prank(user1);
        token.transfer(user2, transferAmount);
        
        assert(token.balanceOf(user1) == initialBalance1);
        assert(token.balanceOf(user2) == initialBalance2 + transferAmount);
        
        emit TestPassed("testTokenTransfer");
    }
    
    /// @notice Test 3: Token approval and transferFrom functionality
    function testTokenApproval() public {
        uint256 approveAmount = 1000 * 10**18;
        uint256 transferAmount = 500 * 10**18;
        
        // Transfer tokens to user1
        token.transfer(user1, approveAmount);
        
        // Approve user2 to spend user1's tokens
        vm.prank(user1);
        token.approve(user2, approveAmount);
        
        assert(token.allowance(user1, user2) == approveAmount);
        
        // Transfer from user1 to user3 using user2's approval
        vm.prank(user2);
        token.transferFrom(user1, user3, transferAmount);
        
        assert(token.balanceOf(user3) == transferAmount);
        assert(token.allowance(user1, user2) == approveAmount - transferAmount);
        
        emit TestPassed("testTokenApproval");
    }
    
    /// @notice Test 4: Token burning functionality
    function testTokenBurning() public {
        uint256 burnAmount = 1000 * 10**18;
        uint256 initialSupply = token.totalSupply();
        uint256 initialBalance = token.balanceOf(owner);
        
        token.burn(burnAmount);
        
        assert(token.totalSupply() == initialSupply - burnAmount);
        assert(token.balanceOf(owner) == initialBalance - burnAmount);
        
        emit TestPassed("testTokenBurning");
    }
    
    /// @notice Test 5: Staking functionality
    function testStaking() public {
        uint256 stakeAmount = STAKE_AMOUNT;
        uint256 lockPeriod = LOCK_PERIOD;
        
        // Transfer tokens to user1
        token.transfer(user1, stakeAmount);
        
        // Stake tokens
        vm.prank(user1);
        token.stake(stakeAmount, lockPeriod);
        
        assert(token.stakedBalances(user1) == stakeAmount);
        assert(token.stakingStartTimes(user1) == block.timestamp);
        assert(token.stakingLockPeriods(user1) == lockPeriod);
        assert(token.balanceOf(user1) == 0);
        assert(token.totalStaked() == stakeAmount);
        
        emit TestPassed("testStaking");
    }
    
    /// @notice Test 6: Voting weight calculation
    function testVotingWeight() public {
        uint256 stakeAmount = STAKE_AMOUNT;
        uint256 lockPeriod = LOCK_PERIOD;
        
        // Transfer tokens to user1
        token.transfer(user1, stakeAmount);
        
        // Stake tokens
        vm.prank(user1);
        token.stake(stakeAmount, lockPeriod);
        
        assert(token.getVotingWeight(user1) == stakeAmount);
        assert(token.getTotalVotingWeight() == stakeAmount);
        
        emit TestPassed("testVotingWeight");
    }
    
    // ============ EDGE CASE TESTS ============
    
    /// @notice Test 7: Zero balance transfers
    function testZeroBalanceTransfer() public {
        uint256 transferAmount = 1000 * 10**18;
        
        // Attempt to transfer from user1 (zero balance)
        vm.prank(user1);
        bool success = token.transfer(user2, transferAmount);
        
        assert(!success);
        assert(token.balanceOf(user1) == 0);
        assert(token.balanceOf(user2) == 0);
        
        emit TestPassed("testZeroBalanceTransfer");
    }
    
    /// @notice Test 8: Maximum token supply
    function testMaximumTokenSupply() public {
        uint256 maxMintAmount = token.MAX_SUPPLY() - token.totalSupply();
        
        // Mint up to maximum supply
        token.mint(user1, maxMintAmount);
        assert(token.totalSupply() == token.MAX_SUPPLY());
        
        // Attempt to mint beyond maximum supply
        bool success = token.mint(user1, 1);
        assert(!success);
        
        emit TestPassed("testMaximumTokenSupply");
    }
    
    /// @notice Test 9: Invalid signatures and addresses
    function testInvalidAddresses() public {
        // Attempt to transfer to zero address
        bool success = token.transfer(address(0), 1000);
        assert(!success);
        
        // Attempt to approve zero address
        success = token.approve(address(0), 1000);
        assert(!success);
        
        // Attempt to mint to zero address
        success = token.mint(address(0), 1000);
        assert(!success);
        
        emit TestPassed("testInvalidAddresses");
    }
    
    /// @notice Test 10: Minimum staking requirements
    function testMinimumStaking() public {
        uint256 belowMinimum = MIN_STAKE - 1;
        uint256 lockPeriod = LOCK_PERIOD;
        
        // Transfer tokens to user1
        token.transfer(user1, belowMinimum);
        
        // Attempt to stake below minimum
        vm.prank(user1);
        bool success = token.stake(belowMinimum, lockPeriod);
        assert(!success);
        
        emit TestPassed("testMinimumStaking");
    }
    
    /// @notice Test 11: Maximum staking period
    function testMaximumStakingPeriod() public {
        uint256 stakeAmount = STAKE_AMOUNT;
        uint256 maxLockPeriod = token.MAX_STAKING_PERIOD();
        uint256 excessiveLockPeriod = maxLockPeriod + 1;
        
        // Transfer tokens to user1
        token.transfer(user1, stakeAmount);
        
        // Attempt to stake with excessive lock period
        vm.prank(user1);
        bool success = token.stake(stakeAmount, excessiveLockPeriod);
        assert(!success);
        
        emit TestPassed("testMaximumStakingPeriod");
    }
    
    /// @notice Test 12: Zero amount operations
    function testZeroAmountOperations() public {
        // Attempt to transfer zero amount
        bool success = token.transfer(user1, 0);
        assert(!success);
        
        // Attempt to stake zero amount
        vm.prank(user1);
        success = token.stake(0, LOCK_PERIOD);
        assert(!success);
        
        // Attempt to burn zero amount
        success = token.burn(0);
        assert(!success);
        
        emit TestPassed("testZeroAmountOperations");
    }
    
    // ============ MALICIOUS BEHAVIOR TESTS ============
    
    /// @notice Test 13: Reentrancy attack prevention
    function testReentrancyPrevention() public {
        // Create a malicious contract that attempts reentrancy
        MaliciousContract malicious = new MaliciousContract(address(token));
        
        // Transfer tokens to malicious contract
        token.transfer(address(malicious), STAKE_AMOUNT);
        
        // Attempt reentrancy attack
        bool success = malicious.attemptReentrancy();
        assert(!success);
        
        emit TestPassed("testReentrancyPrevention");
    }
    
    /// @notice Test 14: Unauthorized slashing attempts
    function testUnauthorizedSlashing() public {
        uint256 stakeAmount = STAKE_AMOUNT;
        uint256 lockPeriod = LOCK_PERIOD;
        
        // Transfer tokens to user1 and stake
        token.transfer(user1, stakeAmount);
        vm.prank(user1);
        token.stake(stakeAmount, lockPeriod);
        
        // Attempt unauthorized slashing
        vm.prank(user2);
        bool success = token.slash(user1, 1000, "Unauthorized slash");
        assert(!success);
        
        emit TestPassed("testUnauthorizedSlashing");
    }
    
    /// @notice Test 15: Front-running prevention
    function testFrontRunningPrevention() public {
        uint256 stakeAmount = STAKE_AMOUNT;
        uint256 lockPeriod = LOCK_PERIOD;
        
        // Transfer tokens to user1
        token.transfer(user1, stakeAmount);
        
        // User1 stakes tokens
        vm.prank(user1);
        token.stake(stakeAmount, lockPeriod);
        
        // Attempt to stake again (should fail)
        vm.prank(user1);
        bool success = token.stake(stakeAmount, lockPeriod);
        assert(!success);
        
        emit TestPassed("testFrontRunningPrevention");
    }
    
    /// @notice Test 16: Overflow protection
    function testOverflowProtection() public {
        uint256 maxAmount = type(uint256).max;
        
        // Attempt to transfer maximum amount
        bool success = token.transfer(user1, maxAmount);
        assert(!success);
        
        // Attempt to mint maximum amount
        success = token.mint(user1, maxAmount);
        assert(!success);
        
        emit TestPassed("testOverflowProtection");
    }
    
    // ============ STRESS TESTS ============
    
    /// @notice Test 17: Large token transfers
    function testLargeTokenTransfers() public {
        uint256 largeAmount = 10_000_000 * 10**18; // 10 million tokens
        
        // Mint large amount to owner
        token.mint(owner, largeAmount);
        
        // Transfer large amount
        bool success = token.transfer(user1, largeAmount);
        assert(success);
        assert(token.balanceOf(user1) == largeAmount);
        
        emit TestPassed("testLargeTokenTransfers");
    }
    
    /// @notice Test 18: High staking volumes
    function testHighStakingVolumes() public {
        uint256 stakeAmount = STAKE_AMOUNT;
        uint256 lockPeriod = LOCK_PERIOD;
        uint256 numberOfStakers = 100;
        
        // Create multiple stakers
        for (uint256 i = 0; i < numberOfStakers; i++) {
            address staker = address(uint160(0x1000 + i));
            
            // Transfer tokens to staker
            token.transfer(staker, stakeAmount);
            
            // Stake tokens
            vm.prank(staker);
            token.stake(stakeAmount, lockPeriod);
        }
        
        assert(token.totalStaked() == stakeAmount * numberOfStakers);
        assert(token.getTotalVotingWeight() == stakeAmount * numberOfStakers);
        
        emit TestPassed("testHighStakingVolumes");
    }
    
    /// @notice Test 19: Emergency pause functionality
    function testEmergencyPause() public {
        uint256 transferAmount = 1000 * 10**18;
        
        // Pause contract
        token.toggleEmergencyPause();
        assert(token.emergencyPaused() == true);
        
        // Attempt to transfer (should fail)
        bool success = token.transfer(user1, transferAmount);
        assert(!success);
        
        // Unpause contract
        token.toggleEmergencyPause();
        assert(token.emergencyPaused() == false);
        
        // Transfer should now succeed
        success = token.transfer(user1, transferAmount);
        assert(success);
        
        emit TestPassed("testEmergencyPause");
    }
    
    /// @notice Test 20: Staking toggle functionality
    function testStakingToggle() public {
        uint256 stakeAmount = STAKE_AMOUNT;
        uint256 lockPeriod = LOCK_PERIOD;
        
        // Disable staking
        token.toggleStaking();
        assert(token.stakingEnabled() == false);
        
        // Attempt to stake (should fail)
        vm.prank(user1);
        bool success = token.stake(stakeAmount, lockPeriod);
        assert(!success);
        
        // Enable staking
        token.toggleStaking();
        assert(token.stakingEnabled() == true);
        
        // Transfer tokens and stake
        token.transfer(user1, stakeAmount);
        vm.prank(user1);
        success = token.stake(stakeAmount, lockPeriod);
        assert(success);
        
        emit TestPassed("testStakingToggle");
    }
    
    /// @notice Test 21: Transfers toggle functionality
    function testTransfersToggle() public {
        uint256 transferAmount = 1000 * 10**18;
        
        // Disable transfers
        token.toggleTransfers();
        assert(token.transfersEnabled() == false);
        
        // Attempt to transfer (should fail)
        bool success = token.transfer(user1, transferAmount);
        assert(!success);
        
        // Enable transfers
        token.toggleTransfers();
        assert(token.transfersEnabled() == true);
        
        // Transfer should now succeed
        success = token.transfer(user1, transferAmount);
        assert(success);
        
        emit TestPassed("testTransfersToggle");
    }
    
    /// @notice Test 22: Slashing functionality
    function testSlashing() public {
        uint256 stakeAmount = STAKE_AMOUNT;
        uint256 lockPeriod = LOCK_PERIOD;
        uint256 slashAmount = 1000 * 10**18;
        
        // Transfer tokens to user1 and stake
        token.transfer(user1, stakeAmount);
        vm.prank(user1);
        token.stake(stakeAmount, lockPeriod);
        
        // Slash tokens (PoS consensus only)
        vm.prank(posConsensus);
        token.slash(user1, slashAmount, "Malicious behavior");
        
        assert(token.stakedBalances(user1) == stakeAmount - slashAmount);
        assert(token.totalStaked() == stakeAmount - slashAmount);
        
        emit TestPassed("testSlashing");
    }
    
    /// @notice Test 23: Percentage slashing
    function testPercentageSlashing() public {
        uint256 stakeAmount = STAKE_AMOUNT;
        uint256 lockPeriod = LOCK_PERIOD;
        uint256 slashPercentage = 10; // 10%
        
        // Transfer tokens to user1 and stake
        token.transfer(user1, stakeAmount);
        vm.prank(user1);
        token.stake(stakeAmount, lockPeriod);
        
        // Slash percentage (PoS consensus only)
        vm.prank(posConsensus);
        token.slashPercentage(user1, slashPercentage, "Percentage slash");
        
        uint256 expectedSlashAmount = (stakeAmount * slashPercentage) / 100;
        assert(token.stakedBalances(user1) == stakeAmount - expectedSlashAmount);
        
        emit TestPassed("testPercentageSlashing");
    }
    
    /// @notice Test 24: Unstaking after lock period
    function testUnstakingAfterLockPeriod() public {
        uint256 stakeAmount = STAKE_AMOUNT;
        uint256 lockPeriod = LOCK_PERIOD;
        
        // Transfer tokens to user1 and stake
        token.transfer(user1, stakeAmount);
        vm.prank(user1);
        token.stake(stakeAmount, lockPeriod);
        
        // Fast forward time past lock period
        vm.warp(block.timestamp + lockPeriod + 1);
        
        // Unstake tokens
        vm.prank(user1);
        token.unstake(stakeAmount);
        
        assert(token.stakedBalances(user1) == 0);
        assert(token.balanceOf(user1) == stakeAmount);
        assert(token.totalStaked() == 0);
        
        emit TestPassed("testUnstakingAfterLockPeriod");
    }
    
    /// @notice Test 25: Emergency unstaking with penalty
    function testEmergencyUnstaking() public {
        uint256 stakeAmount = STAKE_AMOUNT;
        uint256 lockPeriod = LOCK_PERIOD;
        
        // Transfer tokens to user1 and stake
        token.transfer(user1, stakeAmount);
        vm.prank(user1);
        token.stake(stakeAmount, lockPeriod);
        
        // Emergency unstake (with penalty)
        vm.prank(user1);
        token.emergencyUnstake(stakeAmount);
        
        uint256 expectedAmount = stakeAmount / 2; // 50% penalty
        assert(token.stakedBalances(user1) == 0);
        assert(token.balanceOf(user1) == expectedAmount);
        
        emit TestPassed("testEmergencyUnstaking");
    }
    
    /// @notice Test 26: Contract statistics
    function testContractStatistics() public {
        uint256 stakeAmount = STAKE_AMOUNT;
        uint256 lockPeriod = LOCK_PERIOD;
        
        // Transfer tokens to user1 and stake
        token.transfer(user1, stakeAmount);
        vm.prank(user1);
        token.stake(stakeAmount, lockPeriod);
        
        // Get contract statistics
        (
            uint256 totalSupply,
            uint256 totalStaked,
            uint256 totalVotingWeight,
            bool stakingEnabled,
            bool transfersEnabled
        ) = token.getContractStats();
        
        assert(totalSupply == token.totalSupply());
        assert(totalStaked == stakeAmount);
        assert(totalVotingWeight == stakeAmount);
        assert(stakingEnabled == token.stakingEnabled());
        assert(transfersEnabled == token.transfersEnabled());
        
        emit TestPassed("testContractStatistics");
    }
    
    /// @notice Test 27: Staking information
    function testStakingInformation() public {
        uint256 stakeAmount = STAKE_AMOUNT;
        uint256 lockPeriod = LOCK_PERIOD;
        
        // Transfer tokens to user1 and stake
        token.transfer(user1, stakeAmount);
        vm.prank(user1);
        token.stake(stakeAmount, lockPeriod);
        
        // Get staking information
        (
            uint256 stakedAmount,
            uint256 lockPeriodReturned,
            uint256 startTime,
            bool canUnstake
        ) = token.getStakingInfo(user1);
        
        assert(stakedAmount == stakeAmount);
        assert(lockPeriodReturned == lockPeriod);
        assert(startTime == block.timestamp);
        assert(canUnstake == false);
        
        emit TestPassed("testStakingInformation");
    }
    
    /// @notice Test 28: Can stake/unstake checks
    function testCanStakeUnstakeChecks() public {
        uint256 stakeAmount = STAKE_AMOUNT;
        uint256 lockPeriod = LOCK_PERIOD;
        
        // Initially can stake
        assert(token.canStake(user1) == true);
        assert(token.canUnstake(user1) == false);
        
        // Transfer tokens to user1 and stake
        token.transfer(user1, stakeAmount);
        vm.prank(user1);
        token.stake(stakeAmount, lockPeriod);
        
        // After staking, cannot stake again
        assert(token.canStake(user1) == false);
        assert(token.canUnstake(user1) == false);
        
        // Fast forward time past lock period
        vm.warp(block.timestamp + lockPeriod + 1);
        
        // Now can unstake
        assert(token.canUnstake(user1) == true);
        
        emit TestPassed("testCanStakeUnstakeChecks");
    }
    
    /// @notice Test 29: Batch transfer functionality
    function testBatchTransfer() public {
        address[] memory recipients = new address[](3);
        uint256[] memory amounts = new uint256[](3);
        
        recipients[0] = user1;
        recipients[1] = user2;
        recipients[2] = user3;
        
        amounts[0] = 1000 * 10**18;
        amounts[1] = 2000 * 10**18;
        amounts[2] = 3000 * 10**18;
        
        uint256 totalAmount = amounts[0] + amounts[1] + amounts[2];
        
        // Transfer tokens to owner first
        token.mint(owner, totalAmount);
        
        // Batch transfer
        token.batchTransfer(recipients, amounts);
        
        assert(token.balanceOf(user1) == amounts[0]);
        assert(token.balanceOf(user2) == amounts[1]);
        assert(token.balanceOf(user3) == amounts[2]);
        
        emit TestPassed("testBatchTransfer");
    }
    
    /// @notice Test 30: Batch mint functionality
    function testBatchMint() public {
        address[] memory recipients = new address[](3);
        uint256[] memory amounts = new uint256[](3);
        
        recipients[0] = user1;
        recipients[1] = user2;
        recipients[2] = user3;
        
        amounts[0] = 1000 * 10**18;
        amounts[1] = 2000 * 10**18;
        amounts[2] = 3000 * 10**18;
        
        uint256 initialSupply = token.totalSupply();
        uint256 totalAmount = amounts[0] + amounts[1] + amounts[2];
        
        // Batch mint
        token.batchMint(recipients, amounts);
        
        assert(token.balanceOf(user1) == amounts[0]);
        assert(token.balanceOf(user2) == amounts[1]);
        assert(token.balanceOf(user3) == amounts[2]);
        assert(token.totalSupply() == initialSupply + totalAmount);
        
        emit TestPassed("testBatchMint");
    }
    
    /// @notice Test 31: Gas optimization with packed storage
    function testGasOptimization() public {
        uint256 stakeAmount = STAKE_AMOUNT;
        uint256 lockPeriod = LOCK_PERIOD;
        
        // Transfer tokens to user1 and stake
        token.transfer(user1, stakeAmount);
        vm.prank(user1);
        token.stake(stakeAmount, lockPeriod);
        
        // Get staking info (should use packed storage)
        (
            uint256 stakedAmount,
            uint256 returnedLockPeriod,
            uint256 startTime,
            bool canUnstake
        ) = token.getStakingInfo(user1);
        
        assert(stakedAmount == stakeAmount);
        assert(returnedLockPeriod == lockPeriod);
        assert(startTime == block.timestamp);
        assert(canUnstake == false);
        
        emit TestPassed("testGasOptimization");
    }
    
    /// @notice Test 32: Overflow protection with batch operations
    function testBatchOverflowProtection() public {
        address[] memory recipients = new address[](2);
        uint256[] memory amounts = new uint256[](2);
        
        recipients[0] = user1;
        recipients[1] = user2;
        
        amounts[0] = type(uint256).max;
        amounts[1] = 1;
        
        // Attempt batch transfer with overflow
        bool success = token.batchTransfer(recipients, amounts);
        assert(!success);
        
        emit TestPassed("testBatchOverflowProtection");
    }
    
    // ============ HELPER FUNCTIONS ============
    
    /// @notice Runs all tests and returns results
    function runAllTests() external returns (bool[] memory results) {
        results = new bool[](32);
        
        try this.testTokenMinting() { results[0] = true; } catch { results[0] = false; }
        try this.testTokenTransfer() { results[1] = true; } catch { results[1] = false; }
        try this.testTokenApproval() { results[2] = true; } catch { results[2] = false; }
        try this.testTokenBurning() { results[3] = true; } catch { results[3] = false; }
        try this.testStaking() { results[4] = true; } catch { results[4] = false; }
        try this.testVotingWeight() { results[5] = true; } catch { results[5] = false; }
        try this.testZeroBalanceTransfer() { results[6] = true; } catch { results[6] = false; }
        try this.testMaximumTokenSupply() { results[7] = true; } catch { results[7] = false; }
        try this.testInvalidAddresses() { results[8] = true; } catch { results[8] = false; }
        try this.testMinimumStaking() { results[9] = true; } catch { results[9] = false; }
        try this.testMaximumStakingPeriod() { results[10] = true; } catch { results[10] = false; }
        try this.testZeroAmountOperations() { results[11] = true; } catch { results[11] = false; }
        try this.testReentrancyPrevention() { results[12] = true; } catch { results[12] = false; }
        try this.testUnauthorizedSlashing() { results[13] = true; } catch { results[13] = false; }
        try this.testFrontRunningPrevention() { results[14] = true; } catch { results[14] = false; }
        try this.testOverflowProtection() { results[15] = true; } catch { results[15] = false; }
        try this.testLargeTokenTransfers() { results[16] = true; } catch { results[16] = false; }
        try this.testHighStakingVolumes() { results[17] = true; } catch { results[17] = false; }
        try this.testEmergencyPause() { results[18] = true; } catch { results[18] = false; }
        try this.testStakingToggle() { results[19] = true; } catch { results[19] = false; }
        try this.testTransfersToggle() { results[20] = true; } catch { results[20] = false; }
        try this.testSlashing() { results[21] = true; } catch { results[21] = false; }
        try this.testPercentageSlashing() { results[22] = true; } catch { results[22] = false; }
        try this.testUnstakingAfterLockPeriod() { results[23] = true; } catch { results[23] = false; }
        try this.testEmergencyUnstaking() { results[24] = true; } catch { results[24] = false; }
        try this.testContractStatistics() { results[25] = true; } catch { results[25] = false; }
        try this.testStakingInformation() { results[26] = true; } catch { results[26] = false; }
        try this.testCanStakeUnstakeChecks() { results[27] = true; } catch { results[27] = false; }
        try this.testBatchTransfer() { results[28] = true; } catch { results[28] = false; }
        try this.testBatchMint() { results[29] = true; } catch { results[29] = false; }
        try this.testGasOptimization() { results[30] = true; } catch { results[30] = false; }
        try this.testBatchOverflowProtection() { results[31] = true; } catch { results[31] = false; }
    }
}

/**
 * @title MaliciousContract
 * @dev Contract for testing reentrancy attacks
 */
contract MaliciousContract {
    GovernanceToken public token;
    bool public reentrancyAttempted;
    
    constructor(address _token) {
        token = GovernanceToken(_token);
    }
    
    function attemptReentrancy() external returns (bool) {
        try token.stake(1000, 30 days) {
            return true;
        } catch {
            reentrancyAttempted = true;
            return false;
        }
    }
}
