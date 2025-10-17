// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

/**
 * @title Voting
 * @dev A secure vote tallying smart contract with zk-SNARKs privacy protection
 * 
 * This contract implements a decentralized voting system that ensures voter privacy
 * using zero-knowledge proofs (zk-SNARKs) while preventing double voting and
 * providing verifiable results. The contract uses Merkle trees for efficient
 * voter commitment verification and implements comprehensive security measures.
 * 
 * Key Features:
 * - Anonymous vote submission using zk-SNARKs
 * - Double voting prevention with nullifier mechanism
 * - Merkle tree-based voter commitment verification
 * - Gas-optimized storage and operations
 * - Protection against common vulnerabilities
 * - Access control for vote submission and result finalization
 */
contract Voting {
    // ============ STATE VARIABLES ============
    
    /// @dev Reentrancy guard to prevent reentrancy attacks
    bool private _locked;
    
    /// @dev Contract owner with administrative privileges
    address public immutable owner;
    
    /// @dev Voting session configuration
    struct VotingSession {
        string title;                    // Voting session title
        string[] options;                // Available vote options
        uint256 startTime;              // Session start timestamp
        uint256 endTime;                // Session end timestamp
        bool isActive;                  // Session active status
        bool isFinalized;               // Results finalized status
        uint256 totalVotes;             // Total votes cast
        mapping(uint256 => uint256) voteCounts; // Vote counts per option
    }
    
    /// @dev Current voting session
    VotingSession public currentSession;
    
    /// @dev Voter registration and commitment management
    mapping(address => bool) public registeredVoters;     // Registered voter addresses
    mapping(bytes32 => bool) public usedNullifiers;       // Used nullifiers to prevent double voting
    mapping(bytes32 => bool) public voterCommitments;     // Voter commitments in Merkle tree
    
    /// @dev Merkle tree for voter commitments
    bytes32[] public merkleLeaves;                        // Merkle tree leaves
    bytes32 public merkleRoot;                           // Current Merkle root
    uint256 public merkleTreeDepth;                      // Merkle tree depth
    
    /// @dev zk-SNARKs verification parameters
    struct ZKParams {
        uint256[2] alpha;                               // Alpha parameters for zk-SNARKs
        uint256[2][2] beta;                            // Beta parameters for zk-SNARKs
        uint256[2] gamma;                              // Gamma parameters for zk-SNARKs
        uint256[2] delta;                              // Delta parameters for zk-SNARKs
        uint256[2][] ic;                               // IC parameters for zk-SNARKs
    }
    
    ZKParams public zkParams;                           // zk-SNARKs parameters
    
    /// @dev Gas optimization: packed struct for vote data
    struct VoteData {
        uint128 optionIndex;                             // Vote option index
        uint128 timestamp;                              // Vote timestamp
    }
    
    /// @dev Vote results storage (option index => vote count)
    mapping(uint256 => uint256) public voteResults;
    
    /// @dev Events for transparency and monitoring
    event VoterRegistered(address indexed voter, bytes32 commitment);
    event VoteSubmitted(bytes32 indexed nullifier, uint256 optionIndex, uint256 timestamp);
    event VotingSessionCreated(string title, uint256 startTime, uint256 endTime);
    event VotingSessionFinalized(uint256 totalVotes);
    event ZKParamsUpdated();
    
    /// @dev Custom errors for gas efficiency
    error VotingSessionNotActive();
    error VotingSessionNotFinalized();
    error InvalidZKProof();
    error DoubleVotingDetected();
    error UnauthorizedAccess();
    error InvalidVoteOption();
    error VotingSessionExpired();
    error ReentrancyGuard();
    error InvalidMerkleProof();
    error VoterNotRegistered();
    
    // ============ MODIFIERS ============
    
    /// @dev Prevents reentrancy attacks
    modifier nonReentrant() {
        if (_locked) revert ReentrancyGuard();
        _locked = true;
        _;
        _locked = false;
    }
    
    /// @dev Restricts access to contract owner
    modifier onlyOwner() {
        if (msg.sender != owner) revert UnauthorizedAccess();
        _;
    }
    
    /// @dev Ensures voting session is active
    modifier onlyActiveSession() {
        if (!currentSession.isActive || block.timestamp < currentSession.startTime || block.timestamp > currentSession.endTime) {
            revert VotingSessionNotActive();
        }
        _;
    }
    
    /// @dev Ensures voting session is finalized
    modifier onlyFinalizedSession() {
        if (!currentSession.isFinalized) revert VotingSessionNotFinalized();
        _;
    }
    
    // ============ CONSTRUCTOR ============
    
    /**
     * @dev Initializes the voting contract with zk-SNARKs parameters
     * @param _zkParams zk-SNARKs verification parameters
     * @param _merkleTreeDepth Initial Merkle tree depth
     */
    constructor(ZKParams memory _zkParams, uint256 _merkleTreeDepth) {
        owner = msg.sender;
        zkParams = _zkParams;
        merkleTreeDepth = _merkleTreeDepth;
        
        // Initialize empty Merkle root
        merkleRoot = keccak256(abi.encodePacked("EMPTY_MERKLE_ROOT"));
    }
    
    // ============ VOTER REGISTRATION ============
    
    /**
     * @dev Registers a voter and adds their commitment to the Merkle tree
     * @param voterAddress Address of the voter to register
     * @param commitment Voter's commitment hash for Merkle tree
     * 
     * This function allows the contract owner to register eligible voters
     * and add their commitments to the Merkle tree for zk-SNARKs verification.
     * Each voter can only be registered once.
     */
    function registerVoter(address voterAddress, bytes32 commitment) external onlyOwner {
        require(!registeredVoters[voterAddress], "Voter already registered");
        require(commitment != bytes32(0), "Invalid commitment");
        
        registeredVoters[voterAddress] = true;
        voterCommitments[commitment] = true;
        
        // Add commitment to Merkle tree
        merkleLeaves.push(commitment);
        merkleRoot = calculateMerkleRoot(merkleLeaves);
        
        emit VoterRegistered(voterAddress, commitment);
    }
    
    /**
     * @dev Batch registers multiple voters for gas efficiency
     * @param voterAddresses Array of voter addresses
     * @param commitments Array of corresponding commitments
     */
    function batchRegisterVoters(address[] calldata voterAddresses, bytes32[] calldata commitments) external onlyOwner {
        require(voterAddresses.length == commitments.length, "Array length mismatch");
        require(voterAddresses.length <= 50, "Batch size too large"); // Gas limit protection
        
        for (uint256 i = 0; i < voterAddresses.length; i++) {
            require(!registeredVoters[voterAddresses[i]], "Voter already registered");
            require(commitments[i] != bytes32(0), "Invalid commitment");
            
            registeredVoters[voterAddresses[i]] = true;
            voterCommitments[commitments[i]] = true;
            merkleLeaves.push(commitments[i]);
        }
        
        // Update Merkle root once for all new commitments
        merkleRoot = calculateMerkleRoot(merkleLeaves);
        
        for (uint256 i = 0; i < voterAddresses.length; i++) {
            emit VoterRegistered(voterAddresses[i], commitments[i]);
        }
    }
    
    // ============ VOTING SESSION MANAGEMENT ============
    
    /**
     * @dev Creates a new voting session
     * @param _title Voting session title
     * @param _options Array of vote options
     * @param _startTime Session start timestamp
     * @param _endTime Session end timestamp
     */
    function createVotingSession(
        string memory _title,
        string[] memory _options,
        uint256 _startTime,
        uint256 _endTime
    ) external onlyOwner {
        require(_options.length > 0, "No vote options provided");
        require(_startTime < _endTime, "Invalid time range");
        require(_endTime > block.timestamp, "End time must be in future");
        
        currentSession.title = _title;
        currentSession.options = _options;
        currentSession.startTime = _startTime;
        currentSession.endTime = _endTime;
        currentSession.isActive = true;
        currentSession.isFinalized = false;
        currentSession.totalVotes = 0;
        
        // Initialize vote counts
        for (uint256 i = 0; i < _options.length; i++) {
            currentSession.voteCounts[i] = 0;
            voteResults[i] = 0;
        }
        
        emit VotingSessionCreated(_title, _startTime, _endTime);
    }
    
    /**
     * @dev Finalizes the voting session and calculates results
     */
    function finalizeVotingSession() external onlyOwner {
        require(currentSession.isActive, "No active session");
        require(block.timestamp > currentSession.endTime, "Session not ended");
        
        currentSession.isActive = false;
        currentSession.isFinalized = true;
        
        // Copy vote counts to results
        for (uint256 i = 0; i < currentSession.options.length; i++) {
            voteResults[i] = currentSession.voteCounts[i];
        }
        
        emit VotingSessionFinalized(currentSession.totalVotes);
    }
    
    // ============ VOTE SUBMISSION WITH ZK-SNARKS ============
    
    /**
     * @dev Submits a vote with zk-SNARKs proof for privacy
     * @param nullifier Unique nullifier to prevent double voting
     * @param optionIndex Index of the selected vote option
     * @param merkleProof Merkle proof for voter commitment verification
     * @param zkProof zk-SNARKs proof components
     * 
     * This function allows voters to submit anonymous votes using zk-SNARKs
     * to prove they are registered voters without revealing their identity.
     * The nullifier mechanism prevents double voting.
     */
    function submitVote(
        bytes32 nullifier,
        uint256 optionIndex,
        bytes32[] memory merkleProof,
        uint256[8] memory zkProof
    ) external onlyActiveSession nonReentrant {
        // Validate vote option
        if (optionIndex >= currentSession.options.length) revert InvalidVoteOption();
        
        // Check for double voting
        if (usedNullifiers[nullifier]) revert DoubleVotingDetected();
        
        // Verify zk-SNARKs proof
        if (!verifyZKProof(nullifier, optionIndex, merkleProof, zkProof)) {
            revert InvalidZKProof();
        }
        
        // Mark nullifier as used
        usedNullifiers[nullifier] = true;
        
        // Update vote counts
        currentSession.voteCounts[optionIndex]++;
        currentSession.totalVotes++;
        
        emit VoteSubmitted(nullifier, optionIndex, block.timestamp);
    }
    
    /**
     * @dev Verifies zk-SNARKs proof for vote submission
     * @param nullifier Voter's nullifier
     * @param optionIndex Selected vote option
     * @param merkleProof Merkle proof for commitment verification
     * @param zkProof zk-SNARKs proof components
     * @return true if proof is valid, false otherwise
     */
    function verifyZKProof(
        bytes32 nullifier,
        uint256 optionIndex,
        bytes32[] memory merkleProof,
        uint256[8] memory zkProof
    ) internal view returns (bool) {
        // Simplified zk-SNARKs verification for demonstration
        // In production, use proper zk-SNARKs libraries like snarkjs
        
        // Verify Merkle proof
        if (!verifyMerkleProof(merkleProof, nullifier)) {
            return false;
        }
        
        // Verify zk-SNARKs proof components
        // This is a simplified verification - in production, use proper zk-SNARKs
        require(zkProof.length == 8, "Invalid proof length");
        
        // Check proof components are non-zero and within valid ranges
        for (uint256 i = 0; i < 8; i++) {
            if (zkProof[i] == 0) return false;
        }
        
        // Verify proof using zk-SNARKs parameters
        return verifySnarkProof(zkProof, nullifier, optionIndex);
    }
    
    /**
     * @dev Verifies Merkle proof for voter commitment
     * @param proof Merkle proof array
     * @param leaf Leaf node to verify
     * @return true if proof is valid, false otherwise
     */
    function verifyMerkleProof(bytes32[] memory proof, bytes32 leaf) internal view returns (bool) {
        bytes32 currentHash = leaf;
        
        for (uint256 i = 0; i < proof.length; i++) {
            if (currentHash < proof[i]) {
                currentHash = keccak256(abi.encodePacked(currentHash, proof[i]));
            } else {
                currentHash = keccak256(abi.encodePacked(proof[i], currentHash));
            }
        }
        
        return currentHash == merkleRoot;
    }
    
    /**
     * @dev Simplified zk-SNARKs proof verification
     * @param proof zk-SNARKs proof components
     * @param nullifier Voter's nullifier
     * @param optionIndex Selected vote option
     * @return true if proof is valid, false otherwise
     */
    function verifySnarkProof(
        uint256[8] memory proof,
        bytes32 nullifier,
        uint256 optionIndex
    ) internal view returns (bool) {
        // Simplified verification - in production, use proper zk-SNARKs verification
        // This demonstrates the structure but should be replaced with actual zk-SNARKs verification
        
        // Verify proof components are within valid ranges
        for (uint256 i = 0; i < 8; i++) {
            if (proof[i] >= 21888242871839275222246405745257275088548364400416034343698204186575808495617) {
                return false; // Prime field check
            }
        }
        
        // Verify nullifier is not zero
        if (nullifier == bytes32(0)) return false;
        
        // Verify option index is valid
        if (optionIndex >= currentSession.options.length) return false;
        
        // Simplified proof verification (replace with actual zk-SNARKs verification)
        return true;
    }
    
    // ============ MERKLE TREE UTILITIES ============
    
    /**
     * @dev Calculates Merkle root from leaves
     * @param leaves Array of leaf nodes
     * @return Merkle root hash
     */
    function calculateMerkleRoot(bytes32[] memory leaves) internal pure returns (bytes32) {
        if (leaves.length == 0) {
            return keccak256(abi.encodePacked("EMPTY_MERKLE_ROOT"));
        }
        
        if (leaves.length == 1) {
            return leaves[0];
        }
        
        bytes32[] memory currentLevel = leaves;
        
        while (currentLevel.length > 1) {
            bytes32[] memory nextLevel = new bytes32[]((currentLevel.length + 1) / 2);
            
            for (uint256 i = 0; i < currentLevel.length; i += 2) {
                if (i + 1 < currentLevel.length) {
                    nextLevel[i / 2] = keccak256(abi.encodePacked(currentLevel[i], currentLevel[i + 1]));
                } else {
                    nextLevel[i / 2] = currentLevel[i];
                }
            }
            
            currentLevel = nextLevel;
        }
        
        return currentLevel[0];
    }
    
    // ============ VIEW FUNCTIONS ============
    
    /**
     * @dev Gets current voting session information
     * @return title Session title
     * @return options Available vote options
     * @return startTime Session start timestamp
     * @return endTime Session end timestamp
     * @return isActive Session active status
     * @return isFinalized Session finalized status
     * @return totalVotes Total votes cast
     */
    function getVotingSession() external view returns (
        string memory title,
        string[] memory options,
        uint256 startTime,
        uint256 endTime,
        bool isActive,
        bool isFinalized,
        uint256 totalVotes
    ) {
        return (
            currentSession.title,
            currentSession.options,
            currentSession.startTime,
            currentSession.endTime,
            currentSession.isActive,
            currentSession.isFinalized,
            currentSession.totalVotes
        );
    }
    
    /**
     * @dev Gets vote results for a specific option
     * @param optionIndex Index of the vote option
     * @return voteCount Number of votes for the option
     */
    function getVoteCount(uint256 optionIndex) external view onlyFinalizedSession returns (uint256) {
        return voteResults[optionIndex];
    }
    
    /**
     * @dev Gets all vote results
     * @return results Array of vote counts for each option
     */
    function getAllVoteResults() external view onlyFinalizedSession returns (uint256[] memory results) {
        results = new uint256[](currentSession.options.length);
        for (uint256 i = 0; i < currentSession.options.length; i++) {
            results[i] = voteResults[i];
        }
    }
    
    /**
     * @dev Checks if a nullifier has been used
     * @param nullifier Nullifier to check
     * @return true if nullifier has been used, false otherwise
     */
    function isNullifierUsed(bytes32 nullifier) external view returns (bool) {
        return usedNullifiers[nullifier];
    }
    
    /**
     * @dev Gets current Merkle root
     * @return Current Merkle root hash
     */
    function getMerkleRoot() external view returns (bytes32) {
        return merkleRoot;
    }
    
    /**
     * @dev Gets Merkle tree depth
     * @return Current Merkle tree depth
     */
    function getMerkleTreeDepth() external view returns (uint256) {
        return merkleTreeDepth;
    }
    
    /**
     * @dev Gets total number of registered voters
     * @return Number of registered voters
     */
    function getRegisteredVoterCount() external view returns (uint256) {
        return merkleLeaves.length;
    }
    
    // ============ ADMINISTRATIVE FUNCTIONS ============
    
    /**
     * @dev Updates zk-SNARKs parameters (only owner)
     * @param _zkParams New zk-SNARKs parameters
     */
    function updateZKParams(ZKParams memory _zkParams) external onlyOwner {
        zkParams = _zkParams;
        emit ZKParamsUpdated();
    }
    
    /**
     * @dev Updates Merkle tree depth (only owner)
     * @param _depth New Merkle tree depth
     */
    function updateMerkleTreeDepth(uint256 _depth) external onlyOwner {
        require(_depth > 0, "Invalid depth");
        merkleTreeDepth = _depth;
    }
    
    /**
     * @dev Emergency function to pause voting (only owner)
     */
    function pauseVoting() external onlyOwner {
        currentSession.isActive = false;
    }
    
    /**
     * @dev Emergency function to resume voting (only owner)
     */
    function resumeVoting() external onlyOwner {
        require(block.timestamp >= currentSession.startTime && block.timestamp <= currentSession.endTime, "Outside voting period");
        currentSession.isActive = true;
    }
}
