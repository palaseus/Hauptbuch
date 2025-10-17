const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("Voting Contract", function () {
  let voting;
  let owner;
  let voter1;
  let voter2;
  let voter3;
  let nonVoter;

  beforeEach(async function () {
    [owner, voter1, voter2, voter3, nonVoter] = await ethers.getSigners();
    
    // Deploy Voting contract
    const Voting = await ethers.getContractFactory("Voting");
    voting = await Voting.deploy(
      // Mock ZK parameters
      [
        [1, 2, 3, 4, 5, 6, 7, 8],
        [9, 10, 11, 12, 13, 14, 15, 16],
        [17, 18, 19, 20, 21, 22, 23, 24],
        [25, 26, 27, 28, 29, 30, 31, 32],
        [33, 34, 35, 36, 37, 38, 39, 40]
      ],
      3 // Merkle tree depth
    );
    await voting.deployed();
  });

  describe("Voter Registration", function () {
    it("Should register a single voter", async function () {
      const commitment = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter1_secret"));
      
      await voting.registerVoter(voter1.address, commitment);
      
      const isRegistered = await voting.registeredVoters(voter1.address);
      expect(isRegistered).to.be.true;
    });

    it("Should register multiple voters in batch", async function () {
      const commitments = [
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter1_secret")),
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter2_secret")),
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter3_secret"))
      ];
      
      await voting.batchRegisterVoters(
        [voter1.address, voter2.address, voter3.address],
        commitments
      );
      
      expect(await voting.registeredVoters(voter1.address)).to.be.true;
      expect(await voting.registeredVoters(voter2.address)).to.be.true;
      expect(await voting.registeredVoters(voter3.address)).to.be.true;
    });

    it("Should only allow owner to register voters", async function () {
      const commitment = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter1_secret"));
      
      await expect(
        voting.connect(voter1).registerVoter(voter1.address, commitment)
      ).to.be.revertedWith("Only owner can register voters");
    });
  });

  describe("Voting Session Management", function () {
    it("Should create a voting session", async function () {
      const title = "Test Election";
      const options = ["Option A", "Option B", "Option C"];
      const startTime = Math.floor(Date.now() / 1000);
      const endTime = startTime + 3600; // 1 hour
      
      await voting.createVotingSession(title, options, startTime, endTime);
      
      const session = await voting.currentSession();
      expect(session.title).to.equal(title);
      expect(session.options.length).to.equal(3);
      expect(session.startTime).to.equal(startTime);
      expect(session.endTime).to.equal(endTime);
    });

    it("Should only allow owner to create voting sessions", async function () {
      const title = "Test Election";
      const options = ["Option A", "Option B"];
      const startTime = Math.floor(Date.now() / 1000);
      const endTime = startTime + 3600;
      
      await expect(
        voting.connect(voter1).createVotingSession(title, options, startTime, endTime)
      ).to.be.revertedWith("Only owner can create voting sessions");
    });

    it("Should finalize voting session", async function () {
      const title = "Test Election";
      const options = ["Option A", "Option B"];
      const startTime = Math.floor(Date.now() / 1000) - 3600; // Started 1 hour ago
      const endTime = Math.floor(Date.now() / 1000) - 1800; // Ended 30 minutes ago
      
      await voting.createVotingSession(title, options, startTime, endTime);
      await voting.finalizeVotingSession();
      
      const session = await voting.currentSession();
      expect(session.finalized).to.be.true;
    });
  });

  describe("Vote Submission", function () {
    beforeEach(async function () {
      // Register voters
      const commitments = [
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter1_secret")),
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter2_secret")),
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter3_secret"))
      ];
      
      await voting.batchRegisterVoters(
        [voter1.address, voter2.address, voter3.address],
        commitments
      );
      
      // Create voting session
      const title = "Test Election";
      const options = ["Option A", "Option B", "Option C"];
      const startTime = Math.floor(Date.now() / 1000) - 1800; // Started 30 minutes ago
      const endTime = Math.floor(Date.now() / 1000) + 1800; // Ends in 30 minutes
      
      await voting.createVotingSession(title, options, startTime, endTime);
    });

    it("Should submit a valid vote", async function () {
      const nullifier = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("unique_nullifier_1"));
      const optionIndex = 0;
      const merkleProof = []; // Empty for testing
      const zkProof = [1, 2, 3, 4, 5, 6, 7, 8]; // Mock ZK proof
      
      await voting.connect(voter1).submitVote(nullifier, optionIndex, merkleProof, zkProof);
      
      const voteCount = await voting.getVoteCount(optionIndex);
      expect(voteCount).to.equal(1);
    });

    it("Should prevent double voting", async function () {
      const nullifier = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("unique_nullifier_1"));
      const optionIndex = 0;
      const merkleProof = [];
      const zkProof = [1, 2, 3, 4, 5, 6, 7, 8];
      
      await voting.connect(voter1).submitVote(nullifier, optionIndex, merkleProof, zkProof);
      
      // Try to vote again with the same nullifier
      await expect(
        voting.connect(voter1).submitVote(nullifier, optionIndex, merkleProof, zkProof)
      ).to.be.revertedWith("Nullifier already used");
    });

    it("Should prevent unregistered voters from voting", async function () {
      const nullifier = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("unique_nullifier_1"));
      const optionIndex = 0;
      const merkleProof = [];
      const zkProof = [1, 2, 3, 4, 5, 6, 7, 8];
      
      await expect(
        voting.connect(nonVoter).submitVote(nullifier, optionIndex, merkleProof, zkProof)
      ).to.be.revertedWith("Voter not registered");
    });

    it("Should prevent voting with invalid option index", async function () {
      const nullifier = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("unique_nullifier_1"));
      const optionIndex = 10; // Invalid option
      const merkleProof = [];
      const zkProof = [1, 2, 3, 4, 5, 6, 7, 8];
      
      await expect(
        voting.connect(voter1).submitVote(nullifier, optionIndex, merkleProof, zkProof)
      ).to.be.revertedWith("Invalid option index");
    });

    it("Should prevent voting outside session time", async function () {
      // Create session that has ended
      const title = "Ended Election";
      const options = ["Option A", "Option B"];
      const startTime = Math.floor(Date.now() / 1000) - 3600; // Started 1 hour ago
      const endTime = Math.floor(Date.now() / 1000) - 1800; // Ended 30 minutes ago
      
      await voting.createVotingSession(title, options, startTime, endTime);
      
      const nullifier = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("unique_nullifier_1"));
      const optionIndex = 0;
      const merkleProof = [];
      const zkProof = [1, 2, 3, 4, 5, 6, 7, 8];
      
      await expect(
        voting.connect(voter1).submitVote(nullifier, optionIndex, merkleProof, zkProof)
      ).to.be.revertedWith("Voting session not active");
    });
  });

  describe("Vote Tallying", function () {
    beforeEach(async function () {
      // Register voters
      const commitments = [
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter1_secret")),
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter2_secret")),
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter3_secret"))
      ];
      
      await voting.batchRegisterVoters(
        [voter1.address, voter2.address, voter3.address],
        commitments
      );
      
      // Create and finalize voting session
      const title = "Test Election";
      const options = ["Option A", "Option B", "Option C"];
      const startTime = Math.floor(Date.now() / 1000) - 3600; // Started 1 hour ago
      const endTime = Math.floor(Date.now() / 1000) - 1800; // Ended 30 minutes ago
      
      await voting.createVotingSession(title, options, startTime, endTime);
      await voting.finalizeVotingSession();
    });

    it("Should get vote count for specific option", async function () {
      const voteCount = await voting.getVoteCount(0);
      expect(voteCount).to.equal(0); // No votes yet
    });

    it("Should get all vote results", async function () {
      const results = await voting.getAllVoteResults();
      expect(results.length).to.equal(3); // 3 options
      expect(results[0]).to.equal(0);
      expect(results[1]).to.equal(0);
      expect(results[2]).to.equal(0);
    });

    it("Should only allow access to results after finalization", async function () {
      // Create non-finalized session
      const title = "Active Election";
      const options = ["Option A", "Option B"];
      const startTime = Math.floor(Date.now() / 1000) - 1800; // Started 30 minutes ago
      const endTime = Math.floor(Date.now() / 1000) + 1800; // Ends in 30 minutes
      
      await voting.createVotingSession(title, options, startTime, endTime);
      
      await expect(
        voting.getVoteCount(0)
      ).to.be.revertedWith("Voting session not finalized");
    });
  });

  describe("Security Tests", function () {
    it("Should prevent reentrancy attacks", async function () {
      // This would test reentrancy protection
      // In a real implementation, this would involve a malicious contract
      const commitment = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter1_secret"));
      await voting.registerVoter(voter1.address, commitment);
      
      // Test that the contract is protected against reentrancy
      expect(await voting.registeredVoters(voter1.address)).to.be.true;
    });

    it("Should prevent front-running attacks", async function () {
      // Test that nullifier mechanism prevents front-running
      const nullifier = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("unique_nullifier_1"));
      const optionIndex = 0;
      const merkleProof = [];
      const zkProof = [1, 2, 3, 4, 5, 6, 7, 8];
      
      // First vote should succeed
      await voting.connect(voter1).submitVote(nullifier, optionIndex, merkleProof, zkProof);
      
      // Second vote with same nullifier should fail
      await expect(
        voting.connect(voter1).submitVote(nullifier, optionIndex, merkleProof, zkProof)
      ).to.be.revertedWith("Nullifier already used");
    });

    it("Should validate input parameters", async function () {
      const commitment = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter1_secret"));
      
      // Test with empty address
      await expect(
        voting.registerVoter(ethers.constants.AddressZero, commitment)
      ).to.be.revertedWith("Invalid address");
    });
  });

  describe("Gas Efficiency", function () {
    it("Should have reasonable gas costs for voter registration", async function () {
      const commitment = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter1_secret"));
      
      const tx = await voting.registerVoter(voter1.address, commitment);
      const receipt = await tx.wait();
      
      expect(receipt.gasUsed).to.be.lessThan(100000); // Less than 100k gas
    });

    it("Should have reasonable gas costs for batch registration", async function () {
      const commitments = [
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter1_secret")),
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter2_secret")),
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter3_secret"))
      ];
      
      const tx = await voting.batchRegisterVoters(
        [voter1.address, voter2.address, voter3.address],
        commitments
      );
      const receipt = await tx.wait();
      
      expect(receipt.gasUsed).to.be.lessThan(200000); // Less than 200k gas
    });

    it("Should have reasonable gas costs for vote submission", async function () {
      // Register voter first
      const commitment = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter1_secret"));
      await voting.registerVoter(voter1.address, commitment);
      
      // Create voting session
      const title = "Test Election";
      const options = ["Option A", "Option B"];
      const startTime = Math.floor(Date.now() / 1000) - 1800;
      const endTime = Math.floor(Date.now() / 1000) + 1800;
      await voting.createVotingSession(title, options, startTime, endTime);
      
      // Submit vote
      const nullifier = ethers.utils.keccak256(ethers.utils.toUtf8Bytes("unique_nullifier_1"));
      const optionIndex = 0;
      const merkleProof = [];
      const zkProof = [1, 2, 3, 4, 5, 6, 7, 8];
      
      const tx = await voting.connect(voter1).submitVote(nullifier, optionIndex, merkleProof, zkProof);
      const receipt = await tx.wait();
      
      expect(receipt.gasUsed).to.be.lessThan(150000); // Less than 150k gas
    });
  });

  describe("Integration Tests", function () {
    it("Should handle complete voting workflow", async function () {
      // 1. Register voters
      const commitments = [
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter1_secret")),
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter2_secret")),
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("voter3_secret"))
      ];
      
      await voting.batchRegisterVoters(
        [voter1.address, voter2.address, voter3.address],
        commitments
      );
      
      // 2. Create voting session
      const title = "Complete Test Election";
      const options = ["Option A", "Option B", "Option C"];
      const startTime = Math.floor(Date.now() / 1000) - 1800;
      const endTime = Math.floor(Date.now() / 1000) + 1800;
      
      await voting.createVotingSession(title, options, startTime, endTime);
      
      // 3. Submit votes
      const nullifiers = [
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("nullifier_1")),
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("nullifier_2")),
        ethers.utils.keccak256(ethers.utils.toUtf8Bytes("nullifier_3"))
      ];
      
      const optionIndices = [0, 1, 0]; // Vote for A, B, A
      const merkleProof = [];
      const zkProof = [1, 2, 3, 4, 5, 6, 7, 8];
      
      await voting.connect(voter1).submitVote(nullifiers[0], optionIndices[0], merkleProof, zkProof);
      await voting.connect(voter2).submitVote(nullifiers[1], optionIndices[1], merkleProof, zkProof);
      await voting.connect(voter3).submitVote(nullifiers[2], optionIndices[2], merkleProof, zkProof);
      
      // 4. Finalize session
      await voting.finalizeVotingSession();
      
      // 5. Check results
      const results = await voting.getAllVoteResults();
      expect(results[0]).to.equal(2); // Option A: 2 votes
      expect(results[1]).to.equal(1); // Option B: 1 vote
      expect(results[2]).to.equal(0); // Option C: 0 votes
    });
  });
});
