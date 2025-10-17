#!/usr/bin/env python3
"""
Hauptbuch Core RPC Tests
Tests for all RPC methods from the API documentation.
"""

import asyncio
import pytest
import json
from typing import Dict, Any
from hauptbuch_client import HauptbuchClient

class TestCoreRPC:
    """Test suite for core RPC methods"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_get_network_info(self, client):
        """Test hauptbuch_getNetworkInfo RPC method"""
        network_info = await client.get_network_info()
        
        assert network_info.chain_id is not None
        assert network_info.network_id is not None
        assert network_info.node_version is not None
        assert network_info.protocol_version is not None
        assert network_info.genesis_hash is not None
        assert network_info.latest_block is not None
        assert network_info.latest_block_number is not None
        
        # Verify data types
        assert isinstance(network_info.chain_id, str)
        assert isinstance(network_info.network_id, str)
        assert isinstance(network_info.node_version, str)
        assert isinstance(network_info.protocol_version, str)
        assert network_info.genesis_hash.startswith("0x")
        assert network_info.latest_block.startswith("0x")
        assert network_info.latest_block_number.startswith("0x")
    
    @pytest.mark.asyncio
    async def test_get_node_status(self, client):
        """Test hauptbuch_getNodeStatus RPC method"""
        node_status = await client.get_node_status()
        
        assert node_status.status is not None
        assert node_status.sync_status is not None
        assert node_status.peer_count is not None
        assert node_status.uptime is not None
        assert node_status.memory_usage is not None
        assert node_status.cpu_usage is not None
        
        # Verify data types
        assert isinstance(node_status.status, str)
        assert isinstance(node_status.sync_status, dict)
        assert isinstance(node_status.peer_count, int)
        assert isinstance(node_status.uptime, int)
        assert isinstance(node_status.memory_usage, dict)
        assert isinstance(node_status.cpu_usage, float)
        
        # Verify status values
        assert node_status.status in ["healthy", "synced", "syncing", "error"]
        assert node_status.peer_count >= 0
        assert node_status.uptime >= 0
        assert 0 <= node_status.cpu_usage <= 100
    
    @pytest.mark.asyncio
    async def test_get_chain_info(self, client):
        """Test hauptbuch_getChainInfo RPC method"""
        chain_info = await client.get_chain_info()
        
        assert chain_info.block_height is not None
        assert chain_info.total_transactions is not None
        assert chain_info.total_gas_used is not None
        assert chain_info.average_block_time is not None
        assert chain_info.difficulty is not None
        assert chain_info.total_supply is not None
        
        # Verify data types
        assert isinstance(chain_info.block_height, str)
        assert isinstance(chain_info.total_transactions, str)
        assert isinstance(chain_info.total_gas_used, str)
        assert isinstance(chain_info.average_block_time, int)
        assert isinstance(chain_info.difficulty, str)
        assert isinstance(chain_info.total_supply, str)
        
        # Verify hex values
        assert chain_info.block_height.startswith("0x")
        assert chain_info.total_transactions.startswith("0x")
        assert chain_info.total_gas_used.startswith("0x")
        assert chain_info.difficulty.startswith("0x")
        assert chain_info.total_supply.startswith("0x")
    
    @pytest.mark.asyncio
    async def test_get_validator_set(self, client):
        """Test hauptbuch_getValidatorSet RPC method"""
        validator_set = await client.get_validator_set()
        
        assert validator_set.validators is not None
        assert validator_set.total_stake is not None
        assert validator_set.active_validators is not None
        assert validator_set.total_validators is not None
        
        # Verify data types
        assert isinstance(validator_set.validators, list)
        assert isinstance(validator_set.total_stake, str)
        assert isinstance(validator_set.active_validators, int)
        assert isinstance(validator_set.total_validators, int)
        
        # Verify validator structure
        for validator in validator_set.validators:
            assert validator.address.startswith("0x")
            assert len(validator.address) == 42
            assert validator.stake.startswith("0x")
            assert isinstance(validator.voting_power, int)
            assert validator.status in ["active", "inactive", "slashed"]
            assert isinstance(validator.last_seen, int)
    
    @pytest.mark.asyncio
    async def test_get_block(self, client):
        """Test hauptbuch_getBlock RPC method"""
        # Get latest block
        chain_info = await client.get_chain_info()
        latest_height = int(chain_info.block_height, 16)
        
        # Test getting block by number
        block = await client.get_block(latest_height, include_transactions=True)
        
        assert block.number is not None
        assert block.hash is not None
        assert block.parent_hash is not None
        assert block.timestamp is not None
        assert block.gas_limit is not None
        assert block.gas_used is not None
        assert block.transactions is not None
        
        # Verify data types
        assert isinstance(block.number, str)
        assert isinstance(block.hash, str)
        assert isinstance(block.parent_hash, str)
        assert isinstance(block.timestamp, str)
        assert isinstance(block.gas_limit, str)
        assert isinstance(block.gas_used, str)
        assert isinstance(block.transactions, list)
        
        # Verify hex values
        assert block.number.startswith("0x")
        assert block.hash.startswith("0x")
        assert block.parent_hash.startswith("0x")
        assert block.timestamp.startswith("0x")
        assert block.gas_limit.startswith("0x")
        assert block.gas_used.startswith("0x")
    
    @pytest.mark.asyncio
    async def test_get_transaction(self, client):
        """Test hauptbuch_getTransaction RPC method"""
        # This would test with a real transaction hash
        # For now, test with a mock hash
        tx_hash = "0x1234567890123456789012345678901234567890"
        
        try:
            transaction = await client.get_transaction(tx_hash)
            
            assert transaction.hash is not None
            assert transaction.from_address is not None
            assert transaction.to is not None
            assert transaction.value is not None
            assert transaction.gas is not None
            assert transaction.gas_price is not None
            assert transaction.nonce is not None
            assert transaction.data is not None
            assert transaction.block_number is not None
            assert transaction.block_hash is not None
            assert transaction.transaction_index is not None
            
            # Verify data types
            assert isinstance(transaction.hash, str)
            assert isinstance(transaction.from_address, str)
            assert isinstance(transaction.to, str)
            assert isinstance(transaction.value, str)
            assert isinstance(transaction.gas, str)
            assert isinstance(transaction.gas_price, str)
            assert isinstance(transaction.nonce, str)
            assert isinstance(transaction.data, str)
            assert isinstance(transaction.block_number, str)
            assert isinstance(transaction.block_hash, str)
            assert isinstance(transaction.transaction_index, str)
            
        except Exception as e:
            # Transaction might not exist in test environment
            assert "Transaction not found" in str(e) or "Invalid transaction hash" in str(e)
    
    @pytest.mark.asyncio
    async def test_get_peer_list(self, client):
        """Test hauptbuch_getPeerList RPC method"""
        peers = await client.get_peer_list()
        
        assert isinstance(peers, list)
        
        for peer in peers:
            assert "id" in peer
            assert "address" in peer
            assert "status" in peer
            assert "capabilities" in peer
            assert "lastSeen" in peer
            assert "latency" in peer
            
            # Verify data types
            assert isinstance(peer["id"], str)
            assert isinstance(peer["address"], str)
            assert isinstance(peer["status"], str)
            assert isinstance(peer["capabilities"], list)
            assert isinstance(peer["lastSeen"], int)
            assert isinstance(peer["latency"], (int, float))
            
            # Verify status values
            assert peer["status"] in ["connected", "disconnected", "connecting"]
            assert peer["latency"] >= 0
    
    @pytest.mark.asyncio
    async def test_add_peer(self, client):
        """Test hauptbuch_addPeer RPC method"""
        peer_address = "127.0.0.1:30303"
        peer_id = "QmTestPeerId"
        
        result = await client.add_peer(peer_address, peer_id)
        
        assert "success" in result
        assert "peerId" in result
        assert "status" in result
        
        # Verify data types
        assert isinstance(result["success"], bool)
        assert isinstance(result["peerId"], str)
        assert isinstance(result["status"], str)
        
        assert result["success"] is True
        assert result["peerId"] == peer_id
        assert result["status"] == "connected"
    
    @pytest.mark.asyncio
    async def test_remove_peer(self, client):
        """Test hauptbuch_removePeer RPC method"""
        peer_id = "QmTestPeerId"
        
        result = await client.remove_peer(peer_id)
        
        assert "success" in result
        assert "peerId" in result
        
        # Verify data types
        assert isinstance(result["success"], bool)
        assert isinstance(result["peerId"], str)
        
        assert result["success"] is True
        assert result["peerId"] == peer_id
    
    @pytest.mark.asyncio
    async def test_generate_keypair(self, client):
        """Test hauptbuch_generateKeypair RPC method"""
        result = await client.generate_keypair("ml-dsa", 256)
        
        assert "privateKey" in result
        assert "publicKey" in result
        assert "address" in result
        assert "algorithm" in result
        
        # Verify data types
        assert isinstance(result["privateKey"], str)
        assert isinstance(result["publicKey"], str)
        assert isinstance(result["address"], str)
        assert isinstance(result["algorithm"], str)
        
        # Verify values
        assert result["privateKey"].startswith("0x")
        assert result["publicKey"].startswith("0x")
        assert result["address"].startswith("0x")
        assert result["algorithm"] == "ml-dsa"
    
    @pytest.mark.asyncio
    async def test_sign_message(self, client):
        """Test hauptbuch_signMessage RPC method"""
        message = "0x48656c6c6f2c20486175707462756321"  # "Hello, Hauptbuch!" in hex
        private_key = "0x1234567890123456789012345678901234567890123456789012345678901234"
        
        result = await client.sign_message(message, private_key, "ml-dsa")
        
        assert "signature" in result
        assert "publicKey" in result
        assert "algorithm" in result
        
        # Verify data types
        assert isinstance(result["signature"], str)
        assert isinstance(result["publicKey"], str)
        assert isinstance(result["algorithm"], str)
        
        # Verify values
        assert result["signature"].startswith("0x")
        assert result["publicKey"].startswith("0x")
        assert result["algorithm"] == "ml-dsa"
    
    @pytest.mark.asyncio
    async def test_verify_signature(self, client):
        """Test hauptbuch_verifySignature RPC method"""
        message = "0x48656c6c6f2c20486175707462756321"
        signature = "0x1234567890123456789012345678901234567890123456789012345678901234"
        public_key = "0x0987654321098765432109876543210987654321098765432109876543210987"
        
        result = await client.verify_signature(message, signature, public_key, "ml-dsa")
        
        # Verify data type
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_get_balance(self, client):
        """Test hauptbuch_getBalance RPC method"""
        address = "0x1234567890123456789012345678901234567890"
        
        balance = await client.get_balance(address)
        
        # Verify data type
        assert isinstance(balance, int)
        assert balance >= 0
    
    @pytest.mark.asyncio
    async def test_get_nonce(self, client):
        """Test hauptbuch_getNonce RPC method"""
        address = "0x1234567890123456789012345678901234567890"
        
        nonce = await client.get_nonce(address)
        
        # Verify data type
        assert isinstance(nonce, int)
        assert nonce >= 0
    
    @pytest.mark.asyncio
    async def test_get_code(self, client):
        """Test hauptbuch_getCode RPC method"""
        address = "0x1234567890123456789012345678901234567890"
        
        code = await client.get_code(address)
        
        # Verify data type
        assert isinstance(code, str)
        assert code.startswith("0x")
    
    @pytest.mark.asyncio
    async def test_send_transaction(self, client):
        """Test hauptbuch_sendTransaction RPC method"""
        # Create a mock signed transaction
        signed_transaction = {
            "from": "0x1234567890123456789012345678901234567890",
            "to": "0x0987654321098765432109876543210987654321",
            "value": "0x1000",
            "gas": "0x5208",
            "gasPrice": "0x3b9aca00",
            "nonce": "0x1",
            "data": "0x",
            "signature": "0x1234567890123456789012345678901234567890123456789012345678901234",
            "publicKey": "0x0987654321098765432109876543210987654321098765432109876543210987"
        }
        
        try:
            tx_hash = await client.send_transaction(signed_transaction)
            
            # Verify data type
            assert isinstance(tx_hash, str)
            assert tx_hash.startswith("0x")
            assert len(tx_hash) == 66
            
        except Exception as e:
            # Transaction might fail in test environment
            assert "Insufficient funds" in str(e) or "Invalid transaction" in str(e)
    
    @pytest.mark.asyncio
    async def test_get_transaction_status(self, client):
        """Test hauptbuch_getTransactionStatus RPC method"""
        tx_hash = "0x1234567890123456789012345678901234567890"
        
        status = await client.get_transaction_status(tx_hash)
        
        # Verify data type
        assert status in ["pending", "confirmed", "failed"]
    
    @pytest.mark.asyncio
    async def test_get_transaction_history(self, client):
        """Test hauptbuch_getTransactionHistory RPC method"""
        address = "0x1234567890123456789012345678901234567890"
        
        history = await client.get_transaction_history(address, limit=10)
        
        # Verify data type
        assert isinstance(history, list)
        
        for tx in history:
            assert tx.hash.startswith("0x")
            assert tx.from_address.startswith("0x")
            assert tx.to.startswith("0x")
            assert tx.value.startswith("0x")
            assert tx.gas.startswith("0x")
            assert tx.gas_price.startswith("0x")
            assert tx.nonce.startswith("0x")
            assert tx.data.startswith("0x")
            assert tx.block_number.startswith("0x")
            assert tx.block_hash.startswith("0x")
            assert tx.transaction_index.startswith("0x")

class TestCrossChainRPC:
    """Test suite for cross-chain RPC methods"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_get_bridge_status(self, client):
        """Test hauptbuch_getBridgeStatus RPC method"""
        bridge_status = await client.get_bridge_status()
        
        assert "bridges" in bridge_status
        assert "totalBridges" in bridge_status
        assert "activeBridges" in bridge_status
        
        # Verify data types
        assert isinstance(bridge_status["bridges"], list)
        assert isinstance(bridge_status["totalBridges"], int)
        assert isinstance(bridge_status["activeBridges"], int)
        
        # Verify bridge structure
        for bridge in bridge_status["bridges"]:
            assert "name" in bridge
            assert "sourceChain" in bridge
            assert "targetChain" in bridge
            assert "status" in bridge
            assert "totalTransfers" in bridge
            assert "pendingTransfers" in bridge
            
            assert isinstance(bridge["name"], str)
            assert isinstance(bridge["sourceChain"], str)
            assert isinstance(bridge["targetChain"], str)
            assert isinstance(bridge["status"], str)
            assert isinstance(bridge["totalTransfers"], int)
            assert isinstance(bridge["pendingTransfers"], int)
            
            assert bridge["status"] in ["active", "inactive", "error"]
    
    @pytest.mark.asyncio
    async def test_transfer_asset(self, client):
        """Test hauptbuch_transferAsset RPC method"""
        result = await client.transfer_asset(
            from_address="0x1234567890123456789012345678901234567890",
            to_address="0x0987654321098765432109876543210987654321",
            amount="1000000000000000000",
            source_chain="hauptbuch",
            target_chain="ethereum",
            asset="HBK"
        )
        
        assert "transactionHash" in result
        assert "bridgeId" in result
        assert "status" in result
        assert "estimatedTime" in result
        
        # Verify data types
        assert isinstance(result["transactionHash"], str)
        assert isinstance(result["bridgeId"], str)
        assert isinstance(result["status"], str)
        assert isinstance(result["estimatedTime"], int)
        
        assert result["transactionHash"].startswith("0x")
        assert result["status"] in ["pending", "confirmed", "failed"]
        assert result["estimatedTime"] > 0
    
    @pytest.mark.asyncio
    async def test_get_transfer_status(self, client):
        """Test hauptbuch_getTransferStatus RPC method"""
        tx_hash = "0x1234567890123456789012345678901234567890"
        
        transfer_status = await client.get_transfer_status(tx_hash)
        
        assert "status" in transfer_status
        assert "sourceTransaction" in transfer_status
        assert "targetTransaction" in transfer_status
        assert "bridgeId" in transfer_status
        assert "completionTime" in transfer_status
        
        # Verify data types
        assert isinstance(transfer_status["status"], str)
        assert isinstance(transfer_status["sourceTransaction"], str)
        assert isinstance(transfer_status["targetTransaction"], str)
        assert isinstance(transfer_status["bridgeId"], str)
        assert isinstance(transfer_status["completionTime"], int)
        
        assert transfer_status["status"] in ["pending", "completed", "failed"]
        assert transfer_status["sourceTransaction"].startswith("0x")
        assert transfer_status["targetTransaction"].startswith("0x")

class TestGovernanceRPC:
    """Test suite for governance RPC methods"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_get_proposals(self, client):
        """Test hauptbuch_getProposals RPC method"""
        proposals = await client.get_proposals()
        
        assert "proposals" in proposals
        assert "totalProposals" in proposals
        assert "activeProposals" in proposals
        
        # Verify data types
        assert isinstance(proposals["proposals"], list)
        assert isinstance(proposals["totalProposals"], int)
        assert isinstance(proposals["activeProposals"], int)
        
        # Verify proposal structure
        for proposal in proposals["proposals"]:
            assert "id" in proposal
            assert "title" in proposal
            assert "description" in proposal
            assert "author" in proposal
            assert "status" in proposal
            assert "startTime" in proposal
            assert "endTime" in proposal
            assert "votes" in proposal
            
            assert isinstance(proposal["id"], int)
            assert isinstance(proposal["title"], str)
            assert isinstance(proposal["description"], str)
            assert isinstance(proposal["author"], str)
            assert isinstance(proposal["status"], str)
            assert isinstance(proposal["startTime"], int)
            assert isinstance(proposal["endTime"], int)
            assert isinstance(proposal["votes"], dict)
            
            assert proposal["status"] in ["active", "passed", "failed", "executed"]
            assert proposal["author"].startswith("0x")
            assert proposal["startTime"] > 0
            assert proposal["endTime"] > proposal["startTime"]
    
    @pytest.mark.asyncio
    async def test_submit_proposal(self, client):
        """Test hauptbuch_submitProposal RPC method"""
        result = await client.submit_proposal(
            title="Test Proposal",
            description="This is a test proposal",
            author="0x1234567890123456789012345678901234567890",
            proposal_type="parameter_change",
            parameters={"blockTime": 5000, "gasLimit": 10000000}
        )
        
        assert "proposalId" in result
        assert "transactionHash" in result
        assert "status" in result
        
        # Verify data types
        assert isinstance(result["proposalId"], int)
        assert isinstance(result["transactionHash"], str)
        assert isinstance(result["status"], str)
        
        assert result["transactionHash"].startswith("0x")
        assert result["status"] == "submitted"
    
    @pytest.mark.asyncio
    async def test_vote(self, client):
        """Test hauptbuch_vote RPC method"""
        result = await client.vote(
            proposal_id=1,
            voter="0x1234567890123456789012345678901234567890",
            choice="yes",
            voting_power=1000
        )
        
        assert "success" in result
        assert "transactionHash" in result
        assert "votingPower" in result
        
        # Verify data types
        assert isinstance(result["success"], bool)
        assert isinstance(result["transactionHash"], str)
        assert isinstance(result["votingPower"], int)
        
        assert result["success"] is True
        assert result["transactionHash"].startswith("0x")
        assert result["votingPower"] == 1000

class TestAccountAbstractionRPC:
    """Test suite for account abstraction RPC methods"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_get_user_operations(self, client):
        """Test hauptbuch_getUserOperations RPC method"""
        account = "0x1234567890123456789012345678901234567890"
        
        user_ops = await client.get_user_operations(account)
        
        assert "userOperations" in user_ops
        assert "totalOperations" in user_ops
        assert "pendingOperations" in user_ops
        
        # Verify data types
        assert isinstance(user_ops["userOperations"], list)
        assert isinstance(user_ops["totalOperations"], int)
        assert isinstance(user_ops["pendingOperations"], int)
        
        # Verify user operation structure
        for op in user_ops["userOperations"]:
            assert "hash" in op
            assert "sender" in op
            assert "nonce" in op
            assert "callData" in op
            assert "signature" in op
            assert "status" in op
            
            assert isinstance(op["hash"], str)
            assert isinstance(op["sender"], str)
            assert isinstance(op["nonce"], str)
            assert isinstance(op["callData"], str)
            assert isinstance(op["signature"], str)
            assert isinstance(op["status"], str)
            
            assert op["hash"].startswith("0x")
            assert op["sender"].startswith("0x")
            assert op["nonce"].startswith("0x")
            assert op["callData"].startswith("0x")
            assert op["signature"].startswith("0x")
            assert op["status"] in ["pending", "confirmed", "failed"]
    
    @pytest.mark.asyncio
    async def test_submit_user_operation(self, client):
        """Test hauptbuch_submitUserOperation RPC method"""
        result = await client.submit_user_operation(
            sender="0x1234567890123456789012345678901234567890",
            nonce="0x1",
            call_data="0x608060405234801561001057600080fd5b50",
            signature="0x1234567890123456789012345678901234567890123456789012345678901234",
            paymaster="0x0987654321098765432109876543210987654321",
            gas_limit="0x5208",
            gas_price="0x3b9aca00"
        )
        
        assert "userOperationHash" in result
        assert "status" in result
        assert "estimatedGas" in result
        
        # Verify data types
        assert isinstance(result["userOperationHash"], str)
        assert isinstance(result["status"], str)
        assert isinstance(result["estimatedGas"], str)
        
        assert result["userOperationHash"].startswith("0x")
        assert result["status"] == "submitted"
        assert result["estimatedGas"].startswith("0x")

class TestLayer2RPC:
    """Test suite for Layer 2 RPC methods"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_get_rollup_status(self, client):
        """Test hauptbuch_getRollupStatus RPC method"""
        rollup_status = await client.get_rollup_status()
        
        assert "rollups" in rollup_status
        assert "totalRollups" in rollup_status
        assert "activeRollups" in rollup_status
        
        # Verify data types
        assert isinstance(rollup_status["rollups"], list)
        assert isinstance(rollup_status["totalRollups"], int)
        assert isinstance(rollup_status["activeRollups"], int)
        
        # Verify rollup structure
        for rollup in rollup_status["rollups"]:
            assert "name" in rollup
            assert "status" in rollup
            assert "sequencer" in rollup
            assert "prover" in rollup
            assert "totalTransactions" in rollup
            assert "pendingTransactions" in rollup
            
            assert isinstance(rollup["name"], str)
            assert isinstance(rollup["status"], str)
            assert isinstance(rollup["sequencer"], str)
            assert isinstance(rollup["prover"], str)
            assert isinstance(rollup["totalTransactions"], int)
            assert isinstance(rollup["pendingTransactions"], int)
            
            assert rollup["status"] in ["active", "inactive", "error"]
            assert rollup["sequencer"].startswith("0x")
            assert rollup["prover"].startswith("0x")
    
    @pytest.mark.asyncio
    async def test_submit_rollup_transaction(self, client):
        """Test hauptbuch_submitRollupTransaction RPC method"""
        transaction = {
            "from": "0x1234567890123456789012345678901234567890",
            "to": "0x0987654321098765432109876543210987654321",
            "value": "0x1000",
            "data": "0x608060405234801561001057600080fd5b50"
        }
        
        result = await client.submit_rollup_transaction("optimistic-rollup", transaction)
        
        assert "transactionHash" in result
        assert "rollupId" in result
        assert "status" in result
        assert "estimatedConfirmationTime" in result
        
        # Verify data types
        assert isinstance(result["transactionHash"], str)
        assert isinstance(result["rollupId"], str)
        assert isinstance(result["status"], str)
        assert isinstance(result["estimatedConfirmationTime"], int)
        
        assert result["transactionHash"].startswith("0x")
        assert result["rollupId"] == "optimistic-rollup"
        assert result["status"] == "submitted"
        assert result["estimatedConfirmationTime"] > 0

class TestMonitoringRPC:
    """Test suite for monitoring RPC methods"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, client):
        """Test hauptbuch_getMetrics RPC method"""
        metrics = await client.get_metrics()
        
        assert "metrics" in metrics
        assert "timestamp" in metrics
        
        # Verify data types
        assert isinstance(metrics["metrics"], dict)
        assert isinstance(metrics["timestamp"], int)
        
        # Verify metrics structure
        metrics_data = metrics["metrics"]
        assert "blockHeight" in metrics_data
        assert "transactionCount" in metrics_data
        assert "gasUsed" in metrics_data
        assert "peerCount" in metrics_data
        assert "memoryUsage" in metrics_data
        assert "cpuUsage" in metrics_data
        assert "diskUsage" in metrics_data
        assert "networkLatency" in metrics_data
        
        assert isinstance(metrics_data["blockHeight"], str)
        assert isinstance(metrics_data["transactionCount"], str)
        assert isinstance(metrics_data["gasUsed"], str)
        assert isinstance(metrics_data["peerCount"], int)
        assert isinstance(metrics_data["memoryUsage"], str)
        assert isinstance(metrics_data["cpuUsage"], float)
        assert isinstance(metrics_data["diskUsage"], str)
        assert isinstance(metrics_data["networkLatency"], int)
    
    @pytest.mark.asyncio
    async def test_get_health_status(self, client):
        """Test hauptbuch_getHealthStatus RPC method"""
        health_status = await client.get_health_status()
        
        assert "status" in health_status
        assert "components" in health_status
        assert "uptime" in health_status
        assert "lastHealthCheck" in health_status
        
        # Verify data types
        assert isinstance(health_status["status"], str)
        assert isinstance(health_status["components"], dict)
        assert isinstance(health_status["uptime"], int)
        assert isinstance(health_status["lastHealthCheck"], int)
        
        # Verify status values
        assert health_status["status"] in ["healthy", "unhealthy", "degraded"]
        assert health_status["uptime"] >= 0
        assert health_status["lastHealthCheck"] > 0
        
        # Verify components
        components = health_status["components"]
        assert "consensus" in components
        assert "network" in components
        assert "database" in components
        assert "cryptography" in components
        
        for component, status in components.items():
            assert status in ["healthy", "unhealthy", "degraded"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
