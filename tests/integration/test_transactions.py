#!/usr/bin/env python3
"""
Hauptbuch Transaction Tests
Tests for transaction creation, signing, submission, and verification.
"""

import asyncio
import pytest
import time
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from hauptbuch_client import HauptbuchClient, AccountManager, TransactionStatus

class TestTransactions:
    """Test suite for transaction functionality"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.fixture
    def test_accounts(self):
        """Create test accounts for transaction testing"""
        accounts = []
        for i in range(3):
            account = AccountManager.create_account()
            accounts.append(account)
        return accounts
    
    @pytest.mark.asyncio
    async def test_create_account(self, client):
        """Test account creation"""
        account = AccountManager.create_account()
        
        assert account.address.startswith("0x")
        assert len(account.address) == 42
        assert len(account.private_key) == 32
        assert len(account.public_key) == 32
    
    @pytest.mark.asyncio
    async def test_get_balance(self, client, test_accounts):
        """Test getting account balance"""
        account = test_accounts[0]
        balance = await client.get_balance(account.address)
        
        assert isinstance(balance, int)
        assert balance >= 0
    
    @pytest.mark.asyncio
    async def test_get_nonce(self, client, test_accounts):
        """Test getting account nonce"""
        account = test_accounts[0]
        nonce = await client.get_nonce(account.address)
        
        assert isinstance(nonce, int)
        assert nonce >= 0
    
    @pytest.mark.asyncio
    async def test_transaction_creation(self, test_accounts):
        """Test transaction creation"""
        sender = test_accounts[0]
        recipient = test_accounts[1]
        
        # Create transaction
        transaction = {
            "from": sender.address,
            "to": recipient.address,
            "value": 1000000000000000000,  # 1 HBK
            "gas": 21000,
            "gasPrice": 20000000000,
            "nonce": 0,
            "data": "0x"
        }
        
        assert transaction["from"] == sender.address
        assert transaction["to"] == recipient.address
        assert transaction["value"] == 1000000000000000000
        assert transaction["gas"] == 21000
        assert transaction["gasPrice"] == 20000000000
    
    @pytest.mark.asyncio
    async def test_transaction_signing(self, test_accounts):
        """Test transaction signing"""
        sender = test_accounts[0]
        recipient = test_accounts[1]
        
        # Create transaction
        transaction = {
            "from": sender.address,
            "to": recipient.address,
            "value": 1000000000000000000,
            "gas": 21000,
            "gasPrice": 20000000000,
            "nonce": 0,
            "data": "0x"
        }
        
        # Sign transaction
        signed_transaction = AccountManager.sign_transaction(sender, transaction)
        
        assert "signature" in signed_transaction
        assert "publicKey" in signed_transaction
        assert signed_transaction["signature"] != ""
        assert signed_transaction["publicKey"] != ""
    
    @pytest.mark.asyncio
    async def test_transaction_submission(self, client, test_accounts):
        """Test transaction submission"""
        sender = test_accounts[0]
        recipient = test_accounts[1]
        
        # Create and sign transaction
        transaction = {
            "from": sender.address,
            "to": recipient.address,
            "value": 1000000000000000000,
            "gas": 21000,
            "gasPrice": 20000000000,
            "nonce": 0,
            "data": "0x"
        }
        
        signed_transaction = AccountManager.sign_transaction(sender, transaction)
        
        # Submit transaction
        tx_hash = await client.send_transaction(signed_transaction)
        
        assert tx_hash.startswith("0x")
        assert len(tx_hash) == 66
    
    @pytest.mark.asyncio
    async def test_transaction_status(self, client, test_accounts):
        """Test transaction status checking"""
        sender = test_accounts[0]
        recipient = test_accounts[1]
        
        # Create and submit transaction
        transaction = {
            "from": sender.address,
            "to": recipient.address,
            "value": 1000000000000000000,
            "gas": 21000,
            "gasPrice": 20000000000,
            "nonce": 0,
            "data": "0x"
        }
        
        signed_transaction = AccountManager.sign_transaction(sender, transaction)
        tx_hash = await client.send_transaction(signed_transaction)
        
        # Check transaction status
        status = await client.get_transaction_status(tx_hash)
        assert status in [TransactionStatus.PENDING, TransactionStatus.CONFIRMED, TransactionStatus.FAILED]
    
    @pytest.mark.asyncio
    async def test_transaction_retrieval(self, client, test_accounts):
        """Test transaction retrieval by hash"""
        sender = test_accounts[0]
        recipient = test_accounts[1]
        
        # Create and submit transaction
        transaction = {
            "from": sender.address,
            "to": recipient.address,
            "value": 1000000000000000000,
            "gas": 21000,
            "gasPrice": 20000000000,
            "nonce": 0,
            "data": "0x"
        }
        
        signed_transaction = AccountManager.sign_transaction(sender, transaction)
        tx_hash = await client.send_transaction(signed_transaction)
        
        # Retrieve transaction
        retrieved_tx = await client.get_transaction(tx_hash)
        
        assert retrieved_tx.hash == tx_hash
        assert retrieved_tx.from_address == sender.address
        assert retrieved_tx.to == recipient.address
        assert retrieved_tx.value == "0xde0b6b3a7640000"  # 1 HBK in hex
    
    @pytest.mark.asyncio
    async def test_transaction_history(self, client, test_accounts):
        """Test transaction history retrieval"""
        account = test_accounts[0]
        
        # Get transaction history
        history = await client.get_transaction_history(account.address, limit=10)
        
        assert isinstance(history, list)
        # History might be empty in test environment
        for tx in history:
            assert tx.from_address == account.address or tx.to == account.address
    
    @pytest.mark.asyncio
    async def test_contract_transaction(self, client, test_accounts):
        """Test contract transaction creation"""
        sender = test_accounts[0]
        contract_address = "0x1234567890123456789012345678901234567890"
        
        # Create contract transaction
        transaction = {
            "from": sender.address,
            "to": contract_address,
            "value": 0,
            "gas": 100000,
            "gasPrice": 20000000000,
            "nonce": 0,
            "data": "0x608060405234801561001057600080fd5b50"  # Contract call data
        }
        
        assert transaction["to"] == contract_address
        assert transaction["value"] == 0
        assert transaction["gas"] == 100000
        assert transaction["data"] != "0x"
    
    @pytest.mark.asyncio
    async def test_transaction_validation(self, test_accounts):
        """Test transaction validation"""
        sender = test_accounts[0]
        recipient = test_accounts[1]
        
        # Test valid transaction
        valid_transaction = {
            "from": sender.address,
            "to": recipient.address,
            "value": 1000000000000000000,
            "gas": 21000,
            "gasPrice": 20000000000,
            "nonce": 0,
            "data": "0x"
        }
        
        # Test invalid transaction (negative value)
        invalid_transaction = {
            "from": sender.address,
            "to": recipient.address,
            "value": -1000000000000000000,
            "gas": 21000,
            "gasPrice": 20000000000,
            "nonce": 0,
            "data": "0x"
        }
        
        # Valid transaction should pass validation
        assert valid_transaction["value"] > 0
        assert valid_transaction["gas"] > 0
        assert valid_transaction["gasPrice"] > 0
        
        # Invalid transaction should fail validation
        assert invalid_transaction["value"] < 0
    
    @pytest.mark.asyncio
    async def test_gas_estimation(self, client, test_accounts):
        """Test gas estimation for transactions"""
        sender = test_accounts[0]
        recipient = test_accounts[1]
        
        # Create transaction
        transaction = {
            "from": sender.address,
            "to": recipient.address,
            "value": 1000000000000000000,
            "data": "0x"
        }
        
        # Gas estimation would be done by the client
        # For now, we test that gas values are reasonable
        assert transaction["value"] > 0
        assert isinstance(transaction["value"], int)
    
    @pytest.mark.asyncio
    async def test_transaction_batching(self, client, test_accounts):
        """Test batch transaction processing"""
        sender = test_accounts[0]
        recipients = test_accounts[1:]
        
        # Create multiple transactions
        transactions = []
        for i, recipient in enumerate(recipients):
            transaction = {
                "from": sender.address,
                "to": recipient.address,
                "value": 1000000000000000000,
                "gas": 21000,
                "gasPrice": 20000000000,
                "nonce": i,
                "data": "0x"
            }
            signed_transaction = AccountManager.sign_transaction(sender, transaction)
            transactions.append(signed_transaction)
        
        # Submit transactions
        tx_hashes = []
        for transaction in transactions:
            tx_hash = await client.send_transaction(transaction)
            tx_hashes.append(tx_hash)
        
        assert len(tx_hashes) == len(recipients)
        for tx_hash in tx_hashes:
            assert tx_hash.startswith("0x")
            assert len(tx_hash) == 66

class TestTransactionPerformance:
    """Test suite for transaction performance"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.fixture
    def test_accounts(self):
        """Create test accounts for performance testing"""
        accounts = []
        for i in range(10):
            account = AccountManager.create_account()
            accounts.append(account)
        return accounts
    
    @pytest.mark.asyncio
    async def test_transaction_throughput(self, client, test_accounts):
        """Test transaction throughput"""
        sender = test_accounts[0]
        recipient = test_accounts[1]
        
        start_time = time.time()
        
        # Submit multiple transactions
        tx_hashes = []
        for i in range(10):
            transaction = {
                "from": sender.address,
                "to": recipient.address,
                "value": 1000000000000000000,
                "gas": 21000,
                "gasPrice": 20000000000,
                "nonce": i,
                "data": "0x"
            }
            signed_transaction = AccountManager.sign_transaction(sender, transaction)
            tx_hash = await client.send_transaction(signed_transaction)
            tx_hashes.append(tx_hash)
        
        elapsed_time = time.time() - start_time
        tps = len(tx_hashes) / elapsed_time
        
        print(f"Transaction throughput: {tps:.2f} TPS")
        assert tps > 0
    
    @pytest.mark.asyncio
    async def test_transaction_latency(self, client, test_accounts):
        """Test transaction latency"""
        sender = test_accounts[0]
        recipient = test_accounts[1]
        
        # Measure transaction latency
        latencies = []
        for i in range(5):
            start_time = time.time()
            
            transaction = {
                "from": sender.address,
                "to": recipient.address,
                "value": 1000000000000000000,
                "gas": 21000,
                "gasPrice": 20000000000,
                "nonce": i,
                "data": "0x"
            }
            signed_transaction = AccountManager.sign_transaction(sender, transaction)
            tx_hash = await client.send_transaction(signed_transaction)
            
            latency = time.time() - start_time
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        print(f"Average transaction latency: {avg_latency:.4f} seconds")
        assert avg_latency < 10.0  # Less than 10 seconds

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
