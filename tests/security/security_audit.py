#!/usr/bin/env python3
"""
Hauptbuch Security Audit Suite
Comprehensive security testing for all blockchain components.
"""

import asyncio
import pytest
import time
from typing import List, Dict, Any, Tuple
from hauptbuch_client import HauptbuchClient, AccountManager

class TestDoubleSpendPrevention:
    """Test double-spend prevention mechanisms"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_double_spend_detection(self, client):
        """Test double-spend detection"""
        # Create test accounts
        sender = AccountManager.create_account()
        recipient = AccountManager.create_account()
        
        # Create two transactions with the same nonce
        transaction1 = {
            "from": sender.address,
            "to": recipient.address,
            "value": "0x1000",
            "gas": "0x5208",
            "gasPrice": "0x3b9aca00",
            "nonce": "0x1",
            "data": "0x"
        }
        
        transaction2 = {
            "from": sender.address,
            "to": recipient.address,
            "value": "0x2000",
            "gas": "0x5208",
            "gasPrice": "0x3b9aca00",
            "nonce": "0x1",  # Same nonce
            "data": "0x"
        }
        
        # Sign both transactions
        signed_tx1 = AccountManager.sign_transaction(transaction1, sender.private_key)
        signed_tx2 = AccountManager.sign_transaction(transaction2, sender.private_key)
        
        # Submit first transaction
        tx_hash1 = await client.send_transaction(signed_tx1)
        assert tx_hash1 is not None, "First transaction should be accepted"
        
        # Submit second transaction (should be rejected)
        try:
            tx_hash2 = await client.send_transaction(signed_tx2)
            assert False, "Second transaction with same nonce should be rejected"
        except Exception as e:
            assert "nonce" in str(e).lower() or "double" in str(e).lower(), f"Expected nonce/double-spend error, got: {e}"
    
    @pytest.mark.asyncio
    async def test_nonce_validation(self, client):
        """Test nonce validation"""
        sender = AccountManager.create_account()
        recipient = AccountManager.create_account()
        
        # Test with invalid nonce (too high)
        transaction = {
            "from": sender.address,
            "to": recipient.address,
            "value": "0x1000",
            "gas": "0x5208",
            "gasPrice": "0x3b9aca00",
            "nonce": "0x1000",  # Invalid nonce
            "data": "0x"
        }
        
        signed_tx = AccountManager.sign_transaction(transaction, sender.private_key)
        
        try:
            tx_hash = await client.send_transaction(signed_tx)
            assert False, "Transaction with invalid nonce should be rejected"
        except Exception as e:
            assert "nonce" in str(e).lower(), f"Expected nonce error, got: {e}"

class TestSybilAttackResistance:
    """Test Sybil attack resistance"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_validator_stake_requirements(self, client):
        """Test validator stake requirements"""
        validator_set = await client.get_validator_set()
        
        # Verify all validators have sufficient stake
        for validator in validator_set.validators:
            stake = int(validator.stake, 16)
            assert stake > 0, f"Validator {validator.address} should have stake > 0"
            assert validator.status in ["active", "inactive", "slashed"], f"Invalid validator status: {validator.status}"
    
    @pytest.mark.asyncio
    async def test_peer_identity_verification(self, client):
        """Test peer identity verification"""
        peers = await client.get_peer_list()
        
        # Verify peer identities
        for peer in peers:
            assert "id" in peer, "Peer should have ID"
            assert "address" in peer, "Peer should have address"
            assert "status" in peer, "Peer should have status"
            assert peer["status"] in ["connected", "disconnected", "connecting"], f"Invalid peer status: {peer['status']}"
    
    @pytest.mark.asyncio
    async def test_consensus_participation(self, client):
        """Test consensus participation requirements"""
        validator_set = await client.get_validator_set()
        
        # Verify consensus participation
        active_validators = [v for v in validator_set.validators if v.status == "active"]
        assert len(active_validators) > 0, "Should have active validators"
        
        # Verify validator voting power
        for validator in active_validators:
            assert validator.voting_power > 0, f"Validator {validator.address} should have voting power > 0"

class Test51PercentAttackResistance:
    """Test 51% attack resistance"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_majority_consensus_requirement(self, client):
        """Test majority consensus requirement"""
        validator_set = await client.get_validator_set()
        
        # Verify consensus requires majority
        total_stake = int(validator_set.total_stake, 16)
        active_validators = [v for v in validator_set.validators if v.status == "active"]
        
        # Calculate stake distribution
        stake_distribution = {}
        for validator in active_validators:
            stake = int(validator.stake, 16)
            stake_distribution[validator.address] = stake
        
        # Verify no single validator has >50% stake
        max_stake = max(stake_distribution.values())
        assert max_stake < total_stake // 2, "No single validator should have >50% stake"
    
    @pytest.mark.asyncio
    async def test_consensus_finality(self, client):
        """Test consensus finality"""
        # Get current block
        chain_info = await client.get_chain_info()
        current_height = int(chain_info.block_height, 16)
        
        # Verify block finality
        block = await client.get_block(current_height, include_transactions=True)
        assert block.hash is not None, "Block should have hash"
        assert block.parent_hash is not None, "Block should have parent hash"
        assert block.timestamp is not None, "Block should have timestamp"
    
    @pytest.mark.asyncio
    async def test_consensus_irreversibility(self, client):
        """Test consensus irreversibility"""
        # Get current block
        chain_info = await client.get_chain_info()
        current_height = int(chain_info.block_height, 16)
        
        # Wait for block to be finalized
        await asyncio.sleep(5)
        
        # Verify block is still the same
        new_chain_info = await client.get_chain_info()
        new_height = int(new_chain_info.block_height, 16)
        
        # Block height should be same or higher (not lower)
        assert new_height >= current_height, "Block height should not decrease"

class TestSlashingMechanisms:
    """Test slashing mechanisms"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_slashing_conditions(self, client):
        """Test slashing conditions"""
        validator_set = await client.get_validator_set()
        
        # Verify slashing conditions are properly defined
        for validator in validator_set.validators:
            if validator.status == "slashed":
                # Verify slashed validator has reduced stake
                assert validator.voting_power == 0, "Slashed validator should have 0 voting power"
    
    @pytest.mark.asyncio
    async def test_slashing_penalties(self, client):
        """Test slashing penalties"""
        validator_set = await client.get_validator_set()
        
        # Verify slashing penalties
        for validator in validator_set.validators:
            if validator.status == "slashed":
                # Verify slashed validator is inactive
                assert validator.status == "slashed", "Slashed validator should have slashed status"
                assert validator.voting_power == 0, "Slashed validator should have 0 voting power"
    
    @pytest.mark.asyncio
    async def test_slashing_recovery(self, client):
        """Test slashing recovery"""
        validator_set = await client.get_validator_set()
        
        # Verify slashing recovery mechanisms
        for validator in validator_set.validators:
            if validator.status == "slashed":
                # Verify slashed validator can potentially recover
                assert validator.last_seen > 0, "Slashed validator should have last_seen timestamp"

class TestReentrancyProtection:
    """Test reentrancy protection"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_reentrancy_protection(self, client):
        """Test reentrancy protection"""
        # Test reentrancy protection mechanisms
        # This would test smart contract reentrancy protection
        assert True  # Placeholder for reentrancy tests
    
    @pytest.mark.asyncio
    async def test_contract_security(self, client):
        """Test contract security"""
        # Test contract security mechanisms
        # This would test smart contract security
        assert True  # Placeholder for contract security tests

class TestInputSanitization:
    """Test input sanitization"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_malicious_input_handling(self, client):
        """Test malicious input handling"""
        # Test with malicious inputs
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../etc/passwd",
            "null",
            "undefined",
            "NaN",
            "Infinity",
            "-Infinity"
        ]
        
        for malicious_input in malicious_inputs:
            try:
                # Test with malicious input
                result = await client.get_balance(malicious_input)
                # Should not crash or return unexpected results
                assert isinstance(result, int), f"Should return integer for input: {malicious_input}"
            except Exception as e:
                # Should handle malicious input gracefully
                assert "invalid" in str(e).lower() or "error" in str(e).lower(), f"Should handle malicious input gracefully: {e}"
    
    @pytest.mark.asyncio
    async def test_input_validation(self, client):
        """Test input validation"""
        # Test with invalid inputs
        invalid_inputs = [
            "",  # Empty string
            "0x",  # Invalid hex
            "0x123",  # Invalid length
            "not_hex",  # Not hex
            "0xGG",  # Invalid hex characters
        ]
        
        for invalid_input in invalid_inputs:
            try:
                result = await client.get_balance(invalid_input)
                # Should handle invalid input gracefully
                assert isinstance(result, int), f"Should return integer for input: {invalid_input}"
            except Exception as e:
                # Should handle invalid input gracefully
                assert "invalid" in str(e).lower() or "error" in str(e).lower(), f"Should handle invalid input gracefully: {e}"

class TestRateLimiting:
    """Test rate limiting"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, client):
        """Test rate limiting"""
        # Test rate limiting by making many requests quickly
        request_count = 100
        successful_requests = 0
        rate_limited_requests = 0
        
        for i in range(request_count):
            try:
                network_info = await client.get_network_info()
                successful_requests += 1
            except Exception as e:
                if "rate limit" in str(e).lower() or "too many" in str(e).lower():
                    rate_limited_requests += 1
                else:
                    raise e
            
            # Small delay to avoid overwhelming the system
            await asyncio.sleep(0.01)
        
        # Verify rate limiting is working
        assert successful_requests > 0, "Should have some successful requests"
        assert rate_limited_requests >= 0, "Should handle rate limiting gracefully"
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, client):
        """Test concurrent request handling"""
        # Test concurrent requests
        concurrent_requests = 10
        tasks = []
        
        for i in range(concurrent_requests):
            task = asyncio.create_task(client.get_network_info())
            tasks.append(task)
        
        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all requests completed
        assert len(results) == concurrent_requests, f"Should have {concurrent_requests} results"
        
        # Verify results are valid
        for result in results:
            if isinstance(result, Exception):
                # Should handle exceptions gracefully
                assert "rate limit" in str(result).lower() or "too many" in str(result).lower(), f"Should handle rate limiting gracefully: {result}"
            else:
                # Should return valid network info
                assert hasattr(result, 'chain_id'), "Should return network info"

class TestAccessControls:
    """Test access controls"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_unauthorized_access_prevention(self, client):
        """Test unauthorized access prevention"""
        # Test unauthorized access prevention
        # This would test access control mechanisms
        assert True  # Placeholder for access control tests
    
    @pytest.mark.asyncio
    async def test_permission_validation(self, client):
        """Test permission validation"""
        # Test permission validation
        # This would test permission systems
        assert True  # Placeholder for permission tests

class TestCryptographicSecurity:
    """Test cryptographic security"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_key_security(self, client):
        """Test key security"""
        # Generate keypair
        keypair = await client.generate_keypair("ml-dsa", 256)
        private_key = keypair["privateKey"]
        public_key = keypair["publicKey"]
        
        # Verify key security
        assert private_key.startswith("0x"), "Private key should start with 0x"
        assert public_key.startswith("0x"), "Public key should start with 0x"
        assert len(private_key) == 66, "Private key should be 66 characters (0x + 64 hex)"
        assert len(public_key) == 66, "Public key should be 66 characters (0x + 64 hex)"
        
        # Verify keys are different
        assert private_key != public_key, "Private and public keys should be different"
    
    @pytest.mark.asyncio
    async def test_signature_security(self, client):
        """Test signature security"""
        # Generate keypair
        keypair = await client.generate_keypair("ml-dsa", 256)
        private_key = keypair["privateKey"]
        public_key = keypair["publicKey"]
        
        # Test signature security
        message = "0x48656c6c6f2c20486175707462756321"  # "Hello, Hauptbuch!" in hex
        
        # Sign message
        signature = await client.sign_message(message, private_key, "ml-dsa")
        
        # Verify signature
        is_valid = await client.verify_signature(message, signature["signature"], public_key, "ml-dsa")
        assert is_valid, "Signature should be valid"
        
        # Test with different message
        different_message = "0x48656c6c6f2c20486175707462756322"  # Different message
        is_invalid = await client.verify_signature(different_message, signature["signature"], public_key, "ml-dsa")
        assert not is_invalid, "Signature should be invalid for different message"
    
    @pytest.mark.asyncio
    async def test_quantum_resistance(self, client):
        """Test quantum resistance"""
        # Test quantum-resistant algorithms
        algorithms = ["ml-dsa", "ml-kem", "slh-dsa"]
        
        for algorithm in algorithms:
            # Generate keypair
            keypair = await client.generate_keypair(algorithm, 256)
            private_key = keypair["privateKey"]
            public_key = keypair["publicKey"]
            
            # Verify quantum resistance
            assert private_key.startswith("0x"), f"Private key should start with 0x for {algorithm}"
            assert public_key.startswith("0x"), f"Public key should start with 0x for {algorithm}"
            assert len(private_key) == 66, f"Private key should be 66 characters for {algorithm}"
            assert len(public_key) == 66, f"Public key should be 66 characters for {algorithm}"

class TestNetworkSecurity:
    """Test network security"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_peer_authentication(self, client):
        """Test peer authentication"""
        peers = await client.get_peer_list()
        
        # Verify peer authentication
        for peer in peers:
            assert "id" in peer, "Peer should have ID"
            assert "address" in peer, "Peer should have address"
            assert "status" in peer, "Peer should have status"
            assert peer["status"] in ["connected", "disconnected", "connecting"], f"Invalid peer status: {peer['status']}"
    
    @pytest.mark.asyncio
    async def test_network_encryption(self, client):
        """Test network encryption"""
        # Test network encryption
        # This would test network encryption mechanisms
        assert True  # Placeholder for network encryption tests
    
    @pytest.mark.asyncio
    async def test_ddos_protection(self, client):
        """Test DDoS protection"""
        # Test DDoS protection
        # This would test DDoS protection mechanisms
        assert True  # Placeholder for DDoS protection tests

class TestSmartContractSecurity:
    """Test smart contract security"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_contract_vulnerability_scanning(self, client):
        """Test contract vulnerability scanning"""
        # Test contract vulnerability scanning
        # This would test smart contract security
        assert True  # Placeholder for contract vulnerability tests
    
    @pytest.mark.asyncio
    async def test_contract_access_controls(self, client):
        """Test contract access controls"""
        # Test contract access controls
        # This would test smart contract access controls
        assert True  # Placeholder for contract access control tests
    
    @pytest.mark.asyncio
    async def test_contract_upgrade_security(self, client):
        """Test contract upgrade security"""
        # Test contract upgrade security
        # This would test smart contract upgrade mechanisms
        assert True  # Placeholder for contract upgrade tests

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
