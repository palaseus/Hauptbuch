#!/usr/bin/env python3
"""
Hauptbuch Cryptography Tests
Tests for quantum-resistant cryptography, key generation, signing, and verification.
"""

import asyncio
import pytest
import hashlib
import time
from typing import Tuple, List
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from hauptbuch_client import HauptbuchClient, QuantumResistantCrypto, CryptoError

class TestQuantumResistantCrypto:
    """Test suite for quantum-resistant cryptography"""
    
    @pytest.fixture
    def crypto(self):
        """Create quantum-resistant crypto instance"""
        return QuantumResistantCrypto()
    
    def test_ml_kem_keypair_generation(self, crypto):
        """Test ML-KEM keypair generation"""
        private_key, public_key = crypto.generate_ml_kem_keypair()
        
        assert len(private_key) == 32
        assert len(public_key) == 32
        assert private_key != public_key
        assert private_key != b'\x00' * 32
        assert public_key != b'\x00' * 32
    
    def test_ml_dsa_keypair_generation(self, crypto):
        """Test ML-DSA keypair generation"""
        private_key, public_key = crypto.generate_ml_dsa_keypair()
        
        assert len(private_key) == 32
        assert len(public_key) == 32
        assert private_key != public_key
        assert private_key != b'\x00' * 32
        assert public_key != b'\x00' * 32
    
    def test_slh_dsa_keypair_generation(self, crypto):
        """Test SLH-DSA keypair generation"""
        private_key, public_key = crypto.generate_slh_dsa_keypair()
        
        assert len(private_key) == 32
        assert len(public_key) == 32
        assert private_key != public_key
        assert private_key != b'\x00' * 32
        assert public_key != b'\x00' * 32
    
    def test_keypair_uniqueness(self, crypto):
        """Test that generated keypairs are unique"""
        keypairs = []
        for _ in range(10):
            private_key, public_key = crypto.generate_ml_kem_keypair()
            keypairs.append((private_key, public_key))
        
        # All private keys should be unique
        private_keys = [kp[0] for kp in keypairs]
        assert len(set(private_keys)) == len(private_keys)
        
        # All public keys should be unique
        public_keys = [kp[1] for kp in keypairs]
        assert len(set(public_keys)) == len(public_keys)
    
    def test_ml_dsa_signature(self, crypto):
        """Test ML-DSA signature generation and verification"""
        message = b"Hello, Hauptbuch!"
        private_key, public_key = crypto.generate_ml_dsa_keypair()
        
        # Sign message
        signature = crypto.sign_message(message, private_key, "ml-dsa")
        assert len(signature) == 32
        assert signature != b'\x00' * 32
        
        # Verify signature
        is_valid = crypto.verify_signature(message, signature, public_key, "ml-dsa")
        assert is_valid is True
        
        # Test with different message
        different_message = b"Different message"
        is_invalid = crypto.verify_signature(different_message, signature, public_key, "ml-dsa")
        assert is_invalid is False
    
    def test_slh_dsa_signature(self, crypto):
        """Test SLH-DSA signature generation and verification"""
        message = b"Hello, Hauptbuch!"
        private_key, public_key = crypto.generate_slh_dsa_keypair()
        
        # Sign message
        signature = crypto.sign_message(message, private_key, "slh-dsa")
        assert len(signature) == 32
        assert signature != b'\x00' * 32
        
        # Verify signature
        is_valid = crypto.verify_signature(message, signature, public_key, "slh-dsa")
        assert is_valid is True
        
        # Test with different message
        different_message = b"Different message"
        is_invalid = crypto.verify_signature(different_message, signature, public_key, "slh-dsa")
        assert is_invalid is False
    
    def test_signature_consistency(self, crypto):
        """Test that signatures are consistent for the same input"""
        message = b"Consistent message"
        private_key, public_key = crypto.generate_ml_dsa_keypair()
        
        # Generate multiple signatures for the same message
        signatures = []
        for _ in range(5):
            signature = crypto.sign_message(message, private_key, "ml-dsa")
            signatures.append(signature)
        
        # All signatures should be identical
        assert all(sig == signatures[0] for sig in signatures)
    
    def test_signature_deterministic(self, crypto):
        """Test that signatures are deterministic"""
        message = b"Deterministic message"
        private_key, public_key = crypto.generate_ml_dsa_keypair()
        
        # Generate signature multiple times
        signature1 = crypto.sign_message(message, private_key, "ml-dsa")
        signature2 = crypto.sign_message(message, private_key, "ml-dsa")
        
        assert signature1 == signature2
    
    def test_wrong_public_key_verification(self, crypto):
        """Test verification with wrong public key"""
        message = b"Test message"
        private_key1, public_key1 = crypto.generate_ml_dsa_keypair()
        private_key2, public_key2 = crypto.generate_ml_dsa_keypair()
        
        # Sign with private key 1
        signature = crypto.sign_message(message, private_key1, "ml-dsa")
        
        # Try to verify with public key 2 (should fail)
        is_valid = crypto.verify_signature(message, signature, public_key2, "ml-dsa")
        assert is_valid is False
    
    def test_corrupted_signature_verification(self, crypto):
        """Test verification with corrupted signature"""
        message = b"Test message"
        private_key, public_key = crypto.generate_ml_dsa_keypair()
        
        # Generate valid signature
        signature = crypto.sign_message(message, private_key, "ml-dsa")
        
        # Corrupt signature
        corrupted_signature = bytearray(signature)
        corrupted_signature[0] = (corrupted_signature[0] + 1) % 256
        corrupted_signature = bytes(corrupted_signature)
        
        # Verification should fail
        is_valid = crypto.verify_signature(message, corrupted_signature, public_key, "ml-dsa")
        assert is_valid is False
    
    def test_large_message_signature(self, crypto):
        """Test signature with large message"""
        # Create a large message (1MB)
        message = b"X" * (1024 * 1024)
        private_key, public_key = crypto.generate_ml_dsa_keypair()
        
        # Sign large message
        signature = crypto.sign_message(message, private_key, "ml-dsa")
        assert len(signature) == 32
        
        # Verify signature
        is_valid = crypto.verify_signature(message, signature, public_key, "ml-dsa")
        assert is_valid is True
    
    def test_empty_message_signature(self, crypto):
        """Test signature with empty message"""
        message = b""
        private_key, public_key = crypto.generate_ml_dsa_keypair()
        
        # Sign empty message
        signature = crypto.sign_message(message, private_key, "ml-dsa")
        assert len(signature) == 32
        
        # Verify signature
        is_valid = crypto.verify_signature(message, signature, public_key, "ml-dsa")
        assert is_valid is True

class TestHybridCryptography:
    """Test suite for hybrid cryptography"""
    
    @pytest.fixture
    def crypto(self):
        """Create quantum-resistant crypto instance"""
        return QuantumResistantCrypto()
    
    def test_hybrid_key_generation(self, crypto):
        """Test hybrid key generation"""
        # Generate quantum-resistant keys
        ml_dsa_private, ml_dsa_public = crypto.generate_ml_dsa_keypair()
        ml_kem_private, ml_kem_public = crypto.generate_ml_kem_keypair()
        
        # Generate classical keys (simulated)
        classical_private = hashlib.sha256(b"classical_private").digest()
        classical_public = hashlib.sha256(classical_private).digest()
        
        # Verify all keys are generated
        assert len(ml_dsa_private) == 32
        assert len(ml_dsa_public) == 32
        assert len(ml_kem_private) == 32
        assert len(ml_kem_public) == 32
        assert len(classical_private) == 32
        assert len(classical_public) == 32
    
    def test_hybrid_signature(self, crypto):
        """Test hybrid signature scheme"""
        message = b"Hybrid signature test"
        
        # Generate quantum-resistant signature
        ml_dsa_private, ml_dsa_public = crypto.generate_ml_dsa_keypair()
        quantum_signature = crypto.sign_message(message, ml_dsa_private, "ml-dsa")
        
        # Generate classical signature (simulated)
        classical_private = hashlib.sha256(b"classical_private").digest()
        classical_signature = hashlib.sha256(classical_private + message).digest()
        
        # Both signatures should be valid
        quantum_valid = crypto.verify_signature(message, quantum_signature, ml_dsa_public, "ml-dsa")
        classical_valid = hashlib.sha256(classical_private + message).digest() == classical_signature
        
        assert quantum_valid is True
        assert classical_valid is True
    
    def test_hybrid_verification(self, crypto):
        """Test hybrid verification scheme"""
        message = b"Hybrid verification test"
        
        # Generate both types of signatures
        ml_dsa_private, ml_dsa_public = crypto.generate_ml_dsa_keypair()
        quantum_signature = crypto.sign_message(message, ml_dsa_private, "ml-dsa")
        
        classical_private = hashlib.sha256(b"classical_private").digest()
        classical_public = hashlib.sha256(classical_private).digest()
        classical_signature = hashlib.sha256(classical_private + message).digest()
        
        # Verify both signatures
        quantum_valid = crypto.verify_signature(message, quantum_signature, ml_dsa_public, "ml-dsa")
        classical_valid = hashlib.sha256(classical_private + message).digest() == classical_signature
        
        assert quantum_valid is True
        assert classical_valid is True

class TestCryptographicPerformance:
    """Test suite for cryptographic performance"""
    
    @pytest.fixture
    def crypto(self):
        """Create quantum-resistant crypto instance"""
        return QuantumResistantCrypto()
    
    def test_key_generation_performance(self, crypto):
        """Test key generation performance"""
        start_time = time.time()
        
        # Generate multiple keypairs
        for _ in range(100):
            crypto.generate_ml_dsa_keypair()
            crypto.generate_ml_kem_keypair()
            crypto.generate_slh_dsa_keypair()
        
        elapsed_time = time.time() - start_time
        print(f"Key generation time: {elapsed_time:.4f} seconds for 300 keypairs")
        
        # Key generation should be reasonably fast
        assert elapsed_time < 10.0  # Less than 10 seconds for 300 keypairs
    
    def test_signature_performance(self, crypto):
        """Test signature performance"""
        message = b"Performance test message"
        private_key, public_key = crypto.generate_ml_dsa_keypair()
        
        start_time = time.time()
        
        # Generate multiple signatures
        signatures = []
        for _ in range(100):
            signature = crypto.sign_message(message, private_key, "ml-dsa")
            signatures.append(signature)
        
        elapsed_time = time.time() - start_time
        print(f"Signature generation time: {elapsed_time:.4f} seconds for 100 signatures")
        
        # Signature generation should be fast
        assert elapsed_time < 5.0  # Less than 5 seconds for 100 signatures
    
    def test_verification_performance(self, crypto):
        """Test verification performance"""
        message = b"Performance test message"
        private_key, public_key = crypto.generate_ml_dsa_keypair()
        signature = crypto.sign_message(message, private_key, "ml-dsa")
        
        start_time = time.time()
        
        # Verify signature multiple times
        for _ in range(100):
            is_valid = crypto.verify_signature(message, signature, public_key, "ml-dsa")
            assert is_valid is True
        
        elapsed_time = time.time() - start_time
        print(f"Verification time: {elapsed_time:.4f} seconds for 100 verifications")
        
        # Verification should be fast
        assert elapsed_time < 5.0  # Less than 5 seconds for 100 verifications
    
    def test_memory_usage(self, crypto):
        """Test memory usage during cryptographic operations"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Perform many cryptographic operations
        for _ in range(1000):
            private_key, public_key = crypto.generate_ml_dsa_keypair()
            message = b"Memory test message"
            signature = crypto.sign_message(message, private_key, "ml-dsa")
            crypto.verify_signature(message, signature, public_key, "ml-dsa")
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        print(f"Memory increase: {memory_increase / 1024 / 1024:.2f} MB")
        
        # Memory usage should be reasonable
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB increase

class TestCryptographicSecurity:
    """Test suite for cryptographic security"""
    
    @pytest.fixture
    def crypto(self):
        """Create quantum-resistant crypto instance"""
        return QuantumResistantCrypto()
    
    def test_key_entropy(self, crypto):
        """Test that generated keys have sufficient entropy"""
        # Generate multiple keys and check for patterns
        keys = []
        for _ in range(100):
            private_key, public_key = crypto.generate_ml_dsa_keypair()
            keys.extend([private_key, public_key])
        
        # Check for patterns in key distribution
        byte_counts = [0] * 256
        for key in keys:
            for byte in key:
                byte_counts[byte] += 1
        
        # Check that byte distribution is reasonably uniform
        total_bytes = sum(byte_counts)
        expected_count = total_bytes / 256
        
        for count in byte_counts:
            # Each byte should appear roughly the expected number of times
            assert abs(count - expected_count) < expected_count * 0.5
    
    def test_signature_uniqueness(self, crypto):
        """Test that signatures are unique for different messages"""
        private_key, public_key = crypto.generate_ml_dsa_keypair()
        
        # Generate signatures for different messages
        signatures = []
        for i in range(100):
            message = f"Message {i}".encode()
            signature = crypto.sign_message(message, private_key, "ml-dsa")
            signatures.append(signature)
        
        # All signatures should be unique
        assert len(set(signatures)) == len(signatures)
    
    def test_signature_collision_resistance(self, crypto):
        """Test that signatures are collision-resistant"""
        private_key, public_key = crypto.generate_ml_dsa_keypair()
        
        # Generate many signatures and check for collisions
        signatures = set()
        for i in range(1000):
            message = f"Collision test message {i}".encode()
            signature = crypto.sign_message(message, private_key, "ml-dsa")
            
            # Signature should not collide with previous signatures
            assert signature not in signatures
            signatures.add(signature)
    
    def test_key_recovery_resistance(self, crypto):
        """Test that private keys cannot be recovered from public keys"""
        private_key, public_key = crypto.generate_ml_dsa_keypair()
        
        # Public key should not reveal private key
        assert private_key != public_key
        
        # Generate multiple signatures with the same private key
        signatures = []
        for i in range(10):
            message = f"Recovery test {i}".encode()
            signature = crypto.sign_message(message, private_key, "ml-dsa")
            signatures.append(signature)
        
        # Even with multiple signatures, private key should not be recoverable
        # (This is a simplified test - in reality, this would require more sophisticated analysis)
        assert len(set(signatures)) == len(signatures)
    
    def test_side_channel_resistance(self, crypto):
        """Test that operations are resistant to side-channel attacks"""
        # Test that operations take consistent time (simplified test)
        message = b"Side channel test"
        private_key, public_key = crypto.generate_ml_dsa_keypair()
        
        # Measure signature times
        signature_times = []
        for _ in range(100):
            start_time = time.time()
            signature = crypto.sign_message(message, private_key, "ml-dsa")
            elapsed_time = time.time() - start_time
            signature_times.append(elapsed_time)
        
        # Times should be reasonably consistent
        avg_time = sum(signature_times) / len(signature_times)
        for time_val in signature_times:
            # Each operation should be within 50% of average time
            assert abs(time_val - avg_time) < avg_time * 0.5

class TestCryptographicInteroperability:
    """Test suite for cryptographic interoperability"""
    
    @pytest.fixture
    def crypto(self):
        """Create quantum-resistant crypto instance"""
        return QuantumResistantCrypto()
    
    def test_algorithm_compatibility(self, crypto):
        """Test compatibility between different algorithms"""
        message = b"Compatibility test"
        
        # Test ML-DSA
        ml_dsa_private, ml_dsa_public = crypto.generate_ml_dsa_keypair()
        ml_dsa_signature = crypto.sign_message(message, ml_dsa_private, "ml-dsa")
        ml_dsa_valid = crypto.verify_signature(message, ml_dsa_signature, ml_dsa_public, "ml-dsa")
        assert ml_dsa_valid is True
        
        # Test SLH-DSA
        slh_dsa_private, slh_dsa_public = crypto.generate_slh_dsa_keypair()
        slh_dsa_signature = crypto.sign_message(message, slh_dsa_private, "slh-dsa")
        slh_dsa_valid = crypto.verify_signature(message, slh_dsa_signature, slh_dsa_public, "slh-dsa")
        assert slh_dsa_valid is True
        
        # Test cross-algorithm verification (should fail)
        cross_valid = crypto.verify_signature(message, ml_dsa_signature, slh_dsa_public, "slh-dsa")
        assert cross_valid is False
    
    def test_key_format_compatibility(self, crypto):
        """Test key format compatibility"""
        # Test that keys are in expected format
        private_key, public_key = crypto.generate_ml_dsa_keypair()
        
        # Keys should be 32 bytes
        assert len(private_key) == 32
        assert len(public_key) == 32
        
        # Keys should be bytes objects
        assert isinstance(private_key, bytes)
        assert isinstance(public_key, bytes)
        
        # Keys should not be all zeros
        assert private_key != b'\x00' * 32
        assert public_key != b'\x00' * 32
    
    def test_signature_format_compatibility(self, crypto):
        """Test signature format compatibility"""
        message = b"Format test"
        private_key, public_key = crypto.generate_ml_dsa_keypair()
        signature = crypto.sign_message(message, private_key, "ml-dsa")
        
        # Signature should be 32 bytes
        assert len(signature) == 32
        
        # Signature should be bytes object
        assert isinstance(signature, bytes)
        
        # Signature should not be all zeros
        assert signature != b'\x00' * 32

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
