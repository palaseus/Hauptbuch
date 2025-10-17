#!/usr/bin/env python3
"""
Hauptbuch Crypto FFI Wrapper
Python wrapper for quantum-resistant cryptographic operations via Rust FFI.
"""

import os
import sys
import asyncio
from typing import Dict, List, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Try to import the compiled Rust extension
    import hauptbuch_crypto
    CRYPTO_FFI_AVAILABLE = True
    logger.info("Hauptbuch crypto FFI module loaded successfully")
except ImportError as e:
    logger.warning(f"Hauptbuch crypto FFI module not available: {e}")
    logger.warning("Falling back to mock implementations for testing")
    CRYPTO_FFI_AVAILABLE = False
    hauptbuch_crypto = None

class CryptoFFIError(Exception):
    """Exception raised for crypto FFI errors"""
    pass

class QuantumResistantCrypto:
    """Quantum-resistant cryptography operations via FFI"""
    
    def __init__(self):
        if CRYPTO_FFI_AVAILABLE:
            self.engine = hauptbuch_crypto.PythonCryptoEngine()
            logger.info("Initialized quantum-resistant crypto engine")
        else:
            self.engine = None
            logger.warning("Using mock crypto engine - FFI not available")
    
    def generate_ml_kem_keypair(self, key_size: int = 256) -> Dict[str, bytes]:
        """Generate ML-KEM keypair"""
        if CRYPTO_FFI_AVAILABLE:
            try:
                keypair = self.engine.generate_ml_kem_keypair(key_size)
                return {
                    "private_key": keypair.private_key,
                    "public_key": keypair.public_key,
                    "address": keypair.address
                }
            except Exception as e:
                raise CryptoFFIError(f"Failed to generate ML-KEM keypair: {e}")
        else:
            # Mock implementation for testing
            import secrets
            private_key = secrets.token_bytes(32)
            public_key = secrets.token_bytes(32)
            address = secrets.token_bytes(20)
            return {
                "private_key": private_key,
                "public_key": public_key,
                "address": address
            }
    
    def generate_ml_dsa_keypair(self, key_size: int = 256) -> Dict[str, bytes]:
        """Generate ML-DSA keypair"""
        if CRYPTO_FFI_AVAILABLE:
            try:
                keypair = self.engine.generate_ml_dsa_keypair(key_size)
                return {
                    "private_key": keypair.private_key,
                    "public_key": keypair.public_key,
                    "address": keypair.address
                }
            except Exception as e:
                raise CryptoFFIError(f"Failed to generate ML-DSA keypair: {e}")
        else:
            # Mock implementation for testing
            import secrets
            private_key = secrets.token_bytes(32)
            public_key = secrets.token_bytes(32)
            address = secrets.token_bytes(20)
            return {
                "private_key": private_key,
                "public_key": public_key,
                "address": address
            }
    
    def generate_slh_dsa_keypair(self, key_size: int = 256) -> Dict[str, bytes]:
        """Generate SLH-DSA keypair"""
        if CRYPTO_FFI_AVAILABLE:
            try:
                keypair = self.engine.generate_slh_dsa_keypair(key_size)
                return {
                    "private_key": keypair.private_key,
                    "public_key": keypair.public_key,
                    "address": keypair.address
                }
            except Exception as e:
                raise CryptoFFIError(f"Failed to generate SLH-DSA keypair: {e}")
        else:
            # Mock implementation for testing
            import secrets
            private_key = secrets.token_bytes(32)
            public_key = secrets.token_bytes(32)
            address = secrets.token_bytes(20)
            return {
                "private_key": private_key,
                "public_key": public_key,
                "address": address
            }
    
    def sign_ml_dsa(self, message: bytes, private_key: bytes) -> Dict[str, Any]:
        """Sign message with ML-DSA"""
        if CRYPTO_FFI_AVAILABLE:
            try:
                signature = self.engine.sign_ml_dsa(message, private_key)
                return {
                    "signature": signature.signature,
                    "public_key": signature.public_key,
                    "algorithm": signature.algorithm
                }
            except Exception as e:
                raise CryptoFFIError(f"Failed to sign with ML-DSA: {e}")
        else:
            # Mock implementation for testing
            import hashlib
            import secrets
            signature = hashlib.sha256(private_key + message).digest()
            public_key = hashlib.sha256(private_key).digest()
            return {
                "signature": signature,
                "public_key": public_key,
                "algorithm": "ml-dsa"
            }
    
    def sign_slh_dsa(self, message: bytes, private_key: bytes) -> Dict[str, Any]:
        """Sign message with SLH-DSA"""
        if CRYPTO_FFI_AVAILABLE:
            try:
                signature = self.engine.sign_slh_dsa(message, private_key)
                return {
                    "signature": signature.signature,
                    "public_key": signature.public_key,
                    "algorithm": signature.algorithm
                }
            except Exception as e:
                raise CryptoFFIError(f"Failed to sign with SLH-DSA: {e}")
        else:
            # Mock implementation for testing
            import hashlib
            import secrets
            signature = hashlib.sha256(private_key + message).digest()
            public_key = hashlib.sha256(private_key).digest()
            return {
                "signature": signature,
                "public_key": public_key,
                "algorithm": "slh-dsa"
            }
    
    def verify_ml_dsa(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify ML-DSA signature"""
        if CRYPTO_FFI_AVAILABLE:
            try:
                result = self.engine.verify_ml_dsa(message, signature, public_key)
                return result.valid
            except Exception as e:
                raise CryptoFFIError(f"Failed to verify ML-DSA signature: {e}")
        else:
            # Mock implementation for testing
            import hashlib
            expected = hashlib.sha256(public_key + message).digest()
            return signature == expected
    
    def verify_slh_dsa(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify SLH-DSA signature"""
        if CRYPTO_FFI_AVAILABLE:
            try:
                result = self.engine.verify_slh_dsa(message, signature, public_key)
                return result.valid
            except Exception as e:
                raise CryptoFFIError(f"Failed to verify SLH-DSA signature: {e}")
        else:
            # Mock implementation for testing
            import hashlib
            expected = hashlib.sha256(public_key + message).digest()
            return signature == expected
    
    def encrypt_ml_kem(self, plaintext: bytes, public_key: bytes) -> bytes:
        """Encrypt with ML-KEM"""
        if CRYPTO_FFI_AVAILABLE:
            try:
                return self.engine.encrypt_ml_kem(plaintext, public_key)
            except Exception as e:
                raise CryptoFFIError(f"Failed to encrypt with ML-KEM: {e}")
        else:
            # Mock implementation for testing
            import secrets
            return secrets.token_bytes(len(plaintext) + 32)  # Add some overhead
    
    def decrypt_ml_kem(self, ciphertext: bytes, private_key: bytes) -> bytes:
        """Decrypt with ML-KEM"""
        if CRYPTO_FFI_AVAILABLE:
            try:
                return self.engine.decrypt_ml_kem(ciphertext, private_key)
            except Exception as e:
                raise CryptoFFIError(f"Failed to decrypt with ML-KEM: {e}")
        else:
            # Mock implementation for testing
            return ciphertext[:-32] if len(ciphertext) > 32 else b""
    
    def generate_hybrid_keypair(self, quantum_algorithm: str = "ml-dsa", classical_algorithm: str = "ed25519") -> Dict[str, bytes]:
        """Generate hybrid keypair (quantum-resistant + classical)"""
        if CRYPTO_FFI_AVAILABLE:
            try:
                return self.engine.generate_hybrid_keypair(quantum_algorithm, classical_algorithm)
            except Exception as e:
                raise CryptoFFIError(f"Failed to generate hybrid keypair: {e}")
        else:
            # Mock implementation for testing
            import secrets
            return {
                "quantum_private": secrets.token_bytes(32),
                "quantum_public": secrets.token_bytes(32),
                "classical_private": secrets.token_bytes(32),
                "classical_public": secrets.token_bytes(32),
                "address": secrets.token_bytes(20)
            }
    
    def sign_hybrid(self, message: bytes, quantum_private: bytes, classical_private: bytes) -> Dict[str, bytes]:
        """Sign with hybrid cryptography"""
        if CRYPTO_FFI_AVAILABLE:
            try:
                return self.engine.sign_hybrid(message, quantum_private, classical_private)
            except Exception as e:
                raise CryptoFFIError(f"Failed to sign with hybrid crypto: {e}")
        else:
            # Mock implementation for testing
            import hashlib
            quantum_sig = hashlib.sha256(quantum_private + message).digest()
            classical_sig = hashlib.sha256(classical_private + message).digest()
            combined_sig = hashlib.sha256(quantum_sig + classical_sig).digest()
            return {
                "quantum_signature": quantum_sig,
                "classical_signature": classical_sig,
                "combined_signature": combined_sig
            }
    
    def verify_hybrid(self, message: bytes, signature: Dict[str, bytes], quantum_public: bytes, classical_public: bytes) -> bool:
        """Verify hybrid signature"""
        if CRYPTO_FFI_AVAILABLE:
            try:
                return self.engine.verify_hybrid(message, signature, quantum_public, classical_public)
            except Exception as e:
                raise CryptoFFIError(f"Failed to verify hybrid signature: {e}")
        else:
            # Mock implementation for testing
            import hashlib
            quantum_sig = signature.get("quantum_signature", b"")
            classical_sig = signature.get("classical_signature", b"")
            expected_quantum = hashlib.sha256(quantum_public + message).digest()
            expected_classical = hashlib.sha256(classical_public + message).digest()
            return quantum_sig == expected_quantum and classical_sig == expected_classical

# Global crypto instance
_crypto_instance: Optional[QuantumResistantCrypto] = None

def get_crypto() -> QuantumResistantCrypto:
    """Get global crypto instance"""
    global _crypto_instance
    if _crypto_instance is None:
        _crypto_instance = QuantumResistantCrypto()
    return _crypto_instance

def is_ffi_available() -> bool:
    """Check if FFI is available"""
    return CRYPTO_FFI_AVAILABLE

def build_ffi_extension() -> bool:
    """Build the FFI extension module"""
    try:
        import subprocess
        import os
        
        # Change to the project root
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        os.chdir(project_root)
        
        # Build the Python extension
        result = subprocess.run([
            "maturin", "develop", "--release"
        ], capture_output=True, text=True, check=True)
        
        logger.info("FFI extension built successfully")
        logger.debug(f"Build output: {result.stdout}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to build FFI extension: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("maturin not found. Please install maturin: pip install maturin")
        return False
    except Exception as e:
        logger.error(f"Unexpected error building FFI extension: {e}")
        return False

# Utility functions
def hex_to_bytes(hex_string: str) -> bytes:
    """Convert hex string to bytes"""
    return bytes.fromhex(hex_string)

def bytes_to_hex(data: bytes) -> str:
    """Convert bytes to hex string"""
    return data.hex()

def create_address_from_public_key(public_key: bytes) -> str:
    """Create Ethereum-style address from public key"""
    import hashlib
    hash_bytes = hashlib.sha256(public_key).digest()
    return "0x" + hash_bytes[:20].hex()

# Example usage
if __name__ == "__main__":
    # Test the crypto operations
    crypto = get_crypto()
    
    print(f"FFI Available: {is_ffi_available()}")
    
    # Test ML-DSA
    print("\nTesting ML-DSA...")
    keypair = crypto.generate_ml_dsa_keypair()
    message = b"Hello, Hauptbuch!"
    signature = crypto.sign_ml_dsa(message, keypair["private_key"])
    is_valid = crypto.verify_ml_dsa(message, signature["signature"], keypair["public_key"])
    print(f"ML-DSA signature valid: {is_valid}")
    
    # Test SLH-DSA
    print("\nTesting SLH-DSA...")
    keypair = crypto.generate_slh_dsa_keypair()
    signature = crypto.sign_slh_dsa(message, keypair["private_key"])
    is_valid = crypto.verify_slh_dsa(message, signature["signature"], keypair["public_key"])
    print(f"SLH-DSA signature valid: {is_valid}")
    
    # Test hybrid crypto
    print("\nTesting hybrid crypto...")
    hybrid_keypair = crypto.generate_hybrid_keypair()
    hybrid_signature = crypto.sign_hybrid(
        message, 
        hybrid_keypair["quantum_private"], 
        hybrid_keypair["classical_private"]
    )
    is_valid = crypto.verify_hybrid(
        message, 
        hybrid_signature, 
        hybrid_keypair["quantum_public"], 
        hybrid_keypair["classical_public"]
    )
    print(f"Hybrid signature valid: {is_valid}")
    
    print("\nAll tests completed!")
