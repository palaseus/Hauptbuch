//! Crypto Engine for Hauptbuch
//!
//! This module provides a unified crypto engine that integrates all quantum-resistant
//! cryptographic operations and provides a clean interface for the Python FFI bindings.

use std::sync::{Arc, Mutex};
use sha3::{Digest, Sha3_256};
use crate::crypto::nist_pqc::{
    MLKEMPublicKey, MLKEMSecretKey, MLKEMCiphertext,
    MLDSAPublicKey, MLDSASecretKey, MLDSASignature,
    SLHDSAPublicKey, SLHDSASecretKey, SLHDSASignature,
    MLKEMSecurityLevel, MLDSASecurityLevel, SLHDSASecurityLevel,
    ml_kem_keygen, ml_kem_encapsulate, ml_kem_decapsulate,
    ml_dsa_keygen, ml_dsa_sign, ml_dsa_verify,
    slh_dsa_keygen, slh_dsa_sign, slh_dsa_verify,
    NISTPQCResult, NISTPQCError
};

/// Configuration for the crypto engine
#[derive(Debug, Clone)]
pub struct CryptoConfig {
    pub quantum_resistant: bool,
    pub hybrid_mode: bool,
    pub nist_pqc_enabled: bool,
    pub ml_kem_enabled: bool,
    pub ml_dsa_enabled: bool,
    pub slh_dsa_enabled: bool,
}

impl Default for CryptoConfig {
    fn default() -> Self {
        Self {
            quantum_resistant: true,
            hybrid_mode: true,
            nist_pqc_enabled: true,
            ml_kem_enabled: true,
            ml_dsa_enabled: true,
            slh_dsa_enabled: true,
        }
    }
}

/// Main crypto engine
pub struct CryptoEngine {
    config: CryptoConfig,
}

impl CryptoEngine {
    pub async fn new(config: CryptoConfig) -> Result<Self, NISTPQCError> {
        Ok(Self { config })
    }

    /// Generate ML-KEM keypair
    pub async fn generate_ml_kem_keypair(&self, key_size: usize) -> Result<MLKemKeypair, NISTPQCError> {
        let security_level = match key_size {
            512 => MLKEMSecurityLevel::MLKEM512,
            768 => MLKEMSecurityLevel::MLKEM768,
            1024 => MLKEMSecurityLevel::MLKEM1024,
            _ => MLKEMSecurityLevel::MLKEM768, // Default to 768
        };

        let (public_key, secret_key) = ml_kem_keygen(security_level)?;
        
        // Generate address from public key
        let address = self.generate_address(&public_key.public_key);

        Ok(MLKemKeypair {
            private_key: secret_key.secret_key,
            public_key: public_key.public_key,
            address,
        })
    }

    /// Generate ML-DSA keypair
    pub async fn generate_ml_dsa_keypair(&self, key_size: usize) -> Result<MLDsaKeypair, NISTPQCError> {
        let security_level = match key_size {
            44 => MLDSASecurityLevel::MLDSA44,
            65 => MLDSASecurityLevel::MLDSA65,
            87 => MLDSASecurityLevel::MLDSA87,
            _ => MLDSASecurityLevel::MLDSA65, // Default to 65
        };

        let (public_key, secret_key) = ml_dsa_keygen(security_level)?;
        
        // Generate address from public key
        let address = self.generate_address(&public_key.public_key);

        Ok(MLDsaKeypair {
            private_key: secret_key.secret_key,
            public_key: public_key.public_key,
            address,
        })
    }

    /// Generate SLH-DSA keypair
    pub async fn generate_slh_dsa_keypair(&self, key_size: usize) -> Result<SLHDsaKeypair, NISTPQCError> {
        let security_level = match key_size {
            128 => SLHDSASecurityLevel::SLHDSA128f,
            192 => SLHDSASecurityLevel::SLHDSA192f,
            256 => SLHDSASecurityLevel::SLHDSA256f,
            _ => SLHDSASecurityLevel::SLHDSA128f, // Default to 128f
        };

        let (public_key, secret_key) = slh_dsa_keygen(security_level)?;
        
        // Generate address from public key
        let address = self.generate_address(&public_key.public_key);

        Ok(SLHDsaKeypair {
            private_key: secret_key.secret_key,
            public_key: public_key.public_key,
            address,
        })
    }

    /// Sign message with ML-DSA
    pub async fn sign_ml_dsa(&self, message: &[u8], private_key: &[u8]) -> Result<Signature, NISTPQCError> {
        // Create secret key from bytes
        let secret_key = MLDSASecretKey {
            security_level: MLDSASecurityLevel::MLDSA65,
            secret_key: private_key.to_vec(),
            generated_at: current_timestamp(),
        };

        let signature = ml_dsa_sign(&secret_key, message)?;
        
        // Derive public key from private key
        let public_key = self.derive_public_key_from_private(private_key);

        Ok(Signature {
            signature: signature.signature,
            public_key,
            algorithm: "ml-dsa".to_string(),
        })
    }

    /// Sign message with SLH-DSA
    pub async fn sign_slh_dsa(&self, message: &[u8], private_key: &[u8]) -> Result<Signature, NISTPQCError> {
        // Create secret key from bytes
        let secret_key = SLHDSASecretKey {
            security_level: SLHDSASecurityLevel::SLHDSA128f,
            secret_key: private_key.to_vec(),
            generated_at: current_timestamp(),
        };

        let signature = slh_dsa_sign(&secret_key, message)?;
        
        // Derive public key from private key
        let public_key = self.derive_public_key_from_private(private_key);

        Ok(Signature {
            signature: signature.signature,
            public_key,
            algorithm: "slh-dsa".to_string(),
        })
    }

    /// Verify ML-DSA signature
    pub async fn verify_ml_dsa(&self, message: &[u8], signature: &[u8], public_key: &[u8]) -> Result<VerificationResult, NISTPQCError> {
        // Create public key and signature from bytes
        let ml_dsa_public_key = MLDSAPublicKey {
            security_level: MLDSASecurityLevel::MLDSA65,
            public_key: public_key.to_vec(),
            generated_at: current_timestamp(),
        };

        let ml_dsa_signature = MLDSASignature {
            security_level: MLDSASecurityLevel::MLDSA65,
            signature: signature.to_vec(),
            message_hash: sha3::Sha3_256::digest(message).to_vec(),
            signed_at: current_timestamp(),
        };

        let valid = ml_dsa_verify(&ml_dsa_public_key, message, &ml_dsa_signature)?;

        Ok(VerificationResult {
            valid,
            algorithm: "ml-dsa".to_string(),
            details: if valid { Some("Signature is valid".to_string()) } else { Some("Signature verification failed".to_string()) },
        })
    }

    /// Verify SLH-DSA signature
    pub async fn verify_slh_dsa(&self, message: &[u8], signature: &[u8], public_key: &[u8]) -> Result<VerificationResult, NISTPQCError> {
        // Create public key and signature from bytes
        let slh_dsa_public_key = SLHDSAPublicKey {
            security_level: SLHDSASecurityLevel::SLHDSA128f,
            public_key: public_key.to_vec(),
            generated_at: current_timestamp(),
        };

        let slh_dsa_signature = SLHDSASignature {
            security_level: SLHDSASecurityLevel::SLHDSA128f,
            signature: signature.to_vec(),
            message_hash: sha3::Sha3_256::digest(message).to_vec(),
            signed_at: current_timestamp(),
        };

        let valid = slh_dsa_verify(&slh_dsa_public_key, message, &slh_dsa_signature)?;

        Ok(VerificationResult {
            valid,
            algorithm: "slh-dsa".to_string(),
            details: if valid { Some("Signature is valid".to_string()) } else { Some("Signature verification failed".to_string()) },
        })
    }

    /// Encrypt with ML-KEM
    pub async fn encrypt_ml_kem(&self, _plaintext: &[u8], public_key: &[u8]) -> Result<Vec<u8>, NISTPQCError> {
        let ml_kem_public_key = MLKEMPublicKey {
            security_level: MLKEMSecurityLevel::MLKEM768,
            public_key: public_key.to_vec(),
            generated_at: current_timestamp(),
        };

        let (ciphertext, _shared_secret) = ml_kem_encapsulate(&ml_kem_public_key, 32)?;
        
        // For simplicity, return the ciphertext directly
        // In a real implementation, you'd encrypt the plaintext with the shared secret
        Ok(ciphertext.ciphertext)
    }

    /// Decrypt with ML-KEM
    pub async fn decrypt_ml_kem(&self, ciphertext: &[u8], private_key: &[u8]) -> Result<Vec<u8>, NISTPQCError> {
        let ml_kem_secret_key = MLKEMSecretKey {
            security_level: MLKEMSecurityLevel::MLKEM768,
            secret_key: private_key.to_vec(),
            generated_at: current_timestamp(),
        };

        let ml_kem_ciphertext = MLKEMCiphertext {
            security_level: MLKEMSecurityLevel::MLKEM768,
            ciphertext: ciphertext.to_vec(),
            shared_secret_length: 32,
        };

        let shared_secret = ml_kem_decapsulate(&ml_kem_secret_key, &ml_kem_ciphertext)?;
        
        // For simplicity, return the shared secret directly
        // In a real implementation, you'd decrypt the ciphertext with the shared secret
        Ok(shared_secret.shared_secret)
    }

    /// Generate hybrid keypair
    pub async fn generate_hybrid_keypair(&self, quantum_algorithm: &str, _classical_algorithm: &str) -> Result<HybridKeypair, NISTPQCError> {
        // Generate quantum-resistant keypair
        let quantum_keypair = match quantum_algorithm {
            "ml-dsa" => {
                let ml_dsa = self.generate_ml_dsa_keypair(65).await?;
                (ml_dsa.private_key, ml_dsa.public_key)
            },
            "slh-dsa" => {
                let slh_dsa = self.generate_slh_dsa_keypair(128).await?;
                (slh_dsa.private_key, slh_dsa.public_key)
            },
            _ => return Err(NISTPQCError::InvalidParameters),
        };

        // Generate classical keypair (simplified)
        let classical_private = self.generate_classical_private_key();
        let classical_public = self.derive_public_key_from_private(&classical_private);

        // Generate address
        let address = self.generate_address(&quantum_keypair.1);

        Ok(HybridKeypair {
            quantum_private_key: quantum_keypair.0,
            quantum_public_key: quantum_keypair.1,
            classical_private_key: classical_private,
            classical_public_key: classical_public,
            address,
        })
    }

    /// Sign with hybrid cryptography
    pub async fn sign_hybrid(&self, message: &[u8], quantum_private: &[u8], classical_private: &[u8]) -> Result<HybridSignature, NISTPQCError> {
        // Sign with quantum-resistant algorithm
        let quantum_signature = self.sign_ml_dsa(message, quantum_private).await?;
        
        // Sign with classical algorithm (simplified)
        let classical_signature = self.sign_classical(message, classical_private);
        
        // Combine signatures
        let mut combined_signature = Vec::new();
        combined_signature.extend_from_slice(&quantum_signature.signature);
        combined_signature.extend_from_slice(&classical_signature);

        Ok(HybridSignature {
            quantum_signature: quantum_signature.signature,
            classical_signature,
            combined_signature,
        })
    }

    /// Verify hybrid signature
    pub async fn verify_hybrid(&self, message: &[u8], quantum_signature: &[u8], classical_signature: &[u8], quantum_public: &[u8], classical_public: &[u8]) -> Result<bool, NISTPQCError> {
        // Verify quantum signature
        let quantum_result = self.verify_ml_dsa(message, quantum_signature, quantum_public).await?;
        
        // Verify classical signature (simplified)
        let classical_valid = self.verify_classical(message, classical_signature, classical_public);
        
        Ok(quantum_result.valid && classical_valid)
    }

    // Helper methods
    fn generate_address(&self, public_key: &[u8]) -> Vec<u8> {
        let hash = sha3::Sha3_256::digest(public_key);
        hash[..20].to_vec() // Take first 20 bytes for address
    }

    fn derive_public_key_from_private(&self, private_key: &[u8]) -> Vec<u8> {
        // Simplified derivation - in reality this would be algorithm-specific
        sha3::Sha3_256::digest(private_key).to_vec()
    }

    fn generate_classical_private_key(&self) -> Vec<u8> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut key = vec![0u8; 32];
        for byte in &mut key {
            *byte = rng.gen();
        }
        key
    }

    fn sign_classical(&self, message: &[u8], private_key: &[u8]) -> Vec<u8> {
        // Simplified classical signature
        let mut hasher = sha3::Sha3_256::new();
        hasher.update(private_key);
        hasher.update(message);
        hasher.finalize().to_vec()
    }

    fn verify_classical(&self, message: &[u8], signature: &[u8], public_key: &[u8]) -> bool {
        // Simplified classical verification
        let expected = self.sign_classical(message, public_key);
        signature == expected
    }
}

/// Keypair types for Python FFI
#[derive(Debug, Clone)]
pub struct MLKemKeypair {
    pub private_key: Vec<u8>,
    pub public_key: Vec<u8>,
    pub address: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct MLDsaKeypair {
    pub private_key: Vec<u8>,
    pub public_key: Vec<u8>,
    pub address: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct SLHDsaKeypair {
    pub private_key: Vec<u8>,
    pub public_key: Vec<u8>,
    pub address: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct Signature {
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
    pub algorithm: String,
}

#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub valid: bool,
    pub algorithm: String,
    pub details: Option<String>,
}

#[derive(Debug, Clone)]
pub struct HybridKeypair {
    pub quantum_private_key: Vec<u8>,
    pub quantum_public_key: Vec<u8>,
    pub classical_private_key: Vec<u8>,
    pub classical_public_key: Vec<u8>,
    pub address: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct HybridSignature {
    pub quantum_signature: Vec<u8>,
    pub classical_signature: Vec<u8>,
    pub combined_signature: Vec<u8>,
}

fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}
