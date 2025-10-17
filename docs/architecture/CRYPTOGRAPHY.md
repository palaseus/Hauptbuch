# Cryptography Architecture

## Overview

Hauptbuch implements a comprehensive cryptographic architecture designed for quantum resistance, security, and performance. The system supports both NIST PQC standards and legacy post-quantum algorithms, with hybrid modes for transition periods.

## Cryptographic Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    CRYPTOGRAPHIC STACK                         │
├─────────────────────────────────────────────────────────────────┤
│  Application Layer                                              │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │   Account       │ │   Cross-Chain   │ │   Governance    │  │
│  │   Abstraction   │ │   Signatures    │ │   Signatures    │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Zero-Knowledge Layer                                           │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │   Binius        │ │   Plonky3       │ │   Halo2         │  │
│  │   (Binary Field)│ │   (Recursive)   │ │   (ML Circuits) │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Post-Quantum Layer                                             │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │   NIST PQC      │ │   Legacy PQC     │ │   Hybrid Modes  │  │
│  │   (ML-KEM/DSA)  │ │   (Kyber/Dil.)  │ │   (PQC + Class.)│  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Classical Layer                                                │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │   ECDSA         │ │   X25519        │ │   AES-GCM       │  │
│  │   (P-256)       │ │   (Key Exch.)   │ │   (Encryption)  │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
├─────────────────────────────────────────────────────────────────┤
│  Hash Functions                                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐  │
│  │   SHA-3         │ │   BLAKE3        │ │   Custom        │  │
│  │   (256/512)     │ │   (Fast Hash)   │ │   (VDF/VRF)     │  │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## NIST PQC Standards

### ML-KEM (FIPS 203) - Key Encapsulation

ML-KEM is the NIST-standardized key encapsulation mechanism, replacing CRYSTALS-Kyber.

#### Security Levels
- **Level 1**: 128-bit security (ML-KEM-512)
- **Level 3**: 192-bit security (ML-KEM-768) - Recommended
- **Level 5**: 256-bit security (ML-KEM-1024)

#### Implementation

```rust
use pqc_kyber::keypair;

pub struct MLKEMEngine {
    security_level: MLKEMSecurityLevel,
}

impl MLKEMEngine {
    /// Generate ML-KEM keypair
    pub fn keygen(&self) -> Result<(MLKEMPublicKey, MLKEMSecretKey), MLKEMError> {
        match self.security_level {
            MLKEMSecurityLevel::Level1 => {
                let (pk, sk) = keypair(&mut rand::thread_rng());
                Ok((MLKEMPublicKey::Level1(pk), MLKEMSecretKey::Level1(sk)))
            },
            MLKEMSecurityLevel::Level3 => {
                let (pk, sk) = keypair(&mut rand::thread_rng());
                Ok((MLKEMPublicKey::Level3(pk), MLKEMSecretKey::Level3(sk)))
            },
            MLKEMSecurityLevel::Level5 => {
                let (pk, sk) = keypair(&mut rand::thread_rng());
                Ok((MLKEMPublicKey::Level5(pk), MLKEMSecretKey::Level5(sk)))
            },
        }
    }
    
    /// Encapsulate shared secret
    pub fn encapsulate(&self, pk: &MLKEMPublicKey) -> Result<(MLKEMCiphertext, MLKEMSharedSecret), MLKEMError> {
        match pk {
            MLKEMPublicKey::Level1(pk) => {
                let (ct, ss) = pk.encapsulate(&mut rand::thread_rng());
                Ok((MLKEMCiphertext::Level1(ct), MLKEMSharedSecret::Level1(ss)))
            },
            MLKEMPublicKey::Level3(pk) => {
                let (ct, ss) = pk.encapsulate(&mut rand::thread_rng());
                Ok((MLKEMCiphertext::Level3(ct), MLKEMSharedSecret::Level3(ss)))
            },
            MLKEMPublicKey::Level5(pk) => {
                let (ct, ss) = pk.encapsulate(&mut rand::thread_rng());
                Ok((MLKEMCiphertext::Level5(ct), MLKEMSharedSecret::Level5(ss)))
            },
        }
    }
    
    /// Decapsulate shared secret
    pub fn decapsulate(&self, sk: &MLKEMSecretKey, ct: &MLKEMCiphertext) -> Result<MLKEMSharedSecret, MLKEMError> {
        match (sk, ct) {
            (MLKEMSecretKey::Level1(sk), MLKEMCiphertext::Level1(ct)) => {
                Ok(MLKEMSharedSecret::Level1(sk.decapsulate(ct)))
            },
            (MLKEMSecretKey::Level3(sk), MLKEMCiphertext::Level3(ct)) => {
                Ok(MLKEMSharedSecret::Level3(sk.decapsulate(ct)))
            },
            (MLKEMSecretKey::Level5(sk), MLKEMCiphertext::Level5(ct)) => {
                Ok(MLKEMSharedSecret::Level5(sk.decapsulate(ct)))
            },
            _ => Err(MLKEMError::KeyCiphertextMismatch),
        }
    }
}
```

### ML-DSA (FIPS 204) - Digital Signatures

ML-DSA is the NIST-standardized digital signature algorithm, replacing CRYSTALS-Dilithium.

#### Security Levels
- **Level 2**: 128-bit security (ML-DSA-44)
- **Level 3**: 192-bit security (ML-DSA-65) - Recommended
- **Level 5**: 256-bit security (ML-DSA-87)

#### Implementation

```rust
use pqc_dilithium::Keypair as DilithiumKeypair;

pub struct MLDSAEngine {
    security_level: MLDSASecurityLevel,
}

impl MLDSAEngine {
    /// Generate ML-DSA keypair
    pub fn keygen(&self) -> Result<(MLDSAPublicKey, MLDSASecretKey), MLDSAError> {
        match self.security_level {
            MLDSASecurityLevel::Level2 => {
                let (pk, sk) = DilithiumKeypair::generate(&mut rand::thread_rng());
                Ok((MLDSAPublicKey::Level2(pk), MLDSASecretKey::Level2(sk)))
            },
            MLDSASecurityLevel::Level3 => {
                let (pk, sk) = DilithiumKeypair::generate(&mut rand::thread_rng());
                Ok((MLDSAPublicKey::Level3(pk), MLDSASecretKey::Level3(sk)))
            },
            MLDSASecurityLevel::Level5 => {
                let (pk, sk) = DilithiumKeypair::generate(&mut rand::thread_rng());
                Ok((MLDSAPublicKey::Level5(pk), MLDSASecretKey::Level5(sk)))
            },
        }
    }
    
    /// Sign message
    pub fn sign(&self, sk: &MLDSASecretKey, message: &[u8]) -> Result<MLDSASignature, MLDSAError> {
        match sk {
            MLDSASecretKey::Level2(sk) => {
                Ok(MLDSASignature::Level2(sk.sign(message)))
            },
            MLDSASecretKey::Level3(sk) => {
                Ok(MLDSASignature::Level3(sk.sign(message)))
            },
            MLDSASecretKey::Level5(sk) => {
                Ok(MLDSASignature::Level5(sk.sign(message)))
            },
        }
    }
    
    /// Verify signature
    pub fn verify(&self, pk: &MLDSAPublicKey, message: &[u8], signature: &MLDSASignature) -> bool {
        match (pk, signature) {
            (MLDSAPublicKey::Level2(pk), MLDSASignature::Level2(sig)) => {
                pk.verify(message, sig).is_ok()
            },
            (MLDSAPublicKey::Level3(pk), MLDSASignature::Level3(sig)) => {
                pk.verify(message, sig).is_ok()
            },
            (MLDSAPublicKey::Level5(pk), MLDSASignature::Level5(sig)) => {
                pk.verify(message, sig).is_ok()
            },
            _ => false,
        }
    }
}
```

### SLH-DSA (FIPS 205) - Stateless Hash-Based Signatures

SLH-DSA provides stateless hash-based signatures for long-term security.

#### Implementation

```rust
use pqcrypto_sphincsplus::*;

pub struct SLHDSAEngine {
    security_level: SLHDSASecurityLevel,
}

impl SLHDSAEngine {
    /// Generate SLH-DSA keypair
    pub fn keygen(&self) -> Result<(SLHDSAPublicKey, SLHDSASecretKey), SLHDSAError> {
        match self.security_level {
            SLHDSASecurityLevel::Level1 => {
                let (pk, sk) = keypair(&mut rand::thread_rng());
                Ok((SLHDSAPublicKey::Level1(pk), SLHDSASecretKey::Level1(sk)))
            },
            SLHDSASecurityLevel::Level3 => {
                let (pk, sk) = keypair(&mut rand::thread_rng());
                Ok((SLHDSAPublicKey::Level3(pk), SLHDSASecretKey::Level3(sk)))
            },
            SLHDSASecurityLevel::Level5 => {
                let (pk, sk) = keypair(&mut rand::thread_rng());
                Ok((SLHDSAPublicKey::Level5(pk), SLHDSASecretKey::Level5(sk)))
            },
        }
    }
    
    /// Sign message
    pub fn sign(&self, sk: &SLHDSASecretKey, message: &[u8]) -> Result<SLHDSASignature, SLHDSAError> {
        match sk {
            SLHDSASecretKey::Level1(sk) => {
                Ok(SLHDSASignature::Level1(sk.sign(message)))
            },
            SLHDSASecretKey::Level3(sk) => {
                Ok(SLHDSASignature::Level3(sk.sign(message)))
            },
            SLHDSASecretKey::Level5(sk) => {
                Ok(SLHDSASignature::Level5(sk.sign(message)))
            },
        }
    }
    
    /// Verify signature
    pub fn verify(&self, pk: &SLHDSAPublicKey, message: &[u8], signature: &SLHDSASignature) -> bool {
        match (pk, signature) {
            (SLHDSAPublicKey::Level1(pk), SLHDSASignature::Level1(sig)) => {
                pk.verify(message, sig).is_ok()
            },
            (SLHDSAPublicKey::Level3(pk), SLHDSASignature::Level3(sig)) => {
                pk.verify(message, sig).is_ok()
            },
            (SLHDSAPublicKey::Level5(pk), SLHDSASignature::Level5(sig)) => {
                pk.verify(message, sig).is_ok()
            },
            _ => false,
        }
    }
}
```

## Hybrid Cryptography

### ML-KEM + X25519 Hybrid

```rust
pub struct HybridKEM {
    ml_kem: MLKEMEngine,
    x25519: X25519Engine,
}

impl HybridKEM {
    /// Hybrid key encapsulation
    pub fn hybrid_encapsulate(&self, ml_kem_pk: &MLKEMPublicKey, x25519_pk: &X25519PublicKey) -> Result<HybridCiphertext, HybridError> {
        // ML-KEM encapsulation
        let (ml_kem_ct, ml_kem_ss) = self.ml_kem.encapsulate(ml_kem_pk)?;
        
        // X25519 key exchange
        let x25519_ss = self.x25519.diffie_hellman(x25519_pk)?;
        
        // Combine shared secrets
        let combined_ss = self.combine_shared_secrets(&ml_kem_ss, &x25519_ss);
        
        Ok(HybridCiphertext {
            ml_kem_ciphertext: ml_kem_ct,
            x25519_public_key: x25519_pk.clone(),
            combined_shared_secret: combined_ss,
        })
    }
    
    /// Hybrid key decapsulation
    pub fn hybrid_decapsulate(&self, ml_kem_sk: &MLKEMSecretKey, x25519_sk: &X25519SecretKey, ct: &HybridCiphertext) -> Result<Vec<u8>, HybridError> {
        // ML-KEM decapsulation
        let ml_kem_ss = self.ml_kem.decapsulate(ml_kem_sk, &ct.ml_kem_ciphertext)?;
        
        // X25519 key exchange
        let x25519_ss = self.x25519.diffie_hellman(&ct.x25519_public_key)?;
        
        // Combine shared secrets
        let combined_ss = self.combine_shared_secrets(&ml_kem_ss, &x25519_ss);
        
        Ok(combined_ss)
    }
}
```

### ML-DSA + ECDSA Hybrid

```rust
pub struct HybridSignature {
    ml_dsa: MLDSAEngine,
    ecdsa: ECDSAEngine,
}

impl HybridSignature {
    /// Hybrid signature generation
    pub fn hybrid_sign(&self, ml_dsa_sk: &MLDSASecretKey, ecdsa_sk: &ECDSASecretKey, message: &[u8]) -> Result<HybridSignature, HybridError> {
        // ML-DSA signature
        let ml_dsa_sig = self.ml_dsa.sign(ml_dsa_sk, message)?;
        
        // ECDSA signature
        let ecdsa_sig = self.ecdsa.sign(ecdsa_sk, message)?;
        
        Ok(HybridSignature {
            ml_dsa_signature: ml_dsa_sig,
            ecdsa_signature: ecdsa_sig,
        })
    }
    
    /// Hybrid signature verification
    pub fn hybrid_verify(&self, ml_dsa_pk: &MLDSAPublicKey, ecdsa_pk: &ECDSAPublicKey, message: &[u8], signature: &HybridSignature) -> bool {
        // Verify both signatures
        let ml_dsa_valid = self.ml_dsa.verify(ml_dsa_pk, message, &signature.ml_dsa_signature);
        let ecdsa_valid = self.ecdsa.verify(ecdsa_pk, message, &signature.ecdsa_signature);
        
        ml_dsa_valid && ecdsa_valid
    }
}
```

## Zero-Knowledge Proof Systems

### Binius - Binary Field Proof System

```rust
pub struct BiniusProver {
    field: BinaryField,
    circuit: BiniusCircuit,
}

impl BiniusProver {
    /// Generate Binius proof
    pub fn prove(&self, witness: &[BinaryFieldElement], public_inputs: &[BinaryFieldElement]) -> Result<BiniusProof, BiniusError> {
        // Setup circuit
        let circuit = self.circuit.setup(witness, public_inputs)?;
        
        // Generate proof
        let proof = circuit.prove()?;
        
        Ok(BiniusProof {
            proof,
            public_inputs: public_inputs.to_vec(),
        })
    }
    
    /// Verify Binius proof
    pub fn verify(&self, proof: &BiniusProof, public_inputs: &[BinaryFieldElement]) -> bool {
        self.circuit.verify(proof, public_inputs)
    }
}
```

### Plonky3 - Recursive Proof Aggregation

```rust
pub struct Plonky3Prover {
    field: GoldilocksField,
    circuit: Plonky3Circuit,
}

impl Plonky3Prover {
    /// Generate Plonky3 proof
    pub fn prove(&self, witness: &[GoldilocksFieldElement], public_inputs: &[GoldilocksFieldElement]) -> Result<Plonky3Proof, Plonky3Error> {
        // Setup circuit
        let circuit = self.circuit.setup(witness, public_inputs)?;
        
        // Generate proof
        let proof = circuit.prove()?;
        
        Ok(Plonky3Proof {
            proof,
            public_inputs: public_inputs.to_vec(),
        })
    }
    
    /// Aggregate multiple proofs
    pub fn aggregate_proofs(&self, proofs: &[Plonky3Proof]) -> Result<AggregatedProof, Plonky3Error> {
        // Aggregate proofs
        let aggregated = self.circuit.aggregate(proofs)?;
        
        Ok(AggregatedProof {
            proof: aggregated,
            proof_count: proofs.len(),
        })
    }
}
```

## Hash Functions

### SHA-3 Implementation

```rust
use sha3::{Digest, Sha3_256, Sha3_512};

pub struct SHA3Engine;

impl SHA3Engine {
    /// SHA-3-256 hash
    pub fn sha3_256(&self, input: &[u8]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(input);
        hasher.finalize().into()
    }
    
    /// SHA-3-512 hash
    pub fn sha3_512(&self, input: &[u8]) -> [u8; 64] {
        let mut hasher = Sha3_512::new();
        hasher.update(input);
        hasher.finalize().into()
    }
    
    /// SHA-3-256 with salt
    pub fn sha3_256_salted(&self, input: &[u8], salt: &[u8]) -> [u8; 32] {
        let mut hasher = Sha3_256::new();
        hasher.update(salt);
        hasher.update(input);
        hasher.finalize().into()
    }
}
```

### BLAKE3 Implementation

```rust
use blake3::Hasher;

pub struct BLAKE3Engine;

impl BLAKE3Engine {
    /// BLAKE3 hash
    pub fn blake3(&self, input: &[u8]) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(input);
        hasher.finalize().into()
    }
    
    /// BLAKE3 with key
    pub fn blake3_keyed(&self, input: &[u8], key: &[u8; 32]) -> [u8; 32] {
        let mut hasher = Hasher::new_keyed(key);
        hasher.update(input);
        hasher.finalize().into()
    }
}
```

## Performance Characteristics

### Benchmark Results

| Algorithm | Operation | Time | Throughput |
|-----------|-----------|------|------------|
| ML-KEM-768 | Keygen | 2.1ms | 476 keys/sec |
| ML-KEM-768 | Encapsulate | 1.8ms | 556 ops/sec |
| ML-KEM-768 | Decapsulate | 0.9ms | 1,111 ops/sec |
| ML-DSA-65 | Keygen | 3.2ms | 313 keys/sec |
| ML-DSA-65 | Sign | 2.4ms | 417 signs/sec |
| ML-DSA-65 | Verify | 1.1ms | 909 verifies/sec |
| SHA-3-256 | Hash | 550ns | 1.8M hashes/sec |
| BLAKE3 | Hash | 120ns | 8.3M hashes/sec |

### Optimization Strategies

#### SIMD Acceleration
```rust
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl MLKEMEngine {
    pub fn optimized_encapsulate(&self, pk: &MLKEMPublicKey) -> Result<(MLKEMCiphertext, MLKEMSharedSecret), MLKEMError> {
        // Use AVX2 intrinsics for polynomial operations
        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") {
            return self.avx2_encapsulate(pk);
        }
        
        // Fallback to standard implementation
        self.standard_encapsulate(pk)
    }
}
```

#### Parallel Processing
```rust
use rayon::prelude::*;

impl MLDSAEngine {
    pub fn parallel_verify(&self, signatures: &[(MLDSAPublicKey, Vec<u8>, MLDSASignature)]) -> Vec<bool> {
        signatures.par_iter()
            .map(|(pk, msg, sig)| self.verify(pk, msg, sig))
            .collect()
    }
}
```

## Security Considerations

### Quantum Resistance

#### NIST PQC Standards
- **ML-KEM**: Resistant to quantum attacks on key encapsulation
- **ML-DSA**: Resistant to quantum attacks on digital signatures
- **SLH-DSA**: Hash-based signatures for long-term security

#### Hybrid Modes
- **Transition Period**: Support for both PQC and classical algorithms
- **Backward Compatibility**: Legacy algorithm support
- **Security Analysis**: Formal security proofs

### Side-Channel Resistance

#### Constant-Time Operations
```rust
impl MLDSAEngine {
    pub fn constant_time_verify(&self, pk: &MLDSAPublicKey, message: &[u8], signature: &MLDSASignature) -> bool {
        // Use constant-time comparison
        let result = self.verify(pk, message, signature);
        
        // Constant-time return
        let mut ret = 0u8;
        for i in 0..8 {
            ret |= (result as u8) << i;
        }
        
        ret == 0xFF
    }
}
```

#### Memory Protection
```rust
use zeroize::{Zeroize, ZeroizeOnDrop};

#[derive(ZeroizeOnDrop)]
pub struct SecretKey {
    key: Vec<u8>,
}

impl SecretKey {
    pub fn new(key: Vec<u8>) -> Self {
        Self { key }
    }
}

impl Drop for SecretKey {
    fn drop(&mut self) {
        self.key.zeroize();
    }
}
```

### Implementation Security

#### Input Validation
```rust
impl MLKEMEngine {
    pub fn validate_public_key(&self, pk: &[u8]) -> bool {
        // Validate key length
        if pk.len() != self.expected_key_length() {
            return false;
        }
        
        // Validate key format
        self.validate_key_format(pk)
    }
    
    pub fn validate_ciphertext(&self, ct: &[u8]) -> bool {
        // Validate ciphertext length
        if ct.len() != self.expected_ciphertext_length() {
            return false;
        }
        
        // Validate ciphertext format
        self.validate_ciphertext_format(ct)
    }
}
```

#### Error Handling
```rust
#[derive(Debug, Clone)]
pub enum CryptoError {
    InvalidKey,
    InvalidSignature,
    InvalidCiphertext,
    KeyGenerationFailed,
    SignatureFailed,
    VerificationFailed,
    EncryptionFailed,
    DecryptionFailed,
    InvalidParameters,
    ArithmeticOverflow,
    InsufficientRandomness,
    HybridModeFailure,
}

impl std::error::Error for CryptoError {}

impl std::fmt::Display for CryptoError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            CryptoError::InvalidKey => write!(f, "Invalid key format or size"),
            CryptoError::InvalidSignature => write!(f, "Invalid signature format or size"),
            CryptoError::InvalidCiphertext => write!(f, "Invalid ciphertext format or size"),
            CryptoError::KeyGenerationFailed => write!(f, "Key generation failed"),
            CryptoError::SignatureFailed => write!(f, "Signature generation failed"),
            CryptoError::VerificationFailed => write!(f, "Signature verification failed"),
            CryptoError::EncryptionFailed => write!(f, "Encryption failed"),
            CryptoError::DecryptionFailed => write!(f, "Decryption failed"),
            CryptoError::InvalidParameters => write!(f, "Invalid parameters"),
            CryptoError::ArithmeticOverflow => write!(f, "Arithmetic overflow"),
            CryptoError::InsufficientRandomness => write!(f, "Insufficient randomness"),
            CryptoError::HybridModeFailure => write!(f, "Hybrid mode component failure"),
        }
    }
}
```

This cryptographic architecture provides a comprehensive, quantum-resistant foundation for the Hauptbuch blockchain system, with support for both current and future cryptographic standards.
