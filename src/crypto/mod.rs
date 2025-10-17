//! Cryptographic primitives for quantum-resistant security
//!
//! This module provides post-quantum cryptographic algorithms including
//! NIST-finalized standards (ML-KEM, ML-DSA, SLH-DSA) and legacy
//! CRYSTALS-Kyber/Dilithium implementations for backward compatibility.

pub mod binius;
pub mod engine;
pub mod nist_pqc;
pub mod plonky3;
pub mod quantum_resistant;

// Re-export NIST PQC types (primary)
pub use binius::{
    BinaryFieldElement,
    BinaryFieldPolynomial,
    BiniusCircuit,
    BiniusCommitment,
    // Error types
    BiniusError,
    BiniusMetrics,
    BiniusProof,
    // Core Binius types
    BiniusProver,
    BiniusResult,
    CircuitConstraint,
    ConstraintType,
};
pub use nist_pqc::{
    ml_dsa_ecdsa_hybrid_sign,

    ml_dsa_keygen,
    ml_dsa_sign,
    ml_dsa_verify,
    ml_kem_decapsulate,
    ml_kem_encapsulate,
    // NIST PQC functions
    ml_kem_keygen,
    // Hybrid modes
    ml_kem_x25519_hybrid_encapsulate,
    slh_dsa_keygen,
    slh_dsa_sign,
    slh_dsa_verify,

    // ML-DSA types (FIPS 204)
    MLDSAPublicKey,
    MLDSASecretKey,
    MLDSASecurityLevel,

    MLDSASignature,
    MLKEMCiphertext,
    // ML-KEM types (FIPS 203)
    MLKEMPublicKey,
    MLKEMSecretKey,
    MLKEMSecurityLevel,

    MLKEMSharedSecret,
    // Error types
    NISTPQCError,
    NISTPQCResult,
    // SLH-DSA types (FIPS 205)
    SLHDSAPublicKey,
    SLHDSASecretKey,
    SLHDSASecurityLevel,

    SLHDSASignature,
};
pub use plonky3::{
    AggregatedProof,
    Plonky3Circuit,
    Plonky3Constraint,
    Plonky3ConstraintType,

    // Error types
    Plonky3Error,
    Plonky3Metrics,
    Plonky3Proof,
    // Core Plonky3 types
    Plonky3Prover,
    Plonky3Result,
    RecursiveProof,
};

// Re-export legacy types for backward compatibility
pub use quantum_resistant::{
    dilithium_ecdsa_hybrid_sign,

    dilithium_keygen,
    dilithium_sign,
    dilithium_verify,

    kyber_decapsulate,
    kyber_encapsulate,
    // Main functions (deprecated)
    kyber_keygen,
    // Hybrid modes (deprecated)
    kyber_x25519_hybrid_encapsulate,
    DilithiumParams,
    // Dilithium types (deprecated - use ML-DSA)
    DilithiumPublicKey,
    DilithiumSecretKey,
    DilithiumSecurityLevel,

    DilithiumSignature,
    KyberCiphertext,
    KyberParams,
    // Kyber types (deprecated - use ML-KEM)
    KyberPublicKey,
    KyberSecretKey,
    KyberSecurityLevel,

    KyberSharedSecret,
    ModularArithmetic,

    NTTContext,
    // Error types (deprecated)
    PQCError,
    PQCResult,
    // Core primitives
    PolynomialRing,
};

// Re-export engine types for Python FFI
pub use engine::{
    CryptoConfig,
    CryptoEngine,
    MLKemKeypair,
    MLDsaKeypair,
    SLHDsaKeypair,
    Signature,
    VerificationResult,
    HybridKeypair,
    HybridSignature,
};
