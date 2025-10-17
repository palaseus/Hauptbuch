//! Identity module for decentralized identity (DID) management
//!
//! This module provides W3C-compliant decentralized identity functionality
//! for the decentralized voting blockchain, including DID creation, resolution,
//! authentication, and verifiable credentials.

pub mod did;
pub mod verifiable_credentials;

// Re-export main types for convenience
pub use did::{
    DIDAuthenticationRequest, DIDAuthenticationResponse, DIDConfig, DIDDocument, DIDError,
    DIDMerkleNode, DIDMerkleTree, DIDProof, DIDPublicKey, DIDQuery, DIDResult, DIDService,
    DIDStatistics, DIDSystem, VerifiableCredential, VerifiableCredentialProof,
};
pub use verifiable_credentials::{
    BBSPlusMetrics,

    BBSPlusParameters,
    BBSPlusProof,
    BBSPlusSignature,
    BBSPlusSignatureSystem,
    CredentialProof,
    CredentialRegistry,
    CredentialRegistryMetrics,
    CredentialStatus,
    CredentialSubject,
    Issuer,
    SelectiveDisclosureProof,
    // W3C Verifiable Credentials types
    VerifiableCredential as W3CVerifiableCredential,
    // Error types
    VerifiableCredentialError,
    VerifiableCredentialResult,
    VerifiablePresentation,
};

#[cfg(test)]
pub mod did_test;
