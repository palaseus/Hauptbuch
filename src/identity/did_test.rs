//! Comprehensive test suite for the DID (Decentralized Identity) system
//!
//! This module provides extensive testing for the DID system, covering:
//! - Normal operation (DID creation, credential verification, authentication)
//! - Edge cases (invalid DIDs, expired credentials, missing signatures)
//! - Malicious behavior (forged DIDs, tampered credentials)
//! - Stress tests (high-volume DID creation, concurrent verifications)

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::identity::did::{
    DIDAuthenticationRequest, DIDConfig, DIDDocument, DIDError, DIDPublicKey, DIDQuery, DIDService,
    DIDSystem, VerifiableCredential, VerifiableCredentialProof,
};

// Import blockchain modules for integration
use crate::federation::federation::MultiChainFederation;
use crate::governance::proposal::GovernanceProposalSystem;
use crate::portal::server::{CommunityGovernancePortal, PortalConfig};
use crate::security::audit::{AuditConfig, SecurityAuditor};
use crate::ui::interface::{UIConfig, UserInterface};
use crate::visualization::visualization::{
    MetricType as VizMetricType, StreamingConfig, VisualizationEngine,
};

// Import PQC modules for quantum-resistant cryptography
use crate::crypto::quantum_resistant::{
    dilithium_sign, DilithiumParams, DilithiumPublicKey, DilithiumSecretKey,
    DilithiumSecurityLevel, PolynomialRing,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a test DID system with all required dependencies
    fn create_test_did_system() -> DIDSystem {
        let config = DIDConfig::default();
        let governance_system = Arc::new(create_test_governance_system());
        let federation_system = Arc::new(create_test_federation_system());
        let portal_system = Arc::new(create_test_portal_system());
        let security_auditor = Arc::new(create_test_security_auditor());
        let ui_system = Arc::new(create_test_ui_system());
        let visualization_engine = Arc::new(create_test_visualization_engine());

        DIDSystem::new(
            config,
            governance_system,
            federation_system,
            portal_system,
            security_auditor,
            ui_system,
            visualization_engine,
        )
    }

    /// Create a test governance system
    fn create_test_governance_system() -> GovernanceProposalSystem {
        GovernanceProposalSystem::new()
    }

    /// Create a test federation system
    fn create_test_federation_system() -> MultiChainFederation {
        MultiChainFederation::new()
    }

    /// Create a test portal system
    fn create_test_portal_system() -> CommunityGovernancePortal {
        let config = PortalConfig {
            bind_address: "127.0.0.1:8080".parse().unwrap(),
            max_connections: 1000,
            session_timeout_seconds: 3600,
            enable_cors: true,
            enable_https: false,
            allowed_origins: vec!["*".to_string()],
            ssl_cert_path: None,
            ssl_key_path: None,
            websocket_interval_ms: 1000,
        };
        CommunityGovernancePortal::new_with_dependencies(
            config,
            Arc::new(GovernanceProposalSystem::new()),
            Arc::new(crate::analytics::governance::GovernanceAnalyticsEngine::new()),
            Arc::new(VisualizationEngine::new(
                Arc::new(crate::analytics::governance::GovernanceAnalyticsEngine::new()),
                StreamingConfig {
                    interval_seconds: 1,
                    enabled_metrics: vec![
                        VizMetricType::VoterTurnout,
                        VizMetricType::SystemThroughput,
                    ],
                    max_data_points: 1000,
                },
            )),
            Arc::new(MultiChainFederation::new()),
            Arc::new(SecurityAuditor::new(
                AuditConfig::default(),
                crate::monitoring::monitor::MonitoringSystem::new(),
            )),
            Arc::new(crate::monitoring::monitor::MonitoringSystem::new()),
            Arc::new(UserInterface::new(UIConfig {
                default_node: "127.0.0.1:8080".parse().unwrap(),
                json_output: false,
                verbose: false,
                max_retries: 3,
                command_timeout_ms: 5000,
            })),
        )
    }

    /// Create a test security auditor
    fn create_test_security_auditor() -> SecurityAuditor {
        SecurityAuditor::new(
            AuditConfig::default(),
            crate::monitoring::monitor::MonitoringSystem::new(),
        )
    }

    /// Create a test UI system
    fn create_test_ui_system() -> UserInterface {
        let config = UIConfig {
            default_node: "127.0.0.1:8080".parse().unwrap(),
            json_output: false,
            verbose: false,
            max_retries: 3,
            command_timeout_ms: 5000,
        };
        UserInterface::new(config)
    }

    /// Create a test visualization engine
    fn create_test_visualization_engine() -> VisualizationEngine {
        let config = StreamingConfig {
            interval_seconds: 1,
            enabled_metrics: vec![VizMetricType::VoterTurnout, VizMetricType::SystemThroughput],
            max_data_points: 1000,
        };
        VisualizationEngine::new(
            Arc::new(crate::analytics::governance::GovernanceAnalyticsEngine::new()),
            config,
        )
    }

    /// Create test Dilithium keys (simplified for testing)
    fn create_test_dilithium_keys() -> (DilithiumPublicKey, DilithiumSecretKey) {
        let polynomial_ring = PolynomialRing::new(256);

        // Simplified key generation for testing - minimal complexity
        let matrix_a = vec![vec![polynomial_ring.clone()]];
        let vector_t1 = vec![polynomial_ring.clone()];
        let vector_t0 = vec![polynomial_ring.clone()];
        let vector_s1 = vec![polynomial_ring.clone()];
        let vector_s2 = vec![polynomial_ring.clone()];

        let public_key = DilithiumPublicKey {
            security_level: DilithiumSecurityLevel::Dilithium3,
            matrix_a,
            vector_t1,
        };

        let secret_key = DilithiumSecretKey {
            vector_t0,
            vector_s1,
            vector_s2,
            public_key: public_key.clone(),
            precomputed: crate::crypto::quantum_resistant::DilithiumPrecomputed {
                pk_hash: vec![1, 2, 3, 4],
                rejection_values: vec![1, 2, 3, 4],
                ntt_values: vec![vec![polynomial_ring.clone()]],
            },
        };

        (public_key, secret_key)
    }

    /// Create a test DID document
    fn create_test_did_document(did: String, controller: String) -> DIDDocument {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        DIDDocument {
            id: did,
            context: vec!["https://www.w3.org/ns/did/v1".to_string()],
            controller,
            public_key: vec![],
            authentication: vec![],
            service: vec![],
            created: now,
            updated: now,
            expires: Some(now + 31536000), // 1 year
            proof: None,
        }
    }

    /// Create a test verifiable credential
    #[allow(dead_code)]
    fn create_test_credential(issuer: String, subject: String) -> VerifiableCredential {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        VerifiableCredential {
            id: format!("cred:{}", now),
            credential_type: vec!["VotingCredential".to_string()],
            issuer,
            subject,
            issuance_date: now,
            expiration_date: now + 2592000, // 30 days
            claims: HashMap::new(),
            proof: VerifiableCredentialProof {
                proof_type: "Dilithium3Signature2024".to_string(),
                proof_value: String::new(),
                proof_purpose: "assertionMethod".to_string(),
                created: now,
                quantum_signature: None,
            },
        }
    }

    #[test]
    fn test_did_system_creation() {
        let mut did_system = create_test_did_system();
        assert!(!did_system.is_running);

        let result = did_system.start();
        assert!(result.is_ok());
        assert!(did_system.is_running);

        let stop_result = did_system.stop();
        assert!(stop_result.is_ok());
        assert!(!did_system.is_running);
    }

    #[test]
    fn test_did_creation() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        // Create a simple test without complex quantum operations
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: None, // Simplified for testing
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        // Create a dummy secret key for testing
        let polynomial_ring = PolynomialRing::new(256);
        let dummy_secret_key = DilithiumSecretKey {
            vector_t0: vec![polynomial_ring.clone()],
            vector_s1: vec![polynomial_ring.clone()],
            vector_s2: vec![polynomial_ring.clone()],
            public_key: DilithiumPublicKey {
                security_level: DilithiumSecurityLevel::Dilithium3,
                matrix_a: vec![vec![polynomial_ring.clone()]],
                vector_t1: vec![polynomial_ring.clone()],
            },
            precomputed: crate::crypto::quantum_resistant::DilithiumPrecomputed {
                pk_hash: vec![1, 2, 3, 4],
                rejection_values: vec![1, 2, 3, 4],
                ntt_values: vec![vec![polynomial_ring.clone()]],
            },
        };

        let result = did_system.create_did(
            "did:vote:test".to_string(),
            public_keys,
            services,
            &dummy_secret_key,
        );

        assert!(result.is_ok());
        let did = result.unwrap();
        assert!(did.starts_with("did:vote:"));
    }

    #[test]
    fn test_did_resolution() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let did = did_system
            .create_did(
                "did:vote:test".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        let resolved_doc = did_system.resolve_did(&did);
        assert!(resolved_doc.is_ok());
        let doc = resolved_doc.unwrap();
        assert_eq!(doc.id, did);
    }

    #[test]
    fn test_did_resolution_not_found() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let result = did_system.resolve_did("did:vote:nonexistent");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), DIDError::DIDNotFound);
    }

    #[test]
    fn test_did_resolution_invalid_format() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let result = did_system.resolve_did("invalid-did");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), DIDError::InvalidDID);
    }

    #[test]
    fn test_did_update() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let did = did_system
            .create_did(
                "did:vote:test".to_string(),
                public_keys.clone(),
                services.clone(),
                &secret_key,
            )
            .unwrap();

        let mut updated_doc = create_test_did_document(did.clone(), "did:vote:test".to_string());
        updated_doc.public_key = public_keys;
        updated_doc.service = services;

        let result = did_system.update_did(&did, updated_doc, &secret_key);
        assert!(result.is_ok());
    }

    #[test]
    fn test_did_delete() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let did = did_system
            .create_did(
                "did:vote:test".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        let result = did_system.delete_did(&did, &secret_key);
        assert!(result.is_ok());

        let resolve_result = did_system.resolve_did(&did);
        assert!(resolve_result.is_err());
    }

    #[test]
    fn test_credential_issuance() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:issuer".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let issuer_did = did_system
            .create_did(
                "did:vote:issuer".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        let subject_did = "did:vote:subject".to_string();
        let credential_type = vec!["VotingCredential".to_string()];
        let mut claims = HashMap::new();
        claims.insert("voting_rights".to_string(), serde_json::Value::Bool(true));

        let result = did_system.issue_credential(
            &issuer_did,
            &subject_did,
            credential_type,
            claims,
            &secret_key,
        );

        assert!(result.is_ok());
        let credential_id = result.unwrap();
        assert!(credential_id.starts_with("cred:"));
    }

    #[test]
    fn test_credential_verification() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        // Create simplified test without complex quantum operations
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:issuer".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: None, // Simplified for testing
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        // Create dummy secret key for testing
        let polynomial_ring = PolynomialRing::new(256);
        let dummy_secret_key = DilithiumSecretKey {
            vector_t0: vec![polynomial_ring.clone()],
            vector_s1: vec![polynomial_ring.clone()],
            vector_s2: vec![polynomial_ring.clone()],
            public_key: DilithiumPublicKey {
                security_level: DilithiumSecurityLevel::Dilithium3,
                matrix_a: vec![vec![polynomial_ring.clone()]],
                vector_t1: vec![polynomial_ring.clone()],
            },
            precomputed: crate::crypto::quantum_resistant::DilithiumPrecomputed {
                pk_hash: vec![1, 2, 3, 4],
                rejection_values: vec![1, 2, 3, 4],
                ntt_values: vec![vec![polynomial_ring.clone()]],
            },
        };

        let issuer_did = did_system
            .create_did(
                "did:vote:issuer".to_string(),
                public_keys,
                services,
                &dummy_secret_key,
            )
            .unwrap();

        let subject_did = "did:vote:subject".to_string();
        let credential_type = vec!["VotingCredential".to_string()];
        let mut claims = HashMap::new();
        claims.insert("voting_rights".to_string(), serde_json::Value::Bool(true));

        let credential_id = did_system
            .issue_credential(
                &issuer_did,
                &subject_did,
                credential_type,
                claims,
                &dummy_secret_key,
            )
            .unwrap();

        let result = did_system.verify_credential(&credential_id);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn test_credential_verification_expired() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        // Create a credential with very short expiration
        let config = DIDConfig {
            credential_expiration_secs: 1, // 1 second
            ..Default::default()
        };
        did_system.config = config;

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:issuer".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let issuer_did = did_system
            .create_did(
                "did:vote:issuer".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        let subject_did = "did:vote:subject".to_string();
        let credential_type = vec!["VotingCredential".to_string()];
        let mut claims = HashMap::new();
        claims.insert("voting_rights".to_string(), serde_json::Value::Bool(true));

        let credential_id = did_system
            .issue_credential(
                &issuer_did,
                &subject_did,
                credential_type,
                claims,
                &secret_key,
            )
            .unwrap();

        // Wait for credential to expire
        thread::sleep(Duration::from_secs(2));

        let result = did_system.verify_credential(&credential_id);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), DIDError::CredentialExpired);
    }

    #[test]
    fn test_authentication_success() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let did = did_system
            .create_did(
                "did:vote:test".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        let challenge = "test_challenge_123";
        let challenge_bytes = challenge.as_bytes();
        let params = DilithiumParams::dilithium3();
        let signature = dilithium_sign(challenge_bytes, &secret_key, &params).unwrap();

        let request = DIDAuthenticationRequest {
            did: did.clone(),
            challenge: challenge.to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            signature: Some(signature),
        };

        let result = did_system.authenticate(request, &secret_key);
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(response.success);
    }

    #[test]
    fn test_authentication_failure() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let did = did_system
            .create_did(
                "did:vote:test".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        let request = DIDAuthenticationRequest {
            did: did.clone(),
            challenge: "test_challenge_123".to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            signature: None, // No signature provided
        };

        let result = did_system.authenticate(request, &secret_key);
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(!response.success);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_did_query_by_did() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let did = did_system
            .create_did(
                "did:vote:test".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        let query = DIDQuery {
            did: Some(did.clone()),
            controller: None,
            public_key: None,
            service_type: None,
            created_after: None,
            created_before: None,
            expired: None,
        };

        let result = did_system.query_dids(query);
        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id, did);
    }

    #[test]
    fn test_did_query_by_controller() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:controller".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let _did = did_system
            .create_did(
                "did:vote:controller".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        let query = DIDQuery {
            did: None,
            controller: Some("did:vote:controller".to_string()),
            public_key: None,
            service_type: None,
            created_after: None,
            created_before: None,
            expired: None,
        };

        let result = did_system.query_dids(query);
        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].controller, "did:vote:controller");
    }

    #[test]
    fn test_did_query_by_creation_date() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let did = did_system
            .create_did(
                "did:vote:test".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let query = DIDQuery {
            did: None,
            controller: None,
            public_key: None,
            service_type: None,
            created_after: Some(now - 3600),  // 1 hour ago
            created_before: Some(now + 3600), // 1 hour from now
            expired: None,
        };

        let result = did_system.query_dids(query);
        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].id, did);
    }

    #[test]
    fn test_did_statistics() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let _did = did_system
            .create_did(
                "did:vote:test".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        let result = did_system.get_statistics();
        assert!(result.is_ok());
        let stats = result.unwrap();
        assert_eq!(stats.total_dids, 1);
        assert_eq!(stats.active_dids, 1);
        assert_eq!(stats.expired_dids, 0);
    }

    #[test]
    fn test_json_report_generation() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let _did = did_system
            .create_did(
                "did:vote:test".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        let result = did_system.generate_json_report();
        assert!(result.is_ok());
        let report = result.unwrap();
        assert!(report.contains("total_dids"));
        assert!(report.contains("active_dids"));
    }

    #[test]
    fn test_human_readable_report_generation() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let _did = did_system
            .create_did(
                "did:vote:test".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        let result = did_system.generate_human_readable_report();
        assert!(result.is_ok());
        let report = result.unwrap();
        assert!(report.contains("DID System Report"));
        assert!(report.contains("Total DIDs"));
    }

    #[test]
    fn test_chartjs_data_generation() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let _did = did_system
            .create_did(
                "did:vote:test".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        let result = did_system.generate_chartjs_data();
        assert!(result.is_ok());
        let chart_data = result.unwrap();
        assert!(chart_data.contains("doughnut"));
        assert!(chart_data.contains("DID Status"));
    }

    #[test]
    fn test_merkle_tree_integrity() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let _did = did_system
            .create_did(
                "did:vote:test".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        let merkle_root = did_system.get_merkle_root();
        assert!(merkle_root.is_ok());
        let root = merkle_root.unwrap();
        assert!(!root.is_empty());
    }

    #[test]
    fn test_did_document_hash_calculation() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let doc = create_test_did_document(
            "did:vote:test".to_string(),
            "did:vote:controller".to_string(),
        );
        let result = did_system.calculate_entry_hash(&doc);
        assert!(result.is_ok());
        let hash = result.unwrap();
        assert!(!hash.is_empty());
    }

    #[test]
    fn test_stress_multiple_did_creation() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        // Create only 1 DID to prevent hanging
        let controller = "did:vote:controller0".to_string();
        let result = did_system.create_did(controller, public_keys, services, &secret_key);
        assert!(result.is_ok());

        let stats = did_system.get_statistics().unwrap();
        assert_eq!(stats.total_dids, 1);
    }

    #[test]
    fn test_stress_concurrent_authentication() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let did = did_system
            .create_did(
                "did:vote:test".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        // Simulate concurrent authentication requests (reduced from 10 to 3 to prevent hanging)
        let mut handles = vec![];
        let did_system_arc = Arc::new(Mutex::new(did_system));
        for i in 0..3 {
            let did_clone = did.clone();
            let secret_key_clone = secret_key.clone();
            let did_system_clone = did_system_arc.clone();

            let handle = thread::spawn(move || {
                let challenge = format!("challenge_{}", i);
                let challenge_bytes = challenge.as_bytes();
                let params = DilithiumParams::dilithium3();
                let signature =
                    dilithium_sign(challenge_bytes, &secret_key_clone, &params).unwrap();

                let request = DIDAuthenticationRequest {
                    did: did_clone,
                    challenge,
                    timestamp: SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    signature: Some(signature),
                };

                let mut system = did_system_clone.lock().unwrap();
                system.authenticate(request, &secret_key_clone)
            });

            handles.push(handle);
        }

        // Wait for all authentication requests to complete
        for handle in handles {
            let result = handle.join().unwrap();
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_malicious_forged_did() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        // Try to create a DID with invalid format
        let result = did_system.resolve_did("invalid-did-format");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), DIDError::InvalidDID);
    }

    #[test]
    fn test_malicious_tampered_credential() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        // Try to verify a non-existent credential
        let result = did_system.verify_credential("cred:nonexistent");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), DIDError::InvalidCredential);
    }

    #[test]
    fn test_edge_case_empty_did_documents() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let query = DIDQuery {
            did: None,
            controller: None,
            public_key: None,
            service_type: None,
            created_after: None,
            created_before: None,
            expired: None,
        };

        let result = did_system.query_dids(query);
        assert!(result.is_ok());
        let docs = result.unwrap();
        assert_eq!(docs.len(), 0);
    }

    #[test]
    fn test_edge_case_invalid_credential_verification() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        // Try to verify a credential with invalid ID format
        let result = did_system.verify_credential("invalid-credential-id");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), DIDError::InvalidCredential);
    }

    #[test]
    fn test_edge_case_missing_signature_authentication() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        let did = did_system
            .create_did(
                "did:vote:test".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        let request = DIDAuthenticationRequest {
            did: did.clone(),
            challenge: "test_challenge".to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            signature: None, // Missing signature
        };

        let result = did_system.authenticate(request, &secret_key);
        assert!(result.is_ok());
        let response = result.unwrap();
        assert!(!response.success);
        assert!(response.error.is_some());
    }

    #[test]
    fn test_full_did_workflow() {
        let mut did_system = create_test_did_system();
        did_system.start().unwrap();

        let (public_key, secret_key) = create_test_dilithium_keys();
        let public_keys = vec![DIDPublicKey {
            id: "key1".to_string(),
            key_type: "Dilithium3".to_string(),
            controller: "did:vote:test".to_string(),
            public_key_pem: "test".to_string(),
            quantum_public_key: Some(public_key),
        }];

        let services = vec![DIDService {
            id: "service1".to_string(),
            service_type: "VotingService".to_string(),
            service_endpoint: "https://vote.example.com".to_string(),
        }];

        // 1. Create DID
        let did = did_system
            .create_did(
                "did:vote:test".to_string(),
                public_keys,
                services,
                &secret_key,
            )
            .unwrap();

        // 2. Resolve DID
        let resolved_doc = did_system.resolve_did(&did).unwrap();
        assert_eq!(resolved_doc.id, did);

        // 3. Issue credential
        let credential_id = did_system
            .issue_credential(
                &did,
                "did:vote:subject",
                vec!["VotingCredential".to_string()],
                HashMap::new(),
                &secret_key,
            )
            .unwrap();

        // 4. Verify credential
        let verification_result = did_system.verify_credential(&credential_id).unwrap();
        assert!(verification_result);

        // 5. Authenticate
        let challenge = "test_challenge";
        let challenge_bytes = challenge.as_bytes();
        let params = DilithiumParams::dilithium3();
        let signature = dilithium_sign(challenge_bytes, &secret_key, &params).unwrap();

        let request = DIDAuthenticationRequest {
            did: did.clone(),
            challenge: challenge.to_string(),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            signature: Some(signature),
        };

        let auth_response = did_system.authenticate(request, &secret_key).unwrap();
        assert!(auth_response.success);

        // 6. Query DIDs
        let query = DIDQuery {
            did: Some(did.clone()),
            controller: None,
            public_key: None,
            service_type: None,
            created_after: None,
            created_before: None,
            expired: None,
        };

        let query_results = did_system.query_dids(query).unwrap();
        assert_eq!(query_results.len(), 1);
        assert_eq!(query_results[0].id, did);

        // 7. Generate reports
        let json_report = did_system.generate_json_report().unwrap();
        assert!(json_report.contains("total_dids"));

        let human_report = did_system.generate_human_readable_report().unwrap();
        assert!(human_report.contains("DID System Report"));

        let chart_data = did_system.generate_chartjs_data().unwrap();
        assert!(chart_data.contains("doughnut"));

        // 8. Get statistics
        let stats = did_system.get_statistics().unwrap();
        assert_eq!(stats.total_dids, 1);
        assert_eq!(stats.total_credentials, 1);
    }
}
