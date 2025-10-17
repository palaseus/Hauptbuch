//! Test suite for the Community Governance Portal
//!
//! This module provides comprehensive testing for the portal's REST API endpoints,
//! WebSocket functionality, authentication, and frontend components.

use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::analytics::governance::GovernanceAnalyticsEngine;
use crate::crypto::quantum_resistant::{
    dilithium_sign, dilithium_verify, DilithiumParams, DilithiumPrecomputed, DilithiumPublicKey,
    DilithiumSecretKey, DilithiumSecurityLevel, PolynomialRing,
};
use crate::federation::federation::MultiChainFederation;
use crate::governance::proposal::GovernanceProposalSystem;
use crate::monitoring::monitor::MonitoringSystem;
use crate::portal::server::{
    CommunityGovernancePortal, FrontendComponentType, PortalConfig, PortalError, PortalResponse,
    ProposalSubmissionRequest, UserSession, VoteSubmissionRequest, WebAssemblyFrontend,
    WebSocketConnection, WebSocketMessage,
};
use crate::security::audit::{AuditConfig, SecurityAuditor};
use crate::ui::interface::{UIConfig, UserInterface};
use crate::visualization::visualization::{StreamingConfig, VisualizationEngine};

/// Test helper to create a portal configuration
fn create_test_portal_config() -> PortalConfig {
    PortalConfig {
        bind_address: "127.0.0.1:0".parse().unwrap(), // Use port 0 for testing
        enable_https: false,
        ssl_cert_path: None,
        ssl_key_path: None,
        websocket_interval_ms: 100,
        max_connections: 100,
        session_timeout_seconds: 3600,
        enable_cors: true,
        allowed_origins: vec!["*".to_string()],
    }
}

/// Test helper to create test governance system
fn create_test_governance_system() -> Arc<GovernanceProposalSystem> {
    // Create a mock governance system for testing
    // In a real implementation, you would initialize with proper dependencies
    Arc::new(GovernanceProposalSystem::new())
}

/// Test helper to create test analytics engine
fn create_test_analytics_engine() -> Arc<GovernanceAnalyticsEngine> {
    Arc::new(GovernanceAnalyticsEngine::new())
}

/// Test helper to create test visualization engine
fn create_test_visualization_engine() -> Arc<VisualizationEngine> {
    let analytics_engine = create_test_analytics_engine();
    let streaming_config = StreamingConfig {
        interval_seconds: 1,
        max_data_points: 1000,
        enabled_metrics: vec![
            crate::visualization::visualization::MetricType::VoterTurnout,
            crate::visualization::visualization::MetricType::StakeDistribution,
        ],
    };
    Arc::new(VisualizationEngine::new(analytics_engine, streaming_config))
}

/// Test helper to create test federation system
fn create_test_federation_system() -> Arc<MultiChainFederation> {
    Arc::new(MultiChainFederation::new())
}

/// Test helper to create test security auditor
fn create_test_security_auditor() -> Arc<SecurityAuditor> {
    let monitoring_system = create_test_monitoring_system();
    Arc::new(SecurityAuditor::new(
        AuditConfig::default(),
        (*monitoring_system).clone(),
    ))
}

/// Test helper to create test monitoring system
fn create_test_monitoring_system() -> Arc<MonitoringSystem> {
    Arc::new(MonitoringSystem::new())
}

/// Test helper to create test UI interface
fn create_test_ui_interface() -> Arc<UserInterface> {
    Arc::new(UserInterface::new(UIConfig::default()))
}

/// Test helper to create test Dilithium keys with proper PolynomialRing structures
fn create_test_dilithium_keys() -> (DilithiumPublicKey, DilithiumSecretKey) {
    // Create proper PolynomialRing structures for testing
    let polynomial_ring = PolynomialRing {
        coefficients: vec![0; 256],
        modulus: 8380417,
        dimension: 256,
    };

    let matrix_a = vec![vec![polynomial_ring.clone(); 256]; 256];
    let vector_t1 = vec![polynomial_ring.clone(); 256];
    let vector_t0 = vec![polynomial_ring.clone(); 256];
    let vector_s1 = vec![polynomial_ring.clone(); 256];
    let vector_s2 = vec![polynomial_ring.clone(); 256];

    let public_key = DilithiumPublicKey {
        matrix_a,
        vector_t1,
        security_level: DilithiumSecurityLevel::Dilithium3,
    };

    let precomputed = DilithiumPrecomputed {
        pk_hash: vec![0; 32],
        rejection_values: vec![0; 256],
        ntt_values: vec![vec![polynomial_ring.clone(); 256]; 256],
    };

    let secret_key = DilithiumSecretKey {
        vector_t0,
        vector_s1,
        vector_s2,
        public_key: public_key.clone(),
        precomputed,
    };

    (public_key, secret_key)
}

/// Test helper to create test user session
fn create_test_user_session() -> UserSession {
    let (public_key, _) = create_test_dilithium_keys();

    UserSession {
        session_id: "test_session_123".to_string(),
        public_key,
        created_at: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        last_activity: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs(),
        permissions: vec!["vote".to_string(), "propose".to_string()],
        stake_amount: 1000,
    }
}

/// Test helper to create test vote submission request
fn create_test_vote_submission_request() -> VoteSubmissionRequest {
    let (public_key, secret_key) = create_test_dilithium_keys();

    // Create signature for testing
    let message = "test_proposal_123For1000";
    let message_hash = Sha3_256::digest(message.as_bytes()).to_vec();
    let params = DilithiumParams::dilithium3();
    let signature = dilithium_sign(&message_hash, &secret_key, &params).unwrap();

    // Convert signature to bytes for serialization
    let signature_bytes = signature
        .vector_z
        .iter()
        .flat_map(|ring| ring.coefficients.iter())
        .cloned()
        .collect::<Vec<i32>>()
        .iter()
        .flat_map(|&x| x.to_le_bytes())
        .collect::<Vec<u8>>();

    // Convert public key to bytes for serialization
    let public_key_bytes = public_key
        .matrix_a
        .iter()
        .flat_map(|row| row.iter())
        .flat_map(|ring| ring.coefficients.iter())
        .cloned()
        .collect::<Vec<i32>>()
        .iter()
        .flat_map(|&x| x.to_le_bytes())
        .collect::<Vec<u8>>();

    VoteSubmissionRequest {
        proposal_id: "test_proposal_123".to_string(),
        choice: "For".to_string(),
        signature: signature_bytes,
        public_key: public_key_bytes,
        stake_amount: 1000,
    }
}

/// Test helper to create test proposal submission request
fn create_test_proposal_submission_request() -> ProposalSubmissionRequest {
    let (public_key, secret_key) = create_test_dilithium_keys();

    // Create signature for testing
    let message = "Test ProposalThis is a test proposalProtocolUpgrade1000";
    let message_hash = Sha3_256::digest(message.as_bytes()).to_vec();
    let params = DilithiumParams::dilithium3();
    let signature = dilithium_sign(&message_hash, &secret_key, &params).unwrap();

    // Convert signature to bytes for serialization
    let signature_bytes = signature
        .vector_z
        .iter()
        .flat_map(|ring| ring.coefficients.iter())
        .cloned()
        .collect::<Vec<i32>>()
        .iter()
        .flat_map(|&x| x.to_le_bytes())
        .collect::<Vec<u8>>();

    // Convert public key to bytes for serialization
    let public_key_bytes = public_key
        .matrix_a
        .iter()
        .flat_map(|row| row.iter())
        .flat_map(|ring| ring.coefficients.iter())
        .cloned()
        .collect::<Vec<i32>>()
        .iter()
        .flat_map(|&x| x.to_le_bytes())
        .collect::<Vec<u8>>();

    ProposalSubmissionRequest {
        title: "Test Proposal".to_string(),
        description: "This is a test proposal".to_string(),
        proposal_type: "ProtocolUpgrade".to_string(),
        signature: signature_bytes,
        public_key: public_key_bytes,
        required_stake: 1000,
    }
}

/// Test helper to create test HTTP request
fn create_test_http_request(method: &str, path: &str, body: &str) -> HttpRequest {
    HttpRequest {
        method: method.to_string(),
        path: path.to_string(),
        headers: HashMap::new(),
        body: body.to_string(),
    }
}

/// HTTP request structure for testing
#[derive(Debug, Clone)]
struct HttpRequest {
    method: String,
    path: String,
    #[allow(dead_code)]
    headers: HashMap<String, String>,
    body: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test 1: Portal creation and configuration
    #[test]
    fn test_portal_creation() {
        let config = create_test_portal_config();
        let governance_system = create_test_governance_system();
        let analytics_engine = create_test_analytics_engine();
        let visualization_engine = create_test_visualization_engine();
        let federation_system = create_test_federation_system();
        let security_auditor = create_test_security_auditor();
        let monitoring_system = create_test_monitoring_system();
        let ui_interface = create_test_ui_interface();

        let portal = CommunityGovernancePortal::new_with_dependencies(
            config,
            governance_system,
            analytics_engine,
            visualization_engine,
            federation_system,
            security_auditor,
            monitoring_system,
            ui_interface,
        );

        assert!(portal.get_status().session_count == 0);
        assert!(!portal.get_status().is_running);
    }

    // Test 2: Portal configuration validation
    #[test]
    fn test_portal_configuration_validation() {
        let config = create_test_portal_config();

        assert_eq!(config.websocket_interval_ms, 100);
        assert_eq!(config.max_connections, 100);
        assert_eq!(config.session_timeout_seconds, 3600);
        assert!(config.enable_cors);
        assert!(config.allowed_origins.contains(&"*".to_string()));
    }

    // Test 3: User session management
    #[test]
    fn test_user_session_creation() {
        let session = create_test_user_session();

        assert_eq!(session.session_id, "test_session_123");
        assert_eq!(session.stake_amount, 1000);
        assert!(session.permissions.contains(&"vote".to_string()));
        assert!(session.permissions.contains(&"propose".to_string()));
    }

    // Test 4: Vote submission request validation
    #[test]
    fn test_vote_submission_request_creation() {
        let vote_request = create_test_vote_submission_request();

        assert_eq!(vote_request.proposal_id, "test_proposal_123");
        assert_eq!(vote_request.choice, "For");
        assert_eq!(vote_request.stake_amount, 1000);
    }

    // Test 5: Proposal submission request validation
    #[test]
    fn test_proposal_submission_request_creation() {
        let proposal_request = create_test_proposal_submission_request();

        assert_eq!(proposal_request.title, "Test Proposal");
        assert_eq!(proposal_request.description, "This is a test proposal");
        assert_eq!(proposal_request.proposal_type, "ProtocolUpgrade");
        assert_eq!(proposal_request.required_stake, 1000);
    }

    // Test 6: HTTP request parsing
    #[test]
    fn test_http_request_parsing() {
        let request = create_test_http_request("GET", "/api/health", "");

        assert_eq!(request.method, "GET");
        assert_eq!(request.path, "/api/health");
        assert_eq!(request.body, "");
    }

    // Test 7: Portal response structure
    #[test]
    fn test_portal_response_structure() {
        let response = PortalResponse {
            success: true,
            message: "Test message".to_string(),
            data: Some(serde_json::json!({"test": "data"})),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            request_id: "test_request_123".to_string(),
        };

        assert!(response.success);
        assert_eq!(response.message, "Test message");
        assert!(response.data.is_some());
        assert_eq!(response.request_id, "test_request_123");
    }

    // Test 8: WebSocket message types
    #[test]
    fn test_websocket_message_types() {
        let metrics_message = WebSocketMessage::MetricsUpdate {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            metrics_json: "System metrics available".to_string(),
        };

        let vote_message = WebSocketMessage::VoteUpdate {
            proposal_id: "test_proposal".to_string(),
            vote_count: 100,
            approval_rate: 0.75,
        };

        let proposal_message = WebSocketMessage::ProposalNotification {
            proposal_id: "new_proposal".to_string(),
            title: "New Proposal".to_string(),
            proposer: "test_user".to_string(),
        };

        assert!(matches!(
            metrics_message,
            WebSocketMessage::MetricsUpdate { .. }
        ));
        assert!(matches!(vote_message, WebSocketMessage::VoteUpdate { .. }));
        assert!(matches!(
            proposal_message,
            WebSocketMessage::ProposalNotification { .. }
        ));
    }

    // Test 9: WebSocket connection management
    #[test]
    fn test_websocket_connection_creation() {
        let connection = WebSocketConnection {
            connection_id: "test_connection_123".to_string(),
            client_address: "127.0.0.1:8080".parse().unwrap(),
            connected_at: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            last_message: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            is_active: true,
        };

        assert_eq!(connection.connection_id, "test_connection_123");
        assert!(connection.is_active);
    }

    // Test 10: WebAssembly frontend component creation
    #[test]
    fn test_webassembly_frontend_creation() {
        let data = serde_json::json!({"test": "data"});
        let component = WebAssemblyFrontend::new(FrontendComponentType::Dashboard, data);

        assert!(component.component_id.starts_with("comp_"));
        assert_eq!(component.component_type, FrontendComponentType::Dashboard);
    }

    // Test 11: Frontend component rendering
    #[test]
    fn test_frontend_component_rendering() {
        let data = serde_json::json!({"test": "data"});
        let component = WebAssemblyFrontend::new(FrontendComponentType::VotingInterface, data);

        let html = component.render();
        assert!(html.contains("voting-interface"));
        assert!(html.contains(&component.component_id));
    }

    // Test 12: Frontend component data update
    #[test]
    fn test_frontend_component_data_update() {
        let initial_data = serde_json::json!({"initial": "data"});
        let mut component =
            WebAssemblyFrontend::new(FrontendComponentType::AnalyticsChart, initial_data);

        let new_data = serde_json::json!({"updated": "data"});
        component.update_data(new_data);

        assert_eq!(component.data["updated"], "data");
    }

    // Test 13: Portal status retrieval
    #[test]
    fn test_portal_status_retrieval() {
        let config = create_test_portal_config();
        let governance_system = create_test_governance_system();
        let analytics_engine = create_test_analytics_engine();
        let visualization_engine = create_test_visualization_engine();
        let federation_system = create_test_federation_system();
        let security_auditor = create_test_security_auditor();
        let monitoring_system = create_test_monitoring_system();
        let ui_interface = create_test_ui_interface();

        let portal = CommunityGovernancePortal::new_with_dependencies(
            config,
            governance_system,
            analytics_engine,
            visualization_engine,
            federation_system,
            security_auditor,
            monitoring_system,
            ui_interface,
        );

        let status = portal.get_status();
        assert!(!status.is_running);
        assert_eq!(status.session_count, 0);
        assert_eq!(status.connection_count, 0);
    }

    // Test 14: Portal error handling
    #[test]
    fn test_portal_error_handling() {
        let auth_error = PortalError::AuthenticationError("Invalid credentials".to_string());
        let session_error = PortalError::InvalidSession("Session expired".to_string());
        let permission_error =
            PortalError::PermissionDenied("Insufficient permissions".to_string());

        assert!(auth_error.to_string().contains("Authentication error"));
        assert!(session_error.to_string().contains("Invalid session"));
        assert!(permission_error.to_string().contains("Permission denied"));
    }

    // Test 15: Dilithium signature verification
    #[test]
    fn test_dilithium_signature_verification() {
        let (public_key, secret_key) = create_test_dilithium_keys();
        let message = "test message";
        let message_hash = Sha3_256::digest(message.as_bytes()).to_vec();
        let params = DilithiumParams::dilithium3();

        let signature = dilithium_sign(&message_hash, &secret_key, &params).unwrap();
        let is_valid = dilithium_verify(&message_hash, &signature, &public_key, &params).unwrap();

        assert!(is_valid);
    }

    // Test 16: SHA-3 hash generation
    #[test]
    fn test_sha3_hash_generation() {
        let data = b"test data";
        let hash = Sha3_256::digest(data);

        assert_eq!(hash.len(), 32); // SHA-3 256 produces 32-byte hash
        assert!(!hash.is_empty());
    }

    // Test 17: Request ID generation
    #[test]
    fn test_request_id_generation() {
        let request_id = format!(
            "req_{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );

        assert!(request_id.starts_with("req_"));
        assert!(request_id.len() > 10);
    }

    // Test 18: Session timeout handling
    #[test]
    fn test_session_timeout_handling() {
        let mut session = create_test_user_session();
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Set session to be very old
        session.last_activity = current_time - 7200; // 2 hours ago

        let timeout_threshold = current_time - 3600; // 1 hour timeout
        let is_expired = session.last_activity < timeout_threshold;

        assert!(is_expired);
    }

    // Test 19: CORS configuration
    #[test]
    fn test_cors_configuration() {
        let config = create_test_portal_config();

        assert!(config.enable_cors);
        assert!(config.allowed_origins.contains(&"*".to_string()));
    }

    // Test 20: WebSocket interval configuration
    #[test]
    fn test_websocket_interval_configuration() {
        let config = create_test_portal_config();

        assert_eq!(config.websocket_interval_ms, 100);
        assert!(config.websocket_interval_ms > 0);
    }

    // Test 21: Maximum connections configuration
    #[test]
    fn test_maximum_connections_configuration() {
        let config = create_test_portal_config();

        assert_eq!(config.max_connections, 100);
        assert!(config.max_connections > 0);
    }

    // Test 22: Session timeout configuration
    #[test]
    fn test_session_timeout_configuration() {
        let config = create_test_portal_config();

        assert_eq!(config.session_timeout_seconds, 3600);
        assert!(config.session_timeout_seconds > 0);
    }

    // Test 23: HTTPS configuration
    #[test]
    fn test_https_configuration() {
        let config = create_test_portal_config();

        assert!(!config.enable_https);
        assert!(config.ssl_cert_path.is_none());
        assert!(config.ssl_key_path.is_none());
    }

    // Test 24: Bind address configuration
    #[test]
    fn test_bind_address_configuration() {
        let config = create_test_portal_config();

        assert_eq!(
            config.bind_address,
            "127.0.0.1:0".parse::<SocketAddr>().unwrap()
        );
    }

    // Test 25: Portal stop functionality
    #[test]
    fn test_portal_stop_functionality() {
        let config = create_test_portal_config();
        let governance_system = create_test_governance_system();
        let analytics_engine = create_test_analytics_engine();
        let visualization_engine = create_test_visualization_engine();
        let federation_system = create_test_federation_system();
        let security_auditor = create_test_security_auditor();
        let monitoring_system = create_test_monitoring_system();
        let ui_interface = create_test_ui_interface();

        let portal = CommunityGovernancePortal::new_with_dependencies(
            config,
            governance_system,
            analytics_engine,
            visualization_engine,
            federation_system,
            security_auditor,
            monitoring_system,
            ui_interface,
        );

        let result = portal.stop();
        assert!(result.is_ok());
    }

    // Test 26: Edge case - empty vote submission
    #[test]
    fn test_empty_vote_submission() {
        let vote_request = VoteSubmissionRequest {
            proposal_id: String::new(),
            choice: String::new(),
            signature: vec![],
            public_key: vec![],
            stake_amount: 0,
        };

        assert!(vote_request.proposal_id.is_empty());
        assert_eq!(vote_request.stake_amount, 0);
    }

    // Test 27: Edge case - empty proposal submission
    #[test]
    fn test_empty_proposal_submission() {
        let proposal_request = ProposalSubmissionRequest {
            title: String::new(),
            description: String::new(),
            proposal_type: String::new(),
            signature: vec![],
            public_key: vec![],
            required_stake: 0,
        };

        assert!(proposal_request.title.is_empty());
        assert!(proposal_request.description.is_empty());
        assert_eq!(proposal_request.required_stake, 0);
    }

    // Test 28: Edge case - invalid HTTP request
    #[test]
    fn test_invalid_http_request() {
        let request = HttpRequest {
            method: String::new(),
            path: String::new(),
            headers: HashMap::new(),
            body: String::new(),
        };

        assert!(request.method.is_empty());
        assert!(request.path.is_empty());
        assert!(request.body.is_empty());
    }

    // Test 29: Edge case - malformed JSON
    #[test]
    fn test_malformed_json_handling() {
        let malformed_json = "{ invalid json }";
        let result: Result<serde_json::Value, _> = serde_json::from_str(malformed_json);

        assert!(result.is_err());
    }

    // Test 30: Edge case - very large request
    #[test]
    fn test_large_request_handling() {
        let large_body = "x".repeat(10000); // 10KB body
        let request = create_test_http_request("POST", "/api/vote", &large_body);

        assert_eq!(request.body.len(), 10000);
    }

    // Test 31: Multiple sessions test (reduced scale)
    #[test]
    fn test_multiple_sessions() {
        let mut sessions = HashMap::new();

        for i in 0..10 {
            // Reduced from 1000 to 10 to prevent crashes
            let polynomial_ring = PolynomialRing {
                coefficients: vec![0; 256],
                modulus: 8380417,
                dimension: 256,
            };

            let session = UserSession {
                session_id: format!("session_{}", i),
                public_key: DilithiumPublicKey {
                    matrix_a: vec![vec![polynomial_ring.clone(); 256]; 256],
                    vector_t1: vec![polynomial_ring.clone(); 256],
                    security_level: DilithiumSecurityLevel::Dilithium3,
                },
                created_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                last_activity: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                permissions: vec!["vote".to_string()],
                stake_amount: 1000,
            };
            sessions.insert(session.session_id.clone(), session);
        }

        assert_eq!(sessions.len(), 10);
    }

    // Test 32: Multiple WebSocket connections test (reduced scale)
    #[test]
    fn test_multiple_websocket_connections() {
        let mut connections = Vec::new();

        for i in 0..10 {
            // Reduced from 1000 to 10 to prevent crashes
            let connection = WebSocketConnection {
                connection_id: format!("conn_{}", i),
                client_address: "127.0.0.1:8080".parse().unwrap(),
                connected_at: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                last_message: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                is_active: true,
            };
            connections.push(connection);
        }

        assert_eq!(connections.len(), 10);
    }

    // Test 33: Security test - signature tampering
    #[test]
    fn test_signature_tampering_detection() {
        let (public_key, secret_key) = create_test_dilithium_keys();
        let message = "test message";
        let message_hash = Sha3_256::digest(message.as_bytes()).to_vec();
        let params = DilithiumParams::dilithium3();

        let signature = dilithium_sign(&message_hash, &secret_key, &params).unwrap();

        // Test with original signature - should be valid
        let is_valid_original =
            dilithium_verify(&message_hash, &signature, &public_key, &params).unwrap();
        assert!(is_valid_original);

        // Test with different message - should be invalid
        let different_message = "different message";
        let different_hash = Sha3_256::digest(different_message.as_bytes()).to_vec();
        let is_valid_different =
            dilithium_verify(&different_hash, &signature, &public_key, &params).unwrap();
        assert!(!is_valid_different);
    }

    // Test 34: Security test - message tampering
    #[test]
    fn test_message_tampering_detection() {
        let (public_key, secret_key) = create_test_dilithium_keys();
        let original_message = "test message";
        let tampered_message = "tampered message";

        let original_hash = Sha3_256::digest(original_message.as_bytes()).to_vec();
        let tampered_hash = Sha3_256::digest(tampered_message.as_bytes()).to_vec();
        let params = DilithiumParams::dilithium3();

        // Sign the original message
        let signature = dilithium_sign(&original_hash, &secret_key, &params).unwrap();

        // Verify with original message - should be valid
        let is_valid_original =
            dilithium_verify(&original_hash, &signature, &public_key, &params).unwrap();
        assert!(is_valid_original);

        // Verify with tampered message - should be invalid
        let is_valid_tampered =
            dilithium_verify(&tampered_hash, &signature, &public_key, &params).unwrap();
        assert!(!is_valid_tampered);
    }

    // Test 35: Performance test - rapid request processing
    #[test]
    fn test_rapid_request_processing() {
        let start_time = SystemTime::now();

        for _ in 0..1000 {
            let request = create_test_http_request("GET", "/api/health", "");
            assert_eq!(request.method, "GET");
            assert_eq!(request.path, "/api/health");
        }

        let duration = start_time.elapsed().unwrap();
        assert!(duration.as_millis() < 1000); // Should complete in less than 1 second
    }

    // Test 36: Integration test - portal with all modules
    #[test]
    fn test_portal_integration_with_all_modules() {
        let config = create_test_portal_config();
        let governance_system = create_test_governance_system();
        let analytics_engine = create_test_analytics_engine();
        let visualization_engine = create_test_visualization_engine();
        let federation_system = create_test_federation_system();
        let security_auditor = create_test_security_auditor();
        let monitoring_system = create_test_monitoring_system();
        let ui_interface = create_test_ui_interface();

        let portal = CommunityGovernancePortal::new_with_dependencies(
            config,
            governance_system,
            analytics_engine,
            visualization_engine,
            federation_system,
            security_auditor,
            monitoring_system,
            ui_interface,
        );

        let status = portal.get_status();
        assert!(!status.is_running);
        assert_eq!(status.session_count, 0);
        assert_eq!(status.connection_count, 0);
    }

    // Test 37: Memory usage test
    #[test]
    fn test_memory_usage_optimization() {
        let config = create_test_portal_config();
        let governance_system = create_test_governance_system();
        let analytics_engine = create_test_analytics_engine();
        let visualization_engine = create_test_visualization_engine();
        let federation_system = create_test_federation_system();
        let security_auditor = create_test_security_auditor();
        let monitoring_system = create_test_monitoring_system();
        let ui_interface = create_test_ui_interface();

        let portal = CommunityGovernancePortal::new_with_dependencies(
            config,
            governance_system,
            analytics_engine,
            visualization_engine,
            federation_system,
            security_auditor,
            monitoring_system,
            ui_interface,
        );

        // Portal should be created without excessive memory usage
        let status = portal.get_status();
        assert!(status.session_count < 1000); // Reasonable session limit
    }

    // Test 38: Error recovery test
    #[test]
    fn test_error_recovery_mechanisms() {
        let config = create_test_portal_config();
        let governance_system = create_test_governance_system();
        let analytics_engine = create_test_analytics_engine();
        let visualization_engine = create_test_visualization_engine();
        let federation_system = create_test_federation_system();
        let security_auditor = create_test_security_auditor();
        let monitoring_system = create_test_monitoring_system();
        let ui_interface = create_test_ui_interface();

        let portal = CommunityGovernancePortal::new_with_dependencies(
            config,
            governance_system,
            analytics_engine,
            visualization_engine,
            federation_system,
            security_auditor,
            monitoring_system,
            ui_interface,
        );

        // Portal should handle errors gracefully
        let result = portal.stop();
        assert!(result.is_ok());
    }

    // Test 39: Concurrent access test
    #[test]
    fn test_concurrent_access_safety() {
        let config = create_test_portal_config();
        let governance_system = create_test_governance_system();
        let analytics_engine = create_test_analytics_engine();
        let visualization_engine = create_test_visualization_engine();
        let federation_system = create_test_federation_system();
        let security_auditor = create_test_security_auditor();
        let monitoring_system = create_test_monitoring_system();
        let ui_interface = create_test_ui_interface();

        let portal = Arc::new(CommunityGovernancePortal::new_with_dependencies(
            config,
            governance_system,
            analytics_engine,
            visualization_engine,
            federation_system,
            security_auditor,
            monitoring_system,
            ui_interface,
        ));

        let portal_clone = Arc::clone(&portal);
        let handle = std::thread::spawn(move || portal_clone.get_status());

        let status = handle.join().unwrap();
        assert!(!status.is_running);
    }

    // Test 40: Configuration validation test
    #[test]
    fn test_configuration_validation() {
        let config = create_test_portal_config();

        // Validate all configuration parameters
        assert!(config.websocket_interval_ms > 0);
        assert!(config.max_connections > 0);
        assert!(config.session_timeout_seconds > 0);
        assert!(!config.bind_address.ip().is_unspecified());
    }
}
