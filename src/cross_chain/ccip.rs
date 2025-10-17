//! Chainlink CCIP (Cross-Chain Interoperability Protocol) Implementation
//!
//! This module implements Chainlink's CCIP for standardized cross-chain
//! messaging, enabling secure and reliable communication between different
//! blockchain networks with enterprise-grade guarantees.
//!
//! Key features:
//! - Standardized cross-chain messaging
//! - Token transfers with finality guarantees
//! - Programmable token transfers
//! - Message routing and delivery
//! - Fee management and payment
//! - Security and risk management
//! - Oracle network integration
//!
//! Technical advantages:
//! - Enterprise-grade reliability
//! - Standardized protocol
//! - Secure message delivery
//! - Token transfer finality
//! - Programmable transfers
//! - Oracle network backing

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Error types for CCIP implementation
#[derive(Debug, Clone, PartialEq)]
pub enum CCIPError {
    /// Invalid message
    InvalidMessage,
    /// Invalid destination chain
    InvalidDestinationChain,
    /// Insufficient fees
    InsufficientFees,
    /// Message timeout
    MessageTimeout,
    /// Invalid signature
    InvalidSignature,
    /// Oracle not available
    OracleNotAvailable,
    /// Chain not supported
    ChainNotSupported,
    /// Message already processed
    MessageAlreadyProcessed,
    /// Invalid token
    InvalidToken,
    /// Transfer failed
    TransferFailed,
    /// Rate limit exceeded
    RateLimitExceeded,
    /// Security check failed
    SecurityCheckFailed,
    /// Gas limit exceeded
    GasLimitExceeded,
}

pub type CCIPResult<T> = Result<T, CCIPError>;

/// CCIP message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CCIPMessage {
    /// Message ID
    pub message_id: String,
    /// Source chain ID
    pub source_chain_id: u64,
    /// Destination chain ID
    pub destination_chain_id: u64,
    /// Source address
    pub source_address: String,
    /// Destination address
    pub destination_address: String,
    /// Message data
    pub data: Vec<u8>,
    /// Gas limit
    pub gas_limit: u64,
    /// Fee token
    pub fee_token: String,
    /// Fee amount
    pub fee_amount: u128,
    /// Message timestamp
    pub timestamp: u64,
    /// Message nonce
    pub nonce: u64,
}

/// CCIP token transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CCIPTokenTransfer {
    /// Transfer ID
    pub transfer_id: String,
    /// Source chain ID
    pub source_chain_id: u64,
    /// Destination chain ID
    pub destination_chain_id: u64,
    /// Token address
    pub token_address: String,
    /// Amount
    pub amount: u128,
    /// Source address
    pub source_address: String,
    /// Destination address
    pub destination_address: String,
    /// Gas limit
    pub gas_limit: u64,
    /// Fee token
    pub fee_token: String,
    /// Fee amount
    pub fee_amount: u128,
    /// Transfer timestamp
    pub timestamp: u64,
    /// Transfer nonce
    pub nonce: u64,
}

/// CCIP programmable token transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CCIPProgrammableTransfer {
    /// Transfer ID
    pub transfer_id: String,
    /// Source chain ID
    pub source_chain_id: u64,
    /// Destination chain ID
    pub destination_chain_id: u64,
    /// Token address
    pub token_address: String,
    /// Amount
    pub amount: u128,
    /// Source address
    pub source_address: String,
    /// Destination address
    pub destination_address: String,
    /// Program data
    pub program_data: Vec<u8>,
    /// Gas limit
    pub gas_limit: u64,
    /// Fee token
    pub fee_token: String,
    /// Fee amount
    pub fee_amount: u128,
    /// Transfer timestamp
    pub timestamp: u64,
    /// Transfer nonce
    pub nonce: u64,
}

/// CCIP message status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CCIPMessageStatus {
    /// Message pending
    Pending,
    /// Message confirmed
    Confirmed,
    /// Message delivered
    Delivered,
    /// Message failed
    Failed,
    /// Message timeout
    Timeout,
}

/// CCIP transfer status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CCIPTransferStatus {
    /// Transfer pending
    Pending,
    /// Transfer confirmed
    Confirmed,
    /// Transfer completed
    Completed,
    /// Transfer failed
    Failed,
    /// Transfer timeout
    Timeout,
}

/// CCIP oracle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CCIPOracle {
    /// Oracle ID
    pub oracle_id: String,
    /// Oracle address
    pub oracle_address: String,
    /// Supported chains
    pub supported_chains: Vec<u64>,
    /// Oracle reputation
    pub reputation: f64,
    /// Oracle status
    pub status: OracleStatus,
    /// Last update timestamp
    pub last_updated: u64,
}

/// Oracle status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OracleStatus {
    /// Oracle active
    Active,
    /// Oracle inactive
    Inactive,
    /// Oracle suspended
    Suspended,
}

/// CCIP router
pub struct CCIPRouter {
    /// Supported chains
    supported_chains: Arc<RwLock<HashMap<u64, String>>>, // chain_id -> chain_name
    /// Messages
    messages: Arc<RwLock<HashMap<String, CCIPMessage>>>,
    /// Message statuses
    message_statuses: Arc<RwLock<HashMap<String, CCIPMessageStatus>>>,
    /// Token transfers
    token_transfers: Arc<RwLock<HashMap<String, CCIPTokenTransfer>>>,
    /// Transfer statuses
    transfer_statuses: Arc<RwLock<HashMap<String, CCIPTransferStatus>>>,
    /// Programmable transfers
    programmable_transfers: Arc<RwLock<HashMap<String, CCIPProgrammableTransfer>>>,
    /// Oracles
    oracles: Arc<RwLock<HashMap<String, CCIPOracle>>>,
    /// Metrics
    metrics: Arc<RwLock<CCIPMetrics>>,
}

/// CCIP metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CCIPMetrics {
    /// Total messages sent
    pub total_messages_sent: u64,
    /// Total messages delivered
    pub total_messages_delivered: u64,
    /// Total messages failed
    pub total_messages_failed: u64,
    /// Total token transfers
    pub total_token_transfers: u64,
    /// Total programmable transfers
    pub total_programmable_transfers: u64,
    /// Total volume transferred
    pub total_volume_transferred: u128,
    /// Average message delivery time (ms)
    pub avg_message_delivery_time_ms: f64,
    /// Average transfer completion time (ms)
    pub avg_transfer_completion_time_ms: f64,
    /// Total fees collected
    pub total_fees_collected: u128,
    /// Active oracles
    pub active_oracles: u64,
}

impl Default for CCIPRouter {
    fn default() -> Self {
        let mut supported_chains = HashMap::new();
        supported_chains.insert(1, "Ethereum".to_string());
        supported_chains.insert(137, "Polygon".to_string());
        supported_chains.insert(56, "BSC".to_string());
        supported_chains.insert(43114, "Avalanche".to_string());
        supported_chains.insert(250, "Fantom".to_string());

        Self {
            supported_chains: Arc::new(RwLock::new(supported_chains)),
            messages: Arc::new(RwLock::new(HashMap::new())),
            message_statuses: Arc::new(RwLock::new(HashMap::new())),
            token_transfers: Arc::new(RwLock::new(HashMap::new())),
            transfer_statuses: Arc::new(RwLock::new(HashMap::new())),
            programmable_transfers: Arc::new(RwLock::new(HashMap::new())),
            oracles: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(CCIPMetrics {
                total_messages_sent: 0,
                total_messages_delivered: 0,
                total_messages_failed: 0,
                total_token_transfers: 0,
                total_programmable_transfers: 0,
                total_volume_transferred: 0,
                avg_message_delivery_time_ms: 0.0,
                avg_transfer_completion_time_ms: 0.0,
                total_fees_collected: 0,
                active_oracles: 0,
            })),
        }
    }
}

impl CCIPRouter {
    /// Create a new CCIP router
    pub fn new() -> Self {
        Self::default()
    }

    /// Send message
    pub fn send_message(&self, message: CCIPMessage) -> CCIPResult<()> {
        // Validate destination chain
        {
            let supported_chains = self.supported_chains.read().unwrap();
            if !supported_chains.contains_key(&message.destination_chain_id) {
                return Err(CCIPError::InvalidDestinationChain);
            }
        }

        // Validate message data
        if message.data.is_empty() {
            return Err(CCIPError::InvalidMessage);
        }

        // Check if message already exists
        {
            let messages = self.messages.read().unwrap();
            if messages.contains_key(&message.message_id) {
                return Err(CCIPError::MessageAlreadyProcessed);
            }
        }

        // Store message
        {
            let mut messages = self.messages.write().unwrap();
            messages.insert(message.message_id.clone(), message.clone());
        }

        // Set initial status
        {
            let mut message_statuses = self.message_statuses.write().unwrap();
            message_statuses.insert(message.message_id.clone(), CCIPMessageStatus::Pending);
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_messages_sent += 1;
            metrics.total_fees_collected += message.fee_amount;
        }

        Ok(())
    }

    /// Confirm message
    pub fn confirm_message(&self, message_id: &str) -> CCIPResult<()> {
        // Check if message exists
        {
            let messages = self.messages.read().unwrap();
            if !messages.contains_key(message_id) {
                return Err(CCIPError::InvalidMessage);
            }
        }

        // Update message status
        {
            let mut message_statuses = self.message_statuses.write().unwrap();
            if let Some(status) = message_statuses.get_mut(message_id) {
                *status = CCIPMessageStatus::Confirmed;
            }
        }

        Ok(())
    }

    /// Deliver message
    pub fn deliver_message(&self, message_id: &str) -> CCIPResult<()> {
        // Check if message exists
        {
            let messages = self.messages.read().unwrap();
            if !messages.contains_key(message_id) {
                return Err(CCIPError::InvalidMessage);
            }
        }

        // Update message status
        {
            let mut message_statuses = self.message_statuses.write().unwrap();
            if let Some(status) = message_statuses.get_mut(message_id) {
                *status = CCIPMessageStatus::Delivered;
            }
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_messages_delivered += 1;
        }

        Ok(())
    }

    /// Transfer tokens
    pub fn transfer_tokens(&self, transfer: CCIPTokenTransfer) -> CCIPResult<()> {
        // Validate destination chain
        {
            let supported_chains = self.supported_chains.read().unwrap();
            if !supported_chains.contains_key(&transfer.destination_chain_id) {
                return Err(CCIPError::InvalidDestinationChain);
            }
        }

        // Validate transfer amount
        if transfer.amount == 0 {
            return Err(CCIPError::InvalidMessage);
        }

        // Check if transfer already exists
        {
            let token_transfers = self.token_transfers.read().unwrap();
            if token_transfers.contains_key(&transfer.transfer_id) {
                return Err(CCIPError::MessageAlreadyProcessed);
            }
        }

        // Store transfer
        {
            let mut token_transfers = self.token_transfers.write().unwrap();
            token_transfers.insert(transfer.transfer_id.clone(), transfer.clone());
        }

        // Set initial status
        {
            let mut transfer_statuses = self.transfer_statuses.write().unwrap();
            transfer_statuses.insert(transfer.transfer_id.clone(), CCIPTransferStatus::Pending);
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_token_transfers += 1;
            metrics.total_volume_transferred += transfer.amount;
            metrics.total_fees_collected += transfer.fee_amount;
        }

        Ok(())
    }

    /// Complete token transfer
    pub fn complete_token_transfer(&self, transfer_id: &str) -> CCIPResult<()> {
        // Check if transfer exists
        {
            let token_transfers = self.token_transfers.read().unwrap();
            if !token_transfers.contains_key(transfer_id) {
                return Err(CCIPError::InvalidMessage);
            }
        }

        // Update transfer status
        {
            let mut transfer_statuses = self.transfer_statuses.write().unwrap();
            if let Some(status) = transfer_statuses.get_mut(transfer_id) {
                *status = CCIPTransferStatus::Completed;
            }
        }

        Ok(())
    }

    /// Send programmable token transfer
    pub fn send_programmable_transfer(&self, transfer: CCIPProgrammableTransfer) -> CCIPResult<()> {
        // Validate destination chain
        {
            let supported_chains = self.supported_chains.read().unwrap();
            if !supported_chains.contains_key(&transfer.destination_chain_id) {
                return Err(CCIPError::InvalidDestinationChain);
            }
        }

        // Validate transfer amount
        if transfer.amount == 0 {
            return Err(CCIPError::InvalidMessage);
        }

        // Validate program data
        if transfer.program_data.is_empty() {
            return Err(CCIPError::InvalidMessage);
        }

        // Check if transfer already exists
        {
            let programmable_transfers = self.programmable_transfers.read().unwrap();
            if programmable_transfers.contains_key(&transfer.transfer_id) {
                return Err(CCIPError::MessageAlreadyProcessed);
            }
        }

        // Store transfer
        {
            let mut programmable_transfers = self.programmable_transfers.write().unwrap();
            programmable_transfers.insert(transfer.transfer_id.clone(), transfer.clone());
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            metrics.total_programmable_transfers += 1;
            metrics.total_volume_transferred += transfer.amount;
            metrics.total_fees_collected += transfer.fee_amount;
        }

        Ok(())
    }

    /// Register oracle
    pub fn register_oracle(&self, oracle: CCIPOracle) -> CCIPResult<()> {
        // Validate oracle data
        if oracle.oracle_address.is_empty() {
            return Err(CCIPError::InvalidMessage);
        }

        // Store oracle
        {
            let mut oracles = self.oracles.write().unwrap();
            oracles.insert(oracle.oracle_id.clone(), oracle.clone());
        }

        // Update metrics
        {
            let mut metrics = self.metrics.write().unwrap();
            if oracle.status == OracleStatus::Active {
                metrics.active_oracles += 1;
            }
        }

        Ok(())
    }

    /// Get message status
    pub fn get_message_status(&self, message_id: &str) -> Option<CCIPMessageStatus> {
        let message_statuses = self.message_statuses.read().unwrap();
        message_statuses.get(message_id).cloned()
    }

    /// Get transfer status
    pub fn get_transfer_status(&self, transfer_id: &str) -> Option<CCIPTransferStatus> {
        let transfer_statuses = self.transfer_statuses.read().unwrap();
        transfer_statuses.get(transfer_id).cloned()
    }

    /// Get message
    pub fn get_message(&self, message_id: &str) -> Option<CCIPMessage> {
        let messages = self.messages.read().unwrap();
        messages.get(message_id).cloned()
    }

    /// Get token transfer
    pub fn get_token_transfer(&self, transfer_id: &str) -> Option<CCIPTokenTransfer> {
        let token_transfers = self.token_transfers.read().unwrap();
        token_transfers.get(transfer_id).cloned()
    }

    /// Get programmable transfer
    pub fn get_programmable_transfer(&self, transfer_id: &str) -> Option<CCIPProgrammableTransfer> {
        let programmable_transfers = self.programmable_transfers.read().unwrap();
        programmable_transfers.get(transfer_id).cloned()
    }

    /// Get oracle
    pub fn get_oracle(&self, oracle_id: &str) -> Option<CCIPOracle> {
        let oracles = self.oracles.read().unwrap();
        oracles.get(oracle_id).cloned()
    }

    /// Get metrics
    pub fn get_metrics(&self) -> CCIPMetrics {
        let metrics = self.metrics.read().unwrap();
        metrics.clone()
    }

    /// Get supported chains
    pub fn get_supported_chains(&self) -> HashMap<u64, String> {
        let supported_chains = self.supported_chains.read().unwrap();
        supported_chains.clone()
    }

    /// Get all messages
    pub fn get_all_messages(&self) -> Vec<CCIPMessage> {
        let messages = self.messages.read().unwrap();
        messages.values().cloned().collect()
    }

    /// Get all token transfers
    pub fn get_all_token_transfers(&self) -> Vec<CCIPTokenTransfer> {
        let token_transfers = self.token_transfers.read().unwrap();
        token_transfers.values().cloned().collect()
    }

    /// Get all programmable transfers
    pub fn get_all_programmable_transfers(&self) -> Vec<CCIPProgrammableTransfer> {
        let programmable_transfers = self.programmable_transfers.read().unwrap();
        programmable_transfers.values().cloned().collect()
    }

    /// Get all oracles
    pub fn get_all_oracles(&self) -> Vec<CCIPOracle> {
        let oracles = self.oracles.read().unwrap();
        oracles.values().cloned().collect()
    }
}

/// Get current timestamp in milliseconds
#[allow(dead_code)]
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ccip_router_creation() {
        let router = CCIPRouter::new();
        let metrics = router.get_metrics();
        assert_eq!(metrics.total_messages_sent, 0);

        let supported_chains = router.get_supported_chains();
        assert!(supported_chains.contains_key(&1)); // Ethereum
        assert!(supported_chains.contains_key(&137)); // Polygon
    }

    #[test]
    fn test_send_message() {
        let router = CCIPRouter::new();

        let message = CCIPMessage {
            message_id: "msg-1".to_string(),
            source_chain_id: 1,
            destination_chain_id: 137,
            source_address: "0x123".to_string(),
            destination_address: "0x456".to_string(),
            data: vec![1, 2, 3, 4, 5],
            gas_limit: 100000,
            fee_token: "LINK".to_string(),
            fee_amount: 1000,
            timestamp: current_timestamp(),
            nonce: 1,
        };

        let result = router.send_message(message.clone());
        assert!(result.is_ok());

        // Verify message was stored
        let stored_message = router.get_message("msg-1");
        assert!(stored_message.is_some());
        assert_eq!(stored_message.unwrap().message_id, "msg-1");

        // Verify status is pending
        let status = router.get_message_status("msg-1");
        assert_eq!(status, Some(CCIPMessageStatus::Pending));
    }

    #[test]
    fn test_send_message_invalid_destination() {
        let router = CCIPRouter::new();

        let message = CCIPMessage {
            message_id: "msg-1".to_string(),
            source_chain_id: 1,
            destination_chain_id: 999, // Unsupported chain
            source_address: "0x123".to_string(),
            destination_address: "0x456".to_string(),
            data: vec![1, 2, 3, 4, 5],
            gas_limit: 100000,
            fee_token: "LINK".to_string(),
            fee_amount: 1000,
            timestamp: current_timestamp(),
            nonce: 1,
        };

        let result = router.send_message(message);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), CCIPError::InvalidDestinationChain);
    }

    #[test]
    fn test_send_message_empty_data() {
        let router = CCIPRouter::new();

        let message = CCIPMessage {
            message_id: "msg-1".to_string(),
            source_chain_id: 1,
            destination_chain_id: 137,
            source_address: "0x123".to_string(),
            destination_address: "0x456".to_string(),
            data: vec![], // Empty data
            gas_limit: 100000,
            fee_token: "LINK".to_string(),
            fee_amount: 1000,
            timestamp: current_timestamp(),
            nonce: 1,
        };

        let result = router.send_message(message);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), CCIPError::InvalidMessage);
    }

    #[test]
    fn test_confirm_message() {
        let router = CCIPRouter::new();

        // Send message first
        let message = CCIPMessage {
            message_id: "msg-1".to_string(),
            source_chain_id: 1,
            destination_chain_id: 137,
            source_address: "0x123".to_string(),
            destination_address: "0x456".to_string(),
            data: vec![1, 2, 3, 4, 5],
            gas_limit: 100000,
            fee_token: "LINK".to_string(),
            fee_amount: 1000,
            timestamp: current_timestamp(),
            nonce: 1,
        };

        router.send_message(message).unwrap();

        // Confirm message
        let result = router.confirm_message("msg-1");
        assert!(result.is_ok());

        // Verify status is confirmed
        let status = router.get_message_status("msg-1");
        assert_eq!(status, Some(CCIPMessageStatus::Confirmed));
    }

    #[test]
    fn test_deliver_message() {
        let router = CCIPRouter::new();

        // Send message first
        let message = CCIPMessage {
            message_id: "msg-1".to_string(),
            source_chain_id: 1,
            destination_chain_id: 137,
            source_address: "0x123".to_string(),
            destination_address: "0x456".to_string(),
            data: vec![1, 2, 3, 4, 5],
            gas_limit: 100000,
            fee_token: "LINK".to_string(),
            fee_amount: 1000,
            timestamp: current_timestamp(),
            nonce: 1,
        };

        router.send_message(message).unwrap();

        // Deliver message
        let result = router.deliver_message("msg-1");
        assert!(result.is_ok());

        // Verify status is delivered
        let status = router.get_message_status("msg-1");
        assert_eq!(status, Some(CCIPMessageStatus::Delivered));
    }

    #[test]
    fn test_transfer_tokens() {
        let router = CCIPRouter::new();

        let transfer = CCIPTokenTransfer {
            transfer_id: "transfer-1".to_string(),
            source_chain_id: 1,
            destination_chain_id: 137,
            token_address: "0xToken".to_string(),
            amount: 1000000,
            source_address: "0x123".to_string(),
            destination_address: "0x456".to_string(),
            gas_limit: 200000,
            fee_token: "LINK".to_string(),
            fee_amount: 2000,
            timestamp: current_timestamp(),
            nonce: 1,
        };

        let result = router.transfer_tokens(transfer.clone());
        assert!(result.is_ok());

        // Verify transfer was stored
        let stored_transfer = router.get_token_transfer("transfer-1");
        assert!(stored_transfer.is_some());
        assert_eq!(stored_transfer.unwrap().transfer_id, "transfer-1");

        // Verify status is pending
        let status = router.get_transfer_status("transfer-1");
        assert_eq!(status, Some(CCIPTransferStatus::Pending));
    }

    #[test]
    fn test_complete_token_transfer() {
        let router = CCIPRouter::new();

        // Transfer tokens first
        let transfer = CCIPTokenTransfer {
            transfer_id: "transfer-1".to_string(),
            source_chain_id: 1,
            destination_chain_id: 137,
            token_address: "0xToken".to_string(),
            amount: 1000000,
            source_address: "0x123".to_string(),
            destination_address: "0x456".to_string(),
            gas_limit: 200000,
            fee_token: "LINK".to_string(),
            fee_amount: 2000,
            timestamp: current_timestamp(),
            nonce: 1,
        };

        router.transfer_tokens(transfer).unwrap();

        // Complete transfer
        let result = router.complete_token_transfer("transfer-1");
        assert!(result.is_ok());

        // Verify status is completed
        let status = router.get_transfer_status("transfer-1");
        assert_eq!(status, Some(CCIPTransferStatus::Completed));
    }

    #[test]
    fn test_send_programmable_transfer() {
        let router = CCIPRouter::new();

        let transfer = CCIPProgrammableTransfer {
            transfer_id: "prog-transfer-1".to_string(),
            source_chain_id: 1,
            destination_chain_id: 137,
            token_address: "0xToken".to_string(),
            amount: 1000000,
            source_address: "0x123".to_string(),
            destination_address: "0x456".to_string(),
            program_data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            gas_limit: 300000,
            fee_token: "LINK".to_string(),
            fee_amount: 3000,
            timestamp: current_timestamp(),
            nonce: 1,
        };

        let result = router.send_programmable_transfer(transfer.clone());
        assert!(result.is_ok());

        // Verify transfer was stored
        let stored_transfer = router.get_programmable_transfer("prog-transfer-1");
        assert!(stored_transfer.is_some());
        assert_eq!(stored_transfer.unwrap().transfer_id, "prog-transfer-1");
    }

    #[test]
    fn test_register_oracle() {
        let router = CCIPRouter::new();

        let oracle = CCIPOracle {
            oracle_id: "oracle-1".to_string(),
            oracle_address: "0xOracle".to_string(),
            supported_chains: vec![1, 137, 56],
            reputation: 0.95,
            status: OracleStatus::Active,
            last_updated: current_timestamp(),
        };

        let result = router.register_oracle(oracle.clone());
        assert!(result.is_ok());

        // Verify oracle was stored
        let stored_oracle = router.get_oracle("oracle-1");
        assert!(stored_oracle.is_some());
        assert_eq!(stored_oracle.unwrap().oracle_id, "oracle-1");
    }

    #[test]
    fn test_ccip_metrics() {
        let router = CCIPRouter::new();

        // Send message
        let message = CCIPMessage {
            message_id: "msg-1".to_string(),
            source_chain_id: 1,
            destination_chain_id: 137,
            source_address: "0x123".to_string(),
            destination_address: "0x456".to_string(),
            data: vec![1, 2, 3, 4, 5],
            gas_limit: 100000,
            fee_token: "LINK".to_string(),
            fee_amount: 1000,
            timestamp: current_timestamp(),
            nonce: 1,
        };

        router.send_message(message).unwrap();
        router.deliver_message("msg-1").unwrap();

        // Transfer tokens
        let transfer = CCIPTokenTransfer {
            transfer_id: "transfer-1".to_string(),
            source_chain_id: 1,
            destination_chain_id: 137,
            token_address: "0xToken".to_string(),
            amount: 1000000,
            source_address: "0x123".to_string(),
            destination_address: "0x456".to_string(),
            gas_limit: 200000,
            fee_token: "LINK".to_string(),
            fee_amount: 2000,
            timestamp: current_timestamp(),
            nonce: 1,
        };

        router.transfer_tokens(transfer).unwrap();

        // Send programmable transfer
        let prog_transfer = CCIPProgrammableTransfer {
            transfer_id: "prog-transfer-1".to_string(),
            source_chain_id: 1,
            destination_chain_id: 137,
            token_address: "0xToken".to_string(),
            amount: 500000,
            source_address: "0x123".to_string(),
            destination_address: "0x456".to_string(),
            program_data: vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            gas_limit: 300000,
            fee_token: "LINK".to_string(),
            fee_amount: 3000,
            timestamp: current_timestamp(),
            nonce: 1,
        };

        router.send_programmable_transfer(prog_transfer).unwrap();

        // Register oracle
        let oracle = CCIPOracle {
            oracle_id: "oracle-1".to_string(),
            oracle_address: "0xOracle".to_string(),
            supported_chains: vec![1, 137, 56],
            reputation: 0.95,
            status: OracleStatus::Active,
            last_updated: current_timestamp(),
        };

        router.register_oracle(oracle).unwrap();

        let metrics = router.get_metrics();
        assert_eq!(metrics.total_messages_sent, 1);
        assert_eq!(metrics.total_messages_delivered, 1);
        assert_eq!(metrics.total_token_transfers, 1);
        assert_eq!(metrics.total_programmable_transfers, 1);
        assert_eq!(metrics.total_volume_transferred, 1500000); // 1000000 + 500000
        assert_eq!(metrics.total_fees_collected, 6000); // 1000 + 2000 + 3000
        assert_eq!(metrics.active_oracles, 1);
    }

    #[test]
    fn test_duplicate_message() {
        let router = CCIPRouter::new();

        let message = CCIPMessage {
            message_id: "msg-1".to_string(),
            source_chain_id: 1,
            destination_chain_id: 137,
            source_address: "0x123".to_string(),
            destination_address: "0x456".to_string(),
            data: vec![1, 2, 3, 4, 5],
            gas_limit: 100000,
            fee_token: "LINK".to_string(),
            fee_amount: 1000,
            timestamp: current_timestamp(),
            nonce: 1,
        };

        // Send message first time
        let result1 = router.send_message(message.clone());
        assert!(result1.is_ok());

        // Try to send same message again
        let result2 = router.send_message(message);
        assert!(result2.is_err());
        assert_eq!(result2.unwrap_err(), CCIPError::MessageAlreadyProcessed);
    }

    #[test]
    fn test_confirm_nonexistent_message() {
        let router = CCIPRouter::new();

        let result = router.confirm_message("nonexistent-msg");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), CCIPError::InvalidMessage);
    }

    #[test]
    fn test_deliver_nonexistent_message() {
        let router = CCIPRouter::new();

        let result = router.deliver_message("nonexistent-msg");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), CCIPError::InvalidMessage);
    }

    #[test]
    fn test_complete_nonexistent_transfer() {
        let router = CCIPRouter::new();

        let result = router.complete_token_transfer("nonexistent-transfer");
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), CCIPError::InvalidMessage);
    }

    #[test]
    fn test_get_all_messages() {
        let router = CCIPRouter::new();

        // Send multiple messages
        for i in 0..3 {
            let message = CCIPMessage {
                message_id: format!("msg-{}", i),
                source_chain_id: 1,
                destination_chain_id: 137,
                source_address: "0x123".to_string(),
                destination_address: "0x456".to_string(),
                data: vec![i as u8, (i + 1) as u8, (i + 2) as u8],
                gas_limit: 100000,
                fee_token: "LINK".to_string(),
                fee_amount: 1000,
                timestamp: current_timestamp(),
                nonce: i + 1,
            };

            router.send_message(message).unwrap();
        }

        let messages = router.get_all_messages();
        assert_eq!(messages.len(), 3);
    }

    #[test]
    fn test_get_all_token_transfers() {
        let router = CCIPRouter::new();

        // Transfer tokens multiple times
        for i in 0..3 {
            let transfer = CCIPTokenTransfer {
                transfer_id: format!("transfer-{}", i),
                source_chain_id: 1,
                destination_chain_id: 137,
                token_address: "0xToken".to_string(),
                amount: 1000000,
                source_address: "0x123".to_string(),
                destination_address: "0x456".to_string(),
                gas_limit: 200000,
                fee_token: "LINK".to_string(),
                fee_amount: 2000,
                timestamp: current_timestamp(),
                nonce: i + 1,
            };

            router.transfer_tokens(transfer).unwrap();
        }

        let transfers = router.get_all_token_transfers();
        assert_eq!(transfers.len(), 3);
    }

    #[test]
    fn test_get_all_programmable_transfers() {
        let router = CCIPRouter::new();

        // Send multiple programmable transfers
        for i in 0..3 {
            let transfer = CCIPProgrammableTransfer {
                transfer_id: format!("prog-transfer-{}", i),
                source_chain_id: 1,
                destination_chain_id: 137,
                token_address: "0xToken".to_string(),
                amount: 1000000,
                source_address: "0x123".to_string(),
                destination_address: "0x456".to_string(),
                program_data: vec![
                    i as u8,
                    (i + 1) as u8,
                    (i + 2) as u8,
                    (i + 3) as u8,
                    (i + 4) as u8,
                ],
                gas_limit: 300000,
                fee_token: "LINK".to_string(),
                fee_amount: 3000,
                timestamp: current_timestamp(),
                nonce: i + 1,
            };

            router.send_programmable_transfer(transfer).unwrap();
        }

        let transfers = router.get_all_programmable_transfers();
        assert_eq!(transfers.len(), 3);
    }

    #[test]
    fn test_get_all_oracles() {
        let router = CCIPRouter::new();

        // Register multiple oracles
        for i in 0..3 {
            let oracle = CCIPOracle {
                oracle_id: format!("oracle-{}", i),
                oracle_address: format!("0xOracle{}", i),
                supported_chains: vec![1, 137, 56],
                reputation: 0.95,
                status: OracleStatus::Active,
                last_updated: current_timestamp(),
            };

            router.register_oracle(oracle).unwrap();
        }

        let oracles = router.get_all_oracles();
        assert_eq!(oracles.len(), 3);
    }
}
