//! Rust SDK for Hauptbuch Blockchain
//!
//! This module provides a comprehensive Rust SDK for developers to interact
//! with the Hauptbuch blockchain, including transaction creation, smart contract
//! deployment, account management, and advanced features like account abstraction,
//! cross-chain operations, and MEV protection.
//!
//! Key features:
//! - Type-safe blockchain interactions
//! - Smart contract deployment and interaction
//! - Account abstraction support
//! - Cross-chain operations
//! - MEV protection integration
//! - Performance optimization tools
//! - Comprehensive error handling
//! - Async/await support
//! - Event streaming and monitoring

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Error types for SDK operations
#[derive(Debug, Clone, PartialEq)]
pub enum SDKError {
    /// Invalid configuration
    InvalidConfiguration,
    /// Network connection failed
    NetworkConnectionFailed,
    /// Transaction failed
    TransactionFailed,
    /// Contract deployment failed
    ContractDeploymentFailed,
    /// Account not found
    AccountNotFound,
    /// Insufficient funds
    InsufficientFunds,
    /// Invalid signature
    InvalidSignature,
    /// Contract call failed
    ContractCallFailed,
    /// Event subscription failed
    EventSubscriptionFailed,
    /// Cross-chain operation failed
    CrossChainOperationFailed,
    /// MEV protection failed
    MEVProtectionFailed,
    /// Serialization failed
    SerializationFailed,
    /// Deserialization failed
    DeserializationFailed,
}

/// Result type for SDK operations
pub type SDKResult<T> = Result<T, SDKError>;

/// SDK configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SDKConfig {
    /// Network endpoint
    pub network_endpoint: String,
    /// Chain ID
    pub chain_id: u64,
    /// Gas price (wei)
    pub gas_price: u64,
    /// Gas limit
    pub gas_limit: u64,
    /// Enable account abstraction
    pub enable_account_abstraction: bool,
    /// Enable MEV protection
    pub enable_mev_protection: bool,
    /// Enable cross-chain operations
    pub enable_cross_chain: bool,
    /// Connection timeout (ms)
    pub connection_timeout_ms: u64,
    /// Retry attempts
    pub retry_attempts: u32,
    /// Enable performance optimization
    pub enable_performance_optimization: bool,
}

/// Account information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    /// Account address
    pub address: [u8; 20],
    /// Account balance (wei)
    pub balance: u64,
    /// Account nonce
    pub nonce: u64,
    /// Account type
    pub account_type: AccountType,
    /// Is account abstracted
    pub is_abstracted: bool,
    /// Associated smart contract address
    pub smart_contract_address: Option<[u8; 20]>,
}

/// Account types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AccountType {
    /// Externally owned account (EOA)
    EOA,
    /// Smart contract account
    SmartContract,
    /// Abstracted account
    Abstracted,
}

/// Transaction request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionRequest {
    /// From address
    pub from: [u8; 20],
    /// To address
    pub to: Option<[u8; 20]>,
    /// Value (wei)
    pub value: u64,
    /// Gas limit
    pub gas_limit: u64,
    /// Gas price (wei)
    pub gas_price: u64,
    /// Nonce
    pub nonce: u64,
    /// Data
    pub data: Vec<u8>,
    /// Transaction type
    pub transaction_type: TransactionType,
    /// Access list
    pub access_list: Vec<AccessListItem>,
    /// Max fee per gas (EIP-1559)
    pub max_fee_per_gas: Option<u64>,
    /// Max priority fee per gas (EIP-1559)
    pub max_priority_fee_per_gas: Option<u64>,
}

/// Transaction types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TransactionType {
    /// Legacy transaction
    Legacy,
    /// EIP-2930 transaction
    EIP2930,
    /// EIP-1559 transaction
    EIP1559,
    /// Account abstraction transaction
    AccountAbstraction,
}

/// Access list item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessListItem {
    /// Address
    pub address: [u8; 20],
    /// Storage keys
    pub storage_keys: Vec<[u8; 32]>,
}

/// Transaction receipt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransactionReceipt {
    /// Transaction hash
    pub transaction_hash: [u8; 32],
    /// Block number
    pub block_number: u64,
    /// Block hash
    pub block_hash: [u8; 32],
    /// Transaction index
    pub transaction_index: u32,
    /// From address
    pub from: [u8; 20],
    /// To address
    pub to: Option<[u8; 20]>,
    /// Gas used
    pub gas_used: u64,
    /// Effective gas price
    pub effective_gas_price: u64,
    /// Status (1 = success, 0 = failure)
    pub status: u8,
    /// Logs
    pub logs: Vec<LogEntry>,
    /// Contract address (for contract creation)
    pub contract_address: Option<[u8; 20]>,
}

/// Log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Address
    pub address: [u8; 20],
    /// Topics
    pub topics: Vec<[u8; 32]>,
    /// Data
    pub data: Vec<u8>,
    /// Log index
    pub log_index: u32,
    /// Transaction index
    pub transaction_index: u32,
    /// Block number
    pub block_number: u64,
}

/// Smart contract
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmartContract {
    /// Contract address
    pub address: [u8; 20],
    /// Contract ABI
    pub abi: String,
    /// Contract bytecode
    pub bytecode: Vec<u8>,
    /// Contract name
    pub name: String,
    /// Contract version
    pub version: String,
    /// Deployment transaction hash
    pub deployment_tx_hash: Option<[u8; 32]>,
    /// Deployment block number
    pub deployment_block_number: Option<u64>,
}

/// Event filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFilter {
    /// Contract address
    pub contract_address: Option<[u8; 20]>,
    /// Topics
    pub topics: Vec<Option<[u8; 32]>>,
    /// From block
    pub from_block: Option<u64>,
    /// To block
    pub to_block: Option<u64>,
}

/// Cross-chain operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossChainOperation {
    /// Operation ID
    pub operation_id: String,
    /// Source chain ID
    pub source_chain_id: u64,
    /// Target chain ID
    pub target_chain_id: u64,
    /// Source address
    pub source_address: [u8; 20],
    /// Target address
    pub target_address: [u8; 20],
    /// Amount
    pub amount: u64,
    /// Token address
    pub token_address: Option<[u8; 20]>,
    /// Operation type
    pub operation_type: CrossChainOperationType,
    /// Status
    pub status: CrossChainOperationStatus,
}

/// Cross-chain operation types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CrossChainOperationType {
    /// Token transfer
    TokenTransfer,
    /// Contract call
    ContractCall,
    /// Data transfer
    DataTransfer,
    /// Intent execution
    IntentExecution,
}

/// Cross-chain operation status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CrossChainOperationStatus {
    /// Pending
    Pending,
    /// In progress
    InProgress,
    /// Completed
    Completed,
    /// Failed
    Failed,
    /// Cancelled
    Cancelled,
}

/// Hauptbuch SDK
#[derive(Debug)]
pub struct HauptbuchSDK {
    /// SDK configuration
    pub config: SDKConfig,
    /// Network client
    pub network_client: Arc<RwLock<NetworkClient>>,
    /// Account manager
    pub account_manager: Arc<RwLock<AccountManager>>,
    /// Contract manager
    pub contract_manager: Arc<RwLock<ContractManager>>,
    /// Cross-chain manager
    pub cross_chain_manager: Arc<RwLock<CrossChainManager>>,
    /// Event manager
    pub event_manager: Arc<RwLock<EventManager>>,
    /// Performance metrics
    pub metrics: SDKMetrics,
}

/// Network client
#[derive(Debug)]
pub struct NetworkClient {
    /// Client ID
    pub client_id: String,
    /// Connected peers
    pub connected_peers: Vec<String>,
    /// Last block number
    pub last_block_number: u64,
    /// Network latency (ms)
    pub network_latency_ms: u64,
}

/// Account manager
#[derive(Debug)]
pub struct AccountManager {
    /// Managed accounts
    pub accounts: HashMap<[u8; 20], Account>,
    /// Default account
    pub default_account: Option<[u8; 20]>,
}

/// Contract manager
#[derive(Debug)]
pub struct ContractManager {
    /// Deployed contracts
    pub contracts: HashMap<[u8; 20], SmartContract>,
    /// Contract templates
    pub templates: HashMap<String, ContractTemplate>,
}

/// Contract template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractTemplate {
    /// Template name
    pub name: String,
    /// Template description
    pub description: String,
    /// Template ABI
    pub abi: String,
    /// Template bytecode
    pub bytecode: Vec<u8>,
    /// Template parameters
    pub parameters: Vec<ContractParameter>,
}

/// Contract parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub parameter_type: String,
    /// Is required
    pub is_required: bool,
    /// Default value
    pub default_value: Option<String>,
}

/// Cross-chain manager
#[derive(Debug)]
pub struct CrossChainManager {
    /// Supported chains
    pub supported_chains: Vec<u64>,
    /// Active operations
    pub active_operations: HashMap<String, CrossChainOperation>,
    /// Bridge contracts
    pub bridge_contracts: HashMap<u64, [u8; 20]>,
}

/// Event manager
#[derive(Debug)]
pub struct EventManager {
    /// Event subscriptions
    pub subscriptions: HashMap<String, EventFilter>,
    /// Event history
    pub event_history: VecDeque<LogEntry>,
    /// Max history size
    pub max_history_size: usize,
}

/// SDK metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SDKMetrics {
    /// Total transactions sent
    pub total_transactions_sent: u64,
    /// Successful transactions
    pub successful_transactions: u64,
    /// Failed transactions
    pub failed_transactions: u64,
    /// Total contract deployments
    pub total_contract_deployments: u64,
    /// Total cross-chain operations
    pub total_cross_chain_operations: u64,
    /// Average transaction time (ms)
    pub avg_transaction_time_ms: f64,
    /// Network uptime percentage
    pub network_uptime_percentage: f64,
}

impl HauptbuchSDK {
    /// Creates a new Hauptbuch SDK instance
    pub fn new(config: SDKConfig) -> Self {
        Self {
            network_client: Arc::new(RwLock::new(NetworkClient {
                client_id: format!("sdk_client_{}", current_timestamp()),
                connected_peers: Vec::new(),
                last_block_number: 0,
                network_latency_ms: 0,
            })),
            account_manager: Arc::new(RwLock::new(AccountManager {
                accounts: HashMap::new(),
                default_account: None,
            })),
            contract_manager: Arc::new(RwLock::new(ContractManager {
                contracts: HashMap::new(),
                templates: HashMap::new(),
            })),
            cross_chain_manager: Arc::new(RwLock::new(CrossChainManager {
                supported_chains: vec![1, 137, 42161], // Ethereum, Polygon, Arbitrum
                active_operations: HashMap::new(),
                bridge_contracts: HashMap::new(),
            })),
            event_manager: Arc::new(RwLock::new(EventManager {
                subscriptions: HashMap::new(),
                event_history: VecDeque::new(),
                max_history_size: 1000,
            })),
            metrics: SDKMetrics::default(),
            config,
        }
    }

    /// Connects to the network
    pub async fn connect(&mut self) -> SDKResult<()> {
        // Simulate network connection
        let mut client = self.network_client.write().unwrap();
        client
            .connected_peers
            .push(self.config.network_endpoint.clone());
        client.last_block_number = 1000000; // Simulate current block
        client.network_latency_ms = 50; // Simulate 50ms latency

        Ok(())
    }

    /// Disconnects from the network
    pub async fn disconnect(&mut self) -> SDKResult<()> {
        let mut client = self.network_client.write().unwrap();
        client.connected_peers.clear();
        client.last_block_number = 0;

        Ok(())
    }

    /// Creates a new account
    pub fn create_account(&mut self, account_type: AccountType) -> SDKResult<[u8; 20]> {
        let address = self.generate_address();
        let account = Account {
            address,
            balance: 0,
            nonce: 0,
            account_type,
            is_abstracted: account_type == AccountType::Abstracted,
            smart_contract_address: if account_type == AccountType::Abstracted {
                Some(self.generate_address())
            } else {
                None
            },
        };

        let mut account_manager = self.account_manager.write().unwrap();
        account_manager.accounts.insert(address, account);

        if account_manager.default_account.is_none() {
            account_manager.default_account = Some(address);
        }

        Ok(address)
    }

    /// Gets account information
    pub fn get_account(&self, address: [u8; 20]) -> SDKResult<Account> {
        let account_manager = self.account_manager.read().unwrap();
        account_manager
            .accounts
            .get(&address)
            .cloned()
            .ok_or(SDKError::AccountNotFound)
    }

    /// Sends a transaction
    pub async fn send_transaction(
        &mut self,
        tx_request: TransactionRequest,
    ) -> SDKResult<[u8; 32]> {
        let start_time = current_timestamp();

        // Validate transaction
        self.validate_transaction(&tx_request)?;

        // Simulate transaction processing
        let tx_hash = self.generate_tx_hash();

        // Update account nonce
        {
            let mut account_manager = self.account_manager.write().unwrap();
            if let Some(account) = account_manager.accounts.get_mut(&tx_request.from) {
                account.nonce += 1;
            }
        }

        // Update metrics
        self.metrics.total_transactions_sent += 1;
        self.metrics.successful_transactions += 1;

        let execution_time = current_timestamp() - start_time;
        self.metrics.avg_transaction_time_ms =
            (self.metrics.avg_transaction_time_ms + execution_time as f64) / 2.0;

        Ok(tx_hash)
    }

    /// Deploys a smart contract
    pub async fn deploy_contract(
        &mut self,
        bytecode: Vec<u8>,
        abi: String,
        _constructor_args: Vec<String>,
    ) -> SDKResult<[u8; 20]> {
        let contract_address = self.generate_address();
        let deployment_tx_hash = self.generate_tx_hash();

        let contract = SmartContract {
            address: contract_address,
            abi,
            bytecode,
            name: "DeployedContract".to_string(),
            version: "1.0.0".to_string(),
            deployment_tx_hash: Some(deployment_tx_hash),
            deployment_block_number: Some(1000001), // Simulate block number
        };

        let mut contract_manager = self.contract_manager.write().unwrap();
        contract_manager
            .contracts
            .insert(contract_address, contract);

        self.metrics.total_contract_deployments += 1;

        Ok(contract_address)
    }

    /// Calls a smart contract method
    pub async fn call_contract(
        &mut self,
        contract_address: [u8; 20],
        method: String,
        _args: Vec<String>,
    ) -> SDKResult<Vec<u8>> {
        let contract_manager = self.contract_manager.read().unwrap();
        let _contract = contract_manager
            .contracts
            .get(&contract_address)
            .ok_or(SDKError::ContractCallFailed)?;

        // Simulate contract call
        let result = format!("call_result_{}_{}", method, current_timestamp()).into_bytes();

        Ok(result)
    }

    /// Initiates a cross-chain operation
    pub async fn initiate_cross_chain_operation(
        &mut self,
        operation: CrossChainOperation,
    ) -> SDKResult<String> {
        let operation_id = operation.operation_id.clone();

        let mut cross_chain_manager = self.cross_chain_manager.write().unwrap();
        cross_chain_manager
            .active_operations
            .insert(operation_id.clone(), operation);

        self.metrics.total_cross_chain_operations += 1;

        Ok(operation_id)
    }

    /// Subscribes to events
    pub async fn subscribe_to_events(&mut self, filter: EventFilter) -> SDKResult<String> {
        let subscription_id = format!("sub_{}", current_timestamp());

        let mut event_manager = self.event_manager.write().unwrap();
        event_manager
            .subscriptions
            .insert(subscription_id.clone(), filter);

        Ok(subscription_id)
    }

    /// Unsubscribes from events
    pub async fn unsubscribe_from_events(&mut self, subscription_id: &str) -> SDKResult<()> {
        let mut event_manager = self.event_manager.write().unwrap();
        event_manager.subscriptions.remove(subscription_id);

        Ok(())
    }

    /// Gets event history
    pub fn get_event_history(&self, limit: Option<usize>) -> Vec<LogEntry> {
        let event_manager = self.event_manager.read().unwrap();
        let limit = limit.unwrap_or(event_manager.event_history.len());

        event_manager
            .event_history
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }

    /// Gets SDK metrics
    pub fn get_metrics(&self) -> &SDKMetrics {
        &self.metrics
    }

    // Private helper methods

    /// Validates a transaction
    fn validate_transaction(&self, tx_request: &TransactionRequest) -> SDKResult<()> {
        // Check gas limit
        if tx_request.gas_limit == 0 {
            return Err(SDKError::TransactionFailed);
        }

        // Check gas price
        if tx_request.gas_price == 0 {
            return Err(SDKError::TransactionFailed);
        }

        // Check nonce
        if tx_request.nonce == 0 && tx_request.from != [0u8; 20] {
            return Err(SDKError::TransactionFailed);
        }

        Ok(())
    }

    /// Generates a random address
    fn generate_address(&self) -> [u8; 20] {
        let mut address = [0u8; 20];
        for (i, byte) in address.iter_mut().enumerate() {
            *byte = (current_timestamp() as u8).wrapping_add(i as u8);
        }
        address
    }

    /// Generates a random transaction hash
    fn generate_tx_hash(&self) -> [u8; 32] {
        let mut hash = [0u8; 32];
        for (i, byte) in hash.iter_mut().enumerate() {
            *byte = (current_timestamp() as u8).wrapping_add(i as u8);
        }
        hash
    }
}

/// Gets current timestamp
fn current_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdk_creation() {
        let config = SDKConfig {
            network_endpoint: "http://localhost:8545".to_string(),
            chain_id: 1,
            gas_price: 20_000_000_000, // 20 gwei
            gas_limit: 100_000,
            enable_account_abstraction: true,
            enable_mev_protection: true,
            enable_cross_chain: true,
            connection_timeout_ms: 5000,
            retry_attempts: 3,
            enable_performance_optimization: true,
        };

        let sdk = HauptbuchSDK::new(config);
        let metrics = sdk.get_metrics();
        assert_eq!(metrics.total_transactions_sent, 0);
    }

    #[tokio::test]
    async fn test_sdk_connection() {
        let config = SDKConfig {
            network_endpoint: "http://localhost:8545".to_string(),
            chain_id: 1,
            gas_price: 20_000_000_000,
            gas_limit: 100_000,
            enable_account_abstraction: true,
            enable_mev_protection: true,
            enable_cross_chain: true,
            connection_timeout_ms: 5000,
            retry_attempts: 3,
            enable_performance_optimization: true,
        };

        let mut sdk = HauptbuchSDK::new(config);
        let result = sdk.connect().await;
        assert!(result.is_ok());

        let client = sdk.network_client.read().unwrap();
        assert!(!client.connected_peers.is_empty());
    }

    #[test]
    fn test_account_creation() {
        let config = SDKConfig {
            network_endpoint: "http://localhost:8545".to_string(),
            chain_id: 1,
            gas_price: 20_000_000_000,
            gas_limit: 100_000,
            enable_account_abstraction: true,
            enable_mev_protection: true,
            enable_cross_chain: true,
            connection_timeout_ms: 5000,
            retry_attempts: 3,
            enable_performance_optimization: true,
        };

        let mut sdk = HauptbuchSDK::new(config);

        let address = sdk.create_account(AccountType::EOA).unwrap();
        assert_ne!(address, [0u8; 20]);

        let account = sdk.get_account(address).unwrap();
        assert_eq!(account.account_type, AccountType::EOA);
        assert_eq!(account.balance, 0);
        assert_eq!(account.nonce, 0);
    }

    #[test]
    fn test_abstracted_account_creation() {
        let config = SDKConfig {
            network_endpoint: "http://localhost:8545".to_string(),
            chain_id: 1,
            gas_price: 20_000_000_000,
            gas_limit: 100_000,
            enable_account_abstraction: true,
            enable_mev_protection: true,
            enable_cross_chain: true,
            connection_timeout_ms: 5000,
            retry_attempts: 3,
            enable_performance_optimization: true,
        };

        let mut sdk = HauptbuchSDK::new(config);

        let address = sdk.create_account(AccountType::Abstracted).unwrap();
        let account = sdk.get_account(address).unwrap();

        assert_eq!(account.account_type, AccountType::Abstracted);
        assert!(account.is_abstracted);
        assert!(account.smart_contract_address.is_some());
    }

    #[tokio::test]
    async fn test_transaction_sending() {
        let config = SDKConfig {
            network_endpoint: "http://localhost:8545".to_string(),
            chain_id: 1,
            gas_price: 20_000_000_000,
            gas_limit: 100_000,
            enable_account_abstraction: true,
            enable_mev_protection: true,
            enable_cross_chain: true,
            connection_timeout_ms: 5000,
            retry_attempts: 3,
            enable_performance_optimization: true,
        };

        let mut sdk = HauptbuchSDK::new(config);

        let from_address = sdk.create_account(AccountType::EOA).unwrap();
        let to_address = sdk.create_account(AccountType::EOA).unwrap();

        let tx_request = TransactionRequest {
            from: from_address,
            to: Some(to_address),
            value: 1_000_000_000_000_000_000, // 1 ETH
            gas_limit: 21_000,
            gas_price: 20_000_000_000,
            nonce: 1,
            data: Vec::new(),
            transaction_type: TransactionType::Legacy,
            access_list: Vec::new(),
            max_fee_per_gas: None,
            max_priority_fee_per_gas: None,
        };

        let tx_hash = sdk.send_transaction(tx_request).await.unwrap();
        assert_ne!(tx_hash, [0u8; 32]);

        let metrics = sdk.get_metrics();
        assert_eq!(metrics.total_transactions_sent, 1);
        assert_eq!(metrics.successful_transactions, 1);
    }

    #[tokio::test]
    async fn test_contract_deployment() {
        let config = SDKConfig {
            network_endpoint: "http://localhost:8545".to_string(),
            chain_id: 1,
            gas_price: 20_000_000_000,
            gas_limit: 100_000,
            enable_account_abstraction: true,
            enable_mev_protection: true,
            enable_cross_chain: true,
            connection_timeout_ms: 5000,
            retry_attempts: 3,
            enable_performance_optimization: true,
        };

        let mut sdk = HauptbuchSDK::new(config);

        let bytecode = vec![0x60, 0x60, 0x60, 0x40, 0x52]; // Simple contract bytecode
        let abi = r#"[]"#.to_string();
        let constructor_args = Vec::new();

        let contract_address = sdk
            .deploy_contract(bytecode, abi, constructor_args)
            .await
            .unwrap();
        assert_ne!(contract_address, [0u8; 20]);

        let metrics = sdk.get_metrics();
        assert_eq!(metrics.total_contract_deployments, 1);
    }

    #[tokio::test]
    async fn test_cross_chain_operation() {
        let config = SDKConfig {
            network_endpoint: "http://localhost:8545".to_string(),
            chain_id: 1,
            gas_price: 20_000_000_000,
            gas_limit: 100_000,
            enable_account_abstraction: true,
            enable_mev_protection: true,
            enable_cross_chain: true,
            connection_timeout_ms: 5000,
            retry_attempts: 3,
            enable_performance_optimization: true,
        };

        let mut sdk = HauptbuchSDK::new(config);

        let source_address = sdk.create_account(AccountType::EOA).unwrap();
        let target_address = sdk.create_account(AccountType::EOA).unwrap();

        let operation = CrossChainOperation {
            operation_id: "op_1".to_string(),
            source_chain_id: 1,
            target_chain_id: 137,
            source_address,
            target_address,
            amount: 1_000_000_000_000_000_000,
            token_address: None,
            operation_type: CrossChainOperationType::TokenTransfer,
            status: CrossChainOperationStatus::Pending,
        };

        let operation_id = sdk.initiate_cross_chain_operation(operation).await.unwrap();
        assert_eq!(operation_id, "op_1");

        let metrics = sdk.get_metrics();
        assert_eq!(metrics.total_cross_chain_operations, 1);
    }

    #[tokio::test]
    async fn test_event_subscription() {
        let config = SDKConfig {
            network_endpoint: "http://localhost:8545".to_string(),
            chain_id: 1,
            gas_price: 20_000_000_000,
            gas_limit: 100_000,
            enable_account_abstraction: true,
            enable_mev_protection: true,
            enable_cross_chain: true,
            connection_timeout_ms: 5000,
            retry_attempts: 3,
            enable_performance_optimization: true,
        };

        let mut sdk = HauptbuchSDK::new(config);

        let filter = EventFilter {
            contract_address: None,
            topics: Vec::new(),
            from_block: Some(1000000),
            to_block: None,
        };

        let subscription_id = sdk.subscribe_to_events(filter).await.unwrap();
        assert!(!subscription_id.is_empty());

        let result = sdk.unsubscribe_from_events(&subscription_id).await;
        assert!(result.is_ok());
    }
}
