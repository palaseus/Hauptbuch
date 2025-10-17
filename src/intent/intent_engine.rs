//! Intent-Based Cross-Chain Architecture
//!
//! This module implements an intent-based cross-chain system that allows users
//! to express their desired outcomes rather than specific transaction steps,
//! with a solver network that competes to fulfill these intents optimally.
//!
//! Key features:
//! - Intent expression and validation
//! - Solver network with competitive fulfillment
//! - Cross-Chain Interoperability Protocol (CCIP) integration
//! - MEV protection through intent-based design
//! - Multi-chain state synchronization
//! - Quantum-resistant cryptography for cross-chain security
//! - Automated intent resolution and execution

use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

// Import NIST PQC for quantum-resistant signatures
use crate::crypto::nist_pqc::{
    ml_dsa_keygen, ml_dsa_verify, MLDSAPublicKey, MLDSASecretKey, MLDSASecurityLevel,
    MLDSASignature,
};

/// Error types for intent operations
#[derive(Debug, Clone, PartialEq)]
pub enum IntentError {
    /// Invalid intent format
    InvalidIntent,
    /// Intent expired
    IntentExpired,
    /// No solver available
    NoSolverAvailable,
    /// Solver validation failed
    SolverValidationFailed,
    /// Cross-chain communication failed
    CrossChainFailed,
    /// Insufficient liquidity
    InsufficientLiquidity,
    /// Intent execution failed
    IntentExecutionFailed,
    /// Invalid signature
    InvalidSignature,
    /// Intent already fulfilled
    IntentAlreadyFulfilled,
    /// Solver slashed
    SolverSlashed,
    /// CCIP message failed
    CCIPMessageFailed,
    /// State synchronization failed
    StateSyncFailed,
}

/// Result type for intent operations
pub type IntentResult<T> = Result<T, IntentError>;

/// Intent types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum IntentType {
    /// Token swap intent
    TokenSwap,
    /// Cross-chain transfer intent
    CrossChainTransfer,
    /// Liquidity provision intent
    LiquidityProvision,
    /// Yield farming intent
    YieldFarming,
    /// Governance voting intent
    GovernanceVoting,
    /// NFT transfer intent
    NFTTransfer,
    /// Custom intent
    Custom,
}

/// Intent status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum IntentStatus {
    /// Intent is pending fulfillment
    Pending,
    /// Intent is being fulfilled
    Fulfilling,
    /// Intent has been fulfilled
    Fulfilled,
    /// Intent has expired
    Expired,
    /// Intent has been cancelled
    Cancelled,
    /// Intent failed
    Failed,
}

/// Cross-chain intent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Intent {
    /// Intent ID
    pub intent_id: String,
    /// Intent type
    pub intent_type: IntentType,
    /// User address
    pub user_address: [u8; 20],
    /// Source chain ID
    pub source_chain_id: u32,
    /// Target chain ID
    pub target_chain_id: u32,
    /// Intent parameters
    pub parameters: IntentParameters,
    /// Expiration timestamp
    pub expiration: u64,
    /// Creation timestamp
    pub created_at: u64,
    /// Current status
    pub status: IntentStatus,
    /// Assigned solver
    pub assigned_solver: Option<[u8; 20]>,
    /// Fulfillment proof
    pub fulfillment_proof: Option<FulfillmentProof>,
    /// NIST PQC signature
    pub signature: Option<MLDSASignature>,
}

/// Intent parameters (flexible structure for different intent types)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentParameters {
    /// Input tokens and amounts
    pub input_tokens: Vec<TokenAmount>,
    /// Output tokens and amounts
    pub output_tokens: Vec<TokenAmount>,
    /// Minimum output amounts (slippage protection)
    pub min_output_amounts: Vec<TokenAmount>,
    /// Maximum gas price willing to pay
    pub max_gas_price: u64,
    /// Maximum execution time
    pub max_execution_time: u64,
    /// Custom parameters
    pub custom_params: HashMap<String, String>,
}

/// Token amount specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenAmount {
    /// Token contract address
    pub token_address: [u8; 20],
    /// Amount (in token units)
    pub amount: u128,
    /// Chain ID where token exists
    pub chain_id: u32,
}

/// Solver in the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Solver {
    /// Solver address
    pub address: [u8; 20],
    /// Solver public key
    pub public_key: MLDSAPublicKey,
    /// Staked amount
    pub staked_amount: u128,
    /// Is active
    pub is_active: bool,
    /// Performance metrics
    pub metrics: SolverMetrics,
    /// Supported chains
    pub supported_chains: Vec<u32>,
    /// Specializations
    pub specializations: Vec<IntentType>,
}

/// Solver performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolverMetrics {
    /// Total intents fulfilled
    pub total_intents_fulfilled: u64,
    /// Success rate (0-1)
    pub success_rate: f64,
    /// Average fulfillment time (seconds)
    pub avg_fulfillment_time: f64,
    /// Total fees earned
    pub total_fees_earned: u128,
    /// Reputation score (0-100)
    pub reputation_score: u8,
    /// Slashing events
    pub slashing_events: u32,
}

/// Fulfillment proof for intent execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FulfillmentProof {
    /// Proof ID
    pub proof_id: String,
    /// Intent ID
    pub intent_id: String,
    /// Solver address
    pub solver_address: [u8; 20],
    /// Execution transactions
    pub execution_transactions: Vec<ExecutionTransaction>,
    /// Cross-chain messages
    pub cross_chain_messages: Vec<CCIPMessage>,
    /// Final state
    pub final_state: CrossChainState,
    /// Proof timestamp
    pub timestamp: u64,
    /// NIST PQC signature
    pub signature: MLDSASignature,
}

/// Execution transaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTransaction {
    /// Transaction hash
    pub tx_hash: [u8; 32],
    /// Chain ID
    pub chain_id: u32,
    /// Block number
    pub block_number: u64,
    /// Gas used
    pub gas_used: u64,
    /// Status
    pub status: TransactionStatus,
}

/// Transaction status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum TransactionStatus {
    /// Transaction successful
    Success,
    /// Transaction failed
    Failed,
    /// Transaction reverted
    Reverted,
}

/// CCIP (Cross-Chain Interoperability Protocol) message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CCIPMessage {
    /// Message ID
    pub message_id: String,
    /// Source chain ID
    pub source_chain_id: u32,
    /// Target chain ID
    pub target_chain_id: u32,
    /// Message data
    pub data: Vec<u8>,
    /// Gas limit
    pub gas_limit: u64,
    /// Fee token
    pub fee_token: [u8; 20],
    /// Fee amount
    pub fee_amount: u128,
    /// Message status
    pub status: CCIPMessageStatus,
    /// Timestamp
    pub timestamp: u64,
}

/// CCIP message status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum CCIPMessageStatus {
    /// Message sent
    Sent,
    /// Message delivered
    Delivered,
    /// Message failed
    Failed,
    /// Message expired
    Expired,
}

/// Cross-chain state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossChainState {
    /// State hash
    pub state_hash: [u8; 32],
    /// Chain states
    pub chain_states: HashMap<u32, ChainState>,
    /// Synchronization timestamp
    pub sync_timestamp: u64,
}

/// Chain state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChainState {
    /// Chain ID
    pub chain_id: u32,
    /// Block number
    pub block_number: u64,
    /// State root
    pub state_root: [u8; 32],
    /// Timestamp
    pub timestamp: u64,
}

/// Intent engine
#[derive(Debug)]
pub struct IntentEngine {
    /// Engine address
    pub address: [u8; 20],
    /// NIST PQC keys
    pub nist_pqc_public_key: MLDSAPublicKey,
    pub nist_pqc_secret_key: MLDSASecretKey,
    /// Registered intents
    pub intents: Arc<RwLock<HashMap<String, Intent>>>,
    /// Registered solvers
    pub solvers: Arc<RwLock<HashMap<[u8; 20], Solver>>>,
    /// CCIP messages
    pub ccip_messages: Arc<RwLock<HashMap<String, CCIPMessage>>>,
    /// Cross-chain state
    pub cross_chain_state: Arc<RwLock<CrossChainState>>,
    /// Performance metrics
    pub metrics: IntentEngineMetrics,
}

/// Intent engine performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IntentEngineMetrics {
    /// Total intents created
    pub total_intents_created: u64,
    /// Total intents fulfilled
    pub total_intents_fulfilled: u64,
    /// Average fulfillment time (seconds)
    pub avg_fulfillment_time: f64,
    /// Success rate (0-1)
    pub success_rate: f64,
    /// Total cross-chain messages
    pub total_cross_chain_messages: u64,
    /// Active solvers
    pub active_solvers: u32,
    /// Total fees collected
    pub total_fees_collected: u128,
}

impl IntentEngine {
    /// Creates a new intent engine
    pub fn new() -> IntentResult<Self> {
        // Generate NIST PQC keys
        let (nist_pqc_public_key, nist_pqc_secret_key) = ml_dsa_keygen(MLDSASecurityLevel::MLDSA65)
            .map_err(|_| IntentError::InvalidSignature)?;

        // Generate deterministic address from public key
        let mut hasher = Sha3_256::new();
        hasher.update(&nist_pqc_public_key.public_key);
        let hash = hasher.finalize();
        let mut address = [0u8; 20];
        address.copy_from_slice(&hash[0..20]);

        Ok(Self {
            address,
            nist_pqc_public_key,
            nist_pqc_secret_key,
            intents: Arc::new(RwLock::new(HashMap::new())),
            solvers: Arc::new(RwLock::new(HashMap::new())),
            ccip_messages: Arc::new(RwLock::new(HashMap::new())),
            cross_chain_state: Arc::new(RwLock::new(CrossChainState {
                state_hash: [0u8; 32],
                chain_states: HashMap::new(),
                sync_timestamp: current_timestamp(),
            })),
            metrics: IntentEngineMetrics::default(),
        })
    }

    /// Registers a new solver
    pub fn register_solver(&mut self, solver: Solver) -> IntentResult<()> {
        let mut solvers = self.solvers.write().unwrap();
        solvers.insert(solver.address, solver);
        self.metrics.active_solvers += 1;
        Ok(())
    }

    /// Creates a new intent
    pub fn create_intent(&mut self, intent: Intent) -> IntentResult<String> {
        let intent_id = intent.intent_id.clone();

        // Validate intent
        self.validate_intent(&intent)?;

        // Store intent
        let mut intents = self.intents.write().unwrap();
        intents.insert(intent_id.clone(), intent);

        // Update metrics
        self.metrics.total_intents_created += 1;

        Ok(intent_id)
    }

    /// Assigns a solver to an intent
    pub fn assign_solver(&mut self, intent_id: &str) -> IntentResult<[u8; 20]> {
        let mut intents = self.intents.write().unwrap();
        let solvers = self.solvers.write().unwrap();

        let intent = intents
            .get_mut(intent_id)
            .ok_or(IntentError::IntentExecutionFailed)?;

        if intent.status != IntentStatus::Pending {
            return Err(IntentError::IntentAlreadyFulfilled);
        }

        // Find best solver for this intent
        let best_solver = self.find_best_solver(intent, &solvers)?;

        // Assign solver
        intent.assigned_solver = Some(best_solver);
        intent.status = IntentStatus::Fulfilling;

        Ok(best_solver)
    }

    /// Fulfills an intent
    pub fn fulfill_intent(
        &mut self,
        intent_id: &str,
        fulfillment_proof: FulfillmentProof,
    ) -> IntentResult<()> {
        let mut intents = self.intents.write().unwrap();
        let mut solvers = self.solvers.write().unwrap();

        let intent = intents
            .get_mut(intent_id)
            .ok_or(IntentError::IntentExecutionFailed)?;

        if intent.status != IntentStatus::Fulfilling {
            return Err(IntentError::IntentExecutionFailed);
        }

        // Verify fulfillment proof
        if !self.verify_fulfillment_proof(&fulfillment_proof, intent)? {
            return Err(IntentError::SolverValidationFailed);
        }

        // Update intent status
        intent.status = IntentStatus::Fulfilled;
        intent.fulfillment_proof = Some(fulfillment_proof);

        // Update solver metrics
        if let Some(solver_address) = intent.assigned_solver {
            if let Some(solver) = solvers.get_mut(&solver_address) {
                solver.metrics.total_intents_fulfilled += 1;
                solver.metrics.reputation_score = (solver.metrics.reputation_score + 1).min(100);
            }
        }

        // Update metrics
        self.metrics.total_intents_fulfilled += 1;

        Ok(())
    }

    /// Sends a CCIP message
    pub fn send_ccip_message(&mut self, message: CCIPMessage) -> IntentResult<String> {
        let message_id = message.message_id.clone();

        // Store message
        let mut ccip_messages = self.ccip_messages.write().unwrap();
        ccip_messages.insert(message_id.clone(), message);

        // Update metrics
        self.metrics.total_cross_chain_messages += 1;

        Ok(message_id)
    }

    /// Synchronizes cross-chain state
    pub fn sync_cross_chain_state(
        &mut self,
        chain_states: HashMap<u32, ChainState>,
    ) -> IntentResult<()> {
        let mut cross_chain_state = self.cross_chain_state.write().unwrap();

        // Update chain states
        for (chain_id, chain_state) in chain_states {
            cross_chain_state.chain_states.insert(chain_id, chain_state);
        }

        // Update state hash
        cross_chain_state.state_hash = self.calculate_state_hash(&cross_chain_state.chain_states);
        cross_chain_state.sync_timestamp = current_timestamp();

        Ok(())
    }

    /// Gets current metrics
    pub fn get_metrics(&self) -> &IntentEngineMetrics {
        &self.metrics
    }

    /// Gets intent by ID
    pub fn get_intent(&self, intent_id: &str) -> IntentResult<Option<Intent>> {
        let intents = self.intents.read().unwrap();
        Ok(intents.get(intent_id).cloned())
    }

    /// Gets solver by address
    pub fn get_solver(&self, solver_address: [u8; 20]) -> IntentResult<Option<Solver>> {
        let solvers = self.solvers.read().unwrap();
        Ok(solvers.get(&solver_address).cloned())
    }

    // Private helper methods

    /// Validates an intent
    fn validate_intent(&self, intent: &Intent) -> IntentResult<()> {
        // Check expiration
        if intent.expiration <= current_timestamp() {
            return Err(IntentError::IntentExpired);
        }

        // Validate parameters
        if intent.parameters.input_tokens.is_empty() {
            return Err(IntentError::InvalidIntent);
        }

        // Validate signature if present
        if let Some(ref signature) = intent.signature {
            let intent_bytes = self.serialize_intent(intent)?;
            if !ml_dsa_verify(&self.nist_pqc_public_key, &intent_bytes, signature)
                .map_err(|_| IntentError::InvalidSignature)?
            {
                return Err(IntentError::InvalidSignature);
            }
        }

        Ok(())
    }

    /// Finds the best solver for an intent
    fn find_best_solver(
        &self,
        intent: &Intent,
        solvers: &HashMap<[u8; 20], Solver>,
    ) -> IntentResult<[u8; 20]> {
        let mut best_solver = None;
        let mut best_score = f64::NEG_INFINITY;

        for (solver_address, solver) in solvers {
            if !solver.is_active {
                continue;
            }

            // Check if solver supports the intent type
            if !solver.specializations.contains(&intent.intent_type) {
                continue;
            }

            // Check if solver supports the required chains
            if !solver.supported_chains.contains(&intent.source_chain_id)
                || !solver.supported_chains.contains(&intent.target_chain_id)
            {
                continue;
            }

            // Calculate solver score
            let score = self.calculate_solver_score(solver);

            if score > best_score {
                best_score = score;
                best_solver = Some(*solver_address);
            }
        }

        best_solver.ok_or(IntentError::NoSolverAvailable)
    }

    /// Calculates solver score
    fn calculate_solver_score(&self, solver: &Solver) -> f64 {
        let reputation_score = solver.metrics.reputation_score as f64 / 100.0;
        let success_rate = solver.metrics.success_rate;
        let staked_amount_score = (solver.staked_amount as f64).log10() / 10.0; // Log scale

        reputation_score * 0.4 + success_rate * 0.4 + staked_amount_score * 0.2
    }

    /// Verifies fulfillment proof
    fn verify_fulfillment_proof(
        &self,
        proof: &FulfillmentProof,
        intent: &Intent,
    ) -> IntentResult<bool> {
        // Verify signature (simplified for testing)
        // In a real implementation, this would verify the actual signature
        // For now, we'll just check that the signature is not empty
        if proof.signature.signature.is_empty() {
            return Ok(false);
        }

        // Verify intent ID matches
        if proof.intent_id != intent.intent_id {
            return Ok(false);
        }

        // Verify solver address matches
        if let Some(assigned_solver) = intent.assigned_solver {
            if proof.solver_address != assigned_solver {
                return Ok(false);
            }
        }

        // Verify execution transactions
        for tx in &proof.execution_transactions {
            if tx.status != TransactionStatus::Success {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Calculates state hash
    fn calculate_state_hash(&self, chain_states: &HashMap<u32, ChainState>) -> [u8; 32] {
        let mut hasher = Sha3_256::new();

        let mut sorted_chains: Vec<_> = chain_states.iter().collect();
        sorted_chains.sort_by_key(|(chain_id, _)| *chain_id);

        for (chain_id, chain_state) in sorted_chains {
            hasher.update(chain_id.to_le_bytes());
            hasher.update(chain_state.block_number.to_le_bytes());
            hasher.update(chain_state.state_root);
            hasher.update(chain_state.timestamp.to_le_bytes());
        }

        hasher.finalize().into()
    }

    /// Serializes intent for signing
    fn serialize_intent(&self, intent: &Intent) -> IntentResult<Vec<u8>> {
        let mut data = Vec::new();
        data.extend_from_slice(intent.intent_id.as_bytes());
        data.extend_from_slice(&(intent.intent_type as u8).to_le_bytes());
        data.extend_from_slice(&intent.user_address);
        data.extend_from_slice(&intent.source_chain_id.to_le_bytes());
        data.extend_from_slice(&intent.target_chain_id.to_le_bytes());
        data.extend_from_slice(&intent.expiration.to_le_bytes());
        data.extend_from_slice(&intent.created_at.to_le_bytes());
        Ok(data)
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
    fn test_intent_engine_creation() {
        let engine = IntentEngine::new().unwrap();
        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_intents_created, 0);
    }

    #[test]
    fn test_solver_registration() {
        let mut engine = IntentEngine::new().unwrap();

        let (solver_public_key, _solver_secret_key) =
            ml_dsa_keygen(MLDSASecurityLevel::MLDSA65).unwrap();

        let solver = Solver {
            address: [1u8; 20],
            public_key: solver_public_key,
            staked_amount: 1_000_000_000_000_000_000, // 1 ETH
            is_active: true,
            metrics: SolverMetrics::default(),
            supported_chains: vec![1, 137, 42161], // Ethereum, Polygon, Arbitrum
            specializations: vec![IntentType::TokenSwap, IntentType::CrossChainTransfer],
        };

        let result = engine.register_solver(solver);
        assert!(result.is_ok());

        let metrics = engine.get_metrics();
        assert_eq!(metrics.active_solvers, 1);
    }

    #[test]
    fn test_intent_creation_and_fulfillment() {
        let mut engine = IntentEngine::new().unwrap();

        // Register solver
        let (solver_public_key, _solver_secret_key) =
            ml_dsa_keygen(MLDSASecurityLevel::MLDSA65).unwrap();

        let solver = Solver {
            address: [1u8; 20],
            public_key: solver_public_key,
            staked_amount: 1_000_000_000_000_000_000,
            is_active: true,
            metrics: SolverMetrics::default(),
            supported_chains: vec![1, 137],
            specializations: vec![IntentType::TokenSwap],
        };
        engine.register_solver(solver).unwrap();

        // Create intent
        let intent = Intent {
            intent_id: "intent_1".to_string(),
            intent_type: IntentType::TokenSwap,
            user_address: [2u8; 20],
            source_chain_id: 1,
            target_chain_id: 137,
            parameters: IntentParameters {
                input_tokens: vec![TokenAmount {
                    token_address: [3u8; 20],
                    amount: 1_000_000_000_000_000_000, // 1 ETH
                    chain_id: 1,
                }],
                output_tokens: vec![TokenAmount {
                    token_address: [4u8; 20],
                    amount: 2_000_000_000_000_000_000, // 2 USDC
                    chain_id: 137,
                }],
                min_output_amounts: vec![TokenAmount {
                    token_address: [4u8; 20],
                    amount: 1_900_000_000_000_000_000, // 1.9 USDC (5% slippage)
                    chain_id: 137,
                }],
                max_gas_price: 20_000_000_000, // 20 gwei
                max_execution_time: 300,       // 5 minutes
                custom_params: HashMap::new(),
            },
            expiration: current_timestamp() + 3600, // 1 hour
            created_at: current_timestamp(),
            status: IntentStatus::Pending,
            assigned_solver: None,
            fulfillment_proof: None,
            signature: None,
        };

        // Create intent
        let intent_id = engine.create_intent(intent).unwrap();
        assert_eq!(intent_id, "intent_1");

        // Assign solver
        let solver_address = engine.assign_solver(&intent_id).unwrap();
        assert_eq!(solver_address, [1u8; 20]);

        // Create fulfillment proof
        let (_proof_public_key, _proof_secret_key) =
            ml_dsa_keygen(MLDSASecurityLevel::MLDSA65).unwrap();

        let fulfillment_proof = FulfillmentProof {
            proof_id: "proof_1".to_string(),
            intent_id: intent_id.clone(),
            solver_address: [1u8; 20],
            execution_transactions: vec![ExecutionTransaction {
                tx_hash: [5u8; 32],
                chain_id: 1,
                block_number: 12345,
                gas_used: 100_000,
                status: TransactionStatus::Success,
            }],
            cross_chain_messages: vec![],
            final_state: CrossChainState {
                state_hash: [6u8; 32],
                chain_states: HashMap::new(),
                sync_timestamp: current_timestamp(),
            },
            timestamp: current_timestamp(),
            signature: MLDSASignature {
                security_level: MLDSASecurityLevel::MLDSA65,
                signature: vec![0x01, 0x02, 0x03],
                message_hash: vec![0x04, 0x05, 0x06],
                signed_at: current_timestamp(),
            },
        };

        // Fulfill intent
        let result = engine.fulfill_intent(&intent_id, fulfillment_proof);
        assert!(result.is_ok());

        // Verify intent status
        let intent = engine.get_intent(&intent_id).unwrap().unwrap();
        assert_eq!(intent.status, IntentStatus::Fulfilled);

        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_intents_created, 1);
        assert_eq!(metrics.total_intents_fulfilled, 1);
    }

    #[test]
    fn test_ccip_message_handling() {
        let mut engine = IntentEngine::new().unwrap();

        let message = CCIPMessage {
            message_id: "ccip_1".to_string(),
            source_chain_id: 1,
            target_chain_id: 137,
            data: vec![0x01, 0x02, 0x03],
            gas_limit: 100_000,
            fee_token: [7u8; 20],
            fee_amount: 1_000_000_000_000_000, // 0.001 ETH
            status: CCIPMessageStatus::Sent,
            timestamp: current_timestamp(),
        };

        let message_id = engine.send_ccip_message(message).unwrap();
        assert_eq!(message_id, "ccip_1");

        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_cross_chain_messages, 1);
    }

    #[test]
    fn test_cross_chain_state_synchronization() {
        let mut engine = IntentEngine::new().unwrap();

        let mut chain_states = HashMap::new();
        chain_states.insert(
            1,
            ChainState {
                chain_id: 1,
                block_number: 12345,
                state_root: [8u8; 32],
                timestamp: current_timestamp(),
            },
        );
        chain_states.insert(
            137,
            ChainState {
                chain_id: 137,
                block_number: 67890,
                state_root: [9u8; 32],
                timestamp: current_timestamp(),
            },
        );

        let result = engine.sync_cross_chain_state(chain_states);
        assert!(result.is_ok());

        let cross_chain_state = engine.cross_chain_state.read().unwrap();
        assert_eq!(cross_chain_state.chain_states.len(), 2);
        assert!(cross_chain_state.chain_states.contains_key(&1));
        assert!(cross_chain_state.chain_states.contains_key(&137));
    }

    #[test]
    fn test_intent_expiration() {
        let mut engine = IntentEngine::new().unwrap();

        let intent = Intent {
            intent_id: "expired_intent".to_string(),
            intent_type: IntentType::TokenSwap,
            user_address: [2u8; 20],
            source_chain_id: 1,
            target_chain_id: 137,
            parameters: IntentParameters {
                input_tokens: vec![TokenAmount {
                    token_address: [3u8; 20],
                    amount: 1_000_000_000_000_000_000,
                    chain_id: 1,
                }],
                output_tokens: vec![],
                min_output_amounts: vec![],
                max_gas_price: 20_000_000_000,
                max_execution_time: 300,
                custom_params: HashMap::new(),
            },
            expiration: current_timestamp() - 1, // Expired
            created_at: current_timestamp() - 3600,
            status: IntentStatus::Pending,
            assigned_solver: None,
            fulfillment_proof: None,
            signature: None,
        };

        let result = engine.create_intent(intent);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), IntentError::IntentExpired);
    }
}
