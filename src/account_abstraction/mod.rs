//! Account Abstraction Module
//!
//! This module provides comprehensive account abstraction functionality
//! following the ERC-4337 standard, enabling smart contract wallets,
//! paymasters, session keys, and social recovery mechanisms.
//!
//! Key features:
//! - ERC-4337 compliant smart contract wallets
//! - Paymasters for gasless transactions and sponsored fees
//! - Session keys for improved UX and security
//! - Social recovery mechanisms for key management
//! - Bundler for transaction aggregation and execution
//! - EntryPoint contract for unified transaction processing
//! - Integration with NIST PQC for quantum-resistant security

pub mod erc4337;
pub mod erc6900;
pub mod erc7579;
pub mod erc7702;

// Re-export main types for convenience
pub use erc4337::{
    Bundler,

    BundlerMetrics,

    // Error types
    ERC4337Error,
    ERC4337Result,
    // Core ERC-4337 types
    EntryPoint,
    // Metrics types
    EntryPointMetrics,
    // Social recovery types
    Guardian,
    GuardianType,

    Paymaster,
    // Paymaster types
    PaymasterRules,
    // Session key types
    SessionKey,
    SessionPermissions,

    SmartAccount,
    TimeRestrictions,

    UserOperation,
};
pub use erc6900::{
    DependencyType,
    // Error types
    ERC6900Error,
    // Core ERC-6900 types
    ERC6900PluginManager,
    ERC6900Result,
    FunctionVisibility,
    PermissionLevel,
    PermissionScope,
    PluginDependency,
    PluginError,
    PluginEvent,
    PluginExecutionContext,
    PluginExecutionResult,
    PluginFunction,
    PluginHook,
    PluginInstance,
    PluginInterface,
    PluginManifest,
    PluginMarketplaceEntry,
    PluginMetrics,
    PluginParameter,
    PluginPermission,
    PluginReview,
    PluginStatus,
};
pub use erc7579::{
    AccountMetrics,
    AccountPlugin,
    AccountState,
    AccountStatus,

    CrossChainMessage,
    CrossChainMessageStatus,
    CrossChainMessageType,
    // Core ERC-7579 types
    ERC7579Account,
    // Error types
    ERC7579Error,
    ERC7579Result,
    ExecutionHook,
    ExecutionHookType,
    PluginType,
    RecoveryInfo,
    ValidationHook,
    ValidationHookType,
};
pub use erc7702::{
    AccessListItem,
    // Account types
    AccountType,
    AuditStatus,
    ERC7702Account,
    ERC7702Config,

    // Core ERC-7702 types
    ERC7702Engine,
    // Error types
    ERC7702Error,
    ERC7702ExecutionContext,
    ERC7702ExecutionResult,
    ERC7702Implementation,
    ERC7702Metrics,
    ERC7702Result,
    ERC7702Transaction,
    EventLog,

    GasCosts,
    ImplementationType,
    SecurityLevel,
    SetCodeOperation,
    TransactionType,
};
