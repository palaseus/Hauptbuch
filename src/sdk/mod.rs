//! Developer SDK Module
//!
//! This module provides comprehensive developer tools and SDKs for building
//! on the Hauptbuch blockchain, including Rust SDK, Solidity support via zkEVM,
//! and Move VM support with contract templates and advanced features.
//!
//! Key features:
//! - Rust SDK for type-safe blockchain interactions
//! - Solidity support with zkEVM integration
//! - Move VM support with resource management
//! - Contract templates and libraries
//! - Cross-chain development tools
//! - MEV protection integration
//! - Performance optimization tools
//! - Comprehensive testing frameworks

pub mod move_vm_support;
pub mod rust_sdk;
pub mod solidity_support;

// Re-export main types for convenience
pub use rust_sdk::{
    AccessListItem,
    Account,
    AccountType,
    CrossChainOperation,
    CrossChainOperationStatus,

    CrossChainOperationType,
    EventFilter,
    // Rust SDK types
    HauptbuchSDK,
    LogEntry,
    SDKConfig,
    // Error types
    SDKError,
    SDKMetrics,
    SDKResult,
    SmartContract,
    TransactionReceipt,
    TransactionRequest,
    TransactionType,
};

pub use solidity_support::{
    ABIFunction,
    ABIFunctionType,
    ABIParameter,
    CompilerMetrics,
    ContractCategory,
    ContractMetadata,
    OutputFormat,

    // Solidity support types
    SolidityCompiler,
    SolidityCompilerConfig,
    SolidityContract,
    SolidityContractTemplate,
    // Error types
    SolidityError,
    SolidityResult,
    StateMutability,
    TemplateParameter,
    ZkEVMContractData,
};

pub use move_vm_support::{
    FormalVerificationStatus,

    FunctionVisibility,
    ModuleDependency,
    MoveContractCategory,
    MoveContractTemplate,
    MoveFunction,
    MoveFunctionType,
    MoveModule,
    MoveModuleMetadata,
    MoveParameter,
    MoveResource,
    MoveTemplateParameter,
    MoveVMConfig,
    // Move VM support types
    MoveVMEngine,
    // Error types
    MoveVMError,
    MoveVMMetrics,
    MoveVMResult,
    ResourceMetadata,
};
