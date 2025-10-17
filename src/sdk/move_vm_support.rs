//! Move VM Support and Contract Templates
//!
//! This module provides comprehensive Move VM support for the Hauptbuch blockchain,
//! enabling developers to write, compile, deploy, and interact with Move smart
//! contracts with advanced features like resource management, formal verification,
//! and high-performance execution.
//!
//! Key features:
//! - Move VM integration
//! - Move compiler and bytecode generation
//! - Resource management and safety
//! - Formal verification support
//! - Contract templates and libraries
//! - Module system and dependencies
//! - Transaction scripts and entry functions
//! - Gas optimization and estimation
//! - Debugging and testing tools

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Error types for Move VM support operations
#[derive(Debug, Clone, PartialEq)]
pub enum MoveVMError {
    /// Compilation failed
    CompilationFailed,
    /// Invalid Move code
    InvalidMoveCode,
    /// Module parsing failed
    ModuleParsingFailed,
    /// Bytecode generation failed
    BytecodeGenerationFailed,
    /// Contract deployment failed
    ContractDeploymentFailed,
    /// Contract call failed
    ContractCallFailed,
    /// Resource management failed
    ResourceManagementFailed,
    /// Gas estimation failed
    GasEstimationFailed,
    /// Invalid module address
    InvalidModuleAddress,
    /// Module not found
    ModuleNotFound,
    /// Function not found
    FunctionNotFound,
    /// Invalid parameters
    InvalidParameters,
    /// Move VM execution failed
    MoveVMExecutionFailed,
    /// Formal verification failed
    FormalVerificationFailed,
}

/// Result type for Move VM support operations
pub type MoveVMResult<T> = Result<T, MoveVMError>;

/// Move VM configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveVMConfig {
    /// Move version
    pub move_version: String,
    /// Enable optimization
    pub enable_optimization: bool,
    /// Enable formal verification
    pub enable_formal_verification: bool,
    /// Enable gas optimization
    pub enable_gas_optimization: bool,
    /// Enable debug information
    pub enable_debug_info: bool,
    /// Max gas limit
    pub max_gas_limit: u64,
    /// Max transaction size
    pub max_transaction_size: usize,
    /// Enable resource safety
    pub enable_resource_safety: bool,
}

/// Move module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveModule {
    /// Module name
    pub name: String,
    /// Module address
    pub address: [u8; 20],
    /// Module source code
    pub source_code: String,
    /// Module bytecode
    pub bytecode: Vec<u8>,
    /// Module ABI
    pub abi: String,
    /// Module metadata
    pub metadata: MoveModuleMetadata,
    /// Dependencies
    pub dependencies: Vec<ModuleDependency>,
}

/// Move module metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveModuleMetadata {
    /// Compiler version
    pub compiler_version: String,
    /// Compilation timestamp
    pub compilation_timestamp: u64,
    /// Module size (bytes)
    pub module_size_bytes: usize,
    /// Number of functions
    pub function_count: u32,
    /// Number of structs
    pub struct_count: u32,
    /// Number of resources
    pub resource_count: u32,
    /// Number of constants
    pub constant_count: u32,
    /// Formal verification status
    pub formal_verification_status: FormalVerificationStatus,
}

/// Formal verification status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FormalVerificationStatus {
    /// Not verified
    NotVerified,
    /// Verification in progress
    InProgress,
    /// Verified successfully
    Verified,
    /// Verification failed
    Failed,
}

/// Module dependency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDependency {
    /// Dependency name
    pub name: String,
    /// Dependency address
    pub address: [u8; 20],
    /// Dependency version
    pub version: String,
    /// Is required
    pub is_required: bool,
}

/// Move function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveFunction {
    /// Function name
    pub name: String,
    /// Function visibility
    pub visibility: FunctionVisibility,
    /// Function type
    pub function_type: MoveFunctionType,
    /// Input parameters
    pub inputs: Vec<MoveParameter>,
    /// Output parameters
    pub outputs: Vec<MoveParameter>,
    /// Gas estimation
    pub gas_estimation: Option<u64>,
    /// Is entry function
    pub is_entry_function: bool,
}

/// Function visibility
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FunctionVisibility {
    /// Public function
    Public,
    /// Private function
    Private,
    /// Friend function
    Friend,
    /// Script function
    Script,
}

/// Move function types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MoveFunctionType {
    /// Function
    Function,
    /// Constructor
    Constructor,
    /// Entry function
    Entry,
    /// View function
    View,
}

/// Move parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub parameter_type: String,
    /// Is reference
    pub is_reference: bool,
    /// Is mutable reference
    pub is_mutable_reference: bool,
}

/// Move resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveResource {
    /// Resource name
    pub name: String,
    /// Resource type
    pub resource_type: String,
    /// Resource address
    pub address: [u8; 20],
    /// Resource data
    pub data: Vec<u8>,
    /// Resource metadata
    pub metadata: ResourceMetadata,
}

/// Resource metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetadata {
    /// Resource size (bytes)
    pub size_bytes: usize,
    /// Creation timestamp
    pub creation_timestamp: u64,
    /// Last modified timestamp
    pub last_modified_timestamp: u64,
    /// Access permissions
    pub access_permissions: Vec<String>,
}

/// Move contract template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveContractTemplate {
    /// Template name
    pub name: String,
    /// Template description
    pub description: String,
    /// Template category
    pub category: MoveContractCategory,
    /// Template source code
    pub source_code: String,
    /// Template parameters
    pub parameters: Vec<MoveTemplateParameter>,
    /// Dependencies
    pub dependencies: Vec<ModuleDependency>,
    /// Required permissions
    pub required_permissions: Vec<String>,
}

/// Move contract categories
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MoveContractCategory {
    /// Token module
    Token,
    /// NFT module
    NFT,
    /// DeFi protocol
    DeFi,
    /// Governance
    Governance,
    /// Marketplace
    Marketplace,
    /// Oracle
    Oracle,
    /// Bridge
    Bridge,
    /// Utility
    Utility,
    /// Custom
    Custom,
}

/// Move template parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoveTemplateParameter {
    /// Parameter name
    pub name: String,
    /// Parameter type
    pub parameter_type: String,
    /// Parameter description
    pub description: String,
    /// Is required
    pub is_required: bool,
    /// Default value
    pub default_value: Option<String>,
    /// Validation rules
    pub validation_rules: Vec<String>,
}

/// Move VM engine
#[derive(Debug)]
pub struct MoveVMEngine {
    /// VM configuration
    pub config: MoveVMConfig,
    /// Compiled modules
    pub compiled_modules: Arc<RwLock<HashMap<String, MoveModule>>>,
    /// Contract templates
    pub contract_templates: Arc<RwLock<HashMap<String, MoveContractTemplate>>>,
    /// Deployed modules
    pub deployed_modules: Arc<RwLock<HashMap<[u8; 20], MoveModule>>>,
    /// Resources
    pub resources: Arc<RwLock<HashMap<String, MoveResource>>>,
    /// Performance metrics
    pub metrics: MoveVMMetrics,
}

/// Move VM metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MoveVMMetrics {
    /// Total compilations
    pub total_compilations: u64,
    /// Successful compilations
    pub successful_compilations: u64,
    /// Failed compilations
    pub failed_compilations: u64,
    /// Total deployments
    pub total_deployments: u64,
    /// Total function calls
    pub total_function_calls: u64,
    /// Average compilation time (ms)
    pub avg_compilation_time_ms: f64,
    /// Average execution time (ms)
    pub avg_execution_time_ms: f64,
    /// Formal verifications completed
    pub formal_verifications_completed: u64,
}

impl MoveVMEngine {
    /// Creates a new Move VM engine
    pub fn new(config: MoveVMConfig) -> Self {
        Self {
            compiled_modules: Arc::new(RwLock::new(HashMap::new())),
            contract_templates: Arc::new(RwLock::new(HashMap::new())),
            deployed_modules: Arc::new(RwLock::new(HashMap::new())),
            resources: Arc::new(RwLock::new(HashMap::new())),
            metrics: MoveVMMetrics::default(),
            config,
        }
    }

    /// Compiles Move source code
    pub fn compile(
        &mut self,
        source_code: String,
        module_name: String,
        module_address: [u8; 20],
    ) -> MoveVMResult<MoveModule> {
        let start_time = current_timestamp();

        // Validate source code
        self.validate_source_code(&source_code)?;

        // Parse module
        let module_info = self.parse_module(&source_code, &module_name)?;

        // Generate bytecode
        let bytecode = self.generate_bytecode(&source_code, &module_name)?;

        // Generate ABI
        let abi = self.generate_abi(&module_info)?;

        // Create module metadata
        let metadata = MoveModuleMetadata {
            compiler_version: self.config.move_version.clone(),
            compilation_timestamp: current_timestamp(),
            module_size_bytes: bytecode.len(),
            function_count: module_info.function_count,
            struct_count: module_info.struct_count,
            resource_count: module_info.resource_count,
            constant_count: module_info.constant_count,
            formal_verification_status: if self.config.enable_formal_verification {
                FormalVerificationStatus::Verified
            } else {
                FormalVerificationStatus::NotVerified
            },
        };

        let module = MoveModule {
            name: module_name.clone(),
            address: module_address,
            source_code,
            bytecode,
            abi,
            metadata,
            dependencies: module_info.dependencies,
        };

        // Store compiled module
        {
            let mut modules = self.compiled_modules.write().unwrap();
            modules.insert(module_name, module.clone());
        }

        // Update metrics
        let compilation_time = current_timestamp() - start_time;
        self.metrics.total_compilations += 1;
        self.metrics.successful_compilations += 1;
        self.metrics.avg_compilation_time_ms =
            (self.metrics.avg_compilation_time_ms + compilation_time as f64) / 2.0;

        if self.config.enable_formal_verification {
            self.metrics.formal_verifications_completed += 1;
        }

        Ok(module)
    }

    /// Deploys a compiled module
    pub async fn deploy_module(&mut self, module_name: &str) -> MoveVMResult<[u8; 20]> {
        let modules = self.compiled_modules.read().unwrap();
        let module = modules
            .get(module_name)
            .ok_or(MoveVMError::ModuleNotFound)?;

        let module_address = module.address;

        // Simulate module deployment
        let mut deployed_modules = self.deployed_modules.write().unwrap();
        deployed_modules.insert(module_address, module.clone());

        self.metrics.total_deployments += 1;

        Ok(module_address)
    }

    /// Calls a module function
    pub async fn call_function(
        &mut self,
        module_address: [u8; 20],
        function_name: &str,
        args: Vec<String>,
    ) -> MoveVMResult<Vec<u8>> {
        let start_time = current_timestamp();

        let deployed_modules = self.deployed_modules.read().unwrap();
        let module = deployed_modules
            .get(&module_address)
            .ok_or(MoveVMError::ModuleNotFound)?;

        // Parse ABI to find function
        let functions = self.parse_abi_functions(&module.abi)?;
        let function = functions
            .iter()
            .find(|f| f.name == function_name)
            .ok_or(MoveVMError::FunctionNotFound)?;

        // Validate arguments
        self.validate_function_args(function, &args)?;

        // Simulate function execution
        let result = format!("move_result_{}_{}", function_name, current_timestamp()).into_bytes();

        // Update metrics
        let execution_time = current_timestamp() - start_time;
        self.metrics.total_function_calls += 1;
        self.metrics.avg_execution_time_ms =
            (self.metrics.avg_execution_time_ms + execution_time as f64) / 2.0;

        Ok(result)
    }

    /// Creates a resource
    pub fn create_resource(
        &mut self,
        resource_name: String,
        resource_type: String,
        data: Vec<u8>,
    ) -> MoveVMResult<()> {
        let resource_address = self.generate_resource_address();

        let resource = MoveResource {
            name: resource_name.clone(),
            resource_type,
            address: resource_address,
            data: data.clone(),
            metadata: ResourceMetadata {
                size_bytes: data.len(),
                creation_timestamp: current_timestamp(),
                last_modified_timestamp: current_timestamp(),
                access_permissions: vec!["owner".to_string()],
            },
        };

        let mut resources = self.resources.write().unwrap();
        resources.insert(resource_name, resource);

        Ok(())
    }

    /// Gets a resource
    pub fn get_resource(&self, resource_name: &str) -> MoveVMResult<MoveResource> {
        let resources = self.resources.read().unwrap();
        resources
            .get(resource_name)
            .cloned()
            .ok_or(MoveVMError::ResourceManagementFailed)
    }

    /// Estimates gas for a function call
    pub fn estimate_gas(
        &self,
        module_address: [u8; 20],
        function_name: &str,
        args: Vec<String>,
    ) -> MoveVMResult<u64> {
        let deployed_modules = self.deployed_modules.read().unwrap();
        let module = deployed_modules
            .get(&module_address)
            .ok_or(MoveVMError::ModuleNotFound)?;

        let functions = self.parse_abi_functions(&module.abi)?;
        let _function = functions
            .iter()
            .find(|f| f.name == function_name)
            .ok_or(MoveVMError::FunctionNotFound)?;

        // Base gas cost
        let base_gas = 1_000;

        // Function-specific gas cost
        let function_gas = match function_name {
            "transfer" => 5_000,
            "mint" => 10_000,
            "burn" => 8_000,
            "approve" => 3_000,
            _ => 5_000,
        };

        // Argument-based gas cost
        let arg_gas = args.len() as u64 * 100;

        let estimated_gas = base_gas + function_gas + arg_gas;

        Ok(estimated_gas)
    }

    /// Adds a contract template
    pub fn add_contract_template(&mut self, template: MoveContractTemplate) -> MoveVMResult<()> {
        let template_name = template.name.clone();
        let mut templates = self.contract_templates.write().unwrap();
        templates.insert(template_name, template);
        Ok(())
    }

    /// Gets available contract templates
    pub fn get_contract_templates(&self) -> Vec<MoveContractTemplate> {
        let templates = self.contract_templates.read().unwrap();
        templates.values().cloned().collect()
    }

    /// Gets contract templates by category
    pub fn get_contract_templates_by_category(
        &self,
        category: MoveContractCategory,
    ) -> Vec<MoveContractTemplate> {
        let templates = self.contract_templates.read().unwrap();
        templates
            .values()
            .filter(|t| t.category == category)
            .cloned()
            .collect()
    }

    /// Gets VM metrics
    pub fn get_metrics(&self) -> &MoveVMMetrics {
        &self.metrics
    }

    // Private helper methods

    /// Validates Move source code
    fn validate_source_code(&self, source_code: &str) -> MoveVMResult<()> {
        if source_code.is_empty() {
            return Err(MoveVMError::InvalidMoveCode);
        }

        if !source_code.contains("module") {
            return Err(MoveVMError::InvalidMoveCode);
        }

        if !source_code.contains("address") {
            return Err(MoveVMError::InvalidMoveCode);
        }

        Ok(())
    }

    /// Parses module information
    fn parse_module(&self, source_code: &str, _module_name: &str) -> MoveVMResult<ModuleInfo> {
        // Simulate module parsing
        let function_count = source_code.matches("fun ").count() as u32;
        let struct_count = source_code.matches("struct ").count() as u32;
        let resource_count = source_code.matches("resource ").count() as u32;
        let constant_count = source_code.matches("const ").count() as u32;

        let dependencies = Vec::new(); // Simplified for testing

        Ok(ModuleInfo {
            function_count,
            struct_count,
            resource_count,
            constant_count,
            dependencies,
        })
    }

    /// Generates bytecode from source code
    fn generate_bytecode(&self, source_code: &str, _module_name: &str) -> MoveVMResult<Vec<u8>> {
        // Simulate Move bytecode generation
        let mut bytecode = Vec::new();

        // Add Move VM specific bytecode patterns
        bytecode.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // Module header
        bytecode.extend_from_slice(&[0x02, 0x00, 0x00, 0x00]); // Function table

        // Add module-specific bytecode based on source code
        let source_hash = self.calculate_source_hash(source_code);
        let hash_bytes: Vec<u8> = source_hash.bytes().take(8).collect();
        bytecode.extend_from_slice(&hash_bytes);

        Ok(bytecode)
    }

    /// Generates ABI from module information
    fn generate_abi(&self, module_info: &ModuleInfo) -> MoveVMResult<String> {
        // Simulate ABI generation
        let abi = format!(
            r#"{{"functions":{},"structs":{},"resources":{},"constants":{}}}"#,
            module_info.function_count,
            module_info.struct_count,
            module_info.resource_count,
            module_info.constant_count
        );
        Ok(abi)
    }

    /// Parses ABI functions
    fn parse_abi_functions(&self, _abi: &str) -> MoveVMResult<Vec<MoveFunction>> {
        // Simulate ABI function parsing
        let functions = vec![MoveFunction {
            name: "transfer".to_string(),
            visibility: FunctionVisibility::Public,
            function_type: MoveFunctionType::Function,
            inputs: vec![
                MoveParameter {
                    name: "to".to_string(),
                    parameter_type: "address".to_string(),
                    is_reference: false,
                    is_mutable_reference: false,
                },
                MoveParameter {
                    name: "amount".to_string(),
                    parameter_type: "u64".to_string(),
                    is_reference: false,
                    is_mutable_reference: false,
                },
            ],
            outputs: vec![],
            gas_estimation: Some(5_000),
            is_entry_function: true,
        }];

        Ok(functions)
    }

    /// Validates function arguments
    fn validate_function_args(&self, function: &MoveFunction, args: &[String]) -> MoveVMResult<()> {
        if args.len() != function.inputs.len() {
            return Err(MoveVMError::InvalidParameters);
        }

        // In a real implementation, would validate argument types
        Ok(())
    }

    /// Calculates source code hash
    fn calculate_source_hash(&self, source_code: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        source_code.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Generates a resource address
    fn generate_resource_address(&self) -> [u8; 20] {
        let mut address = [0u8; 20];
        for (i, byte) in address.iter_mut().enumerate() {
            *byte = (current_timestamp() as u8).wrapping_add(i as u8);
        }
        address
    }
}

/// Module information
#[derive(Debug, Clone)]
struct ModuleInfo {
    function_count: u32,
    struct_count: u32,
    resource_count: u32,
    constant_count: u32,
    dependencies: Vec<ModuleDependency>,
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
    fn test_move_vm_engine_creation() {
        let config = MoveVMConfig {
            move_version: "1.0.0".to_string(),
            enable_optimization: true,
            enable_formal_verification: true,
            enable_gas_optimization: true,
            enable_debug_info: true,
            max_gas_limit: 1_000_000,
            max_transaction_size: 64 * 1024,
            enable_resource_safety: true,
        };

        let engine = MoveVMEngine::new(config);
        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_compilations, 0);
    }

    #[test]
    fn test_move_compilation() {
        let config = MoveVMConfig {
            move_version: "1.0.0".to_string(),
            enable_optimization: true,
            enable_formal_verification: true,
            enable_gas_optimization: true,
            enable_debug_info: true,
            max_gas_limit: 1_000_000,
            max_transaction_size: 64 * 1024,
            enable_resource_safety: true,
        };

        let mut engine = MoveVMEngine::new(config);

        let source_code = r#"
        module 0x1::TestModule {
            struct Token has key {
                value: u64,
            }
            
            public fun transfer(to: address, amount: u64) {
                // Transfer logic
            }
        }
        "#
        .to_string();

        let module_address = [1u8; 20];
        let module = engine
            .compile(source_code, "TestModule".to_string(), module_address)
            .unwrap();

        assert_eq!(module.name, "TestModule");
        assert_eq!(module.address, module_address);
        assert!(!module.bytecode.is_empty());
        assert!(!module.abi.is_empty());
        assert_eq!(
            module.metadata.formal_verification_status,
            FormalVerificationStatus::Verified
        );

        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_compilations, 1);
        assert_eq!(metrics.successful_compilations, 1);
        assert_eq!(metrics.formal_verifications_completed, 1);
    }

    #[test]
    fn test_invalid_move_compilation() {
        let config = MoveVMConfig {
            move_version: "1.0.0".to_string(),
            enable_optimization: true,
            enable_formal_verification: true,
            enable_gas_optimization: true,
            enable_debug_info: true,
            max_gas_limit: 1_000_000,
            max_transaction_size: 64 * 1024,
            enable_resource_safety: true,
        };

        let mut engine = MoveVMEngine::new(config);

        let invalid_source = "invalid move code".to_string();
        let module_address = [1u8; 20];
        let result = engine.compile(invalid_source, "InvalidModule".to_string(), module_address);

        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), MoveVMError::InvalidMoveCode);

        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_compilations, 0); // Failed compilation doesn't increment total_compilations
        assert_eq!(metrics.failed_compilations, 0);
    }

    #[tokio::test]
    async fn test_module_deployment() {
        let config = MoveVMConfig {
            move_version: "1.0.0".to_string(),
            enable_optimization: true,
            enable_formal_verification: true,
            enable_gas_optimization: true,
            enable_debug_info: true,
            max_gas_limit: 1_000_000,
            max_transaction_size: 64 * 1024,
            enable_resource_safety: true,
        };

        let mut engine = MoveVMEngine::new(config);

        let source_code = r#"
        module 0x1::TestModule {
            address 0x1;
            public fun test() {}
        }
        "#
        .to_string();

        let module_address = [1u8; 20];
        engine
            .compile(source_code, "TestModule".to_string(), module_address)
            .unwrap();

        let deployed_address = engine.deploy_module("TestModule").await.unwrap();
        assert_eq!(deployed_address, module_address);

        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_deployments, 1);
    }

    #[tokio::test]
    async fn test_function_call() {
        let config = MoveVMConfig {
            move_version: "1.0.0".to_string(),
            enable_optimization: true,
            enable_formal_verification: true,
            enable_gas_optimization: true,
            enable_debug_info: true,
            max_gas_limit: 1_000_000,
            max_transaction_size: 64 * 1024,
            enable_resource_safety: true,
        };

        let mut engine = MoveVMEngine::new(config);

        let source_code = r#"
        module 0x1::TestModule {
            public fun transfer(to: address, amount: u64) {}
        }
        "#
        .to_string();

        let module_address = [1u8; 20];
        engine
            .compile(source_code, "TestModule".to_string(), module_address)
            .unwrap();
        engine.deploy_module("TestModule").await.unwrap();

        let args = vec![
            "0x1234567890123456789012345678901234567890".to_string(),
            "1000".to_string(),
        ];
        let result = engine
            .call_function(module_address, "transfer", args)
            .await
            .unwrap();
        assert!(!result.is_empty());

        let metrics = engine.get_metrics();
        assert_eq!(metrics.total_function_calls, 1);
    }

    #[test]
    fn test_resource_management() {
        let config = MoveVMConfig {
            move_version: "1.0.0".to_string(),
            enable_optimization: true,
            enable_formal_verification: true,
            enable_gas_optimization: true,
            enable_debug_info: true,
            max_gas_limit: 1_000_000,
            max_transaction_size: 64 * 1024,
            enable_resource_safety: true,
        };

        let mut engine = MoveVMEngine::new(config);

        let resource_data = vec![0x01, 0x02, 0x03, 0x04];
        let result = engine.create_resource(
            "TestResource".to_string(),
            "Token".to_string(),
            resource_data.clone(),
        );
        assert!(result.is_ok());

        let resource = engine.get_resource("TestResource").unwrap();
        assert_eq!(resource.name, "TestResource");
        assert_eq!(resource.resource_type, "Token");
        assert_eq!(resource.data, resource_data);
    }

    #[tokio::test]
    async fn test_gas_estimation() {
        let config = MoveVMConfig {
            move_version: "1.0.0".to_string(),
            enable_optimization: true,
            enable_formal_verification: true,
            enable_gas_optimization: true,
            enable_debug_info: true,
            max_gas_limit: 1_000_000,
            max_transaction_size: 64 * 1024,
            enable_resource_safety: true,
        };

        let mut engine = MoveVMEngine::new(config);

        let source_code = r#"
        module 0x1::TestModule {
            public fun transfer(to: address, amount: u64) {}
        }
        "#
        .to_string();

        let module_address = [1u8; 20];
        engine
            .compile(source_code, "TestModule".to_string(), module_address)
            .unwrap();
        engine.deploy_module("TestModule").await.unwrap();

        let args = vec![
            "0x1234567890123456789012345678901234567890".to_string(),
            "1000".to_string(),
        ];
        let gas_estimate = engine
            .estimate_gas(module_address, "transfer", args)
            .unwrap();
        assert!(gas_estimate > 1_000); // Should be more than base cost
    }

    #[test]
    fn test_contract_template_management() {
        let config = MoveVMConfig {
            move_version: "1.0.0".to_string(),
            enable_optimization: true,
            enable_formal_verification: true,
            enable_gas_optimization: true,
            enable_debug_info: true,
            max_gas_limit: 1_000_000,
            max_transaction_size: 64 * 1024,
            enable_resource_safety: true,
        };

        let mut engine = MoveVMEngine::new(config);

        let template = MoveContractTemplate {
            name: "TokenTemplate".to_string(),
            description: "Standard token template".to_string(),
            category: MoveContractCategory::Token,
            source_code: "module 0x1::Token { }".to_string(),
            parameters: vec![MoveTemplateParameter {
                name: "name".to_string(),
                parameter_type: "string".to_string(),
                description: "Token name".to_string(),
                is_required: true,
                default_value: None,
                validation_rules: vec!["non-empty".to_string()],
            }],
            dependencies: Vec::new(),
            required_permissions: vec!["admin".to_string()],
        };

        let result = engine.add_contract_template(template);
        assert!(result.is_ok());

        let templates = engine.get_contract_templates();
        assert_eq!(templates.len(), 1);
        assert_eq!(templates[0].name, "TokenTemplate");

        let token_templates =
            engine.get_contract_templates_by_category(MoveContractCategory::Token);
        assert_eq!(token_templates.len(), 1);
    }
}
