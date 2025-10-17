# Rust SDK

## Overview

The Rust SDK provides a comprehensive development toolkit for building applications on the Hauptbuch blockchain. It offers high-performance client libraries, cryptographic primitives, and development tools optimized for Rust developers.

## Key Features

- **High-Performance Client**: Optimized Rust client for blockchain interaction
- **Cryptographic Primitives**: Quantum-resistant cryptography integration
- **Smart Contract Support**: Full support for smart contract development and deployment
- **Account Abstraction**: Complete account abstraction implementation
- **Cross-Chain Support**: Multi-chain interoperability capabilities
- **Development Tools**: Comprehensive development and testing tools

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RUST SDK ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────────────┐    ┌───────────────────┐    ┌───────────┐  │
│  │   Client Layer   │    │  Crypto Layer     │    │  Tools    │  │
│  │   (RPC, WebSocket)│    │  (NIST PQC, ZKP) │    │  (Dev, Test)│  │
│  └─────────┬─────────┘    └─────────┬─────────┘    └─────┬─────┘  │
│            │                        │                      │        │
│            ▼                        ▼                      ▼        │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │             SDK Core & Application Framework                  │  │
│  │  (Account management, transaction handling, state management) │  │
│  └─────────┬─────────────────────────────────────────────────────┘  │
│            │                                                       │
│            ▼                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                 Hauptbuch Blockchain Network                   │  │
│  │             (Quantum-Resistant Cryptography Integration)      │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### Client Library

The main client library for blockchain interaction:

```rust
use hauptbuch_sdk::{
    Client, ClientBuilder, Account, Transaction, Block, 
    QuantumResistantCrypto, AccountAbstraction
};

pub struct HauptbuchClient {
    client: Client,
    account: Account,
    crypto: QuantumResistantCrypto,
}

impl HauptbuchClient {
    pub fn new(rpc_url: &str, private_key: &[u8]) -> Result<Self, SdkError> {
        let client = ClientBuilder::new()
            .rpc_url(rpc_url)
            .build()?;
        
        let account = Account::from_private_key(private_key)?;
        let crypto = QuantumResistantCrypto::new();
        
        Ok(Self {
            client,
            account,
            crypto,
        })
    }

    pub async fn send_transaction(&self, to: &str, amount: u64, data: &[u8]) -> Result<String, SdkError> {
        // Create transaction
        let transaction = Transaction::new()
            .from(self.account.address())
            .to(to)
            .value(amount)
            .data(data)
            .gas_limit(21000)
            .gas_price(20_000_000_000);

        // Sign transaction with quantum-resistant cryptography
        let signed_tx = self.crypto.sign_transaction(&transaction, &self.account.private_key())?;
        
        // Send transaction
        let tx_hash = self.client.send_transaction(&signed_tx).await?;
        Ok(tx_hash)
    }

    pub async fn get_balance(&self, address: &str) -> Result<u64, SdkError> {
        self.client.get_balance(address).await
    }

    pub async fn get_block(&self, block_number: u64) -> Result<Block, SdkError> {
        self.client.get_block(block_number).await
    }
}
```

### Account Management

Comprehensive account management with quantum-resistant security:

```rust
use hauptbuch_sdk::{Account, AccountType, QuantumResistantKeyPair};

pub struct AccountManager {
    accounts: HashMap<String, Account>,
    default_account: Option<String>,
}

impl AccountManager {
    pub fn new() -> Self {
        Self {
            accounts: HashMap::new(),
            default_account: None,
        }
    }

    pub fn create_account(&mut self, name: &str, account_type: AccountType) -> Result<Account, SdkError> {
        let keypair = match account_type {
            AccountType::QuantumResistant => {
                QuantumResistantKeyPair::generate_ml_kem_ml_dsa()?
            }
            AccountType::Hybrid => {
                QuantumResistantKeyPair::generate_hybrid()?
            }
            AccountType::Classical => {
                QuantumResistantKeyPair::generate_classical()?
            }
        };

        let account = Account::new(keypair, account_type);
        self.accounts.insert(name.to_string(), account.clone());
        
        if self.default_account.is_none() {
            self.default_account = Some(name.to_string());
        }

        Ok(account)
    }

    pub fn import_account(&mut self, name: &str, private_key: &[u8], account_type: AccountType) -> Result<Account, SdkError> {
        let keypair = QuantumResistantKeyPair::from_private_key(private_key, account_type)?;
        let account = Account::new(keypair, account_type);
        self.accounts.insert(name.to_string(), account.clone());
        Ok(account)
    }

    pub fn get_account(&self, name: &str) -> Option<&Account> {
        self.accounts.get(name)
    }

    pub fn set_default_account(&mut self, name: &str) -> Result<(), SdkError> {
        if self.accounts.contains_key(name) {
            self.default_account = Some(name.to_string());
            Ok(())
        } else {
            Err(SdkError::AccountNotFound)
        }
    }
}
```

### Smart Contract Support

Full smart contract development and deployment:

```rust
use hauptbuch_sdk::{Contract, ContractBuilder, ContractCall, ContractEvent};

pub struct SmartContractManager {
    client: Client,
    contracts: HashMap<String, Contract>,
}

impl SmartContractManager {
    pub fn new(client: Client) -> Self {
        Self {
            client,
            contracts: HashMap::new(),
        }
    }

    pub async fn deploy_contract(&mut self, name: &str, bytecode: &[u8], abi: &str) -> Result<String, SdkError> {
        // Create contract builder
        let contract_builder = ContractBuilder::new()
            .bytecode(bytecode)
            .abi(abi)
            .gas_limit(1_000_000)
            .gas_price(20_000_000_000);

        // Deploy contract
        let contract_address = self.client.deploy_contract(&contract_builder).await?;
        
        // Create contract instance
        let contract = Contract::new(contract_address, abi)?;
        self.contracts.insert(name.to_string(), contract);

        Ok(contract_address)
    }

    pub async fn call_contract(&self, contract_name: &str, method: &str, params: &[String]) -> Result<String, SdkError> {
        let contract = self.contracts.get(contract_name)
            .ok_or(SdkError::ContractNotFound)?;

        let call = ContractCall::new()
            .contract(contract.address())
            .method(method)
            .params(params);

        let result = self.client.call_contract(&call).await?;
        Ok(result)
    }

    pub async fn send_contract_transaction(&self, contract_name: &str, method: &str, params: &[String], account: &Account) -> Result<String, SdkError> {
        let contract = self.contracts.get(contract_name)
            .ok_or(SdkError::ContractNotFound)?;

        let call = ContractCall::new()
            .contract(contract.address())
            .method(method)
            .params(params);

        let transaction = self.client.create_contract_transaction(&call, account).await?;
        let signed_tx = account.sign_transaction(&transaction)?;
        let tx_hash = self.client.send_transaction(&signed_tx).await?;
        
        Ok(tx_hash)
    }
}
```

### Cross-Chain Support

Multi-chain interoperability capabilities:

```rust
use hauptbuch_sdk::{CrossChainManager, Bridge, IBC, CCIP};

pub struct CrossChainSDK {
    bridge: Bridge,
    ibc: IBC,
    ccip: CCIP,
}

impl CrossChainSDK {
    pub fn new() -> Result<Self, SdkError> {
        Ok(Self {
            bridge: Bridge::new()?,
            ibc: IBC::new()?,
            ccip: CCIP::new()?,
        })
    }

    pub async fn transfer_asset(&self, from_chain: &str, to_chain: &str, asset: &str, amount: u64, recipient: &str) -> Result<String, SdkError> {
        // Use appropriate cross-chain protocol
        match (from_chain, to_chain) {
            ("hauptbuch", "ethereum") | ("ethereum", "hauptbuch") => {
                self.bridge.transfer_asset(from_chain, to_chain, asset, amount, recipient).await
            }
            ("hauptbuch", "cosmos") | ("cosmos", "hauptbuch") => {
                self.ibc.transfer_asset(from_chain, to_chain, asset, amount, recipient).await
            }
            ("hauptbuch", "chainlink") | ("chainlink", "hauptbuch") => {
                self.ccip.transfer_asset(from_chain, to_chain, asset, amount, recipient).await
            }
            _ => Err(SdkError::UnsupportedChain)
        }
    }

    pub async fn get_cross_chain_balance(&self, chain: &str, address: &str, asset: &str) -> Result<u64, SdkError> {
        match chain {
            "ethereum" => self.bridge.get_balance(address, asset).await,
            "cosmos" => self.ibc.get_balance(address, asset).await,
            "chainlink" => self.ccip.get_balance(address, asset).await,
            "hauptbuch" => {
                // Use local client
                let client = Client::new("http://localhost:8545")?;
                client.get_balance(address).await
            }
            _ => Err(SdkError::UnsupportedChain)
        }
    }
}
```

## Quantum-Resistant Cryptography

### NIST PQC Integration

```rust
use hauptbuch_sdk::{MLKem, MLDsa, SLHDsa, HybridCrypto};

pub struct QuantumResistantCrypto {
    kem: MLKem,
    dsa: MLDsa,
    slh: SLHDsa,
    hybrid: HybridCrypto,
}

impl QuantumResistantCrypto {
    pub fn new() -> Self {
        Self {
            kem: MLKem::new(),
            dsa: MLDsa::new(),
            slh: SLHDsa::new(),
            hybrid: HybridCrypto::new(),
        }
    }

    pub fn generate_keypair(&self, scheme: CryptoScheme) -> Result<QuantumResistantKeyPair, SdkError> {
        match scheme {
            CryptoScheme::MLKem => {
                let (private_key, public_key) = self.kem.generate_keypair()?;
                Ok(QuantumResistantKeyPair::new(private_key, public_key, CryptoScheme::MLKem))
            }
            CryptoScheme::MLDsa => {
                let (private_key, public_key) = self.dsa.generate_keypair()?;
                Ok(QuantumResistantKeyPair::new(private_key, public_key, CryptoScheme::MLDsa))
            }
            CryptoScheme::SLHDsa => {
                let (private_key, public_key) = self.slh.generate_keypair()?;
                Ok(QuantumResistantKeyPair::new(private_key, public_key, CryptoScheme::SLHDsa))
            }
            CryptoScheme::Hybrid => {
                let (private_key, public_key) = self.hybrid.generate_keypair()?;
                Ok(QuantumResistantKeyPair::new(private_key, public_key, CryptoScheme::Hybrid))
            }
        }
    }

    pub fn sign_message(&self, message: &[u8], private_key: &[u8], scheme: CryptoScheme) -> Result<Vec<u8>, SdkError> {
        match scheme {
            CryptoScheme::MLDsa => self.dsa.sign(message, private_key),
            CryptoScheme::SLHDsa => self.slh.sign(message, private_key),
            CryptoScheme::Hybrid => self.hybrid.sign(message, private_key),
            _ => Err(SdkError::UnsupportedScheme)
        }
    }

    pub fn verify_signature(&self, message: &[u8], signature: &[u8], public_key: &[u8], scheme: CryptoScheme) -> Result<bool, SdkError> {
        match scheme {
            CryptoScheme::MLDsa => self.dsa.verify(message, signature, public_key),
            CryptoScheme::SLHDsa => self.slh.verify(message, signature, public_key),
            CryptoScheme::Hybrid => self.hybrid.verify(message, signature, public_key),
            _ => Err(SdkError::UnsupportedScheme)
        }
    }
}
```

### Zero-Knowledge Proof Support

```rust
use hauptbuch_sdk::{ZkProof, ZkVerifier, Binius, Plonky3, Halo2};

pub struct ZkProofManager {
    binius: Binius,
    plonky3: Plonky3,
    halo2: Halo2,
}

impl ZkProofManager {
    pub fn new() -> Self {
        Self {
            binius: Binius::new(),
            plonky3: Plonky3::new(),
            halo2: Halo2::new(),
        }
    }

    pub fn generate_proof(&self, circuit: &ZkCircuit, scheme: ZkScheme) -> Result<ZkProof, SdkError> {
        match scheme {
            ZkScheme::Binius => self.binius.generate_proof(circuit),
            ZkScheme::Plonky3 => self.plonky3.generate_proof(circuit),
            ZkScheme::Halo2 => self.halo2.generate_proof(circuit),
        }
    }

    pub fn verify_proof(&self, proof: &ZkProof, scheme: ZkScheme) -> Result<bool, SdkError> {
        match scheme {
            ZkScheme::Binius => self.binius.verify_proof(proof),
            ZkScheme::Plonky3 => self.plonky3.verify_proof(proof),
            ZkScheme::Halo2 => self.halo2.verify_proof(proof),
        }
    }
}
```

## Development Tools

### Testing Framework

```rust
use hauptbuch_sdk::{TestFramework, MockClient, TestAccount, TestContract};

pub struct SdkTestFramework {
    mock_client: MockClient,
    test_accounts: Vec<TestAccount>,
    test_contracts: Vec<TestContract>,
}

impl SdkTestFramework {
    pub fn new() -> Self {
        Self {
            mock_client: MockClient::new(),
            test_accounts: Vec::new(),
            test_contracts: Vec::new(),
        }
    }

    pub fn create_test_account(&mut self, name: &str, balance: u64) -> TestAccount {
        let account = TestAccount::new(name, balance);
        self.test_accounts.push(account.clone());
        account
    }

    pub fn deploy_test_contract(&mut self, name: &str, bytecode: &[u8], abi: &str) -> TestContract {
        let contract = TestContract::new(name, bytecode, abi);
        self.test_contracts.push(contract.clone());
        contract
    }

    pub fn run_tests(&self, test_suite: &TestSuite) -> Result<TestResults, SdkError> {
        let mut results = TestResults::new();
        
        for test in &test_suite.tests {
            let result = self.execute_test(test);
            results.add_result(test.name.clone(), result);
        }
        
        Ok(results)
    }

    fn execute_test(&self, test: &Test) -> TestResult {
        // Execute test logic
        match test.execute(&self.mock_client) {
            Ok(_) => TestResult::Passed,
            Err(e) => TestResult::Failed(e.to_string()),
        }
    }
}
```

### Development CLI

```rust
use hauptbuch_sdk::{Cli, Command, Subcommand};

pub struct SdkCli {
    cli: Cli,
}

impl SdkCli {
    pub fn new() -> Self {
        let cli = Cli::new()
            .name("hauptbuch-sdk")
            .version("1.0.0")
            .about("Hauptbuch Rust SDK CLI")
            .subcommand(Command::new("account")
                .about("Account management")
                .subcommand(Command::new("create")
                    .about("Create new account")
                    .arg(Arg::new("name").required(true))
                    .arg(Arg::new("type").required(true)))
                .subcommand(Command::new("import")
                    .about("Import existing account")
                    .arg(Arg::new("name").required(true))
                    .arg(Arg::new("private-key").required(true))))
            .subcommand(Command::new("contract")
                .about("Smart contract management")
                .subcommand(Command::new("deploy")
                    .about("Deploy contract")
                    .arg(Arg::new("bytecode").required(true))
                    .arg(Arg::new("abi").required(true)))
                .subcommand(Command::new("call")
                    .about("Call contract method")
                    .arg(Arg::new("address").required(true))
                    .arg(Arg::new("method").required(true))
                    .arg(Arg::new("params").required(true))))
            .subcommand(Command::new("transaction")
                .about("Transaction management")
                .subcommand(Command::new("send")
                    .about("Send transaction")
                    .arg(Arg::new("to").required(true))
                    .arg(Arg::new("amount").required(true))
                    .arg(Arg::new("data").required(false))));

        Self { cli }
    }

    pub fn run(&self) -> Result<(), SdkError> {
        let matches = self.cli.get_matches();
        
        match matches.subcommand() {
            Some(("account", sub_matches)) => {
                self.handle_account_command(sub_matches)?;
            }
            Some(("contract", sub_matches)) => {
                self.handle_contract_command(sub_matches)?;
            }
            Some(("transaction", sub_matches)) => {
                self.handle_transaction_command(sub_matches)?;
            }
            _ => {
                println!("Use --help for more information");
            }
        }
        
        Ok(())
    }
}
```

## Usage Examples

### Basic SDK Usage

```rust
use hauptbuch_sdk::{HauptbuchClient, AccountManager, SmartContractManager};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create client
    let client = HauptbuchClient::new("http://localhost:8545", &[0u8; 32])?;
    
    // Create account manager
    let mut account_manager = AccountManager::new();
    let account = account_manager.create_account("alice", AccountType::QuantumResistant)?;
    
    // Create smart contract manager
    let mut contract_manager = SmartContractManager::new(client.clone());
    
    // Deploy contract
    let bytecode = include_bytes!("contracts/SimpleStorage.bin");
    let abi = include_str!("contracts/SimpleStorage.abi");
    let contract_address = contract_manager.deploy_contract("SimpleStorage", bytecode, abi).await?;
    
    // Call contract method
    let result = contract_manager.call_contract("SimpleStorage", "get", &[]).await?;
    println!("Contract result: {}", result);
    
    Ok(())
}
```

### Quantum-Resistant Cryptography

```rust
use hauptbuch_sdk::{QuantumResistantCrypto, CryptoScheme};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create quantum-resistant crypto
    let crypto = QuantumResistantCrypto::new();
    
    // Generate keypair
    let keypair = crypto.generate_keypair(CryptoScheme::MLKem)?;
    
    // Sign message
    let message = b"Hello, Hauptbuch!";
    let signature = crypto.sign_message(message, &keypair.private_key(), CryptoScheme::MLDsa)?;
    
    // Verify signature
    let is_valid = crypto.verify_signature(message, &signature, &keypair.public_key(), CryptoScheme::MLDsa)?;
    println!("Signature is valid: {}", is_valid);
    
    Ok(())
}
```

### Cross-Chain Development

```rust
use hauptbuch_sdk::{CrossChainSDK, HauptbuchClient};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create cross-chain SDK
    let cross_chain = CrossChainSDK::new()?;
    
    // Transfer asset from Hauptbuch to Ethereum
    let tx_hash = cross_chain.transfer_asset(
        "hauptbuch",
        "ethereum",
        "ETH",
        1000000000000000000, // 1 ETH
        "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6"
    ).await?;
    
    println!("Cross-chain transfer: {}", tx_hash);
    
    // Get balance on Ethereum
    let balance = cross_chain.get_cross_chain_balance(
        "ethereum",
        "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
        "ETH"
    ).await?;
    
    println!("Ethereum balance: {}", balance);
    
    Ok(())
}
```

## Configuration

### SDK Configuration

```toml
[hauptbuch_sdk]
# RPC Configuration
rpc_url = "http://localhost:8545"
websocket_url = "ws://localhost:8546"
timeout_ms = 30000

# Account Configuration
default_account_type = "quantum_resistant"
key_derivation_path = "m/44'/60'/0'/0/0"

# Crypto Configuration
default_crypto_scheme = "ml_kem_ml_dsa"
hybrid_mode = true
classical_fallback = true

# Smart Contract Configuration
gas_limit = 1000000
gas_price = 20000000000
contract_timeout_ms = 60000

# Cross-Chain Configuration
bridge_enabled = true
ibc_enabled = true
ccip_enabled = true

# Development Configuration
test_mode = false
mock_client = false
debug_logging = true
```

## API Reference

### HauptbuchClient

```rust
impl HauptbuchClient {
    pub fn new(rpc_url: &str, private_key: &[u8]) -> Result<Self, SdkError>
    pub async fn send_transaction(&self, to: &str, amount: u64, data: &[u8]) -> Result<String, SdkError>
    pub async fn get_balance(&self, address: &str) -> Result<u64, SdkError>
    pub async fn get_block(&self, block_number: u64) -> Result<Block, SdkError>
    pub async fn get_transaction(&self, tx_hash: &str) -> Result<Transaction, SdkError>
}
```

### AccountManager

```rust
impl AccountManager {
    pub fn new() -> Self
    pub fn create_account(&mut self, name: &str, account_type: AccountType) -> Result<Account, SdkError>
    pub fn import_account(&mut self, name: &str, private_key: &[u8], account_type: AccountType) -> Result<Account, SdkError>
    pub fn get_account(&self, name: &str) -> Option<&Account>
    pub fn set_default_account(&mut self, name: &str) -> Result<(), SdkError>
}
```

### SmartContractManager

```rust
impl SmartContractManager {
    pub fn new(client: Client) -> Self
    pub async fn deploy_contract(&mut self, name: &str, bytecode: &[u8], abi: &str) -> Result<String, SdkError>
    pub async fn call_contract(&self, contract_name: &str, method: &str, params: &[String]) -> Result<String, SdkError>
    pub async fn send_contract_transaction(&self, contract_name: &str, method: &str, params: &[String], account: &Account) -> Result<String, SdkError>
}
```

## Error Handling

### SDK Errors

```rust
#[derive(Debug, thiserror::Error)]
pub enum SdkError {
    #[error("Client error: {0}")]
    ClientError(String),
    
    #[error("Account error: {0}")]
    AccountError(String),
    
    #[error("Contract error: {0}")]
    ContractError(String),
    
    #[error("Crypto error: {0}")]
    CryptoError(String),
    
    #[error("Cross-chain error: {0}")]
    CrossChainError(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Validation error: {0}")]
    ValidationError(String),
}
```

## Testing

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = HauptbuchClient::new("http://localhost:8545", &[0u8; 32]);
        assert!(client.is_ok());
    }

    #[tokio::test]
    async fn test_account_creation() {
        let mut account_manager = AccountManager::new();
        let account = account_manager.create_account("test", AccountType::QuantumResistant);
        assert!(account.is_ok());
    }

    #[tokio::test]
    async fn test_quantum_resistant_crypto() {
        let crypto = QuantumResistantCrypto::new();
        let keypair = crypto.generate_keypair(CryptoScheme::MLKem);
        assert!(keypair.is_ok());
    }
}
```

## Performance Benchmarks

### SDK Performance

```rust
#[cfg(test)]
mod benchmarks {
    use super::*;
    use criterion::{black_box, criterion_group, criterion_main, Criterion};

    fn bench_client_creation(c: &mut Criterion) {
        c.bench_function("client_creation", |b| {
            b.iter(|| {
                black_box(HauptbuchClient::new("http://localhost:8545", &[0u8; 32]).unwrap())
            })
        });
    }

    fn bench_account_creation(c: &mut Criterion) {
        c.bench_function("account_creation", |b| {
            b.iter(|| {
                let mut account_manager = AccountManager::new();
                black_box(account_manager.create_account("test", AccountType::QuantumResistant).unwrap())
            })
        });
    }

    criterion_group!(benches, bench_client_creation, bench_account_creation);
    criterion_main!(benches);
}
```

## Future Enhancements

### Planned Features

1. **Advanced Smart Contracts**: Enhanced smart contract development tools
2. **Performance Optimization**: Further performance improvements
3. **Additional Cryptography**: Support for more cryptographic schemes
4. **Enhanced Cross-Chain**: More cross-chain protocols
5. **Development Tools**: Advanced development tools

## Conclusion

The Rust SDK provides a comprehensive and high-performance development toolkit for the Hauptbuch blockchain. With quantum-resistant cryptography, smart contract support, and cross-chain capabilities, it enables developers to build secure and scalable applications on the Hauptbuch network.
