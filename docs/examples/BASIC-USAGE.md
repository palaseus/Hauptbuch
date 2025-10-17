# Basic Usage Examples

## Overview

This document provides comprehensive examples of basic usage for the Hauptbuch blockchain platform. Learn how to create accounts, send transactions, interact with smart contracts, and use core blockchain functionality.

## Table of Contents

- [Getting Started](#getting-started)
- [Account Management](#account-management)
- [Transaction Handling](#transaction-handling)
- [Smart Contract Interaction](#smart-contract-interaction)
- [Network Operations](#network-operations)
- [Cryptography Examples](#cryptography-examples)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Getting Started

### Basic Client Setup

```rust
use hauptbuch_sdk::{Client, ClientBuilder, SdkError};

#[tokio::main]
async fn main() -> Result<(), SdkError> {
    // Create a new client
    let client = ClientBuilder::new()
        .rpc_url("http://localhost:8080")
        .build()?;
    
    // Get network information
    let network_info = client.get_network_info().await?;
    println!("Network: {}", network_info.network_id);
    println!("Chain ID: {}", network_info.chain_id);
    
    Ok(())
}
```

### Environment Setup

```rust
use hauptbuch_sdk::{Client, Account, QuantumResistantCrypto};

// Set up environment variables
std::env::set_var("HAUPTBUCH_RPC_URL", "http://localhost:8080");
std::env::set_var("HAUPTBUCH_PRIVATE_KEY", "your_private_key_here");

// Create client with environment variables
let client = Client::from_env()?;
```

## Account Management

### Creating a New Account

```rust
use hauptbuch_sdk::{Account, AccountBuilder, QuantumResistantCrypto};

// Create a new account with quantum-resistant cryptography
let account = AccountBuilder::new()
    .generate_keypair()
    .quantum_resistant(true)
    .build()?;

println!("Address: {}", account.address());
println!("Public Key: {}", hex::encode(account.public_key()));
```

### Loading an Existing Account

```rust
use hauptbuch_sdk::{Account, AccountBuilder};

// Load account from private key
let private_key = hex::decode("your_private_key_hex")?;
let account = AccountBuilder::new()
    .private_key(&private_key)
    .build()?;

println!("Address: {}", account.address());
```

### Account Balance

```rust
use hauptbuch_sdk::{Client, Account};

async fn check_balance(client: &Client, account: &Account) -> Result<(), SdkError> {
    let balance = client.get_balance(&account.address()).await?;
    println!("Balance: {} wei", balance);
    println!("Balance: {} HBK", balance / 1_000_000_000_000_000_000);
    Ok(())
}
```

### Account Information

```rust
use hauptbuch_sdk::{Client, Account};

async fn get_account_info(client: &Client, address: &str) -> Result<(), SdkError> {
    let balance = client.get_balance(address).await?;
    let nonce = client.get_nonce(address).await?;
    let code = client.get_code(address).await?;
    
    println!("Account: {}", address);
    println!("Balance: {} wei", balance);
    println!("Nonce: {}", nonce);
    println!("Code: {}", hex::encode(code));
    
    Ok(())
}
```

## Transaction Handling

### Basic Transaction

```rust
use hauptbuch_sdk::{Client, Account, Transaction, TransactionBuilder};

async fn send_transaction(
    client: &Client,
    account: &Account,
    to: &str,
    value: u64,
) -> Result<String, SdkError> {
    // Create transaction
    let transaction = TransactionBuilder::new()
        .from(&account.address())
        .to(to)
        .value(value)
        .gas_limit(21000)
        .gas_price(20_000_000_000)
        .build()?;
    
    // Sign transaction
    let signed_tx = account.sign_transaction(&transaction)?;
    
    // Send transaction
    let tx_hash = client.send_transaction(&signed_tx).await?;
    
    println!("Transaction sent: {}", tx_hash);
    Ok(tx_hash)
}
```

### Transaction with Data

```rust
use hauptbuch_sdk::{Client, Account, Transaction, TransactionBuilder};

async fn send_contract_transaction(
    client: &Client,
    account: &Account,
    contract_address: &str,
    data: &[u8],
) -> Result<String, SdkError> {
    // Create transaction with data
    let transaction = TransactionBuilder::new()
        .from(&account.address())
        .to(contract_address)
        .value(0)
        .data(data)
        .gas_limit(100_000)
        .gas_price(20_000_000_000)
        .build()?;
    
    // Sign and send transaction
    let signed_tx = account.sign_transaction(&transaction)?;
    let tx_hash = client.send_transaction(&signed_tx).await?;
    
    println!("Contract transaction sent: {}", tx_hash);
    Ok(tx_hash)
}
```

### Transaction Status

```rust
use hauptbuch_sdk::{Client, TransactionStatus};

async fn check_transaction_status(
    client: &Client,
    tx_hash: &str,
) -> Result<(), SdkError> {
    let status = client.get_transaction_status(tx_hash).await?;
    
    match status {
        TransactionStatus::Pending => println!("Transaction is pending"),
        TransactionStatus::Confirmed => println!("Transaction is confirmed"),
        TransactionStatus::Failed => println!("Transaction failed"),
    }
    
    Ok(())
}
```

### Transaction History

```rust
use hauptbuch_sdk::{Client, Transaction};

async fn get_transaction_history(
    client: &Client,
    address: &str,
    limit: u32,
) -> Result<(), SdkError> {
    let transactions = client.get_transaction_history(address, limit).await?;
    
    for tx in transactions {
        println!("Hash: {}", tx.hash());
        println!("From: {}", tx.from());
        println!("To: {}", tx.to());
        println!("Value: {}", tx.value());
        println!("Block: {}", tx.block_number());
        println!("---");
    }
    
    Ok(())
}
```

## Smart Contract Interaction

### Deploying a Smart Contract

```rust
use hauptbuch_sdk::{Client, Account, Contract, ContractBuilder};

async fn deploy_contract(
    client: &Client,
    account: &Account,
    bytecode: &[u8],
) -> Result<String, SdkError> {
    // Create contract
    let contract = ContractBuilder::new()
        .bytecode(bytecode)
        .build()?;
    
    // Deploy contract
    let contract_address = client.deploy_contract(&contract).await?;
    
    println!("Contract deployed at: {}", contract_address);
    Ok(contract_address)
}
```

### Calling Contract Functions

```rust
use hauptbuch_sdk::{Client, Account, Contract, ContractFunction};

async fn call_contract_function(
    client: &Client,
    account: &Account,
    contract_address: &str,
    function_name: &str,
    args: Vec<String>,
) -> Result<String, SdkError> {
    // Create contract instance
    let contract = Contract::new(contract_address)?;
    
    // Call function
    let result = client.call_contract_function(
        &contract,
        function_name,
        &args,
    ).await?;
    
    println!("Function result: {}", result);
    Ok(result)
}
```

### Contract Events

```rust
use hauptbuch_sdk::{Client, Contract, ContractEvent};

async fn listen_to_contract_events(
    client: &Client,
    contract_address: &str,
    event_name: &str,
) -> Result<(), SdkError> {
    let contract = Contract::new(contract_address)?;
    
    // Subscribe to events
    let mut event_stream = client.subscribe_to_contract_events(
        &contract,
        event_name,
    ).await?;
    
    while let Some(event) = event_stream.next().await {
        println!("Event received: {:?}", event);
    }
    
    Ok(())
}
```

## Network Operations

### Network Information

```rust
use hauptbuch_sdk::{Client, NetworkInfo};

async fn get_network_info(client: &Client) -> Result<(), SdkError> {
    let info = client.get_network_info().await?;
    
    println!("Network ID: {}", info.network_id);
    println!("Chain ID: {}", info.chain_id);
    println!("Node Version: {}", info.node_version);
    println!("Protocol Version: {}", info.protocol_version);
    println!("Genesis Hash: {}", info.genesis_hash);
    
    Ok(())
}
```

### Peer Information

```rust
use hauptbuch_sdk::{Client, PeerInfo};

async fn get_peer_info(client: &Client) -> Result<(), SdkError> {
    let peers = client.get_peer_list().await?;
    
    for peer in peers {
        println!("Peer ID: {}", peer.id);
        println!("Address: {}", peer.address);
        println!("Status: {}", peer.status);
        println!("Capabilities: {:?}", peer.capabilities);
        println!("---");
    }
    
    Ok(())
}
```

### Block Information

```rust
use hauptbuch_sdk::{Client, Block};

async fn get_block_info(client: &Client, block_number: u64) -> Result<(), SdkError> {
    let block = client.get_block(block_number).await?;
    
    println!("Block Number: {}", block.number());
    println!("Block Hash: {}", block.hash());
    println!("Parent Hash: {}", block.parent_hash());
    println!("Timestamp: {}", block.timestamp());
    println!("Gas Limit: {}", block.gas_limit());
    println!("Gas Used: {}", block.gas_used());
    println!("Transaction Count: {}", block.transactions().len());
    
    Ok(())
}
```

## Cryptography Examples

### Quantum-Resistant Key Generation

```rust
use hauptbuch_crypto::{MLKem, MLDsa, SLHDsa, QuantumResistantCrypto};

fn generate_quantum_resistant_keys() -> Result<(), CryptoError> {
    // Generate ML-KEM keypair
    let (ml_kem_private, ml_kem_public) = MLKem::generate_keypair()?;
    println!("ML-KEM Private Key: {}", hex::encode(&ml_kem_private));
    println!("ML-KEM Public Key: {}", hex::encode(&ml_kem_public));
    
    // Generate ML-DSA keypair
    let (ml_dsa_private, ml_dsa_public) = MLDsa::generate_keypair()?;
    println!("ML-DSA Private Key: {}", hex::encode(&ml_dsa_private));
    println!("ML-DSA Public Key: {}", hex::encode(&ml_dsa_public));
    
    // Generate SLH-DSA keypair
    let (slh_dsa_private, slh_dsa_public) = SLHDsa::generate_keypair()?;
    println!("SLH-DSA Private Key: {}", hex::encode(&slh_dsa_private));
    println!("SLH-DSA Public Key: {}", hex::encode(&slh_dsa_public));
    
    Ok(())
}
```

### Message Signing and Verification

```rust
use hauptbuch_crypto::{MLDsa, SLHDsa, HybridCrypto};

fn sign_and_verify_message() -> Result<(), CryptoError> {
    let message = b"Hello, Hauptbuch!";
    
    // ML-DSA signing
    let (ml_dsa_private, ml_dsa_public) = MLDsa::generate_keypair()?;
    let ml_dsa_signature = MLDsa::sign(message, &ml_dsa_private)?;
    let ml_dsa_valid = MLDsa::verify(message, &ml_dsa_signature, &ml_dsa_public)?;
    println!("ML-DSA Signature Valid: {}", ml_dsa_valid);
    
    // SLH-DSA signing
    let (slh_dsa_private, slh_dsa_public) = SLHDsa::generate_keypair()?;
    let slh_dsa_signature = SLH-Dsa::sign(message, &slh_dsa_private)?;
    let slh_dsa_valid = SLH-Dsa::verify(message, &slh_dsa_signature, &slh_dsa_public)?;
    println!("SLH-DSA Signature Valid: {}", slh_dsa_valid);
    
    // Hybrid signing
    let hybrid_crypto = HybridCrypto::new();
    let hybrid_signature = hybrid_crypto.sign(message)?;
    let hybrid_valid = hybrid_crypto.verify(message, &hybrid_signature)?;
    println!("Hybrid Signature Valid: {}", hybrid_valid);
    
    Ok(())
}
```

### Key Exchange

```rust
use hauptbuch_crypto::{MLKem, KeyExchange};

fn perform_key_exchange() -> Result<(), CryptoError> {
    // Alice generates keypair
    let (alice_private, alice_public) = MLKem::generate_keypair()?;
    
    // Bob generates keypair
    let (bob_private, bob_public) = MLKem::generate_keypair()?;
    
    // Alice encrypts message for Bob
    let message = b"Secret message";
    let ciphertext = MLKem::encrypt(message, &bob_public)?;
    
    // Bob decrypts message
    let decrypted = MLKem::decrypt(&ciphertext, &bob_private)?;
    
    println!("Original: {}", String::from_utf8_lossy(message));
    println!("Decrypted: {}", String::from_utf8_lossy(&decrypted));
    
    Ok(())
}
```

## Error Handling

### Basic Error Handling

```rust
use hauptbuch_sdk::{Client, SdkError};

async fn handle_errors(client: &Client) -> Result<(), SdkError> {
    match client.get_balance("invalid_address").await {
        Ok(balance) => println!("Balance: {}", balance),
        Err(SdkError::ValidationError(msg)) => {
            println!("Validation error: {}", msg);
        }
        Err(SdkError::NetworkError(msg)) => {
            println!("Network error: {}", msg);
        }
        Err(e) => {
            println!("Other error: {}", e);
        }
    }
    
    Ok(())
}
```

### Retry Logic

```rust
use hauptbuch_sdk::{Client, SdkError};
use tokio::time::{sleep, Duration};

async fn retry_operation(client: &Client, max_retries: u32) -> Result<(), SdkError> {
    let mut retries = 0;
    
    loop {
        match client.get_network_info().await {
            Ok(info) => {
                println!("Network info: {:?}", info);
                return Ok(());
            }
            Err(SdkError::NetworkError(_)) if retries < max_retries => {
                retries += 1;
                println!("Retry {} of {}", retries, max_retries);
                sleep(Duration::from_secs(1)).await;
            }
            Err(e) => return Err(e),
        }
    }
}
```

### Error Recovery

```rust
use hauptbuch_sdk::{Client, SdkError};

async fn recover_from_errors(client: &Client) -> Result<(), SdkError> {
    // Try to get balance
    let balance = match client.get_balance("0x1234...").await {
        Ok(balance) => balance,
        Err(SdkError::NetworkError(_)) => {
            println!("Network error, retrying...");
            sleep(Duration::from_secs(1)).await;
            client.get_balance("0x1234...").await?
        }
        Err(e) => return Err(e),
    };
    
    println!("Balance: {}", balance);
    Ok(())
}
```

## Best Practices

### Resource Management

```rust
use hauptbuch_sdk::{Client, Account};

async fn manage_resources() -> Result<(), SdkError> {
    // Use connection pooling
    let client = ClientBuilder::new()
        .rpc_url("http://localhost:8080")
        .connection_pool_size(10)
        .build()?;
    
    // Reuse client instance
    let account = Account::new()?;
    
    // Batch operations
    let addresses = vec!["0x1234...", "0x5678...", "0x9abc..."];
    let mut balances = Vec::new();
    
    for address in addresses {
        let balance = client.get_balance(address).await?;
        balances.push(balance);
    }
    
    Ok(())
}
```

### Security Best Practices

```rust
use hauptbuch_sdk::{Account, QuantumResistantCrypto};

fn secure_key_management() -> Result<(), SdkError> {
    // Use quantum-resistant cryptography
    let account = AccountBuilder::new()
        .quantum_resistant(true)
        .build()?;
    
    // Store private key securely
    let private_key = account.private_key();
    // In production, use secure key storage
    
    // Use strong random number generation
    let crypto = QuantumResistantCrypto::new();
    let random_bytes = crypto.generate_random_bytes(32)?;
    
    Ok(())
}
```

### Performance Optimization

```rust
use hauptbuch_sdk::{Client, Account};
use tokio::task;

async fn optimize_performance() -> Result<(), SdkError> {
    let client = Client::new("http://localhost:8080")?;
    let account = Account::new()?;
    
    // Use async/await for concurrent operations
    let balance_task = task::spawn(async {
        client.get_balance(&account.address()).await
    });
    
    let network_info_task = task::spawn(async {
        client.get_network_info().await
    });
    
    // Wait for both operations
    let (balance, network_info) = tokio::try_join!(
        balance_task,
        network_info_task
    )?;
    
    println!("Balance: {}", balance?);
    println!("Network: {}", network_info?.network_id);
    
    Ok(())
}
```

## Conclusion

These basic usage examples provide a foundation for working with the Hauptbuch blockchain platform. Follow the best practices and error handling patterns to build robust applications on the Hauptbuch network.
