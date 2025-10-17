# Cross-Chain Examples

## Overview

This document provides comprehensive examples of cross-chain functionality in the Hauptbuch blockchain platform. Learn how to transfer assets, communicate between chains, and implement cross-chain applications.

## Table of Contents

- [Getting Started](#getting-started)
- [Bridge Operations](#bridge-operations)
- [IBC Operations](#ibc-operations)
- [CCIP Operations](#ccip-operations)
- [Cross-Chain Communication](#cross-chain-communication)
- [Asset Management](#asset-management)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Getting Started

### Basic Cross-Chain Setup

```rust
use hauptbuch_cross_chain::{CrossChainClient, Bridge, IBC, CCIP};

#[tokio::main]
async fn main() -> Result<(), CrossChainError> {
    // Create cross-chain client
    let client = CrossChainClient::new("http://localhost:8080")?;
    
    // Initialize bridge
    let bridge = Bridge::new("ethereum", "hauptbuch");
    
    // Initialize IBC
    let ibc = IBC::new("cosmos", "hauptbuch");
    
    // Initialize CCIP
    let ccip = CCIP::new("ethereum", "avalanche");
    
    println!("Cross-chain client initialized");
    Ok(())
}
```

### Cross-Chain Configuration

```rust
use hauptbuch_cross_chain::{CrossChainConfig, BridgeConfig, IBCConfig, CCIPConfig};

async fn setup_cross_chain_config() -> Result<(), CrossChainError> {
    let config = CrossChainConfig {
        bridges: vec![
            BridgeConfig {
                name: "ethereum-bridge".to_string(),
                source_chain: "ethereum".to_string(),
                target_chain: "hauptbuch".to_string(),
                bridge_address: "0x1234...".to_string(),
                enabled: true,
            },
            BridgeConfig {
                name: "polygon-bridge".to_string(),
                source_chain: "polygon".to_string(),
                target_chain: "hauptbuch".to_string(),
                bridge_address: "0x5678...".to_string(),
                enabled: true,
            },
        ],
        ibc_connections: vec![
            IBCConfig {
                client_id: "07-tendermint-0".to_string(),
                connection_id: "connection-0".to_string(),
                channel_id: "channel-0".to_string(),
                port_id: "transfer".to_string(),
            },
        ],
        ccip_routers: vec![
            CCIPConfig {
                source_chain: "ethereum".to_string(),
                target_chain: "avalanche".to_string(),
                router_address: "0x9abc...".to_string(),
                enabled: true,
            },
        ],
    };
    
    let client = CrossChainClient::new("http://localhost:8080")?;
    client.set_config(config).await?;
    
    println!("Cross-chain configuration set");
    Ok(())
}
```

## Bridge Operations

### Basic Asset Transfer

```rust
use hauptbuch_cross_chain::{Bridge, CrossChainTransaction, AssetType};

async fn transfer_asset(
    bridge: &Bridge,
    from: &str,
    to: &str,
    amount: u64,
    asset: AssetType,
) -> Result<String, CrossChainError> {
    let transaction = CrossChainTransaction::new()
        .from(from)
        .to(to)
        .value(amount)
        .asset(asset)
        .source_chain("ethereum")
        .target_chain("hauptbuch")
        .build()?;
    
    let tx_hash = bridge.transfer_asset(transaction).await?;
    println!("Asset transfer initiated: {}", tx_hash);
    Ok(tx_hash)
}
```

### ERC-20 Token Transfer

```rust
use hauptbuch_cross_chain::{Bridge, CrossChainTransaction, ERC20Token};

async fn transfer_erc20_token(
    bridge: &Bridge,
    from: &str,
    to: &str,
    amount: u64,
    token_address: &str,
) -> Result<String, CrossChainError> {
    let token = ERC20Token {
        address: token_address.to_string(),
        symbol: "USDC".to_string(),
        decimals: 6,
    };
    
    let transaction = CrossChainTransaction::new()
        .from(from)
        .to(to)
        .value(amount)
        .asset(AssetType::ERC20(token))
        .source_chain("ethereum")
        .target_chain("hauptbuch")
        .build()?;
    
    let tx_hash = bridge.transfer_asset(transaction).await?;
    println!("ERC-20 token transfer initiated: {}", tx_hash);
    Ok(tx_hash)
}
```

### NFT Transfer

```rust
use hauptbuch_cross_chain::{Bridge, CrossChainTransaction, NFT};

async fn transfer_nft(
    bridge: &Bridge,
    from: &str,
    to: &str,
    token_id: u64,
    nft_address: &str,
) -> Result<String, CrossChainError> {
    let nft = NFT {
        address: nft_address.to_string(),
        token_id,
        token_uri: "https://api.example.com/nft/123".to_string(),
    };
    
    let transaction = CrossChainTransaction::new()
        .from(from)
        .to(to)
        .value(1) // NFT quantity
        .asset(AssetType::NFT(nft))
        .source_chain("ethereum")
        .target_chain("hauptbuch")
        .build()?;
    
    let tx_hash = bridge.transfer_asset(transaction).await?;
    println!("NFT transfer initiated: {}", tx_hash);
    Ok(tx_hash)
}
```

### Bridge Status Monitoring

```rust
use hauptbuch_cross_chain::{Bridge, BridgeStatus};

async fn monitor_bridge_status(
    bridge: &Bridge,
) -> Result<(), CrossChainError> {
    let status = bridge.get_bridge_status().await?;
    
    println!("Bridge Status:");
    println!("  Name: {}", status.name);
    println!("  Source Chain: {}", status.source_chain);
    println!("  Target Chain: {}", status.target_chain);
    println!("  Status: {}", status.status);
    println!("  Total Transfers: {}", status.total_transfers);
    println!("  Pending Transfers: {}", status.pending_transfers);
    println!("  Last Transfer: {}", status.last_transfer);
    
    Ok(())
}
```

## IBC Operations

### IBC Token Transfer

```rust
use hauptbuch_cross_chain::{IBC, IBCPacket, IBCDenom};

async fn transfer_ibc_token(
    ibc: &IBC,
    from: &str,
    to: &str,
    amount: u64,
    denom: &str,
) -> Result<String, CrossChainError> {
    let ibc_denom = IBCDenom {
        denom: denom.to_string(),
        path: "transfer/channel-0/uatom".to_string(),
        base_denom: "uatom".to_string(),
    };
    
    let packet = IBCPacket::new()
        .from(from)
        .to(to)
        .amount(amount)
        .denom(ibc_denom)
        .source_chain("cosmos")
        .target_chain("hauptbuch")
        .build()?;
    
    let packet_id = ibc.send_packet(packet).await?;
    println!("IBC packet sent: {}", packet_id);
    Ok(packet_id)
}
```

### IBC Channel Management

```rust
use hauptbuch_cross_chain::{IBC, IBCChannel, IBCChannelStatus};

async fn manage_ibc_channel(
    ibc: &IBC,
) -> Result<(), CrossChainError> {
    // Get channel status
    let status = ibc.get_channel_status().await?;
    
    println!("IBC Channel Status:");
    println!("  Client ID: {}", status.client_id);
    println!("  Connection ID: {}", status.connection_id);
    println!("  Channel ID: {}", status.channel_id);
    println!("  Port ID: {}", status.port_id);
    println!("  State: {}", status.state);
    println!("  Counterparty: {:?}", status.counterparty);
    
    // Create new channel if needed
    if status.state == "CLOSED" {
        let channel = IBCChannel {
            client_id: "07-tendermint-0".to_string(),
            connection_id: "connection-0".to_string(),
            channel_id: "channel-0".to_string(),
            port_id: "transfer".to_string(),
        };
        
        ibc.create_channel(channel).await?;
        println!("New IBC channel created");
    }
    
    Ok(())
}
```

### IBC Packet Acknowledgment

```rust
use hauptbuch_cross_chain::{IBC, IBCPacket, IBCAcknowledgment};

async fn acknowledge_ibc_packet(
    ibc: &IBC,
    packet_id: &str,
) -> Result<(), CrossChainError> {
    let acknowledgment = IBCAcknowledgment {
        packet_id: packet_id.to_string(),
        success: true,
        result: "Transfer completed successfully".to_string(),
    };
    
    ibc.acknowledge_packet(acknowledgment).await?;
    println!("IBC packet acknowledged: {}", packet_id);
    Ok(())
}
```

## CCIP Operations

### CCIP Message Transfer

```rust
use hauptbuch_cross_chain::{CCIP, CCIPMessage, CCIPFunctionCall};

async fn send_ccip_message(
    ccip: &CCIP,
    from: &str,
    to: &str,
    message: &str,
) -> Result<String, CrossChainError> {
    let function_call = CCIPFunctionCall {
        target: to.to_string(),
        data: message.as_bytes().to_vec(),
        value: 0,
    };
    
    let ccip_message = CCIPMessage::new()
        .from(from)
        .to(to)
        .function_call(function_call)
        .source_chain("ethereum")
        .target_chain("avalanche")
        .build()?;
    
    let message_id = ccip.send_message(ccip_message).await?;
    println!("CCIP message sent: {}", message_id);
    Ok(message_id)
}
```

### CCIP Token Transfer

```rust
use hauptbuch_cross_chain::{CCIP, CCIPMessage, CCIPTokenTransfer};

async fn transfer_ccip_token(
    ccip: &CCIP,
    from: &str,
    to: &str,
    amount: u64,
    token: &str,
) -> Result<String, CrossChainError> {
    let token_transfer = CCIPTokenTransfer {
        token: token.to_string(),
        amount,
        recipient: to.to_string(),
    };
    
    let ccip_message = CCIPMessage::new()
        .from(from)
        .to(to)
        .token_transfer(token_transfer)
        .source_chain("ethereum")
        .target_chain("avalanche")
        .build()?;
    
    let message_id = ccip.send_message(ccip_message).await?;
    println!("CCIP token transfer sent: {}", message_id);
    Ok(message_id)
}
```

### CCIP Router Management

```rust
use hauptbuch_cross_chain::{CCIP, CCIPRouter, CCIPRouterStatus};

async fn manage_ccip_router(
    ccip: &CCIP,
) -> Result<(), CrossChainError> {
    let status = ccip.get_router_status().await?;
    
    println!("CCIP Router Status:");
    println!("  Source Chain: {}", status.source_chain);
    println!("  Target Chain: {}", status.target_chain);
    println!("  Router Address: {}", status.router_address);
    println!("  Status: {}", status.status);
    println!("  Total Messages: {}", status.total_messages);
    println!("  Pending Messages: {}", status.pending_messages);
    
    // Get supported chains
    let supported_chains = ccip.get_supported_chains().await?;
    println!("Supported chains: {:?}", supported_chains);
    
    Ok(())
}
```

## Cross-Chain Communication

### Cross-Chain Smart Contract Call

```rust
use hauptbuch_cross_chain::{CrossChainClient, CrossChainCall, CrossChainResult};

async fn call_cross_chain_contract(
    client: &CrossChainClient,
    source_chain: &str,
    target_chain: &str,
    contract_address: &str,
    function_name: &str,
    args: Vec<String>,
) -> Result<CrossChainResult, CrossChainError> {
    let call = CrossChainCall::new()
        .source_chain(source_chain)
        .target_chain(target_chain)
        .contract_address(contract_address)
        .function_name(function_name)
        .arguments(args)
        .build()?;
    
    let result = client.execute_cross_chain_call(call).await?;
    
    println!("Cross-chain call result: {:?}", result);
    Ok(result)
}
```

### Cross-Chain Event Listening

```rust
use hauptbuch_cross_chain::{CrossChainClient, CrossChainEvent, EventType};

async fn listen_cross_chain_events(
    client: &CrossChainClient,
) -> Result<(), CrossChainError> {
    let mut event_stream = client.subscribe_to_cross_chain_events().await?;
    
    while let Some(event) = event_stream.next().await {
        match event.event_type {
            EventType::AssetTransfer => {
                println!("Asset transfer event: {:?}", event);
            }
            EventType::MessageSent => {
                println!("Message sent event: {:?}", event);
            }
            EventType::MessageReceived => {
                println!("Message received event: {:?}", event);
            }
            EventType::BridgeStatusChanged => {
                println!("Bridge status changed: {:?}", event);
            }
        }
    }
    
    Ok(())
}
```

### Cross-Chain State Synchronization

```rust
use hauptbuch_cross_chain::{CrossChainClient, StateSync, StateUpdate};

async fn sync_cross_chain_state(
    client: &CrossChainClient,
    source_chain: &str,
    target_chain: &str,
) -> Result<(), CrossChainError> {
    let state_sync = StateSync::new()
        .source_chain(source_chain)
        .target_chain(target_chain)
        .sync_interval(60) // 60 seconds
        .build()?;
    
    let mut state_stream = client.subscribe_to_state_updates(state_sync).await?;
    
    while let Some(update) = state_stream.next().await {
        match update {
            StateUpdate::BalanceChanged { address, new_balance } => {
                println!("Balance changed for {}: {}", address, new_balance);
            }
            StateUpdate::ContractStateChanged { contract, new_state } => {
                println!("Contract state changed for {}: {:?}", contract, new_state);
            }
            StateUpdate::ValidatorSetChanged { new_validators } => {
                println!("Validator set changed: {:?}", new_validators);
            }
        }
    }
    
    Ok(())
}
```

## Asset Management

### Multi-Chain Asset Tracking

```rust
use hauptbuch_cross_chain::{CrossChainClient, AssetTracker, AssetBalance};

async fn track_multi_chain_assets(
    client: &CrossChainClient,
    address: &str,
) -> Result<(), CrossChainError> {
    let tracker = AssetTracker::new()
        .address(address)
        .tracked_chains(vec!["ethereum", "polygon", "hauptbuch"])
        .build()?;
    
    let balances = client.get_multi_chain_balances(tracker).await?;
    
    for balance in balances {
        println!("Chain: {}", balance.chain);
        println!("  Native Token: {} {}", balance.native_balance, balance.native_symbol);
        for token in balance.tokens {
            println!("  Token: {} {} {}", token.balance, token.symbol, token.address);
        }
    }
    
    Ok(())
}
```

### Cross-Chain Asset Swapping

```rust
use hauptbuch_cross_chain::{CrossChainClient, AssetSwap, SwapRoute};

async fn perform_cross_chain_swap(
    client: &CrossChainClient,
    from_asset: &str,
    to_asset: &str,
    amount: u64,
    source_chain: &str,
    target_chain: &str,
) -> Result<String, CrossChainError> {
    let swap_route = SwapRoute {
        from_asset: from_asset.to_string(),
        to_asset: to_asset.to_string(),
        amount,
        source_chain: source_chain.to_string(),
        target_chain: target_chain.to_string(),
        intermediate_chains: vec!["polygon".to_string()],
    };
    
    let swap = AssetSwap::new()
        .route(swap_route)
        .slippage_tolerance(0.01) // 1%
        .deadline(chrono::Utc::now().timestamp() + 3600) // 1 hour
        .build()?;
    
    let swap_id = client.execute_cross_chain_swap(swap).await?;
    println!("Cross-chain swap initiated: {}", swap_id);
    Ok(swap_id)
}
```

### Cross-Chain Liquidity Provision

```rust
use hauptbuch_cross_chain::{CrossChainClient, LiquidityPool, LiquidityProvider};

async fn provide_cross_chain_liquidity(
    client: &CrossChainClient,
    pool_address: &str,
    asset_a: &str,
    asset_b: &str,
    amount_a: u64,
    amount_b: u64,
) -> Result<String, CrossChainError> {
    let liquidity_provider = LiquidityProvider {
        pool_address: pool_address.to_string(),
        asset_a: asset_a.to_string(),
        asset_b: asset_b.to_string(),
        amount_a,
        amount_b,
        min_liquidity: 0,
    };
    
    let tx_hash = client.provide_cross_chain_liquidity(liquidity_provider).await?;
    println!("Cross-chain liquidity provided: {}", tx_hash);
    Ok(tx_hash)
}
```

## Error Handling

### Cross-Chain Error Handling

```rust
use hauptbuch_cross_chain::{CrossChainClient, CrossChainError};

async fn handle_cross_chain_errors(
    client: &CrossChainClient,
    transaction_id: &str,
) -> Result<(), CrossChainError> {
    match client.get_transaction_status(transaction_id).await {
        Ok(status) => {
            println!("Transaction status: {:?}", status);
        }
        Err(CrossChainError::BridgeUnavailable) => {
            println!("Bridge is currently unavailable");
        }
        Err(CrossChainError::InsufficientLiquidity) => {
            println!("Insufficient liquidity for the transfer");
        }
        Err(CrossChainError::InvalidChain) => {
            println!("Invalid source or target chain");
        }
        Err(CrossChainError::TransferTimeout) => {
            println!("Transfer timed out");
        }
        Err(CrossChainError::InvalidAsset) => {
            println!("Invalid asset type or address");
        }
        Err(e) => {
            println!("Other cross-chain error: {}", e);
        }
    }
    
    Ok(())
}
```

### Retry Logic for Cross-Chain Operations

```rust
use hauptbuch_cross_chain::{CrossChainClient, CrossChainError};
use tokio::time::{sleep, Duration};

async fn retry_cross_chain_operation(
    client: &CrossChainClient,
    operation: impl Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String, CrossChainError>> + Send>>,
    max_retries: u32,
) -> Result<String, CrossChainError> {
    let mut retries = 0;
    
    loop {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(CrossChainError::NetworkError(_)) if retries < max_retries => {
                retries += 1;
                println!("Retry {} of {}", retries, max_retries);
                sleep(Duration::from_secs(2_u64.pow(retries))).await; // Exponential backoff
            }
            Err(e) => return Err(e),
        }
    }
}
```

## Best Practices

### Cross-Chain Security

```rust
use hauptbuch_cross_chain::{CrossChainClient, SecurityConfig, MultiSig};

async fn secure_cross_chain_operations(
    client: &CrossChainClient,
) -> Result<(), CrossChainError> {
    // Configure security settings
    let security_config = SecurityConfig {
        require_multisig: true,
        minimum_confirmations: 3,
        enable_audit_trail: true,
        require_identity_verification: true,
    };
    
    client.set_security_config(security_config).await?;
    
    // Use multi-signature for large transfers
    let multisig = MultiSig {
        signers: vec![
            "0x1234...".to_string(),
            "0x5678...".to_string(),
            "0x9abc...".to_string(),
        ],
        threshold: 2,
    };
    
    client.set_multisig(multisig).await?;
    
    Ok(())
}
```

### Performance Optimization

```rust
use hauptbuch_cross_chain::{CrossChainClient, BatchTransfer, TransferBatch};

async fn optimize_cross_chain_performance(
    client: &CrossChainClient,
) -> Result<(), CrossChainError> {
    // Batch multiple transfers
    let transfers = vec![
        TransferBatch {
            from: "0x1234...".to_string(),
            to: "0x5678...".to_string(),
            amount: 1000,
            asset: "ETH".to_string(),
        },
        TransferBatch {
            from: "0x1234...".to_string(),
            to: "0x9abc...".to_string(),
            amount: 2000,
            asset: "ETH".to_string(),
        },
    ];
    
    let batch_transfer = BatchTransfer {
        transfers,
        source_chain: "ethereum".to_string(),
        target_chain: "hauptbuch".to_string(),
    };
    
    let batch_id = client.execute_batch_transfer(batch_transfer).await?;
    println!("Batch transfer executed: {}", batch_id);
    
    Ok(())
}
```

### Cross-Chain Monitoring

```rust
use hauptbuch_cross_chain::{CrossChainClient, CrossChainMetrics, HealthCheck};

async fn monitor_cross_chain_health(
    client: &CrossChainClient,
) -> Result<(), CrossChainError> {
    // Get cross-chain metrics
    let metrics = client.get_cross_chain_metrics().await?;
    
    println!("Cross-Chain Metrics:");
    println!("  Total Transfers: {}", metrics.total_transfers);
    println!("  Successful Transfers: {}", metrics.successful_transfers);
    println!("  Failed Transfers: {}", metrics.failed_transfers);
    println!("  Average Transfer Time: {} seconds", metrics.average_transfer_time);
    println!("  Active Bridges: {}", metrics.active_bridges);
    
    // Perform health checks
    let health_check = HealthCheck {
        check_bridges: true,
        check_ibc_connections: true,
        check_ccip_routers: true,
    };
    
    let health_status = client.perform_health_check(health_check).await?;
    
    for (component, status) in health_status {
        println!("{}: {}", component, status);
    }
    
    Ok(())
}
```

## Conclusion

These cross-chain examples provide comprehensive guidance for implementing cross-chain functionality on the Hauptbuch blockchain platform. Follow the best practices and security considerations to ensure effective and secure cross-chain operations.
