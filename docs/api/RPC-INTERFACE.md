# RPC Interface

## Overview

This document provides comprehensive documentation for the Hauptbuch RPC (Remote Procedure Call) interface. It covers all available RPC methods, their parameters, return values, and usage examples.

## Table of Contents

- [RPC Overview](#rpc-overview)
- [Core RPC Methods](#core-rpc-methods)
- [Consensus RPC Methods](#consensus-rpc-methods)
- [Network RPC Methods](#network-rpc-methods)
- [Cryptography RPC Methods](#cryptography-rpc-methods)
- [Cross-Chain RPC Methods](#cross-chain-rpc-methods)
- [Governance RPC Methods](#governance-rpc-methods)
- [Account Abstraction RPC Methods](#account-abstraction-rpc-methods)
- [Layer 2 RPC Methods](#layer-2-rpc-methods)
- [Monitoring RPC Methods](#monitoring-rpc-methods)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Authentication](#authentication)

## RPC Overview

### RPC Endpoint

```
HTTP: http://localhost:8080/rpc
WebSocket: ws://localhost:8080/rpc
gRPC: localhost:8080
```

### Request Format

```json
{
  "jsonrpc": "2.0",
  "method": "method_name",
  "params": {
    "param1": "value1",
    "param2": "value2"
  },
  "id": 1
}
```

### Response Format

```json
{
  "jsonrpc": "2.0",
  "result": {
    "data": "response_data"
  },
  "id": 1
}
```

### Error Format

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32000,
    "message": "Error message",
    "data": {
      "details": "Additional error details"
    }
  },
  "id": 1
}
```

## Core RPC Methods

### Get Network Info

**Method**: `hauptbuch_getNetworkInfo`

**Description**: Get network information including chain ID, network ID, and node version.

**Parameters**: None

**Response**:
```json
{
  "chainId": "1337",
  "networkId": "hauptbuch-testnet-1",
  "nodeVersion": "1.0.0",
  "protocolVersion": "1.0.0",
  "genesisHash": "0x...",
  "latestBlock": "0x...",
  "latestBlockNumber": "0x1234"
}
```

**Example**:
```bash
curl -X POST http://localhost:8080/rpc \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "hauptbuch_getNetworkInfo",
    "params": {},
    "id": 1
  }'
```

### Get Node Status

**Method**: `hauptbuch_getNodeStatus`

**Description**: Get current node status including sync status, peer count, and uptime.

**Parameters**: None

**Response**:
```json
{
  "status": "synced",
  "syncStatus": {
    "currentBlock": "0x1234",
    "highestBlock": "0x1234",
    "syncProgress": 100
  },
  "peerCount": 10,
  "uptime": 3600,
  "memoryUsage": {
    "used": "512MB",
    "total": "1GB"
  },
  "cpuUsage": 25.5
}
```

### Get Chain Info

**Method**: `hauptbuch_getChainInfo`

**Description**: Get chain information including block height, total transactions, and gas used.

**Parameters**: None

**Response**:
```json
{
  "blockHeight": "0x1234",
  "totalTransactions": "0x5678",
  "totalGasUsed": "0x9abc",
  "averageBlockTime": 5000,
  "difficulty": "0x1234567890abcdef",
  "totalSupply": "0x1000000000000000000"
}
```

## Consensus RPC Methods

### Get Validator Set

**Method**: `hauptbuch_getValidatorSet`

**Description**: Get current validator set with their addresses, stakes, and voting power.

**Parameters**: None

**Response**:
```json
{
  "validators": [
    {
      "address": "0x1234...",
      "stake": "1000000000000000000",
      "votingPower": 1000,
      "status": "active",
      "lastSeen": 1640995200
    }
  ],
  "totalStake": "10000000000000000000",
  "activeValidators": 10,
  "totalValidators": 10
}
```

### Get Block

**Method**: `hauptbuch_getBlock`

**Description**: Get block information by block number or hash.

**Parameters**:
```json
{
  "blockNumber": "0x1234",
  "includeTransactions": true
}
```

**Response**:
```json
{
  "block": {
    "number": "0x1234",
    "hash": "0x...",
    "parentHash": "0x...",
    "timestamp": "0x1234567890",
    "gasLimit": "0x5208",
    "gasUsed": "0x2100",
    "transactions": [
      {
        "hash": "0x...",
        "from": "0x1234...",
        "to": "0x5678...",
        "value": "0x1000",
        "gas": "0x5208",
        "gasPrice": "0x3b9aca00"
      }
    ]
  }
}
```

### Get Transaction

**Method**: `hauptbuch_getTransaction`

**Description**: Get transaction information by transaction hash.

**Parameters**:
```json
{
  "txHash": "0x..."
}
```

**Response**:
```json
{
  "transaction": {
    "hash": "0x...",
    "from": "0x1234...",
    "to": "0x5678...",
    "value": "0x1000",
    "gas": "0x5208",
    "gasPrice": "0x3b9aca00",
    "nonce": "0x1",
    "data": "0x...",
    "blockNumber": "0x1234",
    "blockHash": "0x...",
    "transactionIndex": "0x0"
  }
}
```

## Network RPC Methods

### Get Peer List

**Method**: `hauptbuch_getPeerList`

**Description**: Get list of connected peers with their information.

**Parameters**: None

**Response**:
```json
{
  "peers": [
    {
      "id": "Qm...",
      "address": "127.0.0.1:8080",
      "status": "connected",
      "capabilities": ["consensus", "network"],
      "lastSeen": 1640995200,
      "latency": 50
    }
  ],
  "totalPeers": 10,
  "connectedPeers": 8
}
```

### Add Peer

**Method**: `hauptbuch_addPeer`

**Description**: Add a new peer to the network.

**Parameters**:
```json
{
  "peerAddress": "127.0.0.1:8080",
  "peerId": "Qm..."
}
```

**Response**:
```json
{
  "success": true,
  "peerId": "Qm...",
  "status": "connected"
}
```

### Remove Peer

**Method**: `hauptbuch_removePeer`

**Description**: Remove a peer from the network.

**Parameters**:
```json
{
  "peerId": "Qm..."
}
```

**Response**:
```json
{
  "success": true,
  "peerId": "Qm..."
}
```

## Cryptography RPC Methods

### Generate Keypair

**Method**: `hauptbuch_generateKeypair`

**Description**: Generate a new quantum-resistant keypair.

**Parameters**:
```json
{
  "algorithm": "ml-dsa",
  "keySize": 256
}
```

**Response**:
```json
{
  "privateKey": "0x...",
  "publicKey": "0x...",
  "address": "0x1234...",
  "algorithm": "ml-dsa"
}
```

### Sign Message

**Method**: `hauptbuch_signMessage`

**Description**: Sign a message with a private key.

**Parameters**:
```json
{
  "message": "0x...",
  "privateKey": "0x...",
  "algorithm": "ml-dsa"
}
```

**Response**:
```json
{
  "signature": "0x...",
  "publicKey": "0x...",
  "algorithm": "ml-dsa"
}
```

### Verify Signature

**Method**: `hauptbuch_verifySignature`

**Description**: Verify a message signature.

**Parameters**:
```json
{
  "message": "0x...",
  "signature": "0x...",
  "publicKey": "0x...",
  "algorithm": "ml-dsa"
}
```

**Response**:
```json
{
  "valid": true,
  "algorithm": "ml-dsa"
}
```

## Cross-Chain RPC Methods

### Get Bridge Status

**Method**: `hauptbuch_getBridgeStatus`

**Description**: Get status of cross-chain bridges.

**Parameters**: None

**Response**:
```json
{
  "bridges": [
    {
      "name": "ethereum-bridge",
      "sourceChain": "ethereum",
      "targetChain": "hauptbuch",
      "status": "active",
      "totalTransfers": 1000,
      "pendingTransfers": 5
    }
  ],
  "totalBridges": 3,
  "activeBridges": 2
}
```

### Transfer Asset

**Method**: `hauptbuch_transferAsset`

**Description**: Transfer assets across chains.

**Parameters**:
```json
{
  "from": "0x1234...",
  "to": "0x5678...",
  "amount": "1000000000000000000",
  "sourceChain": "ethereum",
  "targetChain": "hauptbuch",
  "asset": "ETH"
}
```

**Response**:
```json
{
  "transactionHash": "0x...",
  "bridgeId": "ethereum-bridge",
  "status": "pending",
  "estimatedTime": 300
}
```

### Get Transfer Status

**Method**: `hauptbuch_getTransferStatus`

**Description**: Get status of a cross-chain transfer.

**Parameters**:
```json
{
  "transactionHash": "0x..."
}
```

**Response**:
```json
{
  "status": "completed",
  "sourceTransaction": "0x...",
  "targetTransaction": "0x...",
  "bridgeId": "ethereum-bridge",
  "completionTime": 1640995200
}
```

## Governance RPC Methods

### Get Proposals

**Method**: `hauptbuch_getProposals`

**Description**: Get list of governance proposals.

**Parameters**:
```json
{
  "status": "active",
  "limit": 10,
  "offset": 0
}
```

**Response**:
```json
{
  "proposals": [
    {
      "id": 1,
      "title": "Proposal Title",
      "description": "Proposal Description",
      "author": "0x1234...",
      "status": "active",
      "startTime": 1640995200,
      "endTime": 1641081600,
      "votes": {
        "yes": 1000,
        "no": 500,
        "abstain": 100
      }
    }
  ],
  "totalProposals": 25,
  "activeProposals": 5
}
```

### Submit Proposal

**Method**: `hauptbuch_submitProposal`

**Description**: Submit a new governance proposal.

**Parameters**:
```json
{
  "title": "Proposal Title",
  "description": "Proposal Description",
  "author": "0x1234...",
  "proposalType": "parameter_change",
  "parameters": {
    "blockTime": 5000,
    "gasLimit": 10000000
  }
}
```

**Response**:
```json
{
  "proposalId": 1,
  "transactionHash": "0x...",
  "status": "submitted"
}
```

### Vote on Proposal

**Method**: `hauptbuch_vote`

**Description**: Vote on a governance proposal.

**Parameters**:
```json
{
  "proposalId": 1,
  "voter": "0x1234...",
  "choice": "yes",
  "votingPower": 1000
}
```

**Response**:
```json
{
  "success": true,
  "transactionHash": "0x...",
  "votingPower": 1000
}
```

## Account Abstraction RPC Methods

### Get User Operations

**Method**: `hauptbuch_getUserOperations`

**Description**: Get list of user operations for account abstraction.

**Parameters**:
```json
{
  "account": "0x1234...",
  "status": "pending",
  "limit": 10
}
```

**Response**:
```json
{
  "userOperations": [
    {
      "hash": "0x...",
      "sender": "0x1234...",
      "nonce": "0x1",
      "callData": "0x...",
      "signature": "0x...",
      "status": "pending"
    }
  ],
  "totalOperations": 50,
  "pendingOperations": 5
}
```

### Submit User Operation

**Method**: `hauptbuch_submitUserOperation`

**Description**: Submit a user operation for account abstraction.

**Parameters**:
```json
{
  "sender": "0x1234...",
  "nonce": "0x1",
  "callData": "0x...",
  "signature": "0x...",
  "paymaster": "0x5678...",
  "gasLimit": "0x5208",
  "gasPrice": "0x3b9aca00"
}
```

**Response**:
```json
{
  "userOperationHash": "0x...",
  "status": "submitted",
  "estimatedGas": "0x5208"
}
```

## Layer 2 RPC Methods

### Get Rollup Status

**Method**: `hauptbuch_getRollupStatus`

**Description**: Get status of Layer 2 rollups.

**Parameters**: None

**Response**:
```json
{
  "rollups": [
    {
      "name": "optimistic-rollup",
      "status": "active",
      "sequencer": "0x1234...",
      "prover": "0x5678...",
      "totalTransactions": 10000,
      "pendingTransactions": 100
    }
  ],
  "totalRollups": 2,
  "activeRollups": 1
}
```

### Submit Rollup Transaction

**Method**: `hauptbuch_submitRollupTransaction`

**Description**: Submit a transaction to a Layer 2 rollup.

**Parameters**:
```json
{
  "rollupName": "optimistic-rollup",
  "transaction": {
    "from": "0x1234...",
    "to": "0x5678...",
    "value": "0x1000",
    "data": "0x..."
  }
}
```

**Response**:
```json
{
  "transactionHash": "0x...",
  "rollupId": "optimistic-rollup",
  "status": "submitted",
  "estimatedConfirmationTime": 300
}
```

## Monitoring RPC Methods

### Get Metrics

**Method**: `hauptbuch_getMetrics`

**Description**: Get system metrics and performance data.

**Parameters**: None

**Response**:
```json
{
  "metrics": {
    "blockHeight": "0x1234",
    "transactionCount": "0x5678",
    "gasUsed": "0x9abc",
    "peerCount": 10,
    "memoryUsage": "512MB",
    "cpuUsage": 25.5,
    "diskUsage": "1GB",
    "networkLatency": 50
  },
  "timestamp": 1640995200
}
```

### Get Health Status

**Method**: `hauptbuch_getHealthStatus`

**Description**: Get health status of the node and its components.

**Parameters**: None

**Response**:
```json
{
  "status": "healthy",
  "components": {
    "consensus": "healthy",
    "network": "healthy",
    "database": "healthy",
    "cryptography": "healthy"
  },
  "uptime": 3600,
  "lastHealthCheck": 1640995200
}
```

## Error Handling

### Error Codes

| Code | Name | Description |
|------|------|-------------|
| -32000 | Server Error | Internal server error |
| -32001 | Invalid Request | Invalid request format |
| -32002 | Method Not Found | RPC method not found |
| -32003 | Invalid Params | Invalid parameters |
| -32004 | Internal Error | Internal processing error |
| -32005 | Parse Error | JSON parsing error |
| -32006 | Invalid Address | Invalid address format |
| -32007 | Insufficient Funds | Insufficient balance |
| -32008 | Gas Limit Exceeded | Gas limit exceeded |
| -32009 | Nonce Too Low | Nonce too low |
| -32010 | Nonce Too High | Nonce too high |

### Error Response Format

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32000,
    "message": "Server error",
    "data": {
      "details": "Additional error information",
      "requestId": "req_123456789"
    }
  },
  "id": 1
}
```

## Rate Limiting

### Rate Limit Headers

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1640995200
```

### Rate Limit Response

```json
{
  "jsonrpc": "2.0",
  "error": {
    "code": -32011,
    "message": "Rate limit exceeded",
    "data": {
      "retryAfter": 60,
      "limit": 1000,
      "remaining": 0
    }
  },
  "id": 1
}
```

## Authentication

### API Key Authentication

**Header**: `Authorization: Bearer <api_key>`

**Example**:
```bash
curl -X POST http://localhost:8080/rpc \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your_api_key" \
  -d '{
    "jsonrpc": "2.0",
    "method": "hauptbuch_getNetworkInfo",
    "params": {},
    "id": 1
  }'
```

### JWT Authentication

**Header**: `Authorization: Bearer <jwt_token>`

**Example**:
```bash
curl -X POST http://localhost:8080/rpc \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..." \
  -d '{
    "jsonrpc": "2.0",
    "method": "hauptbuch_getNetworkInfo",
    "params": {},
    "id": 1
  }'
```

## WebSocket Interface

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/rpc');

ws.onopen = function() {
  console.log('Connected to Hauptbuch RPC');
};

ws.onmessage = function(event) {
  const response = JSON.parse(event.data);
  console.log('Received:', response);
};

ws.onclose = function() {
  console.log('Disconnected from Hauptbuch RPC');
};
```

### WebSocket Subscription

```javascript
// Subscribe to new blocks
const subscribeRequest = {
  jsonrpc: "2.0",
  method: "hauptbuch_subscribe",
  params: ["newBlocks"],
  id: 1
};

ws.send(JSON.stringify(subscribeRequest));
```

## gRPC Interface

### gRPC Service Definition

```protobuf
syntax = "proto3";

package hauptbuch.rpc;

service HauptbuchRPC {
  rpc GetNetworkInfo(Empty) returns (NetworkInfo);
  rpc GetNodeStatus(Empty) returns (NodeStatus);
  rpc GetBlock(GetBlockRequest) returns (Block);
  rpc GetTransaction(GetTransactionRequest) returns (Transaction);
  rpc SubmitTransaction(SubmitTransactionRequest) returns (SubmitTransactionResponse);
}

message Empty {}

message GetBlockRequest {
  string block_number = 1;
  bool include_transactions = 2;
}

message GetTransactionRequest {
  string tx_hash = 1;
}

message SubmitTransactionRequest {
  string from = 1;
  string to = 2;
  string value = 3;
  string data = 4;
  string gas_limit = 5;
  string gas_price = 6;
}

message SubmitTransactionResponse {
  string tx_hash = 1;
  string status = 2;
}
```

## Conclusion

This RPC interface documentation provides comprehensive information about all available RPC methods in the Hauptbuch platform. Use this reference to integrate with the Hauptbuch network effectively.
