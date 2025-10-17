# Hauptbuch Blockchain - Quick Start Guide

## 🚀 Starting the Node

```bash
# Start the Hauptbuch node
./target/release/hauptbuch --rpc-port 8084 --ws-port 8085 --log-level info

# Or with custom configuration
./target/release/hauptbuch \
  --rpc-port 8080 \
  --ws-port 8081 \
  --p2p-port 30303 \
  --network-id "my-network" \
  --chain-id 1234 \
  --data-dir ./my-data \
  --log-level debug
```

## 💼 Creating Wallets

```python
from hauptbuch_client import AccountManager

# Create a new wallet
account = AccountManager.create_account()
print(f"Address: {account.address}")
print(f"Balance: {account.balance}")
print(f"Nonce: {account.nonce}")
```

## 📤 Sending Transactions

```python
import asyncio
from hauptbuch_client import HauptbuchClient

async def send_transaction():
    async with HauptbuchClient() as client:
        # Send a transaction
        tx_hash = await client.send_transaction(
            from_address="0x1234...",
            to_address="0x5678...",
            value="0x1000",  # Amount in wei (hex)
            gas_limit="0x5208",
            gas_price="0x3b9aca00"
        )
        print(f"Transaction Hash: {tx_hash}")

asyncio.run(send_transaction())
```

## 🔐 Cryptographic Operations

```python
async def crypto_operations():
    async with HauptbuchClient() as client:
        # Generate quantum-resistant keypair
        keypair = await client.generate_keypair("ml-kem", 2048)
        
        # Sign a message
        signature = await client.sign_message(
            private_key="your_private_key",
            message="Hello, Hauptbuch!",
            algorithm="ml-dsa"
        )
        
        # Verify signature
        verification = await client.verify_signature(
            message="Hello, Hauptbuch!",
            signature=signature['signature'],
            public_key=keypair['publicKey'],
            algorithm="ml-dsa"
        )
        print(f"Signature Valid: {verification['valid']}")

asyncio.run(crypto_operations())
```

## 👤 Account Abstraction (ERC-4337)

```python
async def use_account_abstraction():
    async with HauptbuchClient() as client:
        # Submit a user operation
        user_op = await client.submit_user_operation(
            sender="0x1234...",
            nonce="0x1",
            call_data="0x...",
            signature="0x...",
            gas_limit="0x5208",
            gas_price="0x3b9aca00"
        )
        print(f"User Operation Hash: {user_op['userOperationHash']}")

asyncio.run(use_account_abstraction())
```

## 🔄 Layer 2 Rollups

```python
async def use_rollups():
    async with HauptbuchClient() as client:
        # Submit to optimistic rollup
        tx = {
            "from": "0x1234...",
            "to": "0x5678...",
            "value": "0x1000",
            "data": "0x"
        }
        
        result = await client.submit_rollup_transaction(
            "optimistic-rollup",
            tx
        )
        print(f"Rollup TX Hash: {result['transactionHash']}")

asyncio.run(use_rollups())
```

## 🌉 Cross-Chain Transfers

```python
async def cross_chain_transfer():
    async with HauptbuchClient() as client:
        # Transfer assets to Ethereum
        transfer = await client.transfer_asset(
            from_address="0x1234...",
            to_address="0x5678...",
            amount="1000000000000000000",  # 1 token
            source_chain="hauptbuch",
            target_chain="ethereum",
            asset="HBK"
        )
        print(f"Transfer TX: {transfer['transactionHash']}")
        
        # Check transfer status
        status = await client.get_transfer_status(
            transfer['transactionHash']
        )
        print(f"Status: {status['status']}")

asyncio.run(cross_chain_transfer())
```

## 🗳️ Governance

```python
async def governance_operations():
    async with HauptbuchClient() as client:
        # Create a proposal
        proposal = await client.submit_proposal(
            title="Increase Block Size",
            description="Proposal to increase block size to 2MB",
            author="0x1234...",
            proposal_type="parameter_change",
            parameters={"blockSize": 2000000}
        )
        print(f"Proposal ID: {proposal['proposalId']}")
        
        # Vote on proposal
        vote = await client.vote(
            proposal_id=1,
            voter="0x5678...",
            choice="yes",
            voting_power=1000
        )
        print(f"Vote Success: {vote['success']}")

asyncio.run(governance_operations())
```

## 📊 Monitoring

```python
async def monitor_blockchain():
    async with HauptbuchClient() as client:
        # Get network info
        network = await client.get_network_info()
        print(f"Network: {network.network_id}")
        print(f"Chain ID: {network.chain_id}")
        
        # Get node status
        status = await client.get_node_status()
        print(f"Status: {status.status}")
        print(f"Uptime: {status.uptime}s")
        
        # Get metrics
        metrics = await client.get_metrics()
        print(f"Block Height: {metrics['metrics']['blockHeight']}")
        print(f"TX Count: {metrics['metrics']['transactionCount']}")
        
        # Get health status
        health = await client.get_health_status()
        print(f"Health: {health['status']}")

asyncio.run(monitor_blockchain())
```

## 🧪 Running Tests

```bash
# Activate Python virtual environment
source .venv/bin/activate

# Set RPC URL
export HAUPTBUCH_RPC_URL=http://localhost:8084

# Run all tests
python -m pytest tests/

# Run specific test suites
python -m pytest tests/infrastructure/
python -m pytest tests/contracts/
python -m pytest tests/integration/

# Run comprehensive interaction tests
python tests/comprehensive_blockchain_test.py
python tests/advanced_wallet_test.py
```

## 🔧 Configuration

### Environment Variables
```bash
export HAUPTBUCH_RPC_URL=http://localhost:8084
export HAUPTBUCH_WS_URL=ws://localhost:8085
export HAUPTBUCH_API_KEY=your_api_key_here
```

### Node Configuration
Edit `config.toml`:
```toml
[network]
network_id = "hauptbuch-testnet-1"
chain_id = 1337
bootstrap_nodes = []

[server]
rpc_port = 8084
ws_port = 8085
p2p_port = 30303

[storage]
data_dir = "./data"

[logging]
level = "info"
```

## 📡 RPC Endpoints

### Network Information
- `hauptbuch_getNetworkInfo` - Get network details
- `hauptbuch_getNodeStatus` - Get node status
- `hauptbuch_getChainInfo` - Get chain information
- `hauptbuch_getPeerList` - Get connected peers

### Cryptography
- `hauptbuch_generateKeypair` - Generate quantum-resistant keypair
- `hauptbuch_signMessage` - Sign a message
- `hauptbuch_verifySignature` - Verify a signature

### Accounts
- `hauptbuch_getBalance` - Get account balance
- `hauptbuch_getNonce` - Get account nonce
- `hauptbuch_getCode` - Get contract code

### Transactions
- `hauptbuch_sendTransaction` - Send a transaction
- `hauptbuch_getTransactionStatus` - Get TX status
- `hauptbuch_getTransactionHistory` - Get TX history

### Governance
- `hauptbuch_getProposals` - Get governance proposals
- `hauptbuch_submitProposal` - Submit a proposal
- `hauptbuch_vote` - Vote on a proposal

### Account Abstraction
- `hauptbuch_getUserOperations` - Get user operations
- `hauptbuch_submitUserOperation` - Submit user operation

### Layer 2
- `hauptbuch_getRollupStatus` - Get rollup status
- `hauptbuch_submitRollupTransaction` - Submit rollup TX

### Cross-Chain
- `hauptbuch_getBridgeStatus` - Get bridge status
- `hauptbuch_transferAsset` - Transfer assets cross-chain
- `hauptbuch_getTransferStatus` - Get transfer status

### Monitoring
- `hauptbuch_getMetrics` - Get system metrics
- `hauptbuch_getHealthStatus` - Get health status

## 🛠️ Troubleshooting

### Node won't start
```bash
# Check if port is already in use
lsof -i :8084

# Check logs
tail -f hauptbuch.log

# Try different port
./target/release/hauptbuch --rpc-port 8090
```

### Connection issues
```python
# Increase timeout
client = HauptbuchClient(timeout=60)

# Check node is running
curl -X POST http://localhost:8084 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"hauptbuch_getNetworkInfo","params":{},"id":1}'
```

### Build issues
```bash
# Clean build
cargo clean
cargo build --release

# Install dependencies
./scripts/install-deps.sh

# Build Python extension
maturin develop --release
```

## 📚 Additional Resources

- **API Reference:** `docs/api/API-REFERENCE.md`
- **RPC Interface:** `docs/api/RPC-INTERFACE.md`
- **Architecture:** `docs/architecture/OVERVIEW.md`
- **Examples:** `docs/examples/`
- **Test Report:** `COMPREHENSIVE_TEST_REPORT.md`

## 🎯 Next Steps

1. **Deploy Validators:** Set up validator nodes for consensus
2. **Configure Peers:** Add bootstrap nodes for P2P networking
3. **Set Up Monitoring:** Configure Prometheus/Grafana dashboards
4. **Deploy Smart Contracts:** Deploy your contracts using Hardhat
5. **Build dApps:** Use the Python SDK or JSON-RPC directly

---

**Happy Building! 🚀**

For support, visit the GitHub repository or check the documentation.

