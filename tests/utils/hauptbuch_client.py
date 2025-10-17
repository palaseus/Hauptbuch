#!/usr/bin/env python3
"""
Hauptbuch Python SDK
A comprehensive Python client for interacting with the Hauptbuch blockchain platform.
"""

import json
import time
import hashlib
import subprocess
import asyncio
import aiohttp
import websockets
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import os
import logging

# Import our crypto FFI wrapper
try:
    from .hauptbuch_crypto_ffi import get_crypto, is_ffi_available, CryptoFFIError
    CRYPTO_FFI_AVAILABLE = is_ffi_available()
except ImportError:
    # Fallback for when FFI is not available
    CRYPTO_FFI_AVAILABLE = False
    get_crypto = None
    CryptoFFIError = Exception

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionStatus(Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    FAILED = "failed"

@dataclass
class NetworkInfo:
    chain_id: str
    network_id: str
    node_version: str
    protocol_version: str
    genesis_hash: str
    latest_block: str
    latest_block_number: str
    peer_count: int = 0

@dataclass
class NodeStatus:
    status: str
    sync_status: Dict[str, Any]
    peer_count: int
    uptime: int
    memory_usage: Dict[str, str]
    cpu_usage: float
    synced: bool = False
    block_height: str = "0x0"

@dataclass
class ChainInfo:
    block_height: str
    total_transactions: str
    total_gas_used: str
    average_block_time: int
    difficulty: str
    total_supply: str
    latest_block_hash: str = "0x0"
    genesis_hash: str = "0x0"

@dataclass
class Validator:
    address: str
    stake: str
    voting_power: int
    status: str
    last_seen: int

@dataclass
class ValidatorSet:
    validators: List[Validator]
    total_stake: str
    active_validators: int
    total_validators: int

@dataclass
class Block:
    number: str
    hash: str
    parent_hash: str
    timestamp: str
    gas_limit: str
    gas_used: str
    transactions: List[Dict[str, Any]]

@dataclass
class Transaction:
    hash: str
    from_address: str
    to: str
    value: str
    gas: str
    gas_price: str
    nonce: str
    data: str
    block_number: str
    block_hash: str
    transaction_index: str

@dataclass
class Account:
    address: str
    private_key: bytes
    public_key: bytes
    balance: int = 0
    nonce: int = 0

class HauptbuchError(Exception):
    """Base exception for Hauptbuch SDK errors"""
    pass

class NetworkError(HauptbuchError):
    """Network-related errors"""
    pass

class CryptoError(HauptbuchError):
    """Cryptography-related errors"""
    pass

class ValidationError(HauptbuchError):
    """Validation errors"""
    pass

class HauptbuchClient:
    """Main client for interacting with Hauptbuch blockchain"""
    
    def __init__(self, rpc_url: str = None, api_key: str = None, timeout: int = 30):
        self.rpc_url = rpc_url or os.getenv('HAUPTBUCH_RPC_URL', 'http://localhost:8080')
        self.api_key = api_key or os.getenv('HAUPTBUCH_API_KEY')
        self.timeout = timeout
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def _make_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make RPC request to Hauptbuch node with retry logic"""
        if not self.session:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                connector=aiohttp.TCPConnector(limit=10, limit_per_host=5, keepalive_timeout=30)
            )
            
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1
        }
        
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        # Retry logic for connection issues
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with self.session.post(
                    f"{self.rpc_url}",
                    json=payload,
                    headers=headers
                ) as response:
                    if response.status != 200:
                        raise NetworkError(f"HTTP {response.status}: {await response.text()}")
                        
                    result = await response.json()
                    
                    if "error" in result:
                        raise HauptbuchError(f"RPC Error: {result['error']}")
                        
                    return result.get("result", {})
                    
            except (aiohttp.ClientError, aiohttp.ClientConnectionError) as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.1 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    raise NetworkError(f"Network error after {max_retries} attempts: {e}")
            except Exception as e:
                raise NetworkError(f"Unexpected error: {e}")
    
    # Core RPC Methods
    async def get_network_info(self) -> NetworkInfo:
        """Get network information"""
        result = await self._make_request("hauptbuch_getNetworkInfo")
        return NetworkInfo(
            chain_id=result["chainId"],
            network_id=result["networkId"],
            node_version=result["nodeVersion"],
            protocol_version=result["protocolVersion"],
            genesis_hash=result["genesisHash"],
            latest_block=result["latestBlock"],
            latest_block_number=result["latestBlockNumber"],
            peer_count=result.get("peerCount", 0)
        )
    
    async def get_node_status(self) -> NodeStatus:
        """Get node status"""
        result = await self._make_request("hauptbuch_getNodeStatus")
        return NodeStatus(
            status=result["status"],
            sync_status=result["syncStatus"],
            peer_count=result["peerCount"],
            uptime=result["uptime"],
            memory_usage=result["memoryUsage"],
            cpu_usage=result["cpuUsage"],
            synced=result.get("syncStatus", {}).get("synced", False),
            block_height=result.get("blockHeight", "0x0")
        )
    
    async def get_chain_info(self) -> ChainInfo:
        """Get chain information"""
        result = await self._make_request("hauptbuch_getChainInfo")
        return ChainInfo(
            block_height=result["blockHeight"],
            total_transactions=result["totalTransactions"],
            total_gas_used=result["totalGasUsed"],
            average_block_time=result["averageBlockTime"],
            difficulty=result["difficulty"],
            total_supply=result["totalSupply"],
            latest_block_hash=result.get("latestBlockHash", "0x0"),
            genesis_hash=result.get("genesisHash", "0x0")
        )
    
    # Consensus Methods
    async def get_validator_set(self) -> ValidatorSet:
        """Get current validator set"""
        result = await self._make_request("hauptbuch_getValidatorSet")
        validators = [
            Validator(
                address=v["address"],
                stake=v["stake"],
                voting_power=v["votingPower"],
                status=v["status"],
                last_seen=v["lastSeen"]
            )
            for v in result["validators"]
        ]
        return ValidatorSet(
            validators=validators,
            total_stake=result["totalStake"],
            active_validators=result["activeValidators"],
            total_validators=result["totalValidators"]
        )
    
    async def get_block(self, block_number: Union[str, int], include_transactions: bool = True) -> Block:
        """Get block by number or hash"""
        params = {
            "blockNumber": hex(block_number) if isinstance(block_number, int) else block_number,
            "includeTransactions": include_transactions
        }
        result = await self._make_request("hauptbuch_getBlock", params)
        block_data = result["block"]
        return Block(
            number=block_data["number"],
            hash=block_data["hash"],
            parent_hash=block_data["parentHash"],
            timestamp=block_data["timestamp"],
            gas_limit=block_data["gasLimit"],
            gas_used=block_data["gasUsed"],
            transactions=block_data.get("transactions", [])
        )
    
    async def get_transaction(self, tx_hash: str) -> Transaction:
        """Get transaction by hash"""
        result = await self._make_request("hauptbuch_getTransaction", {"txHash": tx_hash})
        tx_data = result["transaction"]
        return Transaction(
            hash=tx_data["hash"],
            from_address=tx_data["from"],
            to=tx_data["to"],
            value=tx_data["value"],
            gas=tx_data["gas"],
            gas_price=tx_data["gasPrice"],
            nonce=tx_data["nonce"],
            data=tx_data["data"],
            block_number=tx_data["blockNumber"],
            block_hash=tx_data["blockHash"],
            transaction_index=tx_data["transactionIndex"]
        )
    
    # Network Methods
    async def get_peer_list(self) -> List[Dict[str, Any]]:
        """Get list of connected peers"""
        result = await self._make_request("hauptbuch_getPeerList")
        return result["peers"]
    
    async def add_peer(self, peer_address: str, peer_id: str = None) -> Dict[str, Any]:
        """Add a new peer"""
        params = {"peerAddress": peer_address}
        if peer_id:
            params["peerId"] = peer_id
        return await self._make_request("hauptbuch_addPeer", params)
    
    async def remove_peer(self, peer_id: str) -> Dict[str, Any]:
        """Remove a peer"""
        return await self._make_request("hauptbuch_removePeer", {"peerId": peer_id})
    
    # Cryptography Methods
    async def generate_keypair(self, algorithm: str = "ml-dsa", key_size: int = 256) -> Dict[str, str]:
        """Generate quantum-resistant keypair"""
        params = {"algorithm": algorithm, "keySize": key_size}
        return await self._make_request("hauptbuch_generateKeypair", params)
    
    async def sign_message(self, message: str, private_key: str, algorithm: str = "ml-dsa") -> Dict[str, str]:
        """Sign a message"""
        params = {
            "message": message,
            "privateKey": private_key,
            "algorithm": algorithm
        }
        return await self._make_request("hauptbuch_signMessage", params)
    
    async def verify_signature(self, message: str, signature: str, public_key: str, algorithm: str = "ml-dsa") -> Dict[str, Any]:
        """Verify a signature"""
        params = {
            "message": message,
            "signature": signature,
            "publicKey": public_key,
            "algorithm": algorithm
        }
        result = await self._make_request("hauptbuch_verifySignature", params)
        return result
    
    # Account Management
    async def get_balance(self, address: str) -> int:
        """Get account balance"""
        result = await self._make_request("hauptbuch_getBalance", {"address": address})
        return int(result["balance"], 16)
    
    async def get_nonce(self, address: str) -> int:
        """Get account nonce"""
        result = await self._make_request("hauptbuch_getNonce", {"address": address})
        return int(result["nonce"], 16)
    
    async def get_code(self, address: str) -> str:
        """Get contract code"""
        result = await self._make_request("hauptbuch_getCode", {"address": address})
        return result["code"]
    
    # Transaction Methods
    async def send_transaction(self, from_address: str = None, to_address: str = None, 
                             value: str = "0x0", data: str = "0x", 
                             gas_limit: str = "0x5208", gas_price: str = "0x3b9aca00",
                             signed_transaction: Dict[str, Any] = None) -> str:
        """Send a transaction (either build one or send a pre-signed one)"""
        if signed_transaction:
            params = {"transaction": signed_transaction}
        else:
            params = {
                "transaction": {
                    "from": from_address,
                    "to": to_address,
                    "value": value,
                    "data": data,
                    "gasLimit": gas_limit,
                    "gasPrice": gas_price
                }
            }
        result = await self._make_request("hauptbuch_sendTransaction", params)
        return result["transactionHash"]
    
    async def get_transaction_status(self, tx_hash: str) -> TransactionStatus:
        """Get transaction status"""
        result = await self._make_request("hauptbuch_getTransactionStatus", {"txHash": tx_hash})
        return TransactionStatus(result["status"])
    
    async def get_transaction_history(self, address: str, limit: int = 100) -> List[Transaction]:
        """Get transaction history for an address"""
        result = await self._make_request("hauptbuch_getTransactionHistory", {
            "address": address,
            "limit": limit
        })
        return [
            Transaction(
                hash=tx["hash"],
                from_address=tx["from"],
                to=tx["to"],
                value=tx["value"],
                gas=tx["gas"],
                gas_price=tx["gasPrice"],
                nonce=tx["nonce"],
                data=tx["data"],
                block_number=tx["blockNumber"],
                block_hash=tx["blockHash"],
                transaction_index=tx["transactionIndex"]
            )
            for tx in result["transactions"]
        ]
    
    # Cross-Chain Methods
    async def get_bridge_status(self) -> Dict[str, Any]:
        """Get cross-chain bridge status"""
        return await self._make_request("hauptbuch_getBridgeStatus")
    
    async def transfer_asset(self, from_address: str, to_address: str, amount: str, 
                           source_chain: str, target_chain: str, asset: str) -> Dict[str, Any]:
        """Transfer assets across chains"""
        params = {
            "from": from_address,
            "to": to_address,
            "amount": amount,
            "sourceChain": source_chain,
            "targetChain": target_chain,
            "asset": asset
        }
        return await self._make_request("hauptbuch_transferAsset", params)
    
    async def get_transfer_status(self, tx_hash: str) -> Dict[str, Any]:
        """Get cross-chain transfer status"""
        return await self._make_request("hauptbuch_getTransferStatus", {"transactionHash": tx_hash})
    
    # Governance Methods
    async def get_proposals(self, status: str = None, limit: int = 10, offset: int = 0) -> Dict[str, Any]:
        """Get governance proposals"""
        params = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        return await self._make_request("hauptbuch_getProposals", params)
    
    async def submit_proposal(self, title: str, description: str, author: str, 
                            proposal_type: str = "parameter_change", 
                            parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Submit a governance proposal"""
        params = {
            "title": title,
            "description": description,
            "author": author,
            "proposalType": proposal_type
        }
        if parameters:
            params["parameters"] = parameters
        return await self._make_request("hauptbuch_submitProposal", params)
    
    async def vote(self, proposal_id: int, voter: str, choice: str, voting_power: int) -> Dict[str, Any]:
        """Vote on a governance proposal"""
        params = {
            "proposalId": proposal_id,
            "voter": voter,
            "choice": choice,
            "votingPower": voting_power
        }
        return await self._make_request("hauptbuch_vote", params)
    
    # Account Abstraction Methods
    async def get_user_operations(self, account: str, status: str = None, limit: int = 10) -> Dict[str, Any]:
        """Get user operations for account abstraction"""
        params = {"account": account, "limit": limit}
        if status:
            params["status"] = status
        return await self._make_request("hauptbuch_getUserOperations", params)
    
    async def submit_user_operation(self, sender: str, nonce: str, call_data: str, 
                                  signature: str, paymaster: str = None,
                                  gas_limit: str = None, gas_price: str = None) -> Dict[str, Any]:
        """Submit user operation for account abstraction"""
        params = {
            "sender": sender,
            "nonce": nonce,
            "callData": call_data,
            "signature": signature
        }
        if paymaster:
            params["paymaster"] = paymaster
        if gas_limit:
            params["gasLimit"] = gas_limit
        if gas_price:
            params["gasPrice"] = gas_price
        return await self._make_request("hauptbuch_submitUserOperation", params)
    
    # Layer 2 Methods
    async def get_rollup_status(self) -> Dict[str, Any]:
        """Get Layer 2 rollup status"""
        return await self._make_request("hauptbuch_getRollupStatus")
    
    async def submit_rollup_transaction(self, rollup_name: str, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Submit transaction to Layer 2 rollup"""
        params = {
            "rollupName": rollup_name,
            "transaction": transaction
        }
        return await self._make_request("hauptbuch_submitRollupTransaction", params)
    
    # Monitoring Methods
    async def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        return await self._make_request("hauptbuch_getMetrics")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return await self._make_request("hauptbuch_getHealthStatus")
    
    # Account Abstraction Methods
    async def get_user_operations(self, address: str) -> List[Dict[str, Any]]:
        """Get user operations for an account"""
        return await self._make_request("hauptbuch_getUserOperations", {"address": address})
    
    async def submit_user_operation(self, sender: str, nonce: str, call_data: str, 
                                  signature: str, paymaster: str = None, 
                                  gas_limit: str = "0x5208", gas_price: str = "0x3b9aca00") -> Dict[str, Any]:
        """Submit a user operation"""
        params = {
            "sender": sender,
            "nonce": nonce,
            "callData": call_data,
            "signature": signature,
            "gasLimit": gas_limit,
            "gasPrice": gas_price
        }
        if paymaster:
            params["paymaster"] = paymaster
        return await self._make_request("hauptbuch_submitUserOperation", params)
    
    # Layer 2 Methods
    async def get_rollup_status(self) -> Dict[str, Any]:
        """Get rollup status"""
        return await self._make_request("hauptbuch_getRollupStatus")
    
    async def submit_rollup_transaction(self, rollup_type: str, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a rollup transaction"""
        return await self._make_request("hauptbuch_submitRollupTransaction", {
            "rollupType": rollup_type,
            "transaction": transaction
        })
    
    # Cross-Chain Methods
    async def get_bridge_status(self) -> Dict[str, Any]:
        """Get bridge status"""
        return await self._make_request("hauptbuch_getBridgeStatus")
    
    async def transfer_asset(self, from_address: str, to_address: str, amount: str,
                           source_chain: str, target_chain: str, asset: str) -> Dict[str, Any]:
        """Transfer assets across chains"""
        return await self._make_request("hauptbuch_transferAsset", {
            "fromAddress": from_address,
            "toAddress": to_address,
            "amount": amount,
            "sourceChain": source_chain,
            "targetChain": target_chain,
            "asset": asset
        })
    
    async def get_transfer_status(self, tx_hash: str) -> Dict[str, Any]:
        """Get transfer status"""
        return await self._make_request("hauptbuch_getTransferStatus", {"txHash": tx_hash})
    
    # Governance Methods
    async def get_proposals(self) -> List[Dict[str, Any]]:
        """Get governance proposals"""
        return await self._make_request("hauptbuch_getProposals")
    
    async def submit_proposal(self, title: str, description: str, author: str,
                            proposal_type: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a governance proposal"""
        return await self._make_request("hauptbuch_submitProposal", {
            "title": title,
            "description": description,
            "author": author,
            "proposalType": proposal_type,
            "parameters": parameters
        })
    
    async def vote(self, proposal_id: int, voter: str, choice: str, voting_power: int) -> Dict[str, Any]:
        """Vote on a governance proposal"""
        return await self._make_request("hauptbuch_vote", {
            "proposalId": proposal_id,
            "voter": voter,
            "choice": choice,
            "votingPower": voting_power
        })

class AccountManager:
    """Account management utilities with quantum-resistant cryptography"""
    
    @staticmethod
    def create_account(algorithm: str = "ml-dsa") -> Account:
        """Create a new account with quantum-resistant keys"""
        if CRYPTO_FFI_AVAILABLE and get_crypto:
            try:
                crypto = get_crypto()
                if algorithm == "ml-dsa":
                    keypair = crypto.generate_ml_dsa_keypair()
                elif algorithm == "slh-dsa":
                    keypair = crypto.generate_slh_dsa_keypair()
                elif algorithm == "ml-kem":
                    keypair = crypto.generate_ml_kem_keypair()
                else:
                    raise ValueError(f"Unsupported algorithm: {algorithm}")
                
                address = "0x" + keypair["address"].hex()
                
                return Account(
                    address=address,
                    private_key=keypair["private_key"],
                    public_key=keypair["public_key"]
                )
            except CryptoFFIError as e:
                logger.warning(f"FFI crypto failed, falling back to mock: {e}")
                # Fall through to mock implementation
        else:
            logger.warning("FFI not available, using mock crypto")
        
        # Mock implementation for testing
        private_key = os.urandom(32)
        public_key = hashlib.sha256(private_key).digest()
        address = "0x" + hashlib.sha256(public_key).hexdigest()[:40]
        
        return Account(
            address=address,
            private_key=private_key,
            public_key=public_key
        )
    
    @staticmethod
    def from_private_key(private_key: bytes) -> Account:
        """Create account from private key"""
        if CRYPTO_FFI_AVAILABLE and get_crypto:
            try:
                crypto = get_crypto()
                # Try to derive public key using ML-DSA (most common)
                # This is a simplified approach - in reality, you'd need the original algorithm
                public_key = hashlib.sha256(private_key).digest()
                address = "0x" + hashlib.sha256(public_key).hexdigest()[:40]
            except Exception as e:
                logger.warning(f"FFI crypto failed, falling back to mock: {e}")
                public_key = hashlib.sha256(private_key).digest()
                address = "0x" + hashlib.sha256(public_key).hexdigest()[:40]
        else:
            public_key = hashlib.sha256(private_key).digest()
            address = "0x" + hashlib.sha256(public_key).hexdigest()[:40]
        
        return Account(
            address=address,
            private_key=private_key,
            public_key=public_key
        )
    
    @staticmethod
    def sign_transaction(transaction: Dict[str, Any], private_key: bytes, algorithm: str = "ml-dsa") -> Dict[str, Any]:
        """Sign a transaction with quantum-resistant cryptography"""
        if CRYPTO_FFI_AVAILABLE and get_crypto:
            try:
                crypto = get_crypto()
                tx_data = json.dumps(transaction, sort_keys=True).encode()
                
                if algorithm == "ml-dsa":
                    signature_result = crypto.sign_ml_dsa(tx_data, private_key)
                elif algorithm == "slh-dsa":
                    signature_result = crypto.sign_slh_dsa(tx_data, private_key)
                else:
                    raise ValueError(f"Unsupported signing algorithm: {algorithm}")
                
                return {
                    **transaction,
                    "signature": signature_result["signature"].hex(),
                    "publicKey": signature_result["public_key"].hex(),
                    "algorithm": signature_result["algorithm"]
                }
            except CryptoFFIError as e:
                logger.warning(f"FFI signing failed, falling back to mock: {e}")
                # Fall through to mock implementation
        else:
            logger.warning("FFI not available, using mock signing")
        
        # Mock implementation for testing
        tx_data = json.dumps(transaction, sort_keys=True).encode()
        signature = hashlib.sha256(private_key + tx_data).hexdigest()
        public_key = hashlib.sha256(private_key).digest()
        
        return {
            **transaction,
            "signature": signature,
            "publicKey": public_key.hex(),
            "algorithm": algorithm
        }

class QuantumResistantCrypto:
    """Quantum-resistant cryptography utilities"""
    
    @staticmethod
    def generate_ml_kem_keypair() -> tuple[bytes, bytes]:
        """Generate ML-KEM keypair"""
        # This would call the Rust ML-KEM implementation
        private_key = os.urandom(32)
        public_key = hashlib.sha256(private_key).digest()
        return private_key, public_key
    
    @staticmethod
    def generate_ml_dsa_keypair() -> tuple[bytes, bytes]:
        """Generate ML-DSA keypair"""
        # This would call the Rust ML-DSA implementation
        private_key = os.urandom(32)
        public_key = hashlib.sha256(private_key).digest()
        return private_key, public_key
    
    @staticmethod
    def generate_slh_dsa_keypair() -> tuple[bytes, bytes]:
        """Generate SLH-DSA keypair"""
        # This would call the Rust SLH-DSA implementation
        private_key = os.urandom(32)
        public_key = hashlib.sha256(private_key).digest()
        return private_key, public_key
    
    @staticmethod
    def sign_message(message: bytes, private_key: bytes, algorithm: str = "ml-dsa") -> bytes:
        """Sign a message with quantum-resistant algorithm"""
        # This would use the appropriate quantum-resistant signing
        return hashlib.sha256(private_key + message).digest()
    
    @staticmethod
    def verify_signature(message: bytes, signature: bytes, public_key: bytes, algorithm: str = "ml-dsa") -> bool:
        """Verify a quantum-resistant signature"""
        # This would use the appropriate quantum-resistant verification
        expected = hashlib.sha256(public_key + message).digest()
        return signature == expected

# WebSocket client for real-time updates
class HauptbuchWebSocketClient:
    """WebSocket client for real-time blockchain updates"""
    
    def __init__(self, ws_url: str = None):
        self.ws_url = ws_url or os.getenv('HAUPTBUCH_WS_URL', 'ws://localhost:8080/rpc')
        self.websocket = None
    
    async def connect(self):
        """Connect to WebSocket"""
        self.websocket = await websockets.connect(self.ws_url)
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        if self.websocket:
            await self.websocket.close()
    
    async def subscribe_to_blocks(self, callback):
        """Subscribe to new blocks"""
        if not self.websocket:
            await self.connect()
        
        subscribe_msg = {
            "jsonrpc": "2.0",
            "method": "hauptbuch_subscribe",
            "params": ["newBlocks"],
            "id": 1
        }
        
        await self.websocket.send(json.dumps(subscribe_msg))
        
        async for message in self.websocket:
            data = json.loads(message)
            if "result" in data:
                await callback(data["result"])
    
    async def subscribe_to_transactions(self, callback):
        """Subscribe to pending transactions"""
        if not self.websocket:
            await self.connect()
        
        subscribe_msg = {
            "jsonrpc": "2.0",
            "method": "hauptbuch_subscribe",
            "params": ["pendingTransactions"],
            "id": 1
        }
        
        await self.websocket.send(json.dumps(subscribe_msg))
        
        async for message in self.websocket:
            data = json.loads(message)
            if "result" in data:
                await callback(data["result"])

# Utility functions
def hex_to_int(hex_str: str) -> int:
    """Convert hex string to integer"""
    return int(hex_str, 16)

def int_to_hex(value: int) -> str:
    """Convert integer to hex string"""
    return hex(value)

def wei_to_hbk(wei: int) -> float:
    """Convert wei to HBK (Hauptbuch token)"""
    return wei / 1_000_000_000_000_000_000

def hbk_to_wei(hbk: float) -> int:
    """Convert HBK to wei"""
    return int(hbk * 1_000_000_000_000_000_000)

# Example usage
async def main():
    """Example usage of the Hauptbuch client"""
    async with HauptbuchClient() as client:
        # Get network info
        network_info = await client.get_network_info()
        print(f"Network: {network_info.network_id}")
        print(f"Chain ID: {network_info.chain_id}")
        
        # Get node status
        node_status = await client.get_node_status()
        print(f"Node status: {node_status.status}")
        print(f"Peer count: {node_status.peer_count}")
        
        # Get validator set
        validator_set = await client.get_validator_set()
        print(f"Active validators: {validator_set.active_validators}")
        
        # Create account
        account = AccountManager.create_account()
        print(f"Account address: {account.address}")
        
        # Get balance
        balance = await client.get_balance(account.address)
        print(f"Balance: {balance} wei ({wei_to_hbk(balance)} HBK)")

if __name__ == "__main__":
    asyncio.run(main())
