#!/usr/bin/env python3
"""
Comprehensive Blockchain Interaction Test
Tests every conceivable way to interact with the Hauptbuch blockchain
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from hauptbuch_client import HauptbuchClient, AccountManager, QuantumResistantCrypto
import json
from datetime import datetime

class ComprehensiveBlockchainTest:
    def __init__(self):
        self.client = None
        self.test_accounts = []
        self.results = {
            "passed": [],
            "failed": [],
            "warnings": []
        }
    
    async def setup(self):
        """Initialize client and create test accounts"""
        print("\n" + "="*80)
        print("🚀 COMPREHENSIVE BLOCKCHAIN INTERACTION TEST")
        print("="*80 + "\n")
        
        self.client = HauptbuchClient()
        await self.client.__aenter__()
        
        print("📝 Creating test accounts...")
        for i in range(5):
            account = AccountManager.create_account()
            self.test_accounts.append(account)
            print(f"   Account {i+1}: {account.address}")
        print()
    
    async def test_network_info(self):
        """Test 1: Network Information"""
        print("🌐 Test 1: Network Information")
        try:
            network_info = await self.client.get_network_info()
            print(f"   Network ID: {network_info.network_id}")
            print(f"   Chain ID: {network_info.chain_id}")
            print(f"   Protocol Version: {network_info.protocol_version}")
            print(f"   Peer Count: {network_info.peer_count}")
            self.results["passed"].append("Network Info")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["failed"].append(f"Network Info: {e}")
        print()
    
    async def test_node_status(self):
        """Test 2: Node Status"""
        print("⚡ Test 2: Node Status")
        try:
            status = await self.client.get_node_status()
            print(f"   Status: {status.status}")
            print(f"   Uptime: {status.uptime}s")
            print(f"   Synced: {status.synced}")
            print(f"   Block Height: {status.block_height}")
            self.results["passed"].append("Node Status")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["failed"].append(f"Node Status: {e}")
        print()
    
    async def test_chain_info(self):
        """Test 3: Chain Information"""
        print("⛓️  Test 3: Chain Information")
        try:
            chain_info = await self.client.get_chain_info()
            print(f"   Block Height: {chain_info.block_height}")
            print(f"   Latest Block Hash: {chain_info.latest_block_hash}")
            print(f"   Genesis Hash: {chain_info.genesis_hash}")
            self.results["passed"].append("Chain Info")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["failed"].append(f"Chain Info: {e}")
        print()
    
    async def test_validator_set(self):
        """Test 4: Validator Set"""
        print("👥 Test 4: Validator Set")
        try:
            validator_set = await self.client.get_validator_set()
            print(f"   Active Validators: {validator_set.active_validators}")
            print(f"   Total Validators: {validator_set.total_validators}")
            print(f"   Total Stake: {validator_set.total_stake}")
            self.results["passed"].append("Validator Set")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["failed"].append(f"Validator Set: {e}")
        print()
    
    async def test_crypto_operations(self):
        """Test 5: Cryptographic Operations"""
        print("🔐 Test 5: Quantum-Resistant Cryptography")
        try:
            # Generate ML-KEM keypair
            keypair = await self.client.generate_keypair("ml-kem", 2048)
            print(f"   ML-KEM Public Key: {keypair['publicKey'][:64]}...")
            
            # Sign a message
            message = "Hello, Hauptbuch!"
            signature = await self.client.sign_message(
                self.test_accounts[0].private_key.hex(),
                message,
                "ml-dsa"
            )
            print(f"   ML-DSA Signature: {signature['signature'][:64]}...")
            
            # Verify signature
            verification = await self.client.verify_signature(
                signature['signature'],
                message,
                keypair['publicKey'],
                "ml-dsa"
            )
            print(f"   Signature Valid: {verification['valid']}")
            
            self.results["passed"].append("Crypto Operations")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["failed"].append(f"Crypto Operations: {e}")
        print()
    
    async def test_account_operations(self):
        """Test 6: Account Operations"""
        print("💰 Test 6: Account Operations")
        try:
            account = self.test_accounts[0]
            
            # Get balance
            balance = await self.client.get_balance(account.address)
            print(f"   Balance: {balance}")
            
            # Get nonce
            nonce = await self.client.get_nonce(account.address)
            print(f"   Nonce: {nonce}")
            
            # Get code (for contract accounts)
            code = await self.client.get_code(account.address)
            print(f"   Code: {code}")
            
            self.results["passed"].append("Account Operations")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["failed"].append(f"Account Operations: {e}")
        print()
    
    async def test_transactions(self):
        """Test 7: Transaction Operations"""
        print("📤 Test 7: Transaction Operations")
        try:
            sender = self.test_accounts[0]
            recipient = self.test_accounts[1]
            
            # Send transaction
            tx_hash = await self.client.send_transaction(
                from_address=sender.address,
                to_address=recipient.address,
                value="0x1000",
                data="0x",
                gas_limit="0x5208",
                gas_price="0x3b9aca00"
            )
            print(f"   Transaction Hash: {tx_hash}")
            
            # Get transaction status
            status = await self.client.get_transaction_status(tx_hash)
            print(f"   Transaction Status: {status}")
            
            # Get transaction history
            history = await self.client.get_transaction_history(sender.address)
            print(f"   Transaction History Count: {len(history)}")
            
            self.results["passed"].append("Transaction Operations")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["failed"].append(f"Transaction Operations: {e}")
        print()
    
    async def test_block_operations(self):
        """Test 8: Block Operations"""
        print("🧱 Test 8: Block Operations")
        try:
            # Get latest block
            block = await self.client.get_block(0)
            print(f"   Block Number: {block.number}")
            print(f"   Block Hash: {block.hash}")
            print(f"   Timestamp: {block.timestamp}")
            print(f"   Transaction Count: {len(block.transactions)}")
            
            self.results["passed"].append("Block Operations")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["failed"].append(f"Block Operations: {e}")
        print()
    
    async def test_user_operations(self):
        """Test 9: ERC-4337 User Operations"""
        print("👤 Test 9: Account Abstraction (ERC-4337)")
        try:
            sender = self.test_accounts[0]
            
            # Get user operations
            user_ops = await self.client.get_user_operations(sender.address)
            print(f"   Total User Operations: {user_ops.get('totalOperations', 0)}")
            print(f"   Pending Operations: {user_ops.get('pendingOperations', 0)}")
            
            # Submit user operation
            user_op = await self.client.submit_user_operation(
                sender=sender.address,
                nonce="0x1",
                call_data="0x608060405234801561001057600080fd5b50",
                signature="0x1234567890abcdef",
                gas_limit="0x5208",
                gas_price="0x3b9aca00"
            )
            print(f"   User Operation Hash: {user_op.get('userOperationHash', 'N/A')}")
            
            self.results["passed"].append("User Operations")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["failed"].append(f"User Operations: {e}")
        print()
    
    async def test_rollup_operations(self):
        """Test 10: Layer 2 Rollup Operations"""
        print("🔄 Test 10: Layer 2 Rollups")
        try:
            # Get rollup status
            rollup_status = await self.client.get_rollup_status()
            print(f"   Active Rollups: {rollup_status.get('activeRollups', 0)}")
            print(f"   Total Rollups: {rollup_status.get('totalRollups', 0)}")
            
            # Submit rollup transaction
            sender = self.test_accounts[0]
            recipient = self.test_accounts[1]
            
            tx = {
                "from": sender.address,
                "to": recipient.address,
                "value": "0x1000",
                "data": "0x"
            }
            
            result = await self.client.submit_rollup_transaction("optimistic-rollup", tx)
            print(f"   Rollup TX Hash: {result.get('transactionHash', 'N/A')}")
            
            self.results["passed"].append("Rollup Operations")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["failed"].append(f"Rollup Operations: {e}")
        print()
    
    async def test_cross_chain(self):
        """Test 11: Cross-Chain Operations"""
        print("🌉 Test 11: Cross-Chain Bridge")
        try:
            # Get bridge status
            bridge_status = await self.client.get_bridge_status()
            print(f"   Active Bridges: {bridge_status.get('activeBridges', 0)}")
            print(f"   Total Transfers: {bridge_status.get('totalTransfers', 0)}")
            
            # Transfer asset
            sender = self.test_accounts[0]
            recipient = self.test_accounts[1]
            
            transfer = await self.client.transfer_asset(
                from_address=sender.address,
                to_address=recipient.address,
                amount="1000000000000000000",
                source_chain="hauptbuch",
                target_chain="ethereum",
                asset="HBK"
            )
            print(f"   Transfer TX Hash: {transfer.get('transactionHash', 'N/A')}")
            
            self.results["passed"].append("Cross-Chain Operations")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["failed"].append(f"Cross-Chain Operations: {e}")
        print()
    
    async def test_governance(self):
        """Test 12: Governance Operations"""
        print("🗳️  Test 12: Governance")
        try:
            # Get proposals
            proposals = await self.client.get_proposals()
            print(f"   Total Proposals: {len(proposals)}")
            
            # Submit proposal
            author = self.test_accounts[0]
            proposal = await self.client.submit_proposal(
                title="Increase Block Size",
                description="Proposal to increase block size to 2MB",
                author=author.address,
                proposal_type="parameter_change",
                parameters={"blockSize": 2000000}
            )
            print(f"   Proposal ID: {proposal.get('proposalId', 'N/A')}")
            
            # Vote on proposal
            voter = self.test_accounts[1]
            vote = await self.client.vote(
                proposal_id=1,
                voter=voter.address,
                choice="yes",
                voting_power=1000
            )
            print(f"   Vote Success: {vote.get('success', False)}")
            
            self.results["passed"].append("Governance Operations")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["failed"].append(f"Governance Operations: {e}")
        print()
    
    async def test_monitoring(self):
        """Test 13: Monitoring and Metrics"""
        print("📊 Test 13: Monitoring & Metrics")
        try:
            # Get metrics
            metrics = await self.client.get_metrics()
            print(f"   Block Height: {metrics.get('metrics', {}).get('blockHeight', 'N/A')}")
            print(f"   Transaction Count: {metrics.get('metrics', {}).get('transactionCount', 'N/A')}")
            print(f"   Peer Count: {metrics.get('metrics', {}).get('peerCount', 'N/A')}")
            print(f"   Memory Usage: {metrics.get('metrics', {}).get('memoryUsage', 'N/A')}")
            
            # Get health status
            health = await self.client.get_health_status()
            print(f"   Health Status: {health.get('status', 'N/A')}")
            
            self.results["passed"].append("Monitoring Operations")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["failed"].append(f"Monitoring Operations: {e}")
        print()
    
    async def test_peer_management(self):
        """Test 14: Peer Management"""
        print("🤝 Test 14: P2P Network & Peer Management")
        try:
            # Get peer list
            peers = await self.client.get_peer_list()
            print(f"   Connected Peers: {len(peers)}")
            
            # Add peer
            result = await self.client.add_peer("127.0.0.1:30304")
            print(f"   Add Peer Result: {result.get('success', False)}")
            
            self.results["passed"].append("Peer Management")
        except Exception as e:
            print(f"   ❌ Error: {e}")
            self.results["failed"].append(f"Peer Management: {e}")
        print()
    
    async def print_summary(self):
        """Print test summary"""
        print("\n" + "="*80)
        print("📋 TEST SUMMARY")
        print("="*80 + "\n")
        
        total_tests = len(self.results["passed"]) + len(self.results["failed"])
        print(f"✅ Passed: {len(self.results['passed'])}/{total_tests}")
        print(f"❌ Failed: {len(self.results['failed'])}/{total_tests}")
        print(f"⚠️  Warnings: {len(self.results['warnings'])}")
        
        if self.results["passed"]:
            print("\n✅ Passed Tests:")
            for test in self.results["passed"]:
                print(f"   ✓ {test}")
        
        if self.results["failed"]:
            print("\n❌ Failed Tests:")
            for test in self.results["failed"]:
                print(f"   ✗ {test}")
        
        if self.results["warnings"]:
            print("\n⚠️  Warnings:")
            for warning in self.results["warnings"]:
                print(f"   ⚠ {warning}")
        
        success_rate = (len(self.results["passed"]) / total_tests * 100) if total_tests > 0 else 0
        print(f"\n🎯 Success Rate: {success_rate:.1f}%")
        
        if success_rate == 100:
            print("\n🎉 ALL TESTS PASSED! Blockchain is fully operational!")
        elif success_rate >= 80:
            print("\n✅ Most tests passed. Blockchain is operational with minor issues.")
        else:
            print("\n⚠️  Several tests failed. Please review the errors above.")
        
        print("\n" + "="*80 + "\n")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client.__aexit__(None, None, None)
    
    async def run_all_tests(self):
        """Run all comprehensive tests"""
        await self.setup()
        
        # Run all tests
        await self.test_network_info()
        await self.test_node_status()
        await self.test_chain_info()
        await self.test_validator_set()
        await self.test_crypto_operations()
        await self.test_account_operations()
        await self.test_transactions()
        await self.test_block_operations()
        await self.test_user_operations()
        await self.test_rollup_operations()
        await self.test_cross_chain()
        await self.test_governance()
        await self.test_monitoring()
        await self.test_peer_management()
        
        await self.print_summary()
        await self.cleanup()

async def main():
    test = ComprehensiveBlockchainTest()
    await test.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())

