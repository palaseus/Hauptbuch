#!/usr/bin/env python3
"""
Advanced Wallet and Transaction Workflow Test
Creates wallets, runs transactions, and tests complex scenarios
"""

import asyncio
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from hauptbuch_client import HauptbuchClient, AccountManager
import json
from datetime import datetime

class AdvancedWalletTest:
    def __init__(self):
        self.client = None
        self.wallets = []
        
    async def setup(self):
        """Initialize client"""
        print("\n" + "="*80)
        print("💼 ADVANCED WALLET & TRANSACTION WORKFLOW TEST")
        print("="*80 + "\n")
        
        self.client = HauptbuchClient()
        await self.client.__aenter__()
    
    async def create_wallets(self, count=10):
        """Create multiple wallets"""
        print(f"💰 Creating {count} wallets...")
        for i in range(count):
            account = AccountManager.create_account()
            self.wallets.append(account)
            print(f"   Wallet {i+1:2d}: {account.address} (nonce: {account.nonce}, balance: {account.balance})")
        print()
    
    async def test_batch_transactions(self):
        """Test sending multiple transactions"""
        print("📤 Testing Batch Transactions...")
        tx_hashes = []
        
        # Send 5 transactions
        for i in range(5):
            sender = self.wallets[i]
            recipient = self.wallets[i + 5]
            
            tx_hash = await self.client.send_transaction(
                from_address=sender.address,
                to_address=recipient.address,
                value=hex(1000000 * (i + 1)),  # Varying amounts
                data="0x",
                gas_limit="0x5208",
                gas_price=hex(1000000000 + i * 1000000)  # Varying gas prices
            )
            
            print(f"   TX {i+1}: {sender.address[:10]}... → {recipient.address[:10]}... = {tx_hash[:20]}...")
            tx_hashes.append(tx_hash)
        
        print(f"\n   ✅ Sent {len(tx_hashes)} transactions")
        return tx_hashes
    
    async def test_transaction_history(self):
        """Test getting transaction history"""
        print("\n📜 Testing Transaction History...")
        for i in range(3):
            account = self.wallets[i]
            history = await self.client.get_transaction_history(account.address)
            print(f"   Account {i+1} ({account.address[:10]}...): {len(history)} transactions")
        print()
    
    async def test_account_balances(self):
        """Test checking balances"""
        print("💵 Checking Account Balances...")
        for i in range(5):
            account = self.wallets[i]
            balance = await self.client.get_balance(account.address)
            nonce = await self.client.get_nonce(account.address)
            print(f"   Account {i+1}: Balance = {balance} wei, Nonce = {nonce}")
        print()
    
    async def test_crypto_operations(self):
        """Test various cryptographic operations"""
        print("🔐 Testing Cryptographic Operations...")
        
        # Test all 3 quantum-resistant algorithms
        algorithms = ["ml-kem", "ml-dsa", "slh-dsa"]
        
        for algo in algorithms:
            # Generate keypair
            keypair = await self.client.generate_keypair(algo, 2048)
            print(f"   {algo.upper()}: Generated keypair")
            
            # Sign message with ML-DSA
            if algo == "ml-dsa" or algo == "slh-dsa":
                message = f"Test message for {algo}"
                signature = await self.client.sign_message(
                    self.wallets[0].private_key.hex(),
                    message,
                    algo
                )
                print(f"   {algo.upper()}: Signed message")
                
                # Verify signature
                verification = await self.client.verify_signature(
                    message,
                    signature['signature'],
                    keypair['publicKey'],
                    algo
                )
                print(f"   {algo.upper()}: Verified signature = {verification.get('valid', False)}")
        print()
    
    async def test_user_operation_workflow(self):
        """Test ERC-4337 user operation workflow"""
        print("👤 Testing User Operation Workflow...")
        
        account = self.wallets[0]
        
        # Submit 3 user operations
        for i in range(3):
            user_op = await self.client.submit_user_operation(
                sender=account.address,
                nonce=hex(i + 1),
                call_data=f"0x{i:064x}",  # Different call data
                signature="0x" + "ab" * 32,
                gas_limit="0x5208",
                gas_price="0x3b9aca00"
            )
            print(f"   UserOp {i+1}: {user_op.get('userOperationHash', 'N/A')[:20]}...")
        
        # Get user operations
        user_ops = await self.client.get_user_operations(account.address)
        print(f"\n   ✅ Total User Operations: {user_ops.get('totalOperations', 0)}")
        print()
    
    async def test_rollup_workflow(self):
        """Test Layer 2 rollup workflow"""
        print("🔄 Testing Rollup Workflow...")
        
        # Test different rollup types
        rollup_types = ["optimistic-rollup", "zk-rollup", "plasma"]
        
        for rollup_type in rollup_types:
            sender = self.wallets[0]
            recipient = self.wallets[1]
            
            tx = {
                "from": sender.address,
                "to": recipient.address,
                "value": "0x2710",  # 10000 wei
                "data": "0x"
            }
            
            result = await self.client.submit_rollup_transaction(rollup_type, tx)
            print(f"   {rollup_type}: {result.get('transactionHash', 'N/A')[:20]}...")
        
        # Get rollup status
        rollup_status = await self.client.get_rollup_status()
        print(f"\n   Active Rollups: {rollup_status.get('activeRollups', 0)}")
        print()
    
    async def test_governance_workflow(self):
        """Test governance proposal and voting workflow"""
        print("🗳️  Testing Governance Workflow...")
        
        # Create 3 proposals
        proposals_created = []
        for i in range(3):
            author = self.wallets[i]
            proposal = await self.client.submit_proposal(
                title=f"Proposal #{i+1}: Test Parameter Change",
                description=f"This is test proposal number {i+1}",
                author=author.address,
                proposal_type="parameter_change",
                parameters={"testParam": i * 100}
            )
            proposals_created.append(proposal.get('proposalId', f'P{i+1}'))
            print(f"   Created Proposal {i+1}: ID = {proposals_created[-1]}")
        
        # Vote on proposals
        print("\n   Voting on proposals...")
        for i in range(5):
            voter = self.wallets[i]
            proposal_id = (i % len(proposals_created)) + 1
            choice = "yes" if i % 2 == 0 else "no"
            
            vote = await self.client.vote(
                proposal_id=proposal_id,
                voter=voter.address,
                choice=choice,
                voting_power=1000 * (i + 1)
            )
            print(f"   Vote {i+1}: Proposal {proposal_id} - {choice.upper()} (power: {1000*(i+1)})")
        
        # Get all proposals
        proposals = await self.client.get_proposals()
        print(f"\n   ✅ Total Proposals: {len(proposals)}")
        print()
    
    async def test_cross_chain_workflow(self):
        """Test cross-chain bridge workflow"""
        print("🌉 Testing Cross-Chain Workflow...")
        
        # Test transfers to different chains
        chains = ["ethereum", "binance-smart-chain", "polygon"]
        
        for i, target_chain in enumerate(chains):
            sender = self.wallets[i]
            recipient = self.wallets[i + 3]
            
            transfer = await self.client.transfer_asset(
                from_address=sender.address,
                to_address=recipient.address,
                amount=str(int(1e18) * (i + 1)),  # 1, 2, 3 tokens
                source_chain="hauptbuch",
                target_chain=target_chain,
                asset="HBK"
            )
            
            tx_hash = transfer.get('transactionHash', 'N/A')
            print(f"   Transfer {i+1}: hauptbuch → {target_chain} = {tx_hash[:20]}...")
            
            # Check transfer status
            if tx_hash != 'N/A':
                status = await self.client.get_transfer_status(tx_hash)
                print(f"              Status: {status.get('status', 'unknown')}")
        
        # Get bridge status
        bridge_status = await self.client.get_bridge_status()
        print(f"\n   Active Bridges: {bridge_status.get('activeBridges', 0)}")
        print()
    
    async def test_monitoring_and_health(self):
        """Test monitoring and health endpoints"""
        print("📊 Testing Monitoring & Health...")
        
        # Get comprehensive metrics
        metrics = await self.client.get_metrics()
        print("   System Metrics:")
        if 'metrics' in metrics:
            m = metrics['metrics']
            print(f"      Block Height: {m.get('blockHeight', 'N/A')}")
            print(f"      TX Count: {m.get('transactionCount', 'N/A')}")
            print(f"      Peers: {m.get('peerCount', 'N/A')}")
            print(f"      Memory: {m.get('memoryUsage', 'N/A')}")
            print(f"      CPU: {m.get('cpuUsage', 'N/A')}%")
        
        # Get health status
        health = await self.client.get_health_status()
        print(f"\n   Health Status: {health.get('status', 'N/A')}")
        
        if 'components' in health:
            print("   Component Health:")
            for component, status in health['components'].items():
                print(f"      {component}: {status}")
        print()
    
    async def print_summary(self):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE SUMMARY")
        print("="*80 + "\n")
        
        print(f"💼 Total Wallets Created: {len(self.wallets)}")
        print(f"🌐 Network: hauptbuch-testnet-1 (Chain ID: 1337)")
        
        # Get final network state
        network_info = await self.client.get_network_info()
        node_status = await self.client.get_node_status()
        chain_info = await self.client.get_chain_info()
        
        print(f"\n📈 Network State:")
        print(f"   Block Height: {chain_info.block_height}")
        print(f"   Total Transactions: {chain_info.total_transactions}")
        print(f"   Node Status: {node_status.status}")
        print(f"   Uptime: {node_status.uptime}s")
        print(f"   Peers: {network_info.peer_count}")
        
        print(f"\n✅ All interaction tests completed successfully!")
        print("\n" + "="*80 + "\n")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.client:
            await self.client.__aexit__(None, None, None)
    
    async def run_all_tests(self):
        """Run all advanced tests"""
        await self.setup()
        await self.create_wallets(10)
        
        tx_hashes = await self.test_batch_transactions()
        await self.test_transaction_history()
        await self.test_account_balances()
        await self.test_crypto_operations()
        await self.test_user_operation_workflow()
        await self.test_rollup_workflow()
        await self.test_governance_workflow()
        await self.test_cross_chain_workflow()
        await self.test_monitoring_and_health()
        
        await self.print_summary()
        await self.cleanup()

async def main():
    test = AdvancedWalletTest()
    await test.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())

