#!/usr/bin/env python3
"""
Hauptbuch Performance Benchmark Suite
Comprehensive performance testing for all blockchain operations.
"""

import asyncio
import time
import statistics
import psutil
import pytest
from typing import List, Dict, Any, Tuple
from hauptbuch_client import HauptbuchClient, AccountManager

class PerformanceBenchmark:
    """Base class for performance benchmarks"""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def start_timer(self):
        """Start performance timer"""
        self.start_time = time.time()
    
    def end_timer(self):
        """End performance timer"""
        self.end_time = time.time()
        return self.end_time - self.start_time
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "network_io": psutil.net_io_counters()._asdict()
        }

class TestTransactionThroughput:
    """Test transaction throughput performance"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_basic_transaction_throughput(self, client):
        """Test basic transaction throughput"""
        benchmark = PerformanceBenchmark()
        
        # Create test accounts
        accounts = []
        for i in range(10):
            account = AccountManager.create_account()
            accounts.append(account)
        
        # Test transaction throughput
        benchmark.start_timer()
        
        transaction_count = 100
        successful_transactions = 0
        
        for i in range(transaction_count):
            try:
                sender = accounts[i % len(accounts)]
                recipient = accounts[(i + 1) % len(accounts)]
                
                # Create and send transaction
                transaction = {
                    "from": sender.address,
                    "to": recipient.address,
                    "value": "0x1000",
                    "gas": "0x5208",
                    "gasPrice": "0x3b9aca00",
                    "nonce": "0x1",
                    "data": "0x"
                }
                
                # Sign and send transaction
                signed_tx = AccountManager.sign_transaction(transaction, sender.private_key)
                tx_hash = await client.send_transaction(signed_tx)
                
                if tx_hash:
                    successful_transactions += 1
                    
            except Exception as e:
                print(f"Transaction {i} failed: {e}")
        
        benchmark.end_timer()
        
        # Calculate metrics
        total_time = benchmark.end_time - benchmark.start_time
        tps = successful_transactions / total_time if total_time > 0 else 0
        
        # Verify performance
        assert tps > 0, f"Transaction throughput should be > 0, got {tps}"
        assert successful_transactions > 0, f"Should have successful transactions, got {successful_transactions}"
        
        print(f"Transaction Throughput: {tps:.2f} TPS")
        print(f"Successful Transactions: {successful_transactions}/{transaction_count}")
        print(f"Total Time: {total_time:.2f} seconds")
    
    @pytest.mark.asyncio
    async def test_smart_contract_transaction_throughput(self, client):
        """Test smart contract transaction throughput"""
        benchmark = PerformanceBenchmark()
        
        # Create test accounts
        accounts = []
        for i in range(5):
            account = AccountManager.create_account()
            accounts.append(account)
        
        # Test smart contract transaction throughput
        benchmark.start_timer()
        
        transaction_count = 50
        successful_transactions = 0
        
        for i in range(transaction_count):
            try:
                sender = accounts[i % len(accounts)]
                
                # Create smart contract transaction
                transaction = {
                    "from": sender.address,
                    "to": "0x1234567890123456789012345678901234567890",  # Contract address
                    "value": "0x0",
                    "gas": "0x100000",
                    "gasPrice": "0x3b9aca00",
                    "nonce": "0x1",
                    "data": "0x608060405234801561001057600080fd5b50"  # Contract call data
                }
                
                # Sign and send transaction
                signed_tx = AccountManager.sign_transaction(transaction, sender.private_key)
                tx_hash = await client.send_transaction(signed_tx)
                
                if tx_hash:
                    successful_transactions += 1
                    
            except Exception as e:
                print(f"Smart contract transaction {i} failed: {e}")
        
        benchmark.end_timer()
        
        # Calculate metrics
        total_time = benchmark.end_time - benchmark.start_time
        tps = successful_transactions / total_time if total_time > 0 else 0
        
        # Verify performance
        assert tps > 0, f"Smart contract transaction throughput should be > 0, got {tps}"
        assert successful_transactions > 0, f"Should have successful smart contract transactions, got {successful_transactions}"
        
        print(f"Smart Contract Transaction Throughput: {tps:.2f} TPS")
        print(f"Successful Transactions: {successful_transactions}/{transaction_count}")
        print(f"Total Time: {total_time:.2f} seconds")

class TestBlockProductionPerformance:
    """Test block production performance"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_block_production_speed(self, client):
        """Test block production speed"""
        benchmark = PerformanceBenchmark()
        
        # Get initial block height
        initial_chain_info = await client.get_chain_info()
        initial_height = int(initial_chain_info.block_height, 16)
        
        # Wait for new blocks
        benchmark.start_timer()
        
        new_blocks = 0
        max_wait_time = 60  # 60 seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            current_chain_info = await client.get_chain_info()
            current_height = int(current_chain_info.block_height, 16)
            
            if current_height > initial_height:
                new_blocks = current_height - initial_height
                break
            
            await asyncio.sleep(1)
        
        benchmark.end_timer()
        
        # Calculate metrics
        total_time = benchmark.end_time - benchmark.start_time
        block_time = total_time / new_blocks if new_blocks > 0 else float('inf')
        
        # Verify performance
        assert new_blocks > 0, f"Should have new blocks, got {new_blocks}"
        assert block_time > 0, f"Block time should be > 0, got {block_time}"
        
        print(f"Block Production Speed: {block_time:.2f} seconds per block")
        print(f"New Blocks: {new_blocks}")
        print(f"Total Time: {total_time:.2f} seconds")
    
    @pytest.mark.asyncio
    async def test_block_size_performance(self, client):
        """Test block size performance"""
        benchmark = PerformanceBenchmark()
        
        # Get recent blocks and analyze size
        chain_info = await client.get_chain_info()
        latest_height = int(chain_info.block_height, 16)
        
        block_sizes = []
        block_times = []
        
        # Analyze last 10 blocks
        for i in range(10):
            block_height = latest_height - i
            if block_height < 0:
                break
            
            block_start = time.time()
            block = await client.get_block(block_height, include_transactions=True)
            block_end = time.time()
            
            # Calculate block size (approximate)
            block_size = len(str(block))  # Rough estimate
            block_sizes.append(block_size)
            block_times.append(block_end - block_start)
        
        # Calculate metrics
        avg_block_size = statistics.mean(block_sizes) if block_sizes else 0
        avg_block_time = statistics.mean(block_times) if block_times else 0
        
        # Verify performance
        assert avg_block_size > 0, f"Average block size should be > 0, got {avg_block_size}"
        assert avg_block_time > 0, f"Average block time should be > 0, got {avg_block_time}"
        
        print(f"Average Block Size: {avg_block_size:.2f} bytes")
        print(f"Average Block Time: {avg_block_time:.2f} seconds")

class TestValidatorSelectionPerformance:
    """Test validator selection performance"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_validator_selection_speed(self, client):
        """Test validator selection speed"""
        benchmark = PerformanceBenchmark()
        
        # Test validator selection performance
        benchmark.start_timer()
        
        selection_times = []
        
        for i in range(10):
            selection_start = time.time()
            validator_set = await client.get_validator_set()
            selection_end = time.time()
            
            selection_times.append(selection_end - selection_start)
        
        benchmark.end_timer()
        
        # Calculate metrics
        avg_selection_time = statistics.mean(selection_times)
        max_selection_time = max(selection_times)
        min_selection_time = min(selection_times)
        
        # Verify performance
        assert avg_selection_time > 0, f"Average selection time should be > 0, got {avg_selection_time}"
        assert max_selection_time > 0, f"Max selection time should be > 0, got {max_selection_time}"
        assert min_selection_time > 0, f"Min selection time should be > 0, got {min_selection_time}"
        
        print(f"Average Validator Selection Time: {avg_selection_time:.4f} seconds")
        print(f"Max Selection Time: {max_selection_time:.4f} seconds")
        print(f"Min Selection Time: {min_selection_time:.4f} seconds")
    
    @pytest.mark.asyncio
    async def test_validator_set_size_performance(self, client):
        """Test validator set size performance"""
        benchmark = PerformanceBenchmark()
        
        # Test with different validator set sizes
        validator_set = await client.get_validator_set()
        validator_count = len(validator_set.validators)
        
        benchmark.start_timer()
        
        # Test validator set operations
        for i in range(100):
            validator_set = await client.get_validator_set()
        
        benchmark.end_timer()
        
        # Calculate metrics
        total_time = benchmark.end_timer()
        avg_time_per_operation = total_time / 100
        
        # Verify performance
        assert avg_time_per_operation > 0, f"Average time per operation should be > 0, got {avg_time_per_operation}"
        
        print(f"Validator Set Size: {validator_count}")
        print(f"Average Time per Operation: {avg_time_per_operation:.4f} seconds")
        print(f"Total Time for 100 Operations: {total_time:.2f} seconds")

class TestCryptographicPerformance:
    """Test cryptographic operation performance"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_key_generation_performance(self, client):
        """Test key generation performance"""
        benchmark = PerformanceBenchmark()
        
        # Test different key generation algorithms
        algorithms = ["ml-dsa", "ml-kem", "slh-dsa"]
        key_sizes = [256, 512, 1024]
        
        results = {}
        
        for algorithm in algorithms:
            for key_size in key_sizes:
                benchmark.start_timer()
                
                key_generation_times = []
                
                for i in range(10):
                    start_time = time.time()
                    keypair = await client.generate_keypair(algorithm, key_size)
                    end_time = time.time()
                    
                    key_generation_times.append(end_time - start_time)
                
                benchmark.end_timer()
                
                # Calculate metrics
                avg_time = statistics.mean(key_generation_times)
                max_time = max(key_generation_times)
                min_time = min(key_generation_times)
                
                results[f"{algorithm}_{key_size}"] = {
                    "avg_time": avg_time,
                    "max_time": max_time,
                    "min_time": min_time
                }
                
                print(f"{algorithm} {key_size}-bit Key Generation:")
                print(f"  Average Time: {avg_time:.4f} seconds")
                print(f"  Max Time: {max_time:.4f} seconds")
                print(f"  Min Time: {min_time:.4f} seconds")
        
        # Verify performance
        for algorithm in algorithms:
            for key_size in key_sizes:
                key = f"{algorithm}_{key_size}"
                assert results[key]["avg_time"] > 0, f"Average time for {key} should be > 0"
                assert results[key]["max_time"] > 0, f"Max time for {key} should be > 0"
                assert results[key]["min_time"] > 0, f"Min time for {key} should be > 0"
    
    @pytest.mark.asyncio
    async def test_signature_performance(self, client):
        """Test signature performance"""
        benchmark = PerformanceBenchmark()
        
        # Generate test keypair
        keypair = await client.generate_keypair("ml-dsa", 256)
        private_key = keypair["privateKey"]
        public_key = keypair["publicKey"]
        
        # Test signature performance
        benchmark.start_timer()
        
        signature_times = []
        verification_times = []
        
        for i in range(100):
            message = f"Test message {i}".encode('utf-8').hex()
            message_hex = "0x" + message
            
            # Test signing
            sign_start = time.time()
            signature = await client.sign_message(message_hex, private_key, "ml-dsa")
            sign_end = time.time()
            signature_times.append(sign_end - sign_start)
            
            # Test verification
            verify_start = time.time()
            is_valid = await client.verify_signature(message_hex, signature["signature"], public_key, "ml-dsa")
            verify_end = time.time()
            verification_times.append(verify_end - verify_start)
        
        benchmark.end_timer()
        
        # Calculate metrics
        avg_sign_time = statistics.mean(signature_times)
        avg_verify_time = statistics.mean(verification_times)
        max_sign_time = max(signature_times)
        max_verify_time = max(verification_times)
        
        # Verify performance
        assert avg_sign_time > 0, f"Average signature time should be > 0, got {avg_sign_time}"
        assert avg_verify_time > 0, f"Average verification time should be > 0, got {avg_verify_time}"
        assert max_sign_time > 0, f"Max signature time should be > 0, got {max_sign_time}"
        assert max_verify_time > 0, f"Max verification time should be > 0, got {max_verify_time}"
        
        print(f"Signature Performance:")
        print(f"  Average Sign Time: {avg_sign_time:.4f} seconds")
        print(f"  Average Verify Time: {avg_verify_time:.4f} seconds")
        print(f"  Max Sign Time: {max_sign_time:.4f} seconds")
        print(f"  Max Verify Time: {max_verify_time:.4f} seconds")

class TestNetworkPerformance:
    """Test network performance"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_peer_connection_performance(self, client):
        """Test peer connection performance"""
        benchmark = PerformanceBenchmark()
        
        # Test peer connection performance
        benchmark.start_timer()
        
        connection_times = []
        
        for i in range(10):
            start_time = time.time()
            peers = await client.get_peer_list()
            end_time = time.time()
            
            connection_times.append(end_time - start_time)
        
        benchmark.end_timer()
        
        # Calculate metrics
        avg_connection_time = statistics.mean(connection_times)
        max_connection_time = max(connection_times)
        min_connection_time = min(connection_times)
        
        # Verify performance
        assert avg_connection_time > 0, f"Average connection time should be > 0, got {avg_connection_time}"
        assert max_connection_time > 0, f"Max connection time should be > 0, got {max_connection_time}"
        assert min_connection_time > 0, f"Min connection time should be > 0, got {min_connection_time}"
        
        print(f"Peer Connection Performance:")
        print(f"  Average Connection Time: {avg_connection_time:.4f} seconds")
        print(f"  Max Connection Time: {max_connection_time:.4f} seconds")
        print(f"  Min Connection Time: {min_connection_time:.4f} seconds")
    
    @pytest.mark.asyncio
    async def test_network_latency(self, client):
        """Test network latency"""
        benchmark = PerformanceBenchmark()
        
        # Test network latency
        benchmark.start_timer()
        
        latency_times = []
        
        for i in range(50):
            start_time = time.time()
            network_info = await client.get_network_info()
            end_time = time.time()
            
            latency_times.append(end_time - start_time)
        
        benchmark.end_timer()
        
        # Calculate metrics
        avg_latency = statistics.mean(latency_times)
        max_latency = max(latency_times)
        min_latency = min(latency_times)
        median_latency = statistics.median(latency_times)
        
        # Verify performance
        assert avg_latency > 0, f"Average latency should be > 0, got {avg_latency}"
        assert max_latency > 0, f"Max latency should be > 0, got {max_latency}"
        assert min_latency > 0, f"Min latency should be > 0, got {min_latency}"
        assert median_latency > 0, f"Median latency should be > 0, got {median_latency}"
        
        print(f"Network Latency:")
        print(f"  Average Latency: {avg_latency:.4f} seconds")
        print(f"  Max Latency: {max_latency:.4f} seconds")
        print(f"  Min Latency: {min_latency:.4f} seconds")
        print(f"  Median Latency: {median_latency:.4f} seconds")

class TestDatabasePerformance:
    """Test database performance"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_block_retrieval_performance(self, client):
        """Test block retrieval performance"""
        benchmark = PerformanceBenchmark()
        
        # Get current block height
        chain_info = await client.get_chain_info()
        latest_height = int(chain_info.block_height, 16)
        
        # Test block retrieval performance
        benchmark.start_timer()
        
        retrieval_times = []
        
        # Test retrieving last 20 blocks
        for i in range(20):
            block_height = latest_height - i
            if block_height < 0:
                break
            
            start_time = time.time()
            block = await client.get_block(block_height, include_transactions=False)
            end_time = time.time()
            
            retrieval_times.append(end_time - start_time)
        
        benchmark.end_timer()
        
        # Calculate metrics
        avg_retrieval_time = statistics.mean(retrieval_times)
        max_retrieval_time = max(retrieval_times)
        min_retrieval_time = min(retrieval_times)
        
        # Verify performance
        assert avg_retrieval_time > 0, f"Average retrieval time should be > 0, got {avg_retrieval_time}"
        assert max_retrieval_time > 0, f"Max retrieval time should be > 0, got {max_retrieval_time}"
        assert min_retrieval_time > 0, f"Min retrieval time should be > 0, got {min_retrieval_time}"
        
        print(f"Block Retrieval Performance:")
        print(f"  Average Retrieval Time: {avg_retrieval_time:.4f} seconds")
        print(f"  Max Retrieval Time: {max_retrieval_time:.4f} seconds")
        print(f"  Min Retrieval Time: {min_retrieval_time:.4f} seconds")
    
    @pytest.mark.asyncio
    async def test_transaction_history_performance(self, client):
        """Test transaction history performance"""
        benchmark = PerformanceBenchmark()
        
        # Test transaction history performance
        benchmark.start_timer()
        
        history_times = []
        
        for i in range(10):
            start_time = time.time()
            history = await client.get_transaction_history("0x1234567890123456789012345678901234567890", limit=100)
            end_time = time.time()
            
            history_times.append(end_time - start_time)
        
        benchmark.end_timer()
        
        # Calculate metrics
        avg_history_time = statistics.mean(history_times)
        max_history_time = max(history_times)
        min_history_time = min(history_times)
        
        # Verify performance
        assert avg_history_time > 0, f"Average history time should be > 0, got {avg_history_time}"
        assert max_history_time > 0, f"Max history time should be > 0, got {max_history_time}"
        assert min_history_time > 0, f"Min history time should be > 0, got {min_history_time}"
        
        print(f"Transaction History Performance:")
        print(f"  Average History Time: {avg_history_time:.4f} seconds")
        print(f"  Max History Time: {max_history_time:.4f} seconds")
        print(f"  Min History Time: {min_history_time:.4f} seconds")

class TestMemoryUsage:
    """Test memory usage performance"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, client):
        """Test memory usage under load"""
        benchmark = PerformanceBenchmark()
        
        # Get initial memory usage
        initial_memory = psutil.virtual_memory().percent
        
        # Perform memory-intensive operations
        benchmark.start_timer()
        
        memory_usage = []
        
        for i in range(100):
            # Perform various operations
            network_info = await client.get_network_info()
            node_status = await client.get_node_status()
            chain_info = await client.get_chain_info()
            validator_set = await client.get_validator_set()
            
            # Record memory usage
            current_memory = psutil.virtual_memory().percent
            memory_usage.append(current_memory)
        
        benchmark.end_timer()
        
        # Calculate metrics
        max_memory = max(memory_usage)
        min_memory = min(memory_usage)
        avg_memory = statistics.mean(memory_usage)
        memory_increase = max_memory - initial_memory
        
        # Verify performance
        assert max_memory > 0, f"Max memory usage should be > 0, got {max_memory}"
        assert min_memory > 0, f"Min memory usage should be > 0, got {min_memory}"
        assert avg_memory > 0, f"Average memory usage should be > 0, got {avg_memory}"
        
        print(f"Memory Usage Performance:")
        print(f"  Initial Memory: {initial_memory:.2f}%")
        print(f"  Max Memory: {max_memory:.2f}%")
        print(f"  Min Memory: {min_memory:.2f}%")
        print(f"  Average Memory: {avg_memory:.2f}%")
        print(f"  Memory Increase: {memory_increase:.2f}%")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
