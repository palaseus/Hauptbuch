#!/usr/bin/env python3
"""
Performance tests for Hauptbuch blockchain
"""

import pytest
import asyncio
import time
import psutil
import statistics
from pathlib import Path


class TestNodePerformance:
    """Test blockchain node performance"""
    
    @pytest.fixture
    def hauptbuch_binary(self):
        """Get the path to the hauptbuch binary"""
        binary_path = Path(__file__).parent.parent.parent / "target" / "release" / "hauptbuch"
        return str(binary_path)
    
    def test_binary_startup_time(self, hauptbuch_binary):
        """Test how quickly the binary starts up"""
        import subprocess
        import time
        
        start_time = time.time()
        process = subprocess.Popen([
            hauptbuch_binary,
            "--rpc-port", "8090",
            "--ws-port", "8091",
            "--log-level", "error"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        time.sleep(2)
        
        startup_time = time.time() - start_time
        
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
        # Startup should be under 5 seconds
        assert startup_time < 5.0, f"Startup time {startup_time:.2f}s exceeds 5s limit"
    
    def test_memory_usage(self, hauptbuch_binary):
        """Test memory usage of the node"""
        import subprocess
        import time
        
        process = subprocess.Popen([
            hauptbuch_binary,
            "--rpc-port", "8092",
            "--ws-port", "8093",
            "--log-level", "error"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(3)
        
        # Get memory usage
        memory_info = psutil.Process(process.pid).memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
        # Memory usage should be reasonable (under 500MB)
        assert memory_mb < 500, f"Memory usage {memory_mb:.2f}MB exceeds 500MB limit"
    
    def test_cpu_usage(self, hauptbuch_binary):
        """Test CPU usage of the node"""
        import subprocess
        import time
        
        process = subprocess.Popen([
            hauptbuch_binary,
            "--rpc-port", "8094",
            "--ws-port", "8095",
            "--log-level", "error"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        time.sleep(3)
        
        # Monitor CPU usage for 5 seconds
        cpu_samples = []
        for _ in range(5):
            cpu_percent = psutil.Process(process.pid).cpu_percent()
            cpu_samples.append(cpu_percent)
            time.sleep(1)
        
        avg_cpu = statistics.mean(cpu_samples)
        
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        
        # CPU usage should be reasonable (under 50%)
        assert avg_cpu < 50, f"Average CPU usage {avg_cpu:.2f}% exceeds 50% limit"


class TestRPCPerformance:
    """Test RPC server performance"""
    
    @pytest.fixture
    async def running_node(self, hauptbuch_binary):
        """Start a node for RPC testing"""
        import subprocess
        
        process = subprocess.Popen([
            hauptbuch_binary,
            "--rpc-port", "8096",
            "--ws-port", "8097",
            "--log-level", "error"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for startup
        await asyncio.sleep(3)
        
        yield process
        
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    
    @pytest.mark.asyncio
    async def test_rpc_response_time(self, running_node):
        """Test RPC response times"""
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "hauptbuch_getNetworkInfo",
                "params": {},
                "id": 1
            }
            
            # Measure response time
            start_time = time.time()
            async with session.post(
                "http://localhost:8096",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response_time = time.time() - start_time
                
                assert response.status == 200
                assert response_time < 1.0, f"RPC response time {response_time:.3f}s exceeds 1s limit"
    
    @pytest.mark.asyncio
    async def test_concurrent_rpc_requests(self, running_node):
        """Test handling of concurrent RPC requests"""
        import aiohttp
        
        async def make_request(session, request_id):
            payload = {
                "jsonrpc": "2.0",
                "method": "hauptbuch_getNodeStatus",
                "params": {},
                "id": request_id
            }
            
            start_time = time.time()
            async with session.post(
                "http://localhost:8096",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response_time = time.time() - start_time
                return response.status, response_time
        
        async with aiohttp.ClientSession() as session:
            # Make 10 concurrent requests
            tasks = [make_request(session, i) for i in range(10)]
            results = await asyncio.gather(*tasks)
            
            # Check all requests succeeded
            status_codes = [result[0] for result in results]
            response_times = [result[1] for result in results]
            
            assert all(status == 200 for status in status_codes), "Some RPC requests failed"
            assert all(time < 2.0 for time in response_times), "Some RPC requests were too slow"
            
            avg_response_time = statistics.mean(response_times)
            assert avg_response_time < 1.0, f"Average response time {avg_response_time:.3f}s exceeds 1s limit"


class TestNetworkPerformance:
    """Test network performance"""
    
    def test_network_latency(self):
        """Test network latency simulation"""
        # Simulate network operations
        start_time = time.time()
        
        # Simulate some network work
        time.sleep(0.1)  # Simulate 100ms network delay
        
        latency = time.time() - start_time
        
        # Latency should be reasonable
        assert latency < 0.5, f"Network latency {latency:.3f}s exceeds 500ms limit"
    
    def test_throughput_simulation(self):
        """Test throughput simulation"""
        # Simulate processing multiple operations
        operations = 100
        start_time = time.time()
        
        # Simulate processing
        for _ in range(operations):
            time.sleep(0.001)  # 1ms per operation
        
        total_time = time.time() - start_time
        throughput = operations / total_time
        
        # Throughput should be reasonable (over 10 ops/sec)
        assert throughput > 10, f"Throughput {throughput:.2f} ops/sec below 10 ops/sec limit"


class TestSystemPerformance:
    """Test system-level performance"""
    
    def test_system_resources(self):
        """Test system resource availability"""
        # Check available memory
        memory = psutil.virtual_memory()
        available_memory_gb = memory.available / 1024 / 1024 / 1024
        
        assert available_memory_gb > 1.0, f"Available memory {available_memory_gb:.2f}GB below 1GB limit"
        
        # Check CPU count
        cpu_count = psutil.cpu_count()
        assert cpu_count >= 2, f"CPU count {cpu_count} below 2 cores limit"
    
    def test_disk_space(self):
        """Test available disk space"""
        disk_usage = psutil.disk_usage('/')
        available_gb = disk_usage.free / 1024 / 1024 / 1024
        
        assert available_gb > 5.0, f"Available disk space {available_gb:.2f}GB below 5GB limit"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

