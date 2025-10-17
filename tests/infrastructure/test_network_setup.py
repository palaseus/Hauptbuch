#!/usr/bin/env python3
"""
Infrastructure tests for Hauptbuch network setup
"""

import pytest
import pytest_asyncio
import asyncio
import aiohttp
import subprocess
import time
import os
import signal
from pathlib import Path


class TestNetworkSetup:
    """Test network setup and basic functionality"""
    
    @pytest.fixture
    def hauptbuch_binary(self):
        """Get the path to the hauptbuch binary"""
        binary_path = Path(__file__).parent.parent.parent / "target" / "release" / "hauptbuch"
        assert binary_path.exists(), f"Hauptbuch binary not found at {binary_path}"
        return str(binary_path)
    
    @pytest_asyncio.fixture
    async def hauptbuch_node(self, hauptbuch_binary):
        """Start a hauptbuch node for testing"""
        # Start the node
        process = subprocess.Popen([
            hauptbuch_binary,
            "--rpc-port", "8084",
            "--ws-port", "8085",
            "--log-level", "info"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait for the node to start
        await asyncio.sleep(3)
        
        yield process
        
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    
    @pytest.mark.asyncio
    async def test_node_starts(self, hauptbuch_binary):
        """Test that the hauptbuch node can start"""
        process = subprocess.Popen([
            hauptbuch_binary,
            "--rpc-port", "8086",
            "--ws-port", "8087",
            "--log-level", "info"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        await asyncio.sleep(2)
        
        # Check if process is still running
        assert process.poll() is None, "Node process died during startup"
        
        # Clean up
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    
    @pytest.mark.asyncio
    async def test_rpc_endpoint_responds(self, hauptbuch_node):
        """Test that the RPC endpoint responds to requests"""
        async with aiohttp.ClientSession() as session:
            # Test network info endpoint
            payload = {
                "jsonrpc": "2.0",
                "method": "hauptbuch_getNetworkInfo",
                "params": {},
                "id": 1
            }
            
            async with session.post(
                "http://localhost:8084",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                assert response.status == 200
                data = await response.json()
                assert "result" in data
                assert "chainId" in data["result"]
                assert "networkId" in data["result"]
    
    @pytest.mark.asyncio
    async def test_node_status_endpoint(self, hauptbuch_node):
        """Test the node status endpoint"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "hauptbuch_getNodeStatus",
                "params": {},
                "id": 2
            }
            
            async with session.post(
                "http://localhost:8084",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                assert response.status == 200
                data = await response.json()
                assert "result" in data
                assert "status" in data["result"]
                assert data["result"]["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_chain_info_endpoint(self, hauptbuch_node):
        """Test the chain info endpoint"""
        async with aiohttp.ClientSession() as session:
            payload = {
                "jsonrpc": "2.0",
                "method": "hauptbuch_getChainInfo",
                "params": {},
                "id": 3
            }
            
            async with session.post(
                "http://localhost:8084",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                assert response.status == 200
                data = await response.json()
                assert "result" in data
                assert "blockHeight" in data["result"]
    
    @pytest.mark.asyncio
    async def test_crypto_endpoints(self, hauptbuch_node):
        """Test crypto-related endpoints"""
        async with aiohttp.ClientSession() as session:
            # Test keypair generation
            payload = {
                "jsonrpc": "2.0",
                "method": "hauptbuch_generateKeypair",
                "params": {},
                "id": 4
            }
            
            async with session.post(
                "http://localhost:8084",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                assert response.status == 200
                data = await response.json()
                assert "result" in data
                assert "privateKey" in data["result"]
                assert "publicKey" in data["result"]
                assert "address" in data["result"]
                assert "algorithm" in data["result"]
                assert data["result"]["algorithm"] == "ml-dsa"
    
    def test_setup_script_exists(self):
        """Test that the setup script exists and is executable"""
        setup_script = Path(__file__).parent / "setup_local_network.sh"
        assert setup_script.exists(), "Setup script not found"
        assert os.access(setup_script, os.X_OK), "Setup script is not executable"
    
    def test_teardown_script_exists(self):
        """Test that the teardown script exists and is executable"""
        teardown_script = Path(__file__).parent / "teardown_network.sh"
        assert teardown_script.exists(), "Teardown script not found"
        assert os.access(teardown_script, os.X_OK), "Teardown script is not executable"
    
    def test_health_check_script_exists(self):
        """Test that the health check script exists and is executable"""
        health_script = Path(__file__).parent / "health_check.sh"
        assert health_script.exists(), "Health check script not found"
        assert os.access(health_script, os.X_OK), "Health check script is not executable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
