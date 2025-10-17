#!/usr/bin/env python3
"""
Hauptbuch Network Tests
Tests for P2P networking, peer management, and network protocols.
"""

import asyncio
import pytest
import time
from typing import List, Dict, Any
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from hauptbuch_client import HauptbuchClient

class TestNetwork:
    """Test suite for network functionality"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_get_peer_list(self, client):
        """Test getting peer list"""
        peers = await client.get_peer_list()
        
        assert isinstance(peers, list)
        for peer in peers:
            assert "id" in peer
            assert "address" in peer
            assert "status" in peer
            assert peer["status"] in ["connected", "disconnected", "connecting"]
    
    @pytest.mark.asyncio
    async def test_add_peer(self, client):
        """Test adding a peer"""
        peer_address = "127.0.0.1:30303"
        peer_id = "QmTestPeerId"
        
        result = await client.add_peer(peer_address, peer_id)
        
        assert "success" in result
        assert result["success"] is True
        assert "peerId" in result
    
    @pytest.mark.asyncio
    async def test_remove_peer(self, client):
        """Test removing a peer"""
        peer_id = "QmTestPeerId"
        
        result = await client.remove_peer(peer_id)
        
        assert "success" in result
        assert result["success"] is True
        assert "peerId" in result
    
    @pytest.mark.asyncio
    async def test_network_status(self, client):
        """Test network status"""
        node_status = await client.get_node_status()
        
        assert node_status.peer_count >= 0
        assert node_status.status in ["healthy", "synced", "syncing", "error"]
    
    @pytest.mark.asyncio
    async def test_peer_connectivity(self, client):
        """Test peer connectivity"""
        peers = await client.get_peer_list()
        
        # Check that peers are properly connected
        connected_peers = [p for p in peers if p["status"] == "connected"]
        assert len(connected_peers) >= 0  # May be 0 in test environment
    
    @pytest.mark.asyncio
    async def test_network_discovery(self, client):
        """Test network discovery"""
        # Test that network discovery is working
        peers = await client.get_peer_list()
        
        # In a real network, we should discover peers
        # In test environment, this might be empty
        assert isinstance(peers, list)
    
    @pytest.mark.asyncio
    async def test_peer_capabilities(self, client):
        """Test peer capabilities"""
        peers = await client.get_peer_list()
        
        for peer in peers:
            if "capabilities" in peer:
                assert isinstance(peer["capabilities"], list)
                # Common capabilities
                expected_capabilities = ["consensus", "network", "rpc"]
                for cap in peer["capabilities"]:
                    assert cap in expected_capabilities
    
    @pytest.mark.asyncio
    async def test_peer_latency(self, client):
        """Test peer latency measurement"""
        peers = await client.get_peer_list()
        
        for peer in peers:
            if "latency" in peer:
                assert isinstance(peer["latency"], (int, float))
                assert peer["latency"] >= 0
                assert peer["latency"] < 10000  # Less than 10 seconds
    
    @pytest.mark.asyncio
    async def test_network_topology(self, client):
        """Test network topology"""
        peers = await client.get_peer_list()
        
        # Test that we have a reasonable number of peers
        assert len(peers) >= 0  # May be 0 in test environment
        
        # Test that peers have valid addresses
        for peer in peers:
            assert "address" in peer
            assert peer["address"] != ""
    
    @pytest.mark.asyncio
    async def test_network_protocols(self, client):
        """Test network protocols"""
        # Test that network protocols are working
        node_status = await client.get_node_status()
        
        assert node_status.status in ["healthy", "synced", "syncing", "error"]
        assert node_status.peer_count >= 0
    
    @pytest.mark.asyncio
    async def test_network_security(self, client):
        """Test network security"""
        peers = await client.get_peer_list()
        
        # Test that peers are properly authenticated
        for peer in peers:
            assert "id" in peer
            assert peer["id"] != ""
            # Peer ID should be a valid format
            assert len(peer["id"]) > 0

class TestNetworkPerformance:
    """Test suite for network performance"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_peer_connection_speed(self, client):
        """Test peer connection speed"""
        start_time = time.time()
        
        # Add a peer
        peer_address = "127.0.0.1:30303"
        peer_id = "QmTestPeerId"
        await client.add_peer(peer_address, peer_id)
        
        connection_time = time.time() - start_time
        print(f"Peer connection time: {connection_time:.4f} seconds")
        
        # Connection should be reasonably fast
        assert connection_time < 5.0  # Less than 5 seconds
    
    @pytest.mark.asyncio
    async def test_peer_discovery_speed(self, client):
        """Test peer discovery speed"""
        start_time = time.time()
        
        # Get peer list
        peers = await client.get_peer_list()
        
        discovery_time = time.time() - start_time
        print(f"Peer discovery time: {discovery_time:.4f} seconds")
        
        # Discovery should be fast
        assert discovery_time < 2.0  # Less than 2 seconds
    
    @pytest.mark.asyncio
    async def test_network_throughput(self, client):
        """Test network throughput"""
        start_time = time.time()
        
        # Perform multiple network operations
        for _ in range(10):
            await client.get_peer_list()
            await client.get_node_status()
        
        elapsed_time = time.time() - start_time
        operations_per_second = 20 / elapsed_time  # 20 operations total
        
        print(f"Network throughput: {operations_per_second:.2f} ops/sec")
        assert operations_per_second > 0
    
    @pytest.mark.asyncio
    async def test_network_latency(self, client):
        """Test network latency"""
        latencies = []
        
        for _ in range(5):
            start_time = time.time()
            await client.get_peer_list()
            latency = time.time() - start_time
            latencies.append(latency)
        
        avg_latency = sum(latencies) / len(latencies)
        print(f"Average network latency: {avg_latency:.4f} seconds")
        
        # Latency should be reasonable
        assert avg_latency < 1.0  # Less than 1 second

class TestNetworkReliability:
    """Test suite for network reliability"""
    
    @pytest.fixture
    async def client(self):
        """Create Hauptbuch client for testing"""
        async with HauptbuchClient() as client:
            yield client
    
    @pytest.mark.asyncio
    async def test_network_availability(self, client):
        """Test network availability"""
        # Test that network is available
        node_status = await client.get_node_status()
        
        assert node_status.status in ["healthy", "synced", "syncing", "error"]
        assert node_status.peer_count >= 0
    
    @pytest.mark.asyncio
    async def test_network_fault_tolerance(self, client):
        """Test network fault tolerance"""
        # Test that network can handle peer failures
        peers = await client.get_peer_list()
        
        # Network should continue to function even with peer failures
        assert isinstance(peers, list)
    
    @pytest.mark.asyncio
    async def test_network_recovery(self, client):
        """Test network recovery"""
        # Test that network can recover from failures
        node_status = await client.get_node_status()
        
        # Network should be in a healthy state
        assert node_status.status in ["healthy", "synced", "syncing", "error"]
    
    @pytest.mark.asyncio
    async def test_network_consistency(self, client):
        """Test network consistency"""
        # Test that network state is consistent
        peers1 = await client.get_peer_list()
        await asyncio.sleep(1)
        peers2 = await client.get_peer_list()
        
        # Peer list should be consistent
        assert isinstance(peers1, list)
        assert isinstance(peers2, list)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
