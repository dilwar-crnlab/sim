#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
XT-NLI-A-RSA Algorithm Implementation
Channel-Based ICXT- and NLI-Aware Service Provisioning for Multi-Band over SDM Systems
Implementation of Algorithm 1 from the first paper
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass
import time


# Fixed imports and ResourceAllocation creation in xt_nli_rsa.py

# At the top of the file, add proper imports
import sys
import os
# Add the parent directory to path for relative imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connection_manager import ResourceAllocation, ModulationFormat

class SpectrumAllocationMethod(Enum):
    """Spectrum Allocation Methods from the paper"""
    CSB = "Core-Spectrum-Band"  # Core-Spectrum-Band: prioritize cores over bands
    BSC = "Band-Spectrum-Core"  # Band-Spectrum-Core: prioritize bands over cores

@dataclass
class AvailableChannelResource:
    """Available channel resource tuple (core, channel, GSNR result)"""
    core_index: int
    channel_index: int
    gsnr_result: object  # GSNRCalculationResult
    band_name: str
    frequency_hz: float

class SpectrumAllocationMatrix:
    """
    Spectrum allocation matrix for MCF EON
    Matrix dimensions: [link_id][core_index][channel_index] = connection_id (0 = free)
    """
    
    def __init__(self, num_links: int, num_cores: int, num_channels: int):
        self.num_links = num_links
        self.num_cores = num_cores
        self.num_channels = num_channels
        
        # Allocation matrix: 0 = free, >0 = connection_id
        self.allocation = np.zeros((num_links, num_cores, num_channels), dtype=int)
        
        # Track allocated resources per connection
        self.connection_resources: Dict[int, List[Tuple[int, int, int]]] = {}
    
    def is_available(self, link_id: int, core_index: int, channel_index: int) -> bool:
        """Check if resource is available"""
        if (0 <= link_id < self.num_links and 
            0 <= core_index < self.num_cores and 
            0 <= channel_index < self.num_channels):
            return self.allocation[link_id, core_index, channel_index] == 0
        return False
    
    def is_path_available(self, path_links: List[int], core_index: int, channel_index: int) -> bool:
        """Check if channel is available on all links in path"""
        for link_id in path_links:
            if not self.is_available(link_id, core_index, channel_index):
                return False
        return True
    
    def allocate_resource(self, link_id: int, core_index: int, channel_index: int, connection_id: int):
        """Allocate resource to connection"""
        if self.is_available(link_id, core_index, channel_index):
            self.allocation[link_id, core_index, channel_index] = connection_id
            
            if connection_id not in self.connection_resources:
                self.connection_resources[connection_id] = []
            self.connection_resources[connection_id].append((link_id, core_index, channel_index))
    
    def allocate_path_resource(self, path_links: List[int], core_index: int, 
                             channel_index: int, connection_id: int) -> bool:
        """Allocate resource on entire path"""
        # Check availability first
        if not self.is_path_available(path_links, core_index, channel_index):
            return False
        
        # Allocate on all links
        for link_id in path_links:
            self.allocate_resource(link_id, core_index, channel_index, connection_id)
        
        return True
    
    def deallocate_connection(self, connection_id: int):
        """Deallocate all resources for a connection"""
        if connection_id in self.connection_resources:
            for link_id, core_index, channel_index in self.connection_resources[connection_id]:
                self.allocation[link_id, core_index, channel_index] = 0
            
            del self.connection_resources[connection_id]
    
    def get_utilization(self) -> float:
        """Get spectrum utilization ratio"""
        total_resources = self.num_links * self.num_cores * self.num_channels
        allocated_resources = np.count_nonzero(self.allocation)
        return allocated_resources / total_resources if total_resources > 0 else 0.0
    
    def get_utilization_per_core(self) -> Dict[int, float]:
        """Get utilization per core"""
        utilization_per_core = {}
        for core_idx in range(self.num_cores):
            core_allocation = self.allocation[:, core_idx, :]
            total_core_resources = self.num_links * self.num_channels
            allocated_core_resources = np.count_nonzero(core_allocation)
            utilization_per_core[core_idx] = allocated_core_resources / total_core_resources
        
        return utilization_per_core

class XT_NLI_A_RSA_Algorithm:
    """
    Implementation of XT-NLI-A-RSA Algorithm (Algorithm 1)
    Channel-Based ICXT- and NLI-Aware Service Provisioning
    """
    
    def __init__(self, network_topology, mcf_config, gsnr_calculator):
        """
        Initialize XT-NLI-A-RSA algorithm
        
        Args:
            network_topology: NetworkTopology object
            mcf_config: MCF configuration object
            gsnr_calculator: GSNRCalculator object
        """
        self.network = network_topology
        self.mcf_config = mcf_config
        self.gsnr_calculator = gsnr_calculator
        
        # Initialize spectrum allocation matrix
        num_links = len(self.network.links)
        num_cores = mcf_config.mcf_params.num_cores
        num_channels = len(mcf_config.channels)
        
        self.spectrum_allocation = SpectrumAllocationMatrix(num_links, num_cores, num_channels)
        
        # Channel information
        self.channels = mcf_config.channels
        self.num_channels = len(self.channels)
        self.num_cores = num_cores
        
        # Create band-to-channel mapping
        self.band_channels = {}
        for band in ['L', 'C']:  # L-band first, then C-band
            self.band_channels[band] = [ch['index'] for ch in self.channels if ch['band'] == band]
        
        # Statistics
        self.algorithm_stats = {
            'total_requests': 0,
            'successful_allocations': 0,
            'blocked_requests': 0,
            'single_chunk_allocations': 0,
            'sliced_allocations': 0,
            'average_computation_time_ms': 0.0
        }
        
        print(f"XT-NLI-A-RSA Algorithm initialized:")
        print(f"  Network: {len(self.network.links)} links, {len(self.network.nodes)} nodes")
        print(f"  MCF: {num_cores} cores, {num_channels} channels")
        print(f"  Band channels - L: {len(self.band_channels['L'])}, C: {len(self.band_channels['C'])}")
    
    def calculate_available_channel_vector(self, path_links: List[int], 
                                         sam: SpectrumAllocationMethod) -> List[AvailableChannelResource]:
        """
        Calculate Available Channel Vector (ACV) using specified SAM
        
        Args:
            path_links: List of link IDs in the path
            sam: Spectrum Allocation Method (CSB or BSC)
            
        Returns:
            List of available channel resources ordered by SAM
        """
        available_resources = []
        
        if sam == SpectrumAllocationMethod.CSB:
            # Core-Spectrum-Band: prioritize cores over bands
            # All available bands within a single core are fully utilized before moving to next core
            
            for core_idx in range(self.num_cores):
                # Process bands in order: C, L, S (only C and L for this implementation)
                for band_name in ['C', 'L']:
                    if band_name in self.band_channels:
                        for channel_idx in self.band_channels[band_name]:
                            if self.spectrum_allocation.is_path_available(path_links, core_idx, channel_idx):
                                # Calculate GSNR for this resource
                                path_links_objects = [self.network.links[link_id] for link_id in path_links]
                                
                                try:
                                    gsnr_result = self.gsnr_calculator.calculate_gsnr(
                                        path_links_objects, channel_idx, core_idx
                                    )
                                    
                                    # Check if channel can support at least PM-BPSK
                                    if gsnr_result.max_bitrate_gbps >= 100:  # PM-BPSK = 100 Gbps
                                        channel_info = self.channels[channel_idx]
                                        available_resources.append(AvailableChannelResource(
                                            core_index=core_idx,
                                            channel_index=channel_idx,
                                            gsnr_result=gsnr_result,
                                            band_name=channel_info['band'],
                                            frequency_hz=channel_info['frequency_hz']
                                        ))
                                
                                except Exception as e:
                                    # Skip this resource if GSNR calculation fails
                                    continue
        
        elif sam == SpectrumAllocationMethod.BSC:
            # Band-Spectrum-Core: prioritize bands over cores
            # Channels within C-band are allocated sequentially across all cores
            
            for band_name in ['C', 'L']:  # Process C-band first, then L-band
                if band_name in self.band_channels:
                    for channel_idx in self.band_channels[band_name]:
                        for core_idx in range(self.num_cores):
                            if self.spectrum_allocation.is_path_available(path_links, core_idx, channel_idx):
                                # Calculate GSNR for this resource
                                path_links_objects = [self.network.links[link_id] for link_id in path_links]
                                
                                try:
                                    gsnr_result = self.gsnr_calculator.calculate_gsnr(
                                        path_links_objects, channel_idx, core_idx
                                    )
                                    
                                    # Check if channel can support at least PM-BPSK
                                    if gsnr_result.max_bitrate_gbps >= 100:
                                        channel_info = self.channels[channel_idx]
                                        available_resources.append(AvailableChannelResource(
                                            core_index=core_idx,
                                            channel_index=channel_idx,
                                            gsnr_result=gsnr_result,
                                            band_name=channel_info['band'],
                                            frequency_hz=channel_info['frequency_hz']
                                        ))
                                
                                except Exception as e:
                                    continue
        
        return available_resources
    
    def xt_nli_a_rsa_algorithm(self, connection, k_shortest_paths: List[List[int]],
                              sam: SpectrumAllocationMethod = SpectrumAllocationMethod.BSC) -> bool:
        """
        Main XT-NLI-A-RSA Algorithm implementation (Algorithm 1)
        
        Args:
            connection: Connection object with bandwidth requirements
            k_shortest_paths: List of K-shortest paths (node sequences)
            sam: Spectrum Allocation Method
            
        Returns:
            True if connection successfully allocated, False if blocked
        """
        start_time = time.time()
        self.algorithm_stats['total_requests'] += 1
        
        connection_id = int(connection.connection_id, 16) if isinstance(connection.connection_id, str) else connection.connection_id
        bandwidth_demand = connection.bandwidth_demand_gbps
        
        # Convert node paths to link paths
        link_paths = []
        for node_path in k_shortest_paths:
            link_path = []
            for i in range(len(node_path) - 1):
                link = self.network.get_link_by_nodes(node_path[i], node_path[i + 1])
                if link:
                    link_path.append(link.link_id)
                else:
                    link_path = None
                    break
            
            if link_path:
                link_paths.append(link_path)
        
        if not link_paths:
            self._record_computation_time(start_time)
            return False
        
        allocated = False
        
        # Stage 1: Try to allocate as single chunk (Lines 2-9 in Algorithm 1)
        for path_links in link_paths:
            # Find the first available channel using first-fit approach and SAM (Line 3)
            acv = self.calculate_available_channel_vector(path_links, sam)
            
            if not acv:
                continue  # No available channels on this path
            
            # First-fit: find first channel that meets QoT requirements (Lines 4-7)
            for resource in acv:
                if resource.gsnr_result.max_bitrate_gbps >= bandwidth_demand:
                    # Allocate path, core, band, and channel to request
                    success = self.spectrum_allocation.allocate_path_resource(
                        path_links, resource.core_index, resource.channel_index, connection_id
                    )
                    
                    if success:
                        #  Fixed ResourceAllocation creation
                        mod_format = self._get_modulation_format_enum(
                            resource.gsnr_result.supported_modulation
                        )
                        
                        resource_allocation = ResourceAllocation(
                            link_id=path_links[0],
                            core_index=resource.core_index,
                            channel_index=resource.channel_index,
                            modulation_format=mod_format,
                            allocated_bitrate_gbps=resource.gsnr_result.max_bitrate_gbps,
                            gsnr_db=resource.gsnr_result.gsnr_db
                        )
                        
                        #  Calculate node path correctly
                        node_path = self._calculate_node_path(path_links)
                        
                        # Update connection
                        connection.allocated_path = node_path
                        connection.resource_allocations = [resource_allocation]
                        connection.total_allocated_bitrate_gbps = resource.gsnr_result.max_bitrate_gbps
                        connection.end_to_end_gsnr_db = resource.gsnr_result.gsnr_db
                        connection.path_length_km = sum(self.network.links[lid].length_km for lid in path_links)
                        
                        allocated = True
                        self.algorithm_stats['successful_allocations'] += 1
                        self.algorithm_stats['single_chunk_allocations'] += 1
                        
                        self._record_computation_time(start_time)
                        return True
        
        # Stage 2: Bandwidth slicing (Lines 10-29 in Algorithm 1)
        if not allocated:
            for path_links in link_paths:
                bandwidth_remaining = bandwidth_demand
                allocated_segments = []
                
                while bandwidth_remaining > 0:  # Line 13
                    # Find the first available channel in ACV using SAM (Line 14)
                    acv = self.calculate_available_channel_vector(path_links, sam)
                    
                    if not acv:  # Line 23-26: Release all allocated resources
                        for link_id, core_idx, channel_idx in allocated_segments:
                            # Release resources
                            for lid in path_links:
                                if self.spectrum_allocation.allocation[lid, core_idx, channel_idx] == connection_id:
                                    self.spectrum_allocation.allocation[lid, core_idx, channel_idx] = 0
                        allocated_segments.clear()
                        break  # Move to next path
                    
                    # Line 15-22: Allocate segment
                    resource = acv[0]  # First-fit
                    segment_bitrate = min(resource.gsnr_result.max_bitrate_gbps, bandwidth_remaining)
                    
                    # Allocate this segment
                    success = self.spectrum_allocation.allocate_path_resource(
                        path_links, resource.core_index, resource.channel_index, connection_id
                    )
                    
                    if success:
                        allocated_segments.append((path_links[0], resource.core_index, resource.channel_index))
                        bandwidth_remaining -= segment_bitrate
                        
                        if bandwidth_remaining <= 0:  # Line 19-21: Successfully allocated
                            # Create resource allocations for connection
                            resource_allocations = []
                            total_bitrate = 0
                            min_gsnr = float('inf')
                            
                            for i, (_, core_idx, channel_idx) in enumerate(allocated_segments):
                                # Recalculate GSNR for this segment
                                path_links_objects = [self.network.links[link_id] for link_id in path_links]
                                try:
                                    gsnr_result = self.gsnr_calculator.calculate_gsnr(
                                        path_links_objects, channel_idx, core_idx
                                    )
                                    
                                    # Find modulation format
                                    mod_format = None
                                    for mf in ModulationFormat:
                                        if mf.format_name == gsnr_result.supported_modulation:
                                            mod_format = mf
                                            break
                                    if mod_format is None:
                                        mod_format = ModulationFormat.PM_BPSK
                                    
                                    resource_allocation = ResourceAllocation(
                                        link_id=path_links[0],
                                        core_index=core_idx,
                                        channel_index=channel_idx,
                                        modulation_format=mod_format,
                                        allocated_bitrate_gbps=min(gsnr_result.max_bitrate_gbps, bandwidth_demand - total_bitrate),
                                        gsnr_db=gsnr_result.gsnr_db
                                    )
                                    
                                    resource_allocations.append(resource_allocation)
                                    total_bitrate += resource_allocation.allocated_bitrate_gbps
                                    min_gsnr = min(min_gsnr, gsnr_result.gsnr_db)
                                    
                                except Exception:
                                    continue
                            
                            # Calculate path through nodes
                            node_path = []
                            for i, link_id in enumerate(path_links):
                                link = self.network.links[link_id]
                                if i == 0:
                                    node_path.append(link.source_node)
                                node_path.append(link.destination_node)
                            
                            # Update connection
                            connection.allocated_path = node_path
                            connection.resource_allocations = resource_allocations
                            connection.total_allocated_bitrate_gbps = total_bitrate
                            connection.end_to_end_gsnr_db = min_gsnr
                            connection.path_length_km = sum(self.network.links[lid].length_km for lid in path_links)
                            
                            allocated = True
                            self.algorithm_stats['successful_allocations'] += 1
                            self.algorithm_stats['sliced_allocations'] += 1
                            
                            self._record_computation_time(start_time)
                            return True
                    else:
                        # Allocation failed, release previous segments and try next path
                        for link_id, core_idx, channel_idx in allocated_segments:
                            for lid in path_links:
                                if self.spectrum_allocation.allocation[lid, core_idx, channel_idx] == connection_id:
                                    self.spectrum_allocation.allocation[lid, core_idx, channel_idx] = 0
                        allocated_segments.clear()
                        break
        
        # Lines 30-32: Request is blocked
        if not allocated:
            self.algorithm_stats['blocked_requests'] += 1
        
        self._record_computation_time(start_time)
        return allocated
    
    def _get_modulation_format_enum(self, modulation_format_name: str) -> ModulationFormat:
        """Get ModulationFormat enum from string name"""
        format_mapping = {
            'PM-BPSK': ModulationFormat.PM_BPSK,
            'PM-QPSK': ModulationFormat.PM_QPSK, 
            'PM-8QAM': ModulationFormat.PM_8QAM,
            'PM-16QAM': ModulationFormat.PM_16QAM,
            'PM-32QAM': ModulationFormat.PM_32QAM,
            'PM-64QAM': ModulationFormat.PM_64QAM,
            'None': ModulationFormat.PM_BPSK  # Fallback
        }
        
        return format_mapping.get(modulation_format_name, ModulationFormat.PM_BPSK)
    
    def _calculate_node_path(self, path_links: List[int]) -> List[int]:
        """Calculate node path from link path"""
        if not path_links:
            return []
        
        node_path = []
        for i, link_id in enumerate(path_links):
            link = self.network.links[link_id]
            if i == 0:
                node_path.append(link.source_node)
            node_path.append(link.destination_node)
        
        return node_path
    
    def _record_computation_time(self, start_time: float):
        """Record computation time for statistics"""
        computation_time_ms = (time.time() - start_time) * 1000
        
        # Update running average
        total_requests = self.algorithm_stats['total_requests']
        if total_requests == 1:
            self.algorithm_stats['average_computation_time_ms'] = computation_time_ms
        else:
            current_avg = self.algorithm_stats['average_computation_time_ms']
            self.algorithm_stats['average_computation_time_ms'] = (
                (current_avg * (total_requests - 1) + computation_time_ms) / total_requests
            )
    
    def deallocate_connection(self, connection_id: int):
        """Deallocate resources for a connection"""
        # Convert string connection_id to int if needed
        if isinstance(connection_id, str):
            connection_id = int(connection_id, 16)
        
        self.spectrum_allocation.deallocate_connection(connection_id)
    
    def get_algorithm_statistics(self) -> Dict:
        """Get algorithm performance statistics"""
        total_requests = self.algorithm_stats['total_requests']
        
        if total_requests > 0:
            success_rate = self.algorithm_stats['successful_allocations'] / total_requests
            blocking_rate = self.algorithm_stats['blocked_requests'] / total_requests
            single_chunk_rate = self.algorithm_stats['single_chunk_allocations'] / total_requests
            sliced_rate = self.algorithm_stats['sliced_allocations'] / total_requests
        else:
            success_rate = blocking_rate = single_chunk_rate = sliced_rate = 0.0
        
        return {
            'total_requests': total_requests,
            'successful_allocations': self.algorithm_stats['successful_allocations'],
            'blocked_requests': self.algorithm_stats['blocked_requests'],
            'success_rate': success_rate,
            'blocking_rate': blocking_rate,
            'single_chunk_allocations': self.algorithm_stats['single_chunk_allocations'],
            'sliced_allocations': self.algorithm_stats['sliced_allocations'],
            'single_chunk_rate': single_chunk_rate,
            'sliced_allocation_rate': sliced_rate,
            'average_computation_time_ms': self.algorithm_stats['average_computation_time_ms'],
            'spectrum_utilization': self.spectrum_allocation.get_utilization(),
            'utilization_per_core': self.spectrum_allocation.get_utilization_per_core()
        }
    
    def reset_statistics(self):
        """Reset algorithm statistics"""
        self.algorithm_stats = {
            'total_requests': 0,
            'successful_allocations': 0,
            'blocked_requests': 0,
            'single_chunk_allocations': 0,
            'sliced_allocations': 0,
            'average_computation_time_ms': 0.0
        }
    
    def get_network_state_summary(self) -> Dict:
        """Get current network state summary"""
        return {
            'spectrum_allocation_matrix_shape': self.spectrum_allocation.allocation.shape,
            'total_resources': (self.spectrum_allocation.num_links * 
                              self.spectrum_allocation.num_cores * 
                              self.spectrum_allocation.num_channels),
            'allocated_resources': np.count_nonzero(self.spectrum_allocation.allocation),
            'active_connections': len(self.spectrum_allocation.connection_resources),
            'utilization_per_core': self.spectrum_allocation.get_utilization_per_core(),
            'overall_utilization': self.spectrum_allocation.get_utilization()
        }

# Example usage
if __name__ == "__main__":
    print("XT-NLI-A-RSA Algorithm")
    print("Requires network topology, MCF configuration, and GSNR calculator")
    print("Use within the main MCF EON simulator framework")