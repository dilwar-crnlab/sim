#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Updated XT-NLI-A-RSA Algorithm Implementation with Spectrum Allocation State
Modified to pass current spectrum allocation to GSNR calculator for interfering channel detection
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass
import time

# Import the modified GSNR calculator
from modified_gsnr_steps import IntegratedGSNRCalculator

# Import other required modules
import sys
import os
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
    Enhanced Spectrum allocation matrix for MCF EON with state tracking
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
        
        # Track allocation history for debugging
        self.allocation_history = []
    
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
        """Allocate resource to connection with tracking"""
        if self.is_available(link_id, core_index, channel_index):
            self.allocation[link_id, core_index, channel_index] = connection_id
            
            if connection_id not in self.connection_resources:
                self.connection_resources[connection_id] = []
            self.connection_resources[connection_id].append((link_id, core_index, channel_index))
            
            # Track allocation for debugging
            self.allocation_history.append({
                'action': 'allocate',
                'link_id': link_id,
                'core_index': core_index,
                'channel_index': channel_index,
                'connection_id': connection_id,
                'timestamp': time.time()
            })
    
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
                
                # Track deallocation
                self.allocation_history.append({
                    'action': 'deallocate',
                    'link_id': link_id,
                    'core_index': core_index,
                    'channel_index': channel_index,
                    'connection_id': connection_id,
                    'timestamp': time.time()
                })
            
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
    
    def get_interfering_channels_on_path(self, path_links: List[int], core_index: int, 
                                       target_channel: int) -> List[int]:
        """Get all interfering channels on a specific path and core"""
        interfering_channels = []
        
        for ch_idx in range(self.num_channels):
            if ch_idx == target_channel:
                continue
            
            # Check if this channel is allocated on ALL links in the path
            is_interferer = True
            for link_id in path_links:
                if self.allocation[link_id, core_index, ch_idx] == 0:
                    is_interferer = False
                    break
            
            if is_interferer:
                interfering_channels.append(ch_idx)
        
        return interfering_channels

class UpdatedXT_NLI_A_RSA_Algorithm:
    """
    Updated XT-NLI-A-RSA Algorithm with spectrum allocation state awareness
    Passes current allocation state to GSNR calculator for realistic interference modeling
    """
    
    def __init__(self, network_topology, mcf_config, gsnr_calculator=None):
        """
        Initialize updated XT-NLI-A-RSA algorithm
        
        Args:
            network_topology: NetworkTopology object
            mcf_config: MCF configuration object
            gsnr_calculator: GSNR calculator (will be replaced with IntegratedGSNRCalculator)
        """
        self.network = network_topology
        self.mcf_config = mcf_config
        
        # Replace GSNR calculator with integrated version
        band_config = {
            'c_band': mcf_config.band_configs['C'].__dict__,
            'l_band': mcf_config.band_configs['L'].__dict__
        }
        self.gsnr_calculator = IntegratedGSNRCalculator(mcf_config, band_config)
        
        # Initialize enhanced spectrum allocation matrix
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
            'average_computation_time_ms': 0.0,
            'total_gsnr_calculations': 0,
            'total_gsnr_time_ms': 0.0,
            'average_interfering_channels': 0.0
        }
        
        print(f"Updated XT-NLI-A-RSA Algorithm initialized:")
        print(f"  Network: {len(self.network.links)} links, {len(self.network.nodes)} nodes")
        print(f"  MCF: {num_cores} cores, {num_channels} channels")
        print(f"  Using integrated GSNR calculator with existing split-step methods")
        print(f"  Band channels - L: {len(self.band_channels['L'])}, C: {len(self.band_channels['C'])}")
    
    def calculate_available_channel_vector_with_interference(self, path_links: List[int], 
                                                           sam: SpectrumAllocationMethod) -> List[AvailableChannelResource]:
        """
        Calculate Available Channel Vector (ACV) with realistic interference modeling
        
        Args:
            path_links: List of link IDs in the path
            sam: Spectrum Allocation Method (CSB or BSC)
            
        Returns:
            List of available channel resources ordered by SAM with realistic GSNR
        """
        available_resources = []
        
        if sam == SpectrumAllocationMethod.CSB:
            # Core-Spectrum-Band: prioritize cores over bands
            for core_idx in range(self.num_cores):
                for band_name in ['C', 'L']:
                    if band_name in self.band_channels:
                        for channel_idx in self.band_channels[band_name]:
                            if self.spectrum_allocation.is_path_available(path_links, core_idx, channel_idx):
                                # Calculate GSNR with current interference state
                                gsnr_result = self._calculate_gsnr_with_interference(
                                    path_links, channel_idx, core_idx
                                )
                                
                                if gsnr_result and gsnr_result.max_bitrate_gbps >= 100:  # PM-BPSK minimum
                                    channel_info = self.channels[channel_idx]
                                    available_resources.append(AvailableChannelResource(
                                        core_index=core_idx,
                                        channel_index=channel_idx,
                                        gsnr_result=gsnr_result,
                                        band_name=channel_info['band'],
                                        frequency_hz=channel_info['frequency_hz']
                                    ))
        
        elif sam == SpectrumAllocationMethod.BSC:
            # Band-Spectrum-Core: prioritize bands over cores
            for band_name in ['C', 'L']:  # Process C-band first, then L-band
                if band_name in self.band_channels:
                    for channel_idx in self.band_channels[band_name]:
                        for core_idx in range(self.num_cores):
                            if self.spectrum_allocation.is_path_available(path_links, core_idx, channel_idx):
                                # Calculate GSNR with current interference state
                                gsnr_result = self._calculate_gsnr_with_interference(
                                    path_links, channel_idx, core_idx
                                )
                                
                                if gsnr_result and gsnr_result.max_bitrate_gbps >= 100:
                                    channel_info = self.channels[channel_idx]
                                    available_resources.append(AvailableChannelResource(
                                        core_index=core_idx,
                                        channel_index=channel_idx,
                                        gsnr_result=gsnr_result,
                                        band_name=channel_info['band'],
                                        frequency_hz=channel_info['frequency_hz']
                                    ))
        
        return available_resources
    
    def _calculate_gsnr_with_interference(self, path_links: List[int], channel_idx: int, 
                                        core_idx: int) -> object:
        """
        Calculate GSNR with current interference state using modified existing methods
        
        Args:
            path_links: List of link IDs
            channel_idx: Channel index
            core_idx: Core index
            
        Returns:
            GSNR calculation result or None if calculation fails
        """
        
        gsnr_start_time = time.time()
        
        try:
            # Convert link IDs to link objects
            path_links_objects = [self.network.links[link_id] for link_id in path_links]
            
            # Get current number of interfering channels for statistics
            interfering_channels = self.spectrum_allocation.get_interfering_channels_on_path(
                path_links, core_idx, channel_idx
            )
            
            # Calculate GSNR using integrated calculator with spectrum allocation state
            gsnr_result = self.gsnr_calculator.calculate_gsnr(
                path_links_objects, 
                channel_idx, 
                core_idx,
                launch_power_dbm=0.0,
                use_cache=True,
                spectrum_allocation=self.spectrum_allocation  # Pass current allocation state
            )
            
            # Update statistics
            gsnr_time = (time.time() - gsnr_start_time) * 1000
            self.algorithm_stats['total_gsnr_calculations'] += 1
            self.algorithm_stats['total_gsnr_time_ms'] += gsnr_time
            
            # Update average interfering channels
            current_avg = self.algorithm_stats['average_interfering_channels']
            count = self.algorithm_stats['total_gsnr_calculations']
            self.algorithm_stats['average_interfering_channels'] = (
                (current_avg * (count - 1) + len(interfering_channels)) / count
            )
            
            return gsnr_result
            
        except Exception as e:
            print(f"Error calculating GSNR for channel {channel_idx}, core {core_idx}: {e}")
            return None
    
    def xt_nli_a_rsa_algorithm(self, connection, k_shortest_paths: List[List[int]],
                              sam: SpectrumAllocationMethod = SpectrumAllocationMethod.BSC) -> bool:
        """
        Main XT-NLI-A-RSA Algorithm with realistic interference modeling
        
        Args:
            connection: Connection object with bandwidth requirements
            k_shortest_paths: List of K-shortest paths (node sequences)
            sam: Spectrum Allocation Method
            
        Returns:
            True if connection successfully allocated, False if blocked
        """
        start_time = time.time()
        self.algorithm_stats['total_requests'] += 1
        
        connection_id = int(connection.connection_id) if isinstance(connection.connection_id, str) else connection.connection_id
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
        
        # Stage 1: Try to allocate as single chunk with realistic interference
        for path_links in link_paths:
            # Calculate ACV with current interference state
            acv = self.calculate_available_channel_vector_with_interference(path_links, sam)
            
            if not acv:
                continue  # No available channels on this path
            
            # First-fit: find first channel that meets QoT requirements with current interference
            for resource in acv:
                if resource.gsnr_result.max_bitrate_gbps >= bandwidth_demand:
                    # Allocate path, core, band, and channel to request
                    success = self.spectrum_allocation.allocate_path_resource(
                        path_links, resource.core_index, resource.channel_index, connection_id
                    )
                    
                    if success:
                        # Create resource allocation
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
                        
                        # Calculate node path correctly
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
        
        # Stage 2: Bandwidth slicing with realistic interference
        if not allocated:
            for path_links in link_paths:
                bandwidth_remaining = bandwidth_demand
                allocated_segments = []
                
                while bandwidth_remaining > 0:
                    # Recalculate ACV with updated interference state
                    acv = self.calculate_available_channel_vector_with_interference(path_links, sam)
                    
                    if not acv:  # Release all allocated resources
                        for link_id, core_idx, channel_idx in allocated_segments:
                            for lid in path_links:
                                if self.spectrum_allocation.allocation[lid, core_idx, channel_idx] == connection_id:
                                    self.spectrum_allocation.allocation[lid, core_idx, channel_idx] = 0
                        allocated_segments.clear()
                        break  # Move to next path
                    
                    # Allocate segment with updated interference
                    resource = acv[0]  # First-fit
                    segment_bitrate = min(resource.gsnr_result.max_bitrate_gbps, bandwidth_remaining)
                    
                    # Allocate this segment
                    success = self.spectrum_allocation.allocate_path_resource(
                        path_links, resource.core_index, resource.channel_index, connection_id
                    )
                    
                    if success:
                        allocated_segments.append((path_links[0], resource.core_index, resource.channel_index))
                        bandwidth_remaining -= segment_bitrate
                        
                        if bandwidth_remaining <= 0:  # Successfully allocated
                            # Create resource allocations for connection
                            resource_allocations = []
                            total_bitrate = 0
                            min_gsnr = float('inf')
                            
                            for i, (_, core_idx, channel_idx) in enumerate(allocated_segments):
                                # Recalculate GSNR for this segment with final interference state
                                path_links_objects = [self.network.links[link_id] for link_id in path_links]
                                try:
                                    gsnr_result = self.gsnr_calculator.calculate_gsnr(
                                        path_links_objects, channel_idx, core_idx,
                                        spectrum_allocation=self.spectrum_allocation
                                    )
                                    
                                    mod_format = self._get_modulation_format_enum(
                                        gsnr_result.supported_modulation
                                    )
                                    
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
                            node_path = self._calculate_node_path(path_links)
                            
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
        
        # Request is blocked
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
            connection_id = int(connection_id)
        
        self.spectrum_allocation.deallocate_connection(connection_id)
    
    def get_algorithm_statistics(self) -> Dict:
        """Get enhanced algorithm performance statistics"""
        total_requests = self.algorithm_stats['total_requests']
        
        if total_requests > 0:
            success_rate = self.algorithm_stats['successful_allocations'] / total_requests
            blocking_rate = self.algorithm_stats['blocked_requests'] / total_requests
            single_chunk_rate = self.algorithm_stats['single_chunk_allocations'] / total_requests
            sliced_rate = self.algorithm_stats['sliced_allocations'] / total_requests
        else:
            success_rate = blocking_rate = single_chunk_rate = sliced_rate = 0.0
        
        # Calculate average GSNR computation time
        avg_gsnr_time_ms = 0.0
        if self.algorithm_stats['total_gsnr_calculations'] > 0:
            avg_gsnr_time_ms = (self.algorithm_stats['total_gsnr_time_ms'] / 
                               self.algorithm_stats['total_gsnr_calculations'])
        
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
            'utilization_per_core': self.spectrum_allocation.get_utilization_per_core(),
            'gsnr_performance': {
                'total_gsnr_calculations': self.algorithm_stats['total_gsnr_calculations'],
                'average_gsnr_time_ms': avg_gsnr_time_ms,
                'average_interfering_channels': self.algorithm_stats['average_interfering_channels']
            }
        }
    
    def get_network_state_summary(self) -> Dict:
        """Get current network state summary with interference tracking"""
        return {
            'spectrum_allocation_matrix_shape': self.spectrum_allocation.allocation.shape,
            'total_resources': (self.spectrum_allocation.num_links * 
                              self.spectrum_allocation.num_cores * 
                              self.spectrum_allocation.num_channels),
            'allocated_resources': np.count_nonzero(self.spectrum_allocation.allocation),
            'active_connections': len(self.spectrum_allocation.connection_resources),
            'utilization_per_core': self.spectrum_allocation.get_utilization_per_core(),
            'overall_utilization': self.spectrum_allocation.get_utilization(),
            'allocation_history_size': len(self.spectrum_allocation.allocation_history)
        }

# Example usage
if __name__ == "__main__":
    print("Updated XT-NLI-A-RSA Algorithm with Realistic Interference Modeling")
    print("Uses existing split-step methods with spectrum allocation state awareness")
    print("Provides realistic GSNR calculations considering current network loading")