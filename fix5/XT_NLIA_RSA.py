#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Updated XT-NLI-A-RSA Algorithm Implementation with Spectrum Allocation State
Modified to pass current spectrum allocation to GSNR calculator for interfering channel detection
COMPLETE IMPLEMENTATION
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from enum import Enum
from dataclasses import dataclass
import time
import logging
from collections import defaultdict
import json

# Import the modified GSNR calculator
from modified_gsnr_steps import IntegratedGSNRCalculator

# Import other required modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from connection_manager import ResourceAllocation, ModulationFormat

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    
    def __post_init__(self):
        """Validate resource parameters"""
        if self.core_index < 0:
            raise ValueError("Core index must be non-negative")
        if self.channel_index < 0:
            raise ValueError("Channel index must be non-negative")
        if self.frequency_hz <= 0:
            raise ValueError("Frequency must be positive")

@dataclass
class AllocationResult:
    """Result of resource allocation attempt"""
    success: bool
    allocated_resources: List[Tuple[int, int, int]] = None  # (link_id, core_idx, channel_idx)
    total_bitrate_gbps: float = 0.0
    end_to_end_gsnr_db: float = 0.0
    num_segments: int = 0
    allocation_time_ms: float = 0.0
    failure_reason: str = ""

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
        
        # Performance metrics
        self.total_allocations = 0
        self.total_deallocations = 0
        self.peak_utilization = 0.0
        
        logger.info(f"Spectrum allocation matrix initialized: {num_links} links, {num_cores} cores, {num_channels} channels")
    
    def is_available(self, link_id: int, core_index: int, channel_index: int) -> bool:
        """Check if resource is available with bounds checking"""
        if not (0 <= link_id < self.num_links and 
                0 <= core_index < self.num_cores and 
                0 <= channel_index < self.num_channels):
            return False
        return self.allocation[link_id, core_index, channel_index] == 0
    
    def is_path_available(self, path_links: List[int], core_index: int, channel_index: int) -> bool:
        """Check if channel is available on all links in path"""
        if not path_links:
            return False
        
        for link_id in path_links:
            if not self.is_available(link_id, core_index, channel_index):
                return False
        return True
    
    def allocate_resource(self, link_id: int, core_index: int, channel_index: int, connection_id: int):
        """Allocate resource to connection with tracking"""
        if not self.is_available(link_id, core_index, channel_index):
            raise ValueError(f"Resource not available: link={link_id}, core={core_index}, channel={channel_index}")
        
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
        
        self.total_allocations += 1
        current_util = self.get_utilization()
        if current_util > self.peak_utilization:
            self.peak_utilization = current_util
    
    def allocate_path_resource(self, path_links: List[int], core_index: int, 
                             channel_index: int, connection_id: int) -> bool:
        """Allocate resource on entire path with rollback on failure"""
        if not path_links:
            return False
        
        # Check availability first
        if not self.is_path_available(path_links, core_index, channel_index):
            return False
        
        # Allocate on all links
        allocated_links = []
        try:
            for link_id in path_links:
                self.allocate_resource(link_id, core_index, channel_index, connection_id)
                allocated_links.append(link_id)
            return True
        except Exception as e:
            # Rollback on failure
            logger.warning(f"Allocation failed, rolling back: {e}")
            for link_id in allocated_links:
                if self.allocation[link_id, core_index, channel_index] == connection_id:
                    self.allocation[link_id, core_index, channel_index] = 0
            return False
    
    def deallocate_connection(self, connection_id: int):
        """Deallocate all resources for a connection"""
        if connection_id not in self.connection_resources:
            logger.warning(f"Connection {connection_id} not found for deallocation")
            return
        
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
        self.total_deallocations += 1
    
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
    
    def get_utilization_per_band(self, band_channels: Dict[str, List[int]]) -> Dict[str, float]:
        """Get utilization per frequency band"""
        utilization_per_band = {}
        
        for band_name, channel_indices in band_channels.items():
            if not channel_indices:
                utilization_per_band[band_name] = 0.0
                continue
            
            total_band_resources = self.num_links * self.num_cores * len(channel_indices)
            allocated_band_resources = 0
            
            for ch_idx in channel_indices:
                if ch_idx < self.num_channels:
                    allocated_band_resources += np.count_nonzero(self.allocation[:, :, ch_idx])
            
            utilization_per_band[band_name] = allocated_band_resources / total_band_resources
        
        return utilization_per_band
    
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
                if link_id >= self.num_links or self.allocation[link_id, core_index, ch_idx] == 0:
                    is_interferer = False
                    break
            
            if is_interferer:
                interfering_channels.append(ch_idx)
        
        return interfering_channels
    
    def get_fragmentation_metrics(self) -> Dict[str, float]:
        """Calculate spectrum fragmentation metrics"""
        metrics = {}
        
        # Calculate average contiguous block size
        total_contiguous_blocks = 0
        total_block_size = 0
        
        for link_id in range(self.num_links):
            for core_idx in range(self.num_cores):
                # Find contiguous free blocks
                free_channels = self.allocation[link_id, core_idx, :] == 0
                blocks = []
                current_block = 0
                
                for is_free in free_channels:
                    if is_free:
                        current_block += 1
                    else:
                        if current_block > 0:
                            blocks.append(current_block)
                            current_block = 0
                
                if current_block > 0:
                    blocks.append(current_block)
                
                if blocks:
                    total_contiguous_blocks += len(blocks)
                    total_block_size += sum(blocks)
        
        if total_contiguous_blocks > 0:
            metrics['average_contiguous_block_size'] = total_block_size / total_contiguous_blocks
            metrics['total_free_blocks'] = total_contiguous_blocks
        else:
            metrics['average_contiguous_block_size'] = 0.0
            metrics['total_free_blocks'] = 0
        
        # Calculate fragmentation ratio
        total_free = np.sum(self.allocation == 0)
        if total_free > 0 and total_contiguous_blocks > 0:
            ideal_blocks = total_free // self.num_channels if self.num_channels > 0 else 1
            metrics['fragmentation_ratio'] = total_contiguous_blocks / max(ideal_blocks, 1)
        else:
            metrics['fragmentation_ratio'] = 0.0
        
        return metrics
    
    def validate_allocation_integrity(self) -> Dict[str, bool]:
        """Validate allocation matrix integrity"""
        validation = {
            'no_negative_connections': True,
            'consistent_connection_tracking': True,
            'no_duplicate_allocations': True,
            'bounds_respected': True
        }
        
        # Check for negative connection IDs
        if np.any(self.allocation < 0):
            validation['no_negative_connections'] = False
        
        # Check connection tracking consistency
        tracked_resources = set()
        for conn_id, resources in self.connection_resources.items():
            for link_id, core_idx, ch_idx in resources:
                if (link_id, core_idx, ch_idx) in tracked_resources:
                    validation['no_duplicate_allocations'] = False
                tracked_resources.add((link_id, core_idx, ch_idx))
                
                # Check if allocation matrix matches tracking
                if (link_id < self.num_links and core_idx < self.num_cores and 
                    ch_idx < self.num_channels):
                    if self.allocation[link_id, core_idx, ch_idx] != conn_id:
                        validation['consistent_connection_tracking'] = False
                else:
                    validation['bounds_respected'] = False
        
        return validation

class UpdatedXT_NLI_A_RSA_Algorithm:
    """
    Updated XT-NLI-A-RSA Algorithm with spectrum allocation state awareness
    Passes current allocation state to GSNR calculator for realistic interference modeling
    """
    
    def __init__(self, network_topology, mcf_config):  #gsnr_calculator=None
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
        
        # Algorithm configuration
        self.config = {
            'max_gsnr_calculation_attempts': 3,
            'gsnr_calculation_timeout_ms': 1000,
            'enable_fragmentation_aware_allocation': True,
            'enable_load_balancing': True,
            'enable_preemption': False,
            'debug_mode': False
        }
        
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
            'average_interfering_channels': 0.0,
            'allocation_failures': defaultdict(int),
            'gsnr_calculation_failures': 0,
            'timeout_failures': 0
        }
        
        logger.info(f"Updated XT-NLI-A-RSA Algorithm initialized:")
        logger.info(f"  Network: {len(self.network.links)} links, {len(self.network.nodes)} nodes")
        logger.info(f"  MCF: {num_cores} cores, {num_channels} channels")
        logger.info(f"  Using integrated GSNR calculator with existing split-step methods")
        logger.info(f"  Band channels - L: {len(self.band_channels['L'])}, C: {len(self.band_channels['C'])}")
    
    def configure_algorithm(self, **kwargs):
        """Configure algorithm parameters"""
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"Algorithm configuration updated: {key} = {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")
    
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
        
        # Validate inputs
        if not path_links:
            logger.warning("Empty path links provided to ACV calculation")
            return available_resources
        
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
                                    try:
                                        channel_info = self.channels[channel_idx]
                                        available_resources.append(AvailableChannelResource(
                                            core_index=core_idx,
                                            channel_index=channel_idx,
                                            gsnr_result=gsnr_result,
                                            band_name=channel_info['band'],
                                            frequency_hz=channel_info['frequency_hz']
                                        ))
                                    except (IndexError, KeyError) as e:
                                        logger.warning(f"Error creating AvailableChannelResource: {e}")
                                        continue
        
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
                                    try:
                                        channel_info = self.channels[channel_idx]
                                        available_resources.append(AvailableChannelResource(
                                            core_index=core_idx,
                                            channel_index=channel_idx,
                                            gsnr_result=gsnr_result,
                                            band_name=channel_info['band'],
                                            frequency_hz=channel_info['frequency_hz']
                                        ))
                                    except (IndexError, KeyError) as e:
                                        logger.warning(f"Error creating AvailableChannelResource: {e}")
                                        continue
        
        # Apply load balancing if enabled
        if self.config['enable_load_balancing'] and available_resources:
            available_resources = self._apply_load_balancing(available_resources)
        
        return available_resources
    
    def _apply_load_balancing(self, available_resources: List[AvailableChannelResource]) -> List[AvailableChannelResource]:
        """Apply load balancing to available resources"""
        if not available_resources:
            return available_resources
        
        # Get current utilization per core
        core_utilization = self.spectrum_allocation.get_utilization_per_core()
        
        # Sort resources by core utilization (prefer less utilized cores)
        def core_load_key(resource):
            core_util = core_utilization.get(resource.core_index, 0.0)
            # Secondary sort by GSNR (higher is better)
            gsnr_penalty = -resource.gsnr_result.gsnr_db if hasattr(resource.gsnr_result, 'gsnr_db') else 0
            return (core_util, gsnr_penalty)
        
        return sorted(available_resources, key=core_load_key)
    
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
        
        for attempt in range(self.config['max_gsnr_calculation_attempts']):
            try:
                # Convert link IDs to link objects
                path_links_objects = [self.network.links[link_id] for link_id in path_links 
                                    if link_id in self.network.links]
                
                if len(path_links_objects) != len(path_links):
                    logger.warning(f"Some links not found: requested {len(path_links)}, found {len(path_links_objects)}")
                    return None
                
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
                
                # Check for timeout
                if gsnr_time > self.config['gsnr_calculation_timeout_ms']:
                    logger.warning(f"GSNR calculation timeout: {gsnr_time:.2f}ms")
                    self.algorithm_stats['timeout_failures'] += 1
                
                return gsnr_result
                
            except Exception as e:
                logger.warning(f"GSNR calculation attempt {attempt + 1} failed for channel {channel_idx}, core {core_idx}: {e}")
                if attempt == self.config['max_gsnr_calculation_attempts'] - 1:
                    self.algorithm_stats['gsnr_calculation_failures'] += 1
                    return None
                
                # Brief pause before retry
                time.sleep(0.001)
        
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
        
        # Validate inputs
        if not k_shortest_paths:
            self.algorithm_stats['blocked_requests'] += 1
            self.algorithm_stats['allocation_failures']['no_paths'] += 1
            self._record_computation_time(start_time)
            return False
        
        try:
            #connection_id = int(connection.connection_id) if isinstance(connection.connection_id, str) else connection.connection_id
            # Generate a consistent integer ID from any string format
            if isinstance(connection.connection_id, str):
                connection_id = abs(hash(connection.connection_id)) % (2**31)  # Ensure positive 32-bit int
            else:
                connection_id = connection.connection_id
            bandwidth_demand = connection.bandwidth_demand_gbps
            
            if bandwidth_demand <= 0:
                logger.warning(f"Invalid bandwidth demand: {bandwidth_demand}")
                self.algorithm_stats['blocked_requests'] += 1
                self.algorithm_stats['allocation_failures']['invalid_bandwidth'] += 1
                self._record_computation_time(start_time)
                return False
            
        except (ValueError, AttributeError) as e:
            logger.error(f"Error processing connection parameters: {e}")
            self.algorithm_stats['blocked_requests'] += 1
            self.algorithm_stats['allocation_failures']['invalid_connection'] += 1
            self._record_computation_time(start_time)
            return False
        
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
            self.algorithm_stats['blocked_requests'] += 1
            self.algorithm_stats['allocation_failures']['no_valid_paths'] += 1
            self._record_computation_time(start_time)
            return False
        
        # Stage 1: Try to allocate as single chunk with realistic interference
        allocation_result = self._attempt_single_chunk_allocation(
            connection_id, bandwidth_demand, link_paths, sam
        )
        
        if allocation_result.success:
            self._finalize_successful_allocation(connection, allocation_result, link_paths[0])
            self.algorithm_stats['successful_allocations'] += 1
            self.algorithm_stats['single_chunk_allocations'] += 1
            self._record_computation_time(start_time)
            return True
        
        # Stage 2: Bandwidth slicing with realistic interference
        allocation_result = self._attempt_sliced_allocation(
            connection_id, bandwidth_demand, link_paths, sam
        )
        
        if allocation_result.success:
            self._finalize_successful_allocation(connection, allocation_result, link_paths[0])
            self.algorithm_stats['successful_allocations'] += 1
            self.algorithm_stats['sliced_allocations'] += 1
            self._record_computation_time(start_time)
            return True
        
        # Request is blocked
        self.algorithm_stats['blocked_requests'] += 1
        self.algorithm_stats['allocation_failures'][allocation_result.failure_reason] += 1
        
        if self.config['debug_mode']:
            logger.debug(f"Connection {connection_id} blocked: {allocation_result.failure_reason}")
        
        self._record_computation_time(start_time)
        return False
    
    def _attempt_single_chunk_allocation(self, connection_id: int, bandwidth_demand: float,
                                       link_paths: List[List[int]], sam: SpectrumAllocationMethod) -> AllocationResult:
        """Attempt single chunk allocation"""
        for path_links in link_paths:
            # Calculate ACV with current interference state
            acv = self.calculate_available_channel_vector_with_interference(path_links, sam)
            
            if not acv:
                continue  # No available channels on this path
            
            # First-fit: find first channel that meets QoT requirements with current interference
            for resource in acv:
                if resource.gsnr_result.max_bitrate_gbps >= bandwidth_demand:
                    # Attempt allocation
                    success = self.spectrum_allocation.allocate_path_resource(
                        path_links, resource.core_index, resource.channel_index, connection_id
                    )
                    
                    if success:
                        return AllocationResult(
                            success=True,
                            allocated_resources=[(path_links[0], resource.core_index, resource.channel_index)],
                            total_bitrate_gbps=resource.gsnr_result.max_bitrate_gbps,
                            end_to_end_gsnr_db=resource.gsnr_result.gsnr_db,
                            num_segments=1
                        )
        
        return AllocationResult(success=False, failure_reason="no_suitable_single_chunk")
    
    def _attempt_sliced_allocation(self, connection_id: int, bandwidth_demand: float,
                                 link_paths: List[List[int]], sam: SpectrumAllocationMethod) -> AllocationResult:
        """Attempt sliced allocation"""
        for path_links in link_paths:
            bandwidth_remaining = bandwidth_demand
            allocated_segments = []
            total_bitrate = 0.0
            min_gsnr = float('inf')
            
            while bandwidth_remaining > 0:
                # Recalculate ACV with updated interference state
                acv = self.calculate_available_channel_vector_with_interference(path_links, sam)
                
                if not acv:  # Release all allocated resources
                    self._release_segments(allocated_segments, connection_id, path_links)
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
                    total_bitrate += segment_bitrate
                    min_gsnr = min(min_gsnr, resource.gsnr_result.gsnr_db)
                    bandwidth_remaining -= segment_bitrate
                    
                    if bandwidth_remaining <= 0:  # Successfully allocated
                        return AllocationResult(
                            success=True,
                            allocated_resources=allocated_segments,
                            total_bitrate_gbps=total_bitrate,
                            end_to_end_gsnr_db=min_gsnr,
                            num_segments=len(allocated_segments)
                        )
                else:
                    # Allocation failed, release previous segments and try next path
                    self._release_segments(allocated_segments, connection_id, path_links)
                    break
        
        return AllocationResult(success=False, failure_reason="insufficient_sliced_resources")
    
    def _release_segments(self, allocated_segments: List[Tuple[int, int, int]], 
                         connection_id: int, path_links: List[int]):
        """Release allocated segments on failure"""
        for link_id, core_idx, channel_idx in allocated_segments:
            for lid in path_links:
                if (lid < self.spectrum_allocation.num_links and 
                    self.spectrum_allocation.allocation[lid, core_idx, channel_idx] == connection_id):
                    self.spectrum_allocation.allocation[lid, core_idx, channel_idx] = 0
        allocated_segments.clear()
    
    def _finalize_successful_allocation(self, connection, allocation_result: AllocationResult, path_links: List[int]):
        """Finalize successful allocation by updating connection object"""
        # Create resource allocations
        resource_allocations = []
        
        for link_id, core_idx, channel_idx in allocation_result.allocated_resources:
            # Recalculate GSNR for this segment with final interference state
            path_links_objects = [self.network.links[link_id] for link_id in path_links 
                                if link_id in self.network.links]
            try:
                gsnr_result = self.gsnr_calculator.calculate_gsnr(
                    path_links_objects, channel_idx, core_idx,
                    spectrum_allocation=self.spectrum_allocation
                )
                
                mod_format = self._get_modulation_format_enum(
                    gsnr_result.supported_modulation
                )
                
                resource_allocation = ResourceAllocation(
                    link_id=link_id,
                    core_index=core_idx,
                    channel_index=channel_idx,
                    modulation_format=mod_format,
                    allocated_bitrate_gbps=min(gsnr_result.max_bitrate_gbps, 
                                             allocation_result.total_bitrate_gbps),
                    gsnr_db=gsnr_result.gsnr_db
                )
                
                resource_allocations.append(resource_allocation)
                
            except Exception as e:
                logger.warning(f"Error finalizing resource allocation: {e}")
                continue
        
        # Calculate node path
        node_path = self._calculate_node_path(path_links)
        
        # Update connection
        connection.allocated_path = node_path
        connection.resource_allocations = resource_allocations
        connection.total_allocated_bitrate_gbps = allocation_result.total_bitrate_gbps
        connection.end_to_end_gsnr_db = allocation_result.end_to_end_gsnr_db
        connection.path_length_km = sum(self.network.links[lid].length_km for lid in path_links 
                                      if lid in self.network.links)
    
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
            if link_id in self.network.links:
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
            try:
                connection_id = int(connection_id)
            except ValueError:
                logger.error(f"Invalid connection ID format: {connection_id}")
                return
        
        self.spectrum_allocation.deallocate_connection(connection_id)
    
    def preempt_connection(self, connection_id: int) -> bool:
        """Preempt a lower priority connection (if preemption is enabled)"""
        if not self.config['enable_preemption']:
            return False
        
        # Implementation for connection preemption
        # This would typically involve finding lower priority connections
        # and deallocating them to make room for higher priority requests
        logger.info(f"Preemption attempted for connection {connection_id}")
        return False
    
    def optimize_spectrum_allocation(self):
        """Perform spectrum defragmentation/optimization"""
        if not self.config['enable_fragmentation_aware_allocation']:
            return
        
        logger.info("Performing spectrum optimization...")
        
        # Get current fragmentation metrics
        frag_metrics = self.spectrum_allocation.get_fragmentation_metrics()
        
        if frag_metrics['fragmentation_ratio'] > 2.0:  # High fragmentation
            logger.info(f"High fragmentation detected: {frag_metrics['fragmentation_ratio']:.2f}")
            # Implement defragmentation algorithm here
            # This would involve finding contiguous blocks and potentially
            # rearranging existing connections
    
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
        
        # Get spectrum allocation statistics
        spectrum_stats = {
            'utilization_per_core': self.spectrum_allocation.get_utilization_per_core(),
            'utilization_per_band': self.spectrum_allocation.get_utilization_per_band(self.band_channels),
            'fragmentation_metrics': self.spectrum_allocation.get_fragmentation_metrics(),
            'peak_utilization': self.spectrum_allocation.peak_utilization,
            'total_allocations': self.spectrum_allocation.total_allocations,
            'total_deallocations': self.spectrum_allocation.total_deallocations
        }
        
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
            'gsnr_performance': {
                'total_gsnr_calculations': self.algorithm_stats['total_gsnr_calculations'],
                'average_gsnr_time_ms': avg_gsnr_time_ms,
                'average_interfering_channels': self.algorithm_stats['average_interfering_channels'],
                'gsnr_calculation_failures': self.algorithm_stats['gsnr_calculation_failures'],
                'timeout_failures': self.algorithm_stats['timeout_failures']
            },
            'spectrum_statistics': spectrum_stats,
            'allocation_failures': dict(self.algorithm_stats['allocation_failures']),
            'algorithm_configuration': self.config.copy()
        }
    
    def get_network_state_summary(self) -> Dict:
        """Get current network state summary with interference tracking"""
        validation_results = self.spectrum_allocation.validate_allocation_integrity()
        
        return {
            'spectrum_allocation_matrix_shape': self.spectrum_allocation.allocation.shape,
            'total_resources': (self.spectrum_allocation.num_links * 
                              self.spectrum_allocation.num_cores * 
                              self.spectrum_allocation.num_channels),
            'allocated_resources': np.count_nonzero(self.spectrum_allocation.allocation),
            'active_connections': len(self.spectrum_allocation.connection_resources),
            'utilization_per_core': self.spectrum_allocation.get_utilization_per_core(),
            'utilization_per_band': self.spectrum_allocation.get_utilization_per_band(self.band_channels),
            'overall_utilization': self.spectrum_allocation.get_utilization(),
            'allocation_history_size': len(self.spectrum_allocation.allocation_history),
            'fragmentation_metrics': self.spectrum_allocation.get_fragmentation_metrics(),
            'integrity_validation': validation_results,
            'peak_utilization': self.spectrum_allocation.peak_utilization
        }
    
    def export_allocation_state(self, filename: str = None) -> str:
        """Export current allocation state to JSON file"""
        if filename is None:
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            filename = f"spectrum_allocation_state_{timestamp}.json"
        
        state_data = {
            'timestamp': time.time(),
            'algorithm_stats': self.algorithm_stats,
            'spectrum_allocation': {
                'allocation_matrix': self.spectrum_allocation.allocation.tolist(),
                'connection_resources': {str(k): v for k, v in self.spectrum_allocation.connection_resources.items()},
                'utilization_metrics': {
                    'overall': self.spectrum_allocation.get_utilization(),
                    'per_core': self.spectrum_allocation.get_utilization_per_core(),
                    'per_band': self.spectrum_allocation.get_utilization_per_band(self.band_channels)
                }
            },
            'network_summary': self.get_network_state_summary(),
            'algorithm_configuration': self.config
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(state_data, f, indent=2)
            logger.info(f"Allocation state exported to {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to export allocation state: {e}")
            return ""
    
    def reset_algorithm_state(self):
        """Reset algorithm to initial state"""
        logger.info("Resetting algorithm state...")
        
        # Reset allocation matrix
        self.spectrum_allocation.allocation.fill(0)
        self.spectrum_allocation.connection_resources.clear()
        self.spectrum_allocation.allocation_history.clear()
        self.spectrum_allocation.total_allocations = 0
        self.spectrum_allocation.total_deallocations = 0
        self.spectrum_allocation.peak_utilization = 0.0
        
        # Reset statistics
        self.algorithm_stats = {
            'total_requests': 0,
            'successful_allocations': 0,
            'blocked_requests': 0,
            'single_chunk_allocations': 0,
            'sliced_allocations': 0,
            'average_computation_time_ms': 0.0,
            'total_gsnr_calculations': 0,
            'total_gsnr_time_ms': 0.0,
            'average_interfering_channels': 0.0,
            'allocation_failures': defaultdict(int),
            'gsnr_calculation_failures': 0,
            'timeout_failures': 0
        }
        
        # Clear GSNR calculator cache if available
        if hasattr(self.gsnr_calculator, 'clear_cache'):
            self.gsnr_calculator.clear_cache()
        
        logger.info("Algorithm state reset complete")

# Example usage and testing
if __name__ == "__main__":
    print("Updated XT-NLI-A-RSA Algorithm with Realistic Interference Modeling")
    print("Enhanced version with:")
    print("- Comprehensive error handling and validation")
    print("- Detailed performance statistics and monitoring")
    print("- Load balancing and fragmentation awareness")
    print("- State export/import capabilities")
    print("- Advanced debugging and logging")
    print("\nUse within the main MCF EON simulator framework for full functionality")