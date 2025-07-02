#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic Traffic Generator for MCF EON Simulation
File: sim/traffic_generator.py
"""

import numpy as np
from typing import List
from connection_manager import Connection

class DynamicTrafficGenerator:
    """Generates realistic traffic patterns with Poisson arrivals"""
    
    def __init__(self, network_topology, arrival_rate_per_hour: float = 100.0):
        """
        Initialize traffic generator
        
        Args:
            network_topology: Network topology object
            arrival_rate_per_hour: Average arrivals per hour (λ parameter)
        """
        self.network = network_topology
        self.arrival_rate_per_hour = arrival_rate_per_hour
        self.arrival_rate_per_second = arrival_rate_per_hour / 3600.0
        
        # Get core nodes for traffic generation
        self.core_nodes = [node_id for node_id, node in self.network.nodes.items() 
                          if node.add_drop_enabled]
        
        # Traffic demand characteristics (based on realistic telecom patterns)
        self.bandwidth_options = [100, 200, 300, 400, 500, 600]  # Gbps
        self.bandwidth_probabilities = [0.3, 0.25, 0.2, 0.15, 0.07, 0.03]  # Favor lower BW
        
        # Holding time parameters (hours)
        self.holding_time_mean = 2.5  # Average 2.5 hours
        self.holding_time_std = 1.0   # Standard deviation 1 hour
        self.min_holding_time = 0.1   # Minimum 6 minutes
        
        print(f"Dynamic Traffic Generator initialized:")
        print(f"  Arrival rate: {arrival_rate_per_hour:.1f} requests/hour")
        print(f"  Core nodes available: {len(self.core_nodes)}")
        print(f"  Mean holding time: {self.holding_time_mean:.1f} hours")
        print(f"  Bandwidth distribution: {dict(zip(self.bandwidth_options, self.bandwidth_probabilities))}")
    
    def generate_next_arrival_time(self, current_time: float) -> float:
        """
        Generate next arrival time using exponential distribution (Poisson process)
        
        Args:
            current_time: Current simulation time
            
        Returns:
            Next arrival time
        """
        inter_arrival_time = np.random.exponential(1.0 / self.arrival_rate_per_second)
        return current_time + inter_arrival_time
    
    def generate_connection_request(self, arrival_time: float, request_id: int) -> Connection:
        """
        Generate a single connection request with realistic parameters
        
        Args:
            arrival_time: Time when connection request arrives
            request_id: Unique request identifier
            
        Returns:
            Connection object with generated parameters
        """
        
        # Random source and destination (must be different)
        source, dest = np.random.choice(self.core_nodes, 2, replace=False)
        
        # Bandwidth demand with weighted random selection
        bandwidth_gbps = np.random.choice(
            self.bandwidth_options, 
            p=self.bandwidth_probabilities
        )
        
        # Holding time (normal distribution, clipped to positive values)
        holding_time = max(
            self.min_holding_time, 
            np.random.normal(self.holding_time_mean, self.holding_time_std)
        )
        
        # Create connection object
        connection = Connection(
            connection_id=f"dyn_{request_id:06d}",
            source_node=source,
            destination_node=dest,
            bandwidth_demand_gbps=bandwidth_gbps,
            holding_time_hours=holding_time,
            arrival_time=arrival_time,
            priority=1,
            service_type="best_effort"
        )
        
        return connection
    
    def get_traffic_statistics(self) -> dict:
        """Get traffic generation parameters for reporting"""
        return {
            'arrival_rate_per_hour': self.arrival_rate_per_hour,
            'arrival_rate_per_second': self.arrival_rate_per_second,
            'mean_holding_time_hours': self.holding_time_mean,
            'holding_time_std_hours': self.holding_time_std,
            'min_holding_time_hours': self.min_holding_time,
            'core_nodes_count': len(self.core_nodes),
            'bandwidth_options_gbps': self.bandwidth_options,
            'bandwidth_probabilities': self.bandwidth_probabilities,
            'expected_offered_load_erlangs': self.arrival_rate_per_hour * self.holding_time_mean
        }
    
    def estimate_offered_load(self) -> float:
        """
        Calculate offered load in Erlangs (λ × μ)
        
        Returns:
            Offered load in Erlangs
        """
        return self.arrival_rate_per_hour * self.holding_time_mean
    
    def validate_configuration(self) -> bool:
        """
        Validate traffic generator configuration
        
        Returns:
            True if configuration is valid
        """
        checks = [
            len(self.core_nodes) >= 2,
            self.arrival_rate_per_hour > 0,
            self.holding_time_mean > 0,
            len(self.bandwidth_options) == len(self.bandwidth_probabilities),
            abs(sum(self.bandwidth_probabilities) - 1.0) < 1e-6
        ]
        
        if not all(checks):
            print("Traffic generator configuration validation failed!")
            return False
        
        return True