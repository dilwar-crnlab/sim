#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic Connection Manager for MCF EON
Handles dynamic connection requests with arrivals and departures
"""

import numpy as np
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
from enum import Enum
import time

class ConnectionStatus(Enum):
    """Connection status enumeration"""
    PENDING = "pending"
    ALLOCATED = "allocated"
    BLOCKED = "blocked"
    TERMINATED = "terminated"

class ModulationFormat(Enum):
    """Modulation format enumeration with bit rates at 64 GBaud"""
    PM_BPSK = ("PM-BPSK", 1, 100)    # (name, cardinality, bitrate_gbps)
    PM_QPSK = ("PM-QPSK", 2, 200)
    PM_8QAM = ("PM-8QAM", 3, 300)
    PM_16QAM = ("PM-16QAM", 4, 400)
    PM_32QAM = ("PM-32QAM", 5, 500)
    PM_64QAM = ("PM-64QAM", 6, 600)
    
    def __init__(self, format_name: str, cardinality: int, bitrate_gbps: int):
        self.format_name = format_name
        self.cardinality = cardinality
        self.bitrate_gbps = bitrate_gbps

@dataclass
class ResourceAllocation:
    """Resource allocation for a connection segment"""
    link_id: int
    core_index: int
    channel_index: int
    modulation_format: ModulationFormat
    allocated_bitrate_gbps: float
    gsnr_db: float

@dataclass
class Connection:
    """Connection request and allocation information"""
    connection_id: str
    source_node: int
    destination_node: int
    bandwidth_demand_gbps: float
    holding_time_hours: float
    arrival_time: float
    priority: int = 1  
    service_type: str = "best_effort" 
    
    # Allocation results
    status: ConnectionStatus = ConnectionStatus.PENDING
    allocated_path: List[int] = field(default_factory=list)
    resource_allocations: List[ResourceAllocation] = field(default_factory=list)
    total_allocated_bitrate_gbps: float = 0.0
    establishment_time: Optional[float] = None
    termination_time: Optional[float] = None
    blocking_reason: Optional[str] = None
    
    # QoT metrics
    end_to_end_gsnr_db: Optional[float] = None
    worst_case_gsnr_db: Optional[float] = None
    path_length_km: Optional[float] = None

    def __str__(self) -> str:
        """String representation of the connection"""
        basic = f"{self.connection_id}: {self.source_node}→{self.destination_node} ({self.bandwidth_demand_gbps:.0f}G)"
        
        if self.status == ConnectionStatus.ALLOCATED:
            if self.end_to_end_gsnr_db and self.path_length_km:
                status = f"[ACTIVE: {self.path_length_km:.0f}km, {self.end_to_end_gsnr_db:.1f}dB]"
            else:
                status = f"[ACTIVE: {len(self.resource_allocations)} segments]"
        elif self.status == ConnectionStatus.BLOCKED:
            status = f"[BLOCKED: {self.blocking_reason or 'No resources'}]"
        elif self.status == ConnectionStatus.TERMINATED:
            duration = (self.termination_time or 0) - self.arrival_time
            status = f"[TERMINATED: {duration:.1f} time units]"
        else:
            status = "[PENDING]"
        
        return f"{basic} {status}"
    
    def is_satisfied(self, tolerance_gbps: float = 0.1) -> bool:
        """Check if connection bandwidth demand is satisfied"""
        return (self.total_allocated_bitrate_gbps >= 
                (self.bandwidth_demand_gbps - tolerance_gbps))
    
    def get_utilization_ratio(self) -> float:
        """Get bandwidth utilization ratio"""
        if self.bandwidth_demand_gbps <= 0:
            return 0.0
        return self.total_allocated_bitrate_gbps / self.bandwidth_demand_gbps
    
    def get_service_time(self, current_time: float) -> float:
        """Get current service time"""
        if self.establishment_time is None:
            return 0.0
        end_time = self.termination_time or current_time
        return end_time - self.establishment_time

class DynamicConnectionManager:
    """
    Dynamic connection manager for time-based simulation
    Handles arrivals, departures, and connection lifecycle
    """
    
    def __init__(self):
        # Connection storage
        self.connections: Dict[str, Connection] = {}
        self.active_connections: Set[str] = set()
        self.blocked_connections: Set[str] = set()
        self.terminated_connections: Set[str] = set()
        
        # Dynamic statistics
        self.total_arrivals = 0
        self.total_blocked = 0
        self.total_terminated = 0
        self.current_simulation_time = 0.0
        
        # Bandwidth tracking
        self.total_bandwidth_requested_gbps = 0.0
        self.total_bandwidth_allocated_gbps = 0.0
        self.total_bandwidth_terminated_gbps = 0.0
        
        # Time-based metrics
        self.connection_arrival_times = []
        self.connection_service_times = []
        self.blocking_events = []
        
        print("Dynamic Connection Manager initialized")
    
    def update_simulation_time(self, current_time: float):
        """Update current simulation time"""
        self.current_simulation_time = current_time
    
    def process_arrival(self, connection: Connection) -> Connection:
        """
        Process a connection arrival
        
        Args:
            connection: Arriving connection request
            
        Returns:
            The same connection object (now tracked)
        """
        # Add to tracking
        self.connections[connection.connection_id] = connection
        self.total_arrivals += 1
        self.total_bandwidth_requested_gbps += connection.bandwidth_demand_gbps
        self.connection_arrival_times.append(connection.arrival_time)
        
        print(f"📥 Arrival processed: {connection.connection_id} at time {connection.arrival_time:.2f}")
        return connection
    
    def allocate_connection(self, connection_id: str, path: List[int],
                          resource_allocations: List[ResourceAllocation],
                          end_to_end_gsnr_db: float, path_length_km: float) -> bool:
        """
        Allocate resources to a connection
        
        Args:
            connection_id: Connection identifier
            path: Allocated path (node sequence)
            resource_allocations: List of resource allocations
            end_to_end_gsnr_db: End-to-end GSNR
            path_length_km: Path length
            
        Returns:
            True if allocation successful
        """
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        # Calculate total allocated bitrate
        total_bitrate = sum(alloc.allocated_bitrate_gbps for alloc in resource_allocations)
        
        # Update connection
        connection.status = ConnectionStatus.ALLOCATED
        connection.allocated_path = path.copy()
        connection.resource_allocations = resource_allocations.copy()
        connection.total_allocated_bitrate_gbps = total_bitrate
        connection.establishment_time = self.current_simulation_time
        connection.end_to_end_gsnr_db = end_to_end_gsnr_db
        connection.path_length_km = path_length_km
        #connection.worst_case_gsnr_db = min(alloc.gsnr_db for alloc in resource_allocations)
        connection.worst_case_gsnr_db = min([alloc.gsnr_db for alloc in resource_allocations], default=end_to_end_gsnr_db)
        
        # Update tracking
        self.active_connections.add(connection_id)
        self.total_bandwidth_allocated_gbps += total_bitrate
        
        print(f"✅ Allocated: {connection_id} - {total_bitrate:.0f}Gbps, {end_to_end_gsnr_db:.2f}dB")
        return True
    
    def block_connection(self, connection_id: str, reason: str) -> bool:
        """
        Block a connection request
        
        Args:
            connection_id: Connection identifier
            reason: Blocking reason
            
        Returns:
            True if blocking recorded successfully
        """
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.status = ConnectionStatus.BLOCKED
        connection.blocking_reason = reason
        
        # Update tracking
        self.blocked_connections.add(connection_id)
        self.total_blocked += 1
        self.blocking_events.append({
            'time': self.current_simulation_time,
            'connection_id': connection_id,
            'reason': reason,
            'bandwidth_gbps': connection.bandwidth_demand_gbps
        })
        
        print(f"❌ Blocked: {connection_id} - {reason}")
        return True
    
    def process_departure(self, connection_id: str) -> bool:
        """
        Process a connection departure (termination)
        
        Args:
            connection_id: Connection identifier
            
        Returns:
            True if departure processed successfully
        """
        if connection_id not in self.connections:
            print(f"⚠️  Departure event for unknown connection: {connection_id}")
            return False
        
        connection = self.connections[connection_id]
        
        if connection.status != ConnectionStatus.ALLOCATED:
            print(f"⚠️  Departure event for non-active connection: {connection_id} (status: {connection.status})")
            return False
        
        # Update connection
        connection.status = ConnectionStatus.TERMINATED
        connection.termination_time = self.current_simulation_time
        
        # Calculate service time
        service_time = connection.get_service_time(self.current_simulation_time)
        
        # Update tracking
        self.active_connections.discard(connection_id)
        self.terminated_connections.add(connection_id)
        self.total_terminated += 1
        self.total_bandwidth_allocated_gbps -= connection.total_allocated_bitrate_gbps
        self.total_bandwidth_terminated_gbps += connection.total_allocated_bitrate_gbps
        self.connection_service_times.append(service_time)
        
        print(f"⬆️  DEPARTURE at time {self.current_simulation_time:.2f}: {connection_id} "
              f"(served {service_time:.1f} time units)")
        return True
    
    def get_active_connections(self) -> List[Connection]:
        """Get all currently active connections"""
        return [self.connections[conn_id] for conn_id in self.active_connections]
    
    def get_blocked_connections(self) -> List[Connection]:
        """Get all blocked connections"""
        return [self.connections[conn_id] for conn_id in self.blocked_connections]
    
    def get_terminated_connections(self) -> List[Connection]:
        """Get all terminated connections"""
        return [self.connections[conn_id] for conn_id in self.terminated_connections]
    
    def get_connection_by_id(self, connection_id: str) -> Optional[Connection]:
        """Get connection by ID"""
        return self.connections.get(connection_id)
    
    def calculate_blocking_probability(self) -> float:
        """Calculate connection blocking probability"""
        if self.total_arrivals <= 0:
            return 0.0
        return self.total_blocked / self.total_arrivals
    
    def calculate_bandwidth_blocking_probability(self) -> float:
        """Calculate bandwidth blocking probability"""
        if self.total_bandwidth_requested_gbps <= 0:
            return 0.0
        
        blocked_bandwidth = sum(
            self.connections[conn_id].bandwidth_demand_gbps 
            for conn_id in self.blocked_connections
        )
        
        return blocked_bandwidth / self.total_bandwidth_requested_gbps
    
    def get_current_utilization(self, total_network_capacity_gbps: float) -> float:
        """Calculate current network utilization"""
        if total_network_capacity_gbps <= 0:
            return 0.0
        return self.total_bandwidth_allocated_gbps / total_network_capacity_gbps
    
    def get_dynamic_statistics(self) -> Dict:
        """Get comprehensive dynamic simulation statistics"""
        
        # Connection counts
        active_count = len(self.active_connections)
        blocked_count = len(self.blocked_connections)
        terminated_count = len(self.terminated_connections)
        
        # Calculate statistics
        blocking_prob = self.calculate_blocking_probability()
        bandwidth_blocking_prob = self.calculate_bandwidth_blocking_probability()
        
        # Time-based metrics
        if self.connection_service_times:
            mean_service_time = np.mean(self.connection_service_times)
            std_service_time = np.std(self.connection_service_times)
        else:
            mean_service_time = std_service_time = 0.0
        
        # Current active connections statistics
        if active_count > 0:
            active_connections = self.get_active_connections()
            avg_gsnr = np.mean([conn.end_to_end_gsnr_db for conn in active_connections 
                              if conn.end_to_end_gsnr_db is not None])
            avg_path_length = np.mean([conn.path_length_km for conn in active_connections
                                     if conn.path_length_km is not None])
            avg_utilization = np.mean([conn.get_utilization_ratio() for conn in active_connections])
        else:
            avg_gsnr = avg_path_length = avg_utilization = 0.0
        
        # Arrival rate estimation (if we have enough data)
        if len(self.connection_arrival_times) > 1:
            arrival_times = np.array(self.connection_arrival_times)
            time_span = arrival_times[-1] - arrival_times[0]
            if time_span > 0:
                estimated_arrival_rate = (len(arrival_times) - 1) / time_span
            else:
                estimated_arrival_rate = 0.0
        else:
            estimated_arrival_rate = 0.0
        
        return {
            'simulation_time': self.current_simulation_time,
            'connection_counts': {
                'total_arrivals': self.total_arrivals,
                'active': active_count,
                'blocked': blocked_count,
                'terminated': terminated_count
            },
            'blocking_metrics': {
                'connection_blocking_probability': blocking_prob,
                'bandwidth_blocking_probability': bandwidth_blocking_prob,
                'total_blocked': self.total_blocked
            },
            'bandwidth_metrics': {
                'total_requested_gbps': self.total_bandwidth_requested_gbps,
                'currently_allocated_gbps': self.total_bandwidth_allocated_gbps,
                'total_terminated_gbps': self.total_bandwidth_terminated_gbps
            },
            'service_time_metrics': {
                'mean_service_time': mean_service_time,
                'std_service_time': std_service_time,
                'total_completed_services': len(self.connection_service_times)
            },
            'current_active_metrics': {
                'average_gsnr_db': avg_gsnr,
                'average_path_length_km': avg_path_length,
                'average_utilization_ratio': avg_utilization
            },
            'traffic_metrics': {
                'estimated_arrival_rate_per_time_unit': estimated_arrival_rate,
                'current_load_connections': active_count
            }
        }
    
    def generate_simulation_report(self) -> str:
        """Generate a comprehensive simulation report"""
        stats = self.get_dynamic_statistics()
        
        report = "MCF EON Dynamic Simulation Report\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Simulation Time: {stats['simulation_time']:.2f} time units\n\n"
        
        report += f"Connection Statistics:\n"
        report += f"  Total Arrivals: {stats['connection_counts']['total_arrivals']}\n"
        report += f"  Currently Active: {stats['connection_counts']['active']}\n"
        report += f"  Blocked: {stats['connection_counts']['blocked']}\n"
        report += f"  Terminated: {stats['connection_counts']['terminated']}\n\n"
        
        report += f"Performance Metrics:\n"
        report += f"  Connection Blocking Probability: {stats['blocking_metrics']['connection_blocking_probability']:.3%}\n"
        report += f"  Bandwidth Blocking Probability: {stats['blocking_metrics']['bandwidth_blocking_probability']:.3%}\n\n"
        
        report += f"Bandwidth Summary:\n"
        report += f"  Total Requested: {stats['bandwidth_metrics']['total_requested_gbps']:.1f} Gbps\n"
        report += f"  Currently Allocated: {stats['bandwidth_metrics']['currently_allocated_gbps']:.1f} Gbps\n"
        report += f"  Total Terminated: {stats['bandwidth_metrics']['total_terminated_gbps']:.1f} Gbps\n\n"
        
        if stats['service_time_metrics']['total_completed_services'] > 0:
            report += f"Service Time Analysis:\n"
            report += f"  Mean Service Time: {stats['service_time_metrics']['mean_service_time']:.2f} time units\n"
            report += f"  Std Service Time: {stats['service_time_metrics']['std_service_time']:.2f} time units\n"
            report += f"  Completed Services: {stats['service_time_metrics']['total_completed_services']}\n\n"
        
        if stats['connection_counts']['active'] > 0:
            report += f"Current Active Connections:\n"
            report += f"  Average GSNR: {stats['current_active_metrics']['average_gsnr_db']:.2f} dB\n"
            report += f"  Average Path Length: {stats['current_active_metrics']['average_path_length_km']:.1f} km\n"
            report += f"  Average Utilization: {stats['current_active_metrics']['average_utilization_ratio']:.2%}\n\n"
        
        report += f"Traffic Characteristics:\n"
        report += f"  Estimated Arrival Rate: {stats['traffic_metrics']['estimated_arrival_rate_per_time_unit']:.3f} per time unit\n"
        report += f"  Current Load: {stats['traffic_metrics']['current_load_connections']} active connections\n"
        
        return report
    
    def reset(self):
        """Reset the connection manager"""
        self.connections.clear()
        self.active_connections.clear()
        self.blocked_connections.clear()
        self.terminated_connections.clear()
        
        self.total_arrivals = 0
        self.total_blocked = 0
        self.total_terminated = 0
        self.current_simulation_time = 0.0
        
        self.total_bandwidth_requested_gbps = 0.0
        self.total_bandwidth_allocated_gbps = 0.0
        self.total_bandwidth_terminated_gbps = 0.0
        
        self.connection_arrival_times.clear()
        self.connection_service_times.clear()
        self.blocking_events.clear()
        
        print("Dynamic Connection Manager reset")

# Example usage
if __name__ == "__main__":
    print("Dynamic Connection Manager Test")
    print("=" * 40)
    
    # Create manager
    manager = DynamicConnectionManager()
    
    # Simulate some connections
    manager.update_simulation_time(0.0)
    
    # Test connection 1
    conn1 = Connection(
        connection_id="test_001",
        source_node=0,
        destination_node=3,
        bandwidth_demand_gbps=400,
        holding_time_hours=10.0,
        arrival_time=0.0
    )
    
    manager.process_arrival(conn1)
    manager.allocate_connection(
        "test_001", [0, 1, 3], 
        [ResourceAllocation(0, 0, 10, ModulationFormat.PM_16QAM, 400, 15.2)],
        15.2, 1200
    )
    
    # Test connection 2
    manager.update_simulation_time(2.5)
    conn2 = Connection(
        connection_id="test_002", 
        source_node=1,
        destination_node=2,
        bandwidth_demand_gbps=600,
        holding_time_hours=8.0,
        arrival_time=2.5
    )
    
    manager.process_arrival(conn2)
    manager.block_connection("test_002", "Insufficient spectrum")
    
    # Test departure
    manager.update_simulation_time(10.0)
    manager.process_departure("test_001")
    
    # Generate report
    print(manager.generate_simulation_report())