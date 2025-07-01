#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Connection Manager for MCF EON
Handles connection requests, bandwidth allocation, and lightpath management
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid

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
    arrival_time: float = field(default_factory=time.time)
    priority: int = 1  # 1 = highest priority
    service_type: str = "best_effort"  # "guaranteed", "best_effort"
    
    # Allocation results
    status: ConnectionStatus = ConnectionStatus.PENDING
    allocated_path: List[int] = field(default_factory=list)
    resource_allocations: List[ResourceAllocation] = field(default_factory=list)
    total_allocated_bitrate_gbps: float = 0.0
    establishment_time: Optional[float] = None
    blocking_reason: Optional[str] = None
    
    # QoT metrics
    end_to_end_gsnr_db: Optional[float] = None
    worst_case_gsnr_db: Optional[float] = None
    path_length_km: Optional[float] = None

    def __str__(self) -> str:
        """Compact string representation of the connection"""
        
        # Basic format: ID: src→dst (bandwidth) [STATUS]
        basic = f"{self.connection_id}: {self.source_node}→{self.destination_node} ({self.bandwidth_demand_gbps:.0f}G)"
        
        # Add status with context
        if self.status == ConnectionStatus.ALLOCATED:
            if self.end_to_end_gsnr_db and self.path_length_km:
                status = f"[ACTIVE: {self.path_length_km:.0f}km, {self.end_to_end_gsnr_db:.1f}dB]"
            else:
                status = f"[ACTIVE: {len(self.resource_allocations)} segments]"
        elif self.status == ConnectionStatus.BLOCKED:
            status = f"[BLOCKED: {self.blocking_reason or 'No resources'}]"
        elif self.status == ConnectionStatus.TERMINATED:
            status = "[TERMINATED]"
        else:
            status = "[PENDING]"
        
        return f"{basic} {status}"
    
    def __post_init__(self):
        if not self.connection_id:
            self.connection_id = str(uuid.uuid4())[:8]
    
    def is_satisfied(self, tolerance_gbps: float = 0.1) -> bool:
        """Check if connection bandwidth demand is satisfied"""
        return (self.total_allocated_bitrate_gbps >= 
                (self.bandwidth_demand_gbps - tolerance_gbps))
    
    def get_utilization_ratio(self) -> float:
        """Get bandwidth utilization ratio"""
        if self.bandwidth_demand_gbps <= 0:
            return 0.0
        return self.total_allocated_bitrate_gbps / self.bandwidth_demand_gbps
    
    def get_allocation_summary(self) -> Dict:
        """Get allocation summary"""
        return {
            'connection_id': self.connection_id,
            'status': self.status.value,
            'demand_gbps': self.bandwidth_demand_gbps,
            'allocated_gbps': self.total_allocated_bitrate_gbps,
            'utilization_ratio': self.get_utilization_ratio(),
            'path': self.allocated_path,
            'num_segments': len(self.resource_allocations),
            'end_to_end_gsnr_db': self.end_to_end_gsnr_db,
            'path_length_km': self.path_length_km
        }

class ConnectionManager:
    """Manages connections and their resource allocations"""
    
    def __init__(self):
        self.connections: Dict[str, Connection] = {}
        self.active_connections: Set[str] = set()
        self.blocked_connections: Set[str] = set()
        self.terminated_connections: Set[str] = set()
        
        # Statistics
        self.total_requests = 0
        self.total_blocked = 0
        self.total_bandwidth_requested_gbps = 0.0
        self.total_bandwidth_allocated_gbps = 0.0
        
    def add_connection_request(self, source_node: int, destination_node: int,
                             bandwidth_demand_gbps: float, holding_time_hours: float,
                             priority: int = 1, service_type: str = "best_effort",
                             connection_id: str = None) -> Connection:
        """Add a new connection request"""
        
        connection = Connection(
            connection_id=connection_id or str(uuid.uuid4())[:8],
            source_node=source_node,
            destination_node=destination_node,
            bandwidth_demand_gbps=bandwidth_demand_gbps,
            holding_time_hours=holding_time_hours,
            priority=priority,
            service_type=service_type
        )
        
        self.connections[connection.connection_id] = connection
        self.total_requests += 1
        self.total_bandwidth_requested_gbps += bandwidth_demand_gbps
        
        return connection
    
    def allocate_connection(self, connection_id: str, path: List[int],
                          resource_allocations: List[ResourceAllocation],
                          end_to_end_gsnr_db: float, path_length_km: float) -> bool:
        """Allocate resources to a connection"""
        
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
        connection.establishment_time = time.time()
        connection.end_to_end_gsnr_db = end_to_end_gsnr_db
        connection.path_length_km = path_length_km
        connection.worst_case_gsnr_db = min(alloc.gsnr_db for alloc in resource_allocations)
        
        # Update tracking sets
        self.active_connections.add(connection_id)
        self.total_bandwidth_allocated_gbps += total_bitrate
        
        return True
    
    def block_connection(self, connection_id: str, reason: str) -> bool:
        """Block a connection request"""
        
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.status = ConnectionStatus.BLOCKED
        connection.blocking_reason = reason
        
        # Update tracking
        self.blocked_connections.add(connection_id)
        self.total_blocked += 1
        
        return True
    
    def terminate_connection(self, connection_id: str) -> bool:
        """Terminate an active connection"""
        
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        if connection.status != ConnectionStatus.ALLOCATED:
            return False
        
        # Update status
        connection.status = ConnectionStatus.TERMINATED
        
        # Update tracking
        self.active_connections.discard(connection_id)
        self.terminated_connections.add(connection_id)
        self.total_bandwidth_allocated_gbps -= connection.total_allocated_bitrate_gbps
        
        return True
    
    def get_active_connections(self) -> List[Connection]:
        """Get all active connections"""
        return [self.connections[conn_id] for conn_id in self.active_connections]
    
    def get_blocked_connections(self) -> List[Connection]:
        """Get all blocked connections"""
        return [self.connections[conn_id] for conn_id in self.blocked_connections]
    
    def get_connection_by_id(self, connection_id: str) -> Optional[Connection]:
        """Get connection by ID"""
        return self.connections.get(connection_id)
    
    def get_connections_on_link(self, link_id: int) -> List[Connection]:
        """Get all connections using a specific link"""
        connections_on_link = []
        
        for connection in self.get_active_connections():
            for allocation in connection.resource_allocations:
                if allocation.link_id == link_id:
                    connections_on_link.append(connection)
                    break
        
        return connections_on_link
    
    def get_connections_on_core(self, link_id: int, core_index: int) -> List[Connection]:
        """Get all connections using specific link and core"""
        connections_on_core = []
        
        for connection in self.get_active_connections():
            for allocation in connection.resource_allocations:
                if allocation.link_id == link_id and allocation.core_index == core_index:
                    connections_on_core.append(connection)
                    break
        
        return connections_on_core
    
    def calculate_blocking_probability(self) -> float:
        """Calculate bandwidth blocking probability"""
        if self.total_bandwidth_requested_gbps <= 0:
            return 0.0
        
        blocked_bandwidth = sum(
            self.connections[conn_id].bandwidth_demand_gbps 
            for conn_id in self.blocked_connections
        )
        
        return blocked_bandwidth / self.total_bandwidth_requested_gbps
    
    def calculate_resource_utilization(self, total_network_capacity_gbps: float) -> float:
        """Calculate network resource utilization"""
        if total_network_capacity_gbps <= 0:
            return 0.0
        
        return self.total_bandwidth_allocated_gbps / total_network_capacity_gbps
    
    def get_statistics(self) -> Dict:
        """Get comprehensive connection statistics"""
        
        active_connections = self.get_active_connections()
        blocked_connections = self.get_blocked_connections()
        
        # Calculate average metrics for active connections
        if active_connections:
            avg_gsnr = np.mean([conn.end_to_end_gsnr_db for conn in active_connections 
                              if conn.end_to_end_gsnr_db is not None])
            avg_path_length = np.mean([conn.path_length_km for conn in active_connections
                                     if conn.path_length_km is not None])
            avg_utilization = np.mean([conn.get_utilization_ratio() for conn in active_connections])
        else:
            avg_gsnr = avg_path_length = avg_utilization = 0.0
        
        # Modulation format distribution
        mod_format_count = {}
        for connection in active_connections:
            for allocation in connection.resource_allocations:
                mod_format = allocation.modulation_format.format_name
                mod_format_count[mod_format] = mod_format_count.get(mod_format, 0) + 1
        
        return {
            'total_requests': self.total_requests,
            'active_connections': len(self.active_connections),
            'blocked_connections': len(self.blocked_connections),
            'terminated_connections': len(self.terminated_connections),
            'blocking_probability': self.calculate_blocking_probability(),
            'bandwidth_statistics': {
                'total_requested_gbps': self.total_bandwidth_requested_gbps,
                'total_allocated_gbps': self.total_bandwidth_allocated_gbps,
                'allocation_efficiency': (self.total_bandwidth_allocated_gbps / 
                                        max(self.total_bandwidth_requested_gbps, 1e-9))
            },
            'average_metrics': {
                'gsnr_db': avg_gsnr,
                'path_length_km': avg_path_length,
                'utilization_ratio': avg_utilization
            },
            'modulation_format_distribution': mod_format_count
        }
    
    def generate_connection_report(self) -> str:
        """Generate a detailed connection report"""
        stats = self.get_statistics()
        
        report = "MCF EON Connection Manager Report\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Connection Summary:\n"
        report += f"  Total Requests: {stats['total_requests']}\n"
        report += f"  Active: {stats['active_connections']}\n"
        report += f"  Blocked: {stats['blocked_connections']}\n"
        report += f"  Terminated: {stats['terminated_connections']}\n"
        report += f"  Blocking Probability: {stats['blocking_probability']:.2%}\n\n"
        
        report += f"Bandwidth Statistics:\n"
        report += f"  Total Requested: {stats['bandwidth_statistics']['total_requested_gbps']:.1f} Gbps\n"
        report += f"  Total Allocated: {stats['bandwidth_statistics']['total_allocated_gbps']:.1f} Gbps\n"
        report += f"  Allocation Efficiency: {stats['bandwidth_statistics']['allocation_efficiency']:.2%}\n\n"
        
        report += f"Average Quality Metrics:\n"
        report += f"  GSNR: {stats['average_metrics']['gsnr_db']:.2f} dB\n"
        report += f"  Path Length: {stats['average_metrics']['path_length_km']:.1f} km\n"
        report += f"  Utilization Ratio: {stats['average_metrics']['utilization_ratio']:.2%}\n\n"
        
        if stats['modulation_format_distribution']:
            report += f"Modulation Format Distribution:\n"
            for mod_format, count in stats['modulation_format_distribution'].items():
                report += f"  {mod_format}: {count}\n"
        
        return report
    
    def export_connections_to_dict(self) -> Dict:
        """Export all connections to dictionary format"""
        return {
            conn_id: {
                'connection_info': {
                    'connection_id': conn.connection_id,
                    'source_node': conn.source_node,
                    'destination_node': conn.destination_node,
                    'bandwidth_demand_gbps': conn.bandwidth_demand_gbps,
                    'holding_time_hours': conn.holding_time_hours,
                    'arrival_time': conn.arrival_time,
                    'priority': conn.priority,
                    'service_type': conn.service_type
                },
                'allocation_results': {
                    'status': conn.status.value,
                    'allocated_path': conn.allocated_path,
                    'total_allocated_bitrate_gbps': conn.total_allocated_bitrate_gbps,
                    'end_to_end_gsnr_db': conn.end_to_end_gsnr_db,
                    'worst_case_gsnr_db': conn.worst_case_gsnr_db,
                    'path_length_km': conn.path_length_km,
                    'establishment_time': conn.establishment_time,
                    'blocking_reason': conn.blocking_reason
                },
                'resource_allocations': [
                    {
                        'link_id': alloc.link_id,
                        'core_index': alloc.core_index,
                        'channel_index': alloc.channel_index,
                        'modulation_format': alloc.modulation_format.format_name,
                        'allocated_bitrate_gbps': alloc.allocated_bitrate_gbps,
                        'gsnr_db': alloc.gsnr_db
                    }
                    for alloc in conn.resource_allocations
                ]
            }
            for conn_id, conn in self.connections.items()
        }

# Example usage and testing
if __name__ == "__main__":
    # Create connection manager
    conn_mgr = ConnectionManager()
    
    # Add some test connection requests
    conn1 = conn_mgr.add_connection_request(
        source_node=0, destination_node=3, 
        bandwidth_demand_gbps=400, holding_time_hours=2.0
    )
    
    conn2 = conn_mgr.add_connection_request(
        source_node=1, destination_node=2,
        bandwidth_demand_gbps=600, holding_time_hours=1.5
    )
    
    conn3 = conn_mgr.add_connection_request(
        source_node=0, destination_node=2,
        bandwidth_demand_gbps=300, holding_time_hours=3.0
    )
    
    # Simulate allocation for conn1
    resource_allocations = [
        ResourceAllocation(
            link_id=0, core_index=0, channel_index=10,
            modulation_format=ModulationFormat.PM_16QAM,
            allocated_bitrate_gbps=400, gsnr_db=15.2
        )
    ]
    
    conn_mgr.allocate_connection(
        conn1.connection_id, 
        path=[0, 1, 3],
        resource_allocations=resource_allocations,
        end_to_end_gsnr_db=15.2,
        path_length_km=1200
    )
    
    # Simulate blocking for conn2
    conn_mgr.block_connection(conn2.connection_id, "Insufficient spectrum resources")
    
    # Print statistics
    print(conn_mgr.generate_connection_report())
    
    # Test connection queries
    print(f"\nActive connections: {len(conn_mgr.get_active_connections())}")
    print(f"Blocked connections: {len(conn_mgr.get_blocked_connections())}")
    print(f"Blocking probability: {conn_mgr.calculate_blocking_probability():.2%}")