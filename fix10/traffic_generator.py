#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic Traffic Generator for MCF EON Simulation
Generates connection requests with Poisson arrivals and exponential holding times
"""

import numpy as np
import heapq
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum
from connection_manager import Connection, ConnectionStatus

class EventType(Enum):
    """Types of simulation events"""
    ARRIVAL = "arrival"
    DEPARTURE = "departure"

@dataclass
class SimulationEvent:
    """
    Simulation event for priority queue
    Supports comparison for heapq operations
    """
    event_time: float
    event_type: EventType
    connection_id: str
    connection_data: Dict = None
    
    def __lt__(self, other):
        """For priority queue ordering by event time"""
        return self.event_time < other.event_time
    
    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, SimulationEvent):
            return False
        return (self.event_time == other.event_time and 
                self.connection_id == other.connection_id)

class DynamicTrafficGenerator:
    """
    Dynamic traffic generator with Poisson arrivals and exponential holding times
    """
    
    def __init__(self, network_topology, arrival_rate_per_hour: float = 100.0):
        """
        Initialize dynamic traffic generator
        
        Args:
            network_topology: Network topology object
            arrival_rate_per_hour: Average arrivals per hour (λ parameter)
        """
        self.network = network_topology
        self.arrival_rate_per_hour = arrival_rate_per_hour
        self.arrival_rate_per_second = arrival_rate_per_hour / 3600.0
        
        # Mean holding time: 10 time units (as requested)
        self.mean_holding_time = 10.0
        
        # Get core nodes for traffic generation
        self.core_nodes = [node_id for node_id, node in self.network.nodes.items() 
                          if node.add_drop_enabled]
        
        # Traffic demand characteristics
        self.bandwidth_options = [100, 200, 300, 400, 500, 600]  # Gbps
        self.bandwidth_probabilities = [0.3, 0.25, 0.2, 0.15, 0.07, 0.03]  # Favor lower BW
        
        # Event queue and counters
        self.event_queue = []
        self.next_connection_id = 0
        self.current_time = 0.0
        
        print(f"Dynamic Traffic Generator initialized:")
        print(f"  Arrival rate: {arrival_rate_per_hour:.1f} requests/hour")
        print(f"  Mean holding time: {self.mean_holding_time:.1f} time units")
        print(f"  Core nodes: {len(self.core_nodes)}")
        print(f"  Expected load: {self.arrival_rate_per_hour * self.mean_holding_time:.1f} Erlangs")
    
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
    
    def generate_holding_time(self) -> float:
        """
        Generate holding time using exponential distribution
        
        Returns:
            Holding time (exponential with mean 10.0)
        """
        return np.random.exponential(self.mean_holding_time)
    
    def generate_connection_request(self, arrival_time: float) -> Connection:
        """
        Generate a single connection request
        
        Args:
            arrival_time: Time when connection request arrives
            
        Returns:
            Connection object with generated parameters
        """
        
        # Generate unique connection ID
        connection_id = f"dyn_{self.next_connection_id:06d}"
        self.next_connection_id += 1
        
        # Random source and destination (must be different)
        source, dest = np.random.choice(self.core_nodes, 2, replace=False)
        
        # Bandwidth demand with weighted random selection
        bandwidth_gbps = np.random.choice(
            self.bandwidth_options, 
            p=self.bandwidth_probabilities
        )
        
        # Exponential holding time
        holding_time = self.generate_holding_time()
        
        # Create connection object
        connection = Connection(
            connection_id=connection_id,
            source_node=source,
            destination_node=dest,
            bandwidth_demand_gbps=bandwidth_gbps,
            holding_time_hours=holding_time,
            arrival_time=arrival_time,
            priority=1,
            service_type="best_effort"
        )
        
        return connection
    
    def schedule_first_arrival(self, start_time: float = 0.0):
        """
        Schedule the first arrival event
        
        Args:
            start_time: Simulation start time
        """
        self.current_time = start_time
        first_arrival_time = self.generate_next_arrival_time(start_time)
        
        # Create first arrival event
        arrival_event = SimulationEvent(
            event_time=first_arrival_time,
            event_type=EventType.ARRIVAL,
            connection_id="",  # Will be generated when processed
            connection_data={}
        )
        
        heapq.heappush(self.event_queue, arrival_event)
        print(f"First arrival scheduled at time {first_arrival_time:.2f}")
    
    def schedule_next_arrival(self, current_time: float):
        """
        Schedule the next arrival event
        
        Args:
            current_time: Current simulation time
        """
        next_arrival_time = self.generate_next_arrival_time(current_time)
        
        arrival_event = SimulationEvent(
            event_time=next_arrival_time,
            event_type=EventType.ARRIVAL,
            connection_id="",  # Will be generated when processed
            connection_data={}
        )
        
        heapq.heappush(self.event_queue, arrival_event)
    
    def schedule_departure(self, connection: Connection):
        """
        Schedule departure event for a connection
        
        Args:
            connection: Connection object with holding time
        """
        departure_time = connection.arrival_time + connection.holding_time_hours
        
        departure_event = SimulationEvent(
            event_time=departure_time,
            event_type=EventType.DEPARTURE,
            connection_id=connection.connection_id,
            connection_data={'connection': connection}
        )
        
        heapq.heappush(self.event_queue, departure_event)
        print(f"  Departure scheduled for {connection.connection_id} at time {departure_time:.2f}")
    
    def get_next_event(self) -> SimulationEvent:
        """
        Get the next event from the queue
        
        Returns:
            Next simulation event or None if queue is empty
        """
        if self.event_queue:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.event_time
            return event
        return None
    
    def has_events(self) -> bool:
        """Check if there are more events in the queue"""
        return len(self.event_queue) > 0
    
    def process_arrival_event(self, event: SimulationEvent) -> Connection:
        """
        Process an arrival event and generate connection request
        
        Args:
            event: Arrival event
            
        Returns:
            Generated connection request
        """
        # Generate new connection request
        connection = self.generate_connection_request(event.event_time)
        
        print(f"⬇️  ARRIVAL at time {event.event_time:.2f}: {connection.connection_id} "
              f"({connection.source_node}→{connection.destination_node}, "
              f"{connection.bandwidth_demand_gbps}Gbps, hold={connection.holding_time_hours:.1f})")
        
        # Schedule next arrival
        self.schedule_next_arrival(event.event_time)
        
        return connection
    
    def get_current_time(self) -> float:
        """Get current simulation time"""
        return self.current_time
    
    def get_traffic_statistics(self) -> Dict:
        """Get traffic generation statistics"""
        offered_load = self.arrival_rate_per_hour * self.mean_holding_time
        
        return {
            'arrival_rate_per_hour': self.arrival_rate_per_hour,
            'arrival_rate_per_second': self.arrival_rate_per_second,
            'mean_holding_time': self.mean_holding_time,
            'offered_load_erlangs': offered_load,
            'core_nodes_count': len(self.core_nodes),
            'bandwidth_options_gbps': self.bandwidth_options,
            'bandwidth_probabilities': self.bandwidth_probabilities,
            'connections_generated': self.next_connection_id,
            'current_simulation_time': self.current_time
        }
    
    def reset(self):
        """Reset the traffic generator"""
        self.event_queue.clear()
        self.next_connection_id = 0
        self.current_time = 0.0
        print("Traffic generator reset")

# Example usage
if __name__ == "__main__":
    print("Dynamic Traffic Generator with Poisson Arrivals")
    print("=" * 50)
    
    # Mock network for testing
    class MockNetwork:
        def __init__(self):
            self.nodes = {
                0: type('Node', (), {'add_drop_enabled': True})(),
                1: type('Node', (), {'add_drop_enabled': True})(),
                2: type('Node', (), {'add_drop_enabled': True})(),
                3: type('Node', (), {'add_drop_enabled': True})(),
            }
    
    # Create traffic generator
    network = MockNetwork()
    traffic_gen = DynamicTrafficGenerator(network, arrival_rate_per_hour=50.0)
    
    # Schedule first arrival
    traffic_gen.schedule_first_arrival(0.0)
    
    # Process first few events
    print("\nProcessing first 5 events:")
    for i in range(5):
        if traffic_gen.has_events():
            event = traffic_gen.get_next_event()
            if event.event_type == EventType.ARRIVAL:
                connection = traffic_gen.process_arrival_event(event)
                traffic_gen.schedule_departure(connection)
        else:
            break
    
    # Show statistics
    stats = traffic_gen.get_traffic_statistics()
    print(f"\nTraffic Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")