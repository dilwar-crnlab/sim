#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic Simulation Integration Example
Shows how to use the modified traffic generator and connection manager
for dynamic MCF EON simulation with Poisson arrivals and exponential holding times
"""

import numpy as np
import time
from typing import Dict, List

# Import the modified components
from traffic_generator import DynamicTrafficGenerator, EventType, SimulationEvent
from connection_manager import DynamicConnectionManager, Connection, ConnectionStatus

class DynamicMCFEONSimulator:
    """
    Dynamic MCF EON Simulator using event-driven simulation
    Integrates traffic generator and connection manager
    """
    
    def __init__(self, network_topology, mcf_config, rsa_algorithm):
        """
        Initialize dynamic simulator
        
        Args:
            network_topology: Network topology object
            mcf_config: MCF configuration
            rsa_algorithm: RSA algorithm instance
        """
        self.network = network_topology
        self.mcf_config = mcf_config
        self.rsa_algorithm = rsa_algorithm
        
        # Initialize dynamic components
        self.traffic_generator = None
        self.connection_manager = DynamicConnectionManager()
        
        # Simulation parameters
        self.max_simulation_time = 1000.0  # time units
        self.current_time = 0.0
        
        print("Dynamic MCF EON Simulator initialized")
    
    def configure_traffic(self, arrival_rate_per_hour: float = 100.0):
        """
        Configure traffic generation parameters
        
        Args:
            arrival_rate_per_hour: Average arrivals per hour
        """
        self.traffic_generator = DynamicTrafficGenerator(
            self.network, arrival_rate_per_hour
        )
        print(f"Traffic configured: {arrival_rate_per_hour} arrivals/hour, mean holding time: 10 time units")
    
    def run_dynamic_simulation(self, max_time: float = 1000.0, 
                             sam_method: str = "BSC") -> Dict:
        """
        Run dynamic event-driven simulation
        
        Args:
            max_time: Maximum simulation time
            sam_method: Spectrum allocation method
            
        Returns:
            Simulation results dictionary
        """
        print(f"\n{'='*60}")
        print(f"STARTING DYNAMIC SIMULATION")
        print(f"{'='*60}")
        print(f"Max time: {max_time} time units")
        print(f"SAM method: {sam_method}")
        
        if self.traffic_generator is None:
            print("❌ Traffic generator not configured!")
            return {}
        
        # Initialize simulation
        self.max_simulation_time = max_time
        self.current_time = 0.0
        
        # Schedule first arrival
        self.traffic_generator.schedule_first_arrival(0.0)
        
        # Event processing counters
        total_events = 0
        arrival_events = 0
        departure_events = 0
        
        start_time = time.time()
        
        # Main event loop
        print(f"\n🔄 Starting event processing loop...")
        
        while (self.traffic_generator.has_events() and 
               self.current_time < self.max_simulation_time):
            
            # Get next event
            event = self.traffic_generator.get_next_event()
            if event is None:
                break
            
            # Update simulation time
            self.current_time = event.event_time
            self.connection_manager.update_simulation_time(self.current_time)
            
            total_events += 1
            
            # Process event based on type
            if event.event_type == EventType.ARRIVAL:
                self._process_arrival_event(event, sam_method)
                arrival_events += 1
                
            elif event.event_type == EventType.DEPARTURE:
                self._process_departure_event(event)
                departure_events += 1
            
            # Progress reporting
            if total_events % 100 == 0:
                print(f"  Processed {total_events} events, time: {self.current_time:.1f}")
        
        simulation_time = time.time() - start_time
        
        # Collect final results
        results = self._collect_simulation_results(
            simulation_time, total_events, arrival_events, departure_events
        )
        
        print(f"\n✅ Simulation completed!")
        print(f"   Simulation time: {simulation_time:.2f} seconds")
        print(f"   Events processed: {total_events}")
        print(f"   Final time: {self.current_time:.2f} time units")
        
        return results
    
    def _process_arrival_event(self, event: SimulationEvent, sam_method: str):
        """Process a connection arrival event"""
        
        # Generate connection request
        connection = self.traffic_generator.process_arrival_event(event)
        
        # Add to connection manager
        self.connection_manager.process_arrival(connection)
        
        # Try to allocate resources using RSA algorithm
        success = self._attempt_resource_allocation(connection, sam_method)
        
        if success:
            # Schedule departure event
            self.traffic_generator.schedule_departure(connection)
        
    def _process_departure_event(self, event: SimulationEvent):
        """Process a connection departure event"""
        
        print(f"⬆️  Processing departure: {event.connection_id}")
        
        # Release resources in connection manager
        departure_success = self.connection_manager.process_departure(event.connection_id)
        
        if departure_success:
            # Release resources in spectrum allocation matrix
            connection = self.connection_manager.get_connection_by_id(event.connection_id)
            if connection and hasattr(self.rsa_algorithm, 'deallocate_connection'):
                # Convert connection_id to int if needed for RSA algorithm
                try:
                    conn_id_int = int(event.connection_id.split('_')[1])
                    self.rsa_algorithm.deallocate_connection(conn_id_int)
                except (ValueError, IndexError):
                    # If conversion fails, try with string ID
                    self.rsa_algorithm.deallocate_connection(event.connection_id)
    
    def _attempt_resource_allocation(self, connection: Connection, sam_method: str) -> bool:
        """
        Attempt to allocate resources for a connection
        
        Args:
            connection: Connection request
            sam_method: Spectrum allocation method
            
        Returns:
            True if allocation successful
        """
        try:
            # Calculate K-shortest paths
            k_paths = self.network.calculate_k_shortest_paths(
                connection.source_node, connection.destination_node, k=3
            )
            
            if not k_paths:
                self.connection_manager.block_connection(
                    connection.connection_id, "No available paths"
                )
                return False
            
            # Apply RSA algorithm
            # Convert SAM method string to enum if needed
            from XT_NLIA_RSA import SpectrumAllocationMethod
            if sam_method == "BSC":
                sam = SpectrumAllocationMethod.BSC
            else:
                sam = SpectrumAllocationMethod.CSB
            
            success = self.rsa_algorithm.xt_nli_a_rsa_algorithm(
                connection, k_paths, sam
            )
            
            if success:
                # Calculate path details for connection manager
                path_length = sum(
                    self.network.links[self.network.get_link_by_nodes(k_paths[0][j], k_paths[0][j+1]).link_id].length_km
                    for j in range(len(k_paths[0]) - 1)
                    if self.network.get_link_by_nodes(k_paths[0][j], k_paths[0][j+1])
                )
                
                # Update connection manager
                self.connection_manager.allocate_connection(
                    connection.connection_id,
                    k_paths[0],  # Node path
                    connection.resource_allocations or [],
                    connection.end_to_end_gsnr_db or 15.0,
                    path_length
                )
                return True
            else:
                self.connection_manager.block_connection(
                    connection.connection_id, "Insufficient spectrum resources"
                )
                return False
                
        except Exception as e:
            print(f"❌ Error in resource allocation: {e}")
            self.connection_manager.block_connection(
                connection.connection_id, f"Allocation error: {str(e)}"
            )
            return False
    
    def _collect_simulation_results(self, simulation_time: float, total_events: int,
                                  arrival_events: int, departure_events: int) -> Dict:
        """Collect comprehensive simulation results"""
        
        # Get statistics from components
        connection_stats = self.connection_manager.get_dynamic_statistics()
        traffic_stats = self.traffic_generator.get_traffic_statistics()
        
        if hasattr(self.rsa_algorithm, 'get_algorithm_statistics'):
            rsa_stats = self.rsa_algorithm.get_algorithm_statistics()
        else:
            rsa_stats = {}
        
        # Compile comprehensive results
        results = {
            'simulation_parameters': {
                'max_simulation_time': self.max_simulation_time,
                'final_simulation_time': self.current_time,
                'simulation_duration_seconds': simulation_time,
                'total_events_processed': total_events,
                'arrival_events': arrival_events,
                'departure_events': departure_events
            },
            'traffic_characteristics': traffic_stats,
            'connection_performance': connection_stats,
            'rsa_performance': rsa_stats,
            'key_metrics': {
                'connection_blocking_probability': connection_stats['blocking_metrics']['connection_blocking_probability'],
                'bandwidth_blocking_probability': connection_stats['blocking_metrics']['bandwidth_blocking_probability'],
                'average_active_connections': connection_stats['connection_counts']['active'],
                'total_throughput_gbps': connection_stats['bandwidth_metrics']['total_terminated_gbps'],
                'events_per_second': total_events / simulation_time if simulation_time > 0 else 0
            }
        }
        
        return results
    
    def print_simulation_summary(self, results: Dict):
        """Print a summary of simulation results"""
        
        print(f"\n{'='*60}")
        print(f"DYNAMIC SIMULATION SUMMARY")
        print(f"{'='*60}")
        
        sim_params = results['simulation_parameters']
        key_metrics = results['key_metrics']
        conn_perf = results['connection_performance']
        
        print(f"Simulation Duration: {sim_params['final_simulation_time']:.1f} time units "
              f"({sim_params['simulation_duration_seconds']:.2f} seconds)")
        print(f"Events Processed: {sim_params['total_events_processed']} "
              f"({sim_params['events_per_second']:.1f} events/sec)")
        
        print(f"\nTraffic Summary:")
        print(f"  Arrivals: {sim_params['arrival_events']}")
        print(f"  Departures: {sim_params['departure_events']}")
        print(f"  Currently Active: {conn_perf['connection_counts']['active']}")
        
        print(f"\nPerformance Metrics:")
        print(f"  Connection Blocking: {key_metrics['connection_blocking_probability']:.2%}")
        print(f"  Bandwidth Blocking: {key_metrics['bandwidth_blocking_probability']:.2%}")
        print(f"  Total Throughput: {key_metrics['total_throughput_gbps']:.1f} Gbps")
        
        if 'service_time_metrics' in conn_perf and conn_perf['service_time_metrics']['total_completed_services'] > 0:
            print(f"  Mean Service Time: {conn_perf['service_time_metrics']['mean_service_time']:.2f} time units")
        
        print(f"\nBandwidth Summary:")
        bw_metrics = conn_perf['bandwidth_metrics']
        print(f"  Total Requested: {bw_metrics['total_requested_gbps']:.1f} Gbps")
        print(f"  Currently Allocated: {bw_metrics['currently_allocated_gbps']:.1f} Gbps")
        print(f"  Total Completed: {bw_metrics['total_terminated_gbps']:.1f} Gbps")

def demonstrate_dynamic_simulation():
    """Demonstrate the dynamic simulation with mock components"""
    
    print("Dynamic MCF EON Simulation Demonstration")
    print("=" * 50)
    
    # Mock network topology
    class MockNetwork:
        def __init__(self):
            self.nodes = {i: type('Node', (), {'add_drop_enabled': True})() for i in range(5)}
            self.links = {i: type('Link', (), {'length_km': 100.0, 'link_id': i})() for i in range(4)}
        
        def calculate_k_shortest_paths(self, source, dest, k=3):
            # Return simple paths for demonstration
            if source != dest:
                return [[source, dest], [source, (source+1)%5, dest]]
            return []
        
        def get_link_by_nodes(self, node1, node2):
            return self.links.get(0)  # Return first link for simplicity
    
    # Mock RSA algorithm
    class MockRSA:
        def __init__(self):
            self.allocation_count = 0
        
        def xt_nli_a_rsa_algorithm(self, connection, k_paths, sam):
            # Simple success/failure based on load
            self.allocation_count += 1
            # Simulate 80% success rate
            success = np.random.random() < 0.8
            
            if success:
                # Mock successful allocation
                connection.resource_allocations = []
                connection.end_to_end_gsnr_db = 15.0 + np.random.normal(0, 2)
            
            return success
        
        def deallocate_connection(self, connection_id):
            self.allocation_count -= 1
    
    # Create simulator with mock components
    network = MockNetwork()
    mcf_config = None  # Not needed for this demo
    rsa_algorithm = MockRSA()
    
    simulator = DynamicMCFEONSimulator(network, mcf_config, rsa_algorithm)
    
    # Configure traffic (50 arrivals per hour, holding time mean = 10)
    simulator.configure_traffic(arrival_rate_per_hour=50.0)
    
    # Run short simulation
    print(f"\nRunning dynamic simulation...")
    results = simulator.run_dynamic_simulation(max_time=100.0, sam_method="BSC")
    
    # Print results
    if results:
        simulator.print_simulation_summary(results)
        
        # Print detailed connection manager report
        print(f"\n{simulator.connection_manager.generate_simulation_report()}")
    
    print(f"\n✅ Dynamic simulation demonstration completed!")

if __name__ == "__main__":
    demonstrate_dynamic_simulation()