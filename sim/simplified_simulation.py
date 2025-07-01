#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simplified Event-Driven Dynamic Simulator for MCF EON
Processes exactly 100,000 connections with exponential arrival and holding times
"""

import heapq
import time
import numpy as np
import random
from typing import Dict, Set

# Import required modules
from events import SimulationEvent, EventType
from xt_nli_rsa import SpectrumAllocationMethod
from network import NetworkTopology
from config import MCF4CoreCLBandConfig
from connection_manager import ConnectionManager, Connection
from gsnr_calculator import GSNRCalculator
from xt_nli_rsa import XT_NLI_A_RSA_Algorithm

class SimplifiedDynamicSimulator:
    """
    Simplified event-driven simulator for MCF EON
    Uses exponential inter-arrival and holding times
    """
    
    def __init__(self, network: NetworkTopology, mcf_config: MCF4CoreCLBandConfig, 
                 mean_inter_arrival_time: float = 1.0, mean_holding_time: float = 25.0):
        """
        Initialize simplified dynamic simulator
        
        Args:
            network: Network topology
            mcf_config: MCF configuration  
            mean_inter_arrival_time: Mean time between arrivals
            mean_holding_time: Mean service holding time
        """
        self.network = network
        self.mcf_config = mcf_config
        self.mean_service_inter_arrival_time = mean_inter_arrival_time
        self.mean_service_holding_time = mean_holding_time
        
        # Initialize components
        self.connection_manager = ConnectionManager()
        
        # Initialize GSNR calculator
        band_config = {
            'c_band': mcf_config.band_configs['C'].__dict__,
            'l_band': mcf_config.band_configs['L'].__dict__
        }
        self.gsnr_calculator = GSNRCalculator(mcf_config, band_config)
        
        # Initialize RSA algorithm
        self.rsa_algorithm = XT_NLI_A_RSA_Algorithm(network, mcf_config, self.gsnr_calculator)
        
        # Get core nodes for traffic generation
        self.core_nodes = [node_id for node_id, node in self.network.nodes.items() 
                          if node.add_drop_enabled]
        
        # Traffic demand characteristics
        self.bandwidth_options = [100, 200, 300, 400, 500, 600]  # Gbps
        self.bandwidth_probabilities = [0.3, 0.25, 0.2, 0.15, 0.07, 0.03]  # Favor lower BW
        
        # Event queue (priority queue)
        self.event_queue = []
        self.current_time = 0.0
        
        # Simulation parameters
        self.total_connections_to_process = 100000
        
        # Connection tracking counters
        self.connections_arrived = 0
        self.connections_established = 0
        self.connections_blocked = 0
        self.connections_terminated = 0
        
        # Active connections set
        self.active_connections: Set[str] = set()
        
        # Calculate offered load in Erlangs (arrival_rate × holding_time)
        arrival_rate = 1.0 / self.mean_service_inter_arrival_time
        offered_load = arrival_rate * self.mean_service_holding_time
        
        print(f"Simplified Dynamic MCF EON Simulator initialized:")
        print(f"  Target connections: {self.total_connections_to_process:,}")
        print(f"  Mean inter-arrival time: {mean_inter_arrival_time:.2f}")
        print(f"  Mean holding time: {mean_holding_time:.2f}")
        print(f"  Offered load: {offered_load:.2f} Erlangs")
        print(f"  MCF: {mcf_config.mcf_params.num_cores} cores, {len(mcf_config.channels)} channels")
    
    def generate_connection_request(self, arrival_time: float, request_id: int) -> Connection:
        """
        Generate a single connection request with exponential holding time
        
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
        
        # Exponential holding time: ht = expovariate(1 / mean_service_holding_time)
        holding_time = random.expovariate(1.0 / self.mean_service_holding_time)
        
        # Create connection object
        connection = Connection(
            connection_id=str(request_id),
            source_node=source,
            destination_node=dest,
            bandwidth_demand_gbps=bandwidth_gbps,
            holding_time_hours=holding_time,  # Store as time units (not hours)
            arrival_time=arrival_time,
            priority=1,
            service_type="best_effort"
        )
        
        return connection
    
    def generate_all_arrival_events(self):
        """
        Pre-generate all 100,000 arrival events with exponential inter-arrival times
        """
        
        print(f"Generating {self.total_connections_to_process:,} arrival events...")
        
        self.current_time = 0.0
        
        for request_id in range(self.total_connections_to_process):
            # Generate next arrival time: at = current_time + expovariate(1 / mean_inter_arrival_time)
            inter_arrival_time = random.expovariate(1.0 / self.mean_service_inter_arrival_time)
            self.current_time += inter_arrival_time
            
            # Generate connection request with exponential holding time
            connection = self.generate_connection_request(self.current_time, request_id)
            
            # Create arrival event
            arrival_event = SimulationEvent(
                event_time=self.current_time,
                event_type=EventType.CONNECTION_ARRIVAL,
                connection_id=connection.connection_id,
                connection_data={'connection': connection}
            )
            
            # Add to event queue
            heapq.heappush(self.event_queue, arrival_event)
            
            # Progress indicator
            if (request_id + 1) % 10000 == 0:
                print(f"  Generated {request_id + 1:,} events...")
        
        print(f"✅ All {self.total_connections_to_process:,} arrival events generated")
        print(f"   Simulation time span: {self.current_time:.1f} time units")
        
        return self.current_time  # Return simulation span in time units
    
    def process_arrival_event(self, event: SimulationEvent):
        """
        Process connection arrival and attempt establishment
        
        Args:
            event: Arrival event containing connection data
        """
        
        connection = event.connection_data['connection']
        self.connections_arrived += 1
        
        # Add to connection manager
        self.connection_manager.connections[connection.connection_id] = connection
        
        # Calculate K-shortest paths
        k_paths = self.network.calculate_k_shortest_paths(
            connection.source_node, 
            connection.destination_node, 
            k=3
        )
        
        success = False
        if k_paths:
            # Attempt to establish using XT-NLI-A-RSA algorithm
            success = self.rsa_algorithm.xt_nli_a_rsa_algorithm(
                connection, k_paths, sam=SpectrumAllocationMethod.BSC
            )
        
        if success:
            # Connection established successfully
            self.connections_established += 1
            self.active_connections.add(connection.connection_id)
            
            # Schedule automatic teardown event (holding_time is already in time units)
            teardown_time = event.event_time + connection.holding_time_hours
            teardown_event = SimulationEvent(
                event_time=teardown_time,
                event_type=EventType.CONNECTION_TEARDOWN,
                connection_id=connection.connection_id
            )
            heapq.heappush(self.event_queue, teardown_event)
            
            # Update connection manager with allocation details
            path_length = self._calculate_path_length(k_paths[0])
            
            self.connection_manager.allocate_connection(
                connection.connection_id,
                k_paths[0],
                connection.resource_allocations,
                connection.end_to_end_gsnr_db or 15.0,
                path_length
            )
            
        else:
            # Connection blocked due to insufficient resources
            self.connections_blocked += 1
            self.connection_manager.block_connection(
                connection.connection_id, 
                "Insufficient spectrum resources"
            )
    
    def process_teardown_event(self, event: SimulationEvent):
        """
        Process connection teardown and resource release
        
        Args:
            event: Teardown event
        """
        
        connection_id = event.connection_id
        
        if connection_id in self.active_connections:
            # Deallocate spectrum resources
            self.rsa_algorithm.deallocate_connection(connection_id)
            
            # Update connection manager
            self.connection_manager.terminate_connection(connection_id)
            
            # Update tracking
            self.active_connections.remove(connection_id)
            self.connections_terminated += 1
    
    def _calculate_path_length(self, node_path: list) -> float:
        """Calculate total path length from node sequence"""
        total_length = 0.0
        
        for i in range(len(node_path) - 1):
            link = self.network.get_link_by_nodes(node_path[i], node_path[i+1])
            if link:
                total_length += link.length_km
        
        return total_length
    
    def run_simulation(self) -> Dict:
        """
        Run the complete event-driven simulation
        
        Returns:
            Comprehensive simulation results
        """
        
        print(f"\n{'='*60}")
        print(f"RUNNING SIMPLIFIED DYNAMIC MCF EON SIMULATION")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Generate all arrival events
        simulation_span_time_units = self.generate_all_arrival_events()
        
        print(f"\nProcessing events...")
        events_processed = 0
        self.current_time = 0.0  # Reset current time for event processing
        
        # Main event processing loop
        while (self.event_queue and 
               self.connections_arrived < self.total_connections_to_process):
            
            # Get next event from priority queue
            event = heapq.heappop(self.event_queue)
            self.current_time = event.event_time
            
            # Process event based on type
            if event.event_type == EventType.CONNECTION_ARRIVAL:
                self.process_arrival_event(event)
            elif event.event_type == EventType.CONNECTION_TEARDOWN:
                self.process_teardown_event(event)
            
            events_processed += 1
            
            # Progress indicator every 5000 events
            if events_processed % 5000 == 0:
                progress = (self.connections_arrived / self.total_connections_to_process) * 100
                current_blocking = self.connections_blocked / max(self.connections_arrived, 1)
                
                print(f"  Progress: {progress:.1f}% | "
                      f"Processed: {self.connections_arrived:,} | "
                      f"Active: {len(self.active_connections)} | "
                      f"Blocking: {current_blocking:.3f}")
        
        # Continue processing teardown events for remaining active connections
        remaining_teardowns = 0
        while self.event_queue:
            event = heapq.heappop(self.event_queue)
            if event.event_type == EventType.CONNECTION_TEARDOWN:
                self.process_teardown_event(event)
                remaining_teardowns += 1
                events_processed += 1
        
        wall_clock_time = time.time() - start_time
        
        # Calculate final metrics
        blocking_probability = self.connections_blocked / self.connections_arrived
        establishment_rate = self.connections_established / self.connections_arrived
        
        # Calculate offered load
        arrival_rate = 1.0 / self.mean_service_inter_arrival_time
        offered_load = arrival_rate * self.mean_service_holding_time
        
        # Compile comprehensive results
        results = {
            'simulation_type': 'simplified_dynamic_mcf',
            'simulation_parameters': {
                'total_connections_processed': self.connections_arrived,
                'simulation_span_time_units': simulation_span_time_units,
                'wall_clock_time_seconds': wall_clock_time,
                'mean_inter_arrival_time': self.mean_service_inter_arrival_time,
                'mean_holding_time': self.mean_service_holding_time,
                'offered_load_erlangs': offered_load,
                'mcf_type': '4-core C+L band',
                'num_cores': self.mcf_config.mcf_params.num_cores,
                'core_pitch_um': self.mcf_config.mcf_params.core_pitch_um,
                'total_channels': len(self.mcf_config.channels)
            },
            'connection_statistics': {
                'connections_arrived': self.connections_arrived,
                'connections_established': self.connections_established,
                'connections_blocked': self.connections_blocked,
                'connections_terminated': self.connections_terminated,
                'final_active_connections': len(self.active_connections),
                'remaining_teardowns_processed': remaining_teardowns
            },
            'performance_metrics': {
                'blocking_probability': blocking_probability,
                'establishment_rate': establishment_rate,
                'termination_rate': self.connections_terminated / max(self.connections_established, 1),
                'events_processed_total': events_processed,
                'events_per_second': events_processed / wall_clock_time,
                'simulation_speedup': simulation_span_time_units / wall_clock_time
            },
            'network_state': self.rsa_algorithm.get_network_state_summary(),
            'algorithm_statistics': self.rsa_algorithm.get_algorithm_statistics(),
            'mcf_specific_metrics': {
                'spectrum_utilization_per_core': self.rsa_algorithm.spectrum_allocation.get_utilization_per_core(),
                'icxt_aware_allocation': True,
                'nli_aware_allocation': True,
                'multi_band_operation': True
            }
        }
        
        # Print simulation summary
        self._print_simulation_summary(results)
        
        return results
    
    def _print_simulation_summary(self, results: Dict):
        """Print comprehensive simulation summary"""
        
        perf = results['performance_metrics']
        conn_stats = results['connection_statistics']
        sim_params = results['simulation_parameters']
        
        print(f"\n✅ Simplified MCF EON simulation completed!")
        print(f"{'='*60}")
        print(f"Simulation Parameters:")
        print(f"  MCF Type: {sim_params['num_cores']}-core C+L band")
        print(f"  Core Pitch: {sim_params['core_pitch_um']} μm")
        print(f"  Total Channels: {sim_params['total_channels']}")
        print(f"  Mean inter-arrival time: {sim_params['mean_inter_arrival_time']:.2f}")
        print(f"  Mean holding time: {sim_params['mean_holding_time']:.2f}")
        print(f"  Offered load: {sim_params['offered_load_erlangs']:.2f} Erlangs")
        print(f"  Simulation span: {sim_params['simulation_span_time_units']:.1f} time units")
        print(f"")
        print(f"Connection Results:")
        print(f"  Connections processed: {conn_stats['connections_arrived']:,}")
        print(f"  Connections established: {conn_stats['connections_established']:,}")
        print(f"  Connections blocked: {conn_stats['connections_blocked']:,}")
        print(f"  Connections terminated: {conn_stats['connections_terminated']:,}")
        print(f"  Final active connections: {conn_stats['final_active_connections']}")
        print(f"")
        print(f"Key Performance Metrics:")
        print(f"  🎯 Blocking probability: {perf['blocking_probability']:.6f} ({perf['blocking_probability']*100:.4f}%)")
        print(f"  Establishment rate: {perf['establishment_rate']:.6f}")
        print(f"  Events processed: {perf['events_processed_total']:,}")
        print(f"  Processing rate: {perf['events_per_second']:.0f} events/second")
        print(f"  Simulation speedup: {perf['simulation_speedup']:.0f}x real-time")
        print(f"")
        print(f"MCF Network Utilization:")
        core_utils = results['mcf_specific_metrics']['spectrum_utilization_per_core']
        for core_idx, utilization in core_utils.items():
            print(f"  Core {core_idx}: {utilization:.4f} ({utilization*100:.2f}%)")
        print(f"  Overall utilization: {results['network_state']['overall_utilization']:.4f}")
        print(f"")
        print(f"Algorithm Performance:")
        alg_stats = results['algorithm_statistics']
        print(f"  Success rate: {alg_stats['success_rate']:.3f}")
        print(f"  Single-chunk allocations: {alg_stats['single_chunk_rate']:.3f}")
        print(f"  Sliced allocations: {alg_stats['sliced_allocation_rate']:.3f}")
        print(f"  Avg computation time: {alg_stats['average_computation_time_ms']:.2f} ms")

def create_simplified_simulator(mean_inter_arrival_time: float = 1.0, 
                               mean_holding_time: float = 25.0) -> SimplifiedDynamicSimulator:
    """
    Create a simplified dynamic simulator with MCF configuration
    
    Args:
        mean_inter_arrival_time: Mean time between connection arrivals
        mean_holding_time: Mean service holding time
        
    Returns:
        Configured SimplifiedDynamicSimulator instance
    """
    
    print("Setting up Simplified Dynamic MCF EON Simulator...")
    
    # Create MCF configuration
    print("  Initializing 4-core C+L band MCF configuration...")
    mcf_config = MCF4CoreCLBandConfig()
    
    # Create network topology
    print("  Creating network topology...")
    network = NetworkTopology()
    network.create_us_backbone_network()
    
    # Create simulator
    print("  Initializing simplified simulator...")
    simulator = SimplifiedDynamicSimulator(network, mcf_config, 
                                          mean_inter_arrival_time, mean_holding_time)
    
    print("✅ Simplified simulator ready!")
    return simulator

def main():
    """Main function for simplified dynamic MCF EON simulation"""
    
    print("Simplified Dynamic MCF EON Simulator")
    print("=" * 50)
    print("Event-driven simulation with exponential inter-arrival and holding times")
    print("Processes exactly 100,000 connections")
    print("Uses: at = current_time + expovariate(1/mean_inter_arrival)")
    print("      ht = expovariate(1/mean_holding_time)")
    print("")
    
    # Get simulation parameters
    try:
        mean_inter_arrival = float(input("Enter mean inter-arrival time (default 1.0): ") or "1.0")
        mean_holding = float(input("Enter mean holding time (default 25.0): ") or "25.0")
    except ValueError:
        print("Invalid input. Using defaults: inter-arrival=1.0, holding=25.0")
        mean_inter_arrival = 1.0
        mean_holding = 25.0
    
    # Create and run simulator
    simulator = create_simplified_simulator(mean_inter_arrival, mean_holding)
    results = simulator.run_simulation()
    
    print(f"\n🎉 Simplified MCF EON simulation completed!")
    print(f"Blocking probability: {results['performance_metrics']['blocking_probability']:.6f}")
    print(f"Results demonstrate ICXT and NLI aware spectrum allocation")
    print(f"for 4-core C+L band Multi-Core Fiber EON")

if __name__ == "__main__":
    main()