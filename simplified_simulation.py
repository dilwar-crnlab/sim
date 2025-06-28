#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Event-Driven Dynamic Simulator for MCF EON
File: sim/dynamic_simulator.py
"""

import heapq
import time
from typing import Dict, Set
from events import SimulationEvent, EventType
from traffic_generator import DynamicTrafficGenerator
from xt_nli_rsa import SpectrumAllocationMethod

class SimplifiedDynamicSimulator:
    """
    Event-driven simulator that processes exactly 100,000 connections
    Focuses on blocking probability calculation with realistic dynamics
    """
    
    def __init__(self, mcf_simulator, traffic_generator: DynamicTrafficGenerator):
        """
        Initialize event-driven simulator
        
        Args:
            mcf_simulator: Main MCFEONSimulator instance
            traffic_generator: Traffic generator instance
        """
        self.mcf_sim = mcf_simulator
        self.traffic_gen = traffic_generator
        
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
        
        print(f"Event-Driven Simulator initialized:")
        print(f"  Target connections: {self.total_connections_to_process:,}")
        print(f"  Traffic arrival rate: {traffic_generator.arrival_rate_per_hour:.1f} req/hour")
        print(f"  Expected offered load: {traffic_generator.estimate_offered_load():.1f} Erlangs")
    
    def generate_all_arrival_events(self):
        """
        Pre-generate all 100,000 arrival events with Poisson timing
        This ensures exactly 100,000 connections are processed
        """
        
        print(f"Generating {self.total_connections_to_process:,} arrival events...")
        
        current_time = 0.0
        
        for request_id in range(self.total_connections_to_process):
            # Generate next arrival time using Poisson process
            current_time = self.traffic_gen.generate_next_arrival_time(current_time)
            
            # Generate connection request
            connection = self.traffic_gen.generate_connection_request(current_time, request_id)
            
            # Create arrival event
            arrival_event = SimulationEvent(
                event_time=current_time,
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
        print(f"   Simulation time span: {current_time/3600:.1f} hours")
        
        return current_time / 3600.0  # Return simulation span in hours
    
    def process_arrival_event(self, event: SimulationEvent):
        """
        Process connection arrival and attempt establishment
        
        Args:
            event: Arrival event containing connection data
        """
        
        connection = event.connection_data['connection']
        self.connections_arrived += 1
        
        # Add to connection manager
        self.mcf_sim.connection_manager.connections[connection.connection_id] = connection
        
        # Calculate K-shortest paths
        k_paths = self.mcf_sim.network.calculate_k_shortest_paths(
            connection.source_node, 
            connection.destination_node, 
            k=3
        )
        
        success = False
        if k_paths:
            # Attempt to establish using XT-NLI-A-RSA algorithm
            success = self.mcf_sim.rsa_algorithm.xt_nli_a_rsa_algorithm(
                connection, k_paths, sam=SpectrumAllocationMethod.BSC
            )
        
        if success:
            # Connection established successfully
            self.connections_established += 1
            self.active_connections.add(connection.connection_id)
            
            # Schedule automatic teardown event
            teardown_time = event.event_time + (connection.holding_time_hours * 3600.0)
            teardown_event = SimulationEvent(
                event_time=teardown_time,
                event_type=EventType.CONNECTION_TEARDOWN,
                connection_id=connection.connection_id
            )
            heapq.heappush(self.event_queue, teardown_event)
            
            # Update connection manager with allocation details
            path_length = self._calculate_path_length(k_paths[0])
            
            self.mcf_sim.connection_manager.allocate_connection(
                connection.connection_id,
                k_paths[0],
                connection.resource_allocations,
                connection.end_to_end_gsnr_db or 15.0,
                path_length
            )
            
        else:
            # Connection blocked due to insufficient resources
            self.connections_blocked += 1
            self.mcf_sim.connection_manager.block_connection(
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
            self.mcf_sim.rsa_algorithm.deallocate_connection(connection_id)
            
            # Update connection manager
            self.mcf_sim.connection_manager.terminate_connection(connection_id)
            
            # Update tracking
            self.active_connections.remove(connection_id)
            self.connections_terminated += 1
    
    def _calculate_path_length(self, node_path: list) -> float:
        """Calculate total path length from node sequence"""
        total_length = 0.0
        
        for i in range(len(node_path) - 1):
            link = self.mcf_sim.network.get_link_by_nodes(node_path[i], node_path[i+1])
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
        print(f"RUNNING EVENT-DRIVEN DYNAMIC SIMULATION")
        print(f"{'='*60}")
        
        # Validate traffic generator
        if not self.traffic_gen.validate_configuration():
            raise ValueError("Traffic generator configuration is invalid")
        
        start_time = time.time()
        
        # Generate all arrival events
        simulation_span_hours = self.generate_all_arrival_events()
        
        print(f"\nProcessing events...")
        events_processed = 0
        
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
        
        # Compile comprehensive results
        results = self._compile_results(
            simulation_span_hours, wall_clock_time, events_processed, 
            blocking_probability, establishment_rate, remaining_teardowns
        )
        
        # Print final summary
        self._print_simulation_summary(results)
        
        return results
    
    def _compile_results(self, simulation_span_hours, wall_clock_time, events_processed,
                        blocking_probability, establishment_rate, remaining_teardowns) -> Dict:
        """Compile comprehensive simulation results"""
        
        return {
            'simulation_type': 'event_driven_dynamic',
            'simulation_parameters': {
                'total_connections_processed': self.connections_arrived,
                'simulation_span_hours': simulation_span_hours,
                'wall_clock_time_seconds': wall_clock_time,
                'traffic_parameters': self.traffic_gen.get_traffic_statistics()
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
                'simulation_speedup': (simulation_span_hours * 3600) / wall_clock_time
            },
            'network_state': self.mcf_sim.rsa_algorithm.get_network_state_summary(),
            'algorithm_statistics': self.mcf_sim.rsa_algorithm.get_algorithm_statistics()
        }
    
    def _print_simulation_summary(self, results: Dict):
        """Print comprehensive simulation summary"""
        
        perf = results['performance_metrics']
        conn_stats = results['connection_statistics']
        sim_params = results['simulation_parameters']
        
        print(f"\n✅ Dynamic simulation completed successfully!")
        print(f"{'='*50}")
        print(f"Simulation Results:")
        print(f"  Connections processed: {conn_stats['connections_arrived']:,}")
        print(f"  Connections established: {conn_stats['connections_established']:,}")
        print(f"  Connections blocked: {conn_stats['connections_blocked']:,}")
        print(f"  Connections terminated: {conn_stats['connections_terminated']:,}")
        print(f"  Final active connections: {conn_stats['final_active_connections']}")
        print(f"")
        print(f"Performance Metrics:")
        print(f"  Blocking probability: {perf['blocking_probability']:.6f} ({perf['blocking_probability']*100:.4f}%)")
        print(f"  Establishment rate: {perf['establishment_rate']:.6f}")
        print(f"  Events processed: {perf['events_processed_total']:,}")
        print(f"  Processing rate: {perf['events_per_second']:.0f} events/second")
        print(f"  Simulation speedup: {perf['simulation_speedup']:.0f}x real-time")
        print(f"")
        print(f"Network Utilization:")
        print(f"  Spectrum utilization: {results['network_state']['overall_utilization']:.4f}")
        print(f"  Active connections at end: {results['network_state']['active_connections']}")