#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete MCF EON Integration with All Fixes Applied
Main integration file to ensure all components work together for 4-core C+L band EON
"""

import numpy as np
from typing import Dict, List
import time

# Import all fixed components
from config import MCF4CoreCLBandConfig
from network import NetworkTopology, Link, Node
from connection_manager import ConnectionManager, Connection
from gsnr_calculator import GSNRCalculator
from xt_nli_rsa import XT_NLI_A_RSA_Algorithm, SpectrumAllocationMethod




class MCFEONSimulator:
    """
    Complete MCF EON Simulator with all fixes applied
    Implements the 4-core C+L band ICXT and NLI aware spectrum allocation
    """
    
    def __init__(self):
        """Initialize the complete simulator"""
        print("="*80)
        print("MCF EON SIMULATOR - 4-CORE C+L BAND")
        print("="*80)
        
        #  Step 1: Initialize 4-core MCF configuration
        print("1. Initializing 4-core C+L band MCF configuration...")
        self.mcf_config = MCF4CoreCLBandConfig()
        
        #  Validate configuration
        config_summary = self.mcf_config.get_configuration_summary()
        print(f"   MCF Type: {config_summary['mcf_type']}")
        print(f"   Cores: {self.mcf_config.mcf_params.num_cores}")
        print(f"   Core pitch: {self.mcf_config.mcf_params.core_pitch_um} μm")
        print(f"   Total channels: {config_summary['total_channels']}")
        print(f"   C-band: {config_summary['c_band_channels']} channels")
        print(f"   L-band: {config_summary['l_band_channels']} channels")
        
        #  Step 2: Initialize network topology
        print("\n2. Creating network topology...")
        self.network = NetworkTopology()
        self.network.create_us_backbone_network()
        
        net_stats = self.network.get_network_statistics()
        print(f"   Nodes: {net_stats['num_nodes']}")
        print(f"   Links: {net_stats['num_links']}")
        print(f"   Total fiber: {net_stats['total_fiber_km']:.0f} km")
        
        #  Step 3: Initialize GSNR calculator with MCF config
        print("\n3. Initializing GSNR calculator...")
        band_config = {
            'c_band': self.mcf_config.band_configs['C'].__dict__,
            'l_band': self.mcf_config.band_configs['L'].__dict__
        }
        self.gsnr_calculator = GSNRCalculator(self.mcf_config, band_config)
        
        #  Step 4: Initialize XT-NLI-A-RSA algorithm
        print("\n4. Initializing XT-NLI-A-RSA algorithm...")
        self.rsa_algorithm = XT_NLI_A_RSA_Algorithm(
            self.network, self.mcf_config, self.gsnr_calculator
        )
        
        #  Step 5: Initialize connection manager
        print("\n5. Initializing connection manager...")
        self.connection_manager = ConnectionManager()
        
        print("\n MCF EON Simulator initialization complete!")
        print("   Ready for ICXT and NLI aware spectrum allocation")
    
    def run_simulation_scenario(self, num_requests: int = 100, 
                              sam: SpectrumAllocationMethod = SpectrumAllocationMethod.BSC) -> Dict:
        """
        Run simulation scenario with specified parameters
        
        Args:
            num_requests: Number of connection requests to generate
            sam: Spectrum Allocation Method (BSC or CSB)
            
        Returns:
            Simulation results dictionary
        """
        print(f"\n{'='*60}")
        print(f"RUNNING SIMULATION SCENARIO")
        print(f"{'='*60}")
        print(f"Requests: {num_requests}")
        print(f"SAM: {sam.value}")
        
        start_time = time.time()
        
        #  Generate traffic demands
        print(f"\nGenerating {num_requests} connection requests...")
        connections = self._generate_traffic_demands(num_requests)
        
        #  Process each connection request
        processed = 0
        for i, connection in enumerate(connections):
            if (i + 1) % 10 == 0:
                print(f"  Processing request {i + 1}/{num_requests}")
            
            # Calculate K-shortest paths
            k_paths = self.network.calculate_k_shortest_paths(
                connection.source_node, connection.destination_node, k=3
            )
            
            if not k_paths:
                self.connection_manager.block_connection(
                    connection.connection_id, "No available paths"
                )
                continue
            
            # Apply XT-NLI-A-RSA algorithm
            success = self.rsa_algorithm.xt_nli_a_rsa_algorithm(
                connection, k_paths, sam
            )
            
            if success:
                # Calculate path details
                path_length = sum(
                    self.network.links[self.network.get_link_by_nodes(k_paths[0][j], k_paths[0][j+1]).link_id].length_km
                    for j in range(len(k_paths[0]) - 1)
                    if self.network.get_link_by_nodes(k_paths[0][j], k_paths[0][j+1])
                )
                
                # Allocate in connection manager
                self.connection_manager.allocate_connection(
                    connection.connection_id,
                    k_paths[0],
                    connection.resource_allocations,
                    connection.end_to_end_gsnr_db or 15.0,  # Default GSNR
                    path_length
                )
            else:
                self.connection_manager.block_connection(
                    connection.connection_id, "Insufficient spectrum resources"
                )
            
            processed += 1
        
        simulation_time = time.time() - start_time
        
        #  Collect results
        results = self._collect_simulation_results(simulation_time, sam)
        
        print(f"\n Simulation completed in {simulation_time:.2f} seconds")
        print(f"   Processed: {processed}/{num_requests} requests")
        print(f"   Success rate: {results['performance_metrics']['success_rate']:.2%}")
        print(f"   Blocking rate: {results['performance_metrics']['blocking_rate']:.2%}")
        
        return results
    
    def _generate_traffic_demands(self, num_requests: int) -> List[Connection]:
        """Generate realistic traffic demands"""
        connections = []
        
        # Get core nodes (nodes with add/drop capability)
        core_nodes = [node_id for node_id, node in self.network.nodes.items() 
                     if node.add_drop_enabled]
        
        np.random.seed(42)  # For reproducible results
        
        for i in range(num_requests):
            # Random source and destination
            source, dest = np.random.choice(core_nodes, 2, replace=False)
            
            # Random bandwidth demand (100-600 Gbps in 100 Gbps steps)
            bandwidth_gbps = np.random.choice([100, 200, 300, 400, 500, 600])
            
            # Random holding time (1-5 hours)
            holding_time = np.random.uniform(1.0, 5.0)
            
            connection = self.connection_manager.add_connection_request(
                source_node=source,
                destination_node=dest,
                bandwidth_demand_gbps=bandwidth_gbps,
                holding_time_hours=holding_time,
                connection_id=f"conn_{i:04d}"
            )
            
            connections.append(connection)
        
        return connections
    
    def _collect_simulation_results(self, simulation_time: float, 
                                  sam: SpectrumAllocationMethod) -> Dict:
        """Collect comprehensive simulation results"""
        
        # Connection manager statistics
        cm_stats = self.connection_manager.get_statistics()
        
        # Algorithm statistics  
        alg_stats = self.rsa_algorithm.get_algorithm_statistics()
        
        # Network state
        network_state = self.rsa_algorithm.get_network_state_summary()
        
        results = {
            'simulation_parameters': {
                'mcf_type': '4-core C+L band',
                'spectrum_allocation_method': sam.value,
                'num_cores': self.mcf_config.mcf_params.num_cores,
                'core_pitch_um': self.mcf_config.mcf_params.core_pitch_um,
                'total_channels': len(self.mcf_config.channels),
                'simulation_time_s': simulation_time
            },
            'performance_metrics': {
                'total_requests': cm_stats['total_requests'],
                'successful_allocations': cm_stats['active_connections'],
                'blocked_requests': cm_stats['blocked_connections'],
                'success_rate': 1 - cm_stats['blocking_probability'],
                'blocking_rate': cm_stats['blocking_probability'],
                'spectrum_utilization': network_state['overall_utilization'],
                'average_gsnr_db': cm_stats['average_metrics']['gsnr_db'],
                'average_path_length_km': cm_stats['average_metrics']['path_length_km']
            },
            'algorithm_performance': alg_stats,
            'network_state': network_state,
            'connection_statistics': cm_stats,
            'modulation_format_usage': cm_stats['modulation_format_distribution']
        }
        
        return results
    
    def compare_sam_methods(self, num_requests: int = 100) -> Dict:
        """Compare CSB vs BSC spectrum allocation methods"""
        print(f"\n{'='*60}")
        print(f"COMPARING SPECTRUM ALLOCATION METHODS")
        print(f"{'='*60}")
        
        results_comparison = {}
        
        for sam in [SpectrumAllocationMethod.CSB, SpectrumAllocationMethod.BSC]:
            print(f"\n--- Testing {sam.value} ---")
            
            # Reset for fair comparison
            self.rsa_algorithm = XT_NLI_A_RSA_Algorithm(
                self.network, self.mcf_config, self.gsnr_calculator
            )
            self.connection_manager = ConnectionManager()
            
            # Run simulation
            results = self.run_simulation_scenario(num_requests, sam)
            results_comparison[sam.value] = results
        
        #  Generate comparison report
        self._generate_comparison_report(results_comparison)
        
        return results_comparison
    
    def _generate_comparison_report(self, results_comparison: Dict):
        """Generate comparison report between SAM methods"""
        print(f"\n{'='*60}")
        print(f"SPECTRUM ALLOCATION METHOD COMPARISON")
        print(f"{'='*60}")
        
        for method, results in results_comparison.items():
            perf = results['performance_metrics']
            print(f"\n{method}:")
            print(f"  Success Rate: {perf['success_rate']:.2%}")
            print(f"  Blocking Rate: {perf['blocking_rate']:.2%}")
            print(f"  Spectrum Utilization: {perf['spectrum_utilization']:.2%}")
            print(f"  Average GSNR: {perf['average_gsnr_db']:.2f} dB")
        
        #  Performance comparison
        if len(results_comparison) == 2:
            methods = list(results_comparison.keys())
            bsc_results = results_comparison.get('Band-Spectrum-Core', {})
            csb_results = results_comparison.get('Core-Spectrum-Band', {})
            
            if bsc_results and csb_results:
                bsc_perf = bsc_results['performance_metrics']
                csb_perf = csb_results['performance_metrics']
                
                print(f"\nPerformance Improvement (BSC vs CSB):")
                success_improvement = (bsc_perf['success_rate'] - csb_perf['success_rate']) / csb_perf['success_rate'] * 100
                print(f"  Success Rate: {success_improvement:+.1f}%")
                
                blocking_improvement = (csb_perf['blocking_rate'] - bsc_perf['blocking_rate']) / csb_perf['blocking_rate'] * 100
                print(f"  Blocking Reduction: {blocking_improvement:+.1f}%")

def main():
    """Main simulation entry point"""
    # Initialize simulator
    simulator = MCFEONSimulator()
    
    # Run comparison between SAM methods
    results = simulator.compare_sam_methods(num_requests=50)
    
    print(f"\n MCF EON Simulation Complete!")
    print(f"Results show the performance of ICXT and NLI aware spectrum allocation")
    print(f"for 4-core C+L band Multi-Core Fiber EON")

if __name__ == "__main__":
    main()