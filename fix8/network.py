#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MCF EON Core - Network Management
Handles network topology, paths, and link management for Multi-Core Fiber EON
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
import json

@dataclass
class Link:
    """Link between two nodes with proper span division"""
    link_id: int
    source_node: int
    destination_node: int
    length_km: float
    num_spans: int
    span_lengths_km: List[float]
    fiber_type: str = "SSMF"
    
    def __post_init__(self):
        """Automatically divide long links into realistic spans"""
        if not self.span_lengths_km:
            # Standard span length in real networks
            standard_span_length = 80.0  # km
            min_last_span_length = 10.0  # km (avoid very short spans)
            
            if self.length_km <= standard_span_length:
                # Short link: single span
                self.num_spans = 1
                self.span_lengths_km = [self.length_km]
            else:
                # Long link: use standard spans + remainder
                num_full_spans = int(self.length_km // standard_span_length)
                remainder = self.length_km % standard_span_length
                
                if remainder < min_last_span_length and num_full_spans > 0:
                    # If remainder is too short, distribute it to avoid very short spans
                    # Reduce number of full spans by 1 and make last span longer
                    num_full_spans -= 1
                    last_span_length = standard_span_length + remainder
                    
                    self.span_lengths_km = [standard_span_length] * num_full_spans + [last_span_length]
                    self.num_spans = num_full_spans + 1
                else:
                    # Normal case: standard spans + remainder
                    self.span_lengths_km = [standard_span_length] * num_full_spans
                    if remainder > 0:
                        self.span_lengths_km.append(remainder)
                        self.num_spans = num_full_spans + 1
                    else:
                        self.num_spans = num_full_spans
        
        # Validate spans
        total_span_length = sum(self.span_lengths_km)
        if abs(total_span_length - self.length_km) > 0.1:
            print(f"Warning: Link {self.link_id} span total {total_span_length:.1f} km "
                f"doesn't match link length {self.length_km:.1f} km")
        
        # Update num_spans to match actual spans
        self.num_spans = len(self.span_lengths_km)
        
        print(f"Link {self.link_id}: {self.length_km:.1f} km → {self.num_spans} spans: "
            f"{[f'{s:.1f}' for s in self.span_lengths_km]} km")

@dataclass
class Node:
    """Network node with ROADM capabilities"""
    node_id: int
    name: str
    node_type: str  # "core", "metro", "access"
    location: Tuple[float, float]  # (latitude, longitude)
    add_drop_enabled: bool = True
    roadm_insertion_loss_db: float = 5.5
    wss_insertion_loss_db: float = 4.5

class NetworkTopology:
    """Network topology management for MCF EON"""
    
    def __init__(self):
        self.graph = nx.Graph()
        self.nodes: Dict[int, Node] = {}
        self.links: Dict[int, Link] = {}
        self.node_counter = 0
        self.link_counter = 0
        
    def add_node(self, name: str, node_type: str = "core", 
                 location: Tuple[float, float] = (0.0, 0.0),
                 add_drop_enabled: bool = True) -> int:
        """Add a node to the network"""
        node_id = self.node_counter
        
        node = Node(
            node_id=node_id,
            name=name,
            node_type=node_type,
            location=location,
            add_drop_enabled=add_drop_enabled
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **node.__dict__)
        self.node_counter += 1
        
        return node_id
    
    def add_link(self, source_node: int, dest_node: int, length_km: float,
                 num_spans: int = 1, span_lengths_km: List[float] = None,
                 fiber_type: str = "SSMF") -> int:
        """Add a bidirectional link between two nodes"""
        if source_node not in self.nodes or dest_node not in self.nodes:
            raise ValueError("Both nodes must exist before adding link")
        
        link_id = self.link_counter
        
        link = Link(
            link_id=link_id,
            source_node=source_node,
            destination_node=dest_node,
            length_km=length_km,
            num_spans=num_spans,
            span_lengths_km=span_lengths_km or [],
            fiber_type=fiber_type
        )
        
        self.links[link_id] = link
        self.graph.add_edge(source_node, dest_node, 
                           link_id=link_id, 
                           length_km=length_km,
                           weight=length_km)
        self.link_counter += 1
        
        return link_id
    
    def get_link_by_nodes(self, node1: int, node2: int) -> Optional[Link]:
        """Get link between two nodes"""
        if self.graph.has_edge(node1, node2):
            link_id = self.graph[node1][node2]['link_id']
            return self.links[link_id]
        return None
    
    def calculate_k_shortest_paths(self, source: int, destination: int, 
                                 k: int = 3) -> List[List[int]]:
        """Calculate K-shortest paths between source and destination"""
        try:
            paths = list(nx.shortest_simple_paths(self.graph, source, destination, weight='length_km'))
            return paths[:k]
        except nx.NetworkXNoPath:
            return []
    
    def get_path_length(self, path: List[int]) -> float:
        """Calculate total length of a path"""
        total_length = 0.0
        for i in range(len(path) - 1):
            link = self.get_link_by_nodes(path[i], path[i + 1])
            if link:
                total_length += link.length_km
        return total_length
    
    def get_path_links(self, path: List[int]) -> List[Link]:
        """Get all links in a path"""
        links = []
        for i in range(len(path) - 1):
            link = self.get_link_by_nodes(path[i], path[i + 1])
            if link:
                links.append(link)
        return links
    
    def create_us_backbone_network(self):
        """Create simplified US backbone network topology"""
        # Major US cities with approximate coordinates
        cities = [
            (0, "Seattle", (47.6062, -122.3321)),
            (1, "Portland", (45.5152, -122.6784)),
            (2, "San Francisco", (37.7749, -122.4194)),
            (3, "Los Angeles", (34.0522, -118.2437)),
            (4, "Phoenix", (33.4484, -112.0740)),
            (5, "Denver", (39.7392, -104.9903)),
            (6, "Dallas", (32.7767, -96.7970)),
            (7, "Houston", (29.7604, -95.3698)),
            (8, "Chicago", (41.8781, -87.6298)),
            (9, "Minneapolis", (44.9778, -93.2650)),
            (10, "Atlanta", (33.7490, -84.3880)),
            (11, "Miami", (25.7617, -80.1918)),
            (12, "Washington DC", (38.9072, -77.0369)),
            (13, "New York", (40.7128, -74.0060)),
            (14, "Boston", (42.3601, -71.0589))
        ]
        
        # Add nodes
        for node_id, name, location in cities:
            self.add_node(name, "core", location)
        
        # Add links with realistic distances (approximate)
        links = [
            (0, 1, 280),    # Seattle - Portland
            (0, 8, 2800),   # Seattle - Chicago
            (1, 2, 950),    # Portland - San Francisco
            (2, 3, 560),    # San Francisco - Los Angeles
            (3, 4, 570),    # Los Angeles - Phoenix
            (3, 6, 1960),   # Los Angeles - Dallas
            (4, 5, 880),    # Phoenix - Denver
            (4, 6, 1280),   # Phoenix - Dallas
            (5, 8, 1480),   # Denver - Chicago
            (5, 6, 1120),   # Denver - Dallas
            (6, 7, 360),    # Dallas - Houston
            (6, 8, 1290),   # Dallas - Chicago
            (7, 10, 1160),  # Houston - Atlanta
            (8, 9, 570),    # Chicago - Minneapolis
            (8, 10, 950),   # Chicago - Atlanta
            (8, 12, 1090),  # Chicago - Washington DC
            (10, 11, 970),  # Atlanta - Miami
            (10, 12, 640),  # Atlanta - Washington DC
            (12, 13, 360),  # Washington DC - New York
            (13, 14, 310),  # New York - Boston
        ]
        
        for source, dest, distance in links:
            # Calculate number of spans (every ~80-100 km)
            num_spans = max(1, int(distance / 80))
            self.add_link(source, dest, distance, num_spans)
    
    def get_network_statistics(self) -> Dict:
        """Get network topology statistics"""
        return {
            'num_nodes': len(self.nodes),
            'num_links': len(self.links),
            'total_fiber_km': sum(link.length_km for link in self.links.values()),
            'average_link_length_km': np.mean([link.length_km for link in self.links.values()]),
            'network_diameter_km': self._calculate_network_diameter(),
            'average_node_degree': 2 * len(self.links) / len(self.nodes),
            'core_nodes': len([n for n in self.nodes.values() if n.node_type == "core"])
        }
    
    def _calculate_network_diameter(self) -> float:
        """Calculate network diameter (longest shortest path)"""
        max_distance = 0.0
        for source in self.nodes.keys():
            for dest in self.nodes.keys():
                if source != dest:
                    try:
                        path = nx.shortest_path(self.graph, source, dest, weight='length_km')
                        distance = self.get_path_length(path)
                        max_distance = max(max_distance, distance)
                    except nx.NetworkXNoPath:
                        continue
        return max_distance
    
    def save_topology(self, filename: str):
        """Save network topology to JSON file"""
        topology_data = {
            'nodes': {str(k): v.__dict__ for k, v in self.nodes.items()},
            'links': {str(k): v.__dict__ for k, v in self.links.items()},
            'statistics': self.get_network_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(topology_data, f, indent=2)
    
    def load_topology(self, filename: str):
        """Load network topology from JSON file"""
        with open(filename, 'r') as f:
            topology_data = json.load(f)
        
        # Clear existing topology
        self.graph.clear()
        self.nodes.clear()
        self.links.clear()
        self.node_counter = 0
        self.link_counter = 0
        
        # Load nodes
        for node_id_str, node_data in topology_data['nodes'].items():
            node_id = int(node_id_str)
            node = Node(**node_data)
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **node_data)
            self.node_counter = max(self.node_counter, node_id + 1)
        
        # Load links
        for link_id_str, link_data in topology_data['links'].items():
            link_id = int(link_id_str)
            link = Link(**link_data)
            self.links[link_id] = link
            self.graph.add_edge(link.source_node, link.destination_node,
                              link_id=link_id,
                              length_km=link.length_km,
                              weight=link.length_km)
            self.link_counter = max(self.link_counter, link_id + 1)

# Example usage
if __name__ == "__main__":
    # Create a small test network
    network = NetworkTopology()
    
    # Add nodes
    node_a = network.add_node("Node A", "core", (40.7, -74.0))
    node_b = network.add_node("Node B", "core", (41.8, -87.6))
    node_c = network.add_node("Node C", "core", (39.7, -104.9))
    node_d = network.add_node("Node D", "core", (33.7, -84.3))
    
    # Add links
    network.add_link(node_a, node_b, 1200, 15)  # 15 spans of ~80km each
    network.add_link(node_b, node_c, 1480, 18)
    network.add_link(node_c, node_d, 1650, 20)
    network.add_link(node_a, node_d, 1100, 14)
    
    # Test K-shortest paths
    paths = network.calculate_k_shortest_paths(node_a, node_d, k=3)
    print(f"3-shortest paths from {node_a} to {node_d}:")
    for i, path in enumerate(paths):
        length = network.get_path_length(path)
        print(f"  Path {i+1}: {path} (Length: {length:.1f} km)")
    
    # Print statistics
    stats = network.get_network_statistics()
    print(f"\nNetwork Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")