"""
Graph-Based Route Planning
Uses NetworkX for road network representation and routing algorithms
"""

import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import heapq

class RoadType(Enum):
    HIGHWAY = "highway"
    ARTERIAL = "arterial"
    RESIDENTIAL = "residential"
    SERVICE = "service"

@dataclass
class RoadSegment:
    """Road segment in network"""
    id: int
    start_node: int
    end_node: int
    length: float
    speed_limit: float
    road_type: RoadType
    lanes: int
    geometry: np.ndarray  # [N x 2] waypoints

@dataclass
class RouteResult:
    """Route planning result"""
    path: List[int]  # Node IDs
    distance: float
    travel_time: float
    geometry: np.ndarray  # [N x 2] full route geometry
    instructions: List[str]

class RoadNetwork:
    """
    Road network representation using NetworkX
    Supports multiple routing algorithms and network analysis
    """
    
    def __init__(self):
        # Directed graph for road network
        self.graph = nx.DiGraph()
        
        # Store segment details
        self.segments: Dict[Tuple[int, int], RoadSegment] = {}
        
        # Spatial index for nodes
        self.node_positions: Dict[int, np.ndarray] = {}
        
        # Next node ID
        self.next_node_id = 0
        self.next_segment_id = 0
        
    def add_node(self, position: np.ndarray, node_id: Optional[int] = None) -> int:
        """
        Add node to network
        
        Args:
            position: [x, y] position
            node_id: Optional node ID
            
        Returns:
            Node ID
        """
        if node_id is None:
            node_id = self.next_node_id
            self.next_node_id += 1
        
        self.graph.add_node(node_id, pos=position)
        self.node_positions[node_id] = position
        
        return node_id
    
    def add_segment(self, start_id: int, end_id: int, 
                   road_type: RoadType = RoadType.ARTERIAL,
                   speed_limit: float = 13.89,  # 50 km/h
                   lanes: int = 2,
                   geometry: Optional[np.ndarray] = None) -> int:
        """
        Add road segment between nodes
        
        Args:
            start_id: Start node ID
            end_id: End node ID
            road_type: Type of road
            speed_limit: Speed limit in m/s
            lanes: Number of lanes
            geometry: Optional detailed geometry
            
        Returns:
            Segment ID
        """
        # Calculate length
        start_pos = self.node_positions[start_id]
        end_pos = self.node_positions[end_id]
        
        if geometry is not None:
            # Sum of segment lengths
            length = np.sum(np.linalg.norm(np.diff(geometry, axis=0), axis=1))
        else:
            # Straight line distance
            length = np.linalg.norm(end_pos - start_pos)
            geometry = np.array([start_pos, end_pos])
        
        # Travel time (used for routing weight)
        travel_time = length / speed_limit
        
        # Add edge to graph
        self.graph.add_edge(start_id, end_id, 
                          weight=travel_time,
                          length=length,
                          speed_limit=speed_limit,
                          road_type=road_type.value,
                          lanes=lanes)
        
        # Store segment
        segment = RoadSegment(
            id=self.next_segment_id,
            start_node=start_id,
            end_node=end_id,
            length=length,
            speed_limit=speed_limit,
            road_type=road_type,
            lanes=lanes,
            geometry=geometry
        )
        
        self.segments[(start_id, end_id)] = segment
        self.next_segment_id += 1
        
        return segment.id
    
    def find_shortest_path(self, start_id: int, end_id: int,
                          weight: str = 'weight') -> Optional[RouteResult]:
        """
        Find shortest path using Dijkstra's algorithm
        
        Args:
            start_id: Start node ID
            end_id: End node ID
            weight: Edge attribute to minimize ('weight' for time, 'length' for distance)
            
        Returns:
            RouteResult or None if no path
        """
        try:
            path = nx.shortest_path(self.graph, start_id, end_id, weight=weight)
            
            # Calculate total distance and time
            total_distance = 0.0
            total_time = 0.0
            geometry_segments = []
            
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i+1]]
                total_distance += edge_data['length']
                total_time += edge_data['weight']
                
                # Get geometry
                if (path[i], path[i+1]) in self.segments:
                    segment = self.segments[(path[i], path[i+1])]
                    geometry_segments.append(segment.geometry)
            
            # Concatenate geometry
            full_geometry = np.vstack(geometry_segments) if geometry_segments else np.array([])
            
            # Generate instructions
            instructions = self._generate_instructions(path)
            
            return RouteResult(
                path=path,
                distance=total_distance,
                travel_time=total_time,
                geometry=full_geometry,
                instructions=instructions
            )
            
        except nx.NetworkXNoPath:
            return None
    
    def find_k_shortest_paths(self, start_id: int, end_id: int, k: int = 3) -> List[RouteResult]:
        """
        Find k shortest paths using Yen's algorithm
        
        Args:
            start_id: Start node
            end_id: End node
            k: Number of paths to find
            
        Returns:
            List of k RouteResults
        """
        try:
            # Use NetworkX's k shortest paths
            paths = list(nx.shortest_simple_paths(self.graph, start_id, end_id, weight='weight'))
            
            results = []
            for path in paths[:k]:
                # Calculate metrics for this path
                total_distance = 0.0
                total_time = 0.0
                geometry_segments = []
                
                for i in range(len(path) - 1):
                    edge_data = self.graph[path[i]][path[i+1]]
                    total_distance += edge_data['length']
                    total_time += edge_data['weight']
                    
                    if (path[i], path[i+1]) in self.segments:
                        segment = self.segments[(path[i], path[i+1])]
                        geometry_segments.append(segment.geometry)
                
                full_geometry = np.vstack(geometry_segments) if geometry_segments else np.array([])
                instructions = self._generate_instructions(path)
                
                results.append(RouteResult(
                    path=path,
                    distance=total_distance,
                    travel_time=total_time,
                    geometry=full_geometry,
                    instructions=instructions
                ))
            
            return results
            
        except nx.NetworkXNoPath:
            return []
    
    def find_path_avoiding_areas(self, start_id: int, end_id: int,
                                avoid_nodes: Set[int]) -> Optional[RouteResult]:
        """
        Find path while avoiding certain nodes (e.g., traffic jams)
        
        Args:
            start_id: Start node
            end_id: End node
            avoid_nodes: Set of node IDs to avoid
            
        Returns:
            RouteResult or None
        """
        # Create subgraph without avoided nodes
        nodes_to_keep = set(self.graph.nodes()) - avoid_nodes
        subgraph = self.graph.subgraph(nodes_to_keep).copy()
        
        try:
            path = nx.shortest_path(subgraph, start_id, end_id, weight='weight')
            
            # Calculate metrics
            total_distance = 0.0
            total_time = 0.0
            geometry_segments = []
            
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i+1]]
                total_distance += edge_data['length']
                total_time += edge_data['weight']
                
                if (path[i], path[i+1]) in self.segments:
                    segment = self.segments[(path[i], path[i+1])]
                    geometry_segments.append(segment.geometry)
            
            full_geometry = np.vstack(geometry_segments) if geometry_segments else np.array([])
            instructions = self._generate_instructions(path)
            
            return RouteResult(
                path=path,
                distance=total_distance,
                travel_time=total_time,
                geometry=full_geometry,
                instructions=instructions
            )
            
        except nx.NetworkXNoPath:
            return None
    
    def update_traffic_conditions(self, segment_key: Tuple[int, int], 
                                  congestion_factor: float):
        """
        Update edge weight based on traffic conditions
        
        Args:
            segment_key: (start_node, end_node)
            congestion_factor: Multiplier for travel time (1.0 = normal, 2.0 = twice as slow)
        """
        if self.graph.has_edge(*segment_key):
            edge_data = self.graph[segment_key[0]][segment_key[1]]
            base_time = edge_data['length'] / edge_data['speed_limit']
            new_time = base_time * congestion_factor
            self.graph[segment_key[0]][segment_key[1]]['weight'] = new_time
    
    def find_nearest_node(self, position: np.ndarray) -> int:
        """
        Find nearest node to position
        
        Args:
            position: [x, y] position
            
        Returns:
            Node ID
        """
        min_dist = float('inf')
        nearest_id = -1
        
        for node_id, node_pos in self.node_positions.items():
            dist = np.linalg.norm(position - node_pos)
            if dist < min_dist:
                min_dist = dist
                nearest_id = node_id
        
        return nearest_id
    
    def analyze_network(self) -> Dict:
        """
        Analyze road network properties
        
        Returns:
            Dictionary of network statistics
        """
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'total_length': sum(data['length'] for _, _, data in self.graph.edges(data=True)),
            'is_connected': nx.is_weakly_connected(self.graph),
            'avg_degree': sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        }
        
        # Find articulation points (critical intersections)
        if nx.is_connected(self.graph.to_undirected()):
            articulation_points = list(nx.articulation_points(self.graph.to_undirected()))
            stats['critical_intersections'] = len(articulation_points)
        
        # Compute betweenness centrality (importance of nodes)
        betweenness = nx.betweenness_centrality(self.graph, weight='weight')
        stats['most_important_node'] = max(betweenness, key=betweenness.get)
        stats['max_betweenness'] = betweenness[stats['most_important_node']]
        
        return stats
    
    def _generate_instructions(self, path: List[int]) -> List[str]:
        """Generate turn-by-turn instructions"""
        instructions = []
        
        if len(path) < 2:
            return instructions
        
        instructions.append(f"Start at node {path[0]}")
        
        for i in range(len(path) - 1):
            edge_data = self.graph[path[i]][path[i+1]]
            length = edge_data['length']
            road_type = edge_data.get('road_type', 'road')
            
            instructions.append(
                f"Continue on {road_type} for {length:.0f}m to node {path[i+1]}"
            )
        
        instructions.append(f"Arrive at destination (node {path[-1]})")
        
        return instructions
    
    def export_to_graphml(self, filename: str):
        """Export network to GraphML format"""
        nx.write_graphml(self.graph, filename)
    
    def import_from_graphml(self, filename: str):
        """Import network from GraphML format"""
        self.graph = nx.read_graphml(filename)
        
        # Rebuild node positions
        for node_id, data in self.graph.nodes(data=True):
            if 'pos' in data:
                self.node_positions[int(node_id)] = np.array(data['pos'])

class TrafficAwareRouter:
    """
    Advanced router that considers real-time traffic
    Uses A* with dynamic heuristics
    """
    
    def __init__(self, network: RoadNetwork):
        self.network = network
        self.traffic_data: Dict[Tuple[int, int], float] = {}  # edge -> congestion
        
    def update_traffic(self, edge: Tuple[int, int], congestion: float):
        """
        Update traffic congestion for edge
        
        Args:
            edge: (start_node, end_node)
            congestion: Congestion factor (1.0 = free flow, >1 = congested)
        """
        self.traffic_data[edge] = congestion
        self.network.update_traffic_conditions(edge, congestion)
    
    def route_with_alternatives(self, start_id: int, end_id: int,
                               num_alternatives: int = 3) -> List[RouteResult]:
        """
        Find multiple route alternatives considering traffic
        
        Returns:
            List of routes ranked by travel time
        """
        routes = self.network.find_k_shortest_paths(start_id, end_id, num_alternatives)
        
        # Rank by current travel time (considering traffic)
        routes.sort(key=lambda r: r.travel_time)
        
        return routes
    
    def predict_future_traffic(self, current_time: float, 
                              edge: Tuple[int, int]) -> float:
        """
        Predict traffic congestion at future time
        
        Simple model: Could be enhanced with ML
        """
        # Get current congestion
        current_congestion = self.traffic_data.get(edge, 1.0)
        
        # Time-of-day pattern (simplified)
        hour = (current_time / 3600) % 24
        
        # Rush hour multiplier
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            rush_hour_factor = 1.5
        else:
            rush_hour_factor = 1.0
        
        return current_congestion * rush_hour_factor
    
    def optimal_departure_time(self, start_id: int, end_id: int,
                              time_window: Tuple[float, float],
                              time_step: float = 300) -> Tuple[float, RouteResult]:
        """
        Find optimal departure time to minimize travel time
        
        Args:
            start_id: Start node
            end_id: End node
            time_window: (start_time, end_time) to search
            time_step: Time step for search (seconds)
            
        Returns:
            (optimal_time, route)
        """
        best_time = time_window[0]
        best_route = None
        min_travel_time = float('inf')
        
        current_time = time_window[0]
        while current_time <= time_window[1]:
            # Predict traffic for this departure time
            for edge in self.network.graph.edges():
                predicted_congestion = self.predict_future_traffic(current_time, edge)
                self.update_traffic(edge, predicted_congestion)
            
            # Find route
            route = self.network.find_shortest_path(start_id, end_id)
            
            if route and route.travel_time < min_travel_time:
                min_travel_time = route.travel_time
                best_time = current_time
                best_route = route
            
            current_time += time_step
        
        return best_time, best_route

def build_grid_network(grid_size: int = 10, spacing: float = 100.0) -> RoadNetwork:
    """
    Build a simple grid road network for testing
    
    Args:
        grid_size: Number of nodes per side
        spacing: Distance between nodes in meters
        
    Returns:
        RoadNetwork instance
    """
    network = RoadNetwork()
    
    # Create nodes
    node_grid = {}
    for i in range(grid_size):
        for j in range(grid_size):
            pos = np.array([i * spacing, j * spacing])
            node_id = network.add_node(pos)
            node_grid[(i, j)] = node_id
    
    # Create edges (4-connected grid)
    for i in range(grid_size):
        for j in range(grid_size):
            current_id = node_grid[(i, j)]
            
            # Right neighbor
            if i < grid_size - 1:
                right_id = node_grid[(i+1, j)]
                network.add_segment(current_id, right_id, RoadType.ARTERIAL)
                network.add_segment(right_id, current_id, RoadType.ARTERIAL)  # Bidirectional
            
            # Top neighbor
            if j < grid_size - 1:
                top_id = node_grid[(i, j+1)]
                network.add_segment(current_id, top_id, RoadType.ARTERIAL)
                network.add_segment(top_id, current_id, RoadType.ARTERIAL)
    
    return network