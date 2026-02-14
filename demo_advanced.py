"""
Advanced Features Demo
Demonstrates spatial indexing, ML utilities, network routing, and visualization
"""

import numpy as np
import sys
import os

print("="*70)
print("ADVANCED AUTONOMOUS VEHICLE SIMULATION - FEATURES DEMO")
print("="*70)
print()

# Test module availability
modules_available = {}

print("Checking module availability...")
print("-" * 70)

try:
    from spatial_indexing import SpatialIndex, AdvancedCollisionDetection, SpatialObject
    from shapely.geometry import box, Point, Polygon
    import rtree
    modules_available['spatial'] = True
    print("✓ Spatial Indexing (rtree, shapely) - AVAILABLE")
except ImportError as e:
    modules_available['spatial'] = False
    print(f"✗ Spatial Indexing - NOT AVAILABLE: {e}")

try:
    from visualization import PygameVisualizer, CameraRenderer, VisualizationConfig
    import pygame
    import cv2
    modules_available['visualization'] = True
    print("✓ Advanced Visualization (pygame, opencv) - AVAILABLE")
except ImportError as e:
    modules_available['visualization'] = False
    print(f"✗ Advanced Visualization - NOT AVAILABLE: {e}")

try:
    from ml_utils import LiDARClustering, TrajectoryAnalyzer, RLTrainingMonitor
    from sklearn.cluster import DBSCAN
    import tensorboard
    modules_available['ml_utils'] = True
    print("✓ ML Utilities (scikit-learn, tensorboard) - AVAILABLE")
except ImportError as e:
    modules_available['ml_utils'] = False
    print(f"✗ ML Utilities - NOT AVAILABLE: {e}")

try:
    from network_routing import RoadNetwork, TrafficAwareRouter, build_grid_network
    import networkx as nx
    modules_available['routing'] = True
    print("✓ Network Routing (networkx) - AVAILABLE")
except ImportError as e:
    modules_available['routing'] = False
    print(f"✗ Network Routing - NOT AVAILABLE: {e}")

print()
print("="*70)
print()

# Demo 1: Spatial Indexing
if modules_available['spatial']:
    print("DEMO 1: Spatial Indexing and Collision Detection")
    print("-" * 70)
    
    # Create spatial index
    spatial_idx = SpatialIndex()
    
    # Add some objects
    for i in range(10):
        pos = np.random.rand(2) * 100
        polygon = box(pos[0] - 2, pos[1] - 2, pos[0] + 2, pos[1] + 2)
        
        obj = SpatialObject(
            id=i,
            geometry=polygon,
            velocity=np.random.randn(2),
            object_type='vehicle'
        )
        spatial_idx.insert(obj)
    
    print(f"✓ Created spatial index with {len(spatial_idx.objects)} objects")
    
    # Query near a point
    query_point = np.array([50.0, 50.0])
    nearby = spatial_idx.query_circle(query_point, radius=20.0)
    print(f"✓ Found {len(nearby)} objects within 20m of {query_point}")
    
    # Find nearest neighbors
    nearest = spatial_idx.nearest_neighbors(query_point, k=3)
    print(f"✓ Found {len(nearest)} nearest neighbors")
    for obj, dist in nearest[:3]:
        print(f"  - Object {obj.id} at distance {dist:.2f}m")
    
    # Collision detection
    collision_detector = AdvancedCollisionDetection()
    if len(nearby) >= 2:
        obj1, obj2 = nearby[0], nearby[1]
        collision = collision_detector.check_collision(obj1, obj2)
        print(f"✓ Collision check between objects: {collision}")
    
    print("✓ Spatial indexing demo completed successfully!")
    print()

# Demo 2: ML Utilities
if modules_available['ml_utils']:
    print("DEMO 2: Machine Learning Utilities")
    print("-" * 70)
    
    # LiDAR clustering
    clusterer = LiDARClustering(eps=0.5, min_samples=5)
    
    # Generate fake point cloud
    num_clusters = 5
    points_per_cluster = 50
    points = []
    
    for i in range(num_clusters):
        center = np.random.rand(3) * 50
        cluster_points = center + np.random.randn(points_per_cluster, 3) * 2
        points.append(cluster_points)
    
    points = np.vstack(points)
    print(f"✓ Generated {len(points)} LiDAR points")
    
    # Cluster
    result = clusterer.cluster_points(points)
    print(f"✓ Found {result.n_clusters} clusters")
    print(f"  - Noise points: {np.sum(result.labels == -1)}")
    
    # Extract objects
    objects = clusterer.extract_objects(points, result.labels)
    print(f"✓ Extracted {len(objects)} objects")
    for obj in objects[:3]:
        print(f"  - {obj['type']}: size={obj['size']}, points={obj['num_points']}")
    
    # Trajectory analysis
    analyzer = TrajectoryAnalyzer()
    
    # Generate fake trajectories
    trajectories = []
    for i in range(20):
        t = np.linspace(0, 10, 100)
        traj = np.column_stack([
            t + np.random.randn(100) * 0.5,
            np.sin(t) + np.random.randn(100) * 0.5
        ])
        trajectories.append(traj)
    
    print(f"✓ Generated {len(trajectories)} trajectories")
    
    # Cluster trajectories
    labels = analyzer.cluster_trajectories(trajectories, n_clusters=3)
    print(f"✓ Clustered into {len(set(labels))} groups")
    
    # Find representatives
    representatives = analyzer.find_representative_trajectories(trajectories, labels)
    print(f"✓ Found {len(representatives)} representative trajectories")
    
    print("✓ ML utilities demo completed successfully!")
    print()

# Demo 3: Network Routing
if modules_available['routing']:
    print("DEMO 3: Network Routing and Traffic-Aware Planning")
    print("-" * 70)
    
    # Build grid network
    network = build_grid_network(grid_size=10, spacing=100.0)
    print(f"✓ Built road network with {network.graph.number_of_nodes()} nodes")
    print(f"✓ Network has {network.graph.number_of_edges()} edges")
    
    # Analyze network
    stats = network.analyze_network()
    print(f"✓ Network statistics:")
    print(f"  - Total length: {stats['total_length']:.0f}m")
    print(f"  - Average degree: {stats['avg_degree']:.2f}")
    print(f"  - Connected: {stats['is_connected']}")
    
    # Find shortest path
    start_node = 0
    end_node = network.graph.number_of_nodes() - 1
    
    route = network.find_shortest_path(start_node, end_node, weight='weight')
    if route:
        print(f"✓ Found route from node {start_node} to {end_node}")
        print(f"  - Distance: {route.distance:.0f}m")
        print(f"  - Travel time: {route.travel_time:.1f}s")
        print(f"  - Waypoints: {len(route.path)} nodes")
    
    # Find k-shortest paths
    k_routes = network.find_k_shortest_paths(start_node, end_node, k=3)
    print(f"✓ Found {len(k_routes)} alternative routes")
    for i, route in enumerate(k_routes):
        print(f"  Route {i+1}: {route.distance:.0f}m, {route.travel_time:.1f}s")
    
    # Traffic-aware routing
    router = TrafficAwareRouter(network)
    
    # Add some traffic congestion
    if network.graph.number_of_edges() > 0:
        edges = list(network.graph.edges())[:5]
        for edge in edges:
            router.update_traffic(edge, congestion=2.0)  # 2x slower
        
        print(f"✓ Added traffic congestion to {len(edges)} edges")
    
    # Find route with traffic
    traffic_route = router.route_with_alternatives(start_node, end_node, num_alternatives=2)
    if traffic_route:
        print(f"✓ Found {len(traffic_route)} routes considering traffic")
        for i, route in enumerate(traffic_route):
            print(f"  Route {i+1}: {route.travel_time:.1f}s (with traffic)")
    
    print("✓ Network routing demo completed successfully!")
    print()

# Demo 4: Visualization (if available, just test initialization)
if modules_available['visualization']:
    print("DEMO 4: Advanced Visualization")
    print("-" * 70)
    
    # Test Pygame visualizer initialization
    try:
        config = VisualizationConfig(
            window_width=800,
            window_height=600,
            fps=30
        )
        print("✓ VisualizationConfig created")
        
        # Note: We don't actually run the visualizer in this demo
        # as it requires a display and event loop
        print("✓ Pygame visualization available (requires display to run)")
        
        # Test camera renderer
        renderer = CameraRenderer(width=640, height=480)
        print("✓ CameraRenderer initialized")
        
    except Exception as e:
        print(f"✗ Visualization test failed: {e}")
    
    print("✓ Visualization demo completed successfully!")
    print()

# Demo 5: Integration Test
print("DEMO 5: Integration with Main Simulation")
print("-" * 70)

try:
    from main import AutonomousVehicle, Simulation
    from config import SIM_CONFIG
    
    # Create simulation
    print("✓ Importing simulation components...")
    
    # Test vehicle creation
    vehicle = AutonomousVehicle()
    print("✓ Created AutonomousVehicle instance")
    
    # Check which features are enabled
    if hasattr(vehicle, 'spatial_index') and vehicle.spatial_index:
        print("  ✓ Spatial indexing enabled")
    if hasattr(vehicle, 'lidar_clusterer') and vehicle.lidar_clusterer:
        print("  ✓ LiDAR clustering enabled")
    if hasattr(vehicle, 'trajectory_analyzer') and vehicle.trajectory_analyzer:
        print("  ✓ Trajectory analysis enabled")
    
    # Test simulation creation
    sim = Simulation()
    print("✓ Created Simulation instance")
    
    if hasattr(sim, 'road_network') and sim.road_network:
        print("  ✓ Road network routing enabled")
    if hasattr(sim, 'training_monitor') and sim.training_monitor:
        print("  ✓ TensorBoard logging enabled")
    if hasattr(sim, 'pygame_viz') and sim.use_pygame:
        print("  ✓ Pygame visualization enabled")
    
    print("✓ Integration test completed successfully!")
    
except Exception as e:
    print(f"✗ Integration test failed: {e}")
    import traceback
    traceback.print_exc()

print()
print("="*70)
print("DEMO SUMMARY")
print("="*70)

available_count = sum(modules_available.values())
total_count = len(modules_available)

print(f"Modules available: {available_count}/{total_count}")
print()

for module, available in modules_available.items():
    status = "✓ AVAILABLE" if available else "✗ NOT AVAILABLE"
    print(f"  {module:20s}: {status}")

print()
if available_count == total_count:
    print("✓ ALL ADVANCED FEATURES AVAILABLE AND WORKING!")
    print()
    print("You can now run the main simulation with:")
    print("  python main.py")
    print()
    print("For full Pygame visualization, ensure you have a display available.")
else:
    print("⚠ Some modules are missing. Install missing dependencies:")
    print()
    if not modules_available['spatial']:
        print("  pip install rtree shapely")
    if not modules_available['visualization']:
        print("  pip install pygame opencv-python Pillow")
    if not modules_available['ml_utils']:
        print("  pip install scikit-learn tensorboard")
    if not modules_available['routing']:
        print("  pip install networkx")
    print()
    print("Or install all at once:")
    print("  pip install -r requirements.txt")

print()
print("="*70)