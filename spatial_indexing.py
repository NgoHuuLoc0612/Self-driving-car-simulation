"""
Advanced Spatial Indexing and Collision Detection
Uses R-tree spatial indexing and Shapely geometries for efficient queries
"""

import numpy as np
from rtree import index
from shapely.geometry import Point, Polygon, LineString, box
from shapely.ops import unary_union, nearest_points
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import numba
from numba import jit

@dataclass
class SpatialObject:
    """Object with spatial extent"""
    id: int
    geometry: Polygon
    velocity: np.ndarray
    object_type: str
    
class SpatialIndex:
    """
    R-tree based spatial indexing for efficient collision detection
    and nearest neighbor queries
    """
    
    def __init__(self):
        # Create R-tree index with custom properties
        properties = index.Property()
        properties.dimension = 2
        properties.fill_factor = 0.7
        properties.leaf_capacity = 100
        properties.near_minimum_overlap_factor = 32
        
        self.idx = index.Index(properties=properties)
        self.objects = {}  # id -> SpatialObject
        
    def insert(self, obj: SpatialObject):
        """Insert object into spatial index"""
        bounds = obj.geometry.bounds  # (minx, miny, maxx, maxy)
        self.idx.insert(obj.id, bounds)
        self.objects[obj.id] = obj
        
    def delete(self, obj_id: int):
        """Remove object from spatial index"""
        if obj_id in self.objects:
            bounds = self.objects[obj_id].geometry.bounds
            self.idx.delete(obj_id, bounds)
            del self.objects[obj_id]
            
    def update(self, obj: SpatialObject):
        """Update object position in index"""
        if obj.id in self.objects:
            # Delete old entry
            old_bounds = self.objects[obj.id].geometry.bounds
            self.idx.delete(obj.id, old_bounds)
            
        # Insert new entry
        new_bounds = obj.geometry.bounds
        self.idx.insert(obj.id, new_bounds)
        self.objects[obj.id] = obj
        
    def query_bbox(self, bbox: Tuple[float, float, float, float]) -> List[SpatialObject]:
        """
        Query objects within bounding box
        
        Args:
            bbox: (minx, miny, maxx, maxy)
            
        Returns:
            List of objects intersecting bbox
        """
        result_ids = list(self.idx.intersection(bbox))
        return [self.objects[obj_id] for obj_id in result_ids if obj_id in self.objects]
    
    def query_circle(self, center: np.ndarray, radius: float) -> List[SpatialObject]:
        """Query objects within circular region"""
        # Use bounding box query first, then filter
        bbox = (center[0] - radius, center[1] - radius,
                center[0] + radius, center[1] + radius)
        
        candidates = self.query_bbox(bbox)
        
        # Filter by actual circle distance
        center_point = Point(center[0], center[1])
        result = []
        
        for obj in candidates:
            if obj.geometry.distance(center_point) <= radius:
                result.append(obj)
                
        return result
    
    def nearest_neighbors(self, point: np.ndarray, k: int = 5) -> List[Tuple[SpatialObject, float]]:
        """
        Find k nearest neighbors to point
        
        Args:
            point: Query point [x, y]
            k: Number of neighbors
            
        Returns:
            List of (object, distance) tuples
        """
        query_point = Point(point[0], point[1])
        
        # R-tree nearest neighbor query
        nearest_ids = list(self.idx.nearest((point[0], point[1], point[0], point[1]), k))
        
        results = []
        for obj_id in nearest_ids:
            if obj_id in self.objects:
                obj = self.objects[obj_id]
                distance = query_point.distance(obj.geometry)
                results.append((obj, distance))
        
        # Sort by distance
        results.sort(key=lambda x: x[1])
        return results[:k]

class AdvancedCollisionDetection:
    """
    Shapely-based collision detection with support for:
    - Polygon-polygon intersection
    - Swept volume collision prediction
    - Time-to-collision estimation
    """
    
    def __init__(self):
        self.spatial_index = SpatialIndex()
        
    @staticmethod
    @jit(nopython=True)
    def _fast_aabb_check(box1: np.ndarray, box2: np.ndarray) -> bool:
        """
        Fast AABB (Axis-Aligned Bounding Box) intersection check
        Numba-optimized for performance
        
        Args:
            box1: [minx, miny, maxx, maxy]
            box2: [minx, miny, maxx, maxy]
        """
        return not (box1[2] < box2[0] or  # box1 right < box2 left
                   box1[0] > box2[2] or  # box1 left > box2 right
                   box1[3] < box2[1] or  # box1 top < box2 bottom
                   box1[1] > box2[3])    # box1 bottom > box2 top
    
    def check_collision(self, obj1: SpatialObject, obj2: SpatialObject) -> bool:
        """
        Check if two objects collide
        
        Args:
            obj1, obj2: Objects to check
            
        Returns:
            True if collision detected
        """
        # Fast AABB pre-filter
        box1 = np.array(obj1.geometry.bounds)
        box2 = np.array(obj2.geometry.bounds)
        
        if not self._fast_aabb_check(box1, box2):
            return False
        
        # Precise polygon intersection
        return obj1.geometry.intersects(obj2.geometry)
    
    def swept_volume_collision(self, obj: SpatialObject, dt: float,
                               obstacles: List[SpatialObject]) -> Optional[Tuple[SpatialObject, float]]:
        """
        Check collision along predicted trajectory using swept volumes
        
        Args:
            obj: Moving object
            dt: Time step
            obstacles: List of potential obstacles
            
        Returns:
            (colliding_object, time_to_collision) or None
        """
        # Predict future position
        displacement = obj.velocity * dt
        
        # Create swept volume (convex hull of current and future position)
        current_poly = obj.geometry
        
        # Translate to future position
        future_poly = self._translate_polygon(current_poly, displacement)
        
        # Create swept volume as union
        swept_volume = unary_union([current_poly, future_poly])
        
        # Check collisions
        min_ttc = float('inf')
        collision_obj = None
        
        for obstacle in obstacles:
            if obstacle.id == obj.id:
                continue
                
            if swept_volume.intersects(obstacle.geometry):
                # Estimate time to collision
                ttc = self._estimate_ttc(obj, obstacle, dt)
                if ttc < min_ttc:
                    min_ttc = ttc
                    collision_obj = obstacle
        
        if collision_obj is not None:
            return (collision_obj, min_ttc)
        return None
    
    def _translate_polygon(self, poly: Polygon, displacement: np.ndarray) -> Polygon:
        """Translate polygon by displacement vector"""
        from shapely.affinity import translate
        return translate(poly, xoff=displacement[0], yoff=displacement[1])
    
    def _estimate_ttc(self, obj1: SpatialObject, obj2: SpatialObject, dt: float) -> float:
        """
        Estimate time to collision between two moving objects
        
        Uses linear extrapolation of trajectories
        """
        # Get closest points
        p1, p2 = nearest_points(obj1.geometry, obj2.geometry)
        
        # Current distance
        current_dist = p1.distance(p2)
        
        # Relative velocity
        rel_velocity = obj1.velocity - obj2.velocity
        
        # Project onto line connecting objects
        direction = np.array([p2.x - p1.x, p2.y - p1.y])
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
            approach_speed = -np.dot(rel_velocity, direction)
        else:
            approach_speed = 0
        
        # Time to collision
        if approach_speed > 0:
            # Objects approaching
            safety_margin = 2.0  # meters
            ttc = (current_dist - safety_margin) / approach_speed
            return max(0, ttc)
        else:
            # Objects moving apart or parallel
            return float('inf')
    
    def predict_trajectory_collisions(self, obj: SpatialObject, 
                                     trajectory: np.ndarray,
                                     dt: float,
                                     obstacles: List[SpatialObject]) -> List[Tuple[int, SpatialObject]]:
        """
        Check trajectory for collisions
        
        Args:
            obj: Object following trajectory
            trajectory: [N x 2] array of positions
            dt: Time step between positions
            obstacles: List of obstacles
            
        Returns:
            List of (timestep, obstacle) collision pairs
        """
        collisions = []
        
        for i in range(len(trajectory) - 1):
            # Create swept volume for this segment
            pos_current = trajectory[i]
            pos_next = trajectory[i + 1]
            
            # Create polygon at current position
            current_poly = self._create_vehicle_polygon(pos_current, obj.geometry.bounds)
            
            # Create polygon at next position
            next_poly = self._create_vehicle_polygon(pos_next, obj.geometry.bounds)
            
            # Swept volume
            swept = unary_union([current_poly, next_poly])
            
            # Check obstacles
            for obstacle in obstacles:
                if swept.intersects(obstacle.geometry):
                    collisions.append((i, obstacle))
        
        return collisions
    
    def _create_vehicle_polygon(self, position: np.ndarray, bounds: Tuple) -> Polygon:
        """Create polygon representing vehicle at position"""
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        
        # Create rectangle centered at position
        return box(position[0] - width/2, position[1] - height/2,
                  position[0] + width/2, position[1] + height/2)

class VoronoiPlanner:
    """
    Voronoi diagram-based path planning
    Uses Shapely for geometric computations
    """
    
    def __init__(self, obstacles: List[Polygon]):
        self.obstacles = obstacles
        
    def generate_roadmap(self, bounds: Tuple[float, float, float, float]) -> List[LineString]:
        """
        Generate Voronoi roadmap for navigation
        
        Args:
            bounds: (minx, miny, maxx, maxy) of environment
            
        Returns:
            List of safe navigation corridors
        """
        from scipy.spatial import Voronoi
        
        # Sample points from obstacle boundaries
        points = []
        for obstacle in self.obstacles:
            coords = list(obstacle.exterior.coords)
            points.extend(coords[:-1])  # Exclude duplicate last point
        
        if len(points) < 4:
            return []
        
        # Compute Voronoi diagram
        points_array = np.array(points)
        vor = Voronoi(points_array)
        
        # Extract edges that are far from obstacles
        roadmap = []
        
        for ridge_vertices in vor.ridge_vertices:
            if -1 not in ridge_vertices:  # Exclude infinite edges
                v1 = vor.vertices[ridge_vertices[0]]
                v2 = vor.vertices[ridge_vertices[1]]
                
                edge = LineString([v1, v2])
                
                # Check if edge is safe (far from obstacles)
                safe = True
                for obstacle in self.obstacles:
                    if edge.distance(obstacle) < 1.0:  # Safety margin
                        safe = False
                        break
                
                if safe:
                    roadmap.append(edge)
        
        return roadmap

@jit(nopython=True)
def fast_point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Numba-optimized point-in-polygon test using ray casting
    
    Args:
        point: [x, y]
        polygon: [N x 2] array of vertices
        
    Returns:
        True if point is inside polygon
    """
    x, y = point[0], point[1]
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

@jit(nopython=True, parallel=True)
def batch_distance_computation(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Vectorized distance computation between two point sets
    Numba-optimized with parallel execution
    
    Args:
        points1: [N x 2] array
        points2: [M x 2] array
        
    Returns:
        [N x M] distance matrix
    """
    n1 = points1.shape[0]
    n2 = points2.shape[0]
    distances = np.zeros((n1, n2))
    
    for i in numba.prange(n1):
        for j in range(n2):
            dx = points1[i, 0] - points2[j, 0]
            dy = points1[i, 1] - points2[j, 1]
            distances[i, j] = np.sqrt(dx*dx + dy*dy)
    
    return distances