"""
Advanced Perception System
Simulates sensors (LiDAR, RADAR, Camera) and processes their data
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from config import SensorConfig, SENSOR_CONFIG
from environment import DynamicObject, Environment

@dataclass
class LiDARPoint:
    """Single LiDAR point"""
    x: float
    y: float
    z: float
    intensity: float
    distance: float
    angle: float
    layer: int

@dataclass
class RADARDetection:
    """RADAR detection"""
    range: float
    azimuth: float
    range_rate: float  # doppler velocity
    rcs: float  # radar cross section
    snr: float  # signal to noise ratio

@dataclass
class CameraDetection:
    """Camera object detection"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    class_id: int
    confidence: float
    position_3d: Optional[np.ndarray] = None

class LiDARSimulator:
    """
    LiDAR sensor simulator
    Generates point cloud from environment
    """
    
    def __init__(self, config: SensorConfig = SENSOR_CONFIG):
        self.config = config
        
        # Sensor parameters
        self.max_range = config.lidar_range
        self.fov = np.deg2rad(config.lidar_fov)
        self.angular_resolution = np.deg2rad(config.lidar_resolution)
        self.num_layers = config.lidar_layers
        self.noise_std = config.lidar_noise_std
        
        # Mounting position (relative to vehicle center)
        self.mount_position = np.array([2.0, 0.0, 1.5])  # [x, y, z]
        
    def simulate(self, vehicle_position: np.ndarray, vehicle_yaw: float,
                environment: Environment) -> List[LiDARPoint]:
        """
        Simulate LiDAR scan
        
        Args:
            vehicle_position: Vehicle position [x, y]
            vehicle_yaw: Vehicle heading
            environment: Simulation environment
            
        Returns:
            List of LiDAR points
        """
        points = []
        
        # Sensor position in world frame (3D: x, y, z)
        sensor_pos = np.array([
            vehicle_position[0] + self.mount_position[0] * np.cos(vehicle_yaw) - 
            self.mount_position[1] * np.sin(vehicle_yaw),
            vehicle_position[1] + self.mount_position[0] * np.sin(vehicle_yaw) + 
            self.mount_position[1] * np.cos(vehicle_yaw),
            self.mount_position[2]  # z-coordinate (sensor height)
        ])
        
        # Generate rays
        num_azimuth = int(self.fov / self.angular_resolution)
        
        for layer in range(self.num_layers):
            # Vertical angle for this layer
            vertical_angle = np.deg2rad(
                -15 + (layer / self.num_layers) * 30
            )
            
            for i in range(num_azimuth):
                # Horizontal angle
                azimuth = -self.fov/2 + i * self.angular_resolution + vehicle_yaw
                
                # Ray direction
                dx = np.cos(vertical_angle) * np.cos(azimuth)
                dy = np.cos(vertical_angle) * np.sin(azimuth)
                dz = np.sin(vertical_angle)
                
                direction = np.array([dx, dy, dz])
                
                # Raycast
                hit_point, hit_distance, intensity = self._raycast(
                    sensor_pos, direction, environment
                )
                
                if hit_point is not None:
                    # Add noise
                    noisy_distance = hit_distance + np.random.randn() * self.noise_std
                    
                    # Create point
                    point = LiDARPoint(
                        x=hit_point[0],
                        y=hit_point[1],
                        z=hit_point[2] if len(hit_point) > 2 else self.mount_position[2],
                        intensity=intensity,
                        distance=noisy_distance,
                        angle=azimuth,
                        layer=layer
                    )
                    points.append(point)
        
        return points
    
    def _raycast(self, origin: np.ndarray, direction: np.ndarray,
                environment: Environment) -> Tuple[Optional[np.ndarray], float, float]:
        """
        Cast ray and find intersection
        
        Returns:
            Tuple of (hit_point, distance, intensity)
        """
        min_distance = self.max_range
        hit_point = None
        intensity = 0.0
        
        # Check all objects
        for obj in environment.get_all_objects():
            # Simplified: treat as sphere
            obj_center = np.array([obj.position[0], obj.position[1], 
                                  obj.dimensions[2] / 2])
            obj_radius = max(obj.dimensions[0], obj.dimensions[1]) / 2
            
            # Ray-sphere intersection
            oc = origin - obj_center
            a = np.dot(direction, direction)
            b = 2.0 * np.dot(oc, direction)
            c = np.dot(oc, oc) - obj_radius**2
            
            discriminant = b*b - 4*a*c
            
            if discriminant > 0:
                t = (-b - np.sqrt(discriminant)) / (2*a)
                if 0 < t < min_distance:
                    min_distance = t
                    hit_point = origin + t * direction
                    # Intensity based on material (simplified)
                    intensity = 0.5 + 0.5 * np.random.rand()
        
        # Check static obstacles
        for obs_pos, obs_radius in environment.static_obstacles:
            obs_center = np.array([obs_pos[0], obs_pos[1], 0.5])
            
            oc = origin - obs_center
            a = np.dot(direction, direction)
            b = 2.0 * np.dot(oc, direction)
            c = np.dot(oc, oc) - obs_radius**2
            
            discriminant = b*b - 4*a*c
            
            if discriminant > 0:
                t = (-b - np.sqrt(discriminant)) / (2*a)
                if 0 < t < min_distance:
                    min_distance = t
                    hit_point = origin + t * direction
                    intensity = 0.8
        
        if hit_point is not None:
            return hit_point, min_distance, intensity
        else:
            return None, min_distance, 0.0

class RADARSimulator:
    """
    RADAR sensor simulator
    Provides range and doppler velocity measurements
    """
    
    def __init__(self, config: SensorConfig = SENSOR_CONFIG):
        self.config = config
        
        self.max_range = config.radar_range
        self.fov = np.deg2rad(config.radar_fov)
        self.range_resolution = config.radar_resolution_range
        self.azimuth_resolution = np.deg2rad(config.radar_resolution_azimuth)
        
        self.mount_position = np.array([2.5, 0.0, 0.5])
        
    def simulate(self, vehicle_position: np.ndarray, vehicle_yaw: float,
                vehicle_velocity: np.ndarray, environment: Environment) -> List[RADARDetection]:
        """
        Simulate RADAR scan
        
        Args:
            vehicle_position: Vehicle position
            vehicle_yaw: Vehicle heading
            vehicle_velocity: Vehicle velocity [vx, vy]
            environment: Environment
            
        Returns:
            List of RADAR detections
        """
        detections = []
        
        # Sensor position
        sensor_pos = vehicle_position + np.array([
            self.mount_position[0] * np.cos(vehicle_yaw),
            self.mount_position[0] * np.sin(vehicle_yaw)
        ])
        
        # Scan field of view
        for obj in environment.get_all_objects():
            # Vector to object
            to_obj = obj.position - sensor_pos
            range_m = np.linalg.norm(to_obj)
            
            if range_m > self.max_range:
                continue
            
            # Azimuth angle
            azimuth = np.arctan2(to_obj[1], to_obj[0]) - vehicle_yaw
            azimuth = np.arctan2(np.sin(azimuth), np.cos(azimuth))
            
            # Check if in FOV
            if abs(azimuth) > self.fov / 2:
                continue
            
            # Doppler velocity (radial component)
            relative_velocity = obj.velocity - vehicle_velocity
            range_rate = np.dot(relative_velocity, to_obj / range_m)
            
            # Radar cross section (simplified)
            rcs = obj.dimensions[0] * obj.dimensions[1] * 10  # m^2
            
            # Signal to noise ratio (simplified)
            snr = rcs / (range_m**4) * 1e6
            
            # Detection threshold
            if snr > 10.0:
                # Add noise
                range_noise = np.random.randn() * self.range_resolution
                azimuth_noise = np.random.randn() * self.azimuth_resolution
                velocity_noise = np.random.randn() * 0.1
                
                detection = RADARDetection(
                    range=range_m + range_noise,
                    azimuth=azimuth + azimuth_noise,
                    range_rate=range_rate + velocity_noise,
                    rcs=rcs,
                    snr=snr
                )
                detections.append(detection)
        
        return detections

class CameraSimulator:
    """
    Camera sensor with object detection
    Simulates vision-based perception
    """
    
    def __init__(self, config: SensorConfig = SENSOR_CONFIG):
        self.config = config
        
        self.fov = np.deg2rad(config.camera_fov)
        self.resolution = config.camera_resolution
        self.max_detection_range = 100.0
        
        # Mounted facing forward
        self.mount_position = np.array([2.0, 0.0, 1.2])
        
        # Object classes
        self.classes = {
            'vehicle': 0,
            'pedestrian': 1,
            'cyclist': 2,
            'traffic_light': 3,
            'stop_sign': 4
        }
        
    def simulate(self, vehicle_position: np.ndarray, vehicle_yaw: float,
                environment: Environment) -> List[CameraDetection]:
        """
        Simulate camera detections
        
        Returns:
            List of detected objects
        """
        detections = []
        
        # Camera position
        camera_pos = vehicle_position + np.array([
            self.mount_position[0] * np.cos(vehicle_yaw),
            self.mount_position[0] * np.sin(vehicle_yaw)
        ])
        
        for obj in environment.get_all_objects():
            # Check if in view
            to_obj = obj.position - camera_pos
            distance = np.linalg.norm(to_obj)
            
            if distance > self.max_detection_range:
                continue
            
            # Angle relative to camera
            angle = np.arctan2(to_obj[1], to_obj[0]) - vehicle_yaw
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            
            if abs(angle) > self.fov / 2:
                continue
            
            # Project to image plane
            # Simplified pinhole camera model
            focal_length = self.resolution[0] / (2 * np.tan(self.fov / 2))
            
            # Object appears smaller with distance
            apparent_size = (obj.dimensions[1] / distance) * focal_length
            
            # Center in image
            u = self.resolution[0] / 2 + (angle / self.fov) * self.resolution[0]
            v = self.resolution[1] / 2
            
            # Bounding box
            bbox = np.array([
                max(0, u - apparent_size / 2),
                max(0, v - apparent_size / 2),
                min(self.resolution[0], u + apparent_size / 2),
                min(self.resolution[1], v + apparent_size / 2)
            ])
            
            # Classification
            if obj.object_type.value in self.classes:
                class_id = self.classes[obj.object_type.value]
            else:
                class_id = 0
            
            # Confidence (decreases with distance and occlusion)
            confidence = np.exp(-distance / 50.0) * (0.9 + 0.1 * np.random.rand())
            confidence = np.clip(confidence, 0.0, 1.0)
            
            # 3D position estimate
            position_3d = np.array([to_obj[0], to_obj[1], obj.dimensions[2]/2])
            
            detection = CameraDetection(
                bbox=bbox,
                class_id=class_id,
                confidence=confidence,
                position_3d=position_3d
            )
            detections.append(detection)
        
        return detections

class PerceptionSystem:
    """
    Unified perception system
    Combines all sensor data
    """
    
    def __init__(self, config: SensorConfig = SENSOR_CONFIG):
        self.config = config
        
        # Sensor simulators
        self.lidar = LiDARSimulator(config)
        self.radar = RADARSimulator(config)
        self.cameras = [CameraSimulator(config) for _ in range(6)]
        
        # Perception outputs
        self.detected_objects: List[Dict] = []
        self.occupancy_grid: Optional[np.ndarray] = None
        
    def perceive(self, vehicle_position: np.ndarray, vehicle_yaw: float,
                vehicle_velocity: np.ndarray, environment: Environment) -> Dict:
        """
        Run full perception pipeline
        
        Args:
            vehicle_position: Vehicle position [x, y]
            vehicle_yaw: Vehicle heading
            vehicle_velocity: Vehicle velocity [vx, vy]
            environment: Simulation environment
            
        Returns:
            Dictionary with all perception outputs
        """
        # LiDAR
        lidar_points = self.lidar.simulate(
            vehicle_position, vehicle_yaw, environment
        )
        
        # RADAR
        radar_detections = self.radar.simulate(
            vehicle_position, vehicle_yaw, vehicle_velocity, environment
        )
        
        # Camera (front camera only for now)
        camera_detections = self.cameras[0].simulate(
            vehicle_position, vehicle_yaw, environment
        )
        
        # Process LiDAR to occupancy grid
        self.occupancy_grid = self._generate_occupancy_grid(
            lidar_points, vehicle_position, vehicle_yaw
        )
        
        # Cluster LiDAR points to objects
        lidar_objects = self._cluster_lidar_points(lidar_points)
        
        # Combine detections
        self.detected_objects = self._fuse_detections(
            lidar_objects, radar_detections, camera_detections
        )
        
        return {
            'lidar_points': lidar_points,
            'radar_detections': radar_detections,
            'camera_detections': camera_detections,
            'occupancy_grid': self.occupancy_grid,
            'detected_objects': self.detected_objects
        }
    
    def _generate_occupancy_grid(self, points: List[LiDARPoint],
                                vehicle_pos: np.ndarray,
                                vehicle_yaw: float) -> np.ndarray:
        """Generate occupancy grid from LiDAR points"""
        # Grid parameters
        grid_size = 200  # 200x200 meters
        resolution = 0.5  # 0.5m cells
        
        grid_cells = int(grid_size / resolution)
        grid = np.zeros((grid_cells, grid_cells))
        
        for point in points:
            # Convert to vehicle frame
            dx = point.x - vehicle_pos[0]
            dy = point.y - vehicle_pos[1]
            
            # Rotate to vehicle frame
            local_x = dx * np.cos(-vehicle_yaw) - dy * np.sin(-vehicle_yaw)
            local_y = dx * np.sin(-vehicle_yaw) + dy * np.cos(-vehicle_yaw)
            
            # Convert to grid coordinates
            grid_x = int((local_x + grid_size/2) / resolution)
            grid_y = int((local_y + grid_size/2) / resolution)
            
            if 0 <= grid_x < grid_cells and 0 <= grid_y < grid_cells:
                grid[grid_x, grid_y] = 1.0
        
        return grid
    
    def _cluster_lidar_points(self, points: List[LiDARPoint]) -> List[Dict]:
        """
        Cluster LiDAR points into objects using DBSCAN
        
        Returns:
            List of detected objects
        """
        if len(points) == 0:
            return []
        
        # Simple distance-based clustering
        objects = []
        points_array = np.array([[p.x, p.y] for p in points])
        
        visited = np.zeros(len(points), dtype=bool)
        
        for i in range(len(points)):
            if visited[i]:
                continue
            
            # Start new cluster
            cluster_indices = [i]
            visited[i] = True
            
            # Find nearby points
            for j in range(len(points)):
                if visited[j]:
                    continue
                
                if np.linalg.norm(points_array[i] - points_array[j]) < 1.0:
                    cluster_indices.append(j)
                    visited[j] = True
            
            # Create object from cluster
            if len(cluster_indices) > 5:  # Minimum points
                cluster_points = points_array[cluster_indices]
                center = np.mean(cluster_points, axis=0)
                
                # Estimate size
                size = np.max(cluster_points, axis=0) - np.min(cluster_points, axis=0)
                
                objects.append({
                    'position': center,
                    'size': size,
                    'points': cluster_indices,
                    'source': 'lidar'
                })
        
        return objects
    
    def _fuse_detections(self, lidar_objects: List[Dict],
                        radar_detections: List[RADARDetection],
                        camera_detections: List[CameraDetection]) -> List[Dict]:
        """
        Fuse detections from multiple sensors
        
        Returns:
            Fused object list
        """
        fused_objects = []
        
        # Associate LiDAR and camera detections
        for lidar_obj in lidar_objects:
            obj = {
                'position': lidar_obj['position'],
                'size': lidar_obj['size'],
                'velocity': np.zeros(2),
                'classification': 'unknown',
                'confidence': 0.5,
                'sensors': ['lidar']
            }
            
            # Find matching camera detection
            for cam_det in camera_detections:
                if cam_det.position_3d is not None:
                    cam_pos = cam_det.position_3d[0:2]
                    if np.linalg.norm(cam_pos - lidar_obj['position']) < 2.0:
                        obj['classification'] = list(self.cameras[0].classes.keys())[cam_det.class_id]
                        obj['confidence'] = max(obj['confidence'], cam_det.confidence)
                        obj['sensors'].append('camera')
                        break
            
            # Find matching RADAR detection
            for radar_det in radar_detections:
                # Convert RADAR to cartesian
                radar_pos = np.array([
                    radar_det.range * np.cos(radar_det.azimuth),
                    radar_det.range * np.sin(radar_det.azimuth)
                ])
                
                if np.linalg.norm(radar_pos - lidar_obj['position']) < 3.0:
                    # Add velocity from RADAR
                    direction = radar_pos / np.linalg.norm(radar_pos)
                    obj['velocity'] = direction * radar_det.range_rate
                    obj['sensors'].append('radar')
                    break
            
            fused_objects.append(obj)
        
        return fused_objects