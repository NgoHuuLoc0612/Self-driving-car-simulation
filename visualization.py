"""
Advanced Real-Time Visualization System
Uses Pygame for interactive visualization and OpenCV for rendering
"""

import pygame
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import time

from environment import Environment, DynamicObject
from vehicle_dynamics import VehicleState

@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    window_width: int = 1920
    window_height: int = 1080
    pixels_per_meter: float = 10.0
    fps: int = 60
    show_sensors: bool = True
    show_trajectories: bool = True
    show_occupancy_grid: bool = True
    show_metrics: bool = True

class PygameVisualizer:
    """
    Real-time interactive visualization using Pygame
    Provides bird's-eye view with camera controls
    """
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.config.window_width, self.config.window_height)
        )
        pygame.display.set_caption("Autonomous Vehicle Simulation")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.small_font = pygame.font.Font(None, 24)
        
        # Camera control
        self.camera_x = 0
        self.camera_y = 0
        self.camera_zoom = 1.0
        self.follow_vehicle = True
        
        # Colors
        self.COLORS = {
            'background': (30, 30, 30),
            'road': (50, 50, 50),
            'lane_marking': (200, 200, 200),
            'ego_vehicle': (255, 50, 50),
            'traffic': (100, 100, 255),
            'pedestrian': (255, 200, 100),
            'obstacle': (200, 50, 50),
            'planned_path': (100, 255, 100),
            'trajectory': (255, 255, 100),
            'lidar': (0, 255, 255),
            'radar': (255, 0, 255),
            'text': (255, 255, 255)
        }
        
        # Performance tracking
        self.frame_times = []
        self.render_time = 0
        
    def world_to_screen(self, world_pos: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        ppm = self.config.pixels_per_meter * self.camera_zoom
        
        screen_x = int((world_pos[0] - self.camera_x) * ppm + self.config.window_width / 2)
        screen_y = int(self.config.window_height / 2 - (world_pos[1] - self.camera_y) * ppm)
        
        return (screen_x, screen_y)
    
    def screen_to_world(self, screen_pos: Tuple[int, int]) -> np.ndarray:
        """Convert screen coordinates to world coordinates"""
        ppm = self.config.pixels_per_meter * self.camera_zoom
        
        world_x = (screen_pos[0] - self.config.window_width / 2) / ppm + self.camera_x
        world_y = -((screen_pos[1] - self.config.window_height / 2) / ppm) + self.camera_y
        
        return np.array([world_x, world_y])
    
    def render_frame(self, ego_state: VehicleState, environment: Environment,
                    planned_path: List = None, trajectory: List = None,
                    lidar_points: List = None, metrics: Dict = None):
        """
        Render complete visualization frame
        
        Args:
            ego_state: Ego vehicle state
            environment: Simulation environment
            planned_path: Planned path waypoints
            trajectory: Optimized trajectory
            lidar_points: LiDAR point cloud
            metrics: Performance metrics
        """
        start_time = time.time()
        
        # Clear screen
        self.screen.fill(self.COLORS['background'])
        
        # Update camera to follow vehicle
        if self.follow_vehicle:
            self.camera_x = ego_state.x
            self.camera_y = ego_state.y
        
        # Render layers
        self._render_road_network(environment)
        
        if planned_path and self.config.show_trajectories:
            self._render_planned_path(planned_path)
        
        if trajectory and self.config.show_trajectories:
            self._render_trajectory(trajectory)
        
        if lidar_points and self.config.show_sensors:
            self._render_lidar_points(lidar_points)
        
        self._render_traffic(environment)
        self._render_pedestrians(environment)
        self._render_obstacles(environment)
        self._render_ego_vehicle(ego_state)
        
        if self.config.show_metrics and metrics:
            self._render_metrics(metrics, ego_state)
        
        # Render UI overlay
        self._render_ui_overlay()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.config.fps)
        
        # Track performance
        self.render_time = time.time() - start_time
        self.frame_times.append(self.render_time)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
    
    def _render_road_network(self, environment: Environment):
        """Render road network with lanes"""
        for segment in environment.road_network:
            for lane in segment.lanes:
                # Draw lane boundaries
                points = [self.world_to_screen(wp) for wp in lane.waypoints]
                
                if len(points) > 1:
                    # Lane centerline
                    pygame.draw.lines(self.screen, self.COLORS['lane_marking'],
                                    False, points, 1)
    
    def _render_planned_path(self, path: List):
        """Render global planned path"""
        points = []
        for wp in path:
            if hasattr(wp, 'x'):
                points.append(self.world_to_screen(np.array([wp.x, wp.y])))
            else:
                points.append(self.world_to_screen(wp))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.COLORS['planned_path'],
                            False, points, 3)
    
    def _render_trajectory(self, trajectory):
        """Render optimized trajectory"""
        if hasattr(trajectory, 'waypoints'):
            points = [self.world_to_screen(np.array([wp.x, wp.y])) 
                     for wp in trajectory.waypoints]
        else:
            points = [self.world_to_screen(p) for p in trajectory]
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.COLORS['trajectory'],
                            False, points, 2)
    
    def _render_lidar_points(self, lidar_points: List):
        """Render LiDAR point cloud"""
        for point in lidar_points[::10]:  # Subsample for performance
            screen_pos = self.world_to_screen(np.array([point.x, point.y]))
            
            # Color by intensity
            intensity = int(point.intensity * 255)
            color = (0, intensity, intensity)
            
            pygame.draw.circle(self.screen, color, screen_pos, 2)
    
    def _render_ego_vehicle(self, state: VehicleState):
        """Render ego vehicle with direction indicator"""
        pos = self.world_to_screen(np.array([state.x, state.y]))
        
        # Vehicle dimensions in pixels
        ppm = self.config.pixels_per_meter * self.camera_zoom
        length = int(4.5 * ppm)
        width = int(1.8 * ppm)
        
        # Create rotated rectangle
        rect_surface = pygame.Surface((length, width), pygame.SRCALPHA)
        rect_surface.fill(self.COLORS['ego_vehicle'])
        
        # Rotate
        rotated = pygame.transform.rotate(rect_surface, -np.rad2deg(state.yaw))
        rect_rect = rotated.get_rect(center=pos)
        
        self.screen.blit(rotated, rect_rect)
        
        # Direction indicator
        front_offset = np.array([
            length/2 * np.cos(state.yaw),
            length/2 * np.sin(state.yaw)
        ]) * (1.0 / ppm)
        
        front_pos = self.world_to_screen(np.array([state.x, state.y]) + front_offset)
        pygame.draw.circle(self.screen, (255, 255, 0), front_pos, 5)
    
    def _render_traffic(self, environment: Environment):
        """Render traffic vehicles"""
        ppm = self.config.pixels_per_meter * self.camera_zoom
        
        for vehicle in environment.traffic_sim.traffic_vehicles.values():
            pos = self.world_to_screen(vehicle.position)
            
            length = int(vehicle.dimensions[0] * ppm)
            width = int(vehicle.dimensions[1] * ppm)
            
            rect_surface = pygame.Surface((length, width), pygame.SRCALPHA)
            rect_surface.fill(self.COLORS['traffic'])
            
            rotated = pygame.transform.rotate(rect_surface, -np.rad2deg(vehicle.yaw))
            rect_rect = rotated.get_rect(center=pos)
            
            self.screen.blit(rotated, rect_rect)
    
    def _render_pedestrians(self, environment: Environment):
        """Render pedestrians"""
        for ped in environment.pedestrian_sim.pedestrians.values():
            pos = self.world_to_screen(ped.position)
            pygame.draw.circle(self.screen, self.COLORS['pedestrian'], pos, 5)
    
    def _render_obstacles(self, environment: Environment):
        """Render static obstacles"""
        for obs_pos, radius in environment.static_obstacles:
            pos = self.world_to_screen(obs_pos)
            ppm = self.config.pixels_per_meter * self.camera_zoom
            screen_radius = int(radius * ppm)
            pygame.draw.circle(self.screen, self.COLORS['obstacle'], pos, screen_radius)
    
    def _render_metrics(self, metrics: Dict, state: VehicleState):
        """Render performance metrics overlay"""
        y_offset = 20
        line_height = 30
        
        metric_texts = [
            f"Speed: {state.vx:.1f} m/s ({state.vx * 3.6:.1f} km/h)",
            f"Distance: {metrics.get('distance_traveled', 0):.1f} m",
            f"Avg Speed: {metrics.get('average_speed', 0):.1f} m/s",
            f"Comfort Violations: {metrics.get('comfort_violations', 0)}",
            f"FPS: {1.0 / np.mean(self.frame_times) if self.frame_times else 0:.1f}",
            f"Render Time: {self.render_time * 1000:.1f} ms"
        ]
        
        for text in metric_texts:
            surface = self.small_font.render(text, True, self.COLORS['text'])
            self.screen.blit(surface, (20, y_offset))
            y_offset += line_height
    
    def _render_ui_overlay(self):
        """Render UI controls overlay"""
        controls = [
            "Controls:",
            "Arrow Keys: Move Camera",
            "+/- : Zoom",
            "F: Toggle Follow",
            "Space: Pause",
            "ESC: Exit"
        ]
        
        y_offset = self.config.window_height - 200
        for text in controls:
            surface = self.small_font.render(text, True, self.COLORS['text'])
            self.screen.blit(surface, (self.config.window_width - 250, y_offset))
            y_offset += 25
    
    def handle_events(self) -> bool:
        """
        Handle user input events
        
        Returns:
            False if user wants to quit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_f:
                    self.follow_vehicle = not self.follow_vehicle
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.camera_zoom *= 1.2
                elif event.key == pygame.K_MINUS:
                    self.camera_zoom /= 1.2
            
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Mouse wheel up
                    self.camera_zoom *= 1.1
                elif event.button == 5:  # Mouse wheel down
                    self.camera_zoom /= 1.1
        
        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        camera_speed = 2.0 / self.camera_zoom
        
        if keys[pygame.K_LEFT]:
            self.camera_x -= camera_speed
            self.follow_vehicle = False
        if keys[pygame.K_RIGHT]:
            self.camera_x += camera_speed
            self.follow_vehicle = False
        if keys[pygame.K_UP]:
            self.camera_y += camera_speed
            self.follow_vehicle = False
        if keys[pygame.K_DOWN]:
            self.camera_y -= camera_speed
            self.follow_vehicle = False
        
        return True
    
    def cleanup(self):
        """Cleanup resources"""
        pygame.quit()

class CameraRenderer:
    """
    OpenCV-based camera view renderer
    Simulates first-person camera view from vehicle
    """
    
    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.fov = 90.0  # degrees
        
    def render_camera_view(self, vehicle_state: VehicleState,
                          environment: Environment) -> np.ndarray:
        """
        Render first-person camera view with proper 3D perspective
        
        Returns:
            RGB image array [height x width x 3]
        """
        # Create blank image
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Sky gradient with realistic coloring
        for i in range(self.height // 2):
            # Gradient from horizon (lighter) to zenith (darker)
            intensity = int(180 + (220 - 180) * (i / (self.height // 2)))
            sky_color = [intensity - 45, intensity - 25, 255]  # Bluish
            image[i, :] = sky_color
        
        # Ground with perspective darkening
        horizon_y = self.height // 2
        for i in range(horizon_y, self.height):
            # Darker ground farther away (perspective effect)
            distance_factor = (i - horizon_y) / (self.height - horizon_y)
            base_intensity = 100
            intensity = int(base_intensity - 30 * (1 - distance_factor))
            ground_color = [intensity, intensity, intensity]  # Gray ground
            image[i, :] = ground_color
        
        # Draw horizon line
        cv2.line(image, (0, horizon_y), (self.width, horizon_y), 
                (150, 150, 150), 1)
        
        # Camera position and orientation
        camera_pos = np.array([vehicle_state.x, vehicle_state.y, 1.5])  # 1.5m height
        camera_yaw = vehicle_state.yaw
        
        # Render perspective grid on ground for depth perception
        self._render_perspective_grid(image, camera_pos, camera_yaw)
        
        # Render road network
        self._render_road_in_view(image, camera_pos, camera_yaw, environment)
        
        # Render vehicles and objects
        self._render_objects_in_view(image, camera_pos, camera_yaw,
                                    environment.get_all_objects())
        
        # Add HUD overlay
        self._render_hud(image, vehicle_state)
        
        # Add distance markers
        self._render_distance_markers(image, camera_pos, camera_yaw)
        
        return image
    
    def _render_perspective_grid(self, image: np.ndarray, 
                                 camera_pos: np.ndarray, camera_yaw: float):
        """Render perspective grid on ground for depth perception"""
        grid_spacing = 10.0  # meters
        grid_range = 100.0   # meters ahead
        grid_width = 40.0    # meters to each side
        
        # Draw longitudinal lines (parallel to direction of travel)
        for offset in np.arange(-grid_width, grid_width + grid_spacing, grid_spacing):
            points_3d = []
            for distance in np.arange(5.0, grid_range, 5.0):
                # Point in vehicle coordinate frame
                local_x = distance
                local_y = offset
                
                # Transform to world coordinates
                world_x = camera_pos[0] + local_x * np.cos(camera_yaw) - local_y * np.sin(camera_yaw)
                world_y = camera_pos[1] + local_x * np.sin(camera_yaw) + local_y * np.cos(camera_yaw)
                
                point_3d = np.array([world_x, world_y, 0.0])
                points_3d.append(point_3d)
            
            # Project to screen
            screen_points = []
            for p in points_3d:
                sp = self._project_to_screen(p, camera_pos, camera_yaw)
                if sp is not None:
                    screen_points.append(sp)
            
            # Draw line segments
            if len(screen_points) > 1:
                for i in range(len(screen_points) - 1):
                    cv2.line(image, screen_points[i], screen_points[i+1], 
                           (80, 80, 80), 1)
        
        # Draw lateral lines (perpendicular to direction)
        for distance in np.arange(10.0, grid_range, 10.0):
            points_3d = []
            for offset in np.arange(-grid_width, grid_width + 2, 2.0):
                local_x = distance
                local_y = offset
                
                world_x = camera_pos[0] + local_x * np.cos(camera_yaw) - local_y * np.sin(camera_yaw)
                world_y = camera_pos[1] + local_x * np.sin(camera_yaw) + local_y * np.cos(camera_yaw)
                
                point_3d = np.array([world_x, world_y, 0.0])
                points_3d.append(point_3d)
            
            screen_points = []
            for p in points_3d:
                sp = self._project_to_screen(p, camera_pos, camera_yaw)
                if sp is not None:
                    screen_points.append(sp)
            
            if len(screen_points) > 1:
                for i in range(len(screen_points) - 1):
                    cv2.line(image, screen_points[i], screen_points[i+1], 
                           (80, 80, 80), 1)
    
    def _render_distance_markers(self, image: np.ndarray,
                                 camera_pos: np.ndarray, camera_yaw: float):
        """Render distance markers along the road"""
        for distance in [10, 25, 50, 75, 100]:
            # Position ahead
            marker_x = camera_pos[0] + distance * np.cos(camera_yaw)
            marker_y = camera_pos[1] + distance * np.sin(camera_yaw)
            marker_3d = np.array([marker_x, marker_y, 0.0])
            
            screen_pos = self._project_to_screen(marker_3d, camera_pos, camera_yaw)
            
            if screen_pos is not None:
                # Draw small marker
                cv2.circle(image, screen_pos, 3, (255, 255, 0), -1)
                
                # Add distance label
                label = f"{distance}m"
                cv2.putText(image, label, 
                          (screen_pos[0] + 5, screen_pos[1]),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    def _render_road_in_view(self, image: np.ndarray, camera_pos: np.ndarray,
                            camera_yaw: float, environment: Environment):
        """Render road in camera view with proper 3D perspective"""
        # Define road surface height
        road_height = 0.0
        
        for segment in environment.road_network:
            for lane in segment.lanes:
                # Render lane markings with 3D positioning
                lane_points_3d = []
                
                for i in range(len(lane.waypoints)):
                    wp = lane.waypoints[i]
                    # Create 3D point on road surface
                    point_3d = np.array([wp[0], wp[1], road_height])
                    lane_points_3d.append(point_3d)
                
                # Project all points to screen
                screen_points = []
                for point_3d in lane_points_3d:
                    screen_pos = self._project_to_screen(point_3d, camera_pos, camera_yaw)
                    if screen_pos is not None:
                        screen_points.append(screen_pos)
                
                # Draw lane markings
                if len(screen_points) > 1:
                    # Draw dashed lines for lane markings
                    for i in range(len(screen_points) - 1):
                        # Alternate dashed pattern
                        if i % 2 == 0:
                            cv2.line(image, screen_points[i], screen_points[i+1], 
                                   (255, 255, 255), 2)
                
                # Also render lane boundaries (edges)
                # Left boundary
                left_boundary_3d = []
                right_boundary_3d = []
                
                for wp in lane.waypoints:
                    # Calculate perpendicular offset for lane width
                    if len(lane.waypoints) > 1:
                        idx = list(lane.waypoints).index(wp)
                        if idx < len(lane.waypoints) - 1:
                            next_wp = lane.waypoints[idx + 1]
                            direction = next_wp - wp
                            direction = direction / (np.linalg.norm(direction) + 1e-6)
                            perpendicular = np.array([-direction[1], direction[0]])
                        else:
                            prev_wp = lane.waypoints[idx - 1]
                            direction = wp - prev_wp
                            direction = direction / (np.linalg.norm(direction) + 1e-6)
                            perpendicular = np.array([-direction[1], direction[0]])
                    else:
                        perpendicular = np.array([0, 1])
                    
                    half_width = lane.width / 2
                    left_point = wp + perpendicular * half_width
                    right_point = wp - perpendicular * half_width
                    
                    left_boundary_3d.append(np.array([left_point[0], left_point[1], road_height]))
                    right_boundary_3d.append(np.array([right_point[0], right_point[1], road_height]))
                
                # Project boundaries
                left_screen = [self._project_to_screen(p, camera_pos, camera_yaw) 
                              for p in left_boundary_3d]
                right_screen = [self._project_to_screen(p, camera_pos, camera_yaw) 
                               for p in right_boundary_3d]
                
                # Draw boundaries
                left_screen = [p for p in left_screen if p is not None]
                right_screen = [p for p in right_screen if p is not None]
                
                if len(left_screen) > 1:
                    for i in range(len(left_screen) - 1):
                        cv2.line(image, left_screen[i], left_screen[i+1], 
                               (200, 200, 200), 1)
                
                if len(right_screen) > 1:
                    for i in range(len(right_screen) - 1):
                        cv2.line(image, right_screen[i], right_screen[i+1], 
                               (200, 200, 200), 1)
    
    def _render_objects_in_view(self, image: np.ndarray, camera_pos: np.ndarray,
                               camera_yaw: float, objects: List[DynamicObject]):
        """Render objects in camera view with proper 3D positioning"""
        for obj in objects:
            # Calculate relative position
            rel_pos = obj.position - camera_pos[0:2]
            distance = np.linalg.norm(rel_pos)
            
            if distance > 100:  # Too far
                continue
            
            # Determine object height based on type
            if hasattr(obj, 'dimensions') and len(obj.dimensions) >= 3:
                obj_height = obj.dimensions[2]
            else:
                # Default heights by type
                if hasattr(obj, 'object_type'):
                    if 'vehicle' in str(obj.object_type).lower():
                        obj_height = 1.5
                    elif 'pedestrian' in str(obj.object_type).lower():
                        obj_height = 1.7
                    else:
                        obj_height = 1.0
                else:
                    obj_height = 1.5
            
            # Create 3D position for object center (mid-height)
            obj_pos_3d = np.array([obj.position[0], obj.position[1], obj_height / 2])
            
            # Project object center to screen
            screen_pos = self._project_to_screen(obj_pos_3d, camera_pos, camera_yaw)
            
            if screen_pos is not None:
                # Calculate apparent size (inversely proportional to distance)
                base_size = 50.0  # Base pixel size at 1 meter
                size = int(base_size / max(distance, 1.0))
                size = max(5, min(size, 100))  # Clamp size
                
                # Color based on object type
                if hasattr(obj, 'object_type'):
                    obj_type_str = str(obj.object_type).lower()
                    if 'vehicle' in obj_type_str:
                        color = (100, 100, 255)  # Blue for vehicles
                    elif 'pedestrian' in obj_type_str:
                        color = (255, 200, 100)  # Orange for pedestrians
                    else:
                        color = (200, 200, 200)  # Gray for others
                else:
                    color = (100, 100, 255)
                
                # Draw bounding box with depth-based positioning
                # Calculate vertical extent based on object height
                obj_bottom_3d = np.array([obj.position[0], obj.position[1], 0.0])
                obj_top_3d = np.array([obj.position[0], obj.position[1], obj_height])
                
                screen_bottom = self._project_to_screen(obj_bottom_3d, camera_pos, camera_yaw)
                screen_top = self._project_to_screen(obj_top_3d, camera_pos, camera_yaw)
                
                if screen_bottom and screen_top:
                    # Use actual projected height
                    box_height = abs(screen_bottom[1] - screen_top[1])
                    box_height = max(size, box_height)
                    
                    # Draw rectangle from bottom to top
                    top_left = (screen_pos[0] - size, screen_top[1])
                    bottom_right = (screen_pos[0] + size, screen_bottom[1])
                    
                    cv2.rectangle(image, top_left, bottom_right, color, 2)
                    
                    # Add distance label
                    label = f"{distance:.0f}m"
                    cv2.putText(image, label, 
                              (screen_pos[0] - 20, screen_top[1] - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                else:
                    # Fallback: simple rectangle if projection fails
                    cv2.rectangle(image,
                                (screen_pos[0] - size, screen_pos[1] - size),
                                (screen_pos[0] + size, screen_pos[1] + size),
                                color, 2)
    
    def _project_to_screen(self, world_pos: np.ndarray, camera_pos: np.ndarray,
                          camera_yaw: float) -> Optional[Tuple[int, int]]:
        """
        Project 3D world position to 2D screen coordinates
        
        Args:
            world_pos: 2D or 3D world position [x, y] or [x, y, z]
            camera_pos: Camera position [x, y, z]
            camera_yaw: Camera heading in radians
            
        Returns:
            (x, y) screen coordinates or None if behind camera
        """
        # Handle 2D input (assume ground level)
        if len(world_pos) == 2:
            world_pos_3d = np.array([world_pos[0], world_pos[1], 0.0])
        else:
            world_pos_3d = world_pos
        
        # Transform to camera coordinate system
        rel_x = world_pos_3d[0] - camera_pos[0]
        rel_y = world_pos_3d[1] - camera_pos[1]
        rel_z = world_pos_3d[2] - camera_pos[2]
        
        # Rotate to camera frame (yaw only for simplicity)
        cos_yaw = np.cos(-camera_yaw)
        sin_yaw = np.sin(-camera_yaw)
        
        cam_x = rel_x * cos_yaw - rel_y * sin_yaw  # Forward
        cam_y = rel_x * sin_yaw + rel_y * cos_yaw  # Right
        cam_z = rel_z  # Up
        
        # Check if in front of camera
        if cam_x < 0.1:
            return None
        
        # Project to screen using pinhole camera model
        fov_rad = np.deg2rad(self.fov)
        focal_length = self.width / (2 * np.tan(fov_rad / 2))
        
        # Horizontal projection (left-right)
        screen_x = int(self.width / 2 - (cam_y / cam_x) * focal_length)
        
        # Vertical projection (up-down) based on actual height
        # Objects higher than camera appear higher on screen
        screen_y = int(self.height / 2 - (cam_z / cam_x) * focal_length)
        
        # Check if in view
        if 0 <= screen_x < self.width and 0 <= screen_y < self.height:
            return (screen_x, screen_y)
        
        return None
    
    def _render_hud(self, image: np.ndarray, vehicle_state: VehicleState):
        """Render enhanced heads-up display with comprehensive telemetry"""
        # Background panel for HUD
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        # Vehicle speed
        speed_kmh = vehicle_state.vx * 3.6
        speed_text = f"Speed: {speed_kmh:.0f} km/h ({vehicle_state.vx:.1f} m/s)"
        cv2.putText(image, speed_text, (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Speed bar
        max_speed = 150  # km/h
        bar_width = int(300 * min(speed_kmh / max_speed, 1.0))
        cv2.rectangle(image, (20, 45), (20 + bar_width, 55), (0, 255, 0), -1)
        cv2.rectangle(image, (20, 45), (320, 55), (0, 255, 0), 1)
        
        # Steering angle
        steering_angle = 0.0  # Would come from controller
        steering_text = f"Steering: {np.rad2deg(steering_angle):.1f}°"
        cv2.putText(image, steering_text, (20, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
        
        # Acceleration
        accel_text = f"Accel: {vehicle_state.ax:.2f} m/s²"
        cv2.putText(image, accel_text, (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        
        # Lateral acceleration (important for stability)
        lateral_text = f"Lateral: {vehicle_state.ay:.2f} m/s²"
        cv2.putText(image, lateral_text, (20, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 100), 2)
        
        # Yaw rate
        yaw_rate_text = f"Yaw Rate: {np.rad2deg(vehicle_state.yaw_rate):.1f}°/s"
        cv2.putText(image, yaw_rate_text, (20, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 150, 255), 2)
        
        # Heading
        heading_deg = np.rad2deg(vehicle_state.yaw) % 360
        heading_text = f"Heading: {heading_deg:.0f}°"
        cv2.putText(image, heading_text, (20, 175),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Compass rose (top right)
        self._render_compass(image, vehicle_state.yaw)
        
        # Warnings if limits exceeded
        warning_y = self.height - 50
        
        if abs(vehicle_state.ay) > 5.0:
            cv2.putText(image, "⚠ HIGH LATERAL G!", 
                       (self.width // 2 - 100, warning_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        if speed_kmh > 140:
            cv2.putText(image, "⚠ EXCESSIVE SPEED!", 
                       (self.width // 2 - 120, warning_y - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    def _render_compass(self, image: np.ndarray, yaw: float):
        """Render compass rose in top-right corner"""
        center_x = self.width - 60
        center_y = 60
        radius = 40
        
        # Circle background
        overlay = image.copy()
        cv2.circle(overlay, (center_x, center_y), radius, (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        
        cv2.circle(image, (center_x, center_y), radius, (255, 255, 255), 2)
        
        # Cardinal directions
        directions = [
            (0, 'N', (0, 255, 0)),      # North - Green
            (90, 'E', (200, 200, 200)),  # East
            (180, 'S', (200, 200, 200)), # South
            (270, 'W', (200, 200, 200))  # West
        ]
        
        for angle_deg, label, color in directions:
            # Angle relative to vehicle heading
            angle_rad = np.deg2rad(angle_deg) - yaw
            
            # Position on circle
            dx = int(radius * 0.8 * np.sin(angle_rad))
            dy = int(-radius * 0.8 * np.cos(angle_rad))
            
            pos_x = center_x + dx
            pos_y = center_y + dy
            
            cv2.putText(image, label, (pos_x - 8, pos_y + 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Vehicle direction indicator (always pointing up in compass)
        indicator_len = int(radius * 0.6)
        cv2.arrowedLine(image, 
                       (center_x, center_y),
                       (center_x, center_y - indicator_len),
                       (255, 0, 0), 2, tipLength=0.3)
    
    def save_image(self, image: np.ndarray, filename: str):
        """Save rendered image to file"""
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))