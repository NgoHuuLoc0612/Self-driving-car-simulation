"""
Simulation Environment
Handles world simulation, traffic, obstacles, and physics
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from config import SimulationConfig, SIM_CONFIG

class ObjectType(Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"
    STATIC_OBSTACLE = "static"

@dataclass
class DynamicObject:
    """Dynamic object in the environment"""
    id: int
    object_type: ObjectType
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy]
    yaw: float
    dimensions: np.ndarray  # [length, width, height]
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    def update(self, dt: float):
        """Update object state"""
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

@dataclass
class Lane:
    """Road lane representation"""
    id: int
    waypoints: np.ndarray  # [N x 2] centerline points
    width: float
    speed_limit: float
    lane_type: str  # 'driving', 'bus', 'bike', 'parking'
    left_neighbor: Optional[int] = None
    right_neighbor: Optional[int] = None

@dataclass
class RoadSegment:
    """Road segment with multiple lanes"""
    id: int
    lanes: List[Lane]
    length: float
    curvature: float
    road_type: str  # 'highway', 'urban', 'residential'

class TrafficSimulator:
    """
    Intelligent Driver Model (IDM) based traffic simulation
    Simulates realistic traffic behavior
    """
    
    def __init__(self, config: SimulationConfig = SIM_CONFIG):
        self.config = config
        
        # IDM parameters
        self.desired_velocity = 30.0  # m/s
        self.time_headway = 1.5  # seconds
        self.min_spacing = 2.0  # meters
        self.max_acceleration = 2.0  # m/s^2
        self.comfortable_deceleration = 3.0  # m/s^2
        self.acceleration_exponent = 4
        
        # Traffic vehicles
        self.traffic_vehicles: Dict[int, DynamicObject] = {}
        self.next_vehicle_id = 1000
        
    def spawn_traffic(self, road_network: List[RoadSegment], 
                     num_vehicles: int):
        """
        Spawn traffic vehicles on road network
        
        Args:
            road_network: List of road segments
            num_vehicles: Number of vehicles to spawn
        """
        for i in range(num_vehicles):
            # Select random lane
            segment = np.random.choice(road_network)
            lane = np.random.choice(segment.lanes)
            
            # Random position along lane
            idx = np.random.randint(0, len(lane.waypoints))
            position = lane.waypoints[idx].copy()
            
            # Initial velocity (near speed limit with variance)
            velocity_mag = lane.speed_limit * np.random.uniform(0.8, 1.0)
            
            # Direction from lane
            if idx < len(lane.waypoints) - 1:
                direction = lane.waypoints[idx + 1] - lane.waypoints[idx]
                direction = direction / np.linalg.norm(direction)
            else:
                direction = lane.waypoints[idx] - lane.waypoints[idx - 1]
                direction = direction / np.linalg.norm(direction)
            
            velocity = direction * velocity_mag
            yaw = np.arctan2(direction[1], direction[0])
            
            # Vehicle dimensions (random car type)
            dimensions = np.array([
                np.random.uniform(4.0, 5.0),  # length
                np.random.uniform(1.7, 2.0),  # width
                np.random.uniform(1.4, 1.6)   # height
            ])
            
            vehicle = DynamicObject(
                id=self.next_vehicle_id,
                object_type=ObjectType.VEHICLE,
                position=position,
                velocity=velocity,
                yaw=yaw,
                dimensions=dimensions
            )
            
            self.traffic_vehicles[self.next_vehicle_id] = vehicle
            self.next_vehicle_id += 1
    
    def update_traffic(self, dt: float, road_network: List[RoadSegment]):
        """
        Update all traffic vehicles using IDM
        
        Args:
            dt: Time step
            road_network: Road network for lane following
        """
        for vehicle in self.traffic_vehicles.values():
            # Find lead vehicle
            lead_vehicle, distance = self._find_lead_vehicle(vehicle)
            
            # Calculate IDM acceleration
            acceleration = self._idm_acceleration(
                vehicle, lead_vehicle, distance
            )
            
            # Apply acceleration (in vehicle's direction)
            direction = np.array([np.cos(vehicle.yaw), np.sin(vehicle.yaw)])
            vehicle.acceleration = direction * acceleration
            
            # Lane following behavior
            target_point = self._get_lane_target(vehicle, road_network)
            if target_point is not None:
                # Lateral correction
                lateral_error = self._calculate_lateral_error(
                    vehicle.position, vehicle.yaw, target_point
                )
                lateral_correction = -0.5 * lateral_error
                
                # Adjust yaw
                target_yaw = np.arctan2(
                    target_point[1] - vehicle.position[1],
                    target_point[0] - vehicle.position[0]
                )
                yaw_error = target_yaw - vehicle.yaw
                yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))
                vehicle.yaw += 0.1 * yaw_error
            
            # Update state
            vehicle.update(dt)
    
    def _idm_acceleration(self, vehicle: DynamicObject,
                         lead_vehicle: Optional[DynamicObject],
                         distance: float) -> float:
        """
        Calculate acceleration using Intelligent Driver Model
        
        Args:
            vehicle: Subject vehicle
            lead_vehicle: Leading vehicle (if any)
            distance: Distance to lead vehicle
            
        Returns:
            Acceleration command
        """
        v = np.linalg.norm(vehicle.velocity)
        v_desired = self.desired_velocity
        
        # Free road acceleration
        free_term = 1.0 - (v / v_desired) ** self.acceleration_exponent
        
        if lead_vehicle is not None:
            # Interaction term
            v_lead = np.linalg.norm(lead_vehicle.velocity)
            delta_v = v - v_lead
            
            # Desired spacing
            s_star = self.min_spacing + v * self.time_headway + \
                    (v * delta_v) / (2 * np.sqrt(self.max_acceleration * 
                                                 self.comfortable_deceleration))
            
            # Interaction term
            interaction_term = (s_star / max(distance, 0.1)) ** 2
            
            acceleration = self.max_acceleration * (free_term - interaction_term)
        else:
            acceleration = self.max_acceleration * free_term
        
        return acceleration
    
    def _find_lead_vehicle(self, vehicle: DynamicObject) -> Tuple[Optional[DynamicObject], float]:
        """Find leading vehicle in same lane"""
        min_distance = float('inf')
        lead_vehicle = None
        
        direction = np.array([np.cos(vehicle.yaw), np.sin(vehicle.yaw)])
        
        for other in self.traffic_vehicles.values():
            if other.id == vehicle.id:
                continue
            
            # Check if ahead
            to_other = other.position - vehicle.position
            distance = np.linalg.norm(to_other)
            
            # Check if in front (dot product > 0)
            if np.dot(to_other, direction) > 0:
                # Check lateral alignment
                lateral_offset = abs(np.cross(to_other, direction))
                
                if lateral_offset < 2.0 and distance < min_distance:
                    min_distance = distance
                    lead_vehicle = other
        
        return lead_vehicle, min_distance
    
    def _get_lane_target(self, vehicle: DynamicObject,
                        road_network: List[RoadSegment]) -> Optional[np.ndarray]:
        """Get target point for lane following"""
        # Find closest lane waypoint
        min_dist = float('inf')
        target_point = None
        
        for segment in road_network:
            for lane in segment.lanes:
                for i, waypoint in enumerate(lane.waypoints):
                    dist = np.linalg.norm(waypoint - vehicle.position)
                    if dist < min_dist:
                        min_dist = dist
                        # Look ahead
                        lookahead_idx = min(i + 5, len(lane.waypoints) - 1)
                        target_point = lane.waypoints[lookahead_idx]
        
        return target_point
    
    def _calculate_lateral_error(self, position: np.ndarray,
                                 yaw: float, target: np.ndarray) -> float:
        """Calculate lateral error from lane centerline"""
        to_target = target - position
        direction = np.array([np.cos(yaw), np.sin(yaw)])
        lateral = np.cross(to_target, direction)
        return lateral

class PedestrianSimulator:
    """
    Social Force Model for pedestrian simulation
    Simulates realistic pedestrian behavior
    """
    
    def __init__(self):
        # Social force parameters
        self.desired_speed = 1.4  # m/s
        self.relaxation_time = 0.5
        self.personal_space_range = 2.0
        self.personal_space_strength = 10.0
        
        self.pedestrians: Dict[int, DynamicObject] = {}
        self.next_pedestrian_id = 2000
    
    def spawn_pedestrians(self, crosswalks: List[np.ndarray], num_pedestrians: int):
        """Spawn pedestrians at crosswalks"""
        for i in range(num_pedestrians):
            # Random crosswalk
            crosswalk = crosswalks[np.random.randint(0, len(crosswalks))]
            
            # Position on crosswalk
            position = crosswalk + np.random.randn(2) * 0.5
            
            # Random initial velocity
            velocity = np.random.randn(2) * 0.5
            
            pedestrian = DynamicObject(
                id=self.next_pedestrian_id,
                object_type=ObjectType.PEDESTRIAN,
                position=position,
                velocity=velocity,
                yaw=0.0,
                dimensions=np.array([0.5, 0.5, 1.7])
            )
            
            self.pedestrians[self.next_pedestrian_id] = pedestrian
            self.next_pedestrian_id += 1
    
    def update_pedestrians(self, dt: float, goals: Dict[int, np.ndarray]):
        """
        Update pedestrians using Social Force Model
        
        Args:
            dt: Time step
            goals: Dictionary mapping pedestrian ID to goal position
        """
        for ped_id, pedestrian in self.pedestrians.items():
            # Goal attraction force
            goal = goals.get(ped_id, pedestrian.position + np.array([1.0, 0.0]))
            force_goal = self._goal_force(pedestrian, goal)
            
            # Repulsion from other pedestrians
            force_repulsion = np.zeros(2)
            for other in self.pedestrians.values():
                if other.id != pedestrian.id:
                    force_repulsion += self._pedestrian_repulsion(
                        pedestrian, other
                    )
            
            # Total force
            total_force = force_goal + force_repulsion
            
            # Update
            pedestrian.acceleration = total_force
            pedestrian.update(dt)
            
            # Update heading
            if np.linalg.norm(pedestrian.velocity) > 0.1:
                pedestrian.yaw = np.arctan2(
                    pedestrian.velocity[1],
                    pedestrian.velocity[0]
                )
    
    def _goal_force(self, pedestrian: DynamicObject, goal: np.ndarray) -> np.ndarray:
        """Calculate force towards goal"""
        direction = goal - pedestrian.position
        distance = np.linalg.norm(direction)
        
        if distance < 0.1:
            return np.zeros(2)
        
        direction = direction / distance
        desired_velocity = direction * self.desired_speed
        
        force = (desired_velocity - pedestrian.velocity) / self.relaxation_time
        return force
    
    def _pedestrian_repulsion(self, pedestrian: DynamicObject,
                             other: DynamicObject) -> np.ndarray:
        """Calculate repulsion force from other pedestrian"""
        diff = pedestrian.position - other.position
        distance = np.linalg.norm(diff)
        
        if distance < 0.1 or distance > self.personal_space_range:
            return np.zeros(2)
        
        direction = diff / distance
        magnitude = self.personal_space_strength * np.exp(
            -distance / self.personal_space_range
        )
        
        return direction * magnitude

class Environment:
    """
    Complete simulation environment
    Manages all dynamic objects, road network, and physics
    """
    
    def __init__(self, config: SimulationConfig = SIM_CONFIG):
        self.config = config
        
        # Time
        self.current_time = 0.0
        
        # Road network
        self.road_network: List[RoadSegment] = []
        self._initialize_road_network()
        
        # Simulators
        self.traffic_sim = TrafficSimulator(config)
        self.pedestrian_sim = PedestrianSimulator()
        
        # Static obstacles
        self.static_obstacles: List[Tuple[np.ndarray, float]] = []
        self._initialize_static_obstacles()
        
        # Spawn initial traffic
        num_traffic = int(config.traffic_density * 100)
        self.traffic_sim.spawn_traffic(self.road_network, num_traffic)
        
        # Spawn pedestrians
        crosswalks = self._get_crosswalk_positions()
        num_pedestrians = int(config.pedestrian_density * 50)
        self.pedestrian_sim.spawn_pedestrians(crosswalks, num_pedestrians)
    
    def _initialize_road_network(self):
        """Create road network"""
        # Create straight highway segment
        lane_width = self.config.lane_width
        num_lanes = self.config.num_lanes
        
        for lane_idx in range(num_lanes):
            # Create waypoints
            waypoints = []
            y_offset = lane_idx * lane_width - (num_lanes * lane_width) / 2
            
            for x in np.linspace(0, self.config.map_size[0], 100):
                waypoints.append(np.array([x, y_offset]))
            
            waypoints = np.array(waypoints)
            
            lane = Lane(
                id=lane_idx,
                waypoints=waypoints,
                width=lane_width,
                speed_limit=30.0,
                lane_type='driving',
                left_neighbor=lane_idx - 1 if lane_idx > 0 else None,
                right_neighbor=lane_idx + 1 if lane_idx < num_lanes - 1 else None
            )
            
            # Add to segment
            if lane_idx == 0:
                segment = RoadSegment(
                    id=0,
                    lanes=[lane],
                    length=self.config.map_size[0],
                    curvature=0.0,
                    road_type='highway'
                )
                self.road_network.append(segment)
            else:
                self.road_network[0].lanes.append(lane)
    
    def _initialize_static_obstacles(self):
        """Add static obstacles to environment"""
        # Add some random obstacles
        for i in range(20):
            position = np.array([
                np.random.uniform(0, self.config.map_size[0]),
                np.random.uniform(-50, 50)
            ])
            radius = np.random.uniform(0.5, 2.0)
            self.static_obstacles.append((position, radius))
    
    def _get_crosswalk_positions(self) -> List[np.ndarray]:
        """Get crosswalk positions"""
        crosswalks = []
        for x in np.linspace(100, self.config.map_size[0] - 100, 5):
            crosswalks.append(np.array([x, 0.0]))
        return crosswalks
    
    def step(self, dt: float):
        """
        Advance simulation by one time step
        
        Args:
            dt: Time step in seconds
        """
        # Update traffic
        self.traffic_sim.update_traffic(dt, self.road_network)
        
        # Update pedestrians
        pedestrian_goals = {
            ped_id: ped.position + np.array([10.0, 0.0])
            for ped_id, ped in self.pedestrian_sim.pedestrians.items()
        }
        self.pedestrian_sim.update_pedestrians(dt, pedestrian_goals)
        
        # Update time
        self.current_time += dt
    
    def get_all_objects(self) -> List[DynamicObject]:
        """Get all dynamic objects in environment"""
        objects = []
        objects.extend(self.traffic_sim.traffic_vehicles.values())
        objects.extend(self.pedestrian_sim.pedestrians.values())
        return objects
    
    def get_obstacles_in_range(self, position: np.ndarray, 
                               range_m: float) -> List[Tuple[np.ndarray, float]]:
        """Get obstacles within range of position"""
        obstacles = []
        
        # Static obstacles
        for obs_pos, radius in self.static_obstacles:
            if np.linalg.norm(obs_pos - position) < range_m:
                obstacles.append((obs_pos, radius))
        
        # Dynamic objects
        for obj in self.get_all_objects():
            if np.linalg.norm(obj.position - position) < range_m:
                # Approximate as circle
                radius = max(obj.dimensions[0], obj.dimensions[1]) / 2
                obstacles.append((obj.position, radius))
        
        return obstacles
    
    def check_collision(self, position: np.ndarray, 
                       vehicle_dimensions: np.ndarray) -> bool:
        """
        Check if position collides with any object
        
        Args:
            position: Position to check
            vehicle_dimensions: [length, width, height]
            
        Returns:
            True if collision detected
        """
        vehicle_radius = max(vehicle_dimensions[0], vehicle_dimensions[1]) / 2
        
        # Check static obstacles
        for obs_pos, obs_radius in self.static_obstacles:
            if np.linalg.norm(position - obs_pos) < (vehicle_radius + obs_radius):
                return True
        
        # Check dynamic objects
        for obj in self.get_all_objects():
            obj_radius = max(obj.dimensions[0], obj.dimensions[1]) / 2
            if np.linalg.norm(position - obj.position) < (vehicle_radius + obj_radius):
                return True
        
        return False
    
    def get_lane_info(self, position: np.ndarray) -> Optional[Lane]:
        """Get lane information at position"""
        min_dist = float('inf')
        closest_lane = None
        
        for segment in self.road_network:
            for lane in segment.lanes:
                for waypoint in lane.waypoints:
                    dist = np.linalg.norm(waypoint - position)
                    if dist < min_dist:
                        min_dist = dist
                        closest_lane = lane
        
        return closest_lane if min_dist < 10.0 else None