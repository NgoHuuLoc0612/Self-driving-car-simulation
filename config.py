"""
Autonomous Vehicle Simulation Configuration
Supports multiple AI theories, sensor fusion, and advanced control systems
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from enum import Enum

class VehicleType(Enum):
    SEDAN = "sedan"
    SUV = "suv"
    TRUCK = "truck"
    SPORTS = "sports"

class ControllerType(Enum):
    PID = "pid"
    MPC = "mpc"
    DEEP_RL = "deep_rl"
    HYBRID = "hybrid"

class SensorType(Enum):
    LIDAR = "lidar"
    RADAR = "radar"
    CAMERA = "camera"
    ULTRASONIC = "ultrasonic"
    IMU = "imu"
    GPS = "gps"

@dataclass
class SimulationConfig:
    """Master simulation configuration"""
    # Simulation parameters
    dt: float = 0.02  # 50Hz update rate
    simulation_duration: float = 300.0  # seconds
    real_time_factor: float = 1.0
    
    # Physics parameters
    gravity: float = 9.81
    air_density: float = 1.225
    road_friction_coefficient: float = 0.8
    
    # Environment
    map_size: Tuple[int, int] = (2000, 2000)  # meters
    num_lanes: int = 6
    lane_width: float = 3.5
    
    # Traffic
    traffic_density: float = 0.3
    pedestrian_density: float = 0.1
    weather_conditions: str = "clear"
    
@dataclass
class VehicleConfig:
    """Vehicle physical and dynamic properties"""
    vehicle_type: VehicleType = VehicleType.SEDAN
    
    # Physical dimensions
    length: float = 4.5
    width: float = 1.8
    height: float = 1.4
    wheelbase: float = 2.7
    track_width: float = 1.5
    
    # Mass properties
    mass: float = 1500.0  # kg
    moment_of_inertia: float = 2500.0  # kg*m^2
    center_of_gravity_height: float = 0.5
    
    # Powertrain
    max_engine_power: float = 150000.0  # Watts
    max_torque: float = 300.0  # Nm
    max_brake_torque: float = 5000.0  # Nm
    transmission_efficiency: float = 0.9
    
    # Aerodynamics
    drag_coefficient: float = 0.3
    frontal_area: float = 2.2  # m^2
    downforce_coefficient: float = 0.1
    
    # Tire model parameters (Pacejka Magic Formula)
    tire_stiffness_front: float = 120000.0  # N/rad
    tire_stiffness_rear: float = 100000.0
    tire_radius: float = 0.32
    tire_rolling_resistance: float = 0.015
    
    # Steering
    max_steering_angle: float = np.deg2rad(35)
    steering_ratio: float = 16.0
    max_steering_rate: float = np.deg2rad(540)  # deg/s
    
@dataclass
class SensorConfig:
    """Sensor suite configuration"""
    # LiDAR
    lidar_range: float = 200.0  # meters
    lidar_fov: float = 360.0  # degrees
    lidar_resolution: float = 0.1  # degrees
    lidar_layers: int = 64
    lidar_frequency: float = 10.0  # Hz
    lidar_noise_std: float = 0.03
    
    # RADAR
    radar_range: float = 250.0
    radar_fov: float = 120.0
    radar_resolution_range: float = 0.1
    radar_resolution_azimuth: float = 1.0
    radar_frequency: float = 20.0
    
    # Camera
    camera_fov: float = 120.0
    camera_resolution: Tuple[int, int] = (1920, 1080)
    camera_framerate: float = 30.0
    num_cameras: int = 6  # surround view
    
    # IMU
    imu_gyro_noise: float = 0.001  # rad/s
    imu_accel_noise: float = 0.01  # m/s^2
    imu_frequency: float = 100.0
    
    # GPS
    gps_horizontal_accuracy: float = 0.5  # meters
    gps_vertical_accuracy: float = 1.0
    gps_frequency: float = 10.0
    
@dataclass
class PerceptionConfig:
    """Perception system configuration"""
    # Object detection
    detection_confidence_threshold: float = 0.7
    nms_iou_threshold: float = 0.5
    max_detection_range: float = 150.0
    
    # Tracking
    tracking_max_age: int = 30
    tracking_min_hits: int = 3
    tracking_iou_threshold: float = 0.3
    
    # Semantic segmentation
    segmentation_classes: int = 23
    segmentation_resolution: Tuple[int, int] = (640, 480)
    
    # Sensor fusion
    fusion_method: str = "extended_kalman_filter"
    measurement_noise_covariance: float = 0.1
    process_noise_covariance: float = 0.01
    
@dataclass
class PlanningConfig:
    """Path planning and decision making configuration"""
    # Global planning
    planning_horizon: float = 100.0  # meters
    planning_resolution: float = 0.5
    planning_frequency: float = 1.0  # Hz
    
    # Local planning
    local_planning_horizon: float = 50.0
    local_planning_timesteps: int = 50
    trajectory_sampling_width: float = 3.0
    trajectory_sampling_resolution: float = 0.5
    
    # Cost function weights
    weight_progress: float = 1.0
    weight_comfort: float = 0.5
    weight_safety: float = 2.0
    weight_efficiency: float = 0.3
    weight_legality: float = 1.5
    
    # Behavioral planning
    behavior_horizon: float = 5.0  # seconds
    maneuver_types: List[str] = field(default_factory=lambda: [
        'lane_follow', 'lane_change_left', 'lane_change_right',
        'overtake', 'merge', 'stop', 'yield', 'turn_left', 'turn_right'
    ])
    
@dataclass
class ControlConfig:
    """Vehicle control system configuration"""
    controller_type: ControllerType = ControllerType.MPC
    control_frequency: float = 50.0  # Hz
    
    # PID parameters
    pid_kp_longitudinal: float = 0.8
    pid_ki_longitudinal: float = 0.1
    pid_kd_longitudinal: float = 0.2
    
    pid_kp_lateral: float = 1.2
    pid_ki_lateral: float = 0.05
    pid_kd_lateral: float = 0.3
    
    # MPC parameters
    mpc_horizon: int = 20
    mpc_dt: float = 0.1
    mpc_max_iterations: int = 100
    mpc_tolerance: float = 1e-4
    
    # State constraints
    max_velocity: float = 40.0  # m/s (144 km/h)
    max_acceleration: float = 3.0  # m/s^2
    max_deceleration: float = 8.0
    max_jerk: float = 2.0  # m/s^3
    max_lateral_acceleration: float = 5.0
    
    # Deep RL parameters
    rl_state_dim: int = 64
    rl_action_dim: int = 2
    rl_hidden_layers: List[int] = field(default_factory=lambda: [256, 256, 128])
    rl_learning_rate: float = 3e-4
    rl_gamma: float = 0.99
    rl_batch_size: int = 256
    
@dataclass
class SafetyConfig:
    """Safety and collision avoidance configuration"""
    # Collision avoidance
    min_safe_distance: float = 10.0  # meters
    time_to_collision_threshold: float = 3.0  # seconds
    emergency_brake_threshold: float = 1.5
    
    # Safety zones
    personal_space_front: float = 15.0
    personal_space_rear: float = 10.0
    personal_space_side: float = 2.0
    
    # Redundancy
    sensor_redundancy_level: int = 2
    controller_redundancy: bool = True
    fail_safe_mode: str = "minimal_risk_condition"
    
    # ISO 26262 compliance
    asil_level: str = "ASIL-D"
    safety_monitoring_frequency: float = 100.0

# Global configuration instances
SIM_CONFIG = SimulationConfig()
VEHICLE_CONFIG = VehicleConfig()
SENSOR_CONFIG = SensorConfig()
PERCEPTION_CONFIG = PerceptionConfig()
PLANNING_CONFIG = PlanningConfig()
CONTROL_CONFIG = ControlConfig()
SAFETY_CONFIG = SafetyConfig()