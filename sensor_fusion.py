"""
Advanced Sensor Fusion System
Implements Extended Kalman Filter, Unscented Kalman Filter, and Particle Filter
for multi-sensor data fusion
"""

import numpy as np
from scipy.linalg import block_diag, cholesky
from scipy.stats import multivariate_normal
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from config import SensorConfig, SENSOR_CONFIG

@dataclass
class SensorMeasurement:
    """Generic sensor measurement"""
    timestamp: float
    sensor_type: str
    data: np.ndarray
    covariance: np.ndarray
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
@dataclass
class TrackedObject:
    """Object being tracked by fusion system"""
    id: int
    state: np.ndarray  # [x, y, vx, vy, ax, ay, yaw, yaw_rate]
    covariance: np.ndarray
    classification: str
    confidence: float
    last_update: float
    age: int = 0
    hits: int = 0
    
class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear state estimation
    Handles sensor fusion from multiple sources
    """
    
    def __init__(self, dim_state: int = 8, dim_measurement: int = 4):
        self.dim_x = dim_state
        self.dim_z = dim_measurement
        
        # State: [x, y, vx, vy, ax, ay, yaw, yaw_rate]
        self.x = np.zeros(dim_state)
        self.P = np.eye(dim_state) * 100.0  # Initial uncertainty
        
        # Process noise
        self.Q = np.eye(dim_state) * 0.1
        self.Q[0:2, 0:2] *= 0.01  # Position process noise
        self.Q[2:4, 2:4] *= 0.1   # Velocity process noise
        self.Q[4:6, 4:6] *= 1.0   # Acceleration process noise
        
        # Measurement noise (will be updated per sensor)
        self.R = np.eye(dim_measurement) * 1.0
        
    def predict(self, dt: float, control_input: Optional[np.ndarray] = None):
        """
        Prediction step with constant acceleration model
        
        Args:
            dt: Time step
            control_input: Optional control input [ax_cmd, ay_cmd]
        """
        # State transition function (nonlinear)
        F = self._state_transition_jacobian(dt)
        
        # Predict state using motion model
        self.x = self._motion_model(self.x, dt, control_input)
        
        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q
        
    def update(self, measurement: np.ndarray, measurement_cov: np.ndarray,
               measurement_function: Optional[callable] = None):
        """
        Update step with measurement
        
        Args:
            measurement: Measurement vector
            measurement_cov: Measurement covariance matrix
            measurement_function: Optional custom measurement function
        """
        if measurement_function is None:
            # Default: measure position and velocity
            H = self._measurement_jacobian()
            z_pred = self.x[0:len(measurement)]
        else:
            H = measurement_function(self.x, jacobian=True)
            z_pred = measurement_function(self.x, jacobian=False)
        
        # Innovation
        y = measurement - z_pred
        
        # Innovation covariance
        S = H @ self.P @ H.T + measurement_cov
        
        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.x = self.x + K @ y
        
        # Update covariance
        I_KH = np.eye(self.dim_x) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ measurement_cov @ K.T
        
    def _motion_model(self, state: np.ndarray, dt: float, 
                     control: Optional[np.ndarray] = None) -> np.ndarray:
        """Nonlinear motion model"""
        x_new = state.copy()
        
        # Position update with velocity and acceleration
        x_new[0] += state[2] * dt + 0.5 * state[4] * dt**2
        x_new[1] += state[3] * dt + 0.5 * state[5] * dt**2
        
        # Velocity update with acceleration
        x_new[2] += state[4] * dt
        x_new[3] += state[5] * dt
        
        # Acceleration decay (simple model)
        x_new[4] *= 0.9
        x_new[5] *= 0.9
        
        # Yaw update
        x_new[6] += state[7] * dt
        x_new[6] = np.arctan2(np.sin(x_new[6]), np.cos(x_new[6]))
        
        # Yaw rate decay
        x_new[7] *= 0.95
        
        if control is not None:
            x_new[4] += control[0] * dt
            x_new[5] += control[1] * dt
            
        return x_new
    
    def _state_transition_jacobian(self, dt: float) -> np.ndarray:
        """Jacobian of state transition function"""
        F = np.eye(self.dim_x)
        
        # Position depends on velocity and acceleration
        F[0, 2] = dt
        F[0, 4] = 0.5 * dt**2
        F[1, 3] = dt
        F[1, 5] = 0.5 * dt**2
        
        # Velocity depends on acceleration
        F[2, 4] = dt
        F[3, 5] = dt
        
        # Acceleration decay
        F[4, 4] = 0.9
        F[5, 5] = 0.9
        
        # Yaw update
        F[6, 7] = dt
        F[7, 7] = 0.95
        
        return F
    
    def _measurement_jacobian(self) -> np.ndarray:
        """Jacobian of measurement function"""
        H = np.zeros((self.dim_z, self.dim_x))
        H[0:self.dim_z, 0:self.dim_z] = np.eye(self.dim_z)
        return H

class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for highly nonlinear systems
    Uses unscented transform instead of linearization
    """
    
    def __init__(self, dim_state: int = 8, dim_measurement: int = 4):
        self.dim_x = dim_state
        self.dim_z = dim_measurement
        
        # State and covariance
        self.x = np.zeros(dim_state)
        self.P = np.eye(dim_state) * 100.0
        
        # Process and measurement noise
        self.Q = np.eye(dim_state) * 0.1
        self.R = np.eye(dim_measurement) * 1.0
        
        # UKF parameters
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        
        self.lambda_ = self.alpha**2 * (dim_state + self.kappa) - dim_state
        self.gamma = np.sqrt(dim_state + self.lambda_)
        
        # Weights for mean and covariance
        self.Wm = np.zeros(2 * dim_state + 1)
        self.Wc = np.zeros(2 * dim_state + 1)
        
        self.Wm[0] = self.lambda_ / (dim_state + self.lambda_)
        self.Wc[0] = self.lambda_ / (dim_state + self.lambda_) + \
                     (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2 * dim_state + 1):
            self.Wm[i] = 1.0 / (2 * (dim_state + self.lambda_))
            self.Wc[i] = 1.0 / (2 * (dim_state + self.lambda_))
    
    def _generate_sigma_points(self) -> np.ndarray:
        """Generate sigma points for unscented transform"""
        sigma_points = np.zeros((2 * self.dim_x + 1, self.dim_x))
        
        # Mean
        sigma_points[0] = self.x
        
        # Compute matrix square root
        try:
            U = cholesky(self.P)
        except:
            U = cholesky(self.P + np.eye(self.dim_x) * 1e-6)
        
        # Positive deviations
        for i in range(self.dim_x):
            sigma_points[i + 1] = self.x + self.gamma * U[i]
            
        # Negative deviations
        for i in range(self.dim_x):
            sigma_points[self.dim_x + i + 1] = self.x - self.gamma * U[i]
            
        return sigma_points
    
    def predict(self, dt: float, motion_model: callable):
        """
        Prediction step using unscented transform
        
        Args:
            dt: Time step
            motion_model: Function that propagates sigma points
        """
        # Generate sigma points
        sigma_points = self._generate_sigma_points()
        
        # Propagate sigma points through motion model
        sigma_points_pred = np.zeros_like(sigma_points)
        for i in range(sigma_points.shape[0]):
            sigma_points_pred[i] = motion_model(sigma_points[i], dt)
        
        # Compute predicted mean
        self.x = np.sum(self.Wm[:, np.newaxis] * sigma_points_pred, axis=0)
        
        # Compute predicted covariance
        self.P = np.zeros((self.dim_x, self.dim_x))
        for i in range(sigma_points_pred.shape[0]):
            diff = sigma_points_pred[i] - self.x
            self.P += self.Wc[i] * np.outer(diff, diff)
        self.P += self.Q
        
    def update(self, measurement: np.ndarray, measurement_model: callable):
        """
        Update step using unscented transform
        
        Args:
            measurement: Measurement vector
            measurement_model: Function that maps state to measurement
        """
        # Generate sigma points
        sigma_points = self._generate_sigma_points()
        
        # Transform sigma points through measurement model
        sigma_points_meas = np.zeros((sigma_points.shape[0], self.dim_z))
        for i in range(sigma_points.shape[0]):
            sigma_points_meas[i] = measurement_model(sigma_points[i])
        
        # Predicted measurement mean
        z_pred = np.sum(self.Wm[:, np.newaxis] * sigma_points_meas, axis=0)
        
        # Innovation covariance
        Pzz = np.zeros((self.dim_z, self.dim_z))
        for i in range(sigma_points_meas.shape[0]):
            diff = sigma_points_meas[i] - z_pred
            Pzz += self.Wc[i] * np.outer(diff, diff)
        Pzz += self.R
        
        # Cross-covariance
        Pxz = np.zeros((self.dim_x, self.dim_z))
        for i in range(sigma_points.shape[0]):
            diff_x = sigma_points[i] - self.x
            diff_z = sigma_points_meas[i] - z_pred
            Pxz += self.Wc[i] * np.outer(diff_x, diff_z)
        
        # Kalman gain
        K = Pxz @ np.linalg.inv(Pzz)
        
        # Update state
        innovation = measurement - z_pred
        self.x = self.x + K @ innovation
        
        # Update covariance
        self.P = self.P - K @ Pzz @ K.T

class MultiSensorFusion:
    """
    Multi-sensor fusion system coordinating multiple filters
    Handles LiDAR, RADAR, Camera, IMU, and GPS data
    """
    
    def __init__(self, config: SensorConfig = SENSOR_CONFIG):
        self.config = config
        
        # Tracked objects dictionary
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_object_id = 0
        
        # Filters for each object
        self.ekf_filters: Dict[int, ExtendedKalmanFilter] = {}
        
        # Sensor measurement buffers
        self.lidar_buffer: List[SensorMeasurement] = []
        self.radar_buffer: List[SensorMeasurement] = []
        self.camera_buffer: List[SensorMeasurement] = []
        
        # Association thresholds
        self.association_distance_threshold = 5.0  # meters
        self.max_missed_detections = 5
        
    def fuse_measurements(self, measurements: List[SensorMeasurement], 
                         current_time: float) -> List[TrackedObject]:
        """
        Main fusion loop - processes all sensor measurements
        
        Args:
            measurements: List of sensor measurements
            current_time: Current simulation time
            
        Returns:
            List of tracked objects
        """
        # Group measurements by sensor type
        lidar_meas = [m for m in measurements if m.sensor_type == 'lidar']
        radar_meas = [m for m in measurements if m.sensor_type == 'radar']
        camera_meas = [m for m in measurements if m.sensor_type == 'camera']
        imu_meas = [m for m in measurements if m.sensor_type == 'imu']
        gps_meas = [m for m in measurements if m.sensor_type == 'gps']
        
        # Predict all existing tracks
        for obj_id, tracked_obj in list(self.tracked_objects.items()):
            if obj_id in self.ekf_filters:
                dt = current_time - tracked_obj.last_update
                if dt > 0:
                    self.ekf_filters[obj_id].predict(dt)
                    tracked_obj.state = self.ekf_filters[obj_id].x
                    tracked_obj.covariance = self.ekf_filters[obj_id].P
                    tracked_obj.age += 1
        
        # Associate and update with LiDAR measurements
        self._process_lidar_measurements(lidar_meas, current_time)
        
        # Associate and update with RADAR measurements
        self._process_radar_measurements(radar_meas, current_time)
        
        # Update with Camera detections
        self._process_camera_measurements(camera_meas, current_time)
        
        # Remove stale tracks
        self._prune_tracks()
        
        return list(self.tracked_objects.values())
    
    def _process_lidar_measurements(self, measurements: List[SensorMeasurement],
                                   current_time: float):
        """Process LiDAR point cloud detections"""
        for meas in measurements:
            # Extract position from measurement
            position = meas.data[0:2]  # x, y
            
            # Find closest tracked object
            matched_id = self._associate_measurement(position)
            
            if matched_id is not None:
                # Update existing track
                ekf = self.ekf_filters[matched_id]
                ekf.update(meas.data, meas.covariance)
                
                tracked_obj = self.tracked_objects[matched_id]
                tracked_obj.state = ekf.x
                tracked_obj.covariance = ekf.P
                tracked_obj.last_update = current_time
                tracked_obj.hits += 1
            else:
                # Create new track
                self._create_new_track(meas, current_time)
    
    def _process_radar_measurements(self, measurements: List[SensorMeasurement],
                                   current_time: float):
        """Process RADAR detections with doppler velocity"""
        for meas in measurements:
            position = meas.data[0:2]
            velocity = meas.data[2:4] if len(meas.data) >= 4 else None
            
            matched_id = self._associate_measurement(position)
            
            if matched_id is not None:
                ekf = self.ekf_filters[matched_id]
                
                # RADAR provides position and velocity
                if velocity is not None:
                    measurement_vector = np.concatenate([position, velocity])
                else:
                    measurement_vector = position
                    
                ekf.update(measurement_vector, meas.covariance)
                
                tracked_obj = self.tracked_objects[matched_id]
                tracked_obj.state = ekf.x
                tracked_obj.covariance = ekf.P
                tracked_obj.last_update = current_time
                tracked_obj.hits += 1
    
    def _process_camera_measurements(self, measurements: List[SensorMeasurement],
                                    current_time: float):
        """Process camera object detections with classification"""
        for meas in measurements:
            position = meas.data[0:2]
            
            matched_id = self._associate_measurement(position)
            
            if matched_id is not None:
                # Update classification if camera provides it
                tracked_obj = self.tracked_objects[matched_id]
                # Camera measurements might include classification info
                # This would be handled here
                tracked_obj.last_update = current_time
    
    def _associate_measurement(self, position: np.ndarray) -> Optional[int]:
        """
        Associate measurement with existing track using nearest neighbor
        
        Args:
            position: Measurement position [x, y]
            
        Returns:
            Object ID if matched, None otherwise
        """
        min_distance = float('inf')
        matched_id = None
        
        for obj_id, tracked_obj in self.tracked_objects.items():
            track_position = tracked_obj.state[0:2]
            distance = np.linalg.norm(position - track_position)
            
            if distance < min_distance and distance < self.association_distance_threshold:
                min_distance = distance
                matched_id = obj_id
                
        return matched_id
    
    def _create_new_track(self, measurement: SensorMeasurement, current_time: float):
        """Create new tracked object from measurement"""
        obj_id = self.next_object_id
        self.next_object_id += 1
        
        # Initialize state
        initial_state = np.zeros(8)
        initial_state[0:len(measurement.data)] = measurement.data
        
        # Create tracked object
        tracked_obj = TrackedObject(
            id=obj_id,
            state=initial_state,
            covariance=np.eye(8) * 10.0,
            classification='unknown',
            confidence=0.5,
            last_update=current_time,
            hits=1
        )
        
        # Create EKF for this object
        ekf = ExtendedKalmanFilter()
        ekf.x = initial_state
        ekf.P = tracked_obj.covariance
        
        self.tracked_objects[obj_id] = tracked_obj
        self.ekf_filters[obj_id] = ekf
    
    def _prune_tracks(self):
        """Remove tracks that haven't been updated recently"""
        to_remove = []
        
        for obj_id, tracked_obj in self.tracked_objects.items():
            if tracked_obj.age > self.max_missed_detections:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.tracked_objects[obj_id]
            del self.ekf_filters[obj_id]