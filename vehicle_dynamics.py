"""
Advanced Vehicle Dynamics Model
Implements comprehensive multi-body dynamics, tire models, and powertrain simulation
"""

import numpy as np
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Tuple, Optional
from config import VehicleConfig, VEHICLE_CONFIG

@dataclass
class VehicleState:
    """Complete vehicle state representation"""
    # Position and orientation (global frame)
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    
    # Velocities (body frame)
    vx: float = 0.0  # longitudinal velocity
    vy: float = 0.0  # lateral velocity
    yaw_rate: float = 0.0
    
    # Accelerations
    ax: float = 0.0
    ay: float = 0.0
    
    # Wheel states
    wheel_speeds: np.ndarray = None  # 4 wheels
    wheel_slip_angles: np.ndarray = None
    wheel_slip_ratios: np.ndarray = None
    
    # Powertrain
    engine_rpm: float = 800.0
    gear: int = 1
    throttle: float = 0.0
    brake: float = 0.0
    
    def __post_init__(self):
        if self.wheel_speeds is None:
            self.wheel_speeds = np.zeros(4)
        if self.wheel_slip_angles is None:
            self.wheel_slip_angles = np.zeros(4)
        if self.wheel_slip_ratios is None:
            self.wheel_slip_ratios = np.zeros(4)

class PacejkaTireModel:
    """
    Magic Formula tire model (Pacejka 2002)
    Provides accurate tire force characteristics
    """
    
    def __init__(self, config: VehicleConfig):
        self.config = config
        
        # Magic Formula coefficients (lateral force)
        self.B_lat = 10.0  # Stiffness factor
        self.C_lat = 1.9   # Shape factor
        self.D_lat = 1.0   # Peak factor
        self.E_lat = -1.0  # Curvature factor
        
        # Longitudinal force coefficients
        self.B_long = 12.0
        self.C_long = 2.4
        self.D_long = 1.0
        self.E_long = 0.97
        
        # Combined slip coefficients
        self.r_x = 1.0
        self.r_y = 1.0
        
    def lateral_force(self, slip_angle: float, vertical_load: float, 
                     slip_ratio: float = 0.0) -> float:
        """
        Calculate lateral tire force using Magic Formula
        
        Args:
            slip_angle: Tire slip angle in radians
            vertical_load: Normal force on tire in Newtons
            slip_ratio: Longitudinal slip ratio
            
        Returns:
            Lateral force in Newtons
        """
        # Normalize vertical load
        Fz_norm = vertical_load / 4000.0
        
        # Peak lateral force
        D = self.D_lat * Fz_norm
        
        # Combined slip reduction
        if abs(slip_ratio) > 0.01:
            s = np.sqrt(slip_angle**2 + (self.r_x * slip_ratio)**2)
            alpha_eq = np.arctan2(slip_angle, self.r_x * slip_ratio)
        else:
            s = abs(slip_angle)
            alpha_eq = slip_angle
            
        # Magic Formula
        BCD = self.B_lat * self.C_lat * D
        E = self.E_lat
        
        Fy = D * np.sin(self.C_lat * np.arctan(
            self.B_lat * s - E * (self.B_lat * s - np.arctan(self.B_lat * s))
        ))
        
        # Return with correct sign
        return Fy * np.sign(alpha_eq)
    
    def longitudinal_force(self, slip_ratio: float, vertical_load: float,
                          slip_angle: float = 0.0) -> float:
        """
        Calculate longitudinal tire force
        
        Args:
            slip_ratio: Longitudinal slip ratio
            vertical_load: Normal force on tire
            slip_angle: Tire slip angle
            
        Returns:
            Longitudinal force in Newtons
        """
        Fz_norm = vertical_load / 4000.0
        D = self.D_long * Fz_norm
        
        # Combined slip
        if abs(slip_angle) > 0.01:
            s = np.sqrt((self.r_y * slip_angle)**2 + slip_ratio**2)
            kappa_eq = slip_ratio / s if s > 0 else 0
        else:
            s = abs(slip_ratio)
            kappa_eq = slip_ratio
            
        BCD = self.B_long * self.C_long * D
        E = self.E_long
        
        Fx = D * np.sin(self.C_long * np.arctan(
            self.B_long * s - E * (self.B_long * s - np.arctan(self.B_long * s))
        ))
        
        return Fx * np.sign(kappa_eq)

class VehicleDynamics:
    """
    Advanced vehicle dynamics model with:
    - Multi-body dynamics
    - Pacejka tire model
    - Load transfer effects
    - Aerodynamic forces
    - Powertrain dynamics
    """
    
    def __init__(self, config: VehicleConfig = VEHICLE_CONFIG):
        self.config = config
        self.tire_model = PacejkaTireModel(config)
        self.state = VehicleState()
        
        # Derived parameters
        self.lr = config.wheelbase * 0.5  # Distance to rear axle
        self.lf = config.wheelbase * 0.5  # Distance to front axle
        
    def calculate_tire_forces(self, state: VehicleState, 
                             steering_angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate forces at each tire
        
        Returns:
            Tuple of (longitudinal_forces, lateral_forces) for 4 wheels
        """
        # Wheel indices: 0=FL, 1=FR, 2=RL, 3=RR
        Fx = np.zeros(4)
        Fy = np.zeros(4)
        
        # Calculate vertical load distribution with load transfer
        static_front = self.config.mass * 9.81 * self.lr / self.config.wheelbase
        static_rear = self.config.mass * 9.81 * self.lf / self.config.wheelbase
        
        # Longitudinal load transfer
        load_transfer_long = (self.config.mass * state.ax * 
                             self.config.center_of_gravity_height / 
                             self.config.wheelbase)
        
        # Lateral load transfer
        load_transfer_lat = (self.config.mass * state.ay * 
                            self.config.center_of_gravity_height / 
                            self.config.track_width)
        
        # Vertical loads on each wheel
        Fz = np.zeros(4)
        Fz[0] = (static_front / 2) - load_transfer_long / 2 - load_transfer_lat  # FL
        Fz[1] = (static_front / 2) - load_transfer_long / 2 + load_transfer_lat  # FR
        Fz[2] = (static_rear / 2) + load_transfer_long / 2 - load_transfer_lat   # RL
        Fz[3] = (static_rear / 2) + load_transfer_long / 2 + load_transfer_lat   # RR
        
        # Ensure positive vertical loads
        Fz = np.maximum(Fz, 100.0)
        
        # Calculate slip angles
        if abs(state.vx) > 0.1:
            # Front wheels
            alpha_f = np.arctan2(state.vy + self.lf * state.yaw_rate, 
                                abs(state.vx)) - steering_angle
            # Rear wheels
            alpha_r = np.arctan2(state.vy - self.lr * state.yaw_rate, 
                                abs(state.vx))
        else:
            alpha_f = 0.0
            alpha_r = 0.0
            
        state.wheel_slip_angles[0:2] = alpha_f
        state.wheel_slip_angles[2:4] = alpha_r
        
        # Calculate slip ratios (simplified)
        for i in range(4):
            wheel_speed = state.wheel_speeds[i] * self.config.tire_radius
            if abs(state.vx) > 0.1:
                state.wheel_slip_ratios[i] = (wheel_speed - state.vx) / abs(state.vx)
            else:
                state.wheel_slip_ratios[i] = 0.0
                
        # Calculate tire forces using Pacejka model
        for i in range(4):
            Fx[i] = self.tire_model.longitudinal_force(
                state.wheel_slip_ratios[i], Fz[i], state.wheel_slip_angles[i]
            )
            Fy[i] = self.tire_model.lateral_force(
                state.wheel_slip_angles[i], Fz[i], state.wheel_slip_ratios[i]
            )
            
        return Fx, Fy
    
    def calculate_aerodynamic_forces(self, state: VehicleState) -> Tuple[float, float]:
        """
        Calculate aerodynamic drag and downforce
        
        Returns:
            Tuple of (drag_force, downforce)
        """
        v_total = np.sqrt(state.vx**2 + state.vy**2)
        
        drag_force = (0.5 * 1.225 * self.config.drag_coefficient * 
                     self.config.frontal_area * v_total**2)
        
        downforce = (0.5 * 1.225 * self.config.downforce_coefficient * 
                    self.config.frontal_area * v_total**2)
        
        return drag_force, downforce
    
    def powertrain_model(self, throttle: float, brake: float, 
                        wheel_speed: float) -> float:
        """
        Calculate drive torque from powertrain
        
        Args:
            throttle: Throttle input [0, 1]
            brake: Brake input [0, 1]
            wheel_speed: Average wheel speed
            
        Returns:
            Total wheel torque
        """
        # Simple engine model
        engine_speed = wheel_speed * self.config.steering_ratio
        engine_speed = np.clip(engine_speed, 800, 7000)  # rpm limits
        
        # Torque curve (simplified)
        optimal_rpm = 4000
        torque_ratio = np.exp(-((engine_speed - optimal_rpm) / 2000)**2)
        engine_torque = self.config.max_torque * torque_ratio * throttle
        
        # Transmission
        gear_ratios = [3.5, 2.0, 1.4, 1.0, 0.8]
        current_gear = min(self.state.gear - 1, len(gear_ratios) - 1)
        drive_torque = (engine_torque * gear_ratios[current_gear] * 
                       self.config.transmission_efficiency)
        
        # Braking torque
        brake_torque = -brake * self.config.max_brake_torque
        
        return drive_torque + brake_torque
    
    def update(self, dt: float, throttle: float, brake: float, 
               steering_angle: float) -> VehicleState:
        """
        Update vehicle state using dynamic equations
        
        Args:
            dt: Time step
            throttle: Throttle input [0, 1]
            brake: Brake input [0, 1]
            steering_angle: Steering angle in radians
            
        Returns:
            Updated vehicle state
        """
        # Calculate tire forces
        Fx_wheels, Fy_wheels = self.calculate_tire_forces(self.state, steering_angle)
        
        # Sum forces in body frame
        # Front wheels contribute to both Fx and Fy with steering
        Fx_front = (Fx_wheels[0] + Fx_wheels[1]) * np.cos(steering_angle) - \
                   (Fy_wheels[0] + Fy_wheels[1]) * np.sin(steering_angle)
        Fy_front = (Fx_wheels[0] + Fx_wheels[1]) * np.sin(steering_angle) + \
                   (Fy_wheels[0] + Fy_wheels[1]) * np.cos(steering_angle)
        
        Fx_rear = Fx_wheels[2] + Fx_wheels[3]
        Fy_rear = Fy_wheels[2] + Fy_wheels[3]
        
        # Total forces
        Fx_total = Fx_front + Fx_rear
        Fy_total = Fy_front + Fy_rear
        
        # Aerodynamic forces
        drag, downforce = self.calculate_aerodynamic_forces(self.state)
        Fx_total -= drag
        
        # Rolling resistance
        rolling_resistance = (self.config.tire_rolling_resistance * 
                             self.config.mass * 9.81)
        Fx_total -= rolling_resistance * np.sign(self.state.vx)
        
        # Accelerations in body frame
        self.state.ax = Fx_total / self.config.mass + \
                       self.state.vy * self.state.yaw_rate
        self.state.ay = Fy_total / self.config.mass - \
                       self.state.vx * self.state.yaw_rate
        
        # Yaw acceleration
        Mz = self.lf * Fy_front - self.lr * Fy_rear
        yaw_accel = Mz / self.config.moment_of_inertia
        
        # Integrate velocities
        self.state.vx += self.state.ax * dt
        self.state.vy += self.state.ay * dt
        self.state.yaw_rate += yaw_accel * dt
        
        # Update position and orientation (global frame)
        v_global_x = self.state.vx * np.cos(self.state.yaw) - \
                     self.state.vy * np.sin(self.state.yaw)
        v_global_y = self.state.vx * np.sin(self.state.yaw) + \
                     self.state.vy * np.cos(self.state.yaw)
        
        self.state.x += v_global_x * dt
        self.state.y += v_global_y * dt
        self.state.yaw += self.state.yaw_rate * dt
        
        # Normalize yaw to [-pi, pi]
        self.state.yaw = np.arctan2(np.sin(self.state.yaw), 
                                   np.cos(self.state.yaw))
        
        # Update wheel speeds (simplified)
        avg_wheel_torque = self.powertrain_model(throttle, brake, 
                                                 np.mean(self.state.wheel_speeds))
        wheel_accel = avg_wheel_torque / (self.config.mass * 
                                         self.config.tire_radius**2)
        self.state.wheel_speeds += wheel_accel * dt
        
        # Store control inputs
        self.state.throttle = throttle
        self.state.brake = brake
        
        return self.state
    
    def get_state(self) -> VehicleState:
        """Return current vehicle state"""
        return self.state
    
    def reset(self, initial_state: Optional[VehicleState] = None):
        """Reset vehicle to initial state"""
        if initial_state is not None:
            self.state = initial_state
        else:
            self.state = VehicleState()