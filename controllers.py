"""
Advanced Vehicle Control Systems
Implements PID, Model Predictive Control, and Deep Reinforcement Learning controllers
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from config import ControlConfig, CONTROL_CONFIG

@dataclass
class ControlCommand:
    """Vehicle control command"""
    throttle: float  # [0, 1]
    brake: float     # [0, 1]
    steering: float  # radians
    timestamp: float

class PIDController:
    """
    PID controller for longitudinal and lateral control
    Industry-standard baseline controller
    """
    
    def __init__(self, config: ControlConfig = CONTROL_CONFIG):
        self.config = config
        
        # Longitudinal PID
        self.kp_long = config.pid_kp_longitudinal
        self.ki_long = config.pid_ki_longitudinal
        self.kd_long = config.pid_kd_longitudinal
        
        # Lateral PID
        self.kp_lat = config.pid_kp_lateral
        self.ki_lat = config.pid_ki_lateral
        self.kd_lat = config.pid_kd_lateral
        
        # State variables
        self.integral_long = 0.0
        self.integral_lat = 0.0
        self.prev_error_long = 0.0
        self.prev_error_lat = 0.0
        
        # Anti-windup limits
        self.integral_max = 10.0
        
    def compute_longitudinal_control(self, current_velocity: float,
                                    target_velocity: float,
                                    dt: float) -> Tuple[float, float]:
        """
        Compute throttle and brake commands
        
        Args:
            current_velocity: Current velocity in m/s
            target_velocity: Desired velocity in m/s
            dt: Time step
            
        Returns:
            Tuple of (throttle, brake)
        """
        error = target_velocity - current_velocity
        
        # Proportional term
        p_term = self.kp_long * error
        
        # Integral term with anti-windup
        self.integral_long += error * dt
        self.integral_long = np.clip(self.integral_long, 
                                     -self.integral_max, self.integral_max)
        i_term = self.ki_long * self.integral_long
        
        # Derivative term
        d_term = self.kd_long * (error - self.prev_error_long) / dt
        self.prev_error_long = error
        
        # Total control
        control = p_term + i_term + d_term
        
        # Split into throttle and brake
        if control > 0:
            throttle = np.clip(control, 0.0, 1.0)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-control, 0.0, 1.0)
        
        return throttle, brake
    
    def compute_lateral_control(self, current_position: np.ndarray,
                               current_yaw: float,
                               target_waypoint: np.ndarray,
                               target_yaw: float,
                               dt: float) -> float:
        """
        Compute steering command using Stanley controller
        
        Args:
            current_position: [x, y]
            current_yaw: Current heading
            target_waypoint: Target [x, y]
            target_yaw: Target heading
            dt: Time step
            
        Returns:
            Steering angle in radians
        """
        # Heading error
        heading_error = target_yaw - current_yaw
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Cross-track error
        dx = target_waypoint[0] - current_position[0]
        dy = target_waypoint[1] - current_position[1]
        front_axle_vec_rot = np.array([np.cos(current_yaw), np.sin(current_yaw)])
        crosstrack_error = np.dot([dy, -dx], front_axle_vec_rot)
        
        # PID terms
        p_term = self.kp_lat * heading_error
        
        # Integral with anti-windup
        self.integral_lat += crosstrack_error * dt
        self.integral_lat = np.clip(self.integral_lat,
                                    -self.integral_max, self.integral_max)
        i_term = self.ki_lat * self.integral_lat
        
        # Derivative
        d_term = self.kd_lat * (crosstrack_error - self.prev_error_lat) / dt
        self.prev_error_lat = crosstrack_error
        
        # Crosstrack correction (Stanley controller component)
        k_crosstrack = 0.5
        crosstrack_term = np.arctan2(k_crosstrack * crosstrack_error, 1.0)
        
        # Total steering
        steering = p_term + i_term + d_term + crosstrack_term
        
        # Clip to max steering angle
        max_steer = self.config.max_steering_angle if hasattr(self.config, 'max_steering_angle') else np.deg2rad(35)
        steering = np.clip(steering, -max_steer, max_steer)
        
        return steering
    
    def reset(self):
        """Reset controller state"""
        self.integral_long = 0.0
        self.integral_lat = 0.0
        self.prev_error_long = 0.0
        self.prev_error_lat = 0.0

class ModelPredictiveController:
    """
    Nonlinear Model Predictive Control for trajectory tracking
    Handles constraints and multi-objective optimization
    """
    
    def __init__(self, config: ControlConfig = CONTROL_CONFIG):
        self.config = config
        self.horizon = config.mpc_horizon
        self.dt = config.mpc_dt
        
        # Weight matrices for cost function
        self.Q = np.diag([10.0, 10.0, 1.0, 0.1])  # State: [x, y, v, yaw]
        self.R = np.diag([1.0, 0.1])              # Control: [accel, steer]
        self.Rd = np.diag([1.0, 1.0])             # Control rate
        
    def solve(self, current_state: np.ndarray,
             reference_trajectory: np.ndarray,
             obstacles: Optional[List[np.ndarray]] = None) -> Tuple[float, float, float]:
        """
        Solve MPC optimization problem
        
        Args:
            current_state: [x, y, v, yaw]
            reference_trajectory: Reference states [N x 4]
            obstacles: Optional list of obstacle positions
            
        Returns:
            Tuple of (throttle, brake, steering)
        """
        # Initialize control sequence
        n_controls = 2
        u0 = np.zeros(self.horizon * n_controls)
        
        # Define optimization objective
        def objective(u):
            return self._cost_function(u, current_state, reference_trajectory, obstacles)
        
        # Constraints
        bounds = []
        for i in range(self.horizon):
            # Acceleration bounds
            bounds.append((-self.config.max_deceleration, self.config.max_acceleration))
            # Steering rate bounds
            max_steer_rate = self.config.max_steering_rate if hasattr(self.config, 'max_steering_rate') else np.deg2rad(540)
            bounds.append((-max_steer_rate * self.dt, max_steer_rate * self.dt))
        
        # Solve optimization
        result = minimize(
            objective, u0,
            method='SLSQP',
            bounds=bounds,
            options={
                'maxiter': self.config.mpc_max_iterations,
                'ftol': self.config.mpc_tolerance
            }
        )
        
        if result.success:
            # Extract first control action
            accel = result.x[0]
            steer_rate = result.x[1]
            
            # Convert to throttle/brake
            if accel > 0:
                throttle = accel / self.config.max_acceleration
                brake = 0.0
            else:
                throttle = 0.0
                brake = -accel / self.config.max_deceleration
            
            # Integrate steering rate
            steering = steer_rate
            
            return throttle, brake, steering
        else:
            # Fallback to safe controls
            return 0.0, 0.3, 0.0
    
    def _cost_function(self, u: np.ndarray, x0: np.ndarray,
                      x_ref: np.ndarray, obstacles: Optional[List[np.ndarray]]) -> float:
        """
        MPC cost function
        
        Includes:
        - Tracking error
        - Control effort
        - Control rate
        - Collision avoidance
        """
        cost = 0.0
        x = x0.copy()
        u_prev = np.zeros(2)
        
        for i in range(self.horizon):
            # Extract control at this timestep
            accel = u[i * 2]
            steer_rate = u[i * 2 + 1]
            u_current = np.array([accel, steer_rate])
            
            # Predict next state using bicycle model
            x = self._predict_state(x, u_current)
            
            # Tracking cost
            if i < len(x_ref):
                state_error = x - x_ref[i]
                # Normalize yaw error
                state_error[3] = np.arctan2(np.sin(state_error[3]), 
                                           np.cos(state_error[3]))
                cost += state_error.T @ self.Q @ state_error
            
            # Control effort cost
            cost += u_current.T @ self.R @ u_current
            
            # Control rate cost
            if i > 0:
                du = u_current - u_prev
                cost += du.T @ self.Rd @ du
            
            # Collision avoidance cost
            if obstacles is not None:
                for obs in obstacles:
                    dist = np.linalg.norm(x[0:2] - obs[0:2])
                    safety_margin = 5.0
                    if dist < safety_margin:
                        cost += 1000.0 * (safety_margin - dist)**2
            
            u_prev = u_current
        
        return cost
    
    def _predict_state(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Predict next state using kinematic bicycle model
        
        State: [x, y, v, yaw]
        Control: [accel, steer_rate]
        """
        wheelbase = 2.7  # meters
        
        x_next = x.copy()
        
        # Update velocity
        x_next[2] += u[0] * self.dt
        x_next[2] = np.clip(x_next[2], 0.0, self.config.max_velocity)
        
        # Update yaw (integrate steering)
        steering = u[1] * self.dt
        max_steer = np.deg2rad(35)
        steering = np.clip(steering, -max_steer, max_steer)
        
        x_next[3] += (x[2] / wheelbase) * np.tan(steering) * self.dt
        x_next[3] = np.arctan2(np.sin(x_next[3]), np.cos(x_next[3]))
        
        # Update position
        x_next[0] += x[2] * np.cos(x[3]) * self.dt
        x_next[1] += x[2] * np.sin(x[3]) * self.dt
        
        return x_next

class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic neural network for deep RL control
    Uses PPO (Proximal Policy Optimization)
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int]):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared feature extraction
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network"""
        features = self.shared(state)
        
        # Actor output
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)
        
        # Critic output
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        """Sample action from policy"""
        action_mean, action_std, value = self.forward(state)
        
        if deterministic:
            return action_mean, value
        
        # Sample from Gaussian policy
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob, value

class DeepRLController:
    """
    Deep Reinforcement Learning controller using PPO
    Learns end-to-end control from state to actions
    """
    
    def __init__(self, config: ControlConfig = CONTROL_CONFIG):
        self.config = config
        
        # Network
        self.network = ActorCriticNetwork(
            state_dim=config.rl_state_dim,
            action_dim=config.rl_action_dim,
            hidden_dims=config.rl_hidden_layers
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config.rl_learning_rate
        )
        
        # PPO hyperparameters
        self.gamma = config.rl_gamma
        self.gae_lambda = 0.95
        self.clip_epsilon = 0.2
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        
        # Experience buffer
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Get control action from current state
        
        Args:
            state: Current state vector
            deterministic: If True, use mean action
            
        Returns:
            Action vector [accel, steering]
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            if deterministic:
                action, value = self.network.get_action(state_tensor, deterministic=True)
                action = action.squeeze(0).numpy()
            else:
                action, log_prob, value = self.network.get_action(state_tensor)
                action = action.squeeze(0).numpy()
                
                # Store for training
                self.states.append(state)
                self.actions.append(action)
                self.log_probs.append(log_prob.item())
                self.values.append(value.item())
        
        # Scale actions to control ranges
        accel = action[0] * self.config.max_acceleration
        steering = action[1] * np.deg2rad(35)
        
        # Convert to throttle/brake
        if accel > 0:
            throttle = np.clip(accel / self.config.max_acceleration, 0, 1)
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.clip(-accel / self.config.max_deceleration, 0, 1)
        
        return throttle, brake, steering
    
    def store_transition(self, reward: float, done: bool):
        """Store transition for training"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def update(self, next_state: np.ndarray):
        """
        Update policy using PPO
        
        Args:
            next_state: State after transition
        """
        if len(self.states) < self.config.rl_batch_size:
            return
        
        # Compute returns and advantages
        returns = self._compute_returns(next_state)
        advantages = returns - np.array(self.values)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)
        
        # PPO update epochs
        for _ in range(10):
            # Forward pass
            action_mean, action_std, values = self.network(states)
            
            # Compute new log probs
            dist = torch.distributions.Normal(action_mean, action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 
                               1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = 0.5 * ((values.squeeze() - returns) ** 2).mean()
            
            # Total loss
            loss = actor_loss + self.value_coef * value_loss - \
                   self.entropy_coef * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def _compute_returns(self, next_state: np.ndarray) -> np.ndarray:
        """Compute discounted returns using GAE"""
        next_value = 0.0
        if not self.dones[-1]:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                _, _, value = self.network(state_tensor)
                next_value = value.item()
        
        returns = []
        gae = 0.0
        
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                next_value = 0.0
                gae = 0.0
            
            delta = self.rewards[i] + self.gamma * next_value - self.values[i]
            gae = delta + self.gamma * self.gae_lambda * gae
            returns.insert(0, gae + self.values[i])
            next_value = self.values[i]
        
        return np.array(returns)
    
    def save(self, path: str):
        """Save model"""
        torch.save(self.network.state_dict(), path)
    
    def load(self, path: str):
        """Load model"""
        self.network.load_state_dict(torch.load(path))