"""
Main Autonomous Vehicle Simulation
Integrates all systems and runs the complete simulation
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Optional
import time
import os

from config import *
from vehicle_dynamics import VehicleDynamics, VehicleState
from sensor_fusion import MultiSensorFusion, SensorMeasurement
from path_planning import AStarPlanner, RRTStarPlanner, TrajectoryOptimizer, Waypoint
from controllers import PIDController, ModelPredictiveController, DeepRLController, ControlCommand
from environment import Environment, DynamicObject
from perception import PerceptionSystem

# Import new advanced modules
try:
    from spatial_indexing import SpatialIndex, AdvancedCollisionDetection, SpatialObject
    from shapely.geometry import box as shapely_box
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False
    print("Warning: spatial_indexing module not available. Install rtree and shapely.")

try:
    from visualization import PygameVisualizer, CameraRenderer, VisualizationConfig
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: visualization module not available. Install pygame and opencv-python.")

try:
    from ml_utils import LiDARClustering, TrajectoryAnalyzer, RLTrainingMonitor
    ML_UTILS_AVAILABLE = True
except ImportError:
    ML_UTILS_AVAILABLE = False
    print("Warning: ml_utils module not available. Install scikit-learn and tensorboard.")

try:
    from network_routing import RoadNetwork, TrafficAwareRouter, build_grid_network
    NETWORK_ROUTING_AVAILABLE = True
except ImportError:
    NETWORK_ROUTING_AVAILABLE = False
    print("Warning: network_routing module not available. Install networkx.")

class AutonomousVehicle:
    """
    Complete autonomous vehicle system
    Integrates perception, planning, and control
    """
    
    def __init__(self):
        # Configuration
        self.config = SIM_CONFIG
        
        # Core systems
        self.dynamics = VehicleDynamics(VEHICLE_CONFIG)
        self.perception = PerceptionSystem(SENSOR_CONFIG)
        self.sensor_fusion = MultiSensorFusion(SENSOR_CONFIG)
        
        # Planning
        self.global_planner = AStarPlanner(resolution=2.0)
        self.local_planner = RRTStarPlanner(PLANNING_CONFIG)
        self.trajectory_optimizer = TrajectoryOptimizer(PLANNING_CONFIG)
        
        # Control
        self.controller_type = CONTROL_CONFIG.controller_type
        
        if self.controller_type == ControllerType.PID:
            self.controller = PIDController(CONTROL_CONFIG)
        elif self.controller_type == ControllerType.MPC:
            self.controller = ModelPredictiveController(CONTROL_CONFIG)
        else:
            self.controller = DeepRLController(CONTROL_CONFIG)
        
        # Advanced modules (optional)
        if SPATIAL_AVAILABLE:
            self.spatial_index = SpatialIndex()
            self.collision_detector = AdvancedCollisionDetection()
            print("✓ Spatial indexing enabled")
        else:
            self.spatial_index = None
            self.collision_detector = None
        
        if ML_UTILS_AVAILABLE:
            self.lidar_clusterer = LiDARClustering()
            self.trajectory_analyzer = TrajectoryAnalyzer()
            print("✓ ML utilities enabled")
        else:
            self.lidar_clusterer = None
            self.trajectory_analyzer = None
        
        # State
        self.current_waypoint_idx = 0
        self.planned_path: List[Waypoint] = []
        self.optimized_trajectory = None
        
        # Performance metrics
        self.metrics = {
            'distance_traveled': 0.0,
            'average_speed': 0.0,
            'num_lane_changes': 0,
            'comfort_violations': 0,
            'safety_violations': 0,
            'planning_time': [],
            'control_time': []
        }
        
    def plan_global_path(self, start: np.ndarray, goal: np.ndarray,
                        environment: Environment) -> bool:
        """
        Plan global path from start to goal
        
        Args:
            start: Start position [x, y]
            goal: Goal position [x, y]
            environment: Environment
            
        Returns:
            True if path found
        """
        print("Planning global path...")
        
        # Create occupancy grid (simplified)
        grid_size = (int(self.config.map_size[0]), int(self.config.map_size[1]))
        occupancy_grid = np.zeros(grid_size)
        
        # Add obstacles to grid
        for obs_pos, obs_radius in environment.static_obstacles:
            x, y = int(obs_pos[0]), int(obs_pos[1])
            r = int(obs_radius) + 2  # Safety margin
            
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if dx*dx + dy*dy <= r*r:
                        gx, gy = x + dx, y + dy
                        if 0 <= gx < grid_size[0] and 0 <= gy < grid_size[1]:
                            occupancy_grid[gx, gy] = 1
        
        # Plan with A*
        self.planned_path = self.global_planner.plan(
            start, goal, occupancy_grid, grid_size
        )
        
        if self.planned_path is None:
            print("Failed to find global path!")
            return False
        
        print(f"Global path planned with {len(self.planned_path)} waypoints")
        return True
    
    def plan_local_trajectory(self, current_state: VehicleState,
                             environment: Environment) -> bool:
        """
        Plan local trajectory considering dynamic obstacles
        
        Args:
            current_state: Current vehicle state
            environment: Environment
            
        Returns:
            True if trajectory found
        """
        if len(self.planned_path) == 0:
            return False
        
        # Get reference path segment
        lookahead = 50
        end_idx = min(self.current_waypoint_idx + lookahead, len(self.planned_path))
        reference_path = self.planned_path[self.current_waypoint_idx:end_idx]
        
        if len(reference_path) < 5:
            return False
        
        # Get nearby obstacles
        current_pos = np.array([current_state.x, current_state.y])
        obstacles = environment.get_obstacles_in_range(current_pos, 50.0)
        obstacle_positions = [obs[0] for obs in obstacles]
        
        # Use advanced collision detection if available
        if self.collision_detector is not None and SPATIAL_AVAILABLE:
            # Create spatial objects for obstacles
            for obs_pos, obs_radius in obstacles:
                obs_polygon = shapely_box(
                    obs_pos[0] - obs_radius, obs_pos[1] - obs_radius,
                    obs_pos[0] + obs_radius, obs_pos[1] + obs_radius
                )
                spatial_obj = SpatialObject(
                    id=hash(tuple(obs_pos)),
                    geometry=obs_polygon,
                    velocity=np.zeros(2),
                    object_type='obstacle'
                )
                self.spatial_index.insert(spatial_obj)
        
        # Optimize trajectory
        initial_state = np.array([
            current_state.x,
            current_state.y,
            current_state.vx,
            current_state.vy,
            current_state.yaw
        ])
        
        self.optimized_trajectory = self.trajectory_optimizer.optimize_trajectory(
            reference_path, initial_state, obstacle_positions
        )
        
        # Use ML clustering on LiDAR if available
        if self.lidar_clusterer is not None and ML_UTILS_AVAILABLE:
            # This would use LiDAR clustering in actual implementation
            pass
        
        return True
    
    def execute_control(self, current_state: VehicleState, dt: float) -> ControlCommand:
        """
        Execute control to follow trajectory
        
        Args:
            current_state: Current vehicle state
            dt: Time step
            
        Returns:
            Control command
        """
        start_time = time.time()
        
        if self.optimized_trajectory is None or len(self.optimized_trajectory.waypoints) == 0:
            # Emergency stop
            return ControlCommand(0.0, 1.0, 0.0, time.time())
        
        # Find closest waypoint
        current_pos = np.array([current_state.x, current_state.y])
        distances = [np.linalg.norm(wp.position if hasattr(wp, 'position') 
                                   else np.array([wp.x, wp.y]) - current_pos)
                    for wp in self.optimized_trajectory.waypoints]
        closest_idx = np.argmin(distances)
        
        # Lookahead
        target_idx = min(closest_idx + 5, len(self.optimized_trajectory.waypoints) - 1)
        target_waypoint = self.optimized_trajectory.waypoints[target_idx]
        
        # Get target position and velocity
        if hasattr(target_waypoint, 'position'):
            target_pos = target_waypoint.position
        else:
            target_pos = np.array([target_waypoint.x, target_waypoint.y])
        
        target_yaw = target_waypoint.yaw
        target_velocity = target_waypoint.velocity if target_waypoint.velocity > 0 else 15.0
        
        # Execute controller
        if isinstance(self.controller, PIDController):
            # Longitudinal control
            throttle, brake = self.controller.compute_longitudinal_control(
                current_state.vx, target_velocity, dt
            )
            
            # Lateral control
            steering = self.controller.compute_lateral_control(
                current_pos, current_state.yaw, target_pos, target_yaw, dt
            )
            
        elif isinstance(self.controller, ModelPredictiveController):
            # MPC
            current_mpc_state = np.array([
                current_state.x, current_state.y,
                current_state.vx, current_state.yaw
            ])
            
            # Reference trajectory
            ref_traj = np.array([
                [wp.x if hasattr(wp, 'x') else wp.position[0],
                 wp.y if hasattr(wp, 'y') else wp.position[1],
                 wp.velocity if wp.velocity > 0 else 15.0,
                 wp.yaw]
                for wp in self.optimized_trajectory.waypoints[:20]
            ])
            
            throttle, brake, steering = self.controller.solve(
                current_mpc_state, ref_traj
            )
            
        else:  # Deep RL
            # State vector for RL
            state_vector = self._construct_state_vector(current_state, target_pos, target_yaw)
            throttle, brake, steering = self.controller.get_action(state_vector, deterministic=True)
        
        control_time = time.time() - start_time
        self.metrics['control_time'].append(control_time)
        
        return ControlCommand(throttle, brake, steering, time.time())
    
    def _construct_state_vector(self, state: VehicleState, 
                               target_pos: np.ndarray, target_yaw: float) -> np.ndarray:
        """Construct state vector for RL controller"""
        # Simplified state representation
        current_pos = np.array([state.x, state.y])
        distance_to_target = np.linalg.norm(target_pos - current_pos)
        angle_to_target = np.arctan2(target_pos[1] - state.y, 
                                     target_pos[0] - state.x) - state.yaw
        
        state_vector = np.array([
            state.vx, state.vy, state.yaw_rate,
            distance_to_target, angle_to_target,
            target_yaw - state.yaw,
            state.ax, state.ay
        ])
        
        # Pad to required dimension
        padding = np.zeros(CONTROL_CONFIG.rl_state_dim - len(state_vector))
        return np.concatenate([state_vector, padding])
    
    def update_metrics(self, state: VehicleState, dt: float):
        """Update performance metrics"""
        velocity = np.sqrt(state.vx**2 + state.vy**2)
        self.metrics['distance_traveled'] += velocity * dt
        self.metrics['average_speed'] = (
            self.metrics['average_speed'] * 0.99 + velocity * 0.01
        )
        
        # Check comfort violations (high jerk)
        if abs(state.ax) > CONTROL_CONFIG.max_acceleration:
            self.metrics['comfort_violations'] += 1

class Simulation:
    """
    Main simulation manager
    Orchestrates all components
    """
    
    def __init__(self):
        self.environment = Environment(SIM_CONFIG)
        self.ego_vehicle = AutonomousVehicle()
        
        # Simulation state
        self.current_time = 0.0
        self.dt = SIM_CONFIG.dt
        
        # Visualization options
        self.use_pygame = PYGAME_AVAILABLE
        
        if self.use_pygame:
            # Use advanced Pygame visualization
            self.pygame_viz = PygameVisualizer(VisualizationConfig())
            self.camera_renderer = CameraRenderer()
            print("✓ Using Pygame visualization")
        else:
            # Fallback to matplotlib
            self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 12))
            print("✓ Using Matplotlib visualization")
        
        self.visualization_enabled = True
        
        # Network routing (if available)
        if NETWORK_ROUTING_AVAILABLE:
            self.road_network = build_grid_network(grid_size=20, spacing=50.0)
            self.traffic_router = TrafficAwareRouter(self.road_network)
            print("✓ Network routing enabled")
        else:
            self.road_network = None
            self.traffic_router = None
        
        # Training monitor (if available)
        if ML_UTILS_AVAILABLE:
            self.training_monitor = RLTrainingMonitor(log_dir='./runs/simulation')
            print("✓ TensorBoard logging enabled")
        else:
            self.training_monitor = None
        
    def initialize(self, start_pos: np.ndarray, goal_pos: np.ndarray):
        """
        Initialize simulation
        
        Args:
            start_pos: Start position [x, y]
            goal_pos: Goal position [x, y]
        """
        print("Initializing simulation...")
        
        # Set initial vehicle state
        initial_state = VehicleState()
        initial_state.x = start_pos[0]
        initial_state.y = start_pos[1]
        initial_state.vx = 0.0
        initial_state.yaw = 0.0
        
        self.ego_vehicle.dynamics.reset(initial_state)
        
        # Plan global path
        success = self.ego_vehicle.plan_global_path(start_pos, goal_pos, self.environment)
        
        if not success:
            raise RuntimeError("Failed to plan global path")
        
        print("Simulation initialized successfully")
    
    def step(self):
        """Execute one simulation step"""
        # Get current state
        state = self.ego_vehicle.dynamics.get_state()
        
        # Perception
        current_pos = np.array([state.x, state.y])
        current_vel = np.array([state.vx, state.vy])
        
        perception_data = self.ego_vehicle.perception.perceive(
            current_pos, state.yaw, current_vel, self.environment
        )
        
        # Sensor fusion (simplified - using perception directly)
        # In full system, would fuse raw sensor data
        
        # Local planning
        self.ego_vehicle.plan_local_trajectory(state, self.environment)
        
        # Control
        control_cmd = self.ego_vehicle.execute_control(state, self.dt)
        
        # Update vehicle dynamics
        self.ego_vehicle.dynamics.update(
            self.dt, control_cmd.throttle, control_cmd.brake, control_cmd.steering
        )
        
        # Update environment
        self.environment.step(self.dt)
        
        # Update metrics
        self.ego_vehicle.update_metrics(state, self.dt)
        
        # Update time
        self.current_time += self.dt
        
        # Check goal reached
        if len(self.ego_vehicle.planned_path) > 0:
            goal = self.ego_vehicle.planned_path[-1]
            goal_pos = np.array([goal.x, goal.y])
            if np.linalg.norm(current_pos - goal_pos) < 5.0:
                print("Goal reached!")
                return False
        
        return True
    
    def run(self, duration: float = 60.0, visualize: bool = True):
        """
        Run simulation
        
        Args:
            duration: Simulation duration in seconds
            visualize: Enable visualization
        """
        print(f"Running simulation for {duration} seconds...")
        
        steps = int(duration / self.dt)
        
        if self.use_pygame and visualize:
            # Pygame visualization loop
            running = True
            step = 0
            
            while running and step < steps:
                # Handle events
                running = self.pygame_viz.handle_events()
                if not running:
                    break
                
                # Step simulation
                if not self.step():
                    break
                
                # Render
                state = self.ego_vehicle.dynamics.get_state()
                self.pygame_viz.render_frame(
                    ego_state=state,
                    environment=self.environment,
                    planned_path=self.ego_vehicle.planned_path,
                    trajectory=self.ego_vehicle.optimized_trajectory,
                    lidar_points=None,
                    metrics=self.ego_vehicle.metrics
                )
                
                step += 1
                
                if step % 100 == 0:
                    print(f"Step {step}/{steps}, Time: {self.current_time:.2f}s, "
                          f"Speed: {state.vx:.2f} m/s")
            
            self.pygame_viz.cleanup()
            
        else:
            # Original matplotlib loop
            for i in range(steps):
                if not self.step():
                    break
                
                if visualize and i % 10 == 0:
                    self.visualize()
                    plt.pause(0.001)
                
                if i % 100 == 0:
                    print(f"Step {i}/{steps}, Time: {self.current_time:.2f}s, "
                          f"Speed: {self.ego_vehicle.dynamics.state.vx:.2f} m/s")
        
        self.print_metrics()
        
        # Close training monitor if available
        if self.training_monitor is not None:
            self.training_monitor.close()
    
    def visualize(self):
        """Visualize simulation state"""
        for ax in self.axes.flat:
            ax.clear()
        
        state = self.ego_vehicle.dynamics.get_state()
        
        # Main view
        ax_main = self.axes[0, 0]
        ax_main.set_title("Simulation Environment")
        ax_main.set_xlabel("X (m)")
        ax_main.set_ylabel("Y (m)")
        ax_main.grid(True)
        
        # Plot road network
        for segment in self.environment.road_network:
            for lane in segment.lanes:
                ax_main.plot(lane.waypoints[:, 0], lane.waypoints[:, 1], 
                           'k--', alpha=0.3, linewidth=0.5)
        
        # Plot planned path
        if len(self.ego_vehicle.planned_path) > 0:
            path_x = [wp.x for wp in self.ego_vehicle.planned_path]
            path_y = [wp.y for wp in self.ego_vehicle.planned_path]
            ax_main.plot(path_x, path_y, 'b-', linewidth=2, label='Planned Path')
        
        # Plot ego vehicle
        vehicle_rect = patches.Rectangle(
            (state.x - 2, state.y - 1), 4, 2,
            angle=np.rad2deg(state.yaw),
            facecolor='red', edgecolor='darkred', linewidth=2
        )
        ax_main.add_patch(vehicle_rect)
        
        # Plot traffic
        for vehicle in self.environment.traffic_sim.traffic_vehicles.values():
            rect = patches.Rectangle(
                (vehicle.position[0] - vehicle.dimensions[0]/2,
                 vehicle.position[1] - vehicle.dimensions[1]/2),
                vehicle.dimensions[0], vehicle.dimensions[1],
                angle=np.rad2deg(vehicle.yaw),
                facecolor='blue', alpha=0.5
            )
            ax_main.add_patch(rect)
        
        # Set view around ego vehicle
        ax_main.set_xlim(state.x - 50, state.x + 50)
        ax_main.set_ylim(state.y - 50, state.y + 50)
        ax_main.legend()
        
        # Velocity profile
        ax_vel = self.axes[0, 1]
        ax_vel.set_title("Velocity Profile")
        ax_vel.set_xlabel("Time (s)")
        ax_vel.set_ylabel("Velocity (m/s)")
        ax_vel.grid(True)
        
        # Occupancy grid
        ax_grid = self.axes[1, 0]
        ax_grid.set_title("Occupancy Grid")
        if self.ego_vehicle.perception.occupancy_grid is not None:
            ax_grid.imshow(self.ego_vehicle.perception.occupancy_grid.T, 
                         origin='lower', cmap='gray')
        
        # Metrics
        ax_metrics = self.axes[1, 1]
        ax_metrics.set_title("Performance Metrics")
        ax_metrics.axis('off')
        
        metrics_text = f"""
        Time: {self.current_time:.2f} s
        Distance: {self.ego_vehicle.metrics['distance_traveled']:.2f} m
        Avg Speed: {self.ego_vehicle.metrics['average_speed']:.2f} m/s
        Current Speed: {state.vx:.2f} m/s
        Comfort Violations: {self.ego_vehicle.metrics['comfort_violations']}
        """
        ax_metrics.text(0.1, 0.5, metrics_text, fontsize=12, 
                       verticalalignment='center')
        
        plt.tight_layout()
    
    def print_metrics(self):
        """Print final performance metrics"""
        print("\n" + "="*50)
        print("SIMULATION METRICS")
        print("="*50)
        print(f"Total Distance: {self.ego_vehicle.metrics['distance_traveled']:.2f} m")
        print(f"Average Speed: {self.ego_vehicle.metrics['average_speed']:.2f} m/s")
        print(f"Comfort Violations: {self.ego_vehicle.metrics['comfort_violations']}")
        print(f"Safety Violations: {self.ego_vehicle.metrics['safety_violations']}")
        
        if len(self.ego_vehicle.metrics['control_time']) > 0:
            avg_control_time = np.mean(self.ego_vehicle.metrics['control_time'])
            print(f"Avg Control Time: {avg_control_time*1000:.2f} ms")
        
        print("="*50 + "\n")

def main():
    """Main entry point"""
    print("="*60)
    print("AUTONOMOUS VEHICLE SIMULATION")
    print("="*60)
    
    # Create simulation
    sim = Simulation()
    
    # Define start and goal
    start_position = np.array([10.0, 0.0])
    goal_position = np.array([500.0, 0.0])
    
    # Initialize
    sim.initialize(start_position, goal_position)
    
    # Run simulation
    sim.run(duration=120.0, visualize=True)
    
    print("\nSimulation completed successfully!")
    plt.show()

if __name__ == "__main__":
    main()