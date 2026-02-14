"""
Advanced Path Planning and Trajectory Optimization
Implements A*, RRT*, Hybrid A*, and Model Predictive Control for trajectory generation
"""

import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline, splprep, splev
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
import heapq
from config import PlanningConfig, PLANNING_CONFIG

@dataclass
class Waypoint:
    """Single waypoint in a path"""
    x: float
    y: float
    yaw: float = 0.0
    velocity: float = 0.0
    curvature: float = 0.0

@dataclass
class Trajectory:
    """Complete trajectory with states and controls"""
    waypoints: List[Waypoint]
    timestamps: np.ndarray
    curvatures: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    cost: float = 0.0

class AStarPlanner:
    """
    A* path planning algorithm for global path generation
    Optimized for structured road networks
    """
    
    def __init__(self, resolution: float = 1.0):
        self.resolution = resolution
        self.motion = self._get_motion_model()
        
    def _get_motion_model(self) -> List[Tuple[int, int, float]]:
        """
        Motion primitives for A*
        Returns: List of (dx, dy, cost)
        """
        motion = [
            (1, 0, 1.0),
            (0, 1, 1.0),
            (-1, 0, 1.0),
            (0, -1, 1.0),
            (1, 1, np.sqrt(2)),
            (-1, 1, np.sqrt(2)),
            (1, -1, np.sqrt(2)),
            (-1, -1, np.sqrt(2))
        ]
        return motion
    
    def plan(self, start: Tuple[float, float], goal: Tuple[float, float],
             obstacles: np.ndarray, map_bounds: Tuple[float, float]) -> Optional[List[Waypoint]]:
        """
        Plan path from start to goal avoiding obstacles
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            obstacles: Binary occupancy grid
            map_bounds: (width, height) of map
            
        Returns:
            List of waypoints or None if no path found
        """
        # Convert to grid coordinates
        start_node = self._to_grid(start[0], start[1])
        goal_node = self._to_grid(goal[0], goal[1])
        
        # Initialize
        open_set = []
        heapq.heappush(open_set, (0.0, start_node))
        came_from = {}
        g_score = {start_node: 0.0}
        f_score = {start_node: self._heuristic(start_node, goal_node)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal_node:
                return self._reconstruct_path(came_from, current)
            
            for motion in self.motion:
                neighbor = (current[0] + motion[0], current[1] + motion[1])
                
                # Check bounds
                if not self._is_valid(neighbor, obstacles, map_bounds):
                    continue
                
                tentative_g = g_score[current] + motion[2]
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal_node)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))
        
        return None
    
    def _heuristic(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> float:
        """Euclidean distance heuristic"""
        return np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)
    
    def _is_valid(self, node: Tuple[int, int], obstacles: np.ndarray,
                  bounds: Tuple[float, float]) -> bool:
        """Check if node is valid (within bounds and collision-free)"""
        if node[0] < 0 or node[0] >= bounds[0] or \
           node[1] < 0 or node[1] >= bounds[1]:
            return False
        
        if obstacles[int(node[0]), int(node[1])] > 0:
            return False
        
        return True
    
    def _to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid"""
        return (int(x / self.resolution), int(y / self.resolution))
    
    def _to_world(self, ix: int, iy: int) -> Tuple[float, float]:
        """Convert grid to world coordinates"""
        return (ix * self.resolution, iy * self.resolution)
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Waypoint]:
        """Reconstruct path from A* search"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        
        # Convert to waypoints
        waypoints = []
        for i in range(len(path)):
            x, y = self._to_world(path[i][0], path[i][1])
            
            # Calculate heading
            if i < len(path) - 1:
                next_x, next_y = self._to_world(path[i+1][0], path[i+1][1])
                yaw = np.arctan2(next_y - y, next_x - x)
            else:
                yaw = waypoints[-1].yaw if waypoints else 0.0
            
            waypoints.append(Waypoint(x=x, y=y, yaw=yaw))
        
        return waypoints

class RRTStarPlanner:
    """
    RRT* (Rapidly-exploring Random Tree Star) path planner
    Asymptotically optimal sampling-based planning
    """
    
    def __init__(self, config: PlanningConfig = PLANNING_CONFIG):
        self.config = config
        self.max_iterations = 5000
        self.goal_sample_rate = 0.1
        self.search_radius = 20.0
        self.step_size = 5.0
        
    def plan(self, start: np.ndarray, goal: np.ndarray,
             obstacle_list: List[Tuple[float, float, float]],
             bounds: Tuple[float, float, float, float]) -> Optional[List[Waypoint]]:
        """
        Plan path using RRT*
        
        Args:
            start: Start position [x, y]
            goal: Goal position [x, y]
            obstacle_list: List of (x, y, radius) obstacles
            bounds: (xmin, xmax, ymin, ymax)
            
        Returns:
            Path as list of waypoints
        """
        # Initialize tree
        tree = {0: {'pos': start, 'parent': None, 'cost': 0.0}}
        node_count = 1
        
        for i in range(self.max_iterations):
            # Sample random point
            if np.random.random() < self.goal_sample_rate:
                rnd = goal
            else:
                rnd = self._get_random_point(bounds)
            
            # Find nearest node
            nearest_idx = self._get_nearest_node(tree, rnd)
            nearest_pos = tree[nearest_idx]['pos']
            
            # Steer towards random point
            new_pos = self._steer(nearest_pos, rnd, self.step_size)
            
            # Check collision
            if self._check_collision(nearest_pos, new_pos, obstacle_list):
                continue
            
            # Find nearby nodes
            near_indices = self._find_near_nodes(tree, new_pos)
            
            # Choose parent with minimum cost
            min_cost = tree[nearest_idx]['cost'] + \
                      np.linalg.norm(new_pos - nearest_pos)
            min_idx = nearest_idx
            
            for near_idx in near_indices:
                near_pos = tree[near_idx]['pos']
                cost = tree[near_idx]['cost'] + np.linalg.norm(new_pos - near_pos)
                
                if cost < min_cost and \
                   not self._check_collision(near_pos, new_pos, obstacle_list):
                    min_cost = cost
                    min_idx = near_idx
            
            # Add new node
            tree[node_count] = {
                'pos': new_pos,
                'parent': min_idx,
                'cost': min_cost
            }
            
            # Rewire tree
            for near_idx in near_indices:
                near_pos = tree[near_idx]['pos']
                new_cost = min_cost + np.linalg.norm(near_pos - new_pos)
                
                if new_cost < tree[near_idx]['cost'] and \
                   not self._check_collision(new_pos, near_pos, obstacle_list):
                    tree[near_idx]['parent'] = node_count
                    tree[near_idx]['cost'] = new_cost
            
            node_count += 1
            
            # Check if goal reached
            if np.linalg.norm(new_pos - goal) < self.step_size:
                # Add goal
                tree[node_count] = {
                    'pos': goal,
                    'parent': node_count - 1,
                    'cost': min_cost + np.linalg.norm(goal - new_pos)
                }
                
                # Extract path
                return self._extract_path(tree, node_count)
        
        return None
    
    def _get_random_point(self, bounds: Tuple[float, float, float, float]) -> np.ndarray:
        """Sample random point in bounds"""
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[2], bounds[3])
        return np.array([x, y])
    
    def _get_nearest_node(self, tree: dict, point: np.ndarray) -> int:
        """Find nearest node in tree"""
        min_dist = float('inf')
        nearest = 0
        
        for idx, node in tree.items():
            dist = np.linalg.norm(node['pos'] - point)
            if dist < min_dist:
                min_dist = dist
                nearest = idx
        
        return nearest
    
    def _steer(self, from_pos: np.ndarray, to_pos: np.ndarray, 
               step_size: float) -> np.ndarray:
        """Steer from one position towards another"""
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)
        
        if dist < step_size:
            return to_pos
        
        return from_pos + (direction / dist) * step_size
    
    def _find_near_nodes(self, tree: dict, point: np.ndarray) -> List[int]:
        """Find all nodes within search radius"""
        near_nodes = []
        
        for idx, node in tree.items():
            if np.linalg.norm(node['pos'] - point) <= self.search_radius:
                near_nodes.append(idx)
        
        return near_nodes
    
    def _check_collision(self, from_pos: np.ndarray, to_pos: np.ndarray,
                        obstacles: List[Tuple[float, float, float]]) -> bool:
        """Check if path segment collides with obstacles"""
        # Check multiple points along segment
        num_checks = int(np.linalg.norm(to_pos - from_pos) / 0.5) + 1
        
        for i in range(num_checks):
            t = i / max(num_checks - 1, 1)
            point = from_pos + t * (to_pos - from_pos)
            
            for obs in obstacles:
                if np.linalg.norm(point - np.array([obs[0], obs[1]])) < obs[2]:
                    return True
        
        return False
    
    def _extract_path(self, tree: dict, goal_idx: int) -> List[Waypoint]:
        """Extract path from tree"""
        path = []
        current = goal_idx
        
        while current is not None:
            pos = tree[current]['pos']
            path.append(pos)
            current = tree[current]['parent']
        
        path.reverse()
        
        # Convert to waypoints with headings
        waypoints = []
        for i in range(len(path)):
            if i < len(path) - 1:
                yaw = np.arctan2(path[i+1][1] - path[i][1],
                               path[i+1][0] - path[i][0])
            else:
                yaw = waypoints[-1].yaw if waypoints else 0.0
            
            waypoints.append(Waypoint(x=path[i][0], y=path[i][1], yaw=yaw))
        
        return waypoints

class TrajectoryOptimizer:
    """
    Model Predictive Control based trajectory optimization
    Optimizes trajectory considering dynamics, comfort, and safety
    """
    
    def __init__(self, config: PlanningConfig = PLANNING_CONFIG):
        self.config = config
        self.horizon = 50
        self.dt = 0.1
        
    def optimize_trajectory(self, reference_path: List[Waypoint],
                          initial_state: np.ndarray,
                          obstacles: List[np.ndarray]) -> Trajectory:
        """
        Optimize trajectory along reference path
        
        Args:
            reference_path: Reference waypoints
            initial_state: Current vehicle state [x, y, vx, vy, yaw]
            obstacles: List of obstacle positions
            
        Returns:
            Optimized trajectory
        """
        # Convert waypoints to reference trajectory
        ref_x = np.array([wp.x for wp in reference_path])
        ref_y = np.array([wp.y for wp in reference_path])
        
        # Ensure we have enough points for spline fitting
        if len(ref_x) < 4:
            # Need at least 4 points for cubic spline
            # Interpolate additional points
            if len(ref_x) == 2:
                # Linear interpolation to create 4 points
                extra_points = 2
                t = np.linspace(0, 1, extra_points + 2)
                ref_x = ref_x[0] + t * (ref_x[-1] - ref_x[0])
                ref_y = ref_y[0] + t * (ref_y[-1] - ref_y[0])
            elif len(ref_x) == 3:
                # Add one intermediate point
                mid_idx = len(ref_x) // 2
                new_x = (ref_x[mid_idx] + ref_x[mid_idx + 1]) / 2
                new_y = (ref_y[mid_idx] + ref_y[mid_idx + 1]) / 2
                ref_x = np.insert(ref_x, mid_idx + 1, new_x)
                ref_y = np.insert(ref_y, mid_idx + 1, new_y)
        
        # Parametric curve fitting with proper smoothing
        try:
            # Use splprep with optimal smoothing
            tck, u = splprep([ref_x, ref_y], s=len(ref_x) * 0.1, k=min(3, len(ref_x)-1))
        except:
            # Fallback: use linear interpolation if spline fails
            print("Warning: Spline fitting failed, using linear interpolation")
            u = np.linspace(0, 1, len(ref_x))
            from scipy.interpolate import interp1d
            fx = interp1d(u, ref_x, kind='linear', fill_value='extrapolate')
            fy = interp1d(u, ref_y, kind='linear', fill_value='extrapolate')
            u_new = np.linspace(0, 1, self.horizon)
            x_ref = fx(u_new)
            y_ref = fy(u_new)
        else:
            # Sample trajectory at regular intervals
            u_new = np.linspace(0, 1, self.horizon)
            x_ref, y_ref = splev(u_new, tck)
        
        # Initial guess for optimization with velocity profile
        x0 = np.zeros(self.horizon * 2)  # [vx, vy] at each timestep
        
        # Intelligent initial guess based on path curvature
        for i in range(self.horizon):
            if i < len(x_ref) - 1:
                # Calculate path direction
                dx = x_ref[min(i + 1, len(x_ref) - 1)] - x_ref[i]
                dy = y_ref[min(i + 1, len(y_ref) - 1)] - y_ref[i]
                path_direction = np.arctan2(dy, dx)
                
                # Initial velocity aligned with path
                base_speed = 15.0  # m/s
                x0[i*2] = base_speed * np.cos(path_direction)
                x0[i*2 + 1] = base_speed * np.sin(path_direction)
            else:
                # Maintain last velocity
                x0[i*2] = x0[(i-1)*2] if i > 0 else 10.0
                x0[i*2 + 1] = x0[(i-1)*2 + 1] if i > 0 else 0.0
        
        # Define cost function with enhanced weighting
        def cost_function(controls):
            return self._trajectory_cost(controls, initial_state, 
                                        x_ref, y_ref, obstacles)
        
        # Enhanced constraints with velocity limits
        bounds = []
        for i in range(self.horizon):
            # Velocity bounds based on road conditions
            max_vx = 30.0  # Maximum forward velocity
            max_vy = 10.0  # Maximum lateral velocity
            bounds.append((0.0, max_vx))      # vx bounds (forward only)
            bounds.append((-max_vy, max_vy))  # vy bounds (lateral)
        
        # Multi-start optimization for better solutions
        best_result = None
        best_cost = float('inf')
        
        # Try multiple initial conditions
        num_attempts = 3
        for attempt in range(num_attempts):
            if attempt > 0:
                # Perturb initial guess
                x0_perturbed = x0 + np.random.randn(len(x0)) * 2.0
                x0_perturbed = np.clip(x0_perturbed, 
                                      [b[0] for b in bounds], 
                                      [b[1] for b in bounds])
            else:
                x0_perturbed = x0
            
            # Optimize with different methods
            methods = ['SLSQP', 'trust-constr'] if attempt == 0 else ['SLSQP']
            
            for method in methods:
                try:
                    result = minimize(cost_function, x0_perturbed, method=method,
                                    bounds=bounds, 
                                    options={'maxiter': 200, 'ftol': 1e-6})
                    
                    if result.success and result.fun < best_cost:
                        best_cost = result.fun
                        best_result = result
                except:
                    continue
        
        if best_result is not None and best_result.success:
            optimized_controls = best_result.x
            trajectory = self._generate_trajectory(optimized_controls, 
                                                  initial_state, x_ref, y_ref)
            trajectory.cost = best_cost
            return trajectory
        else:
            # Advanced fallback: use minimum jerk trajectory
            print("Warning: Optimization failed, generating minimum jerk trajectory")
            return self._generate_minimum_jerk_trajectory(
                reference_path, initial_state, x_ref, y_ref
            )
    
    def _generate_minimum_jerk_trajectory(self, reference_path: List[Waypoint],
                                          initial_state: np.ndarray,
                                          x_ref: np.ndarray, 
                                          y_ref: np.ndarray) -> Trajectory:
        """
        Generate smooth minimum jerk trajectory as advanced fallback
        Uses quintic polynomial interpolation
        """
        waypoints = []
        timestamps = np.arange(0, self.horizon * self.dt, self.dt)[:self.horizon]
        velocities = np.zeros(self.horizon)
        accelerations = np.zeros(self.horizon)
        curvatures = np.zeros(self.horizon)
        
        # Initial conditions
        x0, y0 = initial_state[0], initial_state[1]
        vx0, vy0 = initial_state[2], initial_state[3]
        v0 = np.sqrt(vx0**2 + vy0**2)
        
        # Target conditions (from reference path)
        if len(x_ref) > 0:
            xf, yf = x_ref[-1], y_ref[-1]
        else:
            xf, yf = x0 + 50, y0
        
        # Desired final velocity (maintain speed)
        vf = max(v0, 10.0)
        
        # Generate quintic polynomial trajectory
        T = self.horizon * self.dt
        
        for i, t in enumerate(timestamps):
            # Normalized time
            s = t / T
            
            # Quintic polynomial: ensures smooth acceleration
            # p(s) = a0 + a1*s + a2*s^2 + a3*s^3 + a4*s^4 + a5*s^5
            # Boundary conditions:
            # p(0) = 0, p'(0) = 0, p''(0) = 0
            # p(1) = 1, p'(1) = 0, p''(1) = 0
            
            p = 10*s**3 - 15*s**4 + 6*s**5
            p_dot = (30*s**2 - 60*s**3 + 30*s**4) / T
            p_ddot = (60*s - 180*s**2 + 120*s**3) / (T**2)
            
            # Position interpolation
            x = x0 + (xf - x0) * p
            y = y0 + (yf - y0) * p
            
            # Velocity
            vx = (xf - x0) * p_dot
            vy = (yf - y0) * p_dot
            v = np.sqrt(vx**2 + vy**2)
            
            # Heading
            yaw = np.arctan2(vy, vx) if v > 0.1 else initial_state[4]
            
            # Acceleration
            ax = (xf - x0) * p_ddot
            ay = (yf - y0) * p_ddot
            
            # Curvature (simplified)
            if i > 0 and v > 0.1:
                prev_yaw = waypoints[-1].yaw
                dyaw = yaw - prev_yaw
                curvatures[i] = dyaw / (v * self.dt)
            
            waypoints.append(Waypoint(
                x=x, y=y, yaw=yaw,
                velocity=v,
                curvature=curvatures[i]
            ))
            
            velocities[i] = v
            if i > 0:
                accelerations[i] = (v - velocities[i-1]) / self.dt
        
        return Trajectory(
            waypoints=waypoints,
            timestamps=timestamps,
            curvatures=curvatures,
            velocities=velocities,
            accelerations=accelerations,
            cost=0.0
        )
    
    def _trajectory_cost(self, controls: np.ndarray, initial_state: np.ndarray,
                        x_ref: np.ndarray, y_ref: np.ndarray,
                        obstacles: List[np.ndarray]) -> float:
        """
        Calculate cost of trajectory
        
        Considers:
        - Tracking error
        - Smoothness
        - Obstacle avoidance
        - Comfort
        """
        cost = 0.0
        state = initial_state.copy()
        
        for i in range(self.horizon):
            vx = controls[i*2]
            vy = controls[i*2 + 1]
            
            # Update state
            state[0] += vx * self.dt * np.cos(state[4])
            state[1] += vx * self.dt * np.sin(state[4])
            state[2] = vx
            state[3] = vy
            
            # Tracking error
            if i < len(x_ref):
                tracking_error = (state[0] - x_ref[i])**2 + (state[1] - y_ref[i])**2
                cost += self.config.weight_progress * tracking_error
            
            # Acceleration penalty (comfort)
            if i > 0:
                ax = (controls[i*2] - controls[(i-1)*2]) / self.dt
                ay = (controls[i*2+1] - controls[(i-1)*2+1]) / self.dt
                cost += self.config.weight_comfort * (ax**2 + ay**2)
            
            # Obstacle avoidance
            for obs in obstacles:
                dist = np.sqrt((state[0] - obs[0])**2 + (state[1] - obs[1])**2)
                if dist < 10.0:  # Safety margin
                    cost += self.config.weight_safety * (10.0 - dist)**2
        
        return cost
    
    def _generate_trajectory(self, controls: np.ndarray, initial_state: np.ndarray,
                           x_ref: np.ndarray, y_ref: np.ndarray) -> Trajectory:
        """Generate trajectory from optimized controls"""
        waypoints = []
        state = initial_state.copy()
        timestamps = np.arange(0, self.horizon * self.dt, self.dt)
        velocities = np.zeros(self.horizon)
        accelerations = np.zeros(self.horizon)
        curvatures = np.zeros(self.horizon)
        
        for i in range(self.horizon):
            vx = controls[i*2]
            vy = controls[i*2 + 1]
            
            # Calculate heading
            yaw = np.arctan2(vy, vx) if vx != 0 or vy != 0 else state[4]
            
            # Calculate curvature
            if i > 0 and vx > 0.1:
                dyaw = yaw - waypoints[-1].yaw
                curvatures[i] = dyaw / (vx * self.dt)
            
            waypoints.append(Waypoint(
                x=state[0],
                y=state[1],
                yaw=yaw,
                velocity=vx,
                curvature=curvatures[i]
            ))
            
            velocities[i] = vx
            if i > 0:
                accelerations[i] = (vx - velocities[i-1]) / self.dt
            
            # Update state
            state[0] += vx * self.dt * np.cos(yaw)
            state[1] += vx * self.dt * np.sin(yaw)
            state[4] = yaw
        
        return Trajectory(
            waypoints=waypoints,
            timestamps=timestamps,
            curvatures=curvatures,
            velocities=velocities,
            accelerations=accelerations
        )