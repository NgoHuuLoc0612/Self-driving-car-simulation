"""
Machine Learning Utilities
Clustering, dimensionality reduction, and training visualization with TensorBoard
"""

import numpy as np
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple, Optional, Dict
import torch
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
import time

@dataclass
class ClusteringResult:
    """Results from clustering algorithm"""
    labels: np.ndarray
    n_clusters: int
    centroids: Optional[np.ndarray] = None
    silhouette_score: float = 0.0

class LiDARClustering:
    """
    Advanced LiDAR point cloud clustering
    Uses DBSCAN for density-based clustering
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 10):
        """
        Args:
            eps: Maximum distance between points in cluster
            min_samples: Minimum points to form dense region
        """
        self.eps = eps
        self.min_samples = min_samples
        self.scaler = StandardScaler()
        
    def cluster_points(self, points: np.ndarray) -> ClusteringResult:
        """
        Cluster 3D point cloud
        
        Args:
            points: [N x 3] array of points
            
        Returns:
            ClusteringResult with labels and statistics
        """
        if len(points) == 0:
            return ClusteringResult(labels=np.array([]), n_clusters=0)
        
        # Normalize points
        points_normalized = self.scaler.fit_transform(points)
        
        # DBSCAN clustering
        clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        labels = clusterer.fit_predict(points_normalized)
        
        # Count clusters (excluding noise label -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Compute centroids
        centroids = []
        for cluster_id in range(n_clusters):
            cluster_points = points[labels == cluster_id]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)
        
        centroids = np.array(centroids) if centroids else None
        
        return ClusteringResult(
            labels=labels,
            n_clusters=n_clusters,
            centroids=centroids
        )
    
    def extract_objects(self, points: np.ndarray, labels: np.ndarray) -> List[Dict]:
        """
        Extract object information from clustered points
        
        Returns:
            List of object dictionaries with position, size, etc.
        """
        objects = []
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise
        
        for cluster_id in unique_labels:
            cluster_points = points[labels == cluster_id]
            
            if len(cluster_points) < 5:  # Too few points
                continue
            
            # Compute bounding box
            min_coords = np.min(cluster_points, axis=0)
            max_coords = np.max(cluster_points, axis=0)
            
            center = (min_coords + max_coords) / 2
            size = max_coords - min_coords
            
            # Classify object type based on size
            object_type = self._classify_object(size)
            
            objects.append({
                'id': cluster_id,
                'center': center,
                'size': size,
                'num_points': len(cluster_points),
                'type': object_type,
                'points': cluster_points
            })
        
        return objects
    
    def _classify_object(self, size: np.ndarray) -> str:
        """Classify object based on dimensions"""
        length = np.max(size[:2])
        width = np.min(size[:2])
        height = size[2] if len(size) > 2 else 0
        
        if length > 3.5 and width > 1.5:
            return 'vehicle'
        elif height > 1.5 and length < 1.0:
            return 'pedestrian'
        elif length < 0.5 and height < 0.5:
            return 'small_object'
        else:
            return 'unknown'

class TrajectoryAnalyzer:
    """
    Analyze and cluster trajectories
    Uses Dynamic Time Warping and hierarchical clustering
    """
    
    def __init__(self):
        self.pca = PCA(n_components=10)
        
    def extract_trajectory_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Extract features from trajectory for learning
        
        Args:
            trajectory: [T x state_dim] trajectory
            
        Returns:
            Feature vector
        """
        features = []
        
        # Statistical features
        features.append(np.mean(trajectory, axis=0))
        features.append(np.std(trajectory, axis=0))
        features.append(np.min(trajectory, axis=0))
        features.append(np.max(trajectory, axis=0))
        
        # Velocity features
        if len(trajectory) > 1:
            velocities = np.diff(trajectory, axis=0)
            features.append(np.mean(velocities, axis=0))
            features.append(np.std(velocities, axis=0))
        
        # Acceleration features
        if len(trajectory) > 2:
            accelerations = np.diff(velocities, axis=0)
            features.append(np.mean(accelerations, axis=0))
            features.append(np.std(accelerations, axis=0))
        
        # Flatten
        feature_vector = np.concatenate([f.flatten() for f in features])
        
        return feature_vector
    
    def cluster_trajectories(self, trajectories: List[np.ndarray], 
                            n_clusters: int = 5) -> np.ndarray:
        """
        Cluster similar trajectories
        
        Args:
            trajectories: List of trajectory arrays
            n_clusters: Number of clusters
            
        Returns:
            Cluster labels for each trajectory
        """
        # Extract features from all trajectories
        features = []
        for traj in trajectories:
            feat = self.extract_trajectory_features(traj)
            features.append(feat)
        
        features = np.array(features)
        
        # Normalize
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # PCA for dimensionality reduction
        if features_normalized.shape[1] > 10:
            features_pca = self.pca.fit_transform(features_normalized)
        else:
            features_pca = features_normalized
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_pca)
        
        return labels
    
    def find_representative_trajectories(self, trajectories: List[np.ndarray],
                                        labels: np.ndarray) -> List[np.ndarray]:
        """
        Find representative trajectory for each cluster
        
        Returns:
            List of representative trajectories
        """
        representatives = []
        
        for cluster_id in range(np.max(labels) + 1):
            cluster_trajs = [trajectories[i] for i in range(len(trajectories)) 
                           if labels[i] == cluster_id]
            
            if not cluster_trajs:
                continue
            
            # Find trajectory closest to cluster mean
            features = np.array([self.extract_trajectory_features(t) 
                               for t in cluster_trajs])
            mean_feature = np.mean(features, axis=0)
            
            distances = [np.linalg.norm(f - mean_feature) for f in features]
            representative_idx = np.argmin(distances)
            
            representatives.append(cluster_trajs[representative_idx])
        
        return representatives

class StateSpaceAnalyzer:
    """
    Analyze high-dimensional state spaces
    Uses t-SNE and PCA for visualization
    """
    
    def __init__(self):
        self.pca = PCA(n_components=50)
        self.tsne = TSNE(n_components=2, random_state=42)
        
    def reduce_dimensionality(self, states: np.ndarray, method: str = 'pca') -> np.ndarray:
        """
        Reduce state space dimensionality for visualization
        
        Args:
            states: [N x state_dim] state vectors
            method: 'pca' or 'tsne'
            
        Returns:
            [N x 2] reduced states
        """
        # Normalize
        scaler = StandardScaler()
        states_normalized = scaler.fit_transform(states)
        
        if method == 'pca':
            # PCA to 2D
            pca_2d = PCA(n_components=2)
            reduced = pca_2d.fit_transform(states_normalized)
            
            variance_explained = pca_2d.explained_variance_ratio_
            print(f"PCA variance explained: {sum(variance_explained):.3f}")
            
        elif method == 'tsne':
            # First reduce with PCA if high-dimensional
            if states_normalized.shape[1] > 50:
                states_pca = self.pca.fit_transform(states_normalized)
            else:
                states_pca = states_normalized
            
            # Then t-SNE
            reduced = self.tsne.fit_transform(states_pca)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return reduced
    
    def identify_state_clusters(self, states: np.ndarray, n_clusters: int = 5) -> np.ndarray:
        """Identify distinct regions in state space"""
        reduced = self.reduce_dimensionality(states, method='pca')
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(reduced)
        
        return labels

class RLTrainingMonitor:
    """
    TensorBoard-based training monitor for RL agents
    Logs metrics, gradients, and visualizations
    """
    
    def __init__(self, log_dir: str = './runs'):
        """
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.writer = SummaryWriter(log_dir)
        self.step = 0
        
        # Tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        
    def log_episode(self, reward: float, length: int, info: Dict = None):
        """
        Log episode statistics
        
        Args:
            reward: Total episode reward
            length: Episode length
            info: Additional information
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        # Log to TensorBoard
        self.writer.add_scalar('Episode/Reward', reward, self.step)
        self.writer.add_scalar('Episode/Length', length, self.step)
        
        if info:
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Episode/{key}', value, self.step)
        
        # Moving averages
        if len(self.episode_rewards) >= 100:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
            
            self.writer.add_scalar('Episode/Avg_Reward_100', avg_reward, self.step)
            self.writer.add_scalar('Episode/Avg_Length_100', avg_length, self.step)
        
        self.step += 1
    
    def log_training_step(self, loss: float, learning_rate: float, 
                         additional_metrics: Dict = None):
        """
        Log training step metrics
        
        Args:
            loss: Training loss
            learning_rate: Current learning rate
            additional_metrics: Additional metrics to log
        """
        self.losses.append(loss)
        
        self.writer.add_scalar('Training/Loss', loss, self.step)
        self.writer.add_scalar('Training/Learning_Rate', learning_rate, self.step)
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f'Training/{key}', value, self.step)
    
    def log_network_weights(self, model: torch.nn.Module, step: Optional[int] = None):
        """Log network weights and gradients"""
        if step is None:
            step = self.step
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Log weights
                self.writer.add_histogram(f'Weights/{name}', param.data, step)
                
                # Log gradients
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, step)
                    
                    # Gradient norms
                    grad_norm = torch.norm(param.grad).item()
                    self.writer.add_scalar(f'Gradient_Norms/{name}', grad_norm, step)
    
    def log_trajectory_visualization(self, trajectory: np.ndarray, step: Optional[int] = None):
        """
        Visualize trajectory in TensorBoard
        
        Args:
            trajectory: [T x 2] positions
            step: Global step
        """
        if step is None:
            step = self.step
        
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, label='Start')
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, label='End')
        ax.legend()
        ax.grid(True)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title(f'Trajectory at Step {step}')
        
        self.writer.add_figure('Trajectory', fig, step)
        plt.close(fig)
    
    def log_state_distribution(self, states: np.ndarray, step: Optional[int] = None):
        """Log state space distribution"""
        if step is None:
            step = self.step
        
        # Log histograms for each state dimension
        for i in range(min(states.shape[1], 10)):  # Limit to first 10 dimensions
            self.writer.add_histogram(f'State_Distribution/dim_{i}', 
                                     states[:, i], step)
    
    def log_hyperparameters(self, hparams: Dict, metrics: Dict):
        """
        Log hyperparameters and final metrics
        
        Args:
            hparams: Dictionary of hyperparameters
            metrics: Dictionary of final metrics
        """
        self.writer.add_hparams(hparams, metrics)
    
    def close(self):
        """Close TensorBoard writer"""
        self.writer.close()

class ExperienceReplayAnalyzer:
    """
    Analyze experience replay buffer for RL training
    Identifies diverse and informative experiences
    """
    
    def __init__(self, state_dim: int):
        self.state_dim = state_dim
        self.nn = NearestNeighbors(n_neighbors=5)
        
    def compute_state_novelty(self, states: np.ndarray, new_state: np.ndarray) -> float:
        """
        Compute novelty of new state relative to buffer
        
        Args:
            states: Existing states in buffer [N x state_dim]
            new_state: New state to evaluate [state_dim]
            
        Returns:
            Novelty score (higher = more novel)
        """
        if len(states) < 5:
            return 1.0
        
        # Fit nearest neighbors
        self.nn.fit(states)
        
        # Find distance to k nearest neighbors
        distances, _ = self.nn.kneighbors(new_state.reshape(1, -1))
        
        # Average distance is novelty score
        novelty = np.mean(distances)
        
        return novelty
    
    def prioritize_experiences(self, states: np.ndarray, rewards: np.ndarray,
                              td_errors: np.ndarray) -> np.ndarray:
        """
        Compute priority scores for experiences
        
        Combines:
        - TD error (how surprising)
        - Reward magnitude (how important)
        - State diversity (how representative)
        
        Returns:
            Priority scores [N]
        """
        # Normalize components
        td_normalized = (td_errors - np.min(td_errors)) / (np.max(td_errors) - np.min(td_errors) + 1e-8)
        reward_normalized = np.abs(rewards) / (np.max(np.abs(rewards)) + 1e-8)
        
        # Compute state diversity scores
        diversity_scores = np.zeros(len(states))
        for i in range(len(states)):
            # Distance to other states
            dists = np.linalg.norm(states - states[i], axis=1)
            dists[i] = np.inf  # Exclude self
            diversity_scores[i] = np.min(dists)
        
        diversity_normalized = (diversity_scores - np.min(diversity_scores)) / \
                              (np.max(diversity_scores) - np.min(diversity_scores) + 1e-8)
        
        # Combine
        priorities = 0.4 * td_normalized + 0.4 * reward_normalized + 0.2 * diversity_normalized
        
        return priorities