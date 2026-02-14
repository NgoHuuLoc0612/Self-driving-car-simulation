#!/usr/bin/env python3
"""
Simulation Launcher
Runs the autonomous vehicle simulation with available features
"""

import sys
import os

def check_dependencies():
    """Check which optional dependencies are available"""
    available = {
        'core': True,
        'spatial': False,
        'visualization': False,
        'ml_utils': False,
        'routing': False
    }
    
    print("Checking dependencies...")
    print("-" * 60)
    
    # Core dependencies (required)
    try:
        import numpy
        import scipy
        import matplotlib
        import torch
        print("✓ Core dependencies (numpy, scipy, matplotlib, torch)")
        available['core'] = True
    except ImportError as e:
        print(f"✗ Missing core dependency: {e}")
        available['core'] = False
    
    # Spatial indexing
    try:
        import rtree
        import shapely
        import numba
        print("✓ Spatial indexing (rtree, shapely, numba)")
        available['spatial'] = True
    except ImportError:
        print("⚠ Spatial indexing not available (optional)")
    
    # Visualization
    try:
        import pygame
        import cv2
        from PIL import Image
        print("✓ Advanced visualization (pygame, opencv, Pillow)")
        available['visualization'] = True
    except ImportError:
        print("⚠ Advanced visualization not available (optional)")
    
    # ML utilities
    try:
        import sklearn
        from torch.utils.tensorboard import SummaryWriter
        print("✓ ML utilities (scikit-learn, tensorboard)")
        available['ml_utils'] = True
    except ImportError:
        print("⚠ ML utilities not available (optional)")
    
    # Network routing
    try:
        import networkx
        print("✓ Network routing (networkx)")
        available['routing'] = True
    except ImportError:
        print("⚠ Network routing not available (optional)")
    
    print("-" * 60)
    print()
    
    return available

def run_simulation(mode='auto', duration=60.0, visualize=True):
    """
    Run the simulation
    
    Args:
        mode: 'auto', 'pygame', or 'matplotlib'
        duration: Simulation duration in seconds
        visualize: Enable visualization
    """
    available = check_dependencies()
    
    if not available['core']:
        print("ERROR: Core dependencies missing!")
        print("Please install: pip install numpy scipy matplotlib torch")
        return False
    
    # Import simulation
    try:
        from main import Simulation
        import numpy as np
    except ImportError as e:
        print(f"ERROR: Could not import simulation: {e}")
        return False
    
    print("="*60)
    print("AUTONOMOUS VEHICLE SIMULATION")
    print("="*60)
    print()
    
    # Show available features
    print("Available features:")
    if available['spatial']:
        print("  ✓ R-tree spatial indexing and advanced collision detection")
    if available['visualization']:
        print("  ✓ Interactive Pygame visualization with camera controls")
    if available['ml_utils']:
        print("  ✓ ML-based clustering and TensorBoard logging")
    if available['routing']:
        print("  ✓ Graph-based routing with traffic awareness")
    print()
    
    # Create and initialize simulation
    try:
        print("Initializing simulation...")
        sim = Simulation()
        
        # Set up scenario
        start_position = np.array([10.0, 0.0])
        goal_position = np.array([500.0, 0.0])
        
        print(f"Start: {start_position}")
        print(f"Goal: {goal_position}")
        print()
        
        # Initialize
        sim.initialize(start_position, goal_position)
        
        # Run
        print(f"Running simulation for {duration} seconds...")
        print("Press ESC to exit (in Pygame mode)")
        print()
        
        sim.run(duration=duration, visualize=visualize)
        
        print()
        print("Simulation completed successfully!")
        return True
        
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
        return True
    except Exception as e:
        print(f"\nERROR during simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Launch autonomous vehicle simulation'
    )
    parser.add_argument(
        '--duration', type=float, default=60.0,
        help='Simulation duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--no-viz', action='store_true',
        help='Disable visualization'
    )
    parser.add_argument(
        '--mode', choices=['auto', 'pygame', 'matplotlib'], default='auto',
        help='Visualization mode (default: auto)'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Run advanced features demo instead of simulation'
    )
    
    args = parser.parse_args()
    
    if args.demo:
        # Run demo
        print("Running advanced features demo...")
        print()
        try:
            import demo_advanced
        except Exception as e:
            print(f"Error running demo: {e}")
            return 1
        return 0
    else:
        # Run simulation
        success = run_simulation(
            mode=args.mode,
            duration=args.duration,
            visualize=not args.no_viz
        )
        return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())