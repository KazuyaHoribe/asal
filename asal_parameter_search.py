#!/usr/bin/env python
"""
ASAL Framework-based Parameter Search for Causal Emergence Analysis

This script leverages the ASAL distributed computation framework to perform
large-scale parameter searches for identifying conditions that maximize
causal emergence metrics (Delta, Gamma, Psi).
"""

import os
import sys
import numpy as np
import pandas as pd
import json
import time
import argparse
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import itertools
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
import jax
import jax.numpy as jnp

# Ensure the script is run from the correct directory
if not os.path.exists('causal_emergence.py'):
    print("Error: This script should be run from the directory containing 'causal_emergence.py'")
    sys.exit(1)

# Add the ASAL directory to the path if needed
asal_path = './asal'
if asal_path not in sys.path:
    sys.path.insert(0, asal_path)

# Import causal emergence modules
from causal_emergence import calculate_emergence_metrics, run_simulation
from parameter_search import analyze_search_results, run_optimal_simulation

# Import foundation models and substrate creation functions
try:
    import foundation_models
    import substrates
    from rollout import rollout_simulation
    import asal_metrics
    HAS_ASAL_MODULES = True
except ImportError as e:
    print(f"Warning: Could not import core ASAL modules: {e}")
    print("Some features will be limited to basic causal emergence analysis.")
    HAS_ASAL_MODULES = False

# --- Parameter Search Configuration ---

# Expanded parameter ranges for Game of Life
DEFAULT_GOL_PARAMS = {
    "grid_size": [16, 24, 32],
    "init_density": [0.1, 0.2, 0.3],
    # B/S notation parameters for Life-like cellular automata
    "birth": [
        [3],                  # Conway's Game of Life B3/S23
        [2, 3],               # Day & Night
        [3, 6],               # HighLife  
        [1, 3, 5, 7],         # Replicator
        [3, 5, 7],            # Diamoeba
        [2],                  # Seeds
        [4, 6, 7, 8],         # Life without Death
        [1],                  # Gnarl
        [3, 6, 7, 8],         # Maze
        [2, 4, 5]             # Serviettes
    ],
    "survive": [
        [2, 3],               # Conway's Game of Life B3/S23
        [0, 2, 4, 6, 8],      # Day & Night
        [2, 3],               # HighLife
        [1, 3, 5, 7],         # Replicator
        [5, 6, 7, 8],         # Diamoeba
        [],                   # Seeds
        [0, 1, 2, 3, 4, 5, 6, 7, 8], # Life without Death
        [],                   # Gnarl
        [1, 2, 3, 4, 5],      # Maze
        [3]                   # Serviettes
    ],
    "n_steps": [250, 500]
}

# Expanded parameter ranges for Boids
DEFAULT_BOIDS_PARAMS = {
    "n_boids": [20, 30, 50, 70, 100, 150, 200, 300],
    "visual_range": [3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0, 30.0],
    "protected_range": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    "max_speed": [1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0],
    "cohesion_factor": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
    "alignment_factor": [0.01, 0.05, 0.1, 0.125, 0.2, 0.3, 0.5],
    "separation_factor": [0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5],
    # New parameters
    "turn_factor": [0.1, 0.2, 0.3, 0.5],         # Factor for boundary avoidance
    "turbulence": [0.0, 0.01, 0.05, 0.1],        # Random movement factor
    "obstacle_avoidance": [0.0, 0.2, 0.5, 1.0],  # Obstacle avoidance factor
    "n_steps": [250, 500, 750, 1000, 1500, 2000]
}

# Expanded parameter ranges for Lenia
DEFAULT_LENIA_PARAMS = {
    "grid_size": [64, 96, 128, 192, 256],
    "time_step": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
    "kernel_radius": [5, 10, 13, 15, 20, 25, 30],
    "kernel_peak": [0.05, 0.1, 0.15, 0.2],     # New: Peak value in kernel
    "growth_center": [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4],
    "growth_width": [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
    "init_radius": [5, 10, 15, 20, 25, 30, 40],
    # New parameters
    "init_pattern": ["random", "circle", "square", "glider"], # Initial pattern type
    "multiple_patterns": [1, 2, 3, 4],                       # Number of initial patterns
    "n_steps": [250, 500, 750, 1000, 1500, 2000]
}

# --- Utility Functions ---

def get_parameter_combinations(param_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters from a dictionary of parameter lists."""
    # Handle special cases like birth/survive rules for GoL
    special_params = []
    special_values = []
    regular_params = {}
    
    for key, values in param_dict.items():
        # Check if the parameter has a list of lists (like birth/survive rules)
        if isinstance(values, list) and len(values) > 0 and isinstance(values[0], list):
            special_params.append(key)
            special_values.append(values)
        else:
            regular_params[key] = values
    
    # Generate regular parameter combinations
    param_names = list(regular_params.keys())
    param_values = list(regular_params.values())
    regular_combinations = list(itertools.product(*param_values))
    regular_dicts = [dict(zip(param_names, combo)) for combo in regular_combinations]
    
    # If no special parameters, return regular combinations
    if not special_params:
        return regular_dicts
    
    # Handle special parameters
    final_combinations = []
    special_combinations = list(itertools.product(*special_values))
    
    for regular_dict in regular_dicts:
        for special_combo in special_combinations:
            combined_dict = regular_dict.copy()
            for i, param in enumerate(special_params):
                combined_dict[param] = special_combo[i]
            final_combinations.append(combined_dict)
    
    return final_combinations

# Enhanced Latin Hypercube Sampling for continuous parameters
def generate_latin_hypercube_samples(param_ranges, n_samples):
    """
    Generate Latin Hypercube samples for efficient parameter space exploration.
    
    Args:
        param_ranges: Dictionary with parameter names as keys and [min, max] ranges as values
        n_samples: Number of samples to generate
        
    Returns:
        List of parameter dictionaries
    """
    from scipy.stats import qmc
    
    # Get parameter names and ranges
    param_names = list(param_ranges.keys())
    n_dims = len(param_names)
    
    # Create parameter bounds for continuous parameters
    bounds = []
    for name in param_names:
        if isinstance(param_ranges[name], list) and len(param_ranges[name]) == 2 and all(isinstance(x, (int, float)) for x in param_ranges[name]):
            bounds.append(param_ranges[name])
        else:
            # Skip non-continuous parameters
            return None
    
    # Generate Latin Hypercube samples
    sampler = qmc.LatinHypercube(d=n_dims)
    samples = sampler.random(n_samples)
    
    # Scale samples to the parameter ranges
    scaled_samples = qmc.scale(samples, [b[0] for b in bounds], [b[1] for b in bounds])
    
    # Convert to list of parameter dictionaries
    param_dicts = []
    for sample in scaled_samples:
        param_dict = {}
        for i, name in enumerate(param_names):
            # Round integer parameters
            if isinstance(param_ranges[name][0], int):
                param_dict[name] = int(round(sample[i]))
            else:
                param_dict[name] = sample[i]
        param_dicts.append(param_dict)
    
    return param_dicts

def run_single_simulation_with_asal(
    params: Dict[str, Any], 
    simulation_type: str, 
    seed: int,
    rollout_steps: int = 500
) -> Dict[str, Any]:
    """Run a single simulation with ASAL framework and calculate emergence metrics."""
    results = {"params": params, "seed": seed, "simulation_type": simulation_type}
    
    try:
        # Clear GPU memory between runs to prevent memory exhaustion
        if hasattr(jax, 'clear_caches'):
            jax.clear_caches()  # Clear JAX compilation cache
        
        # Set up foundation model and substrate
        rng = jax.random.PRNGKey(seed)
        
        # Try with CPU fallback if GPU memory is low
        try:
            fm = foundation_models.create_foundation_model('clip')
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print("GPU memory exhausted, falling back to CPU...")
                # Force CPU execution
                with jax.experimental.maps.Mesh(jax.devices("cpu"), ()):
                    fm = foundation_models.create_foundation_model('clip')
            else:
                raise
        
        if simulation_type == 'gol':
            substrate = substrates.create_substrate('gol')
            # Apply custom parameters 
            substrate_params = substrate.default_params(rng)
            if 'grid_size' in params:
                substrate_params['grid_size'] = params['grid_size']
            if 'birth' in params:
                substrate_params['birth_rule'] = jnp.array(params['birth'])
            if 'survive' in params:
                substrate_params['survive_rule'] = jnp.array(params['survive'])
                
        elif simulation_type == 'boids':
            substrate = substrates.create_substrate('boids')
            # Apply custom parameters
            substrate_params = substrate.default_params(rng)
            for key in ['n_boids', 'visual_range', 'protected_range', 
                        'max_speed', 'cohesion_factor', 'alignment_factor', 
                        'separation_factor']:
                if key in params:
                    substrate_params[key] = params[key]
                    
        elif simulation_type == 'lenia':
            substrate = substrates.create_substrate('lenia')
            # Apply custom parameters
            substrate_params = substrate.default_params(rng)
            for key in ['grid_size', 'time_step', 'kernel_radius', 
                        'growth_center', 'growth_width', 'init_radius']:
                if key in params:
                    substrate_params[key] = params[key]
        else:
            raise ValueError(f"Unsupported simulation type: {simulation_type}")
        
        # Create rollout function
        n_steps = params.get('n_steps', rollout_steps)
        time_sampling = 'video'  # Capture full history for causal analysis
        
        rollout_fn = partial(
            rollout_simulation, 
            s0=None, 
            substrate=substrate, 
            fm=fm, 
            rollout_steps=n_steps,
            time_sampling=time_sampling,
            img_size=224, 
            return_state=True
        )
        
        # JIT compile for performance
        rollout_fn = jax.jit(rollout_fn)
        
        # Run simulation
        rollout_data = rollout_fn(rng, substrate_params)
        
        # Extract data for causal emergence analysis
        rgb_sequence = np.array(rollout_data['rgb'])
        z_sequence = np.array(rollout_data['z'])
        states = np.array(rollout_data['state'])
        
        # Calculate emergence metrics
        
        # 1. ASAL open-endedness score
        oe_score = float(asal_metrics.calc_open_endedness_score(z_sequence))
        results["open_endedness_score"] = oe_score
        
        # 2. Create micro and macro states for causal emergence analysis
        if simulation_type == 'gol':
            # For GoL, the grid itself is the micro state
            S_history = np.array([state.flatten() for state in states])
            # Macro state could be density, pattern entropy, etc.
            M_history = np.array([
                [
                    np.mean(state > 0),  # density
                    np.std(state > 0),   # spatial heterogeneity
                ]
                for state in states
            ])
        elif simulation_type == 'boids':
            # For boids, positions and velocities are micro states
            S_history = states  # Assuming states contains positions and velocities
            # Macro states could be polarization, cohesion, etc.
            M_history = np.array([
                calculate_boids_macro_state(state)
                for state in states
            ])
        elif simulation_type == 'lenia':
            # For Lenia, the grid is the micro state
            S_history = np.array([state.flatten() for state in states])
            # Macro state could be activity measures, pattern statistics, etc.
            M_history = np.array([
                [
                    np.mean(state),          # average activity
                    np.max(state),           # maximum activity
                    np.count_nonzero(state), # active cells count
                ]
                for state in states
            ])
            
        # Calculate causal emergence metrics
        needs_discretization = (simulation_type != 'gol')  # GoL is already discrete
        ce_metrics = calculate_emergence_metrics(S_history, M_history, discretize=needs_discretization)
        
        if ce_metrics is None:
            raise RuntimeError("Failed to calculate emergence metrics")
        
        # Add metrics to results
        for key, value in ce_metrics.items():
            results[key] = float(value)  # Convert numpy values to Python floats
        
        # Add simulation info
        results["success"] = True
        results["rgb_shape"] = list(rgb_sequence.shape)
        results["z_shape"] = list(z_sequence.shape)
        results["S_shape"] = list(S_history.shape)
        results["M_shape"] = list(M_history.shape)
        
        # Save embedding features for future analysis
        results["z_embedding"] = z_sequence[-1].tolist()  # Save final state embedding
        
    except Exception as e:
        # Log the error and mark as failed
        results["success"] = False
        results["error"] = str(e)
        import traceback
        results["traceback"] = traceback.format_exc()
    
    return results

def calculate_boids_macro_state(state):
    """Calculate macro state features for Boids simulation."""
    positions, velocities = state[:, :2], state[:, 2:4]
    n_boids = len(positions)
    
    # Polarization (alignment)
    avg_velocity = np.mean(velocities, axis=0)
    polarization = np.linalg.norm(avg_velocity) / np.mean(np.linalg.norm(velocities, axis=1))
    
    # Cohesion (average distance from center of mass)
    center_of_mass = np.mean(positions, axis=0)
    avg_distance = np.mean(np.linalg.norm(positions - center_of_mass, axis=1))
    
    # Spatial extent (radius of the flock)
    max_distance = np.max(np.linalg.norm(positions - center_of_mass, axis=1))
    
    # Nearest neighbor distance
    nn_distances = []
    for i in range(n_boids):
        distances = np.linalg.norm(positions[i] - positions, axis=1)
        distances[i] = np.inf  # Exclude self
        nn_distances.append(np.min(distances))
    avg_nn_distance = np.mean(nn_distances)
    
    return [polarization, avg_distance, max_distance, avg_nn_distance]

def run_standard_simulation(params: Dict[str, Any], simulation_type: str, seed: int) -> Dict[str, Any]:
    """Run a single simulation with standard causal_emergence framework."""
    results = {"params": params, "seed": seed, "simulation_type": simulation_type}
    
    try:
        # Set up the simulation parameters by modifying globals in causal_emergence.py
        if simulation_type == 'gol':
            # Update Game of Life parameters
            import causal_emergence as ce
            ce.GRID_SIZE = params.get('grid_size', 32)
            ce.RANDOM_DENSITY = params.get('init_density', 0.1)
            n_steps = params.get('n_steps', 500)
        elif simulation_type == 'boids':
            # Update Boids parameters
            import causal_emergence as ce
            ce.N_BOIDS = params.get('n_boids', 50)
            ce.VISUAL_RANGE = params.get('visual_range', 10.0)
            ce.PROTECTED_RANGE = params.get('protected_range', 2.0)
            ce.MAX_SPEED = params.get('max_speed', 5.0)
            ce.COHESION_FACTOR = params.get('cohesion_factor', 0.01)
            ce.ALIGNMENT_FACTOR = params.get('alignment_factor', 0.125)
            ce.SEPARATION_FACTOR = params.get('separation_factor', 0.05)
            n_steps = params.get('n_steps', 500)
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")
        
        # Run the simulation
        S_history, M_history, visual_frames = run_simulation(simulation_type, n_steps, seed=seed)
        
        if S_history is None or M_history is None:
            raise RuntimeError("Simulation failed")
        
        # Calculate metrics
        needs_discretization = (simulation_type == 'boids')
        metrics = calculate_emergence_metrics(S_history, M_history, discretize=needs_discretization)
        
        if metrics is None:
            raise RuntimeError("Failed to calculate emergence metrics")
        
        # Add metrics to results
        for key, value in metrics.items():
            results[key] = float(value)  # Convert numpy values to Python floats
        
        # Add simulation info
        results["success"] = True
        results["S_shape"] = list(S_history.shape)
        results["M_shape"] = list(M_history.shape)
        
    except Exception as e:
        # Log the error and mark as failed
        results["success"] = False
        results["error"] = str(e)
    
    return results

def run_parameter_search(
    simulation_type: str,
    param_dict: Dict[str, List[Any]],
    use_asal: bool = True,
    n_seeds: int = 3,
    start_seed: int = 42,
    n_jobs: int = 1,
    save_dir: Optional[str] = None,
    search_mode: str = "grid",
    n_samples: int = 100,
    adaptive_sampling: bool = False
) -> pd.DataFrame:
    """
    Run a parameter search with multiple combinations of parameters.
    
    Args:
        simulation_type: Type of simulation ('gol', 'boids', 'lenia', etc.)
        param_dict: Dictionary mapping parameter names to lists of values
        use_asal: Whether to use ASAL framework for simulations
        n_seeds: Number of random seeds to try for each parameter combination
        start_seed: Starting random seed
        n_jobs: Number of parallel jobs to run (use -1 for all cores)
        save_dir: Directory to save intermediate results
        search_mode: "grid" for grid search, "random" for random search, or "lhs" for Latin Hypercube Sampling
        n_samples: Number of random samples if search_mode is 'random' or 'lhs'
        adaptive_sampling: Whether to use adaptive sampling (focus on promising regions)
        
    Returns:
        DataFrame containing results for all parameter combinations
    """
    # Generate parameter combinations based on search mode
    if search_mode == "grid":
        param_combinations = get_parameter_combinations(param_dict)
        print(f"Generated {len(param_combinations)} parameter combinations using grid search")
    
    elif search_mode == "random":
        all_combinations = get_parameter_combinations(param_dict)
        if n_samples >= len(all_combinations):
            param_combinations = all_combinations
            print(f"Using all {len(param_combinations)} parameter combinations")
        else:
            random.seed(start_seed)
            param_combinations = random.sample(all_combinations, n_samples)
            print(f"Randomly selected {n_samples} parameter combinations out of {len(all_combinations)} possibilities")
    
    elif search_mode == "lhs":
        # Check if all parameters are suitable for LHS
        continuous_param_dict = {}
        categorical_param_dict = {}
        
        for key, values in param_dict.items():
            # Check if parameter is a continuous range
            if isinstance(values, list) and len(values) == 2 and all(isinstance(v, (int, float)) for v in values):
                continuous_param_dict[key] = values
            else:
                categorical_param_dict[key] = values
        
        if continuous_param_dict:
            try:
                # Generate Latin Hypercube samples for continuous parameters
                continuous_samples = generate_latin_hypercube_samples(continuous_param_dict, n_samples)
                
                if continuous_samples is not None:
                    # If there are categorical parameters, combine with LHS samples
                    if categorical_param_dict:
                        categorical_combinations = get_parameter_combinations(categorical_param_dict)
                        
                        # Select a subset of categorical combinations if needed
                        if len(categorical_combinations) > n_samples:
                            random.seed(start_seed)
                            categorical_combinations = random.sample(categorical_combinations, n_samples)
                        
                        # Combine continuous and categorical parameters
                        param_combinations = []
                        for i in range(min(len(continuous_samples), len(categorical_combinations))):
                            combined = {**continuous_samples[i], **categorical_combinations[i % len(categorical_combinations)]}
                            param_combinations.append(combined)
                    else:
                        param_combinations = continuous_samples
                    
                    print(f"Generated {len(param_combinations)} parameter combinations using Latin Hypercube Sampling")
                else:
                    # Fall back to random sampling if LHS fails
                    search_mode = "random"
                    all_combinations = get_parameter_combinations(param_dict)
                    param_combinations = random.sample(all_combinations, min(n_samples, len(all_combinations)))
                    print(f"LHS failed, falling back to random sampling with {len(param_combinations)} combinations")
            except ImportError:
                print("Latin Hypercube Sampling requires scipy. Falling back to random sampling.")
                search_mode = "random"
                all_combinations = get_parameter_combinations(param_dict)
                param_combinations = random.sample(all_combinations, min(n_samples, len(all_combinations)))
                print(f"Using random sampling with {len(param_combinations)} combinations")
        else:
            # No continuous parameters for LHS, use random sampling
            search_mode = "random"
            all_combinations = get_parameter_combinations(param_dict)
            param_combinations = random.sample(all_combinations, min(n_samples, len(all_combinations)))
            print(f"No continuous parameters for LHS. Using random sampling with {len(param_combinations)} combinations")
    
    else:
        raise ValueError(f"Unknown search mode: {search_mode}")

    print(f"Will run each combination with {n_seeds} different seeds")
    print(f"Total simulations to run: {len(param_combinations) * n_seeds}")
    
    # Setup output directory and timestamp for saving results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base_name = f"{simulation_type}_search_{timestamp}"
    
    if save_dir:
        # Ensure the save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Create a run-specific subdirectory for all results
        run_dir = os.path.join(save_dir, results_base_name)
        os.makedirs(run_dir, exist_ok=True)
        print(f"Results will be saved in: {run_dir}")
        
        # Save search parameters for reproducibility
        search_config = {
            "simulation_type": simulation_type,
            "param_dict": param_dict,
            "use_asal": use_asal,
            "n_seeds": n_seeds,
            "start_seed": start_seed,
            "n_jobs": n_jobs,
            "timestamp": timestamp
        }
        with open(os.path.join(run_dir, "search_config.json"), 'w') as f:
            json.dump(search_config, f, indent=2, default=str)
        
        # Path for intermediate results (JSONL format for append operation)
        results_file = os.path.join(run_dir, "intermediate_results.jsonl")
        
        # Path for backup results (in case of process interruption)
        backup_file = os.path.join(run_dir, "results_backup.pkl")
    else:
        run_dir = None
        results_file = None
        backup_file = None
    
    # Create a list of all jobs to run
    jobs = []
    for params in param_combinations:
        for i in range(n_seeds):
            seed = start_seed + i
            jobs.append((params, simulation_type, seed))
    
    # Choose simulation function based on available modules
    if use_asal and HAS_ASAL_MODULES:
        sim_func = run_single_simulation_with_asal
        print("Using ASAL framework for simulations")
        
        # Check GPU memory and issue warning if running many simulations
        try:
            import subprocess
            if 'nvidia-smi' in subprocess.getoutput('which nvidia-smi'):
                gpu_info = subprocess.getoutput('nvidia-smi')
                if 'MiB /' in gpu_info:
                    # Extract memory usage
                    for line in gpu_info.split('\n'):
                        if 'MiB /' in line:
                            used_mem = int(line.split('|')[2].split('/')[0].strip().split()[0])
                            total_mem = int(line.split('|')[2].split('/')[1].strip().split()[0])
                            percent_used = (used_mem / total_mem) * 100
                            if percent_used > 70 and len(param_combinations) * n_seeds > 100:
                                print(f"WARNING: GPU memory usage is already at {percent_used:.1f}%. "
                                      f"Running {len(param_combinations) * n_seeds} simulations may cause memory errors.")
                                print("Consider reducing parameter combinations or using fewer seeds.")
        except:
            pass
    else:
        sim_func = run_standard_simulation
        print("Using standard causal emergence framework for simulations")
    
    # Run the jobs
    results = []
    
    if n_jobs == 1:
        # Single process
        for params, sim_type, seed in jobs:
            print(f"Running simulation with parameters: {params}")
            result = sim_func(params, sim_type, seed)
            results.append(result)
            
            # Save intermediate results if a directory is provided
            if save_dir:
                with open(results_file, 'a') as f:
                    f.write(json.dumps(result) + '\n')
    else:
        # Parallel processing
        if n_jobs == -1:
            import multiprocessing as mp
            n_jobs = mp.cpu_count()
            
        print(f"Running on {n_jobs} cores in parallel")
        
        try:
            # Try with joblib first (better progress reporting)
            from joblib import Parallel, delayed
            from tqdm import tqdm
            
            # Use smaller batches to avoid memory accumulation
            batch_size = min(100, len(jobs))
            all_results = []
            
            # Process in batches
            for i in range(0, len(jobs), batch_size):
                batch_jobs = jobs[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(jobs)-1)//batch_size + 1} ({len(batch_jobs)} jobs)")
                
                batch_results = Parallel(n_jobs=n_jobs)(
                    delayed(sim_func)(*job) for job in tqdm(batch_jobs, desc="Running simulations")
                )
                
                all_results.extend(batch_results)
                
                # Clear memory between batches
                if hasattr(jax, 'clear_caches'):
                    jax.clear_caches()
                
                # Save intermediate batch results
                if save_dir:
                    for result in batch_results:
                        with open(results_file, 'a') as f:
                            f.write(json.dumps(result) + '\n')
                            
            results = all_results
        except ImportError:
            # Fall back to multiprocessing with similar batching approach
            import multiprocessing as mp
            from tqdm import tqdm
            
            # Use smaller batches
            batch_size = min(100, len(jobs))
            all_results = []
            
            for i in range(0, len(jobs), batch_size):
                batch_jobs = jobs[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(jobs)-1)//batch_size + 1} ({len(batch_jobs)} jobs)")
                
                with mp.Pool(processes=n_jobs) as pool:
                    batch_results = list(tqdm(
                        pool.starmap(sim_func, batch_jobs),
                        total=len(batch_jobs),
                        desc="Running simulations"
                    ))
                
                all_results.extend(batch_results)
                
                # Clear memory between batches
                if hasattr(jax, 'clear_caches'):
                    jax.clear_caches()
                
                # Save intermediate batch results
                if save_dir:
                    for result in batch_results:
                        with open(results_file, 'a') as f:
                            f.write(json.dumps(result) + '\n')
                            
            results = all_results
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save final results in multiple formats
    if save_dir and run_dir:
        # CSV format (easy to open with spreadsheet software)
        csv_path = os.path.join(run_dir, "search_results.csv")
        df.to_csv(csv_path, index=False)
        
        # Calculate success rate here to ensure it's always defined
        success_rate = 0.0
        if 'success' in df.columns and len(df) > 0:
            success_rate = df['success'].mean() * 100
        
        # Excel format (with formatting)
        try:
            excel_path = os.path.join(run_dir, "search_results.xlsx")
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Search Results', index=False)
                # Create a summary sheet with success rate
                pd.DataFrame({
                    'Total Jobs': [len(df)],
                    'Successful': [df['success'].sum() if 'success' in df.columns else 0],
                    'Failed': [len(df) - df['success'].sum() if 'success' in df.columns else len(df)],
                    'Success Rate (%)': [success_rate]
                }).to_excel(writer, sheet_name='Summary')
        except ImportError:
            # Skip Excel export if openpyxl not available
            print("openpyxl not installed - skipping Excel export")
        
        # Pickle format (preserves all Python objects for later analysis)
        pkl_path = os.path.join(run_dir, "search_results.pkl")
        df.to_pickle(pkl_path)
        
        # Create a summary text file
        with open(os.path.join(run_dir, "results_summary.txt"), 'w') as f:
            f.write(f"Parameter Search Results Summary\n")
            f.write(f"===============================\n\n")
            f.write(f"Simulation Type: {simulation_type}\n")
            f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total parameter combinations: {len(param_combinations)}\n")
            f.write(f"Seeds per combination: {n_seeds}\n")
            f.write(f"Total jobs: {len(jobs)}\n")
            f.write(f"Jobs completed successfully: {df['success'].sum() if 'success' in df.columns else 0}\n")
            f.write(f"Success rate: {success_rate:.2f}%\n\n")
            
            if "Delta" in df.columns:
                f.write("Emergence Metrics Statistics:\n")
                f.write("--------------------------\n")
                for metric in ["Delta", "Gamma", "Psi"]:
                    if metric in df.columns:
                        success_df = df[df['success'] == True]
                        if not success_df.empty:
                            f.write(f"{metric}:\n")
                            f.write(f"  Mean: {success_df[metric].mean():.4f}\n")
                            f.write(f"  Max:  {success_df[metric].max():.4f}\n")
                            f.write(f"  Min:  {success_df[metric].min():.4f}\n")
                            f.write(f"  Std:  {success_df[metric].std():.4f}\n\n")
        
        print(f"Saved final results to {run_dir}/")
        print(f"  - CSV: {csv_path}")
        print(f"  - Pickle: {pkl_path}")
        if 'excel_path' in locals():
            print(f"  - Excel: {excel_path}")
    
    return df

def parse_args():
    parser = argparse.ArgumentParser(description="ASAL-based parameter search for causal emergence")
    
    # Simulation parameters
    parser.add_argument("--simulation_type", type=str, default="gol",
                      choices=["gol", "boids", "lenia"],
                      help="Type of simulation to analyze")
    
    # Search parameters
    parser.add_argument("--search_mode", type=str, default="grid",
                      choices=["grid", "random", "lhs"],
                      help="Search strategy: grid search, random sampling, or Latin Hypercube Sampling")
    parser.add_argument("--n_samples", type=int, default=100,
                      help="Number of random samples if search_mode is 'random' or 'lhs'")
    parser.add_argument("--adaptive_sampling", action="store_true",
                      help="Use adaptive sampling to focus on promising regions")
    parser.add_argument("--n_seeds", type=int, default=3,
                      help="Number of random seeds to try for each parameter combination")
    parser.add_argument("--start_seed", type=int, default=42,
                      help="Starting random seed")
    parser.add_argument("--n_jobs", type=int, default=1,
                      help="Number of parallel jobs (-1 for all cores)")
    
    # Framework choice
    parser.add_argument("--use_asal", action="store_true",
                      help="Use ASAL framework for simulations (default: standard framework)")
    
    # Output options
    parser.add_argument("--save_dir", type=str, default="./parameter_search_results",
                      help="Directory to save results")
    parser.add_argument("--optimal_run", action="store_true",
                      help="Run an optimal simulation with the best parameters")
    parser.add_argument("--optimal_steps", type=int, default=1000,
                      help="Number of steps for the optimal simulation")
    parser.add_argument("--visualize", action="store_true",
                      help="Generate visualizations of the results")
    
    # Parameter customization
    parser.add_argument("--custom_params", type=str, default=None,
                      help="Path to a JSON file with custom parameter ranges")
    parser.add_argument("--max_combinations", type=int, default=None,
                      help="Maximum number of parameter combinations to try")
    parser.add_argument("--param_subset", type=str, default="full",
                      choices=["full", "minimal", "focused"],
                      help="Predefined parameter subset for quicker exploration")
    
    return parser.parse_args()

def main():
    """Main entry point for the parameter search."""
    args = parse_args()
    
    print(f"Starting parameter search for {args.simulation_type} simulation")
    
    # Create save directory with timestamp
    if args.save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a timestamped subdirectory to prevent overwriting previous results
        save_dir = os.path.join(args.save_dir, f"{args.simulation_type}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Results will be saved in: {save_dir}")
    else:
        save_dir = None
    
    # Load parameter ranges
    if args.custom_params:
        try:
            with open(args.custom_params, 'r') as f:
                param_dict = json.load(f)
            print(f"Loaded custom parameter ranges from {args.custom_params}")
        except Exception as e:
            print(f"Error loading custom parameters: {e}")
            print("Using default parameter ranges")
            if args.simulation_type == 'gol':
                param_dict = DEFAULT_GOL_PARAMS
            elif args.simulation_type == 'boids':
                param_dict = DEFAULT_BOIDS_PARAMS
            elif args.simulation_type == 'lenia':
                param_dict = DEFAULT_LENIA_PARAMS
            else:
                raise ValueError(f"No default parameters for {args.simulation_type}")
    else:
        if args.simulation_type == 'gol':
            param_dict = DEFAULT_GOL_PARAMS
        elif args.simulation_type == 'boids':
            param_dict = DEFAULT_BOIDS_PARAMS
        elif args.simulation_type == 'lenia':
            param_dict = DEFAULT_LENIA_PARAMS
        else:
            raise ValueError(f"No default parameters for {args.simulation_type}")
    
    # Apply parameter subset if requested
    if args.param_subset != "full":
        param_dict = create_parameter_subset(param_dict, args.param_subset, args.simulation_type)
        print(f"Using {args.param_subset} parameter subset")
    
    # Limit parameter combinations if requested
    if args.search_mode == 'random':
        n_samples = min(args.n_samples, args.max_combinations) if args.max_combinations else args.n_samples
        param_dict_subset = param_dict
    elif args.max_combinations:
        # Generate all combinations and take a random subset
        all_combinations = get_parameter_combinations(param_dict)
        if len(all_combinations) > args.max_combinations:
            print(f"Limiting to {args.max_combinations} random parameter combinations")
            import random
            random.seed(args.start_seed)
            selected_combinations = random.sample(all_combinations, args.max_combinations)
            
            # Convert back to parameter dictionary format
            param_dict_subset = {}
            for key in param_dict:
                param_dict_subset[key] = list(set(comb[key] for comb in selected_combinations if key in comb))
        else:
            param_dict_subset = param_dict
    else:
        param_dict_subset = param_dict
    
    # Print parameter ranges
    print("\nParameter ranges for search:")
    for param, values in param_dict_subset.items():
        print(f"  {param}: {values}")
    
    # Run parameter search
    start_time = time.time()
    results_df = run_parameter_search(
        args.simulation_type,
        param_dict_subset,
        use_asal=args.use_asal,
        n_seeds=args.n_seeds,
        start_seed=args.start_seed,
        n_jobs=args.n_jobs,
        save_dir=save_dir,
        search_mode=args.search_mode,
        n_samples=args.n_samples,
        adaptive_sampling=args.adaptive_sampling
    )
    end_time = time.time()
    run_time = end_time - start_time
    print(f"\nParameter search completed in {run_time:.2f} seconds ({run_time/60:.2f} minutes)")
    
    # Analyze results with better error handling
    if not results_df.empty:
        try:
            best_dir = None
            if save_dir:
                best_dir = os.path.join(save_dir, "best_params")
                os.makedirs(best_dir, exist_ok=True)
                
            best_individual, best_combined = analyze_search_results(results_df, output_dir=best_dir)
            
            # Save run metrics and timing information
            if save_dir:
                with open(os.path.join(save_dir, "run_info.json"), 'w') as f:
                    run_info = {
                        "simulation_type": args.simulation_type,
                        "run_time_seconds": run_time,
                        "total_combinations": len(results_df) // args.n_seeds if args.n_seeds > 0 else 0,
                        "total_seeds": args.n_seeds,
                        "successful_simulations": int(results_df['success'].sum()),
                        "success_rate": float(results_df['success'].mean()),
                        "completed_at": datetime.now().isoformat(),
                        "command_line_args": vars(args)
                    }
                    json.dump(run_info, f, indent=2)
            
            # Only proceed with visualizations and optimal run if there are results to analyze
            if best_individual and best_combined:
                # Create additional visualizations when using ASAL
                if args.use_asal and HAS_ASAL_MODULES and args.visualize:
                    print("\nCreating ASAL-specific visualizations...")
                    vis_dir = best_dir if best_dir else save_dir
                    create_asal_visualizations(results_df, args.simulation_type, vis_dir)
                
                # Run optimal simulation if requested
                if args.optimal_run:
                    print("\nRunning simulation with optimal parameters...")
                    opt_dir = os.path.join(save_dir, "optimal_run") if save_dir else None
                    if opt_dir:
                        os.makedirs(opt_dir, exist_ok=True)
                        
                    run_optimal_simulation(
                        args.simulation_type,
                        best_combined,
                        n_steps=args.optimal_steps,
                        output_dir=opt_dir
                    )
                    
                    if opt_dir:
                        # Save the optimal parameters in a separate JSON for easy access
                        with open(os.path.join(opt_dir, "optimal_parameters.json"), 'w') as f:
                            json.dump(best_combined, f, indent=2)
            else:
                print("No optimal parameters found. Skipping visualizations and optimal run.")
                
                # Save empty results to indicate no optimal parameters were found
                if save_dir:
                    with open(os.path.join(save_dir, "no_optimal_parameters.txt"), 'w') as f:
                        f.write("No optimal parameters found during analysis.\n")
                        f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        except Exception as e:
            print(f"Error during results analysis: {e}")
            import traceback
            traceback.print_exc()
            
            # Save error information
            if save_dir:
                with open(os.path.join(save_dir, "analysis_error.txt"), 'w') as f:
                    f.write(f"Error during analysis: {str(e)}\n\n")
                    f.write(traceback.format_exc())
    else:
        print("No results to analyze")
        
        # Create a file indicating no results
        if save_dir:
            with open(os.path.join(save_dir, "no_results.txt"), 'w') as f:
                f.write("Parameter search completed but no results were generated.\n")
                f.write(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("\nParameter search analysis complete!")
    if save_dir:
        print(f"All results saved to: {save_dir}")

def create_asal_visualizations(df: pd.DataFrame, simulation_type: str, output_dir: str):
    """Create ASAL-specific visualizations from the parameter search results."""
    # Filter out failed simulations
    df_success = df[df['success'] == True].copy()
    if len(df_success) == 0:
        print("No successful simulations to visualize")
        return
    
    # 1. Plot relationship between open-endedness score and causal emergence metrics
    if 'open_endedness_score' in df_success.columns:
        plt.figure(figsize=(15, 5))
        metrics = ['Delta', 'Gamma', 'Psi']
        
        for i, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, i)
            sns.scatterplot(data=df_success, x='open_endedness_score', y=metric)
            
            # Try to fit a trend line
            if len(df_success) >= 5:  # Need enough points for regression
                try:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        df_success['open_endedness_score'], df_success[metric]
                    )
                    plt.plot(df_success['open_endedness_score'], 
                             intercept + slope * df_success['open_endedness_score'], 
                             'r', label=f'R²={r_value**2:.3f}')
                    plt.legend()
                except:
                    pass
            
            plt.xlabel('Open-endedness Score')
            plt.ylabel(metric)
            plt.title(f'{metric} vs Open-endedness')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{simulation_type}_openendedness_vs_emergence.png"), dpi=300)
        plt.close()
    
    # 2. Create cluster analysis of parameter space using embeddings
    if 'z_embedding' in df_success.columns:
        try:
            # Stack all embeddings
            embeddings = np.array([np.array(e) for e in df_success['z_embedding'].values])
            
            if len(embeddings) >= 5:  # Need enough points for clustering
                # Reduce dimensionality for visualization
                from sklearn.manifold import TSNE
                tsne = TSNE(n_components=2, random_state=42)
                embeddings_2d = tsne.fit_transform(embeddings)
                
                # K-means clustering
                from sklearn.cluster import KMeans
                n_clusters = min(5, len(embeddings))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings)
                clusters = kmeans.labels()
                
                # Create DataFrame for visualization
                viz_df = pd.DataFrame({
                    'x': embeddings_2d[:, 0],
                    'y': embeddings_2d[:, 1],
                    'cluster': clusters,
                    'Delta': df_success['Delta'].values,
                    'open_endedness': df_success['open_endedness_score'].values if 'open_endedness_score' in df_success.columns else 0
                })
                
                # Plot clusters
                plt.figure(figsize=(12, 10))
                plt.subplot(1, 1, 1)
                sns.scatterplot(data=viz_df, x='x', y='y', hue='cluster', size='Delta', 
                                palette='viridis', sizes=(20, 200), legend='brief')
                
                # Add cluster centroids
                centroids = kmeans.cluster_centers_
                centroids_2d = tsne.fit_transform(centroids)
                plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1], s=100, 
                            c='red', marker='X', label='Cluster Centers')
                
                plt.title(f'Parameter Space Clusters for {simulation_type.upper()}')
                plt.xlabel('t-SNE Dimension 1')
                plt.ylabel('t-SNE Dimension 2')
                plt.savefig(os.path.join(output_dir, f"{simulation_type}_parameter_clusters.png"), dpi=300)
                plt.close()
                
                # Analyze cluster properties
                cluster_analysis = df_success.copy()
                cluster_analysis['cluster'] = clusters
                cluster_metrics = cluster_analysis.groupby('cluster')[['Delta', 'Gamma', 'Psi', 
                                                                     'open_endedness_score']].mean().reset_index()
                
                # Save cluster metrics
                cluster_metrics.to_csv(os.path.join(output_dir, f"{simulation_type}_cluster_metrics.csv"))
                
                # Create cluster visualization for metrics
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    cluster_metrics[['Delta', 'Gamma', 'Psi', 'open_endedness_score']], 
                    annot=True, 
                    cmap='viridis',
                    yticklabels=cluster_metrics['cluster']
                )
                plt.title(f'Average Metrics by Cluster for {simulation_type.upper()}')
                plt.ylabel('Cluster')
                plt.savefig(os.path.join(output_dir, f"{simulation_type}_cluster_metrics.png"), dpi=300)
                plt.close()
                
        except Exception as e:
            print(f"Error creating embedding visualizations: {e}")
            import traceback
            traceback.print_exc()

def create_parameter_subset(param_dict, subset_type, simulation_type):
    """Create a subset of parameters for quicker exploration"""
    subset = {}
    
    # Choose which parameters to include based on subset type and simulation
    if subset_type == "minimal":
        # Minimal subset includes only the most impactful parameters
        if simulation_type == "gol":
            subset["grid_size"] = [32]
            subset["init_density"] = [0.1, 0.3, 0.5]
            subset["birth"] = param_dict["birth"][:3]  # First 3 rules
            subset["survive"] = param_dict["survive"][:3]  # First 3 rules
            subset["n_steps"] = [500]
        elif simulation_type == "boids":
            subset["n_boids"] = [50]
            subset["visual_range"] = [5.0, 10.0, 15.0]
            subset["cohesion_factor"] = param_dict["cohesion_factor"]
            subset["alignment_factor"] = param_dict["alignment_factor"]
            subset["separation_factor"] = param_dict["separation_factor"]
            subset["n_steps"] = [500]
        elif simulation_type == "lenia":
            subset["grid_size"] = [64]
            subset["time_step"] = [0.1, 0.2, 0.3]
            subset["kernel_radius"] = [10, 15, 20]
            subset["growth_center"] = param_dict["growth_center"]
            subset["growth_width"] = param_dict["growth_width"]
            subset["n_steps"] = [500]
    
    elif subset_type == "focused":
        # Focused subset has fewer values but keeps all parameters
        if simulation_type == "gol":
            subset["grid_size"] = [32, 64]
            subset["init_density"] = [0.1, 0.3, 0.5]
            subset["birth"] = param_dict["birth"][::2]  # Every other rule
            subset["survive"] = param_dict["survive"][::2]  # Every other rule
            subset["n_steps"] = [500, 1000]
        elif simulation_type == "boids":
            for key in param_dict:
                # Take every other value for most parameters
                if key != "n_steps":
                    subset[key] = param_dict[key][::2]
                else:
                    subset["n_steps"] = [500, 1000]
        elif simulation_type == "lenia":
            for key in param_dict:
                # Take every other value for most parameters
                if key != "n_steps" and key != "init_pattern":
                    subset[key] = param_dict[key][::2]
                elif key == "init_pattern":
                    subset[key] = param_dict[key]
                else:
                    subset["n_steps"] = [500, 1000]
    
    # If subset is empty (unknown simulation type or subset), return original
    if not subset:
        return param_dict
        
    return subset

if __name__ == "__main__":
    main()
