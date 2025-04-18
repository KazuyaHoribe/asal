#!/usr/bin/env python
"""
Parameter Search for Causal Emergence Analysis

This script performs a systematic search for initial conditions and parameters
that result in high causal emergence metrics (Delta, Gamma, Psi).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
import time
import argparse
import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import json
from tqdm import tqdm

# Check if needed modules are available
if not os.path.exists('causal_emergence.py'):
    print("Error: This script should be run from the directory containing 'causal_emergence.py'")
    sys.exit(1)

# Import local modules
from causal_emergence import run_simulation, calculate_emergence_metrics

# --- Parameter Search Configuration ---

# Default parameter ranges to explore
DEFAULT_GOL_PARAMS = {
    "grid_size": [16, 24, 32],
    "init_density": [0.1, 0.2, 0.3, 0.4, 0.5],
    "n_steps": [500]
}

DEFAULT_BOIDS_PARAMS = {
    "n_boids": [30, 50, 70],
    "visual_range": [5.0, 10.0, 15.0],
    "protected_range": [1.0, 2.0, 3.0],
    "max_speed": [3.0, 5.0, 7.0],
    "cohesion_factor": [0.005, 0.01, 0.02],
    "alignment_factor": [0.05, 0.125, 0.2],
    "separation_factor": [0.025, 0.05, 0.1],
    "n_steps": [500]
}

# --- Utility Functions ---

def get_parameter_combinations(param_dict: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters from a dictionary of parameter lists."""
    param_names = param_dict.keys()
    param_values = param_dict.values()
    param_combinations = list(itertools.product(*param_values))
    
    return [dict(zip(param_names, combination)) for combination in param_combinations]

def run_single_simulation(params: Dict[str, Any], simulation_type: str, seed: int) -> Dict[str, Any]:
    """Run a single simulation with the given parameters and return the results."""
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
    n_seeds: int = 3,
    start_seed: int = 42,
    n_jobs: int = 1,
    save_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Run a parameter search with multiple combinations of parameters.
    
    Args:
        simulation_type: Type of simulation ('gol' or 'boids')
        param_dict: Dictionary mapping parameter names to lists of values
        n_seeds: Number of random seeds to try for each parameter combination
        start_seed: Starting random seed
        n_jobs: Number of parallel jobs to run (use -1 for all cores)
        save_dir: Directory to save intermediate results
        
    Returns:
        DataFrame containing results for all parameter combinations
    """
    # Generate all parameter combinations
    param_combinations = get_parameter_combinations(param_dict)
    print(f"Generated {len(param_combinations)} parameter combinations")
    print(f"Will run each combination with {n_seeds} different seeds")
    print(f"Total simulations to run: {len(param_combinations) * n_seeds}")
    
    # Create a list of all jobs to run
    jobs = []
    for params in param_combinations:
        for i in range(n_seeds):
            seed = start_seed + i
            jobs.append((params, simulation_type, seed))
    
    # Setup output directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(save_dir, f"{simulation_type}_search_{timestamp}.jsonl")
    
    # Run the jobs
    results = []
    
    if n_jobs == 1:
        # Single process
        for params, sim_type, seed in tqdm(jobs, desc="Running simulations"):
            result = run_single_simulation(params, sim_type, seed)
            results.append(result)
            
            # Save intermediate results if a directory is provided
            if save_dir:
                with open(results_file, 'a') as f:
                    f.write(json.dumps(result) + '\n')
    else:
        # Parallel processing
        if n_jobs == -1:
            n_jobs = mp.cpu_count()
            
        print(f"Running on {n_jobs} cores in parallel")
        with mp.Pool(processes=n_jobs) as pool:
            # Create a generator for results to process them as they complete
            for result in tqdm(
                pool.starmap(run_single_simulation, jobs),
                total=len(jobs),
                desc="Running simulations"
            ):
                results.append(result)
                
                # Save intermediate results if a directory is provided
                if save_dir:
                    with open(results_file, 'a') as f:
                        f.write(json.dumps(result) + '\n')
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save final results
    if save_dir:
        df.to_csv(os.path.join(save_dir, f"{simulation_type}_search_results_{timestamp}.csv"))
        print(f"Saved results to {os.path.join(save_dir, f'{simulation_type}_search_results_{timestamp}.csv')}")
    
    return df

def analyze_search_results(df: pd.DataFrame, output_dir: Optional[str] = None) -> None:
    """
    Analyze parameter search results and visualize the findings.
    
    Args:
        df: DataFrame containing parameter search results
        output_dir: Directory to save figures
    """
    # Filter out failed simulations
    df_success = df[df['success'] == True].copy()
    if len(df_success) == 0:
        print("No successful simulations found.")
        return
    
    print(f"Analyzing {len(df_success)} successful simulations out of {len(df)} total runs")
    
    # Compute the average metrics for each parameter combination
    metrics = ['Delta', 'Gamma', 'Psi']
    
    # Convert parameter columns to strings for better grouping
    param_cols = [col for col in df_success.columns if col in df_success['params'].iloc[0]]
    for col in param_cols:
        df_success[col] = df_success['params'].apply(lambda x: x.get(col, None))
    
    # Group by parameter combination and compute mean metrics
    grouped = df_success.groupby(param_cols)[metrics].mean().reset_index()
    
    # Find the best parameter combinations for each metric
    best_params = {}
    for metric in metrics:
        best_idx = grouped[metric].idxmax()
        best_params[metric] = {
            'value': grouped.loc[best_idx, metric],
            'params': {col: grouped.loc[best_idx, col] for col in param_cols}
        }
    
    # Print the best parameters for each metric
    print("\n=== Best Parameter Combinations ===")
    for metric in metrics:
        print(f"\nBest parameters for {metric} (value: {best_params[metric]['value']:.4f}):")
        for param, value in best_params[metric]['params'].items():
            print(f"  {param}: {value}")
    
    # Create visualizations
    
    # 1. Correlation between metrics
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_success[metrics].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation between Emergence Metrics", fontsize=14)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "metrics_correlation.png"), dpi=300)
    
    # 2. Distribution of metrics
    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)
        sns.histplot(df_success[metric], kde=True)
        plt.title(f"Distribution of {metric}", fontsize=12)
        plt.xlabel(metric, fontsize=10)
        plt.ylabel("Frequency", fontsize=10)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "metrics_distribution.png"), dpi=300)
    
    # 3. Parameter influence on metrics
    
    # Identify which parameters have multiple values
    variable_params = [col for col in param_cols if len(grouped[col].unique()) > 1]
    
    if len(variable_params) > 0:
        for metric in metrics:
            plt.figure(figsize=(5 * min(len(variable_params), 3), 4 * ((len(variable_params) + 2) // 3)))
            for i, param in enumerate(variable_params, 1):
                plt.subplot((len(variable_params) + 2) // 3, min(len(variable_params), 3), i)
                if df_success[param].dtype in [np.float64, np.int64]:
                    # For numerical parameters
                    sns.scatterplot(x=param, y=metric, data=df_success)
                    plt.title(f"{metric} vs {param}", fontsize=12)
                else:
                    # For categorical parameters
                    sns.boxplot(x=param, y=metric, data=df_success)
                    plt.title(f"{metric} by {param}", fontsize=12)
                plt.xticks(rotation=45 if len(grouped[param].unique()) > 4 else 0)
            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir, f"parameter_influence_{metric}.png"), dpi=300)
    
    # 4. Combined metrics score
    # Calculate a combined score as the mean of all metrics
    df_success['Combined'] = df_success[metrics].mean(axis=1)
    
    # Find the best parameter combination for the combined score
    best_combined_idx = grouped['Combined'].idxmax()
    best_combined_params = {col: grouped.loc[best_combined_idx, col] for col in param_cols}
    best_combined_value = grouped.loc[best_combined_idx, 'Combined']
    
    print("\nBest parameters for combined metrics (value: {:.4f}):".format(best_combined_value))
    for param, value in best_combined_params.items():
        print(f"  {param}: {value}")
    
    # 5. Parameter pair interactions (for the top 2 most variable parameters)
    if len(variable_params) >= 2:
        # Choose the 2 parameters with the most influence on metrics
        param_influence = {}
        for param in variable_params:
            # Calculate the variance of mean metrics across parameter values
            param_influence[param] = grouped.groupby(param)[metrics].mean().var().mean()
        
        top_params = sorted(param_influence.items(), key=lambda x: x[1], reverse=True)[:2]
        top_param_names = [p[0] for p in top_params]
        
        for metric in metrics:
            plt.figure(figsize=(10, 8))
            pivot = pd.pivot_table(
                grouped, 
                values=metric, 
                index=top_param_names[0], 
                columns=top_param_names[1],
                aggfunc=np.mean
            )
            sns.heatmap(pivot, cmap="viridis", annot=True, fmt=".3f")
            plt.title(f"{metric} by {top_param_names[0]} and {top_param_names[1]}", fontsize=14)
            plt.tight_layout()
            if output_dir:
                plt.savefig(os.path.join(output_dir, f"parameter_interaction_{metric}.png"), dpi=300)
    
    if output_dir:
        # Save a summary of the best parameters
        with open(os.path.join(output_dir, "best_parameters.json"), "w") as f:
            json.dump({
                "best_individual": best_params,
                "best_combined": {
                    "value": float(best_combined_value),
                    "params": best_combined_params
                }
            }, f, indent=2)
    
    return best_params, best_combined_params

def run_optimal_simulation(
    simulation_type: str,
    best_params: Dict[str, Any],
    n_steps: int = 1000,
    output_dir: Optional[str] = None
) -> None:
    """
    Run a simulation with the optimal parameters and visualize the results.
    
    Args:
        simulation_type: Type of simulation ('gol' or 'boids')
        best_params: Dictionary of best parameters
        n_steps: Number of simulation steps
        output_dir: Directory to save figures and data
    """
    print(f"\nRunning optimal {simulation_type} simulation with best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    try:
        # Set up the simulation parameters
        if simulation_type == 'gol':
            # Update Game of Life parameters
            import causal_emergence as ce
            ce.GRID_SIZE = best_params.get('grid_size', 32)
            ce.RANDOM_DENSITY = best_params.get('init_density', 0.1)
        elif simulation_type == 'boids':
            # Update Boids parameters
            import causal_emergence as ce
            ce.N_BOIDS = best_params.get('n_boids', 50)
            ce.VISUAL_RANGE = best_params.get('visual_range', 10.0)
            ce.PROTECTED_RANGE = best_params.get('protected_range', 2.0)
            ce.MAX_SPEED = best_params.get('max_speed', 5.0)
            ce.COHESION_FACTOR = best_params.get('cohesion_factor', 0.01)
            ce.ALIGNMENT_FACTOR = best_params.get('alignment_factor', 0.125)
            ce.SEPARATION_FACTOR = best_params.get('separation_factor', 0.05)
        
        # Run the simulation
        S_history, M_history, visual_frames = run_simulation(simulation_type, n_steps, seed=42)
        
        if S_history is None or M_history is None:
            raise RuntimeError("Optimal simulation failed")
        
        # Calculate metrics
        needs_discretization = (simulation_type == 'boids')
        metrics = calculate_emergence_metrics(S_history, M_history, discretize=needs_discretization)
        
        if metrics is None:
            raise RuntimeError("Failed to calculate emergence metrics")
        
        print("\nEmergence metrics from optimal simulation:")
        print(f"Delta: {metrics['Delta']:.4f}")
        print(f"Gamma: {metrics['Gamma']:.4f}")
        print(f"Psi: {metrics['Psi']:.4f}")
        
        if output_dir:
            # Save the simulation results
            save_path = os.path.join(output_dir, f"{simulation_type}_optimal_simulation.npz")
            np.savez_compressed(
                save_path,
                S_history=S_history,
                M_history=M_history,
                visual_frames=visual_frames,
                metrics=metrics
            )
            print(f"Saved optimal simulation data to {save_path}")
            
            # Create visualizations using existing functions
            from emergence_visualization import visualize_simulation_states, create_dashboard
            
            # Visualize simulation states
            fig = visualize_simulation_states(visual_frames, simulation_type, metrics)
            fig.savefig(os.path.join(output_dir, f"{simulation_type}_optimal_states.png"), dpi=300)
            plt.close(fig)
            
            # Create dashboard
            dashboard = create_dashboard(metrics, S_history, M_history, simulation_type)
            dashboard.savefig(os.path.join(output_dir, f"{simulation_type}_optimal_dashboard.png"), dpi=300)
            plt.close(dashboard)
            
            print(f"Saved visualization of optimal simulation to {output_dir}")
    
    except Exception as e:
        print(f"Error running optimal simulation: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Parameter search for causal emergence")
    
    # Simulation parameters
    parser.add_argument("--simulation_type", type=str, default="gol",
                      choices=["gol", "boids"],
                      help="Type of simulation to analyze")
    
    # Search parameters
    parser.add_argument("--n_seeds", type=int, default=3,
                      help="Number of random seeds to try for each parameter combination")
    parser.add_argument("--start_seed", type=int, default=42,
                      help="Starting random seed")
    parser.add_argument("--n_jobs", type=int, default=1,
                      help="Number of parallel jobs (-1 for all cores)")
    
    # Output options
    parser.add_argument("--save_dir", type=str, default="./parameter_search_results",
                      help="Directory to save results")
    parser.add_argument("--optimal_run", action="store_true",
                      help="Run an optimal simulation with the best parameters")
    parser.add_argument("--optimal_steps", type=int, default=1000,
                      help="Number of steps for the optimal simulation")
    
    # Parameter customization
    parser.add_argument("--custom_params", type=str, default=None,
                      help="Path to a JSON file with custom parameter ranges")
    
    return parser.parse_args()

def main():
    """Main entry point for the parameter search."""
    args = parse_args()
    
    print(f"Starting parameter search for {args.simulation_type} simulation")
    
    # Create save directory if it doesn't exist
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Load parameter ranges
    if args.custom_params:
        try:
            with open(args.custom_params, 'r') as f:
                param_dict = json.load(f)
            print(f"Loaded custom parameter ranges from {args.custom_params}")
        except Exception as e:
            print(f"Error loading custom parameters: {e}")
            print("Using default parameter ranges")
            param_dict = DEFAULT_GOL_PARAMS if args.simulation_type == 'gol' else DEFAULT_BOIDS_PARAMS
    else:
        param_dict = DEFAULT_GOL_PARAMS if args.simulation_type == 'gol' else DEFAULT_BOIDS_PARAMS
    
    # Print parameter ranges
    print("\nParameter ranges for search:")
    for param, values in param_dict.items():
        print(f"  {param}: {values}")
    
    # Run parameter search
    start_time = time.time()
    results_df = run_parameter_search(
        args.simulation_type,
        param_dict,
        n_seeds=args.n_seeds,
        start_seed=args.start_seed,
        n_jobs=args.n_jobs,
        save_dir=args.save_dir
    )
    end_time = time.time()
    print(f"\nParameter search completed in {end_time - start_time:.2f} seconds")
    
    # Analyze results
    best_individual, best_combined = analyze_search_results(results_df, output_dir=args.save_dir)
    
    # Run optimal simulation if requested
    if args.optimal_run:
        print("\nRunning simulation with optimal parameters...")
        run_optimal_simulation(
            args.simulation_type,
            best_combined,
            n_steps=args.optimal_steps,
            output_dir=args.save_dir
        )
    
    print("\nParameter search analysis complete!")

if __name__ == "__main__":
    main()
