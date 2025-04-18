#!/usr/bin/env python
"""
Run Information Theory Experiments

This script runs experiments to analyze information-theoretic properties
of different simulation types from ASAL and creates visualizations of the results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import argparse
from tqdm import tqdm

# Import local modules
import info_theory_utils as it
from causal_emergence import run_simulation, calculate_emergence_metrics
from emergence_visualization import create_dashboard, plot_metrics_radar

# Configure plot style - Updated to work with newer matplotlib/seaborn versions
try:
    plt.style.use('seaborn-v0_8-whitegrid')  # For newer versions of matplotlib
except:
    try:
        plt.style.use('seaborn-whitegrid')  # For older versions
    except:
        plt.style.use('seaborn')  # Fallback to basic seaborn style
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.7

sns.set_context("paper", font_scale=1.2)

def parse_args():
    parser = argparse.ArgumentParser(description="Run information theory experiments and visualizations")
    
    # Simulation parameters
    parser.add_argument("--simulation_types", nargs='+', default=["gol", "boids"],
                      help="List of simulation types to analyze")
    parser.add_argument("--n_steps", type=int, default=500,
                      help="Number of simulation steps")
    parser.add_argument("--n_runs", type=int, default=5,
                      help="Number of runs per simulation type")
    parser.add_argument("--random_seed", type=int, default=42,
                      help="Base random seed")
    
    # Analysis parameters
    parser.add_argument("--discretize", action="store_true", default=True,
                      help="Discretize data before computing information metrics")
    parser.add_argument("--n_bins", type=int, default=10,
                      help="Number of bins for discretization")
    parser.add_argument("--use_reconciling", action="store_true", default=True,
                      help="Include reconciling emergence metrics")
    
    # Visualization parameters
    parser.add_argument("--save_dir", type=str, default="./info_theory_results",
                      help="Directory to save results and visualizations")
    parser.add_argument("--show_plots", action="store_true", default=True,
                      help="Show plots interactively")
    
    return parser.parse_args()

def run_experiments(args):
    """Run experiments for each simulation type and collect metrics"""
    # Create results directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Initialize results storage
    results = {}
    
    # Run experiments for each simulation type
    for sim_type in args.simulation_types:
        print(f"\n*** Running experiments for {sim_type.upper()} ***")
        
        sim_results = []
        
        for run in range(args.n_runs):
            print(f"\n--- Run {run+1}/{args.n_runs} ---")
            
            # Set random seed for this run
            seed = args.random_seed + run
            
            # Run simulation
            print(f"Running {sim_type} simulation with {args.n_steps} steps...")
            S_history, M_history = run_simulation(sim_type, args.n_steps, seed=seed)
            
            if S_history is None or M_history is None:
                print(f"Error: Simulation failed for {sim_type}, run {run+1}")
                continue
                
            print(f"Generated data shapes: S_history={S_history.shape}, M_history={M_history.shape}")
            
            # Calculate metrics
            metrics = calculate_emergence_metrics(
                S_history, M_history, 
                discretize=args.discretize,
                n_bins=args.n_bins
            )
            
            if metrics:
                print("\n--- Emergence Metrics Results ---")
                for key, value in metrics.items():
                    print(f" {key}: {value:.4f}")
                
                # Save S_history and M_history with the metrics
                run_data = {
                    'metrics': metrics,
                    'S_history': S_history,
                    'M_history': M_history,
                    'seed': seed,
                    'n_steps': args.n_steps
                }
                
                sim_results.append(run_data)
            else:
                print(f"Error calculating metrics for {sim_type}, run {run+1}")
        
        # Store results for this simulation type
        results[sim_type] = sim_results
        
        # Save intermediate results
        np.save(os.path.join(args.save_dir, f"{sim_type}_results.npy"), sim_results)
    
    return results

def plot_metrics_comparison(results, args):
    """Create a comparison plot of metrics across simulation types"""
    sim_types = list(results.keys())
    
    # Collect metrics
    delta_values = {st: [] for st in sim_types}
    gamma_values = {st: [] for st in sim_types}
    psi_values = {st: [] for st in sim_types}
    
    delta_r_values = {st: [] for st in sim_types}
    gamma_r_values = {st: [] for st in sim_types}
    psi_r_values = {st: [] for st in sim_types}
    
    # Process results
    for sim_type, sim_results in results.items():
        for run_data in sim_results:
            metrics = run_data['metrics']
            
            # Standard metrics
            delta_values[sim_type].append(metrics.get('Delta', 0))
            gamma_values[sim_type].append(metrics.get('Gamma', 0))
            psi_values[sim_type].append(metrics.get('Psi', 0))
            
            # Reconciling metrics if available
            if 'Delta_reconciling' in metrics:
                delta_r_values[sim_type].append(metrics.get('Delta_reconciling', 0))
                gamma_r_values[sim_type].append(metrics.get('Gamma_reconciling', 0))
                psi_r_values[sim_type].append(metrics.get('Psi_reconciling', 0))
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # --- Metrics Comparison ---
    metric_names = ['Delta', 'Gamma', 'Psi']
    
    # Plot standard metrics
    ax1 = fig.add_subplot(gs[0, :])
    
    # Set width of bars and positions
    bar_width = 0.2
    index = np.arange(len(sim_types))
    
    # Create bars
    for i, metric_name in enumerate(metric_names):
        values = eval(f"{metric_name.lower()}_values")
        means = [np.mean(values[st]) for st in sim_types]
        errors = [np.std(values[st]) / np.sqrt(len(values[st])) for st in sim_types]
        
        offset = (i - 1) * bar_width  # Center the bars
        ax1.bar(index + offset, means, bar_width, label=metric_name,
               yerr=errors, capsize=5)
    
    # Customize plot
    ax1.set_xlabel('Simulation Type')
    ax1.set_ylabel('Metric Value')
    ax1.set_title('Causal Emergence Metrics Comparison')
    ax1.set_xticks(index)
    ax1.set_xticklabels([st.upper() for st in sim_types])
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # --- Plot Radar Charts for each simulation type ---
    has_reconciling = all('Delta_reconciling' in sim_results[0]['metrics'] for sim_results in results.values() if sim_results)
    
    for i, sim_type in enumerate(sim_types):
        if not results[sim_type]:
            continue
            
        # Average metrics across runs
        avg_metrics = {}
        for metric in ['Delta', 'Gamma', 'Psi']:
            avg_metrics[metric] = np.mean(eval(f"{metric.lower()}_values")[sim_type])
            
            if has_reconciling:
                r_metric = f"{metric}_reconciling"
                avg_metrics[r_metric] = np.mean(eval(f"{metric.lower()}_r_values")[sim_type])
        
        # Plot radar chart
        ax = fig.add_subplot(gs[1, i], polar=True)
        plot_metrics_radar(avg_metrics, ax=ax)
        ax.set_title(f"{sim_type.upper()}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "metrics_comparison.png"), dpi=300)
    
    if args.show_plots:
        plt.show()
    else:
        plt.close()

def plot_time_series_analysis(results, args):
    """Analyze and plot how metrics change with time series length"""
    sim_types = list(results.keys())
    
    # We'll analyze different lengths of the time series
    step_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    for sim_idx, sim_type in enumerate(sim_types):
        if not results[sim_type]:
            continue
            
        # Choose the first run for this analysis
        run_data = results[sim_type][0]
        full_S_history = run_data['S_history']
        full_M_history = run_data['M_history']
        n_steps = full_S_history.shape[0]
        
        # Calculate metrics for different history lengths
        delta_values = []
        gamma_values = []
        psi_values = []
        step_counts = []
        
        for frac in step_fractions:
            steps = max(10, int(frac * n_steps))  # Ensure at least 10 steps
            step_counts.append(steps)
            
            # Truncate histories
            S_history = full_S_history[:steps]
            M_history = full_M_history[:steps]
            
            # Calculate metrics
            metrics = calculate_emergence_metrics(
                S_history, M_history, 
                discretize=args.discretize,
                n_bins=args.n_bins
            )
            
            if metrics:
                delta_values.append(metrics.get('Delta', 0))
                gamma_values.append(metrics.get('Gamma', 0))
                psi_values.append(metrics.get('Psi', 0))
            else:
                delta_values.append(0)
                gamma_values.append(0)
                psi_values.append(0)
        
        # Plot metrics vs time series length
        ax = fig.add_subplot(1, len(sim_types), sim_idx + 1)
        ax.plot(step_counts, delta_values, 'o-', label='Delta')
        ax.plot(step_counts, gamma_values, 's-', label='Gamma')
        ax.plot(step_counts, psi_values, '^-', label='Psi')
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Metric Value')
        ax.set_title(f"{sim_type.upper()} Metrics vs Time Series Length")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "time_series_analysis.png"), dpi=300)
    
    if args.show_plots:
        plt.show()
    else:
        plt.close()

def create_dashboards(results, args):
    """Create detailed dashboards for each simulation run"""
    for sim_type, sim_results in results.items():
        for run_idx, run_data in enumerate(sim_results):
            metrics = run_data['metrics']
            S_history = run_data['S_history']
            M_history = run_data['M_history']
            
            print(f"Creating dashboard for {sim_type.upper()} - Run {run_idx+1}")
            
            # Create dashboard
            fig = create_dashboard(
                metrics=metrics,
                S_history=S_history,
                M_history=M_history,
                simulation_type=sim_type,
                show_pca=True
            )
            
            # Save dashboard
            save_path = os.path.join(args.save_dir, f"{sim_type}_run{run_idx+1}_dashboard.png")
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
            if args.show_plots:
                plt.show()
            else:
                plt.close(fig)

def main():
    """Main function to run experiments and create visualizations"""
    args = parse_args()
    
    # Run experiments or load existing results
    all_files_exist = True
    for sim_type in args.simulation_types:
        file_path = os.path.join(args.save_dir, f"{sim_type}_results.npy")
        if not os.path.exists(file_path):
            all_files_exist = False
            break
    
    if all_files_exist:
        # Load existing results
        print("Loading existing experiment results...")
        results = {}
        for sim_type in args.simulation_types:
            file_path = os.path.join(args.save_dir, f"{sim_type}_results.npy")
            results[sim_type] = np.load(file_path, allow_pickle=True).tolist()
    else:
        # Run new experiments
        print("Running new experiments...")
        results = run_experiments(args)
    
    # Create visualizations
    print("\n--- Creating Visualizations ---")
    
    print("1. Metrics Comparison Plot")
    plot_metrics_comparison(results, args)
    
    print("2. Time Series Analysis Plot")
    plot_time_series_analysis(results, args)
    
    print("3. Detailed Dashboards")
    create_dashboards(results, args)
    
    print(f"\nAll visualizations saved to {args.save_dir}")

if __name__ == "__main__":
    main()
