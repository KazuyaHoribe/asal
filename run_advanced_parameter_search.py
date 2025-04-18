#!/usr/bin/env python
"""
Advanced Parameter Search for ASAL Causal Emergence Analysis

This script implements more sophisticated parameter search strategies:
1. Multi-stage search (coarse-to-fine)
2. Bayesian optimization
3. Genetic algorithm search
4. Surrogate model-based optimization

Use this script when exploring large parameter spaces efficiently.
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
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure the ASAL directory is in the path
asal_path = '.'
if asal_path not in sys.path:
    sys.path.insert(0, asal_path)

# Import from standard parameter search
from asal_parameter_search import (
    run_single_simulation_with_asal,
    run_standard_simulation,
    DEFAULT_GOL_PARAMS,
    DEFAULT_BOIDS_PARAMS,
    DEFAULT_LENIA_PARAMS
)

def parse_args():
    parser = argparse.ArgumentParser(description="Advanced parameter search strategies for causal emergence")
    
    # Simulation parameters
    parser.add_argument("--simulation_type", type=str, default="gol",
                      choices=["gol", "boids", "lenia"],
                      help="Type of simulation to analyze")
    
    # Search method
    parser.add_argument("--search_method", type=str, default="bayesian",
                      choices=["bayesian", "genetic", "multi_stage", "surrogate"],
                      help="Advanced search method to use")
    
    # Common parameters
    parser.add_argument("--n_iterations", type=int, default=50,
                      help="Number of optimization iterations")
    parser.add_argument("--n_initial", type=int, default=20,
                      help="Number of initial random samples")
    parser.add_argument("--n_seeds", type=int, default=3,
                      help="Number of random seeds for each parameter set")
    parser.add_argument("--metric", type=str, default="combined",
                      choices=["Delta", "Gamma", "Psi", "combined"],
                      help="Which metric to optimize")
    
    # Framework choice
    parser.add_argument("--use_asal", action="store_true",
                      help="Use ASAL framework for simulations")
    
    # Output options
    parser.add_argument("--save_dir", type=str, default="./advanced_search_results",
                      help="Directory to save results")
    parser.add_argument("--visualize", action="store_true",
                      help="Generate visualizations of the search progress")
    parser.add_argument("--verbose", action="store_true",
                      help="Show detailed progress")
    
    # Parallel processing
    parser.add_argument("--n_jobs", type=int, default=1,
                      help="Number of parallel jobs (-1 for all cores)")
    
    return parser.parse_args()

def run_bayesian_optimization(simulation_type, n_iterations, n_initial, metric="combined", 
                            use_asal=True, n_seeds=3, n_jobs=1, save_dir=None, verbose=False):
    """
    Run Bayesian optimization for parameter search.
    
    Args:
        simulation_type: Type of simulation to run
        n_iterations: Number of optimization iterations
        n_initial: Number of initial random evaluations
        metric: Which metric to optimize
        use_asal: Whether to use ASAL framework
        n_seeds: Number of seeds for each parameter set
        n_jobs: Number of parallel jobs
        save_dir: Directory to save results
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with optimization results
    """
    try:
        from skopt import gp_minimize, dump
        from skopt.space import Real, Integer, Categorical
        from skopt.utils import use_named_args
    except ImportError:
        print("Bayesian optimization requires scikit-optimize. Install with:")
        print("pip install scikit-optimize")
        return None
    
    print("Setting up Bayesian optimization...")
    
    # Define parameter space based on simulation type
    if simulation_type == "gol":
        search_space = [
            Integer(16, 128, name='grid_size'),
            Real(0.01, 0.6, name='init_density'),
            Categorical([[3], [2, 3], [3, 6], [1, 3, 5, 7]], name='birth_rule'),
            Categorical([[2, 3], [0, 2, 4, 6, 8], [1, 2, 3, 4, 5]], name='survive_rule'),
            Integer(250, 2000, name='n_steps')
        ]
    elif simulation_type == "boids":
        search_space = [
            Integer(20, 300, name='n_boids'),
            Real(3.0, 30.0, name='visual_range'),
            Real(0.5, 5.0, name='protected_range'),
            Real(1.0, 15.0, name='max_speed'),
            Real(0.001, 0.2, name='cohesion_factor'),
            Real(0.01, 0.5, name='alignment_factor'),
            Real(0.01, 0.5, name='separation_factor'),
            Integer(250, 2000, name='n_steps')
        ]
    elif simulation_type == "lenia":
        search_space = [
            Integer(64, 256, name='grid_size'),
            Real(0.05, 0.4, name='time_step'),
            Integer(5, 30, name='kernel_radius'),
            Real(0.05, 0.4, name='growth_center'),
            Real(0.005, 0.2, name='growth_width'),
            Integer(5, 40, name='init_radius'),
            Integer(250, 2000, name='n_steps')
        ]
    else:
        raise ValueError(f"Unknown simulation type: {simulation_type}")
    
    # Define the objective function to minimize (negative of the metric we want to maximize)
    @use_named_args(search_space)
    def objective(**params):
        if verbose:
            print(f"Evaluating parameters: {params}")
        
        # Convert parameters to the format expected by simulation functions
        sim_params = params.copy()
        
        # Handle special cases
        if 'birth_rule' in sim_params:
            sim_params['birth'] = sim_params.pop('birth_rule')
        if 'survive_rule' in sim_params:
            sim_params['survive'] = sim_params.pop('survive_rule')
        
        # Run simulations with multiple seeds
        results = []
        for seed in range(n_seeds):
            if use_asal:
                result = run_single_simulation_with_asal(sim_params, simulation_type, seed)
            else:
                result = run_standard_simulation(sim_params, simulation_type, seed)
                
            if result.get("success", False):
                results.append(result)
        
        # No successful simulations
        if not results:
            return 0.0  # Return a default bad value
        
        # Calculate score based on the chosen metric
        if metric == "Delta":
            score = np.mean([r.get("Delta", 0.0) for r in results])
        elif metric == "Gamma":
            score = np.mean([r.get("Gamma", 0.0) for r in results])
        elif metric == "Psi":
            score = np.mean([r.get("Psi", 0.0) for r in results])
        else:  # combined
            score = np.mean([
                (r.get("Delta", 0.0) + r.get("Gamma", 0.0) + r.get("Psi", 0.0)) / 3.0
                for r in results
            ])
        
        # Save intermediate result
        if save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(save_dir, f"bo_result_{timestamp}.json")
            with open(result_path, 'w') as f:
                json.dump({
                    "params": sim_params,
                    "metric": metric,
                    "score": score,
                    "results": results
                }, f)
        
        if verbose:
            print(f"Score: {score}")
        
        # Negate score because we want to maximize, but optimizer minimizes
        return -score
    
    # Run Bayesian optimization
    print(f"Starting Bayesian optimization with {n_iterations} iterations...")
    result = gp_minimize(
        objective,
        search_space,
        n_calls=n_iterations,
        n_initial_points=n_initial,
        n_jobs=1,  # Must be 1 as our objective already handles parallelism
        verbose=verbose,
        random_state=42
    )
    
    # Extract the best parameters
    best_params = {}
    for i, param_name in enumerate([dim.name for dim in search_space]):
        best_params[param_name] = result.x[i]
    
    # Handle special cases
    if 'birth_rule' in best_params:
        best_params['birth'] = best_params.pop('birth_rule')
    if 'survive_rule' in best_params:
        best_params['survive'] = best_params.pop('survive_rule')
    
    # Save final result
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save skopt result object
        dump(result, os.path.join(save_dir, "bayesian_opt_result.pkl"))
        
        # Save best parameters
        with open(os.path.join(save_dir, "best_parameters.json"), 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Create convergence plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(result.func_vals)), -np.array(result.func_vals), 'o-')
        plt.xlabel('Iteration')
        plt.ylabel(f'{metric} Score')
        plt.title('Bayesian Optimization Convergence')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "convergence.png"), dpi=300)
        
    print(f"Bayesian optimization complete. Best {metric} score: {-result.fun}")
    print(f"Best parameters: {best_params}")
    
    return {
        "best_params": best_params,
        "best_score": -result.fun,
        "all_scores": -np.array(result.func_vals),
        "all_params": result.x_iters
    }

def run_genetic_algorithm_search(simulation_type, n_iterations, population_size=20,
                               metric="combined", use_asal=True, n_seeds=3, 
                               n_jobs=1, save_dir=None, verbose=False):
    """
    Run genetic algorithm for parameter search.
    
    Args:
        simulation_type: Type of simulation to run
        n_iterations: Number of generations
        population_size: Size of the population in each generation
        metric: Which metric to optimize
        use_asal: Whether to use ASAL framework
        n_seeds: Number of seeds for each parameter set
        n_jobs: Number of parallel jobs
        save_dir: Directory to save results
        verbose: Whether to print detailed progress
        
    Returns:
        Dictionary with search results
    """
    print("Setting up genetic algorithm search...")
    
    # Define parameter space and encoding based on simulation type
    if simulation_type == "gol":
        param_ranges = {
            'grid_size': (16, 128),
            'init_density': (0.01, 0.6),
            'birth_idx': (0, len(DEFAULT_GOL_PARAMS['birth']) - 1),
            'survive_idx': (0, len(DEFAULT_GOL_PARAMS['survive']) - 1),
            'n_steps': (250, 2000)
        }
    elif simulation_type == "boids":
        param_ranges = {
            'n_boids': (20, 300),
            'visual_range': (3.0, 30.0),
            'protected_range': (0.5, 5.0),
            'max_speed': (1.0, 15.0),
            'cohesion_factor': (0.001, 0.2),
            'alignment_factor': (0.01, 0.5),
            'separation_factor': (0.01, 0.5),
            'n_steps': (250, 2000)
        }
    elif simulation_type == "lenia":
        param_ranges = {
            'grid_size': (64, 256),
            'time_step': (0.05, 0.4),
            'kernel_radius': (5, 30),
            'growth_center': (0.05, 0.4),
            'growth_width': (0.005, 0.2),
            'init_radius': (5, 40),
            'n_steps': (250, 2000)
        }
    else:
        raise ValueError(f"Unknown simulation type: {simulation_type}")
    
    # Initialize population
    random.seed(42)
    population = []
    for _ in range(population_size):
        individual = {}
        for param, (min_val, max_val) in param_ranges.items():
            if param.endswith('_idx'):
                individual[param] = random.randint(min_val, max_val)
            elif isinstance(min_val, int):
                individual[param] = random.randint(min_val, max_val)
            else:
                individual[param] = min_val + random.random() * (max_val - min_val)
        population.append(individual)
    
    # Helper function to convert genetic algorithm parameters to simulation parameters
    def convert_to_sim_params(individual, simulation_type):
        sim_params = individual.copy()
        
        # Handle special cases
        if 'birth_idx' in sim_params:
            idx = int(sim_params.pop('birth_idx'))
            sim_params['birth'] = DEFAULT_GOL_PARAMS['birth'][min(idx, len(DEFAULT_GOL_PARAMS['birth'])-1)]
            
        if 'survive_idx' in sim_params:
            idx = int(sim_params.pop('survive_idx'))
            sim_params['survive'] = DEFAULT_GOL_PARAMS['survive'][min(idx, len(DEFAULT_GOL_PARAMS['survive'])-1)]
        
        # Round integer parameters
        for param in ['grid_size', 'n_boids', 'kernel_radius', 'init_radius', 'n_steps']:
            if param in sim_params:
                sim_params[param] = int(round(sim_params[param]))
        
        return sim_params
    
    # Fitness function
    def evaluate_fitness(individual, simulation_type, use_asal, n_seeds, metric):
        sim_params = convert_to_sim_params(individual, simulation_type)
        
        # Run simulations with multiple seeds
        results = []
        for seed in range(n_seeds):
            if use_asal:
                result = run_single_simulation_with_asal(sim_params, simulation_type, seed)
            else:
                result = run_standard_simulation(sim_params, simulation_type, seed)
                
            if result.get("success", False):
                results.append(result)
        
        # No successful simulations
        if not results:
            return 0.0
        
        # Calculate score based on the chosen metric
        if metric == "Delta":
            score = np.mean([r.get("Delta", 0.0) for r in results])
        elif metric == "Gamma":
            score = np.mean([r.get("Gamma", 0.0) for r in results])
        elif metric == "Psi":
            score = np.mean([r.get("Psi", 0.0) for r in results])
        else:  # combined
            score = np.mean([
                (r.get("Delta", 0.0) + r.get("Gamma", 0.0) + r.get("Psi", 0.0)) / 3.0
                for r in results
            ])
            
        return score
    
    # Genetic operators
    def crossover(parent1, parent2):
        child = {}
        for param in parent1.keys():
            # 50% chance to inherit from either parent
            if random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child
    
    def mutate(individual, mutation_rate=0.2):
        mutated = individual.copy()
        for param, (min_val, max_val) in param_ranges.items():
            # Apply mutation with probability mutation_rate
            if random.random() < mutation_rate:
                if param.endswith('_idx') or isinstance(min_val, int):
                    # Integer mutation
                    mutated[param] = random.randint(min_val, max_val)
                else:
                    # Float mutation - perturb by up to 20% of range
                    range_size = max_val - min_val
                    perturbation = (random.random() - 0.5) * 0.4 * range_size
                    mutated[param] = max(min_val, min(max_val, mutated[param] + perturbation))
        return mutated
    
    # Run genetic algorithm
    history = {'best_fitness': [], 'avg_fitness': [], 'best_individual': []}
    best_individual = None
    best_fitness = 0.0
    
    print(f"Starting genetic algorithm with {n_iterations} generations...")
    
    for generation in range(n_iterations):
        if verbose:
            print(f"Generation {generation+1}/{n_iterations}")
        
        # Evaluate fitness
        fitness_scores = []
        for individual in population:
            fitness = evaluate_fitness(individual, simulation_type, use_asal, n_seeds, metric)
            fitness_scores.append(fitness)
        
        # Track best individual
        generation_best_idx = np.argmax(fitness_scores)
        generation_best = population[generation_best_idx]
        generation_best_fitness = fitness_scores[generation_best_idx]
        
        if generation_best_fitness > best_fitness:
            best_fitness = generation_best_fitness
            best_individual = generation_best.copy()
            
            if verbose:
                print(f"New best individual found: {best_fitness}")
        
        # Record history
        history['best_fitness'].append(best_fitness)
        history['avg_fitness'].append(np.mean(fitness_scores))
        history['best_individual'].append(best_individual)
        
        # Save intermediate results
        if save_dir and (generation % 5 == 0 or generation == n_iterations - 1):
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save progress
            with open(os.path.join(save_dir, f"ga_progress_{timestamp}.json"), 'w') as f:
                json.dump({
                    "generation": generation,
                    "best_fitness": best_fitness,
                    "best_individual": best_individual,
                    "avg_fitness": np.mean(fitness_scores)
                }, f, indent=2)
        
        # Early stopping if perfect solution found or last generation
        if best_fitness >= 0.99 or generation == n_iterations - 1:
            if generation < n_iterations - 1:
                print(f"Early stopping at generation {generation+1}: found excellent solution")
            break
        
        # Selection - tournament selection
        next_generation = []
        
        # Elitism - keep the best individual
        next_generation.append(best_individual)
        
        # Fill the rest of the population
        while len(next_generation) < population_size:
            # Tournament selection
            tournament_size = 3
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            parent1_idx = tournament_indices[np.argmax(tournament_fitness)]
            parent1 = population[parent1_idx]
            
            tournament_indices = random.sample(range(population_size), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            parent2_idx = tournament_indices[np.argmax(tournament_fitness)]
            parent2 = population[parent2_idx]
            
            # Crossover
            child = crossover(parent1, parent2)
            
            # Mutation
            child = mutate(child)
            
            next_generation.append(child)
        
        # Update population
        population = next_generation
    
    # Convert best individual to simulation parameters
    best_params = convert_to_sim_params(best_individual, simulation_type)
    
    # Save final results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        # Save best parameters
        with open(os.path.join(save_dir, "best_parameters.json"), 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Save history
        with open(os.path.join(save_dir, "ga_history.json"), 'w') as f:
            json.dump({
                "best_fitness": history['best_fitness'],
                "avg_fitness": history['avg_fitness']
            }, f)
        
        # Create convergence plot
        plt.figure(figsize=(10, 6))
        plt.plot(history['best_fitness'], 'b-', label='Best Fitness')
        plt.plot(history['avg_fitness'], 'r-', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel(f'{metric} Score')
        plt.title('Genetic Algorithm Convergence')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "ga_convergence.png"), dpi=300)
    
    print(f"Genetic algorithm search complete. Best {metric} score: {best_fitness}")
    print(f"Best parameters: {best_params}")
    
    return {
        "best_params": best_params,
        "best_score": best_fitness,
        "history": history
    }

def main():
    """Main entry point for advanced parameter search."""
    args = parse_args()
    
    print(f"Starting advanced parameter search for {args.simulation_type} simulation")
    print(f"Search method: {args.search_method}")
    print(f"Optimizing for metric: {args.metric}")
    
    # Create save directory if needed
    if args.save_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(args.save_dir, f"{args.simulation_type}_{args.search_method}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Results will be saved in: {save_dir}")
    else:
        save_dir = None
    
    # Run the selected search method
    start_time = time.time()
    
    if args.search_method == "bayesian":
        result = run_bayesian_optimization(
            args.simulation_type,
            args.n_iterations,
            args.n_initial,
            metric=args.metric,
            use_asal=args.use_asal,
            n_seeds=args.n_seeds,
            n_jobs=args.n_jobs,
            save_dir=save_dir,
            verbose=args.verbose
        )
    
    elif args.search_method == "genetic":
        result = run_genetic_algorithm_search(
            args.simulation_type,
            args.n_iterations,
            population_size=args.n_initial,
            metric=args.metric,
            use_asal=args.use_asal,
            n_seeds=args.n_seeds,
            n_jobs=args.n_jobs,
            save_dir=save_dir,
            verbose=args.verbose
        )
    
    elif args.search_method == "multi_stage":
        # TODO: Implement multi-stage search
        print("Multi-stage search not yet implemented")
        result = None
    
    elif args.search_method == "surrogate":
        # TODO: Implement surrogate model-based search
        print("Surrogate model-based search not yet implemented")
        result = None
    
    else:
        raise ValueError(f"Unknown search method: {args.search_method}")
    
    end_time = time.time()
    print(f"Search completed in {end_time - start_time:.2f} seconds")
    
    # Optionally run final simulation with the best parameters
    if result and args.visualize:
        print("Creating visualization of the search results...")
        # TODO: Generate visualization
    
    print("Advanced parameter search complete!")

if __name__ == "__main__":
    main()
