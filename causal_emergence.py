import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import importlib

# --- Assumed asal structure ---
# Add the 'asal' directory path to sys.path if it's not already discoverable
# Example: Adjust the path as needed
asal_path = './asal' # Or the correct path to your asal directory
if asal_path not in sys.path:
    sys.path.insert(0, asal_path)

# Import necessary modules from the asal codebase
# (Adjust module/function names if they differ in your actual codebase)
try:
    # Simulation Substrates
    gol_module = importlib.import_module('substrates.gol')
    boids_module = importlib.import_module('substrates.boids')
    # Information Theory Utilities
    it_utils = importlib.import_module('info_theory_utils')
    # Utility functions (if needed, e.g., for state processing)
    # util_module = importlib.import_module('util')
except ImportError as e:
    print(f"Error importing modules from 'asal': {e}")
    print("Please ensure the 'asal' directory is in the Python path and contains the necessary modules.")
    sys.exit(1)

# Import the new emergence metrics module
try:
    emergence_metrics_module = importlib.import_module('emergence_metrics')
except ImportError as e:
    print(f"Error importing emergence_metrics: {e}")
    print("Using built-in emergence metrics calculation.")
    emergence_metrics_module = None

# --- Configuration ---
SIMULATION_TYPE = 'gol'  # 'gol' or 'boids'
N_STEPS = 1000          # Number of simulation steps
N_RUNS = 1             # Number of simulation runs (for better statistics if needed)
RANDOM_SEED = 42

# GoL specific parameters
GRID_SIZE = 32
INIT_METHOD_GOL = 'random' # or 'glider', etc.
RANDOM_DENSITY = 0.1

# Boids specific parameters
N_BOIDS = 50
WORLD_SIZE = 100.0
VISUAL_RANGE = 10.0
PROTECTED_RANGE = 2.0
MAX_SPEED = 5.0
COHESION_FACTOR = 0.01
ALIGNMENT_FACTOR = 0.125
SEPARATION_FACTOR = 0.05
TURN_FACTOR = 0.2

# --- Macrostate Definitions ---

def get_gol_macro_state(grid):
    """Calculates macro state(s) for Game of Life.
    Example: Density of live cells.
    """
    density = np.mean(grid > 0) # Assumes live cell is > 0
    # Add other macro variables if needed, return as a flat numpy array
    return np.array([density])

def get_boids_macro_state(positions, velocities):
    """Calculates macro state(s) for Boids.
    Example: Polarization and Cohesion.
    """
    n_boids = positions.shape[0]
    if n_boids == 0:
        return np.array([0.0, 0.0]) # Default values if no boids

    # Polarization: Average alignment of velocities
    avg_velocity = np.mean(velocities, axis=0)
    speed = np.linalg.norm(avg_velocity)
    polarization = speed / MAX_SPEED if MAX_SPEED > 0 else 0.0 # Normalize if possible

    # Cohesion: Average distance from the center of mass
    center_of_mass = np.mean(positions, axis=0)
    distances = np.linalg.norm(positions - center_of_mass, axis=1)
    cohesion = np.mean(distances)

    # Return as a flat numpy array
    return np.array([polarization, cohesion])


# --- Simulation Runner ---

# Add necessary JAX import
import jax
import jax.numpy as jnp

def run_simulation(sim_type, n_steps, seed):
    """Runs the selected simulation and returns micro and macro state histories."""
    np.random.seed(seed)
    micro_history = []
    macro_history = []
    # Add this new list to store visualization frames
    visual_frames = []

    if sim_type == 'gol':
        # Initialize GoL (adapt based on your gol_module.GoL implementation)
        print(f"Initializing GoL (Grid: {GRID_SIZE}x{GRID_SIZE})...")
        
        # Inspect available classes/functions in gol_module
        gol_classes = [name for name in dir(gol_module) 
                      if not name.startswith('__') and 
                      (name == 'GoL' or 'gol' in name.lower() or 'life' in name.lower())]
        
        print(f"Available GoL classes/functions: {gol_classes}")
        
        # Initialize JAX random number generator with seed
        rng_key = jax.random.PRNGKey(seed)
        
        # For the JAX-based GameOfLife implementation
        if hasattr(gol_module, 'GameOfLife'):
            print("Using GameOfLife class with JAX")
            # Initialize with grid_size parameter (from the gol.py source)
            gol_class = gol_module.GameOfLife(grid_size=GRID_SIZE)
            
            # Get default parameters
            key, subkey = jax.random.split(rng_key)
            params = gol_class.params_cgol  # Use Conway's Game of Life parameters
            
            # Initialize state
            key, subkey = jax.random.split(key)
            # For random initialization, we can use the class's init_state method
            state = gol_class.init_state(subkey, params)
            
            # Store initial state
            initial_state = np.array(state)  # Convert from JAX array to numpy
            micro_history.append(initial_state.flatten())
            macro_history.append(get_gol_macro_state(initial_state))
            
            # Store grid state for visualization
            visual_frames.append(initial_state.copy())
            
            # Run simulation steps
            for t in range(n_steps):
                key, subkey = jax.random.split(key)
                state = gol_class.step_state(subkey, state, params)
                
                current_state = np.array(state)  # Convert from JAX array to numpy
                micro_history.append(current_state.flatten())
                macro_history.append(get_gol_macro_state(current_state))
                
                # Save visual frame (only store a subset of frames to save memory)
                if t % max(1, n_steps // 100) == 0:  # Save ~100 frames total
                    visual_frames.append(current_state.copy())
                
                if (t+1) % 100 == 0:
                    print(f"GoL Step: {t+1}/{n_steps}")
                    
        # Try with GameOfLifeInit if available
        elif hasattr(gol_module, 'GameOfLifeInit'):
            print("Using GameOfLifeInit class with JAX")
            # Initialize with grid_size parameter (from the gol.py source)
            gol_class = gol_module.GameOfLifeInit(grid_size=GRID_SIZE)
            
            # Get default parameters - this returns a dictionary with 'params_init'
            key, subkey = jax.random.split(rng_key)
            params = gol_class.default_params(subkey)
            
            # Initialize state
            key, subkey = jax.random.split(key)
            # For random initialization, we can use the class's init_state method
            state = gol_class.init_state(subkey, params)
            
            # Store initial state
            initial_state = np.array(state)  # Convert from JAX array to numpy
            micro_history.append(initial_state.flatten())
            macro_history.append(get_gol_macro_state(initial_state))
            
            # Store grid state for visualization
            visual_frames.append(initial_state.copy())
            
            # Run simulation steps
            for t in range(n_steps):
                key, subkey = jax.random.split(key)
                state = gol_class.step_state(subkey, state, params)
                
                current_state = np.array(state)  # Convert from JAX array to numpy
                micro_history.append(current_state.flatten())
                macro_history.append(get_gol_macro_state(current_state))
                
                # Save visual frame (only store a subset of frames to save memory)
                if t % max(1, n_steps // 100) == 0:  # Save ~100 frames total
                    visual_frames.append(current_state.copy())
                
                if (t+1) % 100 == 0:
                    print(f"GoL Step: {t+1}/{n_steps}")
        
        else:
            print("Error: Could not find a suitable Game of Life implementation.")
            return None, None, None

    elif sim_type == 'boids':
        # Initialize Boids (adapt based on your boids_module.Boids implementation)
        print(f"Initializing Boids (N: {N_BOIDS}, World Size: {WORLD_SIZE})...")
        if hasattr(boids_module, 'Boids'):
            # Placeholder: Adapt to your Boids class constructor
            boids_sim = boids_module.Boids(
                num_boids=N_BOIDS,
                world_size=WORLD_SIZE,
                visual_range=VISUAL_RANGE,
                protected_range=PROTECTED_RANGE,
                max_speed=MAX_SPEED,
                cohesion_factor=COHESION_FACTOR,
                alignment_factor=ALIGNMENT_FACTOR,
                separation_factor=SEPARATION_FACTOR,
                turn_factor=TURN_FACTOR
                # Add other necessary parameters
            )
            if hasattr(boids_sim, 'seed'):
                 boids_sim.seed(seed) # Ensure seeding if available

            # Assuming properties or methods to get initial state
            # Need both positions and velocities for micro and macro state
            if hasattr(boids_sim, 'positions') and hasattr(boids_sim, 'velocities'):
                 initial_pos = boids_sim.positions
                 initial_vel = boids_sim.velocities
                 # Micro state: Flattened positions and velocities
                 micro_history.append(np.concatenate([initial_pos.flatten(), initial_vel.flatten()]))
                 macro_history.append(get_boids_macro_state(initial_pos, initial_vel))
                 
                 # Save visual frame (for boids, store positions and velocities)
                 visual_frames.append((initial_pos.copy(), initial_vel.copy()))
            else:
                 print("Error: Cannot access initial 'positions' and 'velocities' from Boids sim.")
                 return None, None, None

            # Run simulation steps
            for t in range(n_steps):
                if hasattr(boids_sim, 'step'):
                    boids_sim.step()
                    current_pos = boids_sim.positions
                    current_vel = boids_sim.velocities
                    micro_history.append(np.concatenate([current_pos.flatten(), current_vel.flatten()]))
                    macro_history.append(get_boids_macro_state(current_pos, current_vel))
                    
                    # Save visual frame (only store a subset of frames to save memory)
                    if t % max(1, n_steps // 100) == 0:  # Save ~100 frames total
                        visual_frames.append((current_pos.copy(), current_vel.copy()))
                else:
                    print("Error: 'step' method not found in Boids simulation object.")
                    break
                if (t+1) % 100 == 0:
                    print(f"Boids Step: {t+1}/{n_steps}")

        else:
             print("Error: Boids class not found in substrates.boids. Please adapt.")
             return None, None, None

    else:
        print(f"Error: Unknown simulation type '{sim_type}'")
        return None, None, None

    # Convert lists to numpy arrays
    # Note: Stacking assumes consistent shapes across time steps
    try:
        micro_history_np = np.stack(micro_history, axis=0)
        macro_history_np = np.stack(macro_history, axis=0)
        # Convert visual_frames to numpy array if possible
        visual_frames_np = np.array(visual_frames)
    except ValueError as e:
        print(f"Error stacking history arrays: {e}")
        print("Check if state dimensions are consistent across simulation steps.")
        return None, None, None

    return micro_history_np, macro_history_np, visual_frames_np


# --- Emergence Metric Calculation ---

def calculate_emergence_metrics(S_hist, M_hist, discretize=True, n_bins=10):
    """
    Calculates Rosas2020 emergence metrics (Psi, Gamma, Delta).

    Args:
        S_hist (np.ndarray): Time series of micro states (T+1, dim_S).
        M_hist (np.ndarray): Time series of macro states (T+1, dim_M).
        discretize (bool): Whether to discretize continuous data before MI calculation.
        n_bins (int): Number of bins for discretization.

    Returns:
        dict: Dictionary containing Psi, Gamma, Delta values.
              Returns None if MI/CMI functions are unavailable or errors occur.
    """
    if not hasattr(it_utils, 'mutual_info') or not hasattr(it_utils, 'conditional_mutual_info'):
        print("Error: `mutual_info` or `conditional_mutual_info` not found in info_theory_utils.")
        return None

    T = S_hist.shape[0] - 1 # Number of transitions
    if T <= 0:
        print("Error: Not enough time steps for metric calculation.")
        return None

    # Prepare data arrays: S_t, M_t, S_{t+1}, M_{t+1}
    S_t = S_hist[:-1]
    M_t = M_hist[:-1]
    S_tp1 = S_hist[1:]
    M_tp1 = M_hist[1:]
    
    print(f"原データの形状: S_t: {S_t.shape}, M_t: {M_t.shape}, S_tp1: {S_tp1.shape}, M_tp1: {M_tp1.shape}")

    # Check if micro state is very high dimensional (Game of Life has 1024 dimensions!)
    is_high_dim_micro = (S_t.shape[1] > 100)
    
    if is_high_dim_micro:
        print(f"マイクロ状態の次元が高いです ({S_t.shape[1]} 次元)。次元削減を行います。")
        
        # Option 1: Use PCA to reduce dimensionality
        try:
            from sklearn.decomposition import PCA
            
            # Reduce to a more manageable number of dimensions
            reduced_dims = min(20, S_t.shape[0] // 10)  # Reduce more aggressively to 20 dimensions
            print(f"PCAを使用してマイクロ状態を {reduced_dims} 次元に削減します")
            
            # Fit PCA on the micro states
            pca = PCA(n_components=reduced_dims)
            S_t_reduced = pca.fit_transform(S_t)
            S_tp1_reduced = pca.transform(S_tp1)
            
            print(f"分散の {pca.explained_variance_ratio_.sum():.2%} を保持しました")
            print(f"PCA後の形状: S_t: {S_t_reduced.shape}, S_tp1: {S_tp1_reduced.shape}")
            
            # Replace original states with reduced ones
            S_t = S_t_reduced
            S_tp1 = S_tp1_reduced
        except Exception as e:
            print(f"PCA次元削減に失敗しました: {e}")
            print("代替アプローチ: ランダムサンプリングを使用します")
            
            # Option 2: Random sampling of dimensions
            import random
            random.seed(42)  # For reproducibility
            
            # Sample a subset of dimensions (e.g., 20)
            sample_size = 20
            selected_dims = random.sample(range(S_t.shape[1]), sample_size)
            
            print(f"{S_t.shape[1]} 次元から {sample_size} 次元をランダムに選択しました")
            S_t = S_t[:, selected_dims]
            S_tp1 = S_tp1[:, selected_dims]
            print(f"次元サンプリング後の形状: S_t: {S_t.shape}, S_tp1: {S_tp1.shape}")
    
    # --- Discretization (Optional but often necessary for discrete MI estimators) ---
    # Adapt this based on your data and MI estimator.
    # `pyinform` often requires integer arrays representing discrete bins.
    # If your info_theory_utils handles continuous data (e.g., kNN estimators),
    # you might skip this or use a different approach.
    if discretize:
        print(f"Discretizing data using {n_bins} bins...")
        # Simple uniform binning - consider more sophisticated methods if needed
        def _discretize_series(series):
            binned = np.zeros_like(series, dtype=int)
            for i in range(series.shape[1]): # Bin each dimension independently
                col_min, col_max = np.min(series[:, i]), np.max(series[:, i])
                if col_max == col_min: # Handle constant columns
                    # Assign bin 0, or handle as appropriate for your MI estimator
                     binned[:, i] = 0
                else:
                    bins = np.linspace(col_min, col_max, n_bins + 1)
                    binned[:, i] = np.digitize(series[:, i], bins[:-1], right=False) -1
                    # Ensure bins are 0 to n_bins-1
                    binned[:, i] = np.clip(binned[:, i], 0, n_bins-1)
            return binned

        # Apply discretization if data looks continuous (more than 2 unique values often suggests this)
        if len(np.unique(S_t)) > 2:
            S_t = _discretize_series(S_t)
            S_tp1 = _discretize_series(S_tp1)
        if len(np.unique(M_t)) > 2:
             M_t = _discretize_series(M_t)
             M_tp1 = _discretize_series(M_tp1)

        # Ensure integer type if required by underlying MI library (like pyinform)
        S_t, M_t, S_tp1, M_tp1 = S_t.astype(int), M_t.astype(int), S_tp1.astype(int), M_tp1.astype(int)

    # --- Calculate MI/CMI terms ---
    print("情報理論的指標を計算しています...")
    try:
        # Method 1: Using the direct calculation as before
        results = {}
        
        # For Psi = I(S_{t+1} ; M_t | S_t)
        print("Psi (条件付き相互情報量)を計算中...")
        print(f"入力形状: S_tp1: {S_tp1.shape}, M_t: {M_t.shape}, S_t: {S_t.shape}")
        cmi_Snext_Mt_given_St = it_utils.conditional_mutual_info(S_tp1, M_t, S_t, bins=n_bins)
        results["Psi"] = cmi_Snext_Mt_given_St
        print(f"  I(S_tp1; M_t | S_t) = {results['Psi']:.4f}")

        # For Gamma = I(S_{t+1} ; M_t) - I(S_{t+1} ; S_t)
        print("Gamma (相互情報量の差)を計算中...")
        print(f"入力形状 (1): S_tp1: {S_tp1.shape}, M_t: {M_t.shape}")
        mi_Snext_Mt = it_utils.mutual_info(S_tp1, M_t, bins=n_bins)
        print(f"入力形状 (2): S_tp1: {S_tp1.shape}, S_t: {S_t.shape}")
        mi_Snext_St = it_utils.mutual_info(S_tp1, S_t, bins=n_bins)
        results["Gamma"] = mi_Snext_Mt - mi_Snext_St
        print(f"  I(S_tp1; M_t) = {mi_Snext_Mt:.4f}")
        print(f"  I(S_tp1; S_t) = {mi_Snext_St:.4f}")
        print(f"  Gamma = {results['Gamma']:.4f}")

        # For Delta = I(M_{t+1} ; M_t) - I(M_{t+1} ; S_t)
        print("Delta (マクロ予測可能性)を計算中...")
        print(f"入力形状 (1): M_tp1: {M_tp1.shape}, M_t: {M_t.shape}")
        mi_Mnext_Mt = it_utils.mutual_info(M_tp1, M_t, bins=n_bins)
        print(f"入力形状 (2): M_tp1: {M_tp1.shape}, S_t: {S_t.shape}")
        mi_Mnext_St = it_utils.mutual_info(M_tp1, S_t, bins=n_bins)
        results["Delta"] = mi_Mnext_Mt - mi_Mnext_St
        print(f"  I(M_tp1; M_t) = {mi_Mnext_Mt:.4f}")
        print(f"  I(M_tp1; S_t) = {mi_Mnext_St:.4f}")
        print(f"  Delta = {results['Delta']:.4f}")
        
        # Method 2: Using the ReconcilingEmergences framework if available
        if emergence_metrics_module is not None:
            print("\nReconcilingEmergences フレームワークを使用して計算しています...")
            try:
                # Use our new module to calculate metrics
                reconciling_results = emergence_metrics_module.calculate_all_emergence_metrics(
                    S_t, M_t, tau=1,
                    mutual_info_func=it_utils.mutual_info,
                    cond_mutual_info_func=it_utils.conditional_mutual_info
                )
                
                # Print and store results
                for metric, value in reconciling_results.items():
                    print(f"  {metric} (Reconciling) = {value:.4f}")
                    results[f"{metric}_reconciling"] = value
                    
            except Exception as e:
                print(f"ReconcilingEmergences 計算中のエラー: {e}")
                import traceback
                traceback.print_exc()

        return results

    except Exception as e:
        print(f"MI/CMI 計算中のエラー: {e}")
        import traceback
        traceback.print_exc()
        print("info_theory_utils 関数の実装と必要な入力形式を確認してください。")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Running {SIMULATION_TYPE} simulation...")
    all_metrics = []

    # --- Data Generation Loop (Optional: run multiple times for stability) ---
    for run in range(N_RUNS):
        print(f"\n--- Run {run+1}/{N_RUNS} ---")
        current_seed = RANDOM_SEED + run
        S_history, M_history, visual_frames = run_simulation(SIMULATION_TYPE, N_STEPS, seed=current_seed)

        if S_history is None or M_history is None:
            print("Simulation failed. Exiting.")
            sys.exit(1)

        print(f"\nGenerated data shapes: S_history={S_history.shape}, M_history={M_history.shape}")

        # --- Metric Calculation ---
        # Decide whether to discretize based on the simulation and MI estimator
        # GoL state is often binary/discrete already. Boids might need discretization.
        needs_discretization = (SIMULATION_TYPE == 'boids') # Example heuristic

        metrics = calculate_emergence_metrics(S_history, M_history, discretize=needs_discretization)

        if metrics:
            print("\n--- Emergence Metrics (Rosas et al. 2020) ---")
            # Standard metrics
            print(f" Ψ (Psi):   {metrics['Psi']:.4f}")
            print(f" Γ (Gamma): {metrics['Gamma']:.4f}")
            print(f" Δ (Delta): {metrics['Delta']:.4f}")
            
            # Reconciling framework metrics if available
            if "Delta_reconciling" in metrics:
                print("\n--- Reconciling Emergences Framework Metrics ---")
                print(f" Δ_R (Delta): {metrics['Delta_reconciling']:.4f}")
                print(f" Γ_R (Gamma): {metrics['Gamma_reconciling']:.4f}")
                print(f" Ψ_R (Psi):   {metrics['Psi_reconciling']:.4f}")
                
                # Remove this line that's causing the NameError
                # explain_metric_differences(metrics)
            
            all_metrics.append(metrics)
        else:
            print("\nFailed to calculate emergence metrics.")

    # --- Aggregate Results (if N_RUNS > 1) ---
    if N_RUNS > 1 and all_metrics:
        print(f"\n--- Average Metrics over {N_RUNS} runs ---")
        avg_psi = np.mean([m['Psi'] for m in all_metrics])
        avg_gamma = np.mean([m['Gamma'] for m in all_metrics])
        avg_delta = np.mean([m['Delta'] for m in all_metrics])
        std_psi = np.std([m['Psi'] for m in all_metrics])
        std_gamma = np.std([m['Gamma'] for m in all_metrics])
        std_delta = np.std([m['Delta'] for m in all_metrics])
        print(f" Avg Ψ (Psi):   {avg_psi:.4f} +/- {std_psi:.4f}")
        print(f" Avg Γ (Gamma): {avg_gamma:.4f} +/- {std_gamma:.4f}")
        print(f" Avg Δ (Delta): {avg_delta:.4f} +/- {std_delta:.4f}")

    # --- Potential for Causal Blanket Extension ---
    print("\n--- Future Extension: Causal Blankets ---")
    print("The generated S_history and M_history can be used for Causal Blanket analysis.")
    print("This typically involves defining Sensory (Se), Active (Ac), Internal (In), and External (Ex) states")
    print("and calculating information flow between them (e.g., I(Se_tp1; In_t | Se_t, Ac_t)).")
    print("The `calculate_emergence_metrics` function structure can be adapted for this purpose,")
    print("using the same `info_theory_utils`.")

    # --- Visualization (Optional) ---
    # Example: Plot macro-state time series
    if M_history is not None:
        plt.figure(figsize=(12, 4))
        for i in range(M_history.shape[1]):
             plt.plot(M_history[:, i], label=f'Macro Var {i+1}')
        plt.title(f'{SIMULATION_TYPE.upper()} Macro-State Dynamics')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Add visualization of the simulation states
        from emergence_visualization import visualize_simulation_states
        fig = visualize_simulation_states(visual_frames, SIMULATION_TYPE, metrics)
        plt.show()

def explain_metric_differences(metrics):
    """
    Explains the differences between standard and Reconciling Emergences metrics.
    
    Args:
        metrics: Dictionary containing both standard and reconciling metrics
    """
    print("\n--- Explanation of Differences Between Metrics ---")
    
    # Check if we have both types of metrics
    has_reconciling = all(f"{m}_reconciling" in metrics for m in ["Delta", "Gamma", "Psi"])
    
    if not has_reconciling:
        print("Reconciling metrics not available for comparison.")
        return
    
    # Explain Delta
    delta_std = metrics.get("Delta", 0)
    delta_rec = metrics.get("Delta_reconciling", 0)
    delta_diff = abs(delta_rec - delta_std)
    
    print(f"Delta (Standard): {delta_std:.4f} vs Delta (Reconciling): {delta_rec:.4f}")
    print("  • Standard Delta measures if macro variables better predict future macro states than micro variables do")
    print("  • Reconciling Delta compares the predictive power of macro variables with the entire micro system")
    print("  • For each micro variable, it calculates whether macro is more predictive than all micro variables combined")
    print(f"  • The difference ({delta_diff:.4f}) indicates {'strong' if delta_diff > 0.5 else 'moderate' if delta_diff > 0.1 else 'subtle'} effects of specific variable-to-variable downward causation")
    
    # Explain Gamma
    gamma_std = metrics.get("Gamma", 0)
    gamma_rec = metrics.get("Gamma_reconciling", 0)
    gamma_diff = abs(gamma_rec - gamma_std)
    
    print(f"\nGamma (Standard): {gamma_std:.4f} vs Gamma (Reconciling): {gamma_rec:.4f}")
    print("  • Standard Gamma measures if macro variables better predict future micro states than past micro variables do")
    print("  • Reconciling Gamma compares each micro variable's self-prediction with how well the macro predicts it")
    print("  • It identifies variables that are better predicted by the macro level than by their own history")
    print(f"  • The difference ({gamma_diff:.4f}) shows {'strong' if gamma_diff > 0.5 else 'moderate' if gamma_diff > 0.1 else 'minimal'} variation in causal decoupling across individual micro variables")
    
    # Explain Psi
    psi_std = metrics.get("Psi", 0)
    psi_rec = metrics.get("Psi_reconciling", 0)
    psi_diff = abs(psi_rec - psi_std)
    
    print(f"\nPsi (Standard): {psi_std:.4f} vs Psi (Reconciling): {psi_rec:.4f}")
    print("  • Standard Psi measures information gain from macro variables about future micro states beyond what micro history provides")
    print("  • Reconciling Psi is calculated as conditional mutual information between macro, future micro, given past micro")
    print("  • Psi tends to be more consistent between methods as both capture the same conditional information gain")
    print(f"  • The difference ({psi_diff:.4f}) is typically smaller than for other metrics")
    
    print("\nConclusion:")
    if delta_rec > delta_std and delta_rec > 0.5:
        print("  • This system shows strong downward causation effects (high Delta_reconciling)")
    if gamma_rec > gamma_std and gamma_rec > 0.3:
        print("  • The macro level shows enhanced predictive ability for some micro variables (high Gamma_reconciling)")
    if psi_rec > 0.3:
        print("  • The macro level provides significant information beyond what's available at the micro level")
    if delta_rec < 0.1 and gamma_rec < 0.1 and psi_rec < 0.1:
        print("  • Limited evidence of causal emergence in this system")

# Modify the main execution section to include this explanation
if __name__ == "__main__":
    # ...existing code...
    
    # After calculating metrics, add the explanation
    if metrics:
        print("\n--- Emergence Metrics (Rosas et al. 2020) ---")
        # Standard metrics
        print(f" Ψ (Psi):   {metrics['Psi']:.4f}")
        print(f" Γ (Gamma): {metrics['Gamma']:.4f}")
        print(f" Δ (Delta): {metrics['Delta']:.4f}")
        
        # Reconciling framework metrics if available
        if "Delta_reconciling" in metrics:
            print("\n--- Reconciling Emergences Framework Metrics ---")
            print(f" Δ_R (Delta): {metrics['Delta_reconciling']:.4f}")
            print(f" Γ_R (Gamma): {metrics['Gamma_reconciling']:.4f}")
            print(f" Ψ_R (Psi):   {metrics['Psi_reconciling']:.4f}")
            
            # Add explanation of differences
            explain_metric_differences(metrics)
        
        all_metrics.append(metrics)
    else:
        print("\nFailed to calculate emergence metrics.")
    
    # ...existing code...

# --- Add at the end of the file ---

def visualize_emergence_metrics(metrics, S_history, M_history, simulation_type):
    """
    Visualizes emergence metrics and their interpretation.
    
    Args:
        metrics: Dictionary containing emergence metrics
        S_history: Micro state history array
        M_history: Macro state history array
        simulation_type: Type of simulation ('gol', 'boids', etc.)
    """
    # Create a multi-panel figure
    fig = plt.figure(figsize=(18, 12))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # 1. Macro state evolution
    ax1 = plt.subplot(2, 3, 1)
    for i in range(M_history.shape[1]):
        ax1.plot(M_history[:, i], label=f'Macro Var {i+1}')
    ax1.set_title('Macro-State Evolution', fontsize=14)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Value')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Standard vs Reconciling metrics comparison
    ax2 = plt.subplot(2, 3, 2)
    if "Delta_reconciling" in metrics:
        metrics_to_plot = [
            ('Delta', 'Delta_reconciling', 'Downward Causation'),
            ('Gamma', 'Gamma_reconciling', 'Causal Decoupling'),
            ('Psi', 'Psi_reconciling', 'Causal Emergence')
        ]
        
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        standard_vals = [metrics[m[0]] for m in metrics_to_plot]
        reconciling_vals = [metrics[m[1]] for m in metrics_to_plot]
        
        ax2.bar(x - width/2, standard_vals, width, label='Standard')
        ax2.bar(x + width/2, reconciling_vals, width, label='Reconciling')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels([m[2] for m in metrics_to_plot])
        ax2.set_title('Standard vs. Reconciling Metrics', fontsize=14)
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True, axis='y')
    else:
        metrics_to_plot = ['Delta', 'Gamma', 'Psi']
        x = np.arange(len(metrics_to_plot))
        vals = [metrics[m] for m in metrics_to_plot]
        
        ax2.bar(x, vals)
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_to_plot)
        ax2.set_title('Emergence Metrics', fontsize=14)
        ax2.set_ylabel('Value')
        ax2.grid(True, axis='y')
    
    # 3. Interpretation heatmap - what the values mean
    ax3 = plt.subplot(2, 3, 3)
    interpretation_matrix = np.array([
        [metrics.get('Delta', 0), metrics.get('Delta_reconciling', 0) if 'Delta_reconciling' in metrics else 0],
        [metrics.get('Gamma', 0), metrics.get('Gamma_reconciling', 0) if 'Gamma_reconciling' in metrics else 0],
        [metrics.get('Psi', 0), metrics.get('Psi_reconciling', 0) if 'Psi_reconciling' in metrics else 0]
    ])
    
    im = ax3.imshow(interpretation_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax3)
    
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(['Standard', 'Reconciling'])
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(['Delta', 'Gamma', 'Psi'])
    ax3.set_title('Metrics Heatmap', fontsize=14)
    
    for i in range(3):
        for j in range(2):
            if j < interpretation_matrix.shape[1]:
                ax3.text(j, i, f'{interpretation_matrix[i, j]:.3f}', 
                        ha='center', va='center', color='white' if interpretation_matrix[i, j] > 0.5 else 'black')
    
    # 4. Dimensionality-reduced scatter plot of states (if PCA was used)
    ax4 = plt.subplot(2, 3, 4)
    
    try:
        from sklearn.decomposition import PCA
        
        # Apply PCA to micro states for visualization
        if S_history.shape[1] > 2:
            pca = PCA(n_components=2)
            S_reduced = pca.fit_transform(S_history)
            
            # Color points by time
            time_norm = (np.arange(S_history.shape[0]) / S_history.shape[0])
            scatter = ax4.scatter(S_reduced[:, 0], S_reduced[:, 1], c=time_norm, cmap='viridis', 
                                 s=10, alpha=0.7)
            
            plt.colorbar(scatter, ax=ax4, label='Time')
            ax4.set_title('PCA of Micro States', fontsize=14)
            ax4.set_xlabel('PC1')
            ax4.set_ylabel('PC2')
            ax4.text(0.05, 0.95, f'Variance Explained: {pca.explained_variance_ratio_.sum():.2f}',
                     transform=ax4.transAxes, fontsize=10, va='top')
        else:
            ax4.text(0.5, 0.5, 'Micro state dimension too small for PCA', 
                    ha='center', va='center', transform=ax4.transAxes)
    except:
        ax4.text(0.5, 0.5, 'PCA visualization unavailable', 
                ha='center', va='center', transform=ax4.transAxes)
    
    # 5. Emergence interpretation
    ax5 = plt.subplot(2, 3, (5, 6))
    
    emergence_text = "Emergence Interpretation:\n\n"
    
    delta = metrics.get('Delta', 0)
    gamma = metrics.get('Gamma', 0)
    psi = metrics.get('Psi', 0)
    delta_r = metrics.get('Delta_reconciling', 0) if 'Delta_reconciling' in metrics else None
    
    # Delta interpretation
    emergence_text += f"• Delta = {delta:.4f}: "
    if delta > 0.5:
        emergence_text += "Strong downward causation - macro states significantly predict future macro states.\n"
    elif delta > 0.1:
        emergence_text += "Moderate downward causation is present.\n"
    else:
        emergence_text += "Limited downward causation detected.\n"
    
    # Gamma interpretation
    emergence_text += f"• Gamma = {gamma:.4f}: "
    if gamma > 0.5:
        emergence_text += "Strong causal decoupling - macro states provide better future prediction than micro states.\n"
    elif gamma > 0.1:
        emergence_text += "Moderate causal decoupling is present.\n"
    else:
        emergence_text += "Limited causal decoupling detected.\n"
    
    # Psi interpretation
    emergence_text += f"• Psi = {psi:.4f}: "
    if psi > 0.5:
        emergence_text += "Strong emergence - macro variables provide significant additional predictive power.\n"
    elif psi > 0.1:
        emergence_text += "Moderate emergence is present.\n"
    else:
        emergence_text += "Limited emergence detected.\n"
    
    # Overall interpretation
    emergence_text += "\nOverall: "
    if max(delta, gamma, psi, 0 if delta_r is None else delta_r) > 0.5:
        emergence_text += "This system shows STRONG causal emergence properties."
    elif max(delta, gamma, psi, 0 if delta_r is None else delta_r) > 0.1:
        emergence_text += "This system shows MODERATE causal emergence properties."
    else:
        emergence_text += "This system shows LIMITED causal emergence properties."
    
    # Add Reconciling metrics interpretation if available
    if delta_r is not None:
        emergence_text += "\n\nReconciling metrics analysis suggests that "
        if delta_r > delta:
            emergence_text += "specific variable-level interactions show stronger emergence than the system-wide average."
        else:
            emergence_text += "emergence is distributed throughout the system rather than concentrated in specific variable interactions."
    
    ax5.text(0.05, 0.95, emergence_text, transform=ax5.transAxes, 
            va='top', ha='left', fontsize=12)
    ax5.axis('off')
    
    # Main title
    plt.suptitle(f'Emergence Analysis for {simulation_type.upper()} Simulation', fontsize=18)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

# Add to main execution section
if __name__ == "__main__":
    # ...existing code...
    
    # Add after metrics calculation
    if metrics:
        # ...existing code for printing metrics...
        
        # Add visualization
        print("\nGenerating visualization of emergence metrics...")
        fig = visualize_emergence_metrics(metrics, S_history, M_history, SIMULATION_TYPE)
        
        # Save the figure if a directory is specified
        save_dir = os.environ.get('EMERGENCE_SAVE_DIR')
        if save_dir:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            fig_path = os.path.join(save_dir, f"{SIMULATION_TYPE}_emergence_analysis.png")
            fig.savefig(fig_path, dpi=300)
            print(f"Visualization saved to {fig_path}")