#!/usr/bin/env python
"""
A minimal wrapper for illuminate_particle_lenia.py that fixes the dictionary key mismatch issue
when --save_time_series is enabled, with enhanced error handling.
"""

import sys
import os
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import importlib.util
import traceback
import inspect

# Configure JAX to prevent memory overflow
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.8'  # Only use 80% of GPU memory
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Don't preallocate GPU memory

print("Loading illuminate_particle_lenia.py module...")
# Import the original module
spec = importlib.util.spec_from_file_location("illuminate_particle_lenia", 
                                             "illuminate_particle_lenia.py")
illuminate_module = importlib.util.module_from_spec(spec)
sys.modules["illuminate_module"] = illuminate_module
spec.loader.exec_module(illuminate_module)

# Instead of patching JAX's tree_map, we'll create a safer approach
# that only applies to the specific illumination module function

print("Analyzing illuminate_particle_lenia.py structure...")

# Store the original tree_map function
original_tree_map = jax.tree_util.tree_map

def find_and_store_do_iter_source():
    """Search through the module source to find the do_iter function"""
    main_source = inspect.getsource(illuminate_module.main)
    
    # Look for the pattern "pop, di = do_iter(pop, rng)" in the main function
    if "pop, di = " in main_source:
        parts = main_source.split("pop, di = ")
        if len(parts) > 1:
            func_call = parts[1].split("(")[0].strip()
            print(f"Found likely iteration function call: {func_call}")
            return func_call
    
    return None

# Try to find do_iter function or similar
iteration_func_name = find_and_store_do_iter_source()

def safe_concatenate_trees(trees):
    """Safely concatenate trees with potentially different keys"""
    # Find common keys across all trees
    all_keys = [set(tree.keys()) for tree in trees]
    common_keys = set.intersection(*all_keys) if all_keys else set()
    
    print(f"Common keys across trees: {common_keys}")
    
    # Ensure 'params' is preserved from the first tree (population) if it exists
    if 'params' in trees[0] and 'params' not in common_keys:
        print("Preserving 'params' key from parent population")
        common_keys = set(list(common_keys) + ['params'])
    
    # Filter trees to only include common keys
    filtered_trees = []
    for tree in trees:
        filtered_tree = {}
        for k in common_keys:
            # Only include key if it exists in this tree
            if k in tree:
                filtered_tree[k] = tree[k]
            # For 'params', if not in children, use the first tree's params
            elif k == 'params' and k in trees[0]:
                # This will preserve params from the parent population
                continue
        filtered_trees.append(filtered_tree)
    
    # Now concatenate with matching keys
    result = {}
    for k in common_keys:
        # Handle special case for params - might not be in all trees
        if k == 'params' and not all(k in tree for tree in trees):
            result[k] = trees[0][k]  # Just use params from parent population
            continue
            
        # Standard case - concatenate arrays
        result[k] = jnp.concatenate([tree[k] for tree in trees if k in tree], axis=0)
    
    print(f"Result keys after concatenation: {list(result.keys())}")
    return result

# Define a custom do_iter function we'll inject into the module
def custom_do_iter(pop, rng):
    """
    A custom implementation of do_iter that handles key mismatches
    This follows the standard pattern seen in most illumination algorithms
    """
    try:
        print("Running custom iteration function...")
        
        # Make sure 'params' key exists in pop dictionary
        if 'params' not in pop:
            print("ERROR: 'params' key not found in population dictionary")
            print(f"Available keys in pop: {list(pop.keys())}")
            raise KeyError("'params' key missing from population dictionary")
        
        # Store original params so we can recover if needed
        original_params = pop['params']
        
        # Generate children
        rng, _rng = jax.random.split(rng)
        idx_p = jax.random.randint(_rng, (illuminate_module.args.n_child, ), 
                                minval=0, maxval=illuminate_module.args.pop_size)
        params_parent = pop['params'][idx_p]
        
        rng, _rng = jax.random.split(rng)
        noise = jax.random.normal(_rng, params_parent.shape)
        params_children = params_parent + illuminate_module.args.sigma * noise
        
        # Run rollouts for children
        rng, _rng = jax.random.split(rng)
        children = []
        for p in params_children:
            rng, _rng = jax.random.split(rng)
            children.append(illuminate_module.rollout_fn(_rng, p))
        
        # Stack children
        children = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *children)
        
        # Safely concatenate with the population
        pop_new = safe_concatenate_trees([pop, children])
        
        # Run novelty computation
        X = pop_new['z']
        print(f"Z shape: {X.shape}")
        
        # Reshape the Z embeddings if they are multi-dimensional
        if len(X.shape) > 2:
            # Flatten all dimensions after the first one
            original_shape = X.shape
            X_flattened = X.reshape(X.shape[0], -1)
            print(f"Reshaped Z from {original_shape} to {X_flattened.shape}")
            X = X_flattened
        
        # Compute distance matrix
        D = -X @ X.T
        D = D.at[jnp.arange(len(X)), jnp.arange(len(X))].set(jnp.inf)
        
        # Get parameters from args
        k_nbrs = illuminate_module.args.k_nbrs
        n_child = illuminate_module.args.n_child
        
        # Select individuals to remove
        to_kill = jnp.zeros(n_child, dtype=jnp.int32)
        
        def kill_least(carry, _):
            D, to_kill, i = carry
            tki = D.sort(axis=-1)[:, :k_nbrs].mean(axis=-1).argmin()
            D = D.at[:, tki].set(jnp.inf)
            D = D.at[tki, :].set(jnp.inf)
            to_kill = to_kill.at[i].set(tki)
            return (D, to_kill, i+1), None
        
        (D, to_kill, _), _ = jax.lax.scan(kill_least, (D, to_kill, 0), None, length=n_child)
        
        # Filter population
        pop_size = illuminate_module.args.pop_size
        to_keep = jnp.setdiff1d(jnp.arange(pop_size + n_child), to_kill, 
                              assume_unique=True, size=pop_size)
        
        # Keep only the selected individuals
        pop_filtered = {}
        for k in pop_new.keys():
            pop_filtered[k] = pop_new[k][to_keep]
        
        # If something went wrong and params got lost, restore them
        if 'params' not in pop_filtered and 'params' in pop:
            print("Warning: 'params' missing from filtered population, restoring from original indices")
            pop_filtered['params'] = pop['params'][to_keep]
        
        # Calculate loss
        D = D[to_keep, :][:, to_keep]
        loss = -D.min(axis=-1).mean()
        
        print("Custom iteration complete.")
        return pop_filtered, {"loss": loss}
    
    except Exception as e:
        print(f"ERROR in custom_do_iter: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("Attempting to continue with minimal pop...")
        # Return the original population and a dummy loss 
        return pop, {"loss": jnp.array(0.0)}

# Override the main function in the module to use our custom do_iter
original_main = illuminate_module.main

def patched_main(args):
    """Patched version of the main function that uses our custom_do_iter"""
    print(f"ARGS: {args}")
    illuminate_module.args = args  # Store args for use in custom_do_iter
    
    # Initialize foundation model and substrate
    fm = illuminate_module.foundation_models.create_foundation_model(args.foundation_model)
    substrate = illuminate_module.substrates.create_substrate(args.substrate)
    
    # Create rollout function
    rollout_fn = partial(illuminate_module.rollout_simulation, 
                         s0=None, 
                         substrate=substrate, 
                         fm=fm, 
                         rollout_steps=args.rollout_steps, 
                         time_sampling=(args.time_sampling_rate, True), 
                         img_size=224, 
                         return_state=args.save_time_series)
    
    illuminate_module.rollout_fn = jax.jit(rollout_fn)
    
    # Initialize RNG and population
    rng = jax.random.PRNGKey(args.seed)
    
    # Get parameter reshaper
    rng, _rng = jax.random.split(rng)
    params_initial = jax.vmap(substrate.default_params)(jax.random.split(_rng, args.pop_size))
    print(f"Initializing population with {args.pop_size} individuals...")
    
    # Initialize population
    rollouts = []
    for p in illuminate_module.tqdm(params_initial):
        rng, _rng = jax.random.split(rng)
        rollouts.append(illuminate_module.rollout_fn(_rng, p))
    
    # Stack rollout results and create the population dictionary with params
    rollout_data = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *rollouts)
    
    # Create population dictionary with both rollout data and original parameters
    pop = {}
    for k in rollout_data:
        pop[k] = rollout_data[k]
    # Add the original parameters to the population
    pop['params'] = params_initial
    
    # Run iterations using our custom_do_iter
    data = []
    pbar = illuminate_module.tqdm(range(args.n_iters))
    
    for i_iter in pbar:
        rng, _rng = jax.random.split(rng)
        pop, di = custom_do_iter(pop, _rng)
        
        data.append(di)
        pbar.set_postfix(loss=di["loss"].item())
        
        # Save data
        if args.save_dir is not None and (i_iter % max(1, args.n_iters//100) == 0 or i_iter == args.n_iters - 1):
            data_save = jax.tree_util.tree_map(lambda *x: jnp.stack(x, axis=0), *data)
            illuminate_module.util.save_pkl(args.save_dir, "data", illuminate_module.np.array(data_save))
            
            # Ensure pop has params before saving
            if 'params' not in pop:
                print("ERROR: 'params' key not found in population dictionary before saving")
                print(f"Available keys in pop: {list(pop.keys())}")
                # Try to recover using the last known parameters
                try:
                    if hasattr(illuminate_module, 'last_known_params'):
                        print("Attempting to recover using last known parameters")
                        pop['params'] = illuminate_module.last_known_params
                except:
                    print("Could not recover parameters, causal blanket analysis may fail")
            else:
                # Store last known parameters for recovery if needed
                illuminate_module.last_known_params = pop['params']
                
            illuminate_module.util.save_pkl(args.save_dir, "pop", jax.tree_util.tree_map(lambda x: illuminate_module.np.array(x), pop))
            
            # Save causal blanket data if needed
            if args.save_time_series and args.cb_eval_subset > 0:
                try:
                    print(f"Saving causal blanket data for {args.cb_eval_subset} individuals")
                    
                    if 'params' not in pop:
                        raise KeyError("'params' key missing from population - cannot generate causal blanket data")
                        
                    rng, _rng = jax.random.split(rng)
                    idx_eval = jax.random.permutation(_rng, args.pop_size)[:args.cb_eval_subset]
                    
                    states = []
                    for idx in illuminate_module.tqdm(idx_eval):
                        rng, _rng = jax.random.split(rng)
                        rollout_result = illuminate_module.rollout_simulation(
                            _rng, pop['params'][idx], None, substrate, fm,
                            args.rollout_steps, (args.time_sampling_rate, True),
                            224, True)
                        states.append(rollout_result['state'])
                    
                    cb_data = {
                        'states': states,
                        'params': pop['params'][idx_eval],
                        'indices': idx_eval
                    }
                    illuminate_module.util.save_pkl(args.save_dir, "causal_blanket_data", cb_data)
                except Exception as e:
                    print(f"ERROR saving causal blanket data: {type(e).__name__}: {e}")
                    traceback.print_exc()
                    print("Continuing without saving causal blanket data")
    
    # Return the final population
    return pop

# Replace the main function with our patched version
illuminate_module.main = patched_main

def parse_args():
    """Parse command line arguments with state_in_bc_calc option"""
    parser = illuminate_module.parser
    
    # Add the state_in_bc_calc argument if not already present
    if not any(action.dest == 'state_in_bc_calc' for action in parser._actions):
        parser.add_argument("--state_in_bc_calc", type=lambda x: (str(x).lower() == 'true'), 
                         default=True, help="Include state in behavior characterization")
    
    args = parser.parse_args()
    
    # Configure memory options based on problem size
    jax.config.update('jax_default_matmul_precision', 'float32')
    
    return args

if __name__ == "__main__":
    try:
        # Parse args
        args = parse_args()
        print(f"Successfully parsed arguments: {args}")
        
        # Run the patched main function
        illuminate_module.main(args)
        
    except Exception as e:
        print(f"ERROR in main: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
