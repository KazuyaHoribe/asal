"""
Visualization utilities for emergence metrics and causal analysis.

This module provides functions to create various visualizations for understanding
emergence phenomena in simulation data, with a focus on causal emergence metrics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os
import warnings
from typing import Dict, Optional, List, Tuple, Union, Any
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# Suppress non-critical warnings during visualization
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# Define custom color schemes for emergence visualizations
EMERGENCE_COLORS = {
    'delta': '#1f77b4',    # blue
    'gamma': '#2ca02c',    # green
    'psi': '#d62728',      # red
    'delta_r': '#9467bd',  # purple
    'gamma_r': '#8c564b',  # brown
    'psi_r': '#e377c2',    # pink
    'background': '#f5f5f5',
    'grid': '#cccccc'
}

def create_dashboard(metrics: Dict[str, float], 
                    S_history: np.ndarray,
                    M_history: np.ndarray, 
                    simulation_type: str,
                    simulation_images: Optional[np.ndarray] = None,
                    show_pca: bool = True,
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a comprehensive dashboard visualization of emergence metrics and simulation data.
    
    Args:
        metrics: Dictionary containing standard and/or reconciling emergence metrics
        S_history: Micro state history array (T, n_micro_dims)
        M_history: Macro state history array (T, n_macro_dims)
        simulation_type: Type of simulation ('gol', 'boids', etc.)
        simulation_images: Optional array of simulation renderings at different timesteps
        show_pca: Whether to show PCA visualization of state trajectories
        save_path: Optional path to save the dashboard image
        
    Returns:
        matplotlib Figure object of the dashboard
    """
    # Create figure with grid layout
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1.2])
    
    # Apply overall style - Updated for newer matplotlib
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
            
    fig.patch.set_facecolor(EMERGENCE_COLORS['background'])
    
    # 1. Metrics Bar Chart
    ax_metrics = fig.add_subplot(gs[0, 0])
    plot_metrics_comparison(metrics, ax=ax_metrics)
    
    # 2. Macro State Evolution
    ax_macro = fig.add_subplot(gs[0, 1:])
    plot_macro_evolution(M_history, ax=ax_macro)
    
    # 3. Metrics Radar Chart
    ax_radar = fig.add_subplot(gs[1, 0], polar=True)
    plot_metrics_radar(metrics, ax=ax_radar)
    
    # 4. State Space Visualization (PCA or direct)
    ax_states = fig.add_subplot(gs[1, 1:])
    if show_pca:
        plot_state_space(S_history, M_history, ax=ax_states)
    else:
        # Alternative visualization if PCA not desired
        plot_micro_macro_correlation(S_history, M_history, ax=ax_states)
    
    # 5. Interpretation Text
    ax_interp = fig.add_subplot(gs[2, :2])
    plot_interpretation(metrics, simulation_type, ax=ax_interp)
    
    # 6. Simulation Snapshots (if available)
    ax_sim = fig.add_subplot(gs[2, 2])
    if simulation_images is not None:
        plot_simulation_snapshots(simulation_images, ax=ax_sim)
    else:
        ax_sim.text(0.5, 0.5, "Simulation images not available", 
                  ha='center', va='center', fontsize=14)
        ax_sim.axis('off')
    
    # Overall title
    fig.suptitle(f'Emergence Analysis Dashboard: {simulation_type.upper()}', 
               fontsize=24, y=0.98)
    
    # Adjust layout
    try:
        plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    except Warning:
        # If tight layout causes warnings, use a simpler layout
        plt.tight_layout()
    
    # Save if path specified
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Dashboard saved to {save_path}")
    
    return fig

def plot_metrics_comparison(metrics: Dict[str, float], ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create a bar chart comparing standard and reconciling metrics.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Determine if we have reconciling metrics
    has_reconciling = all(f"{m}_reconciling" in metrics for m in ["Delta", "Gamma", "Psi"])
    
    if has_reconciling:
        # Data for comparison
        metrics_labels = ['Delta', 'Gamma', 'Psi']
        standard_values = [metrics[m] for m in metrics_labels]
        reconciling_values = [metrics[f"{m}_reconciling"] for m in metrics_labels]
        
        x = np.arange(len(metrics_labels))
        width = 0.35
        
        # Create bars
        ax.bar(x - width/2, standard_values, width, label='Standard', 
             color=[EMERGENCE_COLORS['delta'], EMERGENCE_COLORS['gamma'], EMERGENCE_COLORS['psi']])
        ax.bar(x + width/2, reconciling_values, width, label='Reconciling', 
             color=[EMERGENCE_COLORS['delta_r'], EMERGENCE_COLORS['gamma_r'], EMERGENCE_COLORS['psi_r']])
        
        # Customize
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels)
    else:
        # Simple bar chart for standard metrics only
        metrics_labels = ['Delta', 'Gamma', 'Psi']
        values = [metrics.get(m, 0) for m in metrics_labels]
        
        x = np.arange(len(metrics_labels))
        
        # Create bars
        ax.bar(x, values, color=[EMERGENCE_COLORS['delta'], EMERGENCE_COLORS['gamma'], EMERGENCE_COLORS['psi']])
        
        # Customize
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels)
    
    # Add labels and legend
    ax.set_title('Emergence Metrics Comparison', fontsize=14, pad=10)
    ax.set_ylabel('Value', fontsize=12)
    if has_reconciling:
        ax.legend(fontsize=10)
    
    # Add value labels on top of bars
    for i, v in enumerate(standard_values if has_reconciling else values):
        offset = -width/2 if has_reconciling else 0
        ax.text(i + offset, v + 0.02, f"{v:.3f}", ha='center', fontsize=9)
    
    if has_reconciling:
        for i, v in enumerate(reconciling_values):
            ax.text(i + width/2, v + 0.02, f"{v:.3f}", ha='center', fontsize=9)
    
    # Set y limits with some padding
    max_val = max(standard_values + (reconciling_values if has_reconciling else []))
    ax.set_ylim(0, max_val * 1.2)
    
    # Grid only on y-axis
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    return ax

def plot_macro_evolution(M_history: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot the evolution of macro variables over time.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get number of macro variables
    n_macro_vars = M_history.shape[1]
    time_steps = np.arange(M_history.shape[0])
    
    # Create color map
    cmap = plt.cm.viridis
    colors = [cmap(i/max(1, n_macro_vars-1)) for i in range(n_macro_vars)]
    
    # Plot each macro variable
    for i in range(n_macro_vars):
        ax.plot(time_steps, M_history[:, i], label=f'Macro {i+1}', 
              linewidth=2, color=colors[i])
    
    # Add a light background to highlight time evolution
    ax.fill_between(time_steps, np.min(M_history) * 0.9, np.max(M_history) * 1.1, 
                   color='gray', alpha=0.1)
    
    # Customize
    ax.set_title('Macro State Evolution', fontsize=14)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Macro Value', fontsize=12)
    
    # Add legend if there are multiple variables
    if n_macro_vars > 1:
        ax.legend(fontsize=10)
    
    # Grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return ax

def plot_metrics_radar(metrics: Dict[str, float], ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create a radar chart of emergence metrics.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Determine if we have reconciling metrics
    has_reconciling = all(f"{m}_reconciling" in metrics for m in ["Delta", "Gamma", "Psi"])
    
    # Prepare data
    categories = ['Delta', 'Gamma', 'Psi']
    N = len(categories)
    
    # Scale factor (for visibility, ensure values are at least 0.05)
    scale_factor = 1.0
    
    # Values for standard metrics
    values_std = [max(0.05, metrics.get(cat, 0)) for cat in categories]
    values_std = np.concatenate((values_std, [values_std[0]]))  # Close the loop
    
    # Angle for each category
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Plot standard metrics
    ax.plot(angles, values_std, 'o-', linewidth=2, label='Standard', color='#1f77b4')
    ax.fill(angles, values_std, alpha=0.25, color='#1f77b4')
    
    if has_reconciling:
        # Values for reconciling metrics
        values_rec = [max(0.05, metrics.get(f"{cat}_reconciling", 0)) for cat in categories]
        values_rec = np.concatenate((values_rec, [values_rec[0]]))  # Close the loop
        
        # Plot reconciling metrics
        ax.plot(angles, values_rec, 'o-', linewidth=2, label='Reconciling', color='#ff7f0e')
        ax.fill(angles, values_rec, alpha=0.25, color='#ff7f0e')
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    
    # Remove radial labels and set limits
    ax.set_yticklabels([])
    max_val = max(np.max(values_std), np.max(values_rec) if has_reconciling else 0)
    ax.set_ylim(0, max_val * 1.2)
    
    # Add subtle grid
    ax.grid(True, alpha=0.3)
    
    # Title and legend
    ax.set_title('Emergence Metrics Radar', fontsize=14, y=1.08)
    if has_reconciling:
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.2), fontsize=10)
    
    # Add value annotations
    for i, angle in enumerate(angles[:-1]):
        if i < len(categories):
            std_val = values_std[i]
            ax.text(angle, std_val + 0.1, f"{std_val:.2f}", 
                  ha='center', va='center', fontsize=9, color='#1f77b4')
            
            if has_reconciling and i < len(values_rec) - 1:
                rec_val = values_rec[i]
                ax.text(angle, rec_val + 0.1, f"{rec_val:.2f}", 
                      ha='center', va='center', fontsize=9, color='#ff7f0e')
    
    return ax

def plot_state_space(S_history: np.ndarray, M_history: np.ndarray, 
                    ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Visualize the state space using PCA reduction.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Apply PCA reduction if dimensionality is high
    if S_history.shape[1] > 2:
        try:
            from sklearn.decomposition import PCA
            
            # Reduce to 2 components
            pca = PCA(n_components=2)
            S_reduced = pca.fit_transform(S_history)
            
            # Get variance explained
            var_explained = pca.explained_variance_ratio_
            
            # Plot with time-based coloring
            points = ax.scatter(S_reduced[:, 0], S_reduced[:, 1], 
                              c=np.arange(S_history.shape[0]), 
                              cmap='viridis', s=30, alpha=0.7)
            
            # Add colorbar for time
            cbar = plt.colorbar(points, ax=ax)
            cbar.set_label('Time Step', fontsize=10)
            
            # Add connecting lines to show trajectory
            ax.plot(S_reduced[:, 0], S_reduced[:, 1], 'k-', alpha=0.2, linewidth=0.5)
            
            # Highlight start and end points
            ax.scatter(S_reduced[0, 0], S_reduced[0, 1], c='green', s=100, 
                     label='Start', edgecolors='black', zorder=5)
            ax.scatter(S_reduced[-1, 0], S_reduced[-1, 1], c='red', s=100, 
                     label='End', edgecolors='black', zorder=5)
            
            # Add labels
            ax.set_xlabel(f'PC1 ({var_explained[0]:.2%} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({var_explained[1]:.2%} variance)', fontsize=12)
            ax.set_title('Micro State Space (PCA)', fontsize=14)
            ax.legend(fontsize=10)
            
            # Equal aspect ratio for better visualization
            ax.set_aspect('equal', adjustable='box')
            
            # Mark key positions where macro state changes significantly
            try:
                # Calculate significant changes in macro state
                macro_changes = np.sqrt(np.sum(np.diff(M_history, axis=0)**2, axis=1))
                thres = np.mean(macro_changes) + np.std(macro_changes)
                change_points = np.where(macro_changes > thres)[0]
                
                # Highlight these points
                if len(change_points) > 0:
                    ax.scatter(S_reduced[change_points, 0], S_reduced[change_points, 1], 
                             facecolors='none', edgecolors='orange', s=80, linewidth=2,
                             label='Macro Change', zorder=4)
                    ax.legend(fontsize=10)
            except:
                pass  # Skip if error in macro change detection
            
        except Exception as e:
            ax.text(0.5, 0.5, f"PCA visualization failed: {e}", 
                  ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    else:
        # Direct plot for 1D or 2D data
        if S_history.shape[1] == 1:
            time_steps = np.arange(S_history.shape[0])
            ax.plot(time_steps, S_history[:, 0], '-o', markersize=3)
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('Micro State Value', fontsize=12)
            ax.set_title('Micro State Evolution (1D)', fontsize=14)
        elif S_history.shape[1] == 2:
            points = ax.scatter(S_history[:, 0], S_history[:, 1], 
                              c=np.arange(S_history.shape[0]), 
                              cmap='viridis', s=30, alpha=0.7)
            ax.plot(S_history[:, 0], S_history[:, 1], 'k-', alpha=0.2, linewidth=0.5)
            cbar = plt.colorbar(points, ax=ax)
            cbar.set_label('Time Step', fontsize=10)
            ax.set_xlabel('Dimension 1', fontsize=12)
            ax.set_ylabel('Dimension 2', fontsize=12)
            ax.set_title('Micro State Space (2D)', fontsize=14)
    
    return ax

def plot_micro_macro_correlation(S_history: np.ndarray, M_history: np.ndarray, 
                                ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Visualize the correlation between micro and macro states.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate correlations between micro and macro variables
    n_micro = min(20, S_history.shape[1])  # Limit to 20 variables for visualization
    n_macro = M_history.shape[1]
    
    # Use a subset of micro variables if there are too many
    S_subset = S_history[:, :n_micro]
    
    # Calculate correlation matrix
    corr_matrix = np.zeros((n_macro, n_micro))
    for i in range(n_macro):
        for j in range(n_micro):
            corr_matrix[i, j] = np.corrcoef(M_history[:, i], S_subset[:, j])[0, 1]
    
    # Plot heatmap
    im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    
    # Add labels
    ax.set_xlabel('Micro Variables', fontsize=12)
    ax.set_ylabel('Macro Variables', fontsize=12)
    ax.set_title('Micro-Macro Correlation', fontsize=14)
    
    # Set ticks
    ax.set_xticks(np.arange(n_micro))
    ax.set_yticks(np.arange(n_macro))
    ax.set_xticklabels([f'μ{i+1}' for i in range(n_micro)], fontsize=8, rotation=90)
    ax.set_yticklabels([f'M{i+1}' for i in range(n_macro)], fontsize=10)
    
    # Add correlation values
    for i in range(n_macro):
        for j in range(n_micro):
            text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
            ax.text(j, i, f"{corr_matrix[i, j]:.2f}", ha="center", va="center", 
                  color=text_color, fontsize=7)
    
    return ax

def plot_interpretation(metrics: Dict[str, float], simulation_type: str, 
                       ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Create a textual interpretation of the emergence metrics.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 10))
    
    # Extract metrics
    delta = metrics.get('Delta', 0)
    gamma = metrics.get('Gamma', 0)
    psi = metrics.get('Psi', 0)
    delta_r = metrics.get('Delta_reconciling')
    gamma_r = metrics.get('Gamma_reconciling')
    psi_r = metrics.get('Psi_reconciling')
    
    # Create detailed interpretations
    interpretation_text = []
    
    # Title and introduction
    interpretation_text.append(f"Emergence Analysis for {simulation_type.upper()} Simulation")
    interpretation_text.append("=" * 50)
    interpretation_text.append("\nInterpretation of Causal Emergence Metrics:\n")
    
    # Delta interpretation
    interpretation_text.append("DOWNWARD CAUSATION (Delta):")
    interpretation_text.append(f"  • Standard Delta: {delta:.4f}")
    if delta_r is not None:
        interpretation_text.append(f"  • Reconciling Delta: {delta_r:.4f}")
    
    if delta > 0.5:
        level = "STRONG"
    elif delta > 0.1:
        level = "MODERATE"
    else:
        level = "WEAK"
        
    interpretation_text.append(f"  • Level: {level} downward causation")
    interpretation_text.append("  • Interpretation: " + {
        "STRONG": "Macro states strongly determine future macro states in ways not reducible to micro state interactions.",
        "MODERATE": "Macro states show some influence on future macro dynamics beyond what micro states account for.",
        "WEAK": "Macro state dynamics are largely reducible to underlying micro state interactions."
    }[level])
    
    # Gamma interpretation
    interpretation_text.append("\nCAUSAL DECOUPLING (Gamma):")
    interpretation_text.append(f"  • Standard Gamma: {gamma:.4f}")
    if gamma_r is not None:
        interpretation_text.append(f"  • Reconciling Gamma: {gamma_r:.4f}")
    
    if gamma > 0.5:
        level = "STRONG"
    elif gamma > 0.1:
        level = "MODERATE"
    else:
        level = "WEAK"
        
    interpretation_text.append(f"  • Level: {level} causal decoupling")
    interpretation_text.append("  • Interpretation: " + {
        "STRONG": "Macro variables provide significantly better predictions of future micro states than past micro states.",
        "MODERATE": "Macro states offer some predictive advantage over micro states for future micro dynamics.",
        "WEAK": "Micro state prediction is not improved by considering macro state information."
    }[level])
    
    # Psi interpretation
    interpretation_text.append("\nCAUSAL EMERGENCE (Psi):")
    interpretation_text.append(f"  • Standard Psi: {psi:.4f}")
    if psi_r is not None:
        interpretation_text.append(f"  • Reconciling Psi: {psi_r:.4f}")
    
    if psi > 0.5:
        level = "STRONG"
    elif psi > 0.1:
        level = "MODERATE"
    else:
        level = "WEAK"
        
    interpretation_text.append(f"  • Level: {level} causal emergence")
    interpretation_text.append("  • Interpretation: " + {
        "STRONG": "Macro variables provide substantial additional information about future micro states beyond what past micro states tell us.",
        "MODERATE": "Macro variables offer some unique information about future micro dynamics not contained in past micro states.",
        "WEAK": "Macro variables contain little to no additional information beyond what's available in the micro states."
    }[level])
    
    # Standard vs. Reconciling comparison if applicable
    if all(x is not None for x in [delta_r, gamma_r, psi_r]):
        interpretation_text.append("\nSTANDARD vs. RECONCILING COMPARISON:")
        
        if delta_r > delta:
            interpretation_text.append("  • Delta: Reconciling > Standard - Specific micro variables show stronger downward causation")
        else:
            interpretation_text.append("  • Delta: Standard > Reconciling - Downward causation is more uniform across variables")
            
        if gamma_r > gamma:
            interpretation_text.append("  • Gamma: Reconciling > Standard - Some micro variables are particularly well predicted by macro")
        else:
            interpretation_text.append("  • Gamma: Standard > Reconciling - Macro prediction advantage is distributed evenly")
            
        if psi_r > psi:
            interpretation_text.append("  • Psi: Reconciling > Standard - Information gain from macro is concentrated in specific micro variables")
        else:
            interpretation_text.append("  • Psi: Standard > Reconciling - Information gain from macro is spread across micro variables")
    
    # Overall conclusion
    interpretation_text.append("\nOVERALL CONCLUSION:")
    max_metric = max(delta, gamma, psi, delta_r or 0, gamma_r or 0, psi_r or 0)
    
    if max_metric > 0.5:
        conclusion = f"This {simulation_type} system exhibits STRONG causal emergence properties."
        if max(delta, delta_r or 0) > 0.5:
            conclusion += " Downward causation is particularly significant."
        if max(gamma, gamma_r or 0) > 0.5:
            conclusion += " Macro variables show substantial predictive advantage."
        if max(psi, psi_r or 0) > 0.5:
            conclusion += " Novel information at the macro level is clearly present."
    elif max_metric > 0.1:
        conclusion = f"This {simulation_type} system exhibits MODERATE causal emergence properties."
        if max(delta, delta_r or 0) > 0.1:
            conclusion += " Some downward causation is present."
        if max(gamma, gamma_r or 0) > 0.1:
            conclusion += " Macro variables show some predictive advantage."
        if max(psi, psi_r or 0) > 0.1:
            conclusion += " Some novel information at the macro level is present."
    else:
        conclusion = f"This {simulation_type} system exhibits WEAK causal emergence properties."
        conclusion += " The system behavior is largely reducible to micro-level dynamics."
    
    interpretation_text.append("  " + conclusion)
    
    # Render the full interpretation
    ax.text(0.01, 0.99, "\n".join(interpretation_text), transform=ax.transAxes,
          va='top', ha='left', fontsize=12, linespacing=1.5)
    ax.axis('off')
    
    return ax

def plot_simulation_snapshots(images: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot snapshots from the simulation at different time points.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # If images is 4D (T, H, W, C), select key frames
    if images.ndim == 4:
        T, H, W, C = images.shape
        
        # Select frames at regular intervals
        n_frames = min(9, T)
        frame_indices = np.linspace(0, T-1, n_frames).astype(int)
        
        # Create a grid of images
        grid_size = int(np.ceil(np.sqrt(n_frames)))
        grid_image = np.ones((grid_size*H, grid_size*W, C)) * 0.9  # Light gray background
        
        for i, idx in enumerate(frame_indices):
            row = i // grid_size
            col = i % grid_size
            grid_image[row*H:(row+1)*H, col*W:(col+1)*W] = images[idx]
        
        # Plot the grid
        ax.imshow(grid_image)
        
        # Add time indicators
        for i, idx in enumerate(frame_indices):
            row = i // grid_size
            col = i % grid_size
            ax.text(col*W + W//2, row*H + 20, f"t={idx}", 
                  ha='center', va='center', color='white', 
                  bbox=dict(boxstyle="round,pad=0.3", fc='black', alpha=0.7))
        
        ax.set_title('Simulation Snapshots', fontsize=14)
    else:
        # Handle case where images format is different
        ax.text(0.5, 0.5, "Image data format not supported", 
              ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    return ax

def plot_interactive_dashboard(metrics, S_history, M_history, simulation_type, save_dir=None):
    """
    Create and save an interactive HTML dashboard for emergence metrics.
    
    Args:
        metrics: Dictionary containing emergence metrics
        S_history: Micro state history
        M_history: Macro state history
        simulation_type: Type of simulation
        save_dir: Directory to save the HTML file
    
    Returns:
        Path to saved HTML file or None if not saved
    """
    try:
        # Check for required libraries
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
        import pandas as pd
        
        # Create a subplot figure
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                  [{"type": "polar"}, {"type": "scatter3d"}]],
            subplot_titles=("Emergence Metrics", "Macro State Evolution", 
                           "Metrics Radar", "Micro State Trajectory"),
            column_widths=[0.5, 0.5],
            row_heights=[0.5, 0.5]
        )
        
        # 1. Bar chart for metrics
        metrics_names = ['Delta', 'Gamma', 'Psi']
        standard_values = [metrics.get(m, 0) for m in metrics_names]
        
        if all(f"{m}_reconciling" in metrics for m in metrics_names):
            # Add reconciling metrics if available
            reconciling_values = [metrics.get(f"{m}_reconciling", 0) for m in metrics_names]
            fig.add_trace(
                go.Bar(
                    x=metrics_names,
                    y=standard_values,
                    name="Standard",
                    marker_color=['royalblue', 'green', 'red']
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(
                    x=metrics_names,
                    y=reconciling_values,
                    name="Reconciling",
                    marker_color=['darkblue', 'darkgreen', 'darkred']
                ),
                row=1, col=1
            )
        else:
            # Only standard metrics
            fig.add_trace(
                go.Bar(
                    x=metrics_names,
                    y=standard_values,
                    marker_color=['royalblue', 'green', 'red']
                ),
                row=1, col=1
            )
        
        # 2. Line chart for macro state evolution
        for i in range(M_history.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(M_history.shape[0]),
                    y=M_history[:, i],
                    mode='lines',
                    name=f'Macro Var {i+1}'
                ),
                row=1, col=2
            )
        
        # 3. Radar chart for metrics
        r_values = standard_values + [standard_values[0]]  # Close the loop
        theta_values = ['Delta', 'Gamma', 'Psi', 'Delta']  # Close the loop
        fig.add_trace(
            go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                fill='toself',
                name='Standard'
            ),
            row=2, col=1
        )
        
        if all(f"{m}_reconciling" in metrics for m in metrics_names):
            r_values_rec = reconciling_values + [reconciling_values[0]]
            fig.add_trace(
                go.Scatterpolar(
                    r=r_values_rec,
                    theta=theta_values,
                    fill='toself',
                    name='Reconciling'
                ),
                row=2, col=1
            )
        
        # 4. 3D scatter for micro state trajectory
        # Apply dimensionality reduction if needed
        if S_history.shape[1] > 3:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                S_reduced = pca.fit_transform(S_history)
                
                # Create a color scale based on time
                colors = np.arange(S_reduced.shape[0])
                
                fig.add_trace(
                    go.Scatter3d(
                        x=S_reduced[:, 0],
                        y=S_reduced[:, 1],
                        z=S_reduced[:, 2],
                        mode='markers+lines',
                        marker=dict(
                            size=4,
                            color=colors,
                            colorscale='Viridis',
                            opacity=0.8
                        ),
                        line=dict(
                            color='darkgray',
                            width=1
                        ),
                        name='Micro State'
                    ),
                    row=2, col=2
                )
            except:
                # Fallback to 2D if PCA fails
                pass
        
        # Update layout
        fig.update_layout(
            title_text=f"Emergence Analysis Dashboard: {simulation_type.upper()}",
            height=900,
            width=1200,
            template="plotly_white"
        )
        
        # Add interpretation text
        overall_level = "STRONG" if max(standard_values) > 0.5 else ("MODERATE" if max(standard_values) > 0.1 else "WEAK")
        interpretation = (
            f"<b>Emergence Level: {overall_level}</b><br>"
            f"Delta (Downward Causation): {metrics.get('Delta', 0):.4f}<br>"
            f"Gamma (Causal Decoupling): {metrics.get('Gamma', 0):.4f}<br>"
            f"Psi (Causal Emergence): {metrics.get('Psi', 0):.4f}"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0,
            text=interpretation,
            showarrow=False,
            font=dict(size=14),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="black",
            borderwidth=1,
            borderpad=10
        )
        
        # Save to HTML file if directory is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            html_path = os.path.join(save_dir, f"{simulation_type}_emergence_dashboard.html")
            fig.write_html(html_path)
            print(f"Interactive dashboard saved to {html_path}")
            return html_path
        else:
            return fig
    
    except ImportError:
        print("Interactive dashboard requires plotly. Install with 'pip install plotly'")
        return None
    except Exception as e:
        print(f"Error creating interactive dashboard: {e}")
        return None

def plot_interactive_dashboard(metrics, S_history, M_history, simulation_type, save_dir=None, template="plotly_white"):
    """
    Create and save an interactive HTML dashboard for emergence metrics.
    
    Args:
        metrics: Dictionary containing emergence metrics
        S_history: Micro state history
        M_history: Macro state history
        simulation_type: Type of simulation
        save_dir: Directory to save the HTML file
        template: Plotly template to use for styling
    
    Returns:
        Path to saved HTML file or None if not saved
    """
    try:
        # Check for required libraries
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.express as px
        import pandas as pd
        
        # Create a subplot figure with more panels
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "polar"}, {"type": "scatter3d"}],
                [{"type": "table", "colspan": 2}, None]
            ],
            subplot_titles=(
                "Emergence Metrics", "Macro State Evolution", 
                "Metrics Radar", "Micro State Trajectory",
                "Detailed Metrics Values"
            ),
            column_widths=[0.5, 0.5],
            row_heights=[0.4, 0.4, 0.2],
            vertical_spacing=0.1
        )
        
        # 1. Bar chart for metrics
        metrics_names = ['Delta', 'Gamma', 'Psi']
        standard_values = [metrics.get(m, 0) for m in metrics_names]
        
        if all(f"{m}_reconciling" in metrics for m in metrics_names):
            # Add reconciling metrics if available
            reconciling_values = [metrics.get(f"{m}_reconciling", 0) for m in metrics_names]
            fig.add_trace(
                go.Bar(
                    x=metrics_names,
                    y=standard_values,
                    name="Standard",
                    marker_color=['royalblue', 'green', 'red'],
                    hovertemplate='<b>%{x}</b>: %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(
                    x=metrics_names,
                    y=reconciling_values,
                    name="Reconciling",
                    marker_color=['darkblue', 'darkgreen', 'darkred'],
                    hovertemplate='<b>%{x} (Reconciling)</b>: %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )
        else:
            # Only standard metrics
            fig.add_trace(
                go.Bar(
                    x=metrics_names,
                    y=standard_values,
                    marker_color=['royalblue', 'green', 'red'],
                    hovertemplate='<b>%{x}</b>: %{y:.4f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 2. Line chart for macro state evolution with improved styling
        for i in range(M_history.shape[1]):
            fig.add_trace(
                go.Scatter(
                    x=np.arange(M_history.shape[0]),
                    y=M_history[:, i],
                    mode='lines',
                    name=f'Macro Var {i+1}',
                    line=dict(width=2),
                    hovertemplate='<b>Step</b>: %{x}<br><b>Value</b>: %{y:.4f}<extra>Macro Var %{fullData.name}</extra>'
                ),
                row=1, col=2
            )
        
        # 3. Radar chart for metrics with improved styling
        r_values = standard_values + [standard_values[0]]  # Close the loop
        theta_values = ['Delta', 'Gamma', 'Psi', 'Delta']  # Close the loop
        fig.add_trace(
            go.Scatterpolar(
                r=r_values,
                theta=theta_values,
                fill='toself',
                name='Standard',
                line=dict(color='royalblue', width=2),
                fillcolor='rgba(65, 105, 225, 0.2)',
                hovertemplate='<b>%{theta}</b>: %{r:.4f}<extra>Standard</extra>'
            ),
            row=2, col=1
        )
        
        if all(f"{m}_reconciling" in metrics for m in metrics_names):
            r_values_rec = [metrics.get(f"{m}_reconciling", 0) for m in metrics_names]
            r_values_rec = r_values_rec + [r_values_rec[0]]  # Close the loop
            fig.add_trace(
                go.Scatterpolar(
                    r=r_values_rec,
                    theta=theta_values,
                    fill='toself',
                    name='Reconciling',
                    line=dict(color='darkred', width=2),
                    fillcolor='rgba(220, 20, 60, 0.2)',
                    hovertemplate='<b>%{theta}</b>: %{r:.4f}<extra>Reconciling</extra>'
                ),
                row=2, col=1
            )
        
        # 4. 3D scatter for micro state trajectory
        # Apply dimensionality reduction if needed
        if S_history.shape[1] > 3:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=3)
                S_reduced = pca.fit_transform(S_history)
                
                # Create a color scale based on time
                colors = np.arange(S_reduced.shape[0])
                
                fig.add_trace(
                    go.Scatter3d(
                        x=S_reduced[:, 0],
                        y=S_reduced[:, 1],
                        z=S_reduced[:, 2],
                        mode='markers+lines',
                        marker=dict(
                            size=4,
                            color=colors,
                            colorscale='Viridis',
                            opacity=0.8,
                            colorbar=dict(
                                title="Time Step",
                                thickness=20,
                                len=0.5,
                                y=0.5,
                                yanchor="middle"
                            )
                        ),
                        line=dict(
                            color='darkgray',
                            width=1
                        ),
                        name='Micro State',
                        hovertemplate='<b>PC1</b>: %{x:.4f}<br><b>PC2</b>: %{y:.4f}<br><b>PC3</b>: %{z:.4f}<br><b>Time</b>: %{marker.color}<extra></extra>'
                    ),
                    row=2, col=2
                )
                
                # 分散説明率の表示
                fig.update_layout(
                    scene=dict(
                        xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                        yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                        zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"
                    )
                )
                
            except Exception as e:
                # Fallback to simple text if PCA fails
                fig.add_trace(
                    go.Scatter3d(
                        x=[0], y=[0], z=[0],
                        mode='markers',
                        marker=dict(size=0.1),
                        name='Error',
                        text=f"PCA Dimensionality reduction failed: {str(e)}",
                        hoverinfo='text'
                    ),
                    row=2, col=2
                )
        else:
            # Direct 3D scatter if dimensions are already 3 or less
            if S_history.shape[1] == 3:
                fig.add_trace(
                    go.Scatter3d(
                        x=S_history[:, 0],
                        y=S_history[:, 1],
                        z=S_history[:, 2],
                        mode='markers+lines',
                        marker=dict(
                            size=4,
                            color=np.arange(S_history.shape[0]),
                            colorscale='Viridis',
                            opacity=0.8
                        ),
                        line=dict(
                            color='darkgray',
                            width=1
                        ),
                        name='Micro State'
                    ),
                    row=2, col=2
                )
            else:
                # Placeholder for low-dimensional data
                fig.add_annotation(
                    x=0.5, y=0.5,
                    xref="x domain", yref="y domain",
                    text=f"Micro state has only {S_history.shape[1]} dimensions",
                    showarrow=False,
                    row=2, col=2
                )
        
        # 5. Add a table with detailed metrics values
        table_data = [
            ['Metric', 'Standard', 'Reconciling', 'Interpretation'],
            ['Delta (Downward Causation)', f"{metrics.get('Delta', 0):.4f}", 
             f"{metrics.get('Delta_reconciling', 'N/A')}", 
             "Measures if macro variables better predict future macro states than micro variables do"],
            ['Gamma (Causal Decoupling)', f"{metrics.get('Gamma', 0):.4f}", 
             f"{metrics.get('Gamma_reconciling', 'N/A')}", 
             "Measures if macro variables better predict future micro states than past micro states do"],
            ['Psi (Causal Emergence)', f"{metrics.get('Psi', 0):.4f}", 
             f"{metrics.get('Psi_reconciling', 'N/A')}",
             "Measures additional information macro variables provide about future micro states"]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=table_data[0],
                    fill_color='paleturquoise',
                    align='center',
                    font=dict(size=12)
                ),
                cells=dict(
                    values=[[row[0] for row in table_data[1:]], 
                            [row[1] for row in table_data[1:]], 
                            [row[2] for row in table_data[1:]], 
                            [row[3] for row in table_data[1:]]],
                    fill_color=[['lightcyan']*len(table_data[1:])],
                    align=['left', 'center', 'center', 'left'],
                    font=dict(size=11)
                )
            ),
            row=3, col=1
        )
        
        # Overall conclusion text based on metrics
        max_metric = max(
            metrics.get('Delta', 0), 
            metrics.get('Gamma', 0), 
            metrics.get('Psi', 0),
            metrics.get('Delta_reconciling', 0) if 'Delta_reconciling' in metrics else 0,
            metrics.get('Gamma_reconciling', 0) if 'Gamma_reconciling' in metrics else 0,
            metrics.get('Psi_reconciling', 0) if 'Psi_reconciling' in metrics else 0
        )
        
        if max_metric > 0.5:
            emergence_level = "STRONG"
            color = "darkgreen"
        elif max_metric > 0.1:
            emergence_level = "MODERATE"
            color = "darkorange"
        else:
            emergence_level = "WEAK"
            color = "darkred"
        
        # Update layout
        fig.update_layout(
            title_text=f"Emergence Analysis Dashboard: {simulation_type.upper()}",
            height=1000,
            width=1200,
            template=template,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ),
            annotations=[
                dict(
                    text=f"<b>Overall Emergence Level: {emergence_level}</b>",
                    xref="paper", yref="paper",
                    x=0.5, y=-0.03,
                    showarrow=False,
                    font=dict(size=16, color=color)
                )
            ]
        )
        
        # Save to HTML file if directory is provided
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            html_path = os.path.join(save_dir, f"{simulation_type}_emergence_dashboard.html")
            
            # Add interactive controls
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="left",
                        buttons=[
                            dict(
                                args=[{"visible": [True] * len(fig.data)}],
                                label="Show All",
                                method="update"
                            ),
                            dict(
                                args=[{"visible": [True, True, True, True, True, False] if len(fig.data) > 5 else [True] * len(fig.data)}],
                                label="Hide Table",
                                method="update"
                            )
                        ],
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.11,
                        xanchor="left",
                        y=1.1,
                        yanchor="top"
                    ),
                ]
            )
            
            # Add download button functionality via a JavaScript callback
            # Generate download link in plot
            download_link = f"""
            <a href="data:text/csv;base64,{metrics_to_csv_base64(metrics)}" 
               download="{simulation_type}_metrics.csv"
               style="float: right; margin: 10px; padding: 5px 10px; background-color: #4CAF50; 
                     color: white; text-decoration: none; border-radius: 4px;">
               Download Metrics CSV
            </a>
            """
            
            # Combine HTML content
            with open(html_path, 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>{simulation_type.upper()} Emergence Analysis</title>
                    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                </head>
                <body>
                    <h1>{simulation_type.upper()} Emergence Analysis Dashboard</h1>
                    {download_link}
                    <div id="plotly-div"></div>
                    <script>
                        var plotlyData = {fig.to_json()};
                        Plotly.newPlot('plotly-div', plotlyData.data, plotlyData.layout, {{responsive: true}});
                    </script>
                </body>
                </html>
                """)
            
            print(f"Interactive dashboard saved to {html_path}")
            return html_path
        else:
            return fig
    
    except ImportError:
        print("Interactive dashboard requires plotly. Install with 'pip install plotly'")
        return None
    except Exception as e:
        print(f"Error creating interactive dashboard: {e}")
        import traceback
        traceback.print_exc()
        return None

def metrics_to_csv_base64(metrics):
    """指標をCSV形式にしてbase64エンコードする（ダウンロード用）"""
    import base64
    import io
    
    buff = io.StringIO()
    buff.write("Metric,Value\n")
    
    for key, value in sorted(metrics.items()):
        buff.write(f"{key},{value:.6f}\n")
    
    csv_data = buff.getvalue().encode()
    return base64.b64encode(csv_data).decode()

def visualize_simulation_states(visual_frames, simulation_type, metrics=None, num_frames=12, include_animation=True, save_path=None):
    """
    Visualizes the state of the simulation during emergence metric calculation.
    
    Args:
        visual_frames: Array of simulation states at different time points
        simulation_type: Type of simulation ('gol', 'boids', etc.)
        metrics: Optional dictionary of emergence metrics to display
        num_frames: Number of key frames to display
        include_animation: Whether to include an animation of frames
        save_path: Optional path to save the visualization
        
    Returns:
        matplotlib Figure object
    """
    if visual_frames is None or len(visual_frames) == 0:
        print("No visual frames available to display")
        return None
    
    if simulation_type == 'gol':
        return visualize_gol_states(visual_frames, metrics, num_frames, include_animation, save_path)
    elif simulation_type == 'boids':
        return visualize_boids_states(visual_frames, metrics, num_frames, include_animation, save_path)
    else:
        print(f"Visualization not implemented for simulation type: {simulation_type}")
        return None

def visualize_gol_states(frames, metrics=None, num_frames=12, include_animation=True, save_path=None):
    """Visualizes Game of Life grid states over time."""
    # Determine frame selection
    total_frames = len(frames)
    if total_frames < num_frames:
        selected_indices = list(range(total_frames))
    else:
        selected_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    
    # Set up the figure
    if include_animation:
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
        
        # Top row: Metrics values if available
        if metrics is not None:
            ax_metrics = fig.add_subplot(gs[0])
            metrics_to_plot = ['Delta', 'Gamma', 'Psi']
            values = [metrics.get(m, 0) for m in metrics_to_plot]
            
            ax_metrics.bar(metrics_to_plot, values, color=['#1f77b4', '#2ca02c', '#d62728'])
            ax_metrics.set_title('Emergence Metrics', fontsize=14)
            ax_metrics.set_ylim(0, max(values) * 1.2)
            
            # Add value labels
            for i, v in enumerate(values):
                ax_metrics.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=12)
        
        # Bottom row: Grid animation
        ax_anim = fig.add_subplot(gs[1])
        
        # Create animation
        def update_frame(frame_idx):
            ax_anim.clear()
            ax_anim.set_title(f'Game of Life Grid (Frame {frame_idx}/{total_frames-1})', fontsize=14)
            im = ax_anim.imshow(frames[frame_idx], cmap='binary', interpolation='nearest')
            ax_anim.set_xticks([])
            ax_anim.set_yticks([])
            ax_anim.grid(False)
            return im,

        from matplotlib.animation import FuncAnimation
        try:
            anim = FuncAnimation(fig, update_frame, frames=range(0, total_frames, max(1, total_frames // 100)),
                                interval=100, blit=True)
            # Add animation controls
            plt.tight_layout()
            
            # Save animation if path provided
            if save_path:
                anim.save(save_path, writer='ffmpeg', fps=10, dpi=100)
        except Exception as e:
            print(f"Animation creation failed: {e}. Displaying static frames instead.")
            include_animation = False
    
    # If animation fails or is not requested, display grid of frames
    if not include_animation:
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(len(selected_indices))))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        
        # Flatten axes for easier iteration
        if grid_size > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Plot each selected frame
        for i, idx in enumerate(selected_indices):
            if i < len(axes):
                if grid_size > 1:
                    ax = axes[i]
                else:
                    ax = axes
                ax.imshow(frames[idx], cmap='binary', interpolation='nearest')
                ax.set_title(f'Frame {idx}/{total_frames-1}')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
        
        # Hide unused subplots
        for j in range(len(selected_indices), len(axes)):
            if grid_size > 1:
                axes[j].axis('off')
        
        # Add overall title with metrics if available
        if metrics is not None:
            plt.suptitle(f"Game of Life Evolution\nPsi: {metrics.get('Psi', 0):.3f}, "
                        f"Gamma: {metrics.get('Gamma', 0):.3f}, "
                        f"Delta: {metrics.get('Delta', 0):.3f}", 
                        fontsize=16, y=0.98)
        else:
            plt.suptitle("Game of Life Evolution", fontsize=16, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save static image if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

def visualize_boids_states(frames, metrics=None, num_frames=12, include_animation=True, save_path=None):
    """Visualizes Boids positions and velocities over time."""
    # Determine frame selection
    total_frames = len(frames)
    if total_frames < num_frames:
        selected_indices = list(range(total_frames))
    else:
        selected_indices = [int(i * (total_frames - 1) / (num_frames - 1)) for i in range(num_frames)]
    
    # Set up the figure
    if include_animation:
        fig = plt.figure(figsize=(18, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 3])
        
        # Top row: Metrics values if available
        if metrics is not None:
            ax_metrics = fig.add_subplot(gs[0])
            metrics_to_plot = ['Delta', 'Gamma', 'Psi']
            values = [metrics.get(m, 0) for m in metrics_to_plot]
            
            ax_metrics.bar(metrics_to_plot, values, color=['#1f77b4', '#2ca02c', '#d62728'])
            ax_metrics.set_title('Emergence Metrics', fontsize=14)
            ax_metrics.set_ylim(0, max(values) * 1.2)
            
            # Add value labels
            for i, v in enumerate(values):
                ax_metrics.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=12)
        
        # Bottom row: Boids animation
        ax_anim = fig.add_subplot(gs[1])
        
        # Create animation
        def update_frame(frame_idx):
            ax_anim.clear()
            ax_anim.set_title(f'Boids Simulation (Frame {frame_idx}/{total_frames-1})', fontsize=14)
            
            # Extract positions and velocities for this frame
            positions, velocities = frames[frame_idx]
            
            # Plot boid positions
            scatter = ax_anim.scatter(positions[:, 0], positions[:, 1], c='blue', s=30)
            
            # Plot velocity vectors (reduced for clarity)
            sample_step = max(1, len(positions) // 20)  # Show only a subset of arrows
            quiver = ax_anim.quiver(positions[::sample_step, 0], positions[::sample_step, 1],
                                   velocities[::sample_step, 0], velocities[::sample_step, 1],
                                   color='red', scale=30, width=0.003)
            
            # Set consistent axis limits
            max_pos = np.max([np.max(frame[0]) for frame in frames])
            ax_anim.set_xlim(0, max_pos)
            ax_anim.set_ylim(0, max_pos)
            ax_anim.set_aspect('equal')
            ax_anim.grid(True, alpha=0.3)
            
            return scatter, quiver

        from matplotlib.animation import FuncAnimation
        try:
            anim = FuncAnimation(fig, update_frame, frames=range(0, total_frames, max(1, total_frames // 100)),
                               interval=100, blit=False)
            # Add animation controls
            plt.tight_layout()
            
            # Save animation if path provided
            if save_path:
                anim.save(save_path, writer='ffmpeg', fps=10, dpi=100)
        except Exception as e:
            print(f"Animation creation failed: {e}. Displaying static frames instead.")
            include_animation = False
    
    # If animation fails or is not requested, display grid of frames
    if not include_animation:
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(len(selected_indices))))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
        
        # Flatten axes for easier iteration
        if grid_size > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        # Find global limits for consistent axes
        max_pos = np.max([np.max(frames[idx][0]) for idx in selected_indices])
        
        # Plot each selected frame
        for i, idx in enumerate(selected_indices):
            if i < len(axes):
                if grid_size > 1:
                    ax = axes[i]
                else:
                    ax = axes
                
                positions, velocities = frames[idx]
                ax.scatter(positions[:, 0], positions[:, 1], c='blue', s=30)
                
                # Plot velocity vectors (reduced for clarity)
                sample_step = max(1, len(positions) // 20)
                ax.quiver(positions[::sample_step, 0], positions[::sample_step, 1],
                        velocities[::sample_step, 0], velocities[::sample_step, 1],
                        color='red', scale=30, width=0.003)
                
                ax.set_title(f'Frame {idx}/{total_frames-1}')
                ax.set_xlim(0, max_pos)
                ax.set_ylim(0, max_pos)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(len(selected_indices), len(axes)):
            if grid_size > 1:
                axes[j].axis('off')
        
        # Add overall title with metrics if available
        if metrics is not None:
            plt.suptitle(f"Boids Simulation Evolution\nPsi: {metrics.get('Psi', 0):.3f}, "
                        f"Gamma: {metrics.get('Gamma', 0):.3f}, "
                        f"Delta: {metrics.get('Delta', 0):.3f}", 
                        fontsize=16, y=0.98)
        else:
            plt.suptitle("Boids Simulation Evolution", fontsize=16, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save static image if path provided
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig

if __name__ == "__main__":
    # Example usage
    print("To use this module, import it from your analysis script.")
    print("Example:")
    print("  from emergence_visualization import create_dashboard")
    print("  fig = create_dashboard(metrics, S_history, M_history, 'gol')")
    print("  plt.show()")
