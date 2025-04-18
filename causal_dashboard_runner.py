#!/usr/bin/env python
"""
Script to run causal emergence analysis with interactive visualizations.

This script loads simulation data (or runs a new simulation), calculates
emergence metrics, and creates visualizations to help interpret the results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from typing import Dict, Optional, Union, List, Tuple
import datetime  # Add this import for timestamp functionality

# Check if we're in the correct directory
if not os.path.exists('causal_emergence.py'):
    print("Error: This script should be run from the main 'asal' directory.")
    print("Please change to the directory containing 'causal_emergence.py'")
    sys.exit(1)

# Import local modules
import emergence_metrics
from causal_emergence import run_simulation, calculate_emergence_metrics, explain_metric_differences
# Fix: Add plot_metrics_radar to the imports
from emergence_visualization import create_dashboard, plot_interactive_dashboard, plot_metrics_radar

def parse_args():
    parser = argparse.ArgumentParser(description="Run causal emergence analysis with visualizations")
    
    # Simulation parameters
    parser.add_argument("--simulation_type", type=str, default="gol",
                        choices=["gol", "boids", "other"],
                        help="Type of simulation to analyze")
    parser.add_argument("--n_steps", type=int, default=1000,
                        help="Number of simulation steps")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for simulation")
    
    # Metrics parameters
    parser.add_argument("--discretize", action="store_true",
                        help="Discretize data before computing information metrics")
    parser.add_argument("--n_bins", type=int, default=10,
                        help="Number of bins for discretization")
    
    # 保存オプションの拡張
    save_group = parser.add_argument_group("Save options")
    save_group.add_argument("--save_dir", type=str, default="./output",
                        help="Directory to save results and visualizations")
    save_group.add_argument("--save_format", type=str, default="both", 
                        choices=["dashboard", "image", "both"],
                        help="How to save results: dashboard (HTML), image (PNG), or both")
    save_group.add_argument("--image_dpi", type=int, default=300,
                        help="DPI for saved images (higher for publication quality)")
    save_group.add_argument("--image_format", type=str, default="png",
                        choices=["png", "pdf", "svg", "jpg"],
                        help="Format for saved images")
    save_group.add_argument("--publication_ready", action="store_true",
                        help="Generate publication-ready figures with larger fonts and better formatting")
    
    # インタラクティブダッシュボードのオプション
    dashboard_group = parser.add_argument_group("Dashboard options")
    dashboard_group.add_argument("--interactive", action="store_true",
                        help="Create interactive dashboard (requires plotly)")
    dashboard_group.add_argument("--dashboard_template", type=str, default="plotly_white",
                        choices=["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"],
                        help="Template for interactive dashboard")
    dashboard_group.add_argument("--include_metadata", action="store_true",
                        help="Include simulation metadata in dashboard")
    
    # Add the missing show_plots argument
    parser.add_argument("--show_plots", action="store_true",
                        help="Show plots interactively")
    
    # Additional options
    parser.add_argument("--load_data", type=str, default=None,
                        help="Path to load existing simulation data instead of running new simulation")
    
    return parser.parse_args()

def save_as_dashboard(metrics, S_history, M_history, simulation_type, args):
    """
    シミュレーション結果をHTML形式のダッシュボードとして保存
    
    Parameters:
    -----------
    metrics : dict
        計算された創発指標
    S_history, M_history : array
        マイクロ・マクロ状態の履歴データ
    simulation_type : str
        シミュレーションのタイプ
    args : argparse.Namespace
        コマンドライン引数
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import datetime
        import json
        
        # ダッシュボード用ディレクトリの作成
        dashboard_dir = os.path.join(args.save_dir, f"{simulation_type}_dashboard")
        os.makedirs(dashboard_dir, exist_ok=True)
        
        # メイン画像ダッシュボードの作成
        interactive_path = plot_interactive_dashboard(
            metrics, S_history, M_history, simulation_type, 
            save_dir=dashboard_dir,
            template=args.dashboard_template
        )
        
        # メタデータファイルの作成（条件探索のため）
        if args.include_metadata:
            metadata = {
                "simulation_type": simulation_type,
                "timestamp": datetime.datetime.now().isoformat(),
                "n_steps": args.n_steps,
                "random_seed": args.random_seed,
                "metrics": {k: float(v) for k, v in metrics.items()},  # numpy値をJSONシリアライズ可能に変換
                "S_history_shape": list(S_history.shape),
                "M_history_shape": list(M_history.shape),
                "command_args": vars(args)
            }
            
            # メタデータのJSON保存
            with open(os.path.join(dashboard_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # 簡易インデックスHTMLの作成
            index_html = f"""<!DOCTYPE html>
            <html>
            <head>
                <title>{simulation_type.upper()} Emergence Analysis</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .metrics {{ background: #f5f5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
                    .dashboard-link {{ display: inline-block; margin-top: 20px; padding: 10px 15px; 
                                     background: #4CAF50; color: white; text-decoration: none; border-radius: 4px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>{simulation_type.upper()} Emergence Analysis</h1>
                    <div class="metrics">
                        <h2>Emergence Metrics:</h2>
                        <ul>
                            <li><strong>Delta (Downward Causation):</strong> {metrics.get('Delta', 0):.4f}</li>
                            <li><strong>Gamma (Causal Decoupling):</strong> {metrics.get('Gamma', 0):.4f}</li>
                            <li><strong>Psi (Causal Emergence):</strong> {metrics.get('Psi', 0):.4f}</li>
                        </ul>
                    </div>
                    <h2>Interactive Dashboard</h2>
                    <a href="{os.path.basename(interactive_path)}" class="dashboard-link">View Interactive Dashboard</a>
                    
                    <h2>Simulation Details</h2>
                    <ul>
                        <li><strong>Type:</strong> {simulation_type}</li>
                        <li><strong>Steps:</strong> {args.n_steps}</li>
                        <li><strong>Random Seed:</strong> {args.random_seed}</li>
                        <li><strong>Generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>
                    </ul>
                </div>
            </body>
            </html>
            """
            
            # インデックスHTMLの保存
            with open(os.path.join(dashboard_dir, "index.html"), 'w') as f:
                f.write(index_html)
            
            print(f"\nComplete dashboard saved to {dashboard_dir}/index.html")
            return dashboard_dir
        
        return os.path.dirname(interactive_path)
        
    except ImportError as e:
        print(f"Error creating dashboard: {e}")
        print("Make sure plotly is installed: pip install plotly")
        return None
    except Exception as e:
        print(f"Error creating dashboard: {e}")
        return None

def save_as_image(metrics, S_history, M_history, simulation_type, args):
    """
    シミュレーション結果を高品質な画像として保存
    
    Parameters:
    -----------
    metrics : dict
        計算された創発指標
    S_history, M_history : array
        マイクロ・マクロ状態の履歴データ
    simulation_type : str
        シミュレーションのタイプ
    args : argparse.Namespace
        コマンドライン引数
    """
    # 画像保存用ディレクトリ
    image_dir = os.path.join(args.save_dir, f"{simulation_type}_images")
    os.makedirs(image_dir, exist_ok=True)
    
    # フォントサイズの調整（論文用）
    if args.publication_ready:
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12
        })
    
    # 1. メインダッシュボード画像
    dashboard_fig = create_dashboard(metrics, S_history, M_history, simulation_type)
    dashboard_path = os.path.join(image_dir, f"{simulation_type}_dashboard.{args.image_format}")
    dashboard_fig.savefig(dashboard_path, dpi=args.image_dpi, bbox_inches='tight')
    plt.close(dashboard_fig)
    
    # 2. マクロ状態変化のより詳細なプロット
    macro_fig, macro_ax = plt.subplots(figsize=(10, 6))
    for i in range(M_history.shape[1]):
        macro_ax.plot(M_history[:, i], label=f'Macro Variable {i+1}', linewidth=2)
    
    macro_ax.set_title(f'Macro State Evolution - {simulation_type.upper()}')
    macro_ax.set_xlabel('Time Step')
    macro_ax.set_ylabel('Value')
    macro_ax.legend()
    macro_ax.grid(True)
    
    macro_path = os.path.join(image_dir, f"{simulation_type}_macro_evolution.{args.image_format}")
    macro_fig.savefig(macro_path, dpi=args.image_dpi, bbox_inches='tight')
    plt.close(macro_fig)
    
    # 3. 指標のレーダーチャート（単独版）
    radar_fig = plt.figure(figsize=(8, 8))
    radar_ax = radar_fig.add_subplot(111, polar=True)
    plot_metrics_radar(metrics, ax=radar_ax)
    
    radar_path = os.path.join(image_dir, f"{simulation_type}_metrics_radar.{args.image_format}")
    radar_fig.savefig(radar_path, dpi=args.image_dpi, bbox_inches='tight')
    plt.close(radar_fig)
    
    # 4. PCA状態空間の時系列変化
    try:
        from sklearn.decomposition import PCA
        
        pca_fig, pca_ax = plt.subplots(figsize=(10, 8))
        if S_history.shape[1] > 2:
            pca = PCA(n_components=2)
            S_reduced = pca.fit_transform(S_history)
            
            # 時間に基づく色付け
            time_norm = np.arange(S_history.shape[0]) / S_history.shape[0]
            scatter = pca_ax.scatter(S_reduced[:, 0], S_reduced[:, 1], 
                                    c=time_norm, cmap='viridis',
                                    s=30, alpha=0.7)
            
            # 軌道を線で表示
            pca_ax.plot(S_reduced[:, 0], S_reduced[:, 1], 'k-', alpha=0.3, linewidth=0.5)
            
            # 開始点と終了点をハイライト
            pca_ax.scatter(S_reduced[0, 0], S_reduced[0, 1], c='green', s=100,
                          label='Start', edgecolors='black', zorder=5)
            pca_ax.scatter(S_reduced[-1, 0], S_reduced[-1, 1], c='red', s=100,
                          label='End', edgecolors='black', zorder=5)
            
            cbar = plt.colorbar(scatter, ax=pca_ax)
            cbar.set_label('Time')
            
            # 軸ラベルと説明
            var_explained = pca.explained_variance_ratio_
            pca_ax.set_xlabel(f'PC1 ({var_explained[0]:.2%} variance)')
            pca_ax.set_ylabel(f'PC2 ({var_explained[1]:.2%} variance)')
            pca_ax.set_title(f'State Space Trajectory - {simulation_type.upper()}')
            pca_ax.legend()
            
            pca_path = os.path.join(image_dir, f"{simulation_type}_state_space.{args.image_format}")
            pca_fig.savefig(pca_path, dpi=args.image_dpi, bbox_inches='tight')
        
        plt.close(pca_fig)
    except:
        print("Skipping PCA visualization.")
    
    # 5. メトリクスバーチャート（標準vs. 調整済み）
    if "Delta_reconciling" in metrics:
        bar_fig, bar_ax = plt.subplots(figsize=(10, 6))
        
        metrics_labels = ['Delta', 'Gamma', 'Psi']
        x = np.arange(len(metrics_labels))
        width = 0.35
        
        # 標準メトリクス
        standard_vals = [metrics[m] for m in metrics_labels]
        bar_ax.bar(x - width/2, standard_vals, width, label='Standard')
        
        # 調整済みメトリクス
        reconciling_vals = [metrics[f"{m}_reconciling"] for m in metrics_labels]
        bar_ax.bar(x + width/2, reconciling_vals, width, label='Reconciling')
        
        bar_ax.set_ylabel('Value')
        bar_ax.set_title(f'Emergence Metrics Comparison - {simulation_type.upper()}')
        bar_ax.set_xticks(x)
        bar_ax.set_xticklabels(metrics_labels)
        bar_ax.legend()
        
        # バーの上に数値を表示
        for i, v in enumerate(standard_vals):
            bar_ax.text(i - width/2, v + 0.02, f"{v:.3f}", ha='center', fontsize=10)
        for i, v in enumerate(reconciling_vals):
            bar_ax.text(i + width/2, v + 0.02, f"{v:.3f}", ha='center', fontsize=10)
        
        bar_path = os.path.join(image_dir, f"{simulation_type}_metrics_comparison.{args.image_format}")
        bar_fig.savefig(bar_path, dpi=args.image_dpi, bbox_inches='tight')
        plt.close(bar_fig)
    
    # メトリクス値をテキストファイルにも保存
    metrics_text = f"# Emergence Metrics for {simulation_type.upper()}\n"
    metrics_text += f"# Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # 標準メトリクス
    metrics_text += "## Standard Metrics\n"
    metrics_text += f"Delta (Downward Causation): {metrics.get('Delta', 0):.6f}\n"
    metrics_text += f"Gamma (Causal Decoupling): {metrics.get('Gamma', 0):.6f}\n"
    metrics_text += f"Psi (Causal Emergence): {metrics.get('Psi', 0):.6f}\n\n"
    
    # 調整済みメトリクス（存在する場合）
    if "Delta_reconciling" in metrics:
        metrics_text += "## Reconciling Metrics\n"
        metrics_text += f"Delta_R (Downward Causation): {metrics.get('Delta_reconciling', 0):.6f}\n"
        metrics_text += f"Gamma_R (Causal Decoupling): {metrics.get('Gamma_reconciling', 0):.6f}\n"
        metrics_text += f"Psi_R (Causal Emergence): {metrics.get('Psi_reconciling', 0):.6f}\n"
    
    # テキストファイルとして保存
    with open(os.path.join(image_dir, f"{simulation_type}_metrics.txt"), 'w') as f:
        f.write(metrics_text)
    
    print(f"\nHigh-quality images saved to {image_dir}")
    return image_dir

def main():
    """Main entry point for the dashboard runner."""
    args = parse_args()
    
    # Create save directory if it doesn't exist
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
    
    # Step 1: Get simulation data (either load or run new)
    if args.load_data:
        print(f"Loading simulation data from: {args.load_data}")
        try:
            data = np.load(args.load_data, allow_pickle=True)
            if "S_history" in data and "M_history" in data:
                S_history = data["S_history"]
                M_history = data["M_history"]
                # Try to load visual frames if available
                visual_frames = data["visual_frames"] if "visual_frames" in data else None
            else:
                print("Error: Loaded data must contain 'S_history' and 'M_history'")
                sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    else:
        print(f"Running new {args.simulation_type.upper()} simulation "
              f"with {args.n_steps} steps (seed: {args.random_seed})...")
              
        # Update to unpack the visual_frames as well
        S_history, M_history, visual_frames = run_simulation(args.simulation_type, args.n_steps, args.random_seed)
        
        if S_history is None or M_history is None:
            print("Error: Simulation failed. Exiting.")
            sys.exit(1)
    
    print(f"Data shapes: S_history={S_history.shape}, M_history={M_history.shape}")
    
    # Save the simulation data if running new simulation
    if args.save_dir and not args.load_data:
        save_path = os.path.join(args.save_dir, f"{args.simulation_type}_simulation_data.npz")
        np.savez_compressed(save_path, S_history=S_history, M_history=M_history, visual_frames=visual_frames)
        print(f"Simulation data saved to {save_path}")
    
    # Step 2: Calculate emergence metrics
    print("Calculating emergence metrics...")
    metrics = calculate_emergence_metrics(S_history, M_history, 
                                         discretize=args.discretize,
                                         n_bins=args.n_bins)
    
    if metrics is None:
        print("Error calculating metrics. Exiting.")
        sys.exit(1)
    
    # Print the metrics
    print("\n--- Emergence Metrics Results ---")
    print(f" Ψ (Psi):   {metrics['Psi']:.4f}")
    print(f" Γ (Gamma): {metrics['Gamma']:.4f}")
    print(f" Δ (Delta): {metrics['Delta']:.4f}")
    
    if "Psi_reconciling" in metrics:
        print("\n--- Reconciling Metrics Results ---")
        print(f" Ψ_R (Psi):   {metrics['Psi_reconciling']:.4f}")
        print(f" Γ_R (Gamma): {metrics['Gamma_reconciling']:.4f}")
        print(f" Δ_R (Delta): {metrics['Delta_reconciling']:.4f}")
    
    # Explain differences between standard and reconciling metrics
    if "Delta_reconciling" in metrics:
        explain_metric_differences(metrics)
    
    # Save metrics if directory specified
    if args.save_dir:
        metrics_path = os.path.join(args.save_dir, f"{args.simulation_type}_metrics.npz")
        np.savez(metrics_path, **metrics)
        print(f"Metrics saved to {metrics_path}")
    
    # Step 3: 結果を保存（ダッシュボードまたは画像形式）
    print("\nSaving results...")
    
    if args.save_format in ["dashboard", "both"]:
        print("Creating interactive dashboard...")
        dashboard_dir = save_as_dashboard(metrics, S_history, M_history, args.simulation_type, args)
    
    if args.save_format in ["image", "both"]:
        print("Creating high-quality images...")
        image_dir = save_as_image(metrics, S_history, M_history, args.simulation_type, args)
        
        # Add simulation state visualization to saved images
        if visual_frames is not None:
            from emergence_visualization import visualize_simulation_states
            print("Creating simulation state visualization...")
            sim_fig = visualize_simulation_states(
                visual_frames, 
                args.simulation_type, 
                metrics,
                include_animation=(args.image_format in ["mp4", "gif"]),
                save_path=os.path.join(image_dir, f"{args.simulation_type}_states.{args.image_format}")
            )
            if sim_fig and not args.image_format in ["mp4", "gif"]:
                sim_fig.savefig(
                    os.path.join(image_dir, f"{args.simulation_type}_states.{args.image_format}"),
                    dpi=args.image_dpi
                )
    
    # Interactive displayは保存後も表示
    if args.show_plots:
        # Add visualization of the simulation states to the shown plots
        if visual_frames is not None:
            from emergence_visualization import visualize_simulation_states
            print("Showing simulation state visualization...")
            sim_fig = visualize_simulation_states(visual_frames, args.simulation_type, metrics)
        plt.show()
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
