#!/usr/bin/env python
"""
Create a visualization dashboard for causal blanket analysis results.
This script generates an HTML dashboard with plots and metrics visualization.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import shutil

def create_dashboard(results_csv, plot_dir):
    """
    Create a dashboard from analysis results.
    
    Args:
        results_csv: Path to CSV file with analysis results
        plot_dir: Directory containing plot images
    """
    try:
        # Load results
        print(f"Loading results from {results_csv}")
        df = pd.read_csv(results_csv)
        
        # Create dashboard directory
        dashboard_dir = "dashboard"
        os.makedirs(dashboard_dir, exist_ok=True)
        
        # Copy necessary image files to dashboard directory
        copy_plot_files(plot_dir, dashboard_dir, df['index'])
        
        # Create summary plots
        print("Generating summary plots")
        generate_summary_plots(df, dashboard_dir)
        
        # Create HTML dashboard
        print("Creating HTML dashboard")
        create_html_dashboard(df, dashboard_dir)
        
        print(f"Dashboard created successfully at {os.path.abspath(os.path.join(dashboard_dir, 'index.html'))}")
        return True
    except Exception as e:
        print(f"Error creating dashboard: {e}")
        return False

def copy_plot_files(source_dir, dest_dir, indices):
    """Copy plot files from source to destination directory"""
    # Create images directory inside dashboard
    images_dir = os.path.join(dest_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Copy files for each index
    for idx in indices:
        for plot_type in ["particle_positions", "time_series", "info_flow", "info_metrics", "synergy_metrics"]:
            source_file = os.path.join(source_dir, f"{plot_type}_{idx}.png")
            if os.path.exists(source_file):
                dest_file = os.path.join(images_dir, f"{plot_type}_{idx}.png")
                shutil.copy2(source_file, dest_file)
                print(f"Copied {source_file} to {dest_file}")

def generate_summary_plots(df, dashboard_dir):
    """Generate summary plots for the dashboard"""
    # Create summary metrics plot
    plt.figure(figsize=(12, 8))
    metrics = ['MI_X_to_X', 'MI_Y_to_Y', 'MI_X_to_Y', 'MI_Y_to_X', 'PID_X_synergy', 'PID_Y_synergy', 'D_XY', 'G_XY']
    
    # Filter metrics that exist in the dataframe
    available_metrics = [m for m in metrics if m in df.columns]
    
    if available_metrics:
        df[available_metrics].mean().plot(kind='bar')
        plt.title('Average Information Metrics Across Particles')
        plt.ylabel('Information (bits)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(dashboard_dir, "summary_metrics.png"))
    
    # Create additional plots if data is available
    if 'D_XY' in df.columns and 'G_XY' in df.columns:
        plt.figure(figsize=(12, 8))
        plt.scatter(df['D_XY'], df['G_XY'], alpha=0.7)
        plt.xlabel('D_XY (Synergy - Redundancy)')
        plt.ylabel('G_XY (Synergy Ratio)')
        plt.title('Synergy vs Redundancy Balance')
        plt.grid(True, alpha=0.3)
        for i, idx in enumerate(df['index']):
            plt.annotate(str(idx), (df['D_XY'].iloc[i], df['G_XY'].iloc[i]))
        plt.savefig(os.path.join(dashboard_dir, "synergy_scatter.png"))

def create_html_dashboard(df, dashboard_dir):
    """Create HTML dashboard with CSS styling"""
    html_path = os.path.join(dashboard_dir, "index.html")
    
    with open(html_path, 'w') as f:
        # HTML header with CSS
        f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Causal Blanket Analysis Dashboard</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1, h2, h3 {
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary {
            margin-bottom: 30px;
        }
        .particles-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .particle-card {
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            background-color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        img {
            max-width: 100%;
            height: auto;
            margin-bottom: 10px;
            border: 1px solid #eee;
        }
        .metrics {
            font-size: 14px;
            margin-top: 10px;
        }
        .metrics table {
            width: 100%;
            border-collapse: collapse;
        }
        .metrics td, .metrics th {
            border: 1px solid #ddd;
            padding: 6px;
            text-align: left;
        }
        .metrics th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Particle Lenia Causal Blanket Analysis</h1>
''')
        
        # Summary section
        f.write('''
        <div class="summary">
            <h2>Summary Metrics</h2>
            <img src="summary_metrics.png" alt="Summary Metrics">
''')
        
        # Add synergy scatter plot if available
        if os.path.exists(os.path.join(dashboard_dir, "synergy_scatter.png")):
            f.write('            <img src="synergy_scatter.png" alt="Synergy Scatter Plot">\n')
            
        f.write('        </div>\n')
        
        # Individual particles section
        f.write('        <h2>Individual Particles</h2>\n')
        f.write('        <div class="particles-grid">\n')
        
        # Add each particle's information
        for _, row in df.iterrows():
            idx = row['index']
            f.write(f'''            <div class="particle-card">
                <h3>Particle {idx}</h3>
                <div class="metrics">
                    <table>
                        <tr><th>Parameter 0</th><td>{row.get('param_0', 'N/A')}</td></tr>
                        <tr><th>Parameter 1</th><td>{row.get('param_1', 'N/A')}</td></tr>
                        <tr><th>Timesteps</th><td>{row.get('timesteps', 'N/A')}</td></tr>
                        <tr><th>MI(X→X)</th><td>{row.get('MI_X_to_X', 'N/A'):.4f}</td></tr>
                        <tr><th>MI(Y→Y)</th><td>{row.get('MI_Y_to_Y', 'N/A'):.4f}</td></tr>
                        <tr><th>D_XY</th><td>{row.get('D_XY', 'N/A'):.4f}</td></tr>
                        <tr><th>G_XY</th><td>{row.get('G_XY', 'N/A'):.4f}</td></tr>
                    </table>
                </div>
                <img src="images/particle_positions_{idx}.png" alt="Particle Positions">
                <img src="images/time_series_{idx}.png" alt="Time Series">
                <img src="images/info_flow_{idx}.png" alt="Information Flow">
            </div>
''')
            
        f.write('        </div>\n')
        f.write('    </div>\n</body>\n</html>')

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_dashboard.py [results_csv] [plot_dir]")
        sys.exit(1)
        
    results_csv = sys.argv[1]
    plot_dir = sys.argv[2]
    
    success = create_dashboard(results_csv, plot_dir)
    sys.exit(0 if success else 1)
