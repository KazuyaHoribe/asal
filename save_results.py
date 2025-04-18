#!/usr/bin/env python
"""
Utility script for managing and saving ASAL parameter search results.

This script helps with post-processing, compressing, and organizing the results
from parameter searches. It can also generate summary reports across multiple runs.
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import shutil
import glob
from typing import Dict, List, Optional, Any, Tuple

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Parameter search results management utility")
    
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing parameter search results")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save processed results (default: input_dir/processed)")
    parser.add_argument("--action", type=str, default="summarize",
                        choices=["summarize", "compress", "clean", "visualize", "extract_best"],
                        help="Action to perform on the results")
    parser.add_argument("--compress_format", type=str, default="zip",
                        choices=["zip", "tar.gz"],
                        help="Format for compression")
    parser.add_argument("--delete_after_compress", action="store_true",
                        help="Delete original files after compression")
    parser.add_argument("--min_metrics", type=float, default=None,
                        help="Minimum value for metrics to include in extraction")
    
    return parser.parse_args()

def summarize_results(input_dir: str, output_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Create a summary of all parameter search results in the input directory.
    
    Args:
        input_dir: Directory containing parameter search results
        output_dir: Directory to save the summary
        
    Returns:
        DataFrame containing the summary
    """
    # Find all results directories
    result_dirs = []
    for root, dirs, files in os.walk(input_dir):
        # Look for directories containing search_results.csv
        if "search_results.csv" in files:
            result_dirs.append(root)
    
    if not result_dirs:
        print(f"No parameter search results found in {input_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(result_dirs)} parameter search result directories")
    
    # Collect summary information
    summary_data = []
    for result_dir in result_dirs:
        # Read the search results
        try:
            df = pd.read_csv(os.path.join(result_dir, "search_results.csv"))
            
            # Get run_info if available
            run_info_path = os.path.join(result_dir, "run_info.json")
            run_info = {}
            if os.path.exists(run_info_path):
                with open(run_info_path, 'r') as f:
                    run_info = json.load(f)
            
            # Get best parameters if available
            best_params_path = os.path.join(result_dir, "best_params", "best_parameters.json")
            best_params = {}
            if os.path.exists(best_params_path):
                with open(best_params_path, 'r') as f:
                    best_params = json.load(f)
            
            # Extract key metrics
            success_rate = df['success'].mean() if 'success' in df.columns else 0
            
            # For successful simulations, get metric statistics
            if 'success' in df.columns and 'Delta' in df.columns:
                success_df = df[df['success'] == True]
                if not success_df.empty:
                    metrics_data = {
                        'Delta_mean': success_df['Delta'].mean(),
                        'Delta_max': success_df['Delta'].max(),
                        'Gamma_mean': success_df['Gamma'].mean() if 'Gamma' in success_df.columns else None,
                        'Gamma_max': success_df['Gamma'].max() if 'Gamma' in success_df.columns else None,
                        'Psi_mean': success_df['Psi'].mean() if 'Psi' in success_df.columns else None,
                        'Psi_max': success_df['Psi'].max() if 'Psi' in success_df.columns else None
                    }
                else:
                    metrics_data = {
                        'Delta_mean': None, 'Delta_max': None,
                        'Gamma_mean': None, 'Gamma_max': None,
                        'Psi_mean': None, 'Psi_max': None
                    }
            else:
                metrics_data = {
                    'Delta_mean': None, 'Delta_max': None,
                    'Gamma_mean': None, 'Gamma_max': None,
                    'Psi_mean': None, 'Psi_max': None
                }
            
            # Get the best combined parameters
            best_combined = best_params.get('best_combined', {})
            best_value = best_combined.get('value', 0) if best_combined else 0
            best_params_dict = best_combined.get('params', {}) if best_combined else {}
            
            # Create summary entry
            entry = {
                'result_dir': result_dir,
                'simulation_type': run_info.get('simulation_type', os.path.basename(result_dir).split('_')[0]),
                'timestamp': run_info.get('completed_at', ''),
                'total_jobs': len(df),
                'success_rate': success_rate,
                'best_combined_value': best_value,
                'best_combined_params': str(best_params_dict)
            }
            entry.update(metrics_data)
            
            summary_data.append(entry)
            
        except Exception as e:
            print(f"Error processing {result_dir}: {e}")
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save the summary if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "parameter_search_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        # Also save as Excel for better formatting
        try:
            excel_path = os.path.join(output_dir, "parameter_search_summary.xlsx")
            with pd.ExcelWriter(excel_path) as writer:
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add a pivot table for deeper analysis
                pivot_df = summary_df.pivot_table(
                    values=['Delta_max', 'Gamma_max', 'Psi_max'],
                    index=['simulation_type'],
                    aggfunc=[np.mean, np.max]
                )
                pivot_df.to_excel(writer, sheet_name='Pivot Analysis')
                
            print(f"Saved summary to {summary_path} and {excel_path}")
        except ImportError:
            print(f"Saved summary to {summary_path}")
        
        # Create visualization of best metrics
        try:
            plt.figure(figsize=(15, 10))
            
            metrics_to_plot = ['Delta_max', 'Gamma_max', 'Psi_max']
            for i, metric in enumerate(metrics_to_plot, 1):
                plt.subplot(1, 3, i)
                if summary_df[metric].notna().any():
                    sns.barplot(x='simulation_type', y=metric, data=summary_df)
                    plt.title(f'Best {metric.split("_")[0]} Values by Simulation Type')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
            
            viz_path = os.path.join(output_dir, "best_metrics_summary.png")
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {viz_path}")
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    return summary_df

def compress_results(input_dir: str, output_dir: Optional[str] = None, 
                    compress_format: str = "zip", delete_originals: bool = False) -> List[str]:
    """
    Compress parameter search results directories to save space.
    
    Args:
        input_dir: Directory containing parameter search results
        output_dir: Directory to save compressed files
        compress_format: Format for compression ("zip" or "tar.gz")
        delete_originals: Whether to delete original directories after compression
        
    Returns:
        List of paths to compressed files
    """
    # Find all results directories
    result_dirs = []
    for root, dirs, files in os.walk(input_dir):
        # Look for directories containing search_results.csv
        if "search_results.csv" in files:
            result_dirs.append(root)
    
    if not result_dirs:
        print(f"No parameter search results found in {input_dir}")
        return []
    
    print(f"Found {len(result_dirs)} parameter search result directories to compress")
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(input_dir, "compressed")
    os.makedirs(output_dir, exist_ok=True)
    
    # Compress each directory
    compressed_files = []
    for result_dir in result_dirs:
        try:
            dir_name = os.path.basename(result_dir)
            archive_path = os.path.join(output_dir, dir_name)
            
            if compress_format == "zip":
                archive_path += ".zip"
                shutil.make_archive(
                    os.path.splitext(archive_path)[0],
                    'zip',
                    result_dir
                )
            else:  # tar.gz
                archive_path += ".tar.gz"
                shutil.make_archive(
                    os.path.splitext(archive_path)[0],
                    'gztar',
                    result_dir
                )
            
            compressed_files.append(archive_path)
            print(f"Compressed {result_dir} to {archive_path}")
            
            # Delete original if requested
            if delete_originals:
                shutil.rmtree(result_dir)
                print(f"Deleted original directory {result_dir}")
                
        except Exception as e:
            print(f"Error compressing {result_dir}: {e}")
    
    # Create a manifest of all compressed files
    with open(os.path.join(output_dir, "compressed_files_manifest.txt"), 'w') as f:
        f.write(f"Compression completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total compressed directories: {len(compressed_files)}\n\n")
        for i, path in enumerate(compressed_files, 1):
            f.write(f"{i}. {os.path.basename(path)}\n")
    
    return compressed_files

def extract_best_results(input_dir: str, output_dir: Optional[str] = None, min_metrics: float = 0.1) -> Dict:
    """
    Extract and copy the best parameter search results to a new directory.
    
    Args:
        input_dir: Directory containing parameter search results
        output_dir: Directory to save extracted results
        min_metrics: Minimum value for metrics to include
        
    Returns:
        Dictionary with paths to best results
    """
    # Get summary of all results
    summary_df = summarize_results(input_dir)
    
    if summary_df.empty:
        print("No results to extract")
        return {}
    
    # Filter by minimum metrics value
    if min_metrics is not None:
        filtered_df = summary_df[
            (summary_df['Delta_max'] >= min_metrics) | 
            (summary_df['Gamma_max'] >= min_metrics) | 
            (summary_df['Psi_max'] >= min_metrics)
        ]
        
        if filtered_df.empty:
            print(f"No results with metrics >= {min_metrics}")
            return {}
        
        summary_df = filtered_df
    
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(input_dir, "best_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy best results
    best_results = {}
    for i, row in summary_df.iterrows():
        src_dir = row['result_dir']
        sim_type = row['simulation_type']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create a descriptive name for the destination directory
        dest_name = f"{sim_type}_D{row['Delta_max']:.2f}_G{row['Gamma_max']:.2f}_P{row['Psi_max']:.2f}_{timestamp}"
        dest_dir = os.path.join(output_dir, dest_name)
        
        # Create directory and copy key files
        os.makedirs(dest_dir, exist_ok=True)
        
        try:
            # Copy optimal parameters if they exist
            opt_params_path = os.path.join(src_dir, "best_params", "best_parameters.json")
            if os.path.exists(opt_params_path):
                shutil.copy2(opt_params_path, os.path.join(dest_dir, "best_parameters.json"))
            
            # Copy optimal simulation results if they exist
            opt_sim_dir = os.path.join(src_dir, "optimal_run")
            if os.path.exists(opt_sim_dir):
                dest_opt_dir = os.path.join(dest_dir, "optimal_run")
                os.makedirs(dest_opt_dir, exist_ok=True)
                
                for file in os.listdir(opt_sim_dir):
                    src_file = os.path.join(opt_sim_dir, file)
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, os.path.join(dest_opt_dir, file))
            
            # Copy visualizations if they exist
            vis_patterns = ["*.png", "*.pdf", "*.jpg", "*.svg"]
            for pattern in vis_patterns:
                for vis_file in glob.glob(os.path.join(src_dir, "**", pattern), recursive=True):
                    if os.path.isfile(vis_file):
                        rel_path = os.path.relpath(vis_file, src_dir)
                        dest_file = os.path.join(dest_dir, rel_path)
                        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
                        shutil.copy2(vis_file, dest_file)
            
            # Copy metrics summary
            with open(os.path.join(dest_dir, "metrics_summary.txt"), 'w') as f:
                f.write(f"Best Metrics Summary\n")
                f.write(f"===================\n\n")
                f.write(f"Simulation Type: {sim_type}\n")
                f.write(f"Delta Max: {row['Delta_max']:.4f}\n")
                f.write(f"Gamma Max: {row['Gamma_max']:.4f}\n")
                f.write(f"Psi Max: {row['Psi_max']:.4f}\n")
                f.write(f"Best Combined Value: {row['best_combined_value']:.4f}\n")
                f.write(f"Best Combined Parameters: {row['best_combined_params']}\n")
                
            best_results[dest_name] = {
                "source": src_dir,
                "destination": dest_dir,
                "metrics": {
                    "Delta_max": row['Delta_max'],
                    "Gamma_max": row['Gamma_max'],
                    "Psi_max": row['Psi_max']
                }
            }
            
            print(f"Extracted best results from {src_dir} to {dest_dir}")
            
        except Exception as e:
            print(f"Error extracting best results from {src_dir}: {e}")
    
    # Create a summary of extracted results
    with open(os.path.join(output_dir, "extracted_results_summary.json"), 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "min_metrics": min_metrics,
            "total_extracted": len(best_results),
            "results": best_results
        }, f, indent=2)
    
    return best_results

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Handle output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(args.input_dir, "processed")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.action == "summarize":
        print(f"Creating summary of results in {args.input_dir}")
        summary_df = summarize_results(args.input_dir, args.output_dir)
        print(f"Summary completed with {len(summary_df)} entries")
        
    elif args.action == "compress":
        print(f"Compressing results in {args.input_dir}")
        compressed_files = compress_results(
            args.input_dir, 
            args.output_dir,
            args.compress_format,
            args.delete_after_compress
        )
        print(f"Compression completed with {len(compressed_files)} archives")
        
    elif args.action == "extract_best":
        print(f"Extracting best results from {args.input_dir}")
        best_results = extract_best_results(
            args.input_dir,
            args.output_dir,
            args.min_metrics
        )
        print(f"Extracted {len(best_results)} best results")
        
    elif args.action == "clean":
        print("Cleaning functionality not yet implemented")
        
    elif args.action == "visualize":
        print("Visualization functionality not yet implemented")
    
    print(f"\nResults processing completed. Output saved to {args.output_dir}")

if __name__ == "__main__":
    main()
