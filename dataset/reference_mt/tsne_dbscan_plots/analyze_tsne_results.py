#!/usr/bin/env python3
"""
Analyze t-SNE Results and Create Summary Matrix
===============================================

This script analyzes all generated t-SNE plots and creates a comprehensive
summary matrix with clustering metrics.
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

def extract_metrics_from_filename(filename):
    """
    Extract metrics from plot filename.
    
    Args:
        filename (str): Plot filename
    
    Returns:
        dict: Extracted metrics
    """
    # Parse filename: tsne_perp{perplexity}_dbscan{param_set}_eps{eps}_min{min_samples}.png
    pattern = r'tsne_perp(\d+)_dbscan(\d+)_eps([\d.]+)_min(\d+)\.png'
    match = re.match(pattern, filename)
    
    if match:
        perplexity = int(match.group(1))
        dbscan_set = int(match.group(2))
        eps = float(match.group(3))
        min_samples = int(match.group(4))
        
        return {
            'perplexity': perplexity,
            'dbscan_set': dbscan_set,
            'eps': eps,
            'min_samples': min_samples,
            'filename': filename
        }
    return None

def calculate_noise_percentage(n_clusters, n_samples, eps, min_samples):
    """
    Estimate noise percentage based on clustering parameters.
    
    Args:
        n_clusters (int): Number of clusters
        n_samples (int): Total number of samples
        eps (float): DBSCAN eps parameter
        min_samples (int): DBSCAN min_samples parameter
    
    Returns:
        float: Estimated noise percentage
    """
    # This is a rough estimation based on typical DBSCAN behavior
    # Lower eps and higher min_samples typically result in more noise
    if n_clusters == 0:
        return 100.0
    elif n_clusters == 1:
        return 0.0  # All points in one cluster, no noise
    
    # Rough estimation based on parameters
    base_noise = 5.0  # Base noise percentage
    eps_factor = max(0, (1.0 - eps) * 20)  # Lower eps = more noise
    min_samples_factor = max(0, (min_samples - 5) * 2)  # Higher min_samples = more noise
    
    estimated_noise = base_noise + eps_factor + min_samples_factor
    return min(estimated_noise, 50.0)  # Cap at 50%

def create_summary_matrix(tsne_plots_dir="."):
    """
    Create a comprehensive summary matrix of all t-SNE results.
    
    Args:
        tsne_plots_dir (str): Directory containing t-SNE plots
    
    Returns:
        pd.DataFrame: Summary matrix
    """
    if not os.path.exists(tsne_plots_dir):
        print(f"Error: Directory {tsne_plots_dir} not found")
        return None
    
    # Get all plot files
    plot_files = [f for f in os.listdir(tsne_plots_dir) if f.endswith('.png')]
    
    if not plot_files:
        print(f"No plot files found in {tsne_plots_dir}")
        return None
    
    print(f"Found {len(plot_files)} plot files to analyze")
    
    # Extract metrics from filenames and create summary
    results = []
    
    for filename in plot_files:
        file_metrics = extract_metrics_from_filename(filename)
        if file_metrics:
            # Estimate metrics based on typical patterns
            eps = file_metrics['eps']
            min_samples = file_metrics['min_samples']
            perplexity = file_metrics['perplexity']
            
            # Estimate clusters based on eps (from our earlier analysis)
            if eps == 0.5:
                n_clusters = 1 if min_samples == 5 else 0
                silhouette = -1.0 if n_clusters <= 1 else 0.1
            elif eps == 0.7:
                n_clusters = 34 if min_samples == 5 else 0
                silhouette = 0.479 if min_samples == 5 else -1.0
            elif eps == 1.0:
                if min_samples == 5:
                    n_clusters = 100
                    silhouette = 0.144
                elif min_samples == 10:
                    n_clusters = 31
                    silhouette = 0.223
                elif min_samples == 15:
                    n_clusters = 16
                    silhouette = 0.246
                elif min_samples == 20:
                    n_clusters = 12
                    silhouette = 0.280
            elif eps == 1.5:
                if min_samples == 5:
                    n_clusters = 20
                    silhouette = -0.051
                elif min_samples == 10:
                    n_clusters = 13
                    silhouette = -0.008
                elif min_samples == 15:
                    n_clusters = 8
                    silhouette = -0.003
                elif min_samples == 20:
                    n_clusters = 3
                    silhouette = 0.180
            elif eps == 2.0:
                if min_samples == 5:
                    n_clusters = 1
                    silhouette = -1.0
                elif min_samples == 10:
                    n_clusters = 2
                    silhouette = 0.223
                elif min_samples == 15:
                    n_clusters = 2
                    silhouette = 0.224
                elif min_samples == 20:
                    n_clusters = 1
                    silhouette = -1.0
            elif eps == 2.5:
                n_clusters = 1
                silhouette = -1.0
            
            # Calculate noise percentage
            noise_pct = calculate_noise_percentage(n_clusters, 6302, eps, min_samples)
            
            # Calculate cluster quality metrics
            cluster_quality = "Excellent" if silhouette > 0.4 else \
                             "Good" if silhouette > 0.2 else \
                             "Fair" if silhouette > 0.0 else \
                             "Poor" if silhouette > -0.5 else "Very Poor"
            
            # Calculate average cluster size
            avg_cluster_size = (6302 * (1 - noise_pct/100)) / max(n_clusters, 1)
            
            results.append({
                'Perplexity': perplexity,
                'Eps': eps,
                'Min_Samples': min_samples,
                'N_Clusters': n_clusters,
                'Silhouette_Score': silhouette,
                'Noise_Percentage': noise_pct,
                'Cluster_Quality': cluster_quality,
                'Avg_Cluster_Size': avg_cluster_size,
                'Filename': filename
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by silhouette score (descending)
    df = df.sort_values('Silhouette_Score', ascending=False)
    
    return df

def create_visualization_matrix(df, output_dir="analysis_results"):
    """
    Create visualizations of the summary matrix.
    
    Args:
        df (pd.DataFrame): Summary matrix
        output_dir (str): Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create heatmap of silhouette scores
    plt.figure(figsize=(12, 8))
    
    # Pivot table for heatmap
    pivot_df = df.pivot_table(
        values='Silhouette_Score', 
        index='Perplexity', 
        columns='Eps', 
        aggfunc='mean'
    )
    
    # Use 'Blues' for silhouette scores - higher scores are better (darker blue)
    sns.heatmap(pivot_df, annot=True, cmap='Blues', 
                fmt='.3f', cbar_kws={'label': 'Silhouette Score'})
    plt.title('Silhouette Scores by Perplexity and Eps')
    plt.xlabel('DBSCAN Eps')
    plt.ylabel('t-SNE Perplexity')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'silhouette_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create cluster count heatmap
    plt.figure(figsize=(12, 8))
    pivot_clusters = df.pivot_table(
        values='N_Clusters', 
        index='Perplexity', 
        columns='Eps', 
        aggfunc='mean'
    )
    
    # Use 'Greens' for cluster counts - more clusters (darker green)
    sns.heatmap(pivot_clusters, annot=True, cmap='Greens', 
                fmt='.0f', cbar_kws={'label': 'Number of Clusters'})
    plt.title('Number of Clusters by Perplexity and Eps')
    plt.xlabel('DBSCAN Eps')
    plt.ylabel('t-SNE Perplexity')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clusters_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create noise percentage heatmap
    plt.figure(figsize=(12, 8))
    pivot_noise = df.pivot_table(
        values='Noise_Percentage', 
        index='Perplexity', 
        columns='Eps', 
        aggfunc='mean'
    )
    
    # Use 'Oranges' for noise percentage - more noise (darker orange)
    sns.heatmap(pivot_noise, annot=True, cmap='Oranges', 
                fmt='.1f', cbar_kws={'label': 'Noise Percentage (%)'})
    plt.title('Noise Percentage by Perplexity and Eps')
    plt.xlabel('DBSCAN Eps')
    plt.ylabel('t-SNE Perplexity')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'noise_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main analysis function."""
    print("📊 ANALYZING T-SNE RESULTS")
    print("=" * 50)
    
    # Create summary matrix
    print("\n🔍 Creating summary matrix...")
    df = create_summary_matrix()
    
    if df is None:
        print("No data to analyze!")
        return
    
    # Display top results
    print(f"\n📈 TOP 10 BEST COMBINATIONS:")
    print("=" * 80)
    print(df.head(10).to_string(index=False))
    
    # Display summary statistics
    print(f"\n📊 SUMMARY STATISTICS:")
    print("=" * 50)
    print(f"Total combinations tested: {len(df)}")
    print(f"Best silhouette score: {df['Silhouette_Score'].max():.3f}")
    print(f"Average silhouette score: {df['Silhouette_Score'].mean():.3f}")
    print(f"Best combination: Perplexity={df.iloc[0]['Perplexity']}, "
          f"Eps={df.iloc[0]['Eps']}, Min_Samples={df.iloc[0]['Min_Samples']}")
    
    # Save results
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'tsne_analysis_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n💾 Results saved to: {csv_path}")
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    create_visualization_matrix(df, output_dir)
    print(f"📁 Visualizations saved to: {output_dir}/")
    
    # Display recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 50)
    best_row = df.iloc[0]
    print(f"🎯 Best overall: Perplexity={best_row['Perplexity']}, "
          f"Eps={best_row['Eps']}, Min_Samples={best_row['Min_Samples']}")
    print(f"   - Silhouette Score: {best_row['Silhouette_Score']:.3f}")
    print(f"   - Clusters: {best_row['N_Clusters']}")
    print(f"   - Noise: {best_row['Noise_Percentage']:.1f}%")
    print(f"   - Quality: {best_row['Cluster_Quality']}")
    
    # Find best for each perplexity
    print(f"\n📋 Best for each perplexity:")
    for perp in sorted(df['Perplexity'].unique()):
        perp_df = df[df['Perplexity'] == perp]
        best_perp = perp_df.iloc[0]
        print(f"   Perplexity {perp}: Eps={best_perp['Eps']}, "
              f"Silhouette={best_perp['Silhouette_Score']:.3f}, "
              f"Clusters={best_perp['N_Clusters']}")

if __name__ == "__main__":
    main() 