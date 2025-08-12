#!/usr/bin/env python3
"""
Generate Clustering Summary Matrix
==================================

This script creates a confusion matrix-style visualization showing
noise percentages and silhouette scores for each clustering parameter combination.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_clustering_summary_matrix(csv_file, output_dir="clustering_summary"):
    """
    Create a confusion matrix-style visualization of clustering results.
    
    Args:
        csv_file (str): Path to clustering results CSV
        output_dir (str): Output directory for plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the clustering results
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} clustering results")
    print(f"Perplexity values: {sorted(df['perplexity'].unique())}")
    print(f"Eps values: {sorted(df['eps'].unique())}")
    print(f"Min_samples values: {sorted(df['min_samples'].unique())}")
    
    # Create parameter labels for better visualization
    df['eps_label'] = df['eps'].apply(lambda x: f"eps={x}")
    df['min_samples_label'] = df['min_samples'].apply(lambda x: f"min={x}")
    df['perplexity_label'] = df['perplexity'].apply(lambda x: f"perp={x}")
    
    # Create pivot tables for visualization
    # 1. Noise percentage matrix
    noise_matrix = df.pivot_table(
        values='noise_percentage', 
        index='perplexity_label', 
        columns=['eps_label', 'min_samples_label'],
        aggfunc='first'
    )
    
    # 2. Silhouette score matrix
    silhouette_matrix = df.pivot_table(
        values='silhouette_score', 
        index='perplexity_label', 
        columns=['eps_label', 'min_samples_label'],
        aggfunc='first'
    )
    
    # 3. Number of clusters matrix
    clusters_matrix = df.pivot_table(
        values='n_clusters', 
        index='perplexity_label', 
        columns=['eps_label', 'min_samples_label'],
        aggfunc='first'
    )
    
    # Create the visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Clustering Parameter Analysis Summary', fontsize=16, fontweight='bold')
    
    # 1. Noise Percentage Heatmap
    ax1 = axes[0, 0]
    sns.heatmap(noise_matrix, annot=True, fmt='.1f', cmap='Reds', 
                cbar_kws={'label': 'Noise Percentage (%)'}, ax=ax1)
    ax1.set_title('Noise Percentage by Parameters', fontweight='bold')
    ax1.set_xlabel('DBSCAN Parameters (eps, min_samples)')
    ax1.set_ylabel('t-SNE Perplexity')
    
    # 2. Silhouette Score Heatmap
    ax2 = axes[0, 1]
    sns.heatmap(silhouette_matrix, annot=True, fmt='.3f', cmap='Blues', 
                cbar_kws={'label': 'Silhouette Score'}, ax=ax2)
    ax2.set_title('Silhouette Score by Parameters', fontweight='bold')
    ax2.set_xlabel('DBSCAN Parameters (eps, min_samples)')
    ax2.set_ylabel('t-SNE Perplexity')
    
    # 3. Number of Clusters Heatmap
    ax3 = axes[1, 0]
    sns.heatmap(clusters_matrix, annot=True, fmt='.0f', cmap='Greens', 
                cbar_kws={'label': 'Number of Clusters'}, ax=ax3)
    ax3.set_title('Number of Clusters by Parameters', fontweight='bold')
    ax3.set_xlabel('DBSCAN Parameters (eps, min_samples)')
    ax3.set_ylabel('t-SNE Perplexity')
    
    # 4. Combined Quality Score (Silhouette * (1 - noise_percentage/100))
    df['quality_score'] = df['silhouette_score'] * (1 - df['noise_percentage']/100)
    quality_matrix = df.pivot_table(
        values='quality_score', 
        index='perplexity_label', 
        columns=['eps_label', 'min_samples_label'],
        aggfunc='first'
    )
    
    ax4 = axes[1, 1]
    sns.heatmap(quality_matrix, annot=True, fmt='.3f', cmap='Purples', 
                cbar_kws={'label': 'Quality Score'}, ax=ax4)
    ax4.set_title('Quality Score (Silhouette × Clustering Rate)', fontweight='bold')
    ax4.set_xlabel('DBSCAN Parameters (eps, min_samples)')
    ax4.set_ylabel('t-SNE Perplexity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clustering_summary_matrix.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a detailed summary table
    print("\n" + "="*80)
    print("CLUSTERING PARAMETER ANALYSIS SUMMARY")
    print("="*80)
    
    # Find best parameters for different criteria
    print(f"\n📊 BEST PARAMETERS BY CRITERIA:")
    print(f"=" * 50)
    
    # Best silhouette score
    best_sil = df.loc[df['silhouette_score'].idxmax()]
    print(f"🎯 Best Silhouette Score: {best_sil['silhouette_score']:.3f}")
    print(f"   Parameters: Perplexity={best_sil['perplexity']}, eps={best_sil['eps']}, min_samples={best_sil['min_samples']}")
    print(f"   Clusters: {best_sil['n_clusters']}, Noise: {best_sil['noise_percentage']:.1f}%")
    
    # Lowest noise percentage
    best_noise = df.loc[df['noise_percentage'].idxmin()]
    print(f"\n🔇 Lowest Noise: {best_noise['noise_percentage']:.1f}%")
    print(f"   Parameters: Perplexity={best_noise['perplexity']}, eps={best_noise['eps']}, min_samples={best_noise['min_samples']}")
    print(f"   Clusters: {best_noise['n_clusters']}, Silhouette: {best_noise['silhouette_score']:.3f}")
    
    # Most clusters
    most_clusters = df.loc[df['n_clusters'].idxmax()]
    print(f"\n🔢 Most Clusters: {most_clusters['n_clusters']}")
    print(f"   Parameters: Perplexity={most_clusters['perplexity']}, eps={most_clusters['eps']}, min_samples={most_clusters['min_samples']}")
    print(f"   Silhouette: {most_clusters['silhouette_score']:.3f}, Noise: {most_clusters['noise_percentage']:.1f}%")
    
    # Best quality score
    best_quality = df.loc[df['quality_score'].idxmax()]
    print(f"\n⭐ Best Quality Score: {best_quality['quality_score']:.3f}")
    print(f"   Parameters: Perplexity={best_quality['perplexity']}, eps={best_quality['eps']}, min_samples={best_quality['min_samples']}")
    print(f"   Silhouette: {best_quality['silhouette_score']:.3f}, Noise: {best_quality['noise_percentage']:.1f}%")
    
    # Create a detailed CSV with all metrics
    summary_df = df[['perplexity', 'eps', 'min_samples', 'n_clusters', 
                     'silhouette_score', 'noise_percentage', 'quality_score']].copy()
    summary_df = summary_df.sort_values('quality_score', ascending=False)
    summary_df.to_csv(os.path.join(output_dir, 'detailed_clustering_summary.csv'), index=False)
    
    print(f"\n📁 Files saved to: {output_dir}/")
    print(f"   - clustering_summary_matrix.png (visualization)")
    print(f"   - detailed_clustering_summary.csv (detailed data)")
    
    return summary_df

def create_parameter_trend_plots(csv_file, output_dir="clustering_summary"):
    """
    Create trend plots showing how parameters affect clustering.
    
    Args:
        csv_file (str): Path to clustering results CSV
        output_dir (str): Output directory for plots
    """
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Calculate quality score
    df['quality_score'] = df['silhouette_score'] * (1 - df['noise_percentage']/100)
    
    # Create trend plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Clustering Parameter Trends', fontsize=16, fontweight='bold')
    
    # 1. Noise percentage vs eps for different perplexities
    ax1 = axes[0, 0]
    for perplexity in sorted(df['perplexity'].unique()):
        subset = df[df['perplexity'] == perplexity]
        subset = subset.groupby('eps')['noise_percentage'].mean().reset_index()
        ax1.plot(subset['eps'], subset['noise_percentage'], 
                marker='o', label=f'Perplexity={perplexity}')
    ax1.set_xlabel('DBSCAN eps')
    ax1.set_ylabel('Average Noise Percentage (%)')
    ax1.set_title('Noise vs eps by Perplexity')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Silhouette score vs eps for different perplexities
    ax2 = axes[0, 1]
    for perplexity in sorted(df['perplexity'].unique()):
        subset = df[df['perplexity'] == perplexity]
        subset = subset.groupby('eps')['silhouette_score'].mean().reset_index()
        ax2.plot(subset['eps'], subset['silhouette_score'], 
                marker='s', label=f'Perplexity={perplexity}')
    ax2.set_xlabel('DBSCAN eps')
    ax2.set_ylabel('Average Silhouette Score')
    ax2.set_title('Silhouette vs eps by Perplexity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Number of clusters vs min_samples for different eps
    ax3 = axes[1, 0]
    for eps in sorted(df['eps'].unique()):
        subset = df[df['eps'] == eps]
        subset = subset.groupby('min_samples')['n_clusters'].mean().reset_index()
        ax3.plot(subset['min_samples'], subset['n_clusters'], 
                marker='^', label=f'eps={eps}')
    ax3.set_xlabel('DBSCAN min_samples')
    ax3.set_ylabel('Average Number of Clusters')
    ax3.set_title('Clusters vs min_samples by eps')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Quality score heatmap
    ax4 = axes[1, 1]
    quality_pivot = df.pivot_table(
        values='quality_score', 
        index='perplexity', 
        columns='eps', 
        aggfunc='mean'
    )
    sns.heatmap(quality_pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax4)
    ax4.set_title('Average Quality Score by Perplexity and eps')
    ax4.set_xlabel('DBSCAN eps')
    ax4.set_ylabel('t-SNE Perplexity')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_trends.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Parameter trend plots saved to: {output_dir}/parameter_trends.png")

def main():
    """Main function to generate clustering summary."""
    print("🔬 GENERATING CLUSTERING SUMMARY MATRIX")
    print("=" * 50)
    
    # Define input file
    csv_file = "tsne_plots/results/clustering_results.csv"
    
    if not os.path.exists(csv_file):
        print(f"Error: Clustering results file not found: {csv_file}")
        print("Please run the main analysis script first.")
        return
    
    # Generate summary matrix
    print(f"\n📊 Creating clustering summary matrix...")
    summary_df = create_clustering_summary_matrix(csv_file)
    
    # Generate trend plots
    print(f"\n📈 Creating parameter trend plots...")
    create_parameter_trend_plots(csv_file)
    
    print(f"\n✅ Clustering summary generation complete!")
    print(f"📁 Check the 'clustering_summary/' directory for all visualizations")

if __name__ == "__main__":
    main() 