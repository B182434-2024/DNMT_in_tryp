#!/usr/bin/env python3
"""
t-SNE Analysis with DNMT Cluster Highlighting - Fixed Version
=============================================================

This script performs:
1. Loads methyltransferase embeddings from individual .pt files (including subdirectories)
2. Runs t-SNE dimensionality reduction with perplexities: 10, 30, 50, 70
3. Labels samples red if they are in the DNMT cluster
4. Creates visualizations for each perplexity
"""

import os
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_dnmt_ids(filepath):
    """
    Load DNMT cluster UniProt IDs from file.
    
    Args:
        filepath (str): Path to the DNMT IDs file
    
    Returns:
        set: Set of DNMT UniProt IDs
    """
    dnmt_ids = set()
    try:
        with open(filepath, 'r') as f:
            for line in f:
                dnmt_ids.add(line.strip())
        print(f"Loaded {len(dnmt_ids)} DNMT cluster IDs")
        return dnmt_ids
    except FileNotFoundError:
        print(f"Warning: DNMT IDs file {filepath} not found")
        return set()

def extract_uniprot_id_from_path(filepath):
    """
    Extract UniProt ID from filepath (handles both files and subdirectories).
    
    Args:
        filepath (str): Full path to the embedding file
    
    Returns:
        str: UniProt ID or None if not found
    """
    import re
    
    # Get the directory name (in case it's a subdirectory)
    dirname = os.path.dirname(filepath)
    basename = os.path.basename(filepath)
    
    # First try to extract from the directory name (for subdirectories)
    if dirname != 'embedded_methyltransferase':
        dir_basename = os.path.basename(dirname)
        match = re.search(r'^(sp|tr)\|([A-Z0-9]+)\|', dir_basename)
        if match:
            return match.group(2)
    
    # If not found in directory, try the filename
    match = re.search(r'^(sp|tr)\|([A-Z0-9]+)\|', basename)
    if match:
        return match.group(2)
    
    return None

def load_methyltransferase_embeddings(directory):
    """
    Load all methyltransferase embeddings from the directory.
    
    Args:
        directory (str): Path to directory containing embedding files
    
    Returns:
        tuple: (embeddings, filenames, uniprot_ids)
    """
    embeddings = []
    filenames = []
    uniprot_ids = []
    
    if not os.path.exists(directory):
        print(f"Error: Directory {directory} does not exist")
        return [], [], []
    
    # Get all .pt files recursively
    pt_files = glob.glob(os.path.join(directory, "**", "*.pt"), recursive=True)
    print(f"Found {len(pt_files)} embedding files")
    
    for filepath in tqdm(pt_files, desc="Loading embeddings"):
        try:
            # Load the embedding
            embedding = torch.load(filepath, map_location='cpu')
            
            # Handle different embedding formats
            if isinstance(embedding, dict):
                if 'mean_representations' in embedding:
                    # ESM format
                    embedding = embedding['mean_representations'][33].numpy()
                elif 'embeddings' in embedding:
                    embedding = embedding['embeddings'].numpy()
                elif 'features' in embedding:
                    embedding = embedding['features'].numpy()
                else:
                    # Try to find any tensor in the dict
                    for key, value in embedding.items():
                        if isinstance(value, torch.Tensor):
                            embedding = value.numpy()
                            break
                    else:
                        continue
            elif isinstance(embedding, torch.Tensor):
                embedding = embedding.numpy()
            else:
                continue
            
            # Handle different embedding dimensions
            if embedding.ndim > 1:
                embedding = np.mean(embedding, axis=0)
            
            # Extract filename and UniProt ID
            filename = os.path.basename(filepath)
            uniprot_id = extract_uniprot_id_from_path(filepath)
            
            embeddings.append(embedding)
            filenames.append(filename)
            uniprot_ids.append(uniprot_id)
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    return np.array(embeddings), filenames, uniprot_ids

def run_tsne(embeddings, perplexity, random_state=42):
    """
    Run t-SNE dimensionality reduction.
    
    Args:
        embeddings (np.array): Input embeddings
        perplexity (int): t-SNE perplexity parameter
        random_state (int): Random seed
    
    Returns:
        np.array: 2D t-SNE embeddings
    """
    print(f"Running t-SNE with perplexity={perplexity}...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, n_jobs=-1)
    tsne_embeddings = tsne.fit_transform(embeddings)
    return tsne_embeddings

def create_tsne_plot(tsne_embeddings, uniprot_ids, dnmt_ids, perplexity, output_dir="tsne_dnmt_plots_fixed"):
    """
    Create t-SNE plot with DNMT cluster highlighting.
    
    Args:
        tsne_embeddings (np.array): 2D t-SNE embeddings
        uniprot_ids (list): List of UniProt IDs
        dnmt_ids (set): Set of DNMT cluster UniProt IDs
        perplexity (int): t-SNE perplexity used
        output_dir (str): Output directory for plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create color mapping
    colors = []
    dnmt_count = 0
    other_count = 0
    
    for uniprot_id in uniprot_ids:
        if uniprot_id in dnmt_ids:
            colors.append('red')
            dnmt_count += 1
        else:
            colors.append('lightblue')
            other_count += 1
    
    # Create the plot
    plt.figure(figsize=(12, 10))
    
    # Plot points
    plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
               c=colors, alpha=0.7, s=30, edgecolors='black', linewidth=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label=f'DNMT Cluster ({dnmt_count} samples)'),
        Patch(facecolor='lightblue', alpha=0.7, label=f'Other Methyltransferases ({other_count} samples)')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Customize plot
    plt.title(f't-SNE of Methyltransferase Embeddings (Perplexity = {perplexity})', 
              fontsize=16, fontweight='bold')
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Total samples: {len(tsne_embeddings)}\nDNMT cluster: {dnmt_count}\nOther: {other_count}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    output_file = os.path.join(output_dir, f'tsne_perplexity_{perplexity}_dnmt_highlight.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {output_file}")
    
    return dnmt_count, other_count

def main():
    """
    Main function to run the t-SNE analysis with DNMT highlighting.
    """
    print("Starting t-SNE Analysis with DNMT Cluster Highlighting (Fixed Version)")
    print("=" * 70)
    
    # Load DNMT cluster IDs
    dnmt_ids = load_dnmt_ids('uniprot_ids_clustered_sam_dnmt.txt')
    
    # Load embeddings
    embeddings_dir = 'embedded_methyltransferase'
    embeddings, filenames, uniprot_ids = load_methyltransferase_embeddings(embeddings_dir)
    
    if len(embeddings) == 0:
        print("No embeddings loaded. Exiting.")
        return
    
    print(f"Loaded {len(embeddings)} embeddings with shape: {embeddings.shape}")
    
    # Count DNMT matches
    dnmt_matches = sum(1 for uid in uniprot_ids if uid in dnmt_ids)
    print(f"Found {dnmt_matches} DNMT cluster matches out of {len(uniprot_ids)} total embeddings")
    
    # Show some examples of DNMT matches
    print("\nFirst 10 DNMT matches found:")
    count = 0
    for i, uid in enumerate(uniprot_ids):
        if uid in dnmt_ids and count < 10:
            print(f"  {uid} -> {filenames[i][:60]}...")
            count += 1
    
    # Standardize embeddings
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Define perplexities
    perplexities = [10, 30, 50, 70]
    
    # Create results summary
    results_summary = []
    
    # Run t-SNE for each perplexity
    for perplexity in perplexities:
        print(f"\n{'='*20} Perplexity: {perplexity} {'='*20}")
        
        # Run t-SNE
        tsne_embeddings = run_tsne(embeddings_scaled, perplexity)
        
        # Create plot and get counts
        dnmt_count, other_count = create_tsne_plot(
            tsne_embeddings, uniprot_ids, dnmt_ids, perplexity
        )
        
        # Store results
        results_summary.append({
            'perplexity': perplexity,
            'total_samples': len(tsne_embeddings),
            'dnmt_count': dnmt_count,
            'other_count': other_count,
            'dnmt_percentage': (dnmt_count / len(tsne_embeddings)) * 100
        })
    
    # Create summary plot
    create_summary_plot(results_summary)
    
    # Save results to CSV
    results_df = pd.DataFrame(results_summary)
    results_df.to_csv('tsne_dnmt_analysis_results_fixed.csv', index=False)
    print(f"\nResults saved to: tsne_dnmt_analysis_results_fixed.csv")
    
    # Print summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    for result in results_summary:
        print(f"Perplexity {result['perplexity']:2d}: "
              f"Total={result['total_samples']:4d}, "
              f"DNMT={result['dnmt_count']:4d} ({result['dnmt_percentage']:5.1f}%), "
              f"Other={result['other_count']:4d}")

def create_summary_plot(results_summary):
    """
    Create a summary plot showing DNMT cluster distribution across perplexities.
    
    Args:
        results_summary (list): List of result dictionaries
    """
    perplexities = [r['perplexity'] for r in results_summary]
    dnmt_counts = [r['dnmt_count'] for r in results_summary]
    other_counts = [r['other_count'] for r in results_summary]
    
    plt.figure(figsize=(10, 6))
    
    # Create stacked bar chart
    x = np.arange(len(perplexities))
    width = 0.35
    
    plt.bar(x, dnmt_counts, width, label='DNMT Cluster', color='red', alpha=0.7)
    plt.bar(x, other_counts, width, bottom=dnmt_counts, label='Other Methyltransferases', 
            color='lightblue', alpha=0.7)
    
    # Customize plot
    plt.xlabel('t-SNE Perplexity', fontsize=14)
    plt.ylabel('Number of Samples', fontsize=14)
    plt.title('Distribution of DNMT Cluster vs Other Methyltransferases\nAcross Different t-SNE Perplexities', 
              fontsize=16, fontweight='bold')
    plt.xticks(x, perplexities)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (dnmt, other) in enumerate(zip(dnmt_counts, other_counts)):
        plt.text(i, dnmt/2, str(dnmt), ha='center', va='center', fontweight='bold')
        plt.text(i, dnmt + other/2, str(other), ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('tsne_dnmt_summary_plot_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Saved summary plot: tsne_dnmt_summary_plot_fixed.png")

if __name__ == "__main__":
    main() 