#!/usr/bin/env python3
"""
Script to create and visualize the correlation matrix of protein features for supervised_2 dataset.
Loads embeddings from embedded_non_sam and embedded_sam directories.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
import os
import torch
import glob
from tqdm import tqdm

def load_embeddings_from_directory(directory_path, label, is_nested=False):
    """Load all embedding files from a directory and return as numpy array with labels."""
    embeddings = []
    labels = []
    file_names = []
    
    if is_nested:
        # Handle nested directory structure (like embedded_non_sam)
        for subdir in os.listdir(directory_path):
            subdir_path = os.path.join(directory_path, subdir)
            if os.path.isdir(subdir_path):
                for file_name in os.listdir(subdir_path):
                    if file_name.endswith('.pt'):
                        file_path = os.path.join(subdir_path, file_name)
                        try:
                            # Load the embedding dictionary
                            embedding_dict = torch.load(file_path, map_location='cpu')
                            
                            # Extract the mean representation from layer 33 (final layer)
                            if isinstance(embedding_dict, dict) and 'mean_representations' in embedding_dict:
                                embedding = embedding_dict['mean_representations'][33].numpy()
                            else:
                                # Fallback: try to convert to numpy directly
                                embedding = embedding_dict.numpy() if isinstance(embedding_dict, torch.Tensor) else np.array(embedding_dict)
                            
                            # Flatten if it's multi-dimensional
                            if embedding.ndim > 1:
                                embedding = embedding.flatten()
                            
                            embeddings.append(embedding)
                            labels.append(label)
                            file_names.append(f"{subdir}/{file_name}")
                            
                        except Exception as e:
                            print(f"Error loading {subdir}/{file_name}: {e}")
    else:
        # Handle flat directory structure (like embedded_sam)
        for file_name in os.listdir(directory_path):
            if file_name.endswith('.pt'):
                file_path = os.path.join(directory_path, file_name)
                try:
                    # Load the embedding dictionary
                    embedding_dict = torch.load(file_path, map_location='cpu')
                    
                    # Extract the mean representation from layer 33 (final layer)
                    if isinstance(embedding_dict, dict) and 'mean_representations' in embedding_dict:
                        embedding = embedding_dict['mean_representations'][33].numpy()
                    else:
                        # Fallback: try to convert to numpy directly
                        embedding = embedding_dict.numpy() if isinstance(embedding_dict, torch.Tensor) else np.array(embedding_dict)
                    
                    # Flatten if it's multi-dimensional
                    if embedding.ndim > 1:
                        embedding = embedding.flatten()
                    
                    embeddings.append(embedding)
                    labels.append(label)
                    file_names.append(file_name)
                    
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
    
    return np.array(embeddings), np.array(labels), file_names

def load_features():
    """Load the features from embedded_non_sam and embedded_sam directories."""
    print("Loading features from embedded_non_sam and embedded_sam directories...")
    
    # Load SAM-dependent methyltransferases (positive class)
    print("Loading SAM-dependent methyltransferases...")
    sam_features, sam_labels, sam_files = load_embeddings_from_directory('supervised_2/embedded_sam', 1, is_nested=False)
    print(f"Loaded {len(sam_features)} SAM embeddings with shape {sam_features.shape}")
    
    # Load non-SAM proteins (negative class)
    print("Loading non-SAM proteins...")
    non_sam_features, non_sam_labels, non_sam_files = load_embeddings_from_directory('supervised_2/embedded_non_sam', 0, is_nested=True)
    print(f"Loaded {len(non_sam_features)} non-SAM embeddings with shape {non_sam_features.shape}")
    
    # Combine datasets
    features = np.vstack([sam_features, non_sam_features])
    labels = np.hstack([sam_labels, non_sam_labels])
    
    print(f"Combined dataset shape: {features.shape}")
    print(f"SAM methyltransferase samples: {sum(labels)}")
    print(f"Non-SAM protein samples: {len(labels) - sum(labels)}")
    
    # Check if all embeddings have the same dimension
    if len(set(emb.shape[0] for emb in features)) > 1:
        print("Warning: Embeddings have different dimensions!")
        # Find the minimum dimension
        min_dim = min(emb.shape[0] for emb in features)
        print(f"Truncating all embeddings to {min_dim} dimensions")
        features = np.array([emb[:min_dim] for emb in features])
    
    return features, labels

def create_correlation_matrix(features, output_dir='.'):
    """Create and save the correlation matrix."""
    print("Calculating correlation matrix...")
    
    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(features.T)
    
    print(f"Correlation matrix shape: {correlation_matrix.shape}")
    print(f"Memory usage: {correlation_matrix.nbytes / 1024**2:.2f} MB")
    
    # Save the correlation matrix
    output_file = os.path.join(output_dir, 'correlation_matrix.npy')
    np.save(output_file, correlation_matrix)
    print(f"Saved correlation matrix to {output_file}")
    
    return correlation_matrix

def analyze_correlations(correlation_matrix):
    """Analyze the correlation patterns."""
    print("\n=== CORRELATION ANALYSIS ===")
    
    # Get upper triangle (avoid duplicates and diagonal)
    upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
    abs_correlations = np.abs(upper_triangle)
    
    # Basic statistics
    print(f"Mean absolute correlation: {np.mean(abs_correlations):.4f}")
    print(f"Std absolute correlation: {np.std(abs_correlations):.4f}")
    print(f"Min absolute correlation: {np.min(abs_correlations):.4f}")
    print(f"Max absolute correlation: {np.max(abs_correlations):.4f}")
    
    # Count correlations above different thresholds
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98]
    print(f"\nCorrelation distribution:")
    for thresh in thresholds:
        count = np.sum(abs_correlations >= thresh)
        percentage = (count / len(abs_correlations)) * 100
        print(f"|r| >= {thresh}: {count:,} pairs ({percentage:.2f}%)")
    
    # Find most correlated feature pairs
    print(f"\nTop 20 most correlated feature pairs:")
    # Get indices of upper triangle
    triu_indices = np.triu_indices_from(correlation_matrix, k=1)
    
    # Create list of (i, j, correlation) tuples
    correlations = []
    for i, j in zip(triu_indices[0], triu_indices[1]):
        correlations.append((i, j, abs(correlation_matrix[i, j])))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: x[2], reverse=True)
    
    for idx, (i, j, corr) in enumerate(correlations[:20]):
        print(f"{idx+1:2d}. Features {i:4d} and {j:4d}: |r| = {corr:.4f}")
    
    return abs_correlations

def create_correlation_heatmap(correlation_matrix, output_dir='.'):
    """Create a heatmap visualization of the correlation matrix."""
    print("\nCreating correlation heatmap...")
    
    # Create figure with appropriate size
    plt.figure(figsize=(12, 10))
    
    # Create heatmap
    # Note: For large matrices, we'll create a downsampled version for visualization
    if correlation_matrix.shape[0] > 100:
        print("Matrix is too large for full visualization. Creating downsampled version...")
        
        # Sample every nth feature for visualization
        n = correlation_matrix.shape[0] // 100
        sampled_matrix = correlation_matrix[::n, ::n]
        
        # Create heatmap
        sns.heatmap(sampled_matrix, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   xticklabels=False,
                   yticklabels=False,
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title(f'Feature Correlation Matrix (Sampled: every {n}th feature)\n{correlation_matrix.shape[0]} features total')
    else:
        # Full heatmap for smaller matrices
        sns.heatmap(correlation_matrix, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   xticklabels=False,
                   yticklabels=False,
                   cbar_kws={'label': 'Correlation Coefficient'})
        
        plt.title(f'Feature Correlation Matrix ({correlation_matrix.shape[0]} features)')
    
    # Save the plot
    output_file = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved correlation heatmap to {output_file}")
    plt.close()

def create_correlation_histogram(abs_correlations, output_dir='.'):
    """Create a histogram of correlation values."""
    print("Creating correlation histogram...")
    
    plt.figure(figsize=(10, 6))
    
    # Create histogram
    plt.hist(abs_correlations, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Absolute Correlation Coefficient')
    plt.ylabel('Frequency of Feature Pairs')
    plt.title('Distribution of Absolute Feature Correlations')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for common thresholds
    thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
    colors = ['orange', 'red', 'purple', 'brown', 'black']
    
    for thresh, color in zip(thresholds, colors):
        count = np.sum(abs_correlations >= thresh)
        plt.axvline(x=thresh, color=color, linestyle='--', alpha=0.7,
                   label=f'|r| ≥ {thresh}: {count:,} pairs')
    
    plt.legend()
    plt.xlim(0, 1)
    
    # Save the plot
    output_file = os.path.join(output_dir, 'correlation_histogram.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved correlation histogram to {output_file}")
    plt.close()

def main():
    print("=== FEATURE CORRELATION MATRIX ANALYSIS (SUPERVISED_2) ===")
    
    # Load features
    features, labels = load_features()
    
    # Create correlation matrix
    correlation_matrix = create_correlation_matrix(features)
    
    # Analyze correlations
    abs_correlations = analyze_correlations(correlation_matrix)
    
    # Create visualizations
    create_correlation_heatmap(correlation_matrix)
    create_correlation_histogram(abs_correlations)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Generated files:")
    print("- correlation_matrix.npy: Raw correlation matrix")
    print("- correlation_heatmap.png: Visual heatmap")
    print("- correlation_histogram.png: Correlation distribution")

if __name__ == "__main__":
    main() 