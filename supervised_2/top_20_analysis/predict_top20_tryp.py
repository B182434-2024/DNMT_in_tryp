#!/usr/bin/env python3
"""
Predict SAM/Non-SAM Classes for embedded_tryp Using Top 20 Features Model
=======================================================================

This script loads the trained GaussianNB model (top 20 features) and predicts
classes for all samples in the unknown set ../embedded_tryp. Results are saved
to a file for downstream analysis.
"""

import torch
import numpy as np
import os
import glob
import pickle
import pandas as pd
from tqdm import tqdm
# Add plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

# --- Utility Function ---
def load_embeddings(embedding_dir):
    """
    Load protein embeddings from a directory (no labels).
    Args:
        embedding_dir: Path to directory containing .pt files
    Returns:
        embeddings: numpy array of shape (n_samples, n_features)
        file_list: list of file paths (for reference)
    """
    embeddings = []
    file_list = []
    pt_files = glob.glob(os.path.join(embedding_dir, "**", "*.pt"), recursive=True)
    print(f"Loading {len(pt_files)} files from {embedding_dir}...")
    for pt_file in tqdm(pt_files):
        try:
            embedding_data = torch.load(pt_file, map_location='cpu')
            if isinstance(embedding_data, dict) and 'mean_representations' in embedding_data:
                embedding_tensor = embedding_data['mean_representations'][33]
            else:
                embedding_tensor = embedding_data
            if isinstance(embedding_tensor, torch.Tensor):
                embedding_array = embedding_tensor.numpy()
            else:
                embedding_array = np.array(embedding_tensor)
            if embedding_array.ndim > 1:
                embedding_array = embedding_array.flatten()
            embeddings.append(embedding_array)
            file_list.append(pt_file)
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
            continue
    return np.array(embeddings), file_list

if __name__ == "__main__":
    # Load model, scaler, and feature indices
    with open('top20_sam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('top20_sam_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('top20_sam_feature_indices.pkl', 'rb') as f:
        feature_indices = pickle.load(f)

    # Load unknown embeddings
    X_tryp, tryp_files = load_embeddings('../embedded_tryp')
    if len(X_tryp) == 0:
        print("No data found in ../embedded_tryp.")
        exit(0)
    X_tryp_reduced = X_tryp[:, feature_indices]
    X_tryp_scaled = scaler.transform(X_tryp_reduced)
    y_tryp_pred = model.predict(X_tryp_scaled)

    # Print class counts
    unique, counts = np.unique(y_tryp_pred, return_counts=True)
    print("Predicted class counts:")
    for u, c in zip(unique, counts):
        label = 'Non-SAM' if u == 0 else 'SAM'
        print(f"  {label}: {c}")

    # --- Plot 1: Bar plot of predicted class counts ---
    plt.figure(figsize=(6,5))
    class_labels = ['Non-SAM', 'SAM']
    count_dict = {label: 0 for label in class_labels}
    for u, c in zip(unique, counts):
        count_dict[class_labels[u]] = c
    
    bars = plt.bar(list(count_dict.keys()), list(count_dict.values()), color=['lightcoral', 'lightblue'], alpha=0.8)
    plt.ylabel('Count', fontsize=12)
    plt.title('Predicted Class Counts', fontsize=14, fontweight='bold')
    
    # Add count values on top of each bar
    for bar, count in zip(bars, count_dict.values()):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(count_dict.values()), 
                str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('predicted_class_counts_embedded_tryp.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 2: Violin plots for first 3 top features by predicted class ---
    n_violin = min(3, len(feature_indices))
    for i in range(n_violin):
        feature_idx = feature_indices[i]
        data = []
        for pred_class in [0, 1]:
            values = X_tryp[y_tryp_pred == pred_class, feature_idx]
            data.append(values)
        plt.figure(figsize=(6,4))
        sns.violinplot(data=data, palette='pastel')
        plt.xticks([0,1], class_labels)
        plt.ylabel(f'Feature {feature_idx}')
        plt.title(f'Feature {feature_idx} Distribution by Predicted Class')
        plt.tight_layout()
        plt.savefig(f'violin_feature{feature_idx}_by_class_embedded_tryp.png')
        plt.close()

    # --- Plot 3: Histogram of prediction probabilities for SAM class (if available) ---
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_tryp_scaled)[:, 1]
        plt.figure(figsize=(6,4))
        sns.histplot(y_proba, bins=20, kde=True, color='skyblue')
        plt.xlabel('Predicted Probability (SAM)')
        plt.ylabel('Count')
        plt.title('Distribution of Predicted Probabilities for SAM (embedded_tryp)')
        plt.tight_layout()
        plt.savefig('predicted_probabilities_sam_embedded_tryp.png')
        plt.close()

    # --- Plot 4: Pie chart of class distribution ---
    plt.figure(figsize=(8,6))
    colors = ['lightcoral', 'lightblue']
    plt.pie(count_dict.values(), labels=count_dict.keys(), autopct='%1.1f%%', 
            colors=colors, startangle=90, textprops={'fontsize': 12})
    plt.title('Class Distribution', fontsize=14, fontweight='bold', pad=20)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('class_distribution_pie.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 5: Feature importance visualization (if model supports it) ---
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(10,6))
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': [f'Feature_{i}' for i in feature_indices],
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        plt.barh(range(len(feature_importance_df)), feature_importance_df['Importance'], 
                color='skyblue', alpha=0.8)
        plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
        plt.xlabel('Feature Importance', fontsize=12)
        plt.title('Top Feature Importances', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()

    # --- Plot 6: Prediction confidence distribution ---
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_tryp_scaled)
        confidence = np.max(y_proba, axis=1)
        
        plt.figure(figsize=(8,6))
        plt.hist(confidence, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Prediction Confidence', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Distribution of Prediction Confidence', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('prediction_confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

    # --- Plot 7: Confusion matrix style plot for predictions ---
    plt.figure(figsize=(6,5))
    # Create a simple visualization showing prediction distribution
    sam_count = count_dict['SAM']
    non_sam_count = count_dict['Non-SAM']
    total = sam_count + non_sam_count
    
    # Create a stacked bar showing proportions
    categories = ['Predicted\nDistribution']
    sam_prop = [sam_count/total * 100]
    non_sam_prop = [non_sam_count/total * 100]
    
    plt.bar(categories, sam_prop, label='SAM', color='lightblue', alpha=0.8)
    plt.bar(categories, non_sam_prop, bottom=sam_prop, label='Non-SAM', color='lightcoral', alpha=0.8)
    
    # Add percentage labels
    plt.text(0, sam_prop[0]/2, f'{sam_prop[0]:.1f}%', ha='center', va='center', fontweight='bold')
    plt.text(0, sam_prop[0] + non_sam_prop[0]/2, f'{non_sam_prop[0]:.1f}%', ha='center', va='center', fontweight='bold')
    
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.title('Predicted Class Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('predicted_distribution_percentage.png', dpi=300, bbox_inches='tight')
    plt.close()

    # --- Plot 8: Box plots for top features by predicted class ---
    n_box = min(3, len(feature_indices))
    fig, axes = plt.subplots(1, n_box, figsize=(15, 5))
    if n_box == 1:
        axes = [axes]
    
    for i in range(n_box):
        feature_idx = feature_indices[i]
        sam_values = X_tryp[y_tryp_pred == 1, feature_idx]
        non_sam_values = X_tryp[y_tryp_pred == 0, feature_idx]
        
        data = [non_sam_values, sam_values]
        labels = ['Non-SAM', 'SAM']
        
        axes[i].boxplot(data, labels=labels, patch_artist=True)
        axes[i].set_ylabel(f'Feature {feature_idx}', fontsize=10)
        axes[i].set_title(f'Feature {feature_idx} by Predicted Class', fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_boxplots_by_class.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save predictions to file
    with open('embedded_tryp_predictions.txt', 'w') as f:
        for file, pred in zip(tryp_files, y_tryp_pred):
            label = 'Non-SAM' if pred == 0 else 'SAM'
            f.write(f"{file}\t{label}\n")
    print("Predictions saved to embedded_tryp_predictions.txt")
    
    # Save predictions with confidence scores
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_tryp_scaled)
        confidence = np.max(y_proba, axis=1)
        
        with open('embedded_tryp_predictions_with_confidence.txt', 'w') as f:
            f.write("File\tPrediction\tConfidence\tSAM_Probability\tNonSAM_Probability\n")
            for file, pred, conf, sam_prob, nonsam_prob in zip(tryp_files, y_tryp_pred, confidence, y_proba[:, 1], y_proba[:, 0]):
                label = 'Non-SAM' if pred == 0 else 'SAM'
                f.write(f"{file}\t{label}\t{conf:.4f}\t{sam_prob:.4f}\t{nonsam_prob:.4f}\n")
        print("Predictions with confidence scores saved to embedded_tryp_predictions_with_confidence.txt")
        
        # Print summary statistics
        print(f"\nConfidence Score Summary:")
        print(f"  Mean confidence: {np.mean(confidence):.4f}")
        print(f"  Median confidence: {np.median(confidence):.4f}")
        print(f"  Min confidence: {np.min(confidence):.4f}")
        print(f"  Max confidence: {np.max(confidence):.4f}")
        print(f"  Std confidence: {np.std(confidence):.4f}")
        
        # Count high-confidence predictions
        high_conf_threshold = 0.8
        high_conf_count = np.sum(confidence >= high_conf_threshold)
        print(f"  High-confidence predictions (≥{high_conf_threshold}): {high_conf_count}/{len(confidence)} ({high_conf_count/len(confidence)*100:.1f}%)") 