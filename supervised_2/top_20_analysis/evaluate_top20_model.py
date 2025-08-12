#!/usr/bin/env python3
"""
Evaluate Top 20 Features SAM Classification Model
================================================

This script evaluates the saved GaussianNB model (trained on top 20 features)
on the test and holdout sets, reporting accuracy and classification metrics.
"""

import torch
import numpy as np
import os
import glob
import pickle
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score
# Add imports for plotting and additional metrics
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, roc_auc_score

# --- Utility Functions ---
def load_embeddings_with_cath_label(embedding_dir):
    """
    Load protein embeddings from a directory and label as 0 if 'cath' in path, else 1.
    Args:
        embedding_dir: Path to directory containing .pt files
    Returns:
        embeddings: numpy array of shape (n_samples, n_features)
        labels: numpy array of labels
        file_list: list of file paths (for reference)
    """
    embeddings = []
    labels = []
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
            label = 0 if 'cath' in pt_file.lower() else 1
            labels.append(label)
            file_list.append(pt_file)
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
            continue
    return np.array(embeddings), np.array(labels), file_list

# --- Main Evaluation ---
def evaluate_on_directory(embedding_dir, model, scaler, feature_indices, set_name):
    print(f"\n=== Evaluating on {set_name} ===")
    X, y, files = load_embeddings_with_cath_label(embedding_dir)
    if len(X) == 0:
        print(f"No data found in {embedding_dir}.")
        return
    X_reduced = X[:, feature_indices]
    X_scaled = scaler.transform(X_reduced)
    y_pred = model.predict(X_scaled)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print(f"Samples: {len(y)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Classification Report:")
    print(classification_report(y, y_pred, target_names=['Non-SAM (cath)', 'SAM']))

    # --- Plot Confusion Matrix ---
    cm_filename = f"confusion_matrix_{set_name.lower().replace(' ', '_')}.png"
    disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, display_labels=['Non-SAM (cath)', 'SAM'], cmap=plt.cm.Blues)
    disp.ax_.set_title(f'Confusion Matrix: {set_name}')
    plt.savefig(cm_filename, bbox_inches='tight')
    plt.close()

    # --- Plot ROC Curve ---
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc = roc_auc_score(y, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {set_name}')
        plt.legend(loc='lower right')
        roc_filename = f"roc_curve_{set_name.lower().replace(' ', '_')}.png"
        plt.savefig(roc_filename, bbox_inches='tight')
        plt.close()
    else:
        print("Model does not support probability prediction; skipping ROC curve.")

if __name__ == "__main__":
    # Load model, scaler, and feature indices
    with open('top20_sam_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('top20_sam_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('top20_sam_feature_indices.pkl', 'rb') as f:
        feature_indices = pickle.load(f)
    # Evaluate on test and holdout sets
    evaluate_on_directory('../embedded_test', model, scaler, feature_indices, 'Test Set')
    evaluate_on_directory('../embedded_holdout_spout', model, scaler, feature_indices, 'Holdout Set') 