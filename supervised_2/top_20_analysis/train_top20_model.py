#!/usr/bin/env python3
"""
Train SAM Classification Model with Top 20 Features
==================================================

This script trains a model using the top 20 most important features
for SAM methyltransferase classification on the training set.
"""

import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_embeddings(embedding_dir, label):
    """
    Load protein embeddings from a directory.
    
    Args:
        embedding_dir: Path to directory containing .pt files
        label: Label for the class (0 or 1)
    
    Returns:
        embeddings: numpy array of shape (n_samples, n_features)
        labels: numpy array of labels
    """
    embeddings = []
    labels = []
    
    # Get all .pt files recursively
    pt_files = glob.glob(os.path.join(embedding_dir, "**", "*.pt"), recursive=True)
    
    print(f"Loading {len(pt_files)} files from {embedding_dir}...")
    
    for pt_file in tqdm(pt_files):
        try:
            # Load the embedding data
            embedding_data = torch.load(pt_file, map_location='cpu')
            
            # Extract the mean representation from layer 33 (ESM-2)
            if isinstance(embedding_data, dict) and 'mean_representations' in embedding_data:
                embedding_tensor = embedding_data['mean_representations'][33]  # Layer 33
            else:
                # Fallback: try to use the data directly
                embedding_tensor = embedding_data
            
            # Convert to numpy array
            if isinstance(embedding_tensor, torch.Tensor):
                embedding_array = embedding_tensor.numpy()
            else:
                embedding_array = np.array(embedding_tensor)
            
            # Flatten if it's multi-dimensional
            if embedding_array.ndim > 1:
                embedding_array = embedding_array.flatten()
            
            embeddings.append(embedding_array)
            labels.append(label)
            
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
            continue
    
    return np.array(embeddings), np.array(labels)

def get_top_features(X, y, n_features=20):
    """
    Get the top n most important features using Random Forest.
    
    Args:
        X: Feature matrix
        y: Labels
        n_features: Number of top features to select (default: 20)
    
    Returns:
        top_feature_indices: Indices of top features
        feature_importance: Feature importance scores
    """
    print(f"Training Random Forest to identify top {n_features} features...")
    
    # Train a Random Forest to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importance
    feature_importance = rf.feature_importances_
    
    # Get top feature indices
    top_feature_indices = np.argsort(feature_importance)[-n_features:]
    
    print(f"Top {n_features} features identified:")
    for i, idx in enumerate(reversed(top_feature_indices)):
        importance = feature_importance[idx]
        print(f"  {i+1:2d}. Feature {idx:4d}: {importance:.6f}")
    
    return top_feature_indices, feature_importance

def train_optimal_model(X_train, y_train, feature_indices):
    """
    Train the optimal model (GaussianNB) on selected features.
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_indices: Indices of features to use
    
    Returns:
        dict: Dictionary containing model and performance metrics
    """
    print(f"\nTraining optimal model (GaussianNB) with {len(feature_indices)} features...")
    
    # Select features
    X_train_reduced = X_train[:, feature_indices]
    
    # Use StandardScaler for GaussianNB
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reduced)
    
    # Train GaussianNB with optimal hyperparameters
    model = GaussianNB(var_smoothing=1e-9)
    model.fit(X_train_scaled, y_train)
    
    # Cross-validation
    cv_f1_scores = cross_val_score(
        model, X_train_scaled, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1'
    )
    cv_accuracy_scores = cross_val_score(
        model, X_train_scaled, y_train, 
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy'
    )
    
    # Get predictions for confusion matrix
    y_pred = model.predict(X_train_scaled)
    
    results = {
        'cv_f1_mean': cv_f1_scores.mean(),
        'cv_f1_std': cv_f1_scores.std(),
        'cv_accuracy_mean': cv_accuracy_scores.mean(),
        'cv_accuracy_std': cv_accuracy_scores.std(),
        'cv_f1_scores': cv_f1_scores,
        'cv_accuracy_scores': cv_accuracy_scores,
        'model': model,
        'scaler': scaler,
        'feature_indices': feature_indices,
        'y_pred': y_pred
    }
    
    return results

def plot_training_results(results, y_train):
    """Create plots for training results."""
    
    # 1. Cross-validation scores
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # F1 scores
    axes[0].boxplot([results['cv_f1_scores']], labels=['F1 Score'])
    axes[0].set_title('Cross-Validation F1 Scores')
    axes[0].set_ylabel('F1 Score')
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy scores
    axes[1].boxplot([results['cv_accuracy_scores']], labels=['Accuracy'])
    axes[1].set_title('Cross-Validation Accuracy Scores')
    axes[1].set_ylabel('Accuracy Score')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('top20_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Confusion matrix
    cm = confusion_matrix(y_train, results['y_pred'])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-SAM', 'SAM'], 
                yticklabels=['Non-SAM', 'SAM'])
    plt.title('Confusion Matrix (Training Set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('top20_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to train the top 20 features model."""
    
    print("=== SAM Classification - Top 20 Features Training ===")
    
    # Safety check
    print("\n🔒 SAFETY CHECK: Using only training data")
    print("Training data sources:")
    print("  - embedded_sam/ (remaining files after test set creation)")
    print("  - embedded_non_sam/ (remaining files after test set creation)")
    print("Excluded data sources:")
    print("  - embedded_test/ (test set - NOT TOUCHED)")
    print("  - embedded_holdout_spout/ (holdout set - NOT TOUCHED)")
    
    # Load training data only
    print("\nLoading training embeddings...")
    X_sam, y_sam = load_embeddings("embedded_sam", 1)
    X_non_sam, y_non_sam = load_embeddings("embedded_non_sam", 0)
    
    # Combine data
    X = np.vstack([X_sam, X_non_sam])
    y = np.hstack([y_sam, y_non_sam])
    
    print(f"\nDataset Summary:")
    print(f"Total samples: {len(X)}")
    print(f"Original features per sample: {X.shape[1]}")
    print(f"SAM methyltransferase samples: {np.sum(y == 1)}")
    print(f"Non-SAM protein samples: {np.sum(y == 0)}")
    
    # Get top 20 features
    top_20_indices, feature_importance = get_top_features(X, y, n_features=20)
    
    # Train optimal model
    results = train_optimal_model(X, y, top_20_indices)
    
    # Print results
    print(f"\n{'='*60}")
    print("TRAINING RESULTS")
    print(f"{'='*60}")
    print(f"Model: GaussianNB")
    print(f"Features used: {len(top_20_indices)}")
    print(f"Feature reduction: {((X.shape[1] - len(top_20_indices)) / X.shape[1] * 100):.1f}%")
    print(f"\nCross-Validation Performance:")
    print(f"  F1 Score: {results['cv_f1_mean']:.4f} ± {results['cv_f1_std']:.4f}")
    print(f"  Accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}")
    
    # Print classification report
    print(f"\nClassification Report:")
    print(classification_report(y, results['y_pred'], 
                               target_names=['Non-SAM', 'SAM']))
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    plot_training_results(results, y)
    
    # Save the trained model
    print(f"\nSaving trained model...")
    
    with open('top20_sam_model.pkl', 'wb') as f:
        pickle.dump(results['model'], f)
    
    with open('top20_sam_scaler.pkl', 'wb') as f:
        pickle.dump(results['scaler'], f)
    
    with open('top20_sam_feature_indices.pkl', 'wb') as f:
        pickle.dump(results['feature_indices'], f)
    
    with open('top20_sam_feature_importance.pkl', 'wb') as f:
        pickle.dump(feature_importance, f)
    
    print(f"Model saved:")
    print(f"  - Model: top20_sam_model.pkl")
    print(f"  - Scaler: top20_sam_scaler.pkl")
    print(f"  - Feature indices: top20_sam_feature_indices.pkl")
    print(f"  - Feature importance: top20_sam_feature_importance.pkl")
    
    print(f"\n✅ TRAINING COMPLETE - Test and holdout sets remain untouched:")
    print(f"  - embedded_test/ directory: UNTOUCHED")
    print(f"  - embedded_holdout_spout/ directory: UNTOUCHED")
    print(f"  - Only used training data from embedded_sam/ and embedded_non_sam/")

if __name__ == "__main__":
    main() 