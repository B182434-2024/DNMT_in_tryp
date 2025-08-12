#!/usr/bin/env python3
"""
Naive Bayes Classification - Feature Count Comparison for Supervised_2 Dataset
==============================================================================

This script trains Naive Bayes classifiers on different numbers of top features
for SAM methyltransferase classification and compares their performance.

Feature counts tested: 100, 50, 20, 10, 5, 1
Dataset: SAM methyltransferases vs non-SAM proteins
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

def get_top_features(X, y, n_features):
    """
    Get the top n most important features using Random Forest.
    
    Args:
        X: Feature matrix
        y: Labels
        n_features: Number of top features to select
    
    Returns:
        top_feature_indices: Indices of top features
        feature_importance: Feature importance scores
    """
    # Train a Random Forest to get feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    # Get feature importance
    feature_importance = rf.feature_importances_
    
    # Get top feature indices
    top_feature_indices = np.argsort(feature_importance)[-n_features:]
    
    return top_feature_indices, feature_importance

def train_naive_bayes_models(X_train, y_train, feature_indices, model_name="Model"):
    """
    Train Naive Bayes models on selected features.
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_indices: Indices of features to use
        model_name: Name for the model
    
    Returns:
        dict: Dictionary containing model performance metrics
    """
    # Select features
    X_train_reduced = X_train[:, feature_indices]
    
    # Define models
    models = {
        'GaussianNB': GaussianNB(),
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB()
    }
    
    # Hyperparameter grids
    param_grids = {
        'GaussianNB': {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
        'MultinomialNB': {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]},
        'BernoulliNB': {'alpha': [0.1, 0.5, 1.0, 2.0, 5.0], 'binarize': [None, 0.0, 0.5, 1.0]}
    }
    
    best_metrics = {}
    
    for model_name, model in models.items():
        # Choose appropriate scaler
        if model_name == 'MultinomialNB':
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train_reduced)
        else:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_reduced)
        
        # Hyperparameter tuning
        if model_name in param_grids:
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='f1', n_jobs=-1
            )
            grid_search.fit(X_train_scaled, y_train)
            best_model = grid_search.best_estimator_
        else:
            best_model = model.fit(X_train_scaled, y_train)
        
        # Cross-validation
        cv_f1_scores = cross_val_score(
            best_model, X_train_scaled, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='f1'
        )
        cv_accuracy_scores = cross_val_score(
            best_model, X_train_scaled, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        best_metrics[model_name] = {
            'cv_f1_mean': cv_f1_scores.mean(),
            'cv_f1_std': cv_f1_scores.std(),
            'cv_accuracy_mean': cv_accuracy_scores.mean(),
            'cv_accuracy_std': cv_accuracy_scores.std(),
            'cv_f1_scores': cv_f1_scores,
            'cv_accuracy_scores': cv_accuracy_scores,
            'best_model': best_model,
            'scaler': scaler
        }
    
    return best_metrics

def plot_feature_comparison(results, save_dir="feature_comparison_results"):
    """
    Create visualization plots for feature comparison results.
    
    Args:
        results: Dictionary containing results for all feature counts
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    feature_counts = list(results.keys())
    models = ['GaussianNB', 'MultinomialNB', 'BernoulliNB']
    
    # 1. F1 Score comparison
    plt.figure(figsize=(12, 8))
    
    for i, model in enumerate(models):
        f1_means = [results[fc][model]['cv_f1_mean'] for fc in feature_counts]
        f1_stds = [results[fc][model]['cv_f1_std'] for fc in feature_counts]
        
        plt.errorbar(feature_counts, f1_means, yerr=f1_stds, 
                    marker='o', linewidth=2, markersize=8, 
                    label=model, capsize=5)
    
    plt.xlabel('Number of Features')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Number of Features (SAM Classification)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.xticks(feature_counts, feature_counts)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'f1_score_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Accuracy comparison
    plt.figure(figsize=(12, 8))
    
    for i, model in enumerate(models):
        acc_means = [results[fc][model]['cv_accuracy_mean'] for fc in feature_counts]
        acc_stds = [results[fc][model]['cv_accuracy_std'] for fc in feature_counts]
        
        plt.errorbar(feature_counts, acc_means, yerr=acc_stds, 
                    marker='s', linewidth=2, markersize=8, 
                    label=model, capsize=5)
    
    plt.xlabel('Number of Features')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy Score vs Number of Features (SAM Classification)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.xticks(feature_counts, feature_counts)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'accuracy_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Best model performance summary
    plt.figure(figsize=(12, 8))
    
    best_f1_scores = []
    best_acc_scores = []
    best_models = []
    
    for fc in feature_counts:
        # Find best model for this feature count
        best_model = max(results[fc].keys(), 
                        key=lambda x: results[fc][x]['cv_f1_mean'])
        best_f1 = results[fc][best_model]['cv_f1_mean']
        best_acc = results[fc][best_model]['cv_accuracy_mean']
        
        best_f1_scores.append(best_f1)
        best_acc_scores.append(best_acc)
        best_models.append(best_model)
    
    # Plot best F1 scores
    plt.subplot(2, 1, 1)
    plt.plot(feature_counts, best_f1_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Features')
    plt.ylabel('Best F1 Score')
    plt.title('Best F1 Score vs Number of Features (SAM Classification)')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.xticks(feature_counts, feature_counts)
    
    # Add model labels
    for i, (fc, model) in enumerate(zip(feature_counts, best_models)):
        plt.annotate(model, (fc, best_f1_scores[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    # Plot best accuracy scores
    plt.subplot(2, 1, 2)
    plt.plot(feature_counts, best_acc_scores, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of Features')
    plt.ylabel('Best Accuracy Score')
    plt.title('Best Accuracy Score vs Number of Features (SAM Classification)')
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    plt.xticks(feature_counts, feature_counts)
    
    # Add model labels
    for i, (fc, model) in enumerate(zip(feature_counts, best_models)):
        plt.annotate(model, (fc, best_acc_scores[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'best_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function to run feature comparison for SAM classification."""
    print("=== Naive Bayes Classification - Feature Count Comparison (SAM Dataset) ===")
    
    # Safety check: Ensure we're only using training data
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
    
    X_train, y_train = X, y
    
    # Feature counts to test
    feature_counts = [100, 50, 20, 10, 5, 1]
    
    # Get feature importance once
    print(f"\nGetting feature importance using Random Forest...")
    _, feature_importance = get_top_features(X_train, y_train, max(feature_counts))
    
    # Store results
    all_results = {}
    
    # Test each feature count
    for n_features in feature_counts:
        print(f"\n{'='*60}")
        print(f"Testing with {n_features} features...")
        print(f"{'='*60}")
        
        # Get top features
        top_feature_indices = np.argsort(feature_importance)[-n_features:]
        
        print(f"Feature reduction: {((X_train.shape[1] - n_features) / X_train.shape[1] * 100):.1f}%")
        print(f"Feature importance range: {feature_importance[top_feature_indices].min():.6f} - {feature_importance[top_feature_indices].max():.6f}")
        
        # Train models
        results = train_naive_bayes_models(X_train, y_train, top_feature_indices)
        
        # Print results
        print(f"\nResults for {n_features} features:")
        print(f"{'Model':<15} {'F1 Score':<12} {'Accuracy':<12}")
        print(f"{'-'*45}")
        
        best_model = max(results.keys(), key=lambda x: results[x]['cv_f1_mean'])
        best_f1 = results[best_model]['cv_f1_mean']
        best_acc = results[best_model]['cv_accuracy_mean']
        
        for model_name, metrics in results.items():
            f1_mean = metrics['cv_f1_mean']
            acc_mean = metrics['cv_accuracy_mean']
            marker = " *" if model_name == best_model else ""
            print(f"{model_name:<15} {f1_mean:<12.4f} {acc_mean:<12.4f}{marker}")
        
        print(f"\nBest model: {best_model}")
        print(f"Best F1: {best_f1:.4f}, Best Accuracy: {best_acc:.4f}")
        
        all_results[n_features] = results
    
    # Create visualizations
    print(f"\n{'='*60}")
    print("Creating visualizations...")
    plot_feature_comparison(all_results)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("FEATURE COUNT COMPARISON SUMMARY (SAM Classification)")
    print(f"{'='*80}")
    print(f"{'Features':<10} {'Best Model':<15} {'F1 Score':<12} {'Accuracy':<12} {'Reduction':<12}")
    print(f"{'-'*80}")
    
    for n_features in feature_counts:
        results = all_results[n_features]
        best_model = max(results.keys(), key=lambda x: results[x]['cv_f1_mean'])
        best_f1 = results[best_model]['cv_f1_mean']
        best_acc = results[best_model]['cv_accuracy_mean']
        reduction = ((X_train.shape[1] - n_features) / X_train.shape[1] * 100)
        
        print(f"{n_features:<10} {best_model:<15} {best_f1:<12.4f} {best_acc:<12.4f} {reduction:<12.1f}%")
    
    print(f"{'='*80}")
    
    # Find overall best
    best_feature_count = max(all_results.keys(), 
                           key=lambda x: max(all_results[x][m]['cv_f1_mean'] 
                                           for m in all_results[x].keys()))
    best_model_name = max(all_results[best_feature_count].keys(), 
                         key=lambda x: all_results[best_feature_count][x]['cv_f1_mean'])
    
    print(f"\nOVERALL BEST:")
    print(f"Feature count: {best_feature_count}")
    print(f"Best model: {best_model_name}")
    print(f"F1 Score: {all_results[best_feature_count][best_model_name]['cv_f1_mean']:.4f}")
    print(f"Accuracy: {all_results[best_feature_count][best_model_name]['cv_accuracy_mean']:.4f}")
    
    # Save best model
    best_model = all_results[best_feature_count][best_model_name]['best_model']
    best_scaler = all_results[best_feature_count][best_model_name]['scaler']
    best_feature_indices = np.argsort(feature_importance)[-best_feature_count:]
    
    with open('naive_bayes_best_overall_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    
    with open('naive_bayes_best_overall_scaler.pkl', 'wb') as f:
        pickle.dump(best_scaler, f)
    
    with open('naive_bayes_best_overall_feature_indices.pkl', 'wb') as f:
        pickle.dump(best_feature_indices, f)
    
    print(f"\nBest overall model saved:")
    print(f"- Model: naive_bayes_best_overall_model.pkl")
    print(f"- Scaler: naive_bayes_best_overall_scaler.pkl")
    print(f"- Feature indices: naive_bayes_best_overall_feature_indices.pkl")
    
    print(f"\n✅ TRAINING COMPLETE - Test and holdout sets remain untouched:")
    print(f"  - embedded_test/ directory: UNTOUCHED")
    print(f"  - embedded_holdout_spout/ directory: UNTOUCHED")
    print(f"  - Only used training data from embedded_sam/ and embedded_non_sam/")

if __name__ == "__main__":
    main() 