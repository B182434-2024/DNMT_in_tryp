#!/usr/bin/env python3
"""
Feature 1071 Analysis - Single Feature Model and Distribution
============================================================

This script trains a model using only feature 1071 and creates a visualization
showing the distribution of proteins from SAM to non-SAM based on this feature.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_embeddings_with_ids(embedding_dir, label, is_nested=False):
    """Load embeddings with file IDs."""
    embeddings = []
    labels = []
    file_ids = []
    
    if is_nested:
        for subdir in os.listdir(embedding_dir):
            subdir_path = os.path.join(embedding_dir, subdir)
            if os.path.isdir(subdir_path):
                for file_name in os.listdir(subdir_path):
                    if file_name.endswith('.pt'):
                        file_path = os.path.join(subdir_path, file_name)
                        try:
                            embedding_dict = torch.load(file_path, map_location='cpu')
                            
                            if isinstance(embedding_dict, dict) and 'mean_representations' in embedding_dict:
                                embedding = embedding_dict['mean_representations'][33].numpy()
                            else:
                                embedding = embedding_dict.numpy() if isinstance(embedding_dict, torch.Tensor) else np.array(embedding_dict)
                            
                            if embedding.ndim > 1:
                                embedding = embedding.flatten()
                            
                            embeddings.append(embedding)
                            labels.append(label)
                            file_ids.append(f"{subdir}/{file_name}")
                            
                        except Exception as e:
                            print(f"Error loading {subdir}/{file_name}: {e}")
    else:
        for file_name in os.listdir(embedding_dir):
            if file_name.endswith('.pt'):
                file_path = os.path.join(embedding_dir, file_name)
                try:
                    embedding_dict = torch.load(file_path, map_location='cpu')
                    
                    if isinstance(embedding_dict, dict) and 'mean_representations' in embedding_dict:
                        embedding = embedding_dict['mean_representations'][33].numpy()
                    else:
                        embedding = embedding_dict.numpy() if isinstance(embedding_dict, torch.Tensor) else np.array(embedding_dict)
                    
                    if embedding.ndim > 1:
                        embedding = embedding.flatten()
                    
                    embeddings.append(embedding)
                    labels.append(label)
                    file_ids.append(file_name)
                    
                except Exception as e:
                    print(f"Error loading {file_name}: {e}")
    
    return np.array(embeddings), np.array(labels), file_ids

def train_feature_1071_model(X, y):
    """Train a model using only feature 1071."""
    
    print("🔬 TRAINING MODEL WITH FEATURE 1071 ONLY")
    print("=" * 50)
    
    # Extract feature 1071
    feature_1071 = X[:, 1071].reshape(-1, 1)
    
    # Scale the feature
    scaler = StandardScaler()
    feature_1071_scaled = scaler.fit_transform(feature_1071)
    
    # Train GaussianNB model
    model = GaussianNB()
    
    # Cross-validation
    cv_scores = cross_val_score(model, feature_1071_scaled, y, cv=5, scoring='accuracy')
    cv_f1_scores = cross_val_score(model, feature_1071_scaled, y, cv=5, scoring='f1')
    
    print(f"Cross-validation results:")
    print(f"  Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  F1 Score: {cv_f1_scores.mean():.4f} ± {cv_f1_scores.std():.4f}")
    
    # Train final model on all data
    model.fit(feature_1071_scaled, y)
    
    # Get predictions
    y_pred = model.predict(feature_1071_scaled)
    y_proba = model.predict_proba(feature_1071_scaled)
    
    # Print classification report
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Non-SAM', 'SAM']))
    
    # Create confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-SAM', 'SAM'], 
                yticklabels=['Non-SAM', 'SAM'])
    plt.title('Confusion Matrix - Feature 1071 Only')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('feature_1071_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model, scaler, feature_1071_scaled, y_pred, y_proba

def create_distribution_plot(feature_values, y, file_ids):
    """Create a distribution plot showing proteins from SAM to non-SAM."""
    
    print("\n📊 CREATING DISTRIBUTION PLOT")
    print("=" * 50)
    
    # Create dataframe for analysis
    df = pd.DataFrame({
        'file_id': file_ids,
        'feature_1071_value': feature_values.flatten(),
        'class': ['SAM' if l == 1 else 'Non-SAM' for l in y]
    })
    
    # Sort by feature value
    df_sorted = df.sort_values('feature_1071_value')
    
    # Create the main distribution plot
    plt.figure(figsize=(20, 12))
    
    # 1. Main distribution plot
    plt.subplot(2, 2, 1)
    
    # Create color mapping
    colors = ['red' if c == 'SAM' else 'blue' for c in df_sorted['class']]
    
    # Plot each protein as a point
    plt.scatter(range(len(df_sorted)), df_sorted['feature_1071_value'], 
               c=colors, alpha=0.6, s=20)
    
    # Add class separation line
    sam_indices = df_sorted[df_sorted['class'] == 'SAM'].index
    non_sam_indices = df_sorted[df_sorted['class'] == 'Non-SAM'].index
    
    if len(sam_indices) > 0 and len(non_sam_indices) > 0:
        # Find the boundary between classes
        sam_values = df_sorted.loc[sam_indices, 'feature_1071_value']
        non_sam_values = df_sorted.loc[non_sam_indices, 'feature_1071_value']
        
        # Calculate optimal threshold
        threshold = (sam_values.mean() + non_sam_values.mean()) / 2
        plt.axhline(y=threshold, color='green', linestyle='--', alpha=0.7, 
                   label=f'Threshold: {threshold:.4f}')
    
    plt.xlabel('Protein Index (sorted by Feature 1071 value)')
    plt.ylabel('Feature 1071 Value')
    plt.title('Distribution of Proteins by Feature 1071 Value\n(Sorted from lowest to highest)')
    plt.legend(['SAM', 'Non-SAM', 'Threshold'])
    plt.grid(True, alpha=0.3)
    
    # 2. Histogram overlay
    plt.subplot(2, 2, 2)
    
    sam_values = df[df['class'] == 'SAM']['feature_1071_value']
    non_sam_values = df[df['class'] == 'Non-SAM']['feature_1071_value']
    
    plt.hist(sam_values, alpha=0.7, label='SAM', bins=30, density=True, color='red')
    plt.hist(non_sam_values, alpha=0.7, label='Non-SAM', bins=30, density=True, color='blue')
    
    if len(sam_values) > 0 and len(non_sam_values) > 0:
        plt.axvline(x=threshold, color='green', linestyle='--', alpha=0.7, 
                   label=f'Threshold: {threshold:.4f}')
    
    plt.xlabel('Feature 1071 Value')
    plt.ylabel('Density')
    plt.title('Feature 1071 Distribution by Class')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Box plot
    plt.subplot(2, 2, 3)
    
    data = [sam_values, non_sam_values]
    plt.boxplot(data, labels=['SAM', 'Non-SAM'])
    plt.ylabel('Feature 1071 Value')
    plt.title('Feature 1071 Distribution - Box Plot')
    plt.grid(True, alpha=0.3)
    
    # 4. Violin plot
    plt.subplot(2, 2, 4)
    
    sns.violinplot(data=df, x='class', y='feature_1071_value')
    plt.title('Feature 1071 Distribution - Violin Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feature_1071_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"\n📈 FEATURE 1071 STATISTICS")
    print("-" * 30)
    
    if len(sam_values) > 0:
        print(f"SAM proteins:")
        print(f"  Count: {len(sam_values)}")
        print(f"  Mean: {sam_values.mean():.6f}")
        print(f"  Std: {sam_values.std():.6f}")
        print(f"  Min: {sam_values.min():.6f}")
        print(f"  Max: {sam_values.max():.6f}")
    
    if len(non_sam_values) > 0:
        print(f"\nNon-SAM proteins:")
        print(f"  Count: {len(non_sam_values)}")
        print(f"  Mean: {non_sam_values.mean():.6f}")
        print(f"  Std: {non_sam_values.std():.6f}")
        print(f"  Min: {non_sam_values.min():.6f}")
        print(f"  Max: {non_sam_values.max():.6f}")
    
    if len(sam_values) > 0 and len(non_sam_values) > 0:
        # Calculate effect size (Cohen's d)
        pooled_std = np.sqrt((sam_values.var() + non_sam_values.var()) / 2)
        cohens_d = (sam_values.mean() - non_sam_values.mean()) / pooled_std
        
        print(f"\nEffect size (Cohen's d): {cohens_d:.4f}")
        print(f"Separation threshold: {threshold:.6f}")
        
        # Calculate overlap
        sam_above_threshold = (sam_values > threshold).sum()
        non_sam_below_threshold = (non_sam_values < threshold).sum()
        
        sam_accuracy = sam_above_threshold / len(sam_values)
        non_sam_accuracy = non_sam_below_threshold / len(non_sam_values)
        
        print(f"\nSimple threshold-based accuracy:")
        print(f"  SAM accuracy: {sam_accuracy:.4f} ({sam_above_threshold}/{len(sam_values)})")
        print(f"  Non-SAM accuracy: {non_sam_accuracy:.4f} ({non_sam_below_threshold}/{len(non_sam_values)})")
        print(f"  Overall accuracy: {(sam_above_threshold + non_sam_below_threshold) / len(df):.4f}")
    
    return df_sorted, threshold

def analyze_extreme_proteins(df_sorted, threshold):
    """Analyze proteins with extreme feature 1071 values."""
    
    print(f"\n🎯 EXTREME PROTEIN ANALYSIS")
    print("=" * 50)
    
    # Find extreme SAM proteins (highest feature 1071 values)
    sam_proteins = df_sorted[df_sorted['class'] == 'SAM']
    extreme_sam_high = sam_proteins.nlargest(10, 'feature_1071_value')
    
    print(f"Top 10 SAM proteins with highest Feature 1071 values:")
    for i, (_, row) in enumerate(extreme_sam_high.iterrows(), 1):
        print(f"  {i:2d}. {row['file_id'][:50]:<50} | {row['feature_1071_value']:.6f}")
    
    # Find extreme SAM proteins (lowest feature 1071 values)
    extreme_sam_low = sam_proteins.nsmallest(10, 'feature_1071_value')
    
    print(f"\nTop 10 SAM proteins with lowest Feature 1071 values:")
    for i, (_, row) in enumerate(extreme_sam_low.iterrows(), 1):
        print(f"  {i:2d}. {row['file_id'][:50]:<50} | {row['feature_1071_value']:.6f}")
    
    # Find extreme non-SAM proteins (highest feature 1071 values)
    non_sam_proteins = df_sorted[df_sorted['class'] == 'Non-SAM']
    extreme_non_sam_high = non_sam_proteins.nlargest(10, 'feature_1071_value')
    
    print(f"\nTop 10 Non-SAM proteins with highest Feature 1071 values:")
    for i, (_, row) in enumerate(extreme_non_sam_high.iterrows(), 1):
        print(f"  {i:2d}. {row['file_id'][:50]:<50} | {row['feature_1071_value']:.6f}")
    
    # Find extreme non-SAM proteins (lowest feature 1071 values)
    extreme_non_sam_low = non_sam_proteins.nsmallest(10, 'feature_1071_value')
    
    print(f"\nTop 10 Non-SAM proteins with lowest Feature 1071 values:")
    for i, (_, row) in enumerate(extreme_non_sam_low.iterrows(), 1):
        print(f"  {i:2d}. {row['file_id'][:50]:<50} | {row['feature_1071_value']:.6f}")
    
    # Save extreme proteins to CSV
    extreme_df = pd.concat([
        extreme_sam_high.assign(type='SAM_high'),
        extreme_sam_low.assign(type='SAM_low'),
        extreme_non_sam_high.assign(type='NonSAM_high'),
        extreme_non_sam_low.assign(type='NonSAM_low')
    ])
    
    extreme_df.to_csv('feature_1071_extreme_proteins.csv', index=False)
    print(f"\nExtreme proteins saved to: feature_1071_extreme_proteins.csv")

def main():
    """Main analysis function."""
    
    print("=== FEATURE 1071 ANALYSIS ===")
    print("Training model and creating distribution plot")
    print("=" * 50)
    
    # Load data
    print("\nLoading training data...")
    X_sam, y_sam, sam_ids = load_embeddings_with_ids("embedded_sam", 1, is_nested=False)
    X_non_sam, y_non_sam, non_sam_ids = load_embeddings_with_ids("embedded_non_sam", 0, is_nested=True)
    
    X = np.vstack([X_sam, X_non_sam])
    y = np.hstack([y_sam, y_non_sam])
    all_ids = sam_ids + non_sam_ids
    
    print(f"Dataset loaded: {len(X)} samples")
    print(f"SAM proteins: {np.sum(y == 1)}")
    print(f"Non-SAM proteins: {np.sum(y == 0)}")
    
    # Train model with feature 1071 only
    model, scaler, feature_1071_scaled, y_pred, y_proba = train_feature_1071_model(X, y)
    
    # Create distribution plot
    df_sorted, threshold = create_distribution_plot(feature_1071_scaled, y, all_ids)
    
    # Analyze extreme proteins
    analyze_extreme_proteins(df_sorted, threshold)
    
    # Save model and results
    import pickle
    with open('feature_1071_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('feature_1071_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save sorted dataframe
    df_sorted.to_csv('feature_1071_sorted_proteins.csv', index=False)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"Files saved:")
    print(f"  - feature_1071_model.pkl (trained model)")
    print(f"  - feature_1071_scaler.pkl (feature scaler)")
    print(f"  - feature_1071_sorted_proteins.csv (sorted protein list)")
    print(f"  - feature_1071_extreme_proteins.csv (extreme proteins)")
    print(f"  - feature_1071_distribution_analysis.png (distribution plots)")
    print(f"  - feature_1071_confusion_matrix.png (confusion matrix)")

if __name__ == "__main__":
    main() 