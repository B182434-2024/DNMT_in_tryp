#!/usr/bin/env python3
"""
Apply Top 20 Features Model to Holdout Set
==========================================

This script applies the trained SAM classification model to the holdout set
and generates predictions with confidence scores.
"""

import torch
import numpy as np
import os
import glob
from tqdm import tqdm
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_embeddings_with_ids(embedding_dir):
    """Load protein embeddings with their file IDs from holdout set."""
    embeddings = []
    file_ids = []
    
    pt_files = glob.glob(os.path.join(embedding_dir, "**", "*.pt"), recursive=True)
    
    print(f"Loading {len(pt_files)} files from {embedding_dir}...")
    
    for pt_file in tqdm(pt_files):
        try:
            # Extract file ID from path
            file_id = os.path.basename(pt_file).replace('.pt', '')
            
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
            file_ids.append(file_id)
            
        except Exception as e:
            print(f"Error loading {pt_file}: {e}")
            continue
    
    return np.array(embeddings), file_ids

def predict_holdout_set():
    """Apply the trained model to the holdout set."""
    
    print("=== SAM Classification - Holdout Set Prediction ===")
    
    # Load the trained model and components
    print("\nLoading trained model...")
    try:
        with open('top20_sam_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('top20_sam_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('top20_sam_feature_indices.pkl', 'rb') as f:
            feature_indices = pickle.load(f)
        
        with open('top20_sam_feature_importance.pkl', 'rb') as f:
            feature_importance = pickle.load(f)
            
        print("✓ Model loaded successfully")
    except FileNotFoundError as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load holdout set embeddings
    print("\nLoading holdout set embeddings...")
    X_holdout, holdout_ids = load_embeddings_with_ids("embedded_holdout_spout")
    
    print(f"Holdout set loaded: {len(X_holdout)} samples")
    
    # Extract the top 20 features
    X_holdout_reduced = X_holdout[:, feature_indices]
    
    # Scale the features
    X_holdout_scaled = scaler.transform(X_holdout_reduced)
    
    # Make predictions
    print("\nMaking predictions...")
    predictions = model.predict(X_holdout_scaled)
    prediction_probas = model.predict_proba(X_holdout_scaled)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'file_id': holdout_ids,
        'predicted_class': ['SAM' if p == 1 else 'Non-SAM' for p in predictions],
        'prediction_confidence': np.max(prediction_probas, axis=1),
        'sam_probability': prediction_probas[:, 1],
        'non_sam_probability': prediction_probas[:, 0]
    })
    
    # Add feature values for analysis
    for i, feat_idx in enumerate(feature_indices):
        results_df[f'feature_{feat_idx}'] = X_holdout_reduced[:, i]
    
    # Print prediction summary
    print(f"\n{'='*60}")
    print("PREDICTION SUMMARY")
    print(f"{'='*60}")
    
    sam_count = sum(predictions == 1)
    non_sam_count = sum(predictions == 0)
    
    print(f"Total predictions: {len(predictions)}")
    print(f"Predicted SAM methyltransferases: {sam_count}")
    print(f"Predicted Non-SAM proteins: {non_sam_count}")
    print(f"Confidence range: {results_df['prediction_confidence'].min():.3f} - {results_df['prediction_confidence'].max():.3f}")
    print(f"Average confidence: {results_df['prediction_confidence'].mean():.3f}")
    
    # Show high-confidence predictions
    print(f"\n{'='*60}")
    print("HIGH CONFIDENCE PREDICTIONS (≥0.95)")
    print(f"{'='*60}")
    
    high_conf = results_df[results_df['prediction_confidence'] >= 0.95]
    print(f"High confidence predictions: {len(high_conf)}")
    
    if len(high_conf) > 0:
        for _, row in high_conf.iterrows():
            print(f"  {row['file_id']:<50} | {row['predicted_class']:<8} | {row['prediction_confidence']:.3f}")
    
    # Show low-confidence predictions
    print(f"\n{'='*60}")
    print("LOW CONFIDENCE PREDICTIONS (<0.8)")
    print(f"{'='*60}")
    
    low_conf = results_df[results_df['prediction_confidence'] < 0.8]
    print(f"Low confidence predictions: {len(low_conf)}")
    
    if len(low_conf) > 0:
        for _, row in low_conf.iterrows():
            print(f"  {row['file_id']:<50} | {row['predicted_class']:<8} | {row['prediction_confidence']:.3f}")
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    
    # 1. Confidence distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(results_df['prediction_confidence'], bins=20, alpha=0.7, color='skyblue')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Count')
    plt.title('Distribution of Prediction Confidence')
    plt.grid(True, alpha=0.3)
    
    # 2. SAM probability distribution
    plt.subplot(2, 2, 2)
    plt.hist(results_df['sam_probability'], bins=20, alpha=0.7, color='lightcoral')
    plt.xlabel('SAM Probability')
    plt.ylabel('Count')
    plt.title('Distribution of SAM Probabilities')
    plt.grid(True, alpha=0.3)
    
    # 3. Prediction counts
    plt.subplot(2, 2, 3)
    prediction_counts = results_df['predicted_class'].value_counts()
    plt.pie(prediction_counts.values, labels=prediction_counts.index, autopct='%1.1f%%')
    plt.title('Prediction Distribution')
    
    # 4. Confidence vs SAM probability
    plt.subplot(2, 2, 4)
    plt.scatter(results_df['sam_probability'], results_df['prediction_confidence'], alpha=0.6)
    plt.xlabel('SAM Probability')
    plt.ylabel('Prediction Confidence')
    plt.title('Confidence vs SAM Probability')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('holdout_predictions_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save results
    print(f"\nSaving results...")
    
    # Save detailed results
    results_df.to_csv('holdout_predictions.csv', index=False)
    
    # Save summary statistics
    summary_stats = {
        'total_predictions': len(predictions),
        'predicted_sam': sam_count,
        'predicted_non_sam': non_sam_count,
        'avg_confidence': results_df['prediction_confidence'].mean(),
        'min_confidence': results_df['prediction_confidence'].min(),
        'max_confidence': results_df['prediction_confidence'].max(),
        'high_confidence_count': len(high_conf),
        'low_confidence_count': len(low_conf)
    }
    
    with open('holdout_prediction_summary.pkl', 'wb') as f:
        pickle.dump(summary_stats, f)
    
    print(f"Results saved:")
    print(f"  - holdout_predictions.csv (detailed predictions)")
    print(f"  - holdout_prediction_summary.pkl (summary statistics)")
    print(f"  - holdout_predictions_analysis.png (visualizations)")
    
    # Print top features analysis
    print(f"\n{'='*60}")
    print("TOP FEATURES ANALYSIS")
    print(f"{'='*60}")
    
    # Show feature importance for the top 5 features
    top_5_indices = np.argsort(feature_importance)[-5:]
    print("Top 5 most important features:")
    for i, idx in enumerate(reversed(top_5_indices)):
        importance = feature_importance[idx]
        print(f"  {i+1}. Feature {idx:4d}: {importance:.6f}")
    
    print(f"\n✅ Holdout set prediction complete!")

if __name__ == "__main__":
    predict_holdout_set() 