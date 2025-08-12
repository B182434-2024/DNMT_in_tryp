import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from Bio import SeqIO
from Bio.SeqUtils.ProtParam import ProteinAnalysis
import warnings
warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

def parse_fasta_sequences(fasta_path):
    """Parse a FASTA file and return a dict of {header: sequence}"""
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        header = record.description
        sequences[header] = str(record.seq)
    return sequences

def calculate_net_charge(sequence):
    """Calculate net charge of a protein sequence using ProteinAnalysis"""
    try:
        protein = ProteinAnalysis(sequence)
        # Get charge at pH 7.0 (physiological pH)
        charge = protein.charge_at_pH(7.0)
        return charge
    except Exception as e:
        print(f"Error calculating charge for sequence: {e}")
        return 0.0

def main():
    # Load the predictions file
    print("Loading data...")
    predictions_df = pd.read_csv('supervised_2/top_1_analysis/feature_1071_sorted_proteins.csv')
    
    # The "class" column contains the TRUE labels
    # We need to apply the feature 1071 model threshold to get PREDICTIONS
    true_labels = []
    feature_1071_values = []
    file_ids = []
    
    for _, row in predictions_df.iterrows():
        file_id = row['file_id']
        feature_1071_val = row['feature_1071_value']
        true_class = row['class']  # This is the TRUE label
        
        # Convert true class to numeric
        true_label = 1 if true_class == 'SAM' else 0
        
        true_labels.append(true_label)
        feature_1071_values.append(feature_1071_val)
        file_ids.append(file_id)
    
    true_labels = np.array(true_labels)
    feature_1071_values = np.array(feature_1071_values)
    
    # Apply feature 1071 model threshold to get predictions
    threshold = 0.0719
    predicted_labels = (feature_1071_values > threshold).astype(int)
    
    # Parse FASTA for sequences
    sam_fasta = 'supervised_2/clustered_sam_methyltrasnferase.fasta'
    nonsam_fasta = 'supervised_2/non_sam.fasta'
    sam_sequences = parse_fasta_sequences(sam_fasta)
    nonsam_sequences = parse_fasta_sequences(nonsam_fasta)
    all_sequences = {**sam_sequences, **nonsam_sequences}
    
    # Calculate net charges
    print("Calculating net charges...")
    net_charges = []
    for file_id in file_ids:
        # Extract the protein identifier from file_id
        if file_id.startswith('cath|'):
            # For cath files, extract the protein ID
            parts = file_id.split('|')
            if len(parts) >= 3:
                protein_id = parts[2].split('/')[0]  # e.g., "1wmaA00" from "1wmaA00/2-276.pt"
        else:
            # For tr| files, extract the protein ID
            parts = file_id.split('|')
            if len(parts) >= 2:
                protein_id = parts[1]  # e.g., "A0A090LXE7" from "tr|A0A090LXE7|..."
        
        # Find matching sequence
        sequence = None
        for fasta_header, seq in all_sequences.items():
            if protein_id in fasta_header:
                sequence = seq
                break
        
        if sequence:
            charge = calculate_net_charge(sequence)
        else:
            charge = 0.0
        
        net_charges.append(charge)
    
    net_charges = np.array(net_charges)
    
    # Calculate accuracy and statistics
    accuracy = np.mean(true_labels == predicted_labels)
    print(f"\n=== ANALYSIS RESULTS ===")
    print(f"Total proteins: {len(true_labels)}")
    print(f"True SAM: {np.sum(true_labels == 1)}")
    print(f"True Non-SAM: {np.sum(true_labels == 0)}")
    print(f"Predicted SAM: {np.sum(predicted_labels == 1)}")
    print(f"Predicted Non-SAM: {np.sum(predicted_labels == 0)}")
    print(f"Feature 1071 threshold: {threshold}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Calculate confusion matrix
    true_sam_pred_sam = np.sum((true_labels == 1) & (predicted_labels == 1))
    true_sam_pred_nonsam = np.sum((true_labels == 1) & (predicted_labels == 0))
    true_nonsam_pred_sam = np.sum((true_labels == 0) & (predicted_labels == 1))
    true_nonsam_pred_nonsam = np.sum((true_labels == 0) & (predicted_labels == 0))
    
    print(f"\n=== CONFUSION MATRIX ===")
    print(f"True SAM, Predicted SAM: {true_sam_pred_sam}")
    print(f"True SAM, Predicted Non-SAM: {true_sam_pred_nonsam}")
    print(f"True Non-SAM, Predicted SAM: {true_nonsam_pred_sam}")
    print(f"True Non-SAM, Predicted Non-SAM: {true_nonsam_pred_nonsam}")
    
    # Create comparison plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: True labels
    sam_mask_true = true_labels == 1
    nonsam_mask_true = true_labels == 0
    
    ax1.scatter(net_charges[nonsam_mask_true], feature_1071_values[nonsam_mask_true], 
                alpha=0.6, s=30, label='True Non-SAM', color='lightcoral')
    ax1.scatter(net_charges[sam_mask_true], feature_1071_values[sam_mask_true], 
                alpha=0.6, s=30, label='True SAM', color='steelblue')
    ax1.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
    ax1.set_xlabel('Net Charge (pH 7.0)')
    ax1.set_ylabel('Feature 1071 Value')
    ax1.set_title('True Labels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Predicted labels
    sam_mask_pred = predicted_labels == 1
    nonsam_mask_pred = predicted_labels == 0
    
    ax2.scatter(net_charges[nonsam_mask_pred], feature_1071_values[nonsam_mask_pred], 
                alpha=0.6, s=30, label='Predicted Non-SAM', color='lightcoral')
    ax2.scatter(net_charges[sam_mask_pred], feature_1071_values[sam_mask_pred], 
                alpha=0.6, s=30, label='Predicted SAM', color='steelblue')
    ax2.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
    ax2.set_xlabel('Net Charge (pH 7.0)')
    ax2.set_ylabel('Feature 1071 Value')
    ax2.set_title('Predicted Labels (Feature 1071 Model)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Correct vs Incorrect predictions
    correct_mask = true_labels == predicted_labels
    incorrect_mask = true_labels != predicted_labels
    
    ax3.scatter(net_charges[correct_mask], feature_1071_values[correct_mask], 
                alpha=0.6, s=30, label='Correct Prediction', color='green')
    ax3.scatter(net_charges[incorrect_mask], feature_1071_values[incorrect_mask], 
                alpha=0.6, s=30, label='Incorrect Prediction', color='red')
    ax3.axhline(y=threshold, color='red', linestyle='--', alpha=0.7, label=f'Threshold ({threshold})')
    ax3.set_xlabel('Net Charge (pH 7.0)')
    ax3.set_ylabel('Feature 1071 Value')
    ax3.set_title('Correct vs Incorrect Predictions')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Net charge distribution by predicted class
    pred_sam_charges = net_charges[sam_mask_pred]
    pred_nonsam_charges = net_charges[nonsam_mask_pred]
    
    ax4.hist(pred_nonsam_charges, bins=30, alpha=0.7, label='Predicted Non-SAM', color='lightcoral', density=True)
    ax4.hist(pred_sam_charges, bins=30, alpha=0.7, label='Predicted SAM', color='steelblue', density=True)
    ax4.set_xlabel('Net Charge (pH 7.0)')
    ax4.set_ylabel('Density')
    ax4.set_title('Net Charge Distribution by Predicted Class')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('supervised_2/plots/net_charge_vs_feature1071.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed statistics
    print(f"\n=== DETAILED STATISTICS ===")
    print(f"Precision: {true_sam_pred_sam / (true_sam_pred_sam + true_nonsam_pred_sam):.4f}")
    print(f"Recall: {true_sam_pred_sam / (true_sam_pred_sam + true_sam_pred_nonsam):.4f}")
    print(f"F1 Score: {2 * true_sam_pred_sam / (2 * true_sam_pred_sam + true_sam_pred_nonsam + true_nonsam_pred_sam):.4f}")
    
    # Show correlation between net charge and classification
    print(f"\n=== CORRELATION ANALYSIS ===")
    # Correlation between net charge and true labels
    corr_true = np.corrcoef(net_charges, true_labels)[0, 1]
    print(f"Correlation (net charge vs true labels): {corr_true:.4f}")
    
    # Correlation between net charge and predicted labels
    corr_pred = np.corrcoef(net_charges, predicted_labels)[0, 1]
    print(f"Correlation (net charge vs predicted labels): {corr_pred:.4f}")
    
    # Correlation between net charge and feature 1071
    corr_feature = np.corrcoef(net_charges, feature_1071_values)[0, 1]
    print(f"Correlation (net charge vs feature 1071): {corr_feature:.4f}")
    
    # Print charge statistics by predicted class
    print(f"\n=== NET CHARGE STATISTICS BY PREDICTED CLASS ===")
    print(f"Predicted SAM proteins:")
    print(f"  Mean charge: {np.mean(pred_sam_charges):.4f} ± {np.std(pred_sam_charges):.4f}")
    print(f"  Range: {np.min(pred_sam_charges):.4f} to {np.max(pred_sam_charges):.4f}")
    print(f"  Count: {len(pred_sam_charges)}")
    
    print(f"\nPredicted Non-SAM proteins:")
    print(f"  Mean charge: {np.mean(pred_nonsam_charges):.4f} ± {np.std(pred_nonsam_charges):.4f}")
    print(f"  Range: {np.min(pred_nonsam_charges):.4f} to {np.max(pred_nonsam_charges):.4f}")
    print(f"  Count: {len(pred_nonsam_charges)}")
    
    # Show some examples of misclassifications with charge info
    print(f"\n=== EXAMPLES OF MISCLASSIFICATIONS WITH CHARGE ===")
    incorrect_indices = np.where(incorrect_mask)[0]
    for i in incorrect_indices[:10]:  # Show first 10 misclassifications
        print(f"File: {file_ids[i]}")
        print(f"  True: {'SAM' if true_labels[i] == 1 else 'Non-SAM'}")
        print(f"  Predicted: {'SAM' if predicted_labels[i] == 1 else 'Non-SAM'}")
        print(f"  Feature 1071: {feature_1071_values[i]:.4f}")
        print(f"  Net Charge: {net_charges[i]:.4f}")
        print()

if __name__ == "__main__":
    main() 