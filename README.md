# DNMT in Trypanosomes: Machine Learning Analysis

## Overview
This repository contains machine learning analyses for SAM-dependent DNA methyltransferase (DNMT) proteins in _Trypanosoma brucei_ using Evolutionary Scale Modelling 2 (ESM2) embeddings, combining both unsupervised and supervised learning approaches to understand protein classification and feature analysis.

## Project Structure

### Unsupervised Machine Learning
ESM2 embeddings were generated (https://github.com/facebookresearch/esm)

The first phase focuses on unsupervised learning techniques for protein clustering and visualization:

- **Combined Genome Analysis**: t-SNE visualization scripts for combined trypanosome and methyltransferase datasets
- **DBSCAN Clustering**: Parameter optimization and clustering analysis for protein classification
- **DNMT-Specific Analysis**: Specialized analysis for DNA methyltransferase proteins
- **Interactive Visualizations**: HTML-based interactive plots for exploration

### Supervised Machine Learning
The second phase implements supervised learning for protein classification:

- **Feature Analysis**: Comprehensive analysis of protein features and their importance
- **Naive Bayes Classification**: Implementation of Naive Bayes models for protein classification
- **Top Feature Analysis**: Focus on the most important features (e.g., feature 1071)
- **Model Evaluation**: Holdout set validation and performance metrics
- **Correlation Analysis**: Feature correlation matrices and visualization

## Key Scripts

### Unsupervised Learning
- `run_combined_tsne.py` - Combined t-SNE analysis for trypanosome and methyltransferase proteins
- `run_combined_tsne_interactive.py` - Interactive t-SNE visualization
- `analyze_tsne_results.py` - Analysis of t-SNE clustering results
- `generate_clustering_summary_matrix.py` - Clustering summary generation
- `tsne_dnmt_analysis_fixed.py` - DNMT-specific t-SNE analysis

### Supervised Learning
- `train_naive_bayes_feature_comparison.py` - Naive Bayes model training
- `analyze_feature_1071.py` - Analysis of the most important feature
- `evaluate_top20_model.py` - Model evaluation scripts
- `predict_holdout_set.py` - Holdout set prediction
- `create_correlation_matrix.py` - Feature correlation analysis

## Data Sources
- **Trypanosome Genomes**: Multiple trypanosome species protein sequences
- **Methyltransferase Proteins**: Reference methyltransferase datasets
- **Combined Datasets**: Merged datasets for comprehensive analysis

## Requirements
- Python 3.x
- Machine learning libraries (scikit-learn, numpy, pandas)
- Visualization libraries (matplotlib, seaborn, plotly)
- Bioinformatics tools for protein analysis

## Usage
1. Clone the repository
2. Install required dependencies
3. Run unsupervised analysis scripts for initial exploration
4. Execute supervised learning scripts for classification
5. Analyze results using the provided visualization tools

## License
Public repository for research purposes
