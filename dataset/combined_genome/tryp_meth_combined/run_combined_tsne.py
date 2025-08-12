import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import re
import matplotlib.cm as cm

# Directories containing embeddings (relative to current directory)
REF_DIR = 'dataset/reference_mt/embedded_methyltransferase'
TRYP_DIR = 'dataset/tryp_genome/tryp_embedding'
OUTPUT_DIR = '.'
SAM_DNMT_FILE = 'dataset/reference_mt/uniprot_ids_clustered_sam_dnmt.txt'

# Helper to recursively load all .pt files from a directory and its subdirectories
def load_embeddings_from_directory(directory):
    embeddings = {}
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.pt'):
                filepath = os.path.join(root, filename)
                try:
                    embedding = torch.load(filepath, map_location='cpu')
                    # Handle dict or tensor
                    if isinstance(embedding, dict):
                        if 'mean_representations' in embedding:
                            embedding = embedding['mean_representations'][33].numpy()
                        elif 'embeddings' in embedding:
                            embedding = embedding['embeddings'].numpy()
                        elif 'features' in embedding:
                            embedding = embedding['features'].numpy()
                        else:
                            for v in embedding.values():
                                if isinstance(v, torch.Tensor):
                                    embedding = v.numpy()
                                    break
                            else:
                                continue
                    elif isinstance(embedding, torch.Tensor):
                        embedding = embedding.numpy()
                    else:
                        continue
                    if embedding.ndim > 1:
                        embedding = np.mean(embedding, axis=0)
                    # Use relative path from directory as key for uniqueness
                    rel_path = os.path.relpath(filepath, directory)
                    embeddings[rel_path] = embedding
                except Exception as e:
                    print(f'Error loading {filepath}: {e}')
    return embeddings

def extract_uniprot_id(filename):
    # Try to extract UniProt ID from filename, e.g. tr|A0A0F8V9K5|...
    # Accepts both 'tr|ID|' and 'sp|ID|' and also just ID_
    match = re.search(r'(?:tr\|([A-Z0-9]{6,10})\||sp\|([A-Z0-9]{6,10})\||([A-Z0-9]{6,10})[_\.])', filename)
    if match:
        for group in match.groups():
            if group:
                return group
    return None

# Load both sets
ref_embeddings = load_embeddings_from_directory(REF_DIR)
tryp_embeddings = load_embeddings_from_directory(TRYP_DIR)

# Combine
all_embeddings = {**ref_embeddings, **tryp_embeddings}
all_ids = list(all_embeddings.keys())
embedding_matrix = np.array([all_embeddings[k] for k in all_ids])

print(f"Loaded {len(all_ids)} embeddings. Shape: {embedding_matrix.shape}")

# Load SAM DNMT UniProt IDs
with open(SAM_DNMT_FILE) as f:
    sam_dnmt_ids = set(line.strip() for line in f if line.strip())

# Load cluster assignments (by UniProt ID)
CLUSTER_FILE = 'dataset/reference_mt/tsne50_dbscan_clusters.csv'
cluster_map = {}
with open(CLUSTER_FILE) as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split(',')
        if len(parts) < 3:
            continue
        seq_id = parts[0]
        group = parts[2]
        # Extract UniProt ID between first pair of |
        match = re.search(r'\|([A-Z0-9]{6,10})\|', seq_id)
        if match:
            uniprot_id = match.group(1)
            cluster_map[uniprot_id] = group

# Standardize
scaler = StandardScaler()
embedding_matrix_scaled = scaler.fit_transform(embedding_matrix)

# Run t-SNE
print("Running t-SNE with perplexity=50...")
tsne = TSNE(n_components=2, random_state=42, perplexity=50, max_iter=1000)
tsne_result = tsne.fit_transform(embedding_matrix_scaled)

# Prepare labels for coloring
labels = []
clusters = []
for id_ in all_ids:
    if id_ in ref_embeddings:
        uniprot_id = extract_uniprot_id(id_)
        if uniprot_id == 'Q38BE3':  # Special highlight for Q38BE3
            labels.append('Q38BE3 Highlight')
            clusters.append('Q38BE3 Highlight')
        elif uniprot_id and uniprot_id in sam_dnmt_ids:
            labels.append('SAM DNMT')
            clusters.append('SAM DNMT')
        elif uniprot_id and uniprot_id in cluster_map:
            labels.append(f'Cluster {cluster_map[uniprot_id]}')
            clusters.append(cluster_map[uniprot_id])
        else:
            labels.append('Unclustered')
            clusters.append('Unclustered')
    else:
        labels.append('Other')
        clusters.append('Other')

# Save CSV (add cluster column)
df = pd.DataFrame({
    'id': all_ids,
    'tsne_x': tsne_result[:,0],
    'tsne_y': tsne_result[:,1],
    'label': labels,
    'cluster': clusters
})
df.to_csv(os.path.join(OUTPUT_DIR, 'q38be3_highlighted_tsne_results.csv'), index=False)

# Plot with cluster colors
# Get unique clusters (excluding special labels)
unique_clusters = sorted(set(c for c in clusters if c not in ['SAM DNMT', 'Other', 'Unclustered', 'Q38BE3 Highlight']), key=lambda x: int(x))
cmap = cm.get_cmap('tab20', len(unique_clusters))
plt.figure(figsize=(12,10))
# Plot clusters first (background)
for idx, cluster in enumerate(unique_clusters):
    mask = (df['cluster'] == cluster)
    # Make clusters fully transparent except for Trypanosome and SAM DNMT
    plt.scatter(df.loc[mask, 'tsne_x'], df.loc[mask, 'tsne_y'], label=f'Cluster {cluster}', alpha=0, s=20, color='#1f77b4')  # fully transparent
# Plot other special cases
plt.scatter(df.loc[df['label'] == 'SAM DNMT', 'tsne_x'], df.loc[df['label'] == 'SAM DNMT', 'tsne_y'], label='SAM DNMT', color='#d62728', s=40, alpha=0.7, marker='o')
plt.scatter(df.loc[df['label'] == 'Other', 'tsne_x'], df.loc[df['label'] == 'Other', 'tsne_y'], label='Other', color='#cccccc', s=20, alpha=0.3, marker='o')
plt.scatter(df.loc[df['label'] == 'Unclustered', 'tsne_x'], df.loc[df['label'] == 'Unclustered', 'tsne_y'], label='Unclustered', color='#1f77b4', s=30, alpha=0, marker='^')  # fully transparent
# Plot Q38BE3 last so it appears on top
plt.scatter(df.loc[df['label'] == 'Q38BE3 Highlight', 'tsne_x'], df.loc[df['label'] == 'Q38BE3 Highlight', 'tsne_y'], label='Q38BE3 Highlight', color='#ff7f0e', s=100, alpha=1.0, marker='*', edgecolors='black', linewidth=2, zorder=10)
plt.title('t-SNE of Combined Embeddings (perplexity=50)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Type/Cluster')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'q38be3_highlighted_tsne_plot.png'), dpi=300)
plt.show()

print('Done! Results saved in', OUTPUT_DIR) 