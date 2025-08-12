import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load the CSV with t-SNE results and labels
CSV_FILE = 'combined_tsne_results_labeled.csv'
df = pd.read_csv(CSV_FILE)

# Define color and marker mapping
color_map = {
    'Other': '#cccccc',         # Trypanosome
    'Unclustered': '#6c7a89',    # Unclustered
    'SAM DNMT': '#d62728',       # SAM DNMT
    'Methyltransferase': '#1f77b4',  # Methyltransferase
}
marker_map = {
    'Other': 'circle',
    'Unclustered': 'triangle-up',
    'SAM DNMT': 'circle',
    'Methyltransferase': 'circle',
}

def break_long_name(name, max_len=40):
    if len(name) <= max_len:
        return name
    # Find the nearest space after max_len
    space_idx = name.find(' ', max_len)
    if space_idx == -1:
        return name  # no space found, don't break
    return name[:space_idx] + '<br>' + name[space_idx+1:]

# Prepare hover text with line break in long protein names
hover_text = df.apply(lambda row: f"Protein: {break_long_name(str(row['id']))}<br>Label: {row['label']}", axis=1)

# Build traces for each group
fig = go.Figure()

# Add 'SAM DNMT' points
sam_mask = (df['label'] == 'SAM DNMT')
fig.add_trace(go.Scattergl(
    x=df.loc[sam_mask, 'tsne_x'],
    y=df.loc[sam_mask, 'tsne_y'],
    mode='markers',
    name='SAM DNMT',
    marker=dict(
        color='#d62728',
        size=10,
        symbol='circle',
        opacity=0.8,
        line=dict(width=0)
    ),
    text=hover_text[sam_mask],
    hoverinfo='text',
))

# Add 'Methyltransferase' points (invisible)
methyl_mask = (df['label'] == 'Methyltransferase')
fig.add_trace(go.Scattergl(
    x=df.loc[methyl_mask, 'tsne_x'],
    y=df.loc[methyl_mask, 'tsne_y'],
    mode='markers',
    name='Methyltransferase',
    marker=dict(
        color='#1f77b4',
        size=7,
        symbol='circle',
        opacity=0,  # completely invisible
        line=dict(width=0)
    ),
    text=hover_text[methyl_mask],
    hoverinfo='text',
    showlegend=False,  # Hide from legend
))

# Add 'Other' points
other_mask = (df['label'] == 'Other')
fig.add_trace(go.Scattergl(
    x=df.loc[other_mask, 'tsne_x'],
    y=df.loc[other_mask, 'tsne_y'],
    mode='markers',
    name='Other',
    marker=dict(
        color='#888888',
        size=6,
        symbol=marker_map['Other'],
        opacity=0.6,
        line=dict(width=0)
    ),
    text=hover_text[other_mask],
    hoverinfo='text',
))

fig.update_layout(
    title='Interactive t-SNE of Combined Embeddings',
    xaxis_title='t-SNE 1',
    yaxis_title='t-SNE 2',
    showlegend=True,
    width=1000,
    height=800,
    template='plotly_white',
    hovermode='closest',
    hoverlabel=dict(
        align='right',
        font=dict(color='black'),
        namelength=-1
    ),
)

fig.write_html('combined_tsne_plot_interactive.html')
fig.show() 