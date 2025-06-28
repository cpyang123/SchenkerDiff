import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import networkx as nx

SCALE_DEGREE_MAP = [
    'A2', 'm2', 'P8', 'A6', 'A1', 'A5',
    'm7', 'M2', 'm6', 'M7',
    'm3', 'M3', 'P5', 'd7',
    'P4', 'M6', 'A4', 'd5'
]

SCALE_DEGREE_Y = {
    'A1': 0, 'm2': 1, 'M2': 2, 'm3': 3, 'M3': 4, 'P4': 5,
    'A4': 6, 'd5': 7, 'P5': 8, 'A5': 9, 'm6': 10, 'M6': 11,
    'A6': 12, 'm7': 13, 'd7': 13.5, 'M7': 14, 'P8': 15, 'A2': -1
}

def parse_txt_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    N, X, E, R = 0, [], [], []
    mode = None
    for line in lines:
        if line.startswith("N="): N = int(line.split('=')[1])
        elif line.startswith("X:"): mode = 'X'; continue
        elif line.startswith("E:"): mode = 'E'; continue
        elif line.startswith("R:"): mode = 'R'; continue
        elif line.startswith("../../../"): break
        if mode == 'X':
            X += list(map(int, line.split()))
        elif mode == 'E':
            E.append(list(map(int, line.split())))
        elif mode == 'R':
            R.append(list(map(float, line.split())))
    return N, X, np.array(E), np.array(R)

# Load Data
filepath = "generated_samples1.txt"
N, X, E, R = parse_txt_file(filepath)
offsets = R[:, 7]
voices = R[:, 8]

# 4 Voices
# voice_indices = np.minimum((voices * 4).astype(int), 3)
# VOICE_LAYER_BASE = {0: 0, 1: 30, 2: 60, 3: 90}

voice_indices = np.round(voices).astype(int)
VOICE_LAYER_BASE = {0: 0, 1: 40}

pitch_relative_y = [SCALE_DEGREE_Y.get(SCALE_DEGREE_MAP[x], 0) for x in X]
pitch_y = [
    VOICE_LAYER_BASE[voice_indices[i]] + pitch_relative_y[i]
    for i in range(N)
]
pos = {i: (offsets[i], pitch_y[i]) for i in range(N)}

# Color Mapping for 4 voices
# norm = mcolors.Normalize(vmin=0, vmax=1)
# cmap = plt.colormaps['viridis']
# voice_colors = [cmap(norm(v)) for v in voices]

# Graph Structure
G = nx.DiGraph()
for i in range(N):
    pitch = SCALE_DEGREE_MAP[X[i]]
    G.add_node(i, pitch=pitch, offset=offsets[i], voice=voices[i])


EDGE_COLORS = {
    "onset": "blue",
    "voice": "purple",
    "forward": "black",
    "melisma": "green"
}

for i in range(N):
    for j in range(N):
        if E[i, j] == 1:
            G.add_edge(i, j, color=EDGE_COLORS["onset"])
        elif E[i, j] == 2:
            G.add_edge(i, j, color=EDGE_COLORS["forward"])
        elif E[i, j] == 3:
            G.add_edge(i, j, color=EDGE_COLORS["voice"])
        elif E[i, j] == 4:
            G.add_edge(i, j, color=EDGE_COLORS["melisma"])


fig, ax = plt.subplots(figsize=(14, 8))

# nx.draw_networkx_nodes(G, pos, node_color=voice_colors, node_size=500, edgecolors='black', ax=ax)
nx.draw_networkx_nodes(G, pos, node_size=500, edgecolors='black', ax=ax)
labels = {i: G.nodes[i]['pitch'] for i in G.nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=10, ax=ax)

# Edges
for u, v, d in G.edges(data=True):
    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], edge_color=d['color'], width=2, ax=ax)

# Offset lines
for x in offsets:
    ax.axvline(x=x, color='red', linestyle='--', linewidth=0.5)

# colorbar
# sm = cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# fig.colorbar(sm, ax=ax, label='Normalized Voice (0=Low, 1=High)')

ax.set_title("Note Graph: Per-Voice Layered Pitch with Offset and Edge Types")
ax.axis('off')
legend_elements = [
    Line2D([0], [0], color=EDGE_COLORS["onset"], lw=2, label='Onset'),
    Line2D([0], [0], color=EDGE_COLORS["forward"], lw=2, label='Forward'),
    Line2D([0], [0], color=EDGE_COLORS["voice"], lw=2, label='Voice'),
    Line2D([0], [0], color=EDGE_COLORS["melisma"], lw=2, label='Melisma'),
]

ax.legend(handles=legend_elements, loc='upper right')
VOICE_LAYER_BASE = {0: 0, 1: 40}
VOICE_LABEL_Y = {v: y + 7 for v, y in VOICE_LAYER_BASE.items()}

for v, y in VOICE_LABEL_Y.items():
    ax.text(
        x=min(offsets) - 0.03,
        y=y,
        s=f"Voice {v}",
        fontsize=12,
        va='center',
        ha='right',
        bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3')
    )

plt.tight_layout()
plt.show()