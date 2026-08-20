import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, ArrowStyle
import matplotlib.patches as patches

out_dir = r'C:\Users\Utkarsh Upadhyay\Desktop\imagesofrunsure'

def draw_box(ax, x, y, width, height, text, facecolor='#E8F0F8', edgecolor='#2B5B84'):
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.1,rounding_size=0.1",
                         ec=edgecolor, fc=facecolor, lw=1.5)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', 
            fontsize=10, fontweight='bold', color='#112233', family='sans-serif')
    return (x + width/2, y, x + width/2, y + height) # (center_x, bottom_y, center_x, top_y)

def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color='#2B5B84', lw=1.5, shrinkA=0, shrinkB=0))

# --- 01. System Architecture ---
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Blocks
b1 = draw_box(ax, 3, 10.5, 4, 1, 'Input\n(Raw GitHub Actions YAML)')
b2 = draw_box(ax, 3, 8.5, 4, 1, 'Data Preprocessing\n(YAML Parse & Canonical Normalization)')
b3 = draw_box(ax, 1, 6.5, 3.5, 1, 'Feature Extraction\n(80 Structural Features)')
b4 = draw_box(ax, 5.5, 6.5, 3.5, 1, 'Text Representation\n(TF-IDF N-grams)')
b5 = draw_box(ax, 3, 4.5, 4, 1, 'Fusion\n(Scaled Structure + Sparse Text)')
b6 = draw_box(ax, 3, 2.5, 4, 1, 'Model\n(Logistic Regression)')
b7 = draw_box(ax, 3, 0.5, 4, 1, 'Output\n(Risk Class & Probabilities)')

# Arrows
draw_arrow(ax, b1[0], b1[1], b2[0], b2[3])
draw_arrow(ax, b2[0], b2[1], b3[0], b3[3])
draw_arrow(ax, b2[0], b2[1], b4[0], b4[3])
draw_arrow(ax, b3[0], b3[1], b5[0], b5[3])
draw_arrow(ax, b4[0], b4[1], b5[0], b5[3])
draw_arrow(ax, b5[0], b5[1], b6[0], b6[3])
draw_arrow(ax, b6[0], b6[1], b7[0], b7[3])

fig.tight_layout()
fig.savefig(os.path.join(out_dir, '01_system_architecture.png'), dpi=300)
fig.savefig(os.path.join(out_dir, '01_system_architecture.svg'))
plt.close(fig)

# --- 02. Data Pipeline ---
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

b1 = draw_box(ax, 3, 12.5, 4, 1, 'Raw BSON Dumps\n(Configs & Executions)')
b2 = draw_box(ax, 3, 10.5, 4, 1, 'Execution Filtering\n(Min 10 runs, Valid Conclusions)')
b3 = draw_box(ax, 3, 8.5, 4, 1, 'Label Generation\n(Empirical Failure Rate Tertiles)')
b4 = draw_box(ax, 3, 6.5, 4, 1, 'Repository Capping\n(Max 20 Workflows per Repo)')
b5 = draw_box(ax, 3, 4.5, 4, 1, 'Train / Val / Test Split\n(Zero Repo Overlap)')
b6 = draw_box(ax, 3, 2.5, 4, 1, 'Feature Preparation\n(Fit TF-IDF/Scaler on Train Only)')
b7 = draw_box(ax, 3, 0.5, 4, 1, 'Model Ready Data')

draw_arrow(ax, b1[0], b1[1], b2[0], b2[3])
draw_arrow(ax, b2[0], b2[1], b3[0], b3[3])
draw_arrow(ax, b3[0], b3[1], b4[0], b4[3])
draw_arrow(ax, b4[0], b4[1], b5[0], b5[3])
draw_arrow(ax, b5[0], b5[1], b6[0], b6[3])
draw_arrow(ax, b6[0], b6[1], b7[0], b7[3])

fig.tight_layout()
fig.savefig(os.path.join(out_dir, '02_data_pipeline.png'), dpi=300)
fig.savefig(os.path.join(out_dir, '02_data_pipeline.svg'))
plt.close(fig)

# --- 03. Model Architecture ---
fig, ax = plt.subplots(figsize=(8, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

b1 = draw_box(ax, 1, 10.5, 3.5, 1, 'Structural Input\n(80 Features)')
b2 = draw_box(ax, 5.5, 10.5, 3.5, 1, 'Text Input\n(Normalized YAML String)')
b3 = draw_box(ax, 1, 8.5, 3.5, 1, 'StandardScaler')
b4 = draw_box(ax, 5.5, 8.5, 3.5, 1, 'TfidfVectorizer\n(Max 250k Vocab)')
b5 = draw_box(ax, 3, 6.5, 4, 1, 'SciPy Horizontal Stack\n(Sparse + Dense Matrix)')
b6 = draw_box(ax, 3, 4.5, 4, 1, 'Logistic Regression\n(L2 Penalty, C=1.0)')
b7 = draw_box(ax, 3, 2.5, 4, 1, 'Softmax Probabilities')
b8 = draw_box(ax, 3, 0.5, 4, 1, 'Output Class\n(LOW, MEDIUM, HIGH)')

draw_arrow(ax, b1[0], b1[1], b3[0], b3[3])
draw_arrow(ax, b2[0], b2[1], b4[0], b4[3])
draw_arrow(ax, b3[0], b3[1], b5[0], b5[3])
draw_arrow(ax, b4[0], b4[1], b5[0], b5[3])
draw_arrow(ax, b5[0], b5[1], b6[0], b6[3])
draw_arrow(ax, b6[0], b6[1], b7[0], b7[3])
draw_arrow(ax, b7[0], b7[1], b8[0], b8[3])

fig.tight_layout()
fig.savefig(os.path.join(out_dir, '03_model_architecture.png'), dpi=300)
fig.savefig(os.path.join(out_dir, '03_model_architecture.svg'))
plt.close(fig)

# --- 08. Prediction Workflow ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(0, 12)
ax.set_ylim(0, 10)
ax.axis('off')

b1 = draw_box(ax, 4, 8.5, 4, 1, 'Client (React UI)\nReads YAML file')
b2 = draw_box(ax, 4, 6.5, 4, 1, 'FastAPI Gateway\n(1MB Limit, CORS Validation)')
b3 = draw_box(ax, 4, 4.5, 4, 1, 'WorkflowPreprocessor\n(Extract Structure & Text)')
b4 = draw_box(ax, 4, 2.5, 4, 1, 'E06InferenceService\n(Frozen Artifacts)')
b5 = draw_box(ax, 4, 0.5, 4, 1, 'JSON Response\n(Probabilities & Metadata)')

draw_arrow(ax, b1[0], b1[1], b2[0], b2[3])
draw_arrow(ax, b2[0], b2[1], b3[0], b3[3])
draw_arrow(ax, b3[0], b3[1], b4[0], b4[3])
draw_arrow(ax, b4[0], b4[1], b5[0], b5[3])

# Add side note boxes
sn1 = draw_box(ax, 8.5, 4.5, 3, 1, 'Strips unused fields', facecolor='#F0F0F0', edgecolor='gray')
draw_arrow(ax, 8.5, 5, 8, 5) # Pointing left
ax.annotate('', xy=(8, 5), xytext=(8.5, 5), arrowprops=dict(arrowstyle="-", color='gray', linestyle='--'))

fig.tight_layout()
fig.savefig(os.path.join(out_dir, '08_prediction_workflow.png'), dpi=300)
fig.savefig(os.path.join(out_dir, '08_prediction_workflow.svg'))
plt.close(fig)

print("Diagrams created.")
