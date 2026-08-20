import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

fig, ax = plt.subplots(figsize=(8, 10))
ax.axis('off')

def draw_box(ax, x, y, width, height, text, bg_color='#f0f0f0'):
    rect = patches.Rectangle((x, y), width, height, linewidth=1.5, edgecolor='black', facecolor=bg_color)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=10, fontweight='bold', wrap=True)

def draw_arrow(ax, x, y, dx, dy):
    ax.annotate('', xy=(x+dx, y+dy), xytext=(x, y), arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))

# Define blocks
blocks = [
    (0.3, 0.9, 0.4, 0.08, "RAW YAML"),
    (0.3, 0.75, 0.4, 0.08, "React frontend"),
    (0.3, 0.60, 0.4, 0.08, "FastAPI"),
    (0.2, 0.45, 0.6, 0.08, "WorkflowPreprocessor\n- frozen normalization\n- 80 structural features"),
    (0.2, 0.30, 0.6, 0.08, "E06 Model\n- TF-IDF text representation\n- StandardScaler structural representation"),
    (0.3, 0.15, 0.4, 0.08, "Logistic Regression"),
    (0.25, 0.00, 0.5, 0.08, "LOW / MEDIUM / HIGH\n+ probabilities")
]

for i in range(len(blocks)):
    x, y, w, h, t = blocks[i]
    draw_box(ax, x, y, w, h, t, bg_color='#e8f4f8')
    if i < len(blocks) - 1:
        # draw arrow down
        nx, ny, nw, nh, nt = blocks[i+1]
        draw_arrow(ax, x + w/2, y, 0, ny + nh - y)

os.makedirs('docs', exist_ok=True)
plt.savefig('docs/architecture.png', bbox_inches='tight', dpi=300)
print("Saved docs/architecture.png")
