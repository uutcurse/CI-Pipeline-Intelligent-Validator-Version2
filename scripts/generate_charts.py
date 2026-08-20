import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

out_dir = r'C:\Users\Utkarsh Upadhyay\Desktop\imagesofrunsure'

# --- 04. Dataset Distribution ---
dist = pd.read_csv('tables/class_distribution.csv')
# Plot just the Test split for clarity, or stacked. Let's do a grouped bar chart.
splits = ['train', 'validation', 'test']
classes = [0, 1, 2]
labels = ['LOW (0)', 'MEDIUM (1)', 'HIGH (2)']

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(classes))
width = 0.25

colors = ['#4477AA', '#DDCC77', '#CC6677']

for i, split in enumerate(splits):
    subset = dist[dist['split'] == split].sort_values('final_label')
    counts = subset['count'].values
    ax.bar(x + (i - 1) * width, counts, width, label=split.capitalize(), color=colors[i])

ax.set_ylabel('Number of Samples')
ax.set_title('Dataset Class Distribution by Split')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.tight_layout()
fig.savefig(os.path.join(out_dir, '04_dataset_distribution.png'), dpi=300)
fig.savefig(os.path.join(out_dir, '04_dataset_distribution.svg'))
plt.close(fig)


# --- 06. Confusion Matrix ---
cm_df = pd.read_csv('tables/confusion_matrix.csv', index_col=0)
cm = cm_df.values
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=['LOW', 'MEDIUM', 'HIGH'],
       yticklabels=['LOW', 'MEDIUM', 'HIGH'],
       title='Confusion Matrix (E06 - Test Set)',
       ylabel='True Label',
       xlabel='Predicted Label')

# Loop over data dimensions and create text annotations.
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
fig.savefig(os.path.join(out_dir, '06_confusion_matrix.png'), dpi=300)
fig.savefig(os.path.join(out_dir, '06_confusion_matrix.svg'))
plt.close(fig)


# --- 07. Performance Comparison ---
comp = pd.read_csv('tables/model_comparison.csv')
fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(comp['experiment']))
width = 0.35

ax.bar(x - width/2, comp['val_macro_f1'], width, label='Validation Macro F1', color='#88CCEE')
ax.bar(x + width/2, comp['test_macro_f1'], width, label='Test Macro F1', color='#332288')

ax.set_ylabel('Macro F1 Score')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(comp['experiment'], rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_ylim([0.4, 0.52])

fig.tight_layout()
fig.savefig(os.path.join(out_dir, '07_performance_comparison.png'), dpi=300)
fig.savefig(os.path.join(out_dir, '07_performance_comparison.svg'))
plt.close(fig)

print("Charts created.")
