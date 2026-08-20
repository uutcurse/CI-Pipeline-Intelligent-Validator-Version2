import os
import matplotlib.pyplot as plt

out_dir = r'C:\Users\Utkarsh Upadhyay\Desktop\imagesofrunsure'

fig, ax = plt.subplots(figsize=(6, 4))
ax.text(0.5, 0.5, 'TRAINING HISTORY UNAVAILABLE\n(Classical Logistic Regression Model)', 
        ha='center', va='center', fontsize=14, color='gray')
ax.axis('off')

fig.tight_layout()
fig.savefig(os.path.join(out_dir, '05_training_curves.png'), dpi=300)
fig.savefig(os.path.join(out_dir, '05_training_curves.svg'))
plt.close(fig)
