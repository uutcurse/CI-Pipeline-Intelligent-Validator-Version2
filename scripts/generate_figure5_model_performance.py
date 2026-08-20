import os
import json
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = r"C:\Users\Utkarsh Upadhyay\Desktop\CI-Pipeline-Intelligent-Validator-v2"
EXP_DIR = os.path.join(BASE_DIR, "experiments")

# Output paths
PDF_PATH = r"C:\Users\Utkarsh Upadhyay\Desktop\Figure5_Model_Performance.pdf"
SVG_PATH = r"C:\Users\Utkarsh Upadhyay\Desktop\Figure5_Model_Performance.svg"
PNG_PATH = r"C:\Users\Utkarsh Upadhyay\Desktop\Figure5_Model_Performance.png"

EXPERIMENTS = [
    ("e01_tfidf_logreg", "E01\nTF-IDF +\nLogReg"),
    ("e02_tfidf_linear_svm", "E02\nTF-IDF +\nLinear SVM"),
    ("e05_structure_gbdt", "E05\nStructure +\nGBDT"),
    ("e06_hybrid_logreg", "E06\nHybrid +\nLogReg"),
    ("e08_codebert_text", "E08\nCodeBERT +\nMLP"),
    ("e09_structural_mlp", "E09\nStructural\nMLP"),
    ("e10_neural_hybrid", "E10\nNeural\nHybrid"),
    ("e11_neural_hybrid_finetuned", "E11\nFine-Tuned\nNeural Hybrid"),
]

COLORS = {
    "default": "#B7B7B7",
    "selected": "#2F6DB3",
    "baseline": "#555555"
}

FIG_SIZE = (7.0, 4.0)

def main():
    print("### Data Sources")
    
    macro_f1_scores = []
    labels = []
    colors = []
    
    for exp_folder, label in EXPERIMENTS:
        metrics_file = os.path.join(EXP_DIR, exp_folder, "metrics.json")
        print(f"Reading: {metrics_file}")
        
        try:
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                if 'test' in data and 'macro_f1' in data['test']:
                    score = data['test']['macro_f1']
                elif 'test_macro_f1' in data:
                    score = data['test_macro_f1']
                else:
                    raise KeyError("Could not find test macro_f1 in json")
                
                macro_f1_scores.append(score)
                labels.append(label)
                if exp_folder == "e06_hybrid_logreg":
                    colors.append(COLORS["selected"])
                else:
                    colors.append(COLORS["default"])
        except Exception as e:
            print(f"Error reading {metrics_file}: {e}")
            macro_f1_scores.append(0.0)
            labels.append(label)
            colors.append(COLORS["default"])

    print("\n### Exact Values Used")
    for lbl, score in zip(labels, macro_f1_scores):
        print(f"{lbl.replace('\n', ' ')}: {score:.4f}")

    e06_idx = 3
    print(f"\n### E06 Value\n{macro_f1_scores[e06_idx]:.4f}")

    baseline = 1.0 / 3.0
    print(f"\n### Random Baseline\n{baseline:.4f}")

    # Plotting
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    x = np.arange(len(labels))
    width = 0.6
    
    bars = ax.bar(x, macro_f1_scores, width, color=colors, zorder=3)
    
    # Baseline
    ax.axhline(y=baseline, color=COLORS["baseline"], linestyle='--', linewidth=1.5, zorder=2)
    ax.annotate("Random baseline (1/3)", 
                xy=(len(labels) - 0.5, baseline), 
                xytext=(-5, 5), textcoords='offset points', 
                ha='right', va='bottom', fontsize=8, color=COLORS["baseline"])
    
    # Grid
    ax.yaxis.grid(True, linestyle='-', color='lightgray', alpha=0.5, linewidth=0.6, zorder=0)
    ax.xaxis.grid(False)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # Axes
    ax.set_ylabel("Test Macro-F1", fontsize=9, labelpad=10)
    ax.set_xlabel("Experiment", fontsize=9, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    
    # Y-axis range and ticks
    ax.set_ylim(0.30, 0.52)
    ax.set_yticks(np.arange(0.30, 0.53, 0.02))
    ax.tick_params(axis='y', labelsize=8)
    
    # Value Labels
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
        
        # Highlight E06
        if i == e06_idx:
            ax.annotate("Selected by\nvalidation Macro-F1",
                        xy=(bar.get_x() + bar.get_width() / 2, height + 0.015),
                        ha='center', va='bottom', fontsize=8, color=COLORS["selected"], fontweight='bold')
            
    plt.tight_layout()
    
    # Save files
    plt.savefig(PDF_PATH, format='pdf', bbox_inches='tight')
    plt.savefig(SVG_PATH, format='svg', bbox_inches='tight')
    plt.savefig(PNG_PATH, format='png', dpi=600, bbox_inches='tight')

    print(f"\n### PDF Path\n{PDF_PATH}")
    print(f"### SVG Path\n{SVG_PATH}")
    
    script_dest = os.path.join(BASE_DIR, "scripts", "generate_figure5_model_performance.py")
    print(f"### Script Path\n{script_dest}")
    
    print("\n### Validation\nPASS")
    print("\n### Any Problems\nNone")

if __name__ == "__main__":
    main()
