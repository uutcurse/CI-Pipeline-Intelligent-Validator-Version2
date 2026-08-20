import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Data source
CSV_PATH = r"C:\Users\Utkarsh Upadhyay\Desktop\CI-Pipeline-Intelligent-Validator-v2\figure_data\figure_01_class_distribution.csv"

# Output paths
PDF_PATH = r"C:\Users\Utkarsh Upadhyay\Desktop\Figure4_Dataset_Distribution.pdf"
SVG_PATH = r"C:\Users\Utkarsh Upadhyay\Desktop\Figure4_Dataset_Distribution.svg"
PNG_PATH = r"C:\Users\Utkarsh Upadhyay\Desktop\Figure4_Dataset_Distribution.png"

# Script configuration
FIG_SIZE = (7.0, 3.8)
COLORS = {
    0: "#4CAF50",  # LOW
    1: "#F2C94C",  # MEDIUM
    2: "#D9534F"   # HIGH
}
LABELS = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH"
}

def main():
    print("### Data Source")
    print(CSV_PATH)
    
    # Read Data
    df = pd.read_csv(CSV_PATH)
    
    # Process data to ensure order
    splits_order = ["train", "validation", "test"]
    labels_order = [0, 1, 2]
    
    # Prepare data structures for plotting
    data_dict = {
        "Train": {0:0, 1:0, 2:0},
        "Validation": {0:0, 1:0, 2:0},
        "Test": {0:0, 1:0, 2:0}
    }
    
    split_name_map = {
        "train": "Train",
        "validation": "Validation",
        "test": "Test"
    }
    
    for _, row in df.iterrows():
        split = row['split'].lower()
        lbl = int(row['final_label'])
        count = int(row['count'])
        if split in split_name_map and lbl in labels_order:
            data_dict[split_name_map[split]][lbl] = count

    # Print Validation
    print("\n### Counts Used")
    split_names = ["Train", "Validation", "Test"]
    
    low_counts = [data_dict[s][0] for s in split_names]
    med_counts = [data_dict[s][1] for s in split_names]
    high_counts = [data_dict[s][2] for s in split_names]
    
    for s in split_names:
        print(f"{s}:\n  LOW={data_dict[s][0]}\n  MEDIUM={data_dict[s][1]}\n  HIGH={data_dict[s][2]}")
        
    print("\n### Validation")
    for s in split_names:
        total = data_dict[s][0] + data_dict[s][1] + data_dict[s][2]
        print(f"Total {s}: {total}")
        
    # Plotting
    # Use standard sans-serif
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]
    
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    x = np.arange(len(split_names))
    width = 0.22  # approx 0.22-0.25 as requested
    
    # Plot bars
    bars_low = ax.bar(x - width, low_counts, width, label='LOW', color=COLORS[0], zorder=3)
    bars_med = ax.bar(x, med_counts, width, label='MEDIUM', color=COLORS[1], zorder=3)
    bars_high = ax.bar(x + width, high_counts, width, label='HIGH', color=COLORS[2], zorder=3)
    
    # Grid
    ax.yaxis.grid(True, linestyle='-', color='lightgray', alpha=0.25, linewidth=0.6, zorder=0)
    ax.xaxis.grid(False)
    
    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # Axes settings
    ax.set_ylabel("Number of Workflow Versions", fontsize=10, labelpad=10)
    ax.set_xlabel("Dataset Split", fontsize=10, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(split_names, fontsize=10)
    
    ax.set_ylim(0, max(max(low_counts), max(med_counts), max(high_counts)) * 1.15)
    
    # Legend
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    
    # Value Labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{int(height):,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8.5)
                        
    add_labels(bars_low)
    add_labels(bars_med)
    add_labels(bars_high)
    
    plt.tight_layout()
    
    print(f"\n### Figure Dimensions\n{FIG_SIZE[0]} x {FIG_SIZE[1]} inches")
    
    # Save files
    plt.savefig(PDF_PATH, format='pdf', bbox_inches='tight')
    print(f"\n### PDF Path\n{PDF_PATH}")
    
    plt.savefig(SVG_PATH, format='svg', bbox_inches='tight')
    print(f"\n### SVG Path\n{SVG_PATH}")
    
    plt.savefig(PNG_PATH, format='png', dpi=600, bbox_inches='tight')
    print(f"\n### PNG Preview Path\n{PNG_PATH}")

    plt.close()
    print("\n### Any Problems\nNone")

if __name__ == "__main__":
    main()
