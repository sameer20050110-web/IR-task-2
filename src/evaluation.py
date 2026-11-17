"""
evaluation.py
-------------
Evaluation and Publication-Quality Visualization Suite for CLA-2 Task 2.
Generates:
1. Metric Comparison Charts
2. Confusion Matrices (Individual & 2x2 Combined)
3. Class Imbalance and Text Length Distribution Plots
4. Latency and Computational Complexity Charts
5. What-If Experiment Plots

Author: Mohammed Sameer P (RA2411026050199)
Course: 21CSE377T - Information Retrieval
Assignment: CLA-2 Task 2
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "sms_spam_collection.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

# Styling configurations for academic publications
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8


def setup_directories():
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_dataset_distributions():
    """Plot class balance and message length distributions."""
    print("[*] Generating dataset distribution figures...")
    if not os.path.exists(DATA_PATH):
        print(f"[-] Dataset missing at {DATA_PATH}")
        return
        
    df = pd.read_csv(DATA_PATH)
    df['char_length'] = df['message'].astype(str).apply(len)
    df['word_count'] = df['message'].astype(str).apply(lambda s: len(s.split()))
    
    # 1. Class Distribution (Bar + Donut)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts = df['label'].value_counts()
    colors = ['#2b5c8f', '#d9534f']
    
    # Bar chart
    bars = axes[0].bar(counts.index.str.upper(), counts.values, color=colors, width=0.55, edgecolor='black', alpha=0.9)
    axes[0].set_title("SMS Spam Collection: Class Frequency Distribution", fontsize=12, fontweight='bold', pad=12)
    axes[0].set_ylabel("Number of Messages", fontsize=11)
    axes[0].set_xlabel("Message Class", fontsize=11)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        axes[0].annotate(f"{h:,} ({h/len(df)*100:.1f}%)",
                         xy=(bar.get_x() + bar.get_width() / 2, h),
                         xytext=(0, 4), textcoords="offset points",
                         ha='center', va='bottom', fontsize=10, fontweight='bold')
                         
    # Donut Chart
    wedges, texts, autotexts = axes[1].pie(
        counts.values,
        labels=[f"{k.upper()}\n({v:,})" for k, v in counts.items()],
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        explode=(0, 0.08),
        wedgeprops=dict(width=0.45, edgecolor='black')
    )
    for at in autotexts:
        at.set_color('white')
        at.set_fontweight('bold')
    axes[1].set_title("Class Imbalance Ratio (Ham vs Spam)", fontsize=12, fontweight='bold', pad=12)
    
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "class_distribution.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")
    
    # 2. Length Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(data=df, x='char_length', hue='label', bins=60, kde=True,
                 palette={'ham': '#2b5c8f', 'spam': '#d9534f'}, ax=axes[0], element='step', alpha=0.4)
    axes[0].set_title("Character Length Distribution by Class", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Character Count")
    axes[0].set_ylabel("Frequency")
    axes[0].set_xlim(0, 350)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    sns.histplot(data=df, x='word_count', hue='label', bins=40, kde=True,
                 palette={'ham': '#2b5c8f', 'spam': '#d9534f'}, ax=axes[1], element='step', alpha=0.4)
    axes[1].set_title("Word Count Distribution by Class", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Word Count")
    axes[1].set_ylabel("Frequency")
    axes[1].set_xlim(0, 70)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_len = os.path.join(FIG_DIR, "message_length_distribution.png")
    plt.savefig(out_len, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_len}")


def plot_model_comparisons():
    """Plot comparative metrics across the 4 classifiers."""
    csv_path = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
    if not os.path.exists(csv_path):
        print(f"[-] Metrics CSV missing at {csv_path}")
        return
        
    df_metrics = pd.read_csv(csv_path)
    print("[*] Generating model comparison plots...")
    
    # 1. Grouped Bar Chart of Core Metrics
    metrics_to_plot = ['Accuracy', 'Spam Precision', 'Spam Recall', 'Spam F1-Score']
    x = np.arange(len(df_metrics))
    width = 0.18
    
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    for i, metric in enumerate(metrics_to_plot):
        offset = (i - 1.5) * width
        rects = ax.bar(x + offset, df_metrics[metric], width, label=metric, color=palette[i], edgecolor='black')
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=45)

    ax.set_title("Classification Performance Battle (Test Set: 1,115 messages)", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(df_metrics['Model'], fontsize=11, fontweight='bold')
    ax.set_ylim(0.75, 1.05)
    ax.legend(loc='lower right', frameon=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    out_metric = os.path.join(FIG_DIR, "models_metric_comparison.png")
    plt.savefig(out_metric, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_metric}")
    
    # 2. Computational Latency Comparison (Training vs Inference)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Train Time
    ax1.barh(df_metrics['Model'], df_metrics['Training Time (s)'], color='#3498db', edgecolor='black')
    ax1.set_title("Model Training Latency", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Time (seconds)")
    ax1.grid(axis='x', linestyle='--', alpha=0.5)
    for i, v in enumerate(df_metrics['Training Time (s)']):
        ax1.text(v + 0.005, i, f" {v:.4f}s", va='center', fontweight='bold')
        
    # Inference Latency per Sample
    ax2.barh(df_metrics['Model'], df_metrics['Latency per Sample (ms)'], color='#e67e22', edgecolor='black')
    ax2.set_title("Per-Sample Inference Latency", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Latency (milliseconds / sample)")
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    for i, v in enumerate(df_metrics['Latency per Sample (ms)']):
        ax2.text(v + 0.01, i, f" {v:.3f} ms", va='center', fontweight='bold')
        
    plt.tight_layout()
    out_lat = os.path.join(FIG_DIR, "latency_comparison.png")
    plt.savefig(out_lat, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_lat}")


def plot_confusion_matrices():
    """Plot individual and 2x2 grid of confusion matrices."""
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrices.json")
    if not os.path.exists(cm_path):
        print(f"[-] Confusion matrices JSON missing at {cm_path}")
        return
        
    with open(cm_path, "r", encoding="utf-8") as f:
        cms = json.load(f)
        
    print("[*] Generating confusion matrix figures...")
    
    # 2x2 Combined Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()
    
    categories = ['Ham (0)', 'Spam (1)']
    
    for idx, (model_name, data) in enumerate(cms.items()):
        cm = np.array(data['matrix'])
        ax = axes_flat[idx]
        
        # Annotations with descriptive labels
        annot = np.array([
            [f"TN\n{cm[0,0]}", f"FP\n{cm[0,1]}"],
            [f"FN\n{cm[1,0]}", f"TP\n{cm[1,1]}"]
        ])
        
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=categories, yticklabels=categories, annot_kws={"fontsize": 13, "fontweight": "bold"})
        ax.set_title(f"{model_name}\n(TN={data['TN']}, FP={data['FP']}, FN={data['FN']}, TP={data['TP']})",
                     fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Predicted Class", fontsize=10)
        ax.set_ylabel("True Class", fontsize=10)
        
        # Also save individual CM
        fig_single, ax_single = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax_single, cbar=True,
                    xticklabels=categories, yticklabels=categories, annot_kws={"fontsize": 12, "fontweight": "bold"})
        ax_single.set_title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight='bold', pad=10)
        ax_single.set_xlabel("Predicted Class", fontsize=10)
        ax_single.set_ylabel("Actual Class", fontsize=10)
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
        single_path = os.path.join(FIG_DIR, f"confusion_matrix_{safe_name}.png")
        plt.savefig(single_path, dpi=300, bbox_inches='tight')
        plt.close(fig_single)

    plt.tight_layout()
    out_comb = os.path.join(FIG_DIR, "combined_confusion_matrices.png")
    plt.savefig(out_comb, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved combined confusion matrices to: {out_comb}")


def plot_what_if_results():
    """Plot What-If experiment graphs."""
    print("[*] Generating What-If experiment graphs...")
    
    # 1. KNN k-Sensitivity
    knn_path = os.path.join(RESULTS_DIR, "what_if_knn_k.csv")
    if os.path.exists(knn_path):
        df_k = pd.read_csv(knn_path)
        fig, ax1 = plt.subplots(figsize=(10, 5))
        
        ax1.plot(df_k['k_neighbors'], df_k['Accuracy'], marker='o', linewidth=2, color='#2b5c8f', label='Overall Accuracy')
        ax1.plot(df_k['k_neighbors'], df_k['Spam Precision'], marker='s', linewidth=2, color='#27ae60', label='Spam Precision')
        ax1.plot(df_k['k_neighbors'], df_k['Spam Recall'], marker='^', linewidth=2, color='#d9534f', label='Spam Recall')
        ax1.plot(df_k['k_neighbors'], df_k['Spam F1-Score'], marker='d', linewidth=2.5, linestyle='--', color='#8e44ad', label='Spam F1-Score')
        
        ax1.set_title("What-If Analysis A: Impact of K-Neighbors (k) on KNN Performance", fontsize=12, fontweight='bold', pad=12)
        ax1.set_xlabel("Number of Nearest Neighbours (k)", fontsize=11)
        ax1.set_ylabel("Metric Score", fontsize=11)
        ax1.set_xticks(df_k['k_neighbors'])
        ax1.set_ylim(0.0, 1.05)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='lower left', frameon=True, shadow=True)
        
        # Add annotation explaining collapse of recall
        ax1.annotate("High k causes Recall collapse\ndue to Ham majority dominance",
                     xy=(21, df_k.loc[df_k['k_neighbors']==21, 'Spam Recall'].values[0]),
                     xytext=(15, 0.35),
                     arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
                     fontsize=9, fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="#fffae6", ec="#b38f00"))
                     
        plt.tight_layout()
        out_k = os.path.join(FIG_DIR, "what_if_knn_accuracy.png")
        plt.savefig(out_k, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved: {out_k}")

    # 2. TF-IDF max_features
    tfidf_path = os.path.join(RESULTS_DIR, "what_if_tfidf_features.csv")
    if os.path.exists(tfidf_path):
        df_feat = pd.read_csv(tfidf_path)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(df_feat['Actual Vocab Size'], df_feat['SVM Spam F1'], marker='o', linewidth=2, color='#2980b9', label='Linear SVM Spam F1')
        ax.plot(df_feat['Actual Vocab Size'], df_feat['NB Spam F1'], marker='s', linewidth=2, color='#e74c3c', label='Multinomial NB Spam F1')
        ax.plot(df_feat['Actual Vocab Size'], df_feat['SVM Accuracy'], marker='^', linestyle=':', color='#34495e', label='Linear SVM Accuracy')
        ax.plot(df_feat['Actual Vocab Size'], df_feat['NB Accuracy'], marker='v', linestyle=':', color='#e67e22', label='Multinomial NB Accuracy')
        
        ax.set_title("What-If Analysis B: Effect of TF-IDF Vocabulary Dimensionality", fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Actual Extracted Vocabulary Size (Features)", fontsize=11)
        ax.set_ylabel("Metric Score", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', frameon=True, shadow=True)
        
        plt.tight_layout()
        out_f = os.path.join(FIG_DIR, "what_if_tfidf_max_features.png")
        plt.savefig(out_f, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved: {out_f}")


def run_all_evaluations():
    setup_directories()
    plot_dataset_distributions()
    plot_model_comparisons()
    plot_confusion_matrices()
    plot_what_if_results()
    print("[+] All evaluation figures generated successfully.")


if __name__ == "__main__":
    run_all_evaluations()

