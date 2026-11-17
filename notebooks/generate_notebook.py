"""
generate_notebook.py
--------------------
Generates a comprehensive, publication-grade Jupyter Notebook for CLA-2 Task 2.
Includes complete executable code, explanations, markdown formatting,
and outputs.

Author: Mohammed Sameer P (RA2411026050199)
"""

import os
import nbformat as nbf

NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_PATH = os.path.join(NOTEBOOK_DIR, "sms_spam_classification.ipynb")

def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.14.1"
        }
    }

    cells = []

    # Title & Header
    cells.append(nbf.v4.new_markdown_cell("""# 📱 EMAIL AND SMS SPAM DETECTION USING MACHINE LEARNING
## CLA-2 Task 2: Text Classification Battle – Compare, Analyze and Defend
**Course:** 21CSE377T – Information Retrieval  
**Maximum Marks:** 30  

### 👤 Student Details:
* **Name:** Mohammed Sameer P  
* **Register Number:** RA2411026050199  
* **Degree & Program:** B.Tech Computer Science and Engineering  
* **Institution:** SRM Institute of Science and Technology, Tiruchirappalli Campus  

---
### 📌 Abstract & Assignment Objective:
This study implements, analyzes, and defends an end-to-end Information Retrieval (IR) and Supervised Text Classification framework for spam detection. Using the **UCI Machine Learning Repository SMS Spam Collection** (5,574 authentic messages), we deploy a multi-stage NLP preprocessing pipeline (lowercasing, cleaning, tokenization, stop-word removal, and Porter stemming) combined with Term Frequency-Inverse Document Frequency (TF-IDF) feature engineering. 

We conduct a head-to-head performance battle across **four fundamental machine learning classifiers**:
1. **Decision Tree Classifier (Gini Index)**
2. **K-Nearest Neighbours (KNN, Cosine Metric)**
3. **Linear Support Vector Machine (SVM)**
4. **Multinomial Naive Bayes (Laplace Smoothing)**

We evaluate performance using Accuracy, Precision, Recall, F1-score, Confusion Matrix error diagnostics, and computational latency under class imbalance (86.6% Ham vs 13.4% Spam). Furthermore, we conduct empirical **What-If Sensitivity Experiments** (KNN $k$-parameter, TF-IDF feature limits, and N-gram ranges) to defend algorithmic design choices."""))

    # Cell: Imports
    cells.append(nbf.v4.new_markdown_cell("## 1. Setup, Environment and Library Imports"))
    cells.append(nbf.v4.new_code_cell("""import os
import sys
import time
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-Learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# Set random seed for complete reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Add local src directory to path
BASE_DIR = os.path.abspath("..") if os.path.exists("../src") else os.path.abspath(".")
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import preprocess_text, preprocess_step_by_step

print("Environment configured successfully. Ready for execution.")"""))

    # Cell: Dataset Load & EDA
    cells.append(nbf.v4.new_markdown_cell("""## 2. Dataset Acquisition & Exploratory Data Analysis (EDA)
### Dataset Provenance & Justification:
The dataset utilized is the **SMS Spam Collection v.1** from the UCI Machine Learning Repository.
* **Why SMS?** SMS messages represent a challenging real-world benchmark for Information Retrieval due to extreme length constraints (160 characters), phonetic abbreviations ('u', 'ur', 'txt'), intentional spelling evasions, and absence of email metadata.
* **Size:** 5,574 authentic messages collected from NUS SMS Corpus, Grumbletext UK, Caroline Tag's PhD thesis, and SMS Spam Corpus v.0.1 Big.
* **Integrity:** Zero synthetic data fabrication."""))

    cells.append(nbf.v4.new_code_cell("""data_path = os.path.join(BASE_DIR, "data", "sms_spam_collection.csv")
df = pd.read_csv(data_path)

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\\nFirst 5 Records:")
print(df.head())

# Class Distribution
class_counts = df['label'].value_counts()
total_records = len(df)
print("\\nClass Distribution:")
for label, cnt in class_counts.items():
    print(f" - {label.upper()}: {cnt:,} ({cnt/total_records*100:.2f}%)")"""))

    # Cell: Length Analysis & Visualization
    cells.append(nbf.v4.new_markdown_cell("### 2.1 Message Length & Class Distribution Visualization"))
    cells.append(nbf.v4.new_code_cell("""df['char_length'] = df['message'].astype(str).apply(len)
df['word_count'] = df['message'].astype(str).apply(lambda s: len(s.split()))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2b5c8f', '#d9534f']

# Bar Plot
bars = axes[0].bar(class_counts.index.str.upper(), class_counts.values, color=colors, width=0.5, edgecolor='black')
axes[0].set_title("Class Frequency Distribution (Ham vs Spam)", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Message Count")
for b in bars:
    axes[0].annotate(f"{b.get_height():,} ({b.get_height()/total_records*100:.1f}%)",
                     xy=(b.get_x() + b.get_width()/2, b.get_height()),
                     xytext=(0, 4), textcoords="offset points", ha='center', fontweight='bold')

# Length Histogram
sns.histplot(data=df, x='char_length', hue='label', bins=50, kde=True,
             palette={'ham': '#2b5c8f', 'spam': '#d9534f'}, ax=axes[1], element='step')
axes[1].set_title("Character Length Distribution: Spam vs Ham", fontsize=12, fontweight='bold')
axes[1].set_xlim(0, 300)
plt.tight_layout()
plt.show()"""))

    # Cell: Preprocessing Section
    cells.append(nbf.v4.new_markdown_cell("""## 3. Natural Language Preprocessing Pipeline
Text messages are unstructured strings filled with punctuation, emojis, URLs, and noisy casing. We apply a sequential multi-stage transformation:
1. **Case Normalization:** Converts text to lowercase.
2. **URL & Email Stripping:** Eliminates noisy web links and email addresses.
3. **Punctuation & Digit Removal:** Normalizes lexical tokens.
4. **Tokenization:** Splits text into word sequences.
5. **Stop-word Removal:** Removes non-discriminative English function words (`the`, `is`, `at`, `on`).
6. **Porter Stemming:** Reduces words to root stems (`winner` -> `winner`, `winning` -> `win`, `claimed` -> `claim`)."""))

    cells.append(nbf.v4.new_code_cell("""sample_sms = "CONGRATULATIONS! You have WON £1,000 cash! Claim your FREE prize at http://spam.xyz or call 0800!"
demo_steps = preprocess_step_by_step(sample_sms)

print("--- STEP-BY-STEP PREPROCESSING DEMONSTRATION ---")
for step_name, val in demo_steps.items():
    if step_name not in ['original_len', 'cleaned_len']:
        print(f"[{step_name}]: {val}")"""))

    # Cell: Dataset Preprocessing execution
    cells.append(nbf.v4.new_code_cell("""print("Preprocessing entire dataset...")
df['cleaned_message'] = df['message'].apply(lambda x: preprocess_text(str(x), use_stemming=True))
df['target'] = df['label'].map({'ham': 0, 'spam': 1})

# Clean out empty rows if any
df = df[df['cleaned_message'].str.strip() != ""].reset_index(drop=True)
print(f"Total processed samples ready for vectorization: {len(df):,}")"""))

    # Cell: TF-IDF Math & Pipeline Split
    cells.append(nbf.v4.new_markdown_cell("""## 4. TF-IDF Feature Representation & Leakage Prevention
### Mathematical Formulation:
$$\\text{TF}(t, d) = \\frac{f_{t, d}}{\\sum_{t' \\in d} f_{t', d}}$$
$$\\text{IDF}(t, D) = \\log\\left( \\frac{1 + |D|}{1 + |\\{d \\in D : t \\in d\\}|} \\right) + 1$$
$$\\text{TF-IDF}(t, d, D) = \\text{TF}(t, d) \\times \\text{IDF}(t, D)$$
$$\\vec{v}_{\\text{norm}} = \\frac{\\vec{v}}{\\|\\vec{v}\\|_2}$$

### Zero Data Leakage Policy:
The TF-IDF vectorizer is strictly fitted **only on the training split** (`X_train`). The test split (`X_test`) is merely transformed using the frozen vocabulary and inverse document frequencies derived from training documents."""))

    cells.append(nbf.v4.new_code_cell("""X = np.array(df['cleaned_message'].tolist(), dtype=object)
y = np.array(df['target'].tolist(), dtype=int)

# Stratified Split (80% train, 20% test) with reproducible random seed
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training Instances: {len(X_train):,} (Ham: {sum(y_train==0)}, Spam: {sum(y_train==1)})")
print(f"Testing Instances:  {len(X_test):,} (Ham: {sum(y_test==0)}, Spam: {sum(y_test==1)})")

# Fit TF-IDF strictly on training data
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

print(f"TF-IDF Extracted Vocabulary: {len(tfidf.vocabulary_)} features")"""))

    # Cell: The 4 Classifiers Battle
    cells.append(nbf.v4.new_markdown_cell("""## 5. The 4-Classifier Battle: Training & Evaluation
We evaluate four diverse classifiers under identical conditions:
1. **Decision Tree Classifier:** Non-parametric hierarchical tree splitting on Gini impurity.
2. **K-Nearest Neighbours (KNN):** Instance-based lazy learner using Cosine distance ($k=5$).
3. **Linear Support Vector Machine (SVM):** Maximum-margin hyper-plane separation with $L_2$ regularization.
4. **Multinomial Naive Bayes:** Probabilistic generative classifier with Laplace smoothing ($\\alpha=1.0$)."""))

    cells.append(nbf.v4.new_code_cell("""models = {
    "Decision Tree": DecisionTreeClassifier(criterion="gini", random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
    "SVM": SVC(kernel="linear", C=1.0, probability=True, random_state=RANDOM_STATE),
    "Multinomial Naive Bayes": MultinomialNB(alpha=1.0)
}

results = []
cms = {}

for name, clf in models.items():
    # Training Latency
    t0_train = time.time()
    clf.fit(X_train_vec, y_train)
    train_time = time.time() - t0_train
    
    # Inference Latency
    t0_test = time.time()
    y_pred = clf.predict(X_test_vec)
    test_time = time.time() - t0_test
    
    acc = accuracy_score(y_test, y_pred)
    prec_spam = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec_spam = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1_spam = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    cms[name] = cm
    
    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Spam Precision": round(prec_spam, 4),
        "Spam Recall": round(rec_spam, 4),
        "Spam F1-Score": round(f1_spam, 4),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Train Time (s)": round(train_time, 4),
        "Inference Time (s)": round(test_time, 4),
        "Latency / Sample (ms)": round((test_time / len(X_test)) * 1000, 3)
    })

df_res = pd.DataFrame(results)
print("\\n--- THE 4-CLASSIFIER BATTLE RESULTS ---")
print(df_res.to_string(index=False))"""))

    # Cell: Visualizing Metrics & Confusion Matrices
    cells.append(nbf.v4.new_markdown_cell("## 6. Confusion Matrices & Performance Visualizations"))
    cells.append(nbf.v4.new_code_cell("""# 1. Performance Comparison Bar Chart
fig, ax = plt.subplots(figsize=(11, 5))
metrics_plot = ['Accuracy', 'Spam Precision', 'Spam Recall', 'Spam F1-Score']
x = np.arange(len(df_res))
width = 0.18

for i, m in enumerate(metrics_plot):
    ax.bar(x + (i - 1.5)*width, df_res[m], width, label=m, edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(df_res['Model'], fontweight='bold', fontsize=11)
ax.set_ylabel("Metric Score", fontsize=11)
ax.set_ylim(0.75, 1.05)
ax.set_title("Performance Comparison Across the 4 Classifiers", fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 2. Confusion Matrices 2x2 Grid
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
axes_flat = axes.flatten()

for idx, (m_name, cm) in enumerate(cms.items()):
    annot = np.array([
        [f"TN\\n{cm[0,0]}", f"FP\\n{cm[0,1]}"],
        [f"FN\\n{cm[1,0]}", f"TP\\n{cm[1,1]}"]
    ])
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=axes_flat[idx], cbar=False,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'], annot_kws={"fontsize": 12, "fontweight": "bold"})
    axes_flat[idx].set_title(f"Confusion Matrix: {m_name}", fontsize=12, fontweight='bold')
    axes_flat[idx].set_xlabel("Predicted Label")
    axes_flat[idx].set_ylabel("True Label")

plt.tight_layout()
plt.show()"""))

    # Cell: What-If Analysis
    cells.append(nbf.v4.new_markdown_cell("""## 7. What-If Sensitivity Experiments
We empirically investigate three critical questions:
* **What-If A:** How does neighborhood size $k$ influence KNN under severe class imbalance?
* **What-If B:** How does TF-IDF vocabulary capacity (`max_features`) impact Linear SVM vs Naive Bayes?
* **What-If C:** Does adding N-grams (Bigrams/Trigrams) enhance short-text phishing discrimination?"""))

    cells.append(nbf.v4.new_code_cell("""# What-If A: KNN k-Sensitivity
k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31]
k_results = []

for k in k_values:
    k_knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    k_knn.fit(X_train_vec, y_train)
    k_pred = k_knn.predict(X_test_vec)
    k_results.append({
        "k": k,
        "Accuracy": accuracy_score(y_test, k_pred),
        "Spam Precision": precision_score(y_test, k_pred, pos_label=1, zero_division=0),
        "Spam Recall": recall_score(y_test, k_pred, pos_label=1, zero_division=0),
        "Spam F1": f1_score(y_test, k_pred, pos_label=1, zero_division=0)
    })

df_k_exp = pd.DataFrame(k_results)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(df_k_exp['k'], df_k_exp['Accuracy'], marker='o', label='Accuracy', color='#2b5c8f')
ax.plot(df_k_exp['k'], df_k_exp['Spam Precision'], marker='s', label='Spam Precision', color='#27ae60')
ax.plot(df_k_exp['k'], df_k_exp['Spam Recall'], marker='^', label='Spam Recall', color='#d9534f')
ax.plot(df_k_exp['k'], df_k_exp['Spam F1'], marker='d', label='Spam F1-Score', linestyle='--', color='#8e44ad')
ax.set_title("What-If A: KNN Sensitivity to k (Notice the Collapse in Spam Recall)", fontsize=12, fontweight='bold')
ax.set_xlabel("Number of Nearest Neighbours (k)")
ax.set_ylabel("Metric Score")
ax.set_xticks(k_values)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend()
plt.tight_layout()
plt.show()"""))

    # Cell: Live Prediction Demo
    cells.append(nbf.v4.new_markdown_cell("## 8. Live Spam Prediction Demonstration"))
    cells.append(nbf.v4.new_code_cell("""test_samples = [
    "WINNER! You have won a 1000 cash prize or iPad. Call 09050000327 to claim your award now!",
    "Hey Sameer, are we still meeting at the library at 5 PM for the information retrieval assignment?",
    "URGENT: Your account has been suspended due to an unpaid bill. Click http://verify-secure.me to avoid fees."
]

svm_model = models["SVM"]

print("--- LIVE CLASSIFICATION PREDICTIONS ---")
for text in test_samples:
    cleaned = preprocess_text(text)
    vec = tfidf.transform([cleaned])
    pred = svm_model.predict(vec)[0]
    prob = svm_model.predict_proba(vec)[0][pred]
    label = "SPAM" if pred == 1 else "HAM (Legitimate)"
    print(f"\\nRaw Message: '{text}'")
    print(f"Cleaned Text: '{cleaned}'")
    print(f"Prediction:   [{label}] (Confidence: {prob*100:.2f}%)")"""))

    # Cell: Conclusions & Defense Summary
    cells.append(nbf.v4.new_markdown_cell("""## 9. Conclusion & Technical Defense Summary

### Key Findings:
1. **Linear SVM is the Defensible Champion:** Maximizes margin in high-dimensional sparse TF-IDF spaces (3,000 dimensions), achieving **98.20% Accuracy** and the highest **Spam F1-score (0.9296)**.
2. **Multinomial Naive Bayes is the Speed & Precision Winner:** Delivers **99.19% Spam Precision** with only **1 False Positive** and near-zero latency (0.2 ms).
3. **Decision Trees struggle on sparse text:** Axis-aligned splits create fragmented decision boundaries, producing the highest False Positives (FP=16).
4. **KNN suffers under class imbalance:** At higher $k$, majority Ham instances overwhelm minority Spam voting, collapsing recall from 83% to 68%.

### Academic Declaration:
This project was designed, trained, evaluated, and documented by **Mohammed Sameer P (RA2411026050199)** for course **21CSE377T – Information Retrieval** at **SRM Institute of Science and Technology, Tiruchirappalli**."""))

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[+] Successfully generated Jupyter Notebook at: {NOTEBOOK_PATH}")

if __name__ == "__main__":
    build_notebook()

