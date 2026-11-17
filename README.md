# 🛡️ EMAIL AND SMS SPAM DETECTION USING MACHINE LEARNING
### CLA-2 Task 2: Text Classification Battle – Compare, Analyze and Defend
**Course:** 21CSE377T – Information Retrieval | **Maximum Marks:** 30  
**Institution:** SRM Institute of Science and Technology, Tiruchirappalli Campus  

---

### 👤 Student Information
* **Student Name:** Mohammed Sameer P
* **Register Number:** `RA2411026050199`
* **Degree / Program:** B.Tech Computer Science and Engineering
* **Department:** Department of Computer Science & Engineering
* **Course Code & Title:** 21CSE377T – Information Retrieval
* **Academic Year:** 2026

---

## 📌 Executive Summary & Assignment Scope
This project provides a complete, mathematically grounded, reproducible Information Retrieval (IR) and Supervised Text Classification framework. Using the gold-standard **UCI SMS Spam Collection benchmark (5,574 authentic messages)**, we design and implement a multi-stage NLP text preprocessing pipeline and an inverse document frequency (TF-IDF) feature engineering engine with strict data leakage prevention.

We pit **four classic machine learning algorithms** against each other under identical stratified test conditions:
1. **Decision Tree Classifier (CART, Gini Impurity)**
2. **K-Nearest Neighbours (KNN, Cosine Metric, k=5)**
3. **Linear Support Vector Machine (Linear SVM, C=1.0)**
4. **Multinomial Naive Bayes (MNB, Laplace Smoothing $\alpha=1.0$)**

### 🏆 Key Battle Takeaways:
* **Defensible Champion:** **Linear SVM** achieved the highest overall **Accuracy (98.20%)**, **Spam Recall (88.00%)**, and **Spam F1-score (0.9296)**, with exceptional False Positive control (only 2 out of 963 legitimate messages).
* **Speed & Precision Champion:** **Multinomial Naive Bayes** achieved **99.19% Spam Precision** (only 1 False Positive) with sub-microsecond inference latency ($0.2\,\mu\text{s}$).
* **Decision Tree struggles:** High-dimensional sparse text vectors caused fragmented tree splits, resulting in 16 False Positives (8x worse than SVM).
* **KNN collapses under class imbalance:** What-If experiments prove that expanding $k$ from 3 to 31 reduces Spam Recall from 83.33% to 68.00% as the 86.6% majority Ham class overwhelms voting.

---

## 🏛️ System Architecture

```
┌─────────────────────────────────┐
│     Raw Text Communications     │
│   (UCI Benchmark: 5,574 msgs)   │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│   Multi-Tier NLP Preprocessor   │
│  • Case Normalization (lower)   │
│  • Regex URL & Email Stripping  │
│  • Punctuation & Digit Removal  │
│  • Tokenization (Word-level)    │
│  • Stop-word Removal (NLTK)     │
│  • Porter Morphological Stem    │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│ Stratified Split (80/20, Seed=42│
│ Train: 4,448 | Test: 1,113 msgs │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  TF-IDF Feature Representation  │
│  • Sublinear Term Frequency     │
│  • Smooth Inverse Document Freq │
│  • Strict Fit on Train Partition│
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                 THE 4-CLASSIFIER BATTLE                     │
│  1. Decision Tree       │  2. K-Nearest Neighbours (k=5)    │
│  3. Linear SVM (C=1.0)  │  4. Multinomial Naive Bayes (α=1) │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│                 EMPIRICAL DEFENCE & WHAT-IF                 │
│  • Accuracy, Precision, Recall, F1, Latency                 │
│  • Asymmetric Error Diagnostics (Cost of FP vs FN)          │
│  • What-If Sensitivity Experiments (k, max_features, n-gram)│
│  • Interactive Streamlit Web UI + Multi-Model Consensus     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 The 4-Classifier Battle Results

All metrics computed on the identical stratified test partition of **1,113 messages (963 Ham, 150 Spam)**:

| Classifier | Accuracy | Spam Precision | Spam Recall | Spam F1-Score | Macro F1 | ROC-AUC | Training Time | Latency / Sample |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** 🥇 | **98.20%** | **98.51%** | **88.00%** | **0.9296** | **0.9596** | **0.9880** | 1.4489s | 0.046 ms |
| **Multinomial NB** 🥈 | **97.39%** | **99.19%** | **81.33%** | **0.8938** | **0.9395** | **0.9854** | **0.0017s** | **0.0002 ms** |
| **KNN (k=5)** 🥉 | **97.12%** | **98.36%** | **80.00%** | **0.8824** | **0.9331** | **0.9634** | 0.0009s | 0.047 ms |
| **Decision Tree** | **96.68%** | **88.97%** | **86.00%** | **0.8746** | **0.9278** | **0.9217** | 0.1176s | 0.001 ms |

---

## 🎯 Confusion Matrix & Error Profile Analysis

| Classifier | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | FP Rate (Cost Impact) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | 961 | 2 | 18 | 132 | **0.21% (Very Safe)** |
| **Multinomial NB** | **962** | **1** | 28 | 122 | **0.10% (Safest)** |
| **KNN (k=5)** | 961 | 2 | 30 | 120 | 0.21% (Very Safe) |
| **Decision Tree** | 947 | **16** | 21 | 129 | **1.66% (High Risk)** |

> **Asymmetric Error Cost Defense:**  
> In real-world spam filtering, **False Positives (FP) are catastrophic**. When a legitimate message (such as a banking OTP, appointment confirmation, or flight ticket) is misclassified as spam, direct user disruption occurs.  
> * **Multinomial Naive Bayes** produced only **1 False Positive** out of 963 legitimate messages.  
> * **Linear SVM** produced **2 False Positives** while capturing 132 out of 150 spam messages (88% recall).  
> * **Decision Tree** produced **16 False Positives**, an unacceptable 8x to 16x higher error rate.

---

## 🔬 Empirical What-If Experiments Summary

1. **What-If A (KNN Neighborhood Size $k$):**  
   Increasing $k$ from 3 to 31 caused Spam Recall to plummet from **83.33% to 68.00%**. In an 86.6% Ham dataset, larger neighborhood spheres become inundated by majority-class points, starving the minority spam class of majority votes.
2. **What-If B (TF-IDF Feature Truncation):**  
   Constraining features to 200 led to underfitting (F1 ~0.87). Scaling to 1,000–2,000 features yielded peak stability (SVM F1: 0.9412). At unlimited features (30,508), Naive Bayes recall collapsed to 70.0% due to tail-frequency distortions, whereas SVM remained resilient due to $L_2$ weight regularization.
3. **What-If C (N-Gram Dynamics):**  
   Bigrams alone (2,2) degraded F1 to 0.8289 due to text sparsity. Unigrams + Bigrams (1,2) achieved peak precision (98.51%) by capturing critical phishing collocations (`free cash`, `claim prize`, `call now`).

---

## 📁 Repository Directory Structure

```
d:\IR task2 sameer\
├── .gitignore                      # Git configuration & ignore rules
├── README.md                       # Master project documentation (this file)
├── requirements.txt                # Pinned dependencies for reproduction
├── app.py                          # Modern Streamlit 8-tab interactive dashboard
│
├── data/                           # Data acquisition and validation
│   ├── download_dataset.py         # Automated UCI dataset downloader
│   ├── dataset_info.json           # Validated dataset metadata & distributions
│   ├── README.md                   # Dataset citations & provenance documentation
│   ├── SMSSpamCollection           # Original raw UCI tab-separated text corpus
│   └── sms_spam_collection.csv     # Clean structured CSV representation
│
├── src/                            # Modular, production-ready Python source
│   ├── __init__.py                 # Package declaration
│   ├── preprocessing.py            # 6-stage NLP cleaning & scikit-learn transformer
│   ├── model_training.py           # Model training, serialization, & evaluation
│   ├── evaluation.py               # Publication-grade chart & matrix generation
│   ├── what_if_analysis.py         # Empirical sensitivity experiment suite
│   └── prediction.py               # Live inference engine & multi-model consensus
│
├── models/                         # Serialized Scikit-Learn pipelines
│   ├── tfidf_vectorizer.joblib     # Standalone TF-IDF vectorizer (train-fitted)
│   ├── decision_tree.joblib        # Fitted Decision Tree Pipeline
│   ├── knn.joblib                  # Fitted KNN Pipeline
│   ├── svm.joblib                  # Fitted Linear SVM Pipeline (Champion)
│   └── multinomial_naive_bayes.joblib # Fitted Multinomial Naive Bayes Pipeline
│
├── results/                        # Experimental artifacts & metrics
│   ├── metrics_comparison.csv      # Test set metrics table (Accuracy, P, R, F1, Latency)
│   ├── classification_reports.json # Full per-class classification reports
│   ├── confusion_matrices.json     # Numerical confusion matrix elements
│   ├── what_if_knn_k.csv           # KNN k-sensitivity experimental results
│   ├── what_if_tfidf_features.csv  # TF-IDF feature capacity results
│   ├── what_if_ngram.csv           # N-gram range experimental results
│   └── figures/                    # High-resolution (300 DPI) publication charts
│       ├── class_distribution.png
│       ├── message_length_distribution.png
│       ├── models_metric_comparison.png
│       ├── combined_confusion_matrices.png
│       ├── latency_comparison.png
│       ├── what_if_knn_accuracy.png
│       └── what_if_tfidf_max_features.png
│
├── notebooks/                      # Interactive Jupyter Notebook
│   ├── generate_notebook.py        # Automated notebook builder script
│   └── sms_spam_classification.ipynb # Complete executed end-to-end notebook
│
├── presentation/                   # PowerPoint slides & defence materials
│   ├── generate_presentation.py    # Python-pptx 14-slide deck builder
│   ├── SLIDES_OUTLINE.md           # Slide-by-slide outline & speaker notes
│   └── SMS_Spam_Detection_Presentation.pptx # Professional 16:9 widescreen deck
│
├── report/                         # Formal academic submission documents
│   ├── ACADEMIC_REPORT.md          # Full academic report in Markdown
│   ├── generate_docx_report.py     # Academic Word document generator
│   └── ACADEMIC_REPORT.docx        # Professional formatted Word submission document
│
└── docs/                           # Defence & Viva preparation materials
    ├── VIVA_QUESTIONS.md           # 25 in-depth technical viva questions & answers
    └── TECHNICAL_DEFENCE_GUIDE.md  # 1-page quick technical defence cheat sheet
```

---

## 🚀 Execution & Reproduction Guide

### Windows Command Prompt (CMD) Execution

#### 1. Navigate to Project Directory
```cmd
cd /d "d:\IR task2 sameer"
```

#### 2. Install Required Dependencies
```cmd
pip install -r requirements.txt
```

#### 3. Download Dataset (If not already present)
```cmd
python data\download_dataset.py
```

#### 4. Run Complete Training & Pipeline
```cmd
python src\model_training.py
```

#### 5. Run What-If Experiments & Visualization
```cmd
python src\what_if_analysis.py
python src\evaluation.py
```

#### 6. Test Live Inference Engine
```cmd
python src\prediction.py
```

#### 7. Launch the Streamlit Interactive Dashboard
```cmd
streamlit run app.py
```
*The web dashboard will automatically open in your default browser at `http://localhost:8501`.*

---

## 🖥️ Streamlit Dashboard Features
* **🏠 Project Overview:** Student details, assignment context, problem statement, and system architecture.
* **📊 Dataset Exploration:** Class imbalance charts, character & word count distributions, and searchable data table.
* **🧹 Preprocessing Pipeline:** Interactive text box demonstrating live step-by-step transformations.
* **🧮 TF-IDF Formulation:** Detailed KaTeX mathematical equations and top discriminative spam vs ham keywords.
* **⚔️ 4-Classifier Battle:** Interactive metric tables, grouped bar charts, and individual confusion matrix inspection.
* **🔬 What-If Experiments:** Toggle between KNN $k$-analysis, TF-IDF feature spectrum, and N-gram trade-offs.
* **🛡️ Live Spam Predictor:** Test preset or custom messages, select active model, or trigger a 4-model consensus voting battle with latency timer.
* **📚 Viva Reference Guide:** Expandable interactive technical questions and model defence answers.

---

## 🔒 Security & Operational Notice
This classification system employs statistical Information Retrieval and machine learning techniques. While achieving a **98.20% accuracy**, statistical heuristics do not provide an infallible defense against evolving adversarial tactics, zero-width character insertions, or homoglyph character substitutions (e.g., substituting Latin 'a' with Cyrillic 'а'). In production telecommunications or enterprise email environments, this classifier must be integrated within a multi-tiered security defense-in-depth architecture.

---

## 📖 Academic References
1. Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: New collection and results. *Information Sciences*, 181(11), 2422-2438.
2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
3. Joachims, T. (1998). Text categorization with Support Vector Machines: Learning with many relevant features. *European Conference on Machine Learning (ECML)*, 137-142.
4. McCallum, A., & Nigam, K. (1998). A comparison of event models for Naive Bayes text classification. *AAAI-98 Workshop on Learning for Text Categorization*, 752, 41-48.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research (JMLR)*, 12, 2825-2830.

---

**Developed and Defended by:**  
**Mohammed Sameer P (RA2411026050199)**  
Department of Computer Science & Engineering  
SRM Institute of Science and Technology, Tiruchirappalli

