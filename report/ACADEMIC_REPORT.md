# SRM INSTITUTE OF SCIENCE AND TECHNOLOGY
### FACULTY OF ENGINEERING AND TECHNOLOGY
### DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING — TIRUCHIRAPPALLI CAMPUS

---

## CONTINUOUS LEARNING ASSESSMENT – II (CLA-2 TASK 2)
### COURSE CODE: 21CSE377T – INFORMATION RETRIEVAL
### ASSIGNMENT TITLE: TEXT CLASSIFICATION BATTLE: COMPARE, ANALYZE AND DEFEND
**MAXIMUM MARKS: 30**

---

# PROJECT TITLE:
# EMAIL AND SMS SPAM DETECTION USING MACHINE LEARNING

### STUDENT DETAILS:
* **Student Name:** Mohammed Sameer P
* **Register Number:** RA2411026050199
* **Degree / Program:** B.Tech Computer Science and Engineering
* **Course:** 21CSE377T – Information Retrieval
* **Academic Year:** 2026
* **Institution:** SRM Institute of Science and Technology, Tiruchirappalli Campus

---

## ABSTRACT
Unsolicited electronic messaging, encompassing email spam and mobile Short Message Service (SMS) spam, imposes massive operational costs on communication infrastructure, threatens user privacy, and exposes consumers to phishing and financial fraud. This project designs, evaluates, and defends a comprehensive Information Retrieval (IR) and Supervised Text Classification framework to distinguish legitimate messages (**Ham**) from unsolicited intrusions (**Spam**). Using the gold-standard **UCI SMS Spam Collection benchmark (5,574 authentic messages)**, we deploy a multi-tier Natural Language Processing (NLP) pipeline (case folding, URL/symbol stripping, word tokenization, stop-word elimination, and Porter stemming) combined with Term Frequency-Inverse Document Frequency (TF-IDF) feature engineering fitted with strict data leakage prevention. We orchestrate an empirical battle across four classic machine learning algorithms: **Decision Tree (CART)**, **K-Nearest Neighbours (KNN)**, **Linear Support Vector Machine (SVM)**, and **Multinomial Naive Bayes (MNB)** under an identical stratified 80/20 train-test split. The empirical results reveal that **Linear SVM** is the definitive champion, achieving **98.20% Accuracy**, **0.9296 Spam F1-score**, and an **88.00% Spam Recall** with exceptional False Positive control (only 2 out of 963 legitimate messages). **Multinomial Naive Bayes** achieved the highest Precision (**99.19%**) with sub-microsecond latency. Extensive What-If sensitivity experiments confirm that KNN recall collapses as $k$ expands due to severe class imbalance (86.6% Ham), while TF-IDF feature truncation between 1,000–2,000 n-grams optimizes discriminative stability.

---

## 1. INTRODUCTION & BACKGROUND
Information Retrieval (IR) has transitioned from traditional keyword-matching indexing toward semantic categorization and predictive text classification. Unsolicited mobile messaging (SMS spam) has emerged as an especially perilous threat vector due to mobile devices' personal nature, high open rates (often exceeding 95% within 3 minutes), and the severe lack of built-in transport security metadata compared to desktop email protocols. 

Text classification in mobile messaging operates under stringent constraints:
1. **Length Limitation:** SMS messages are bounded by telecom protocols to 160 septets (characters), yielding an average document length of merely 15 to 20 words.
2. **Lexical Noise:** Users and malicious adversaries employ phonetic abbreviations (`u`, `ur`, `txt`), non-standard orthography, deliberate character substitutions, and emoji obfuscations.
3. **Severe Class Imbalance:** The vast majority of mobile traffic consists of legitimate interpersonal conversations, with spam representing a minority fraction (~13-15%).

This investigation systematically addresses these challenges by developing a transparent, mathematically grounded, and reproducible classification pipeline.

---

## 2. PROBLEM STATEMENT & OBJECTIVES

### 2.1 Problem Statement
Given an arbitrary raw text string $d$ arriving at an electronic gateway, classify $d$ into one of two mutually exclusive categories:
$$c \in \{\text{Ham (0)}, \text{Spam (1)}\}$$
such that:
1. **False Positives (FP)**—legitimate messages erroneously classified as spam—are minimized to prevent catastrophic communication disruptions.
2. **False Negatives (FN)**—spam messages that evade detection—are minimized to preserve inbox security.
3. The computational latency per message remains well within real-time streaming constraints ($< 50\text{ ms}$).

### 2.2 Project Objectives
* **Objective 1:** Ingest and statistically profile the authentic UCI SMS Spam Collection benchmark without synthetic fabrication.
* **Objective 2:** Design a robust, multi-stage NLP preprocessing pipeline that handles casing, URLs, punctuation, stopwords, and morphological inflection.
* **Objective 3:** Implement TF-IDF feature extraction with zero data leakage using Scikit-Learn pipelines.
* **Objective 4:** Implement four distinct machine learning classifiers: Decision Tree, KNN, Linear SVM, and Multinomial Naive Bayes.
* **Objective 5:** Evaluate performance across Accuracy, Precision, Recall, F1-Score, Confusion Matrices, and Latency under stratified test conditions.
* **Objective 6:** Conduct real-world What-If sensitivity experiments on hyperparameter variations.
* **Objective 7:** Deploy an interactive Streamlit application and provide a defensible viva reference guide.

---

## 3. DATASET DESCRIPTION & QUALITY VERIFICATION

### 3.1 Provenance and Acquisition
The dataset employed is the **SMS Spam Collection v.1**, hosted by the UCI Machine Learning Repository (compiled by Almeida, Gómez Hidalgo, and Yamakami, 2011). It consolidates four verified, non-synthetic mobile text sources:
1. **NUS SMS Corpus:** 3,375 legitimate messages collected from students at the National University of Singapore.
2. **Grumbletext UK:** 425 spam messages submitted to a public UK consumer grievance portal.
3. **Caroline Tag's PhD Thesis:** 450 legitimate SMS messages gathered for linguistic research.
4. **SMS Spam Corpus v.0.1 Big:** 1,002 legitimate and 322 spam messages from prior academic baselines.

### 3.2 Dataset Statistics & Class Imbalance
* **Total Records:** 5,574 messages.
* **Legitimate Messages (Ham):** 4,827 records (**86.60%**).
* **Spam Messages:** 747 records (**13.40%**).
* **Class Ratio:** Approximately $6.46 : 1$.
* **Missing/Corrupt Values:** 0 null records.

### 3.3 Text Length Characteristics
Empirical analysis reveals distinct structural divergence between classes:
* **Ham Messages:** Character length mean = 71.0 characters (median = 52 characters, word count mean = 14.3 words). Ham texts are concise, informal, and conversational.
* **Spam Messages:** Character length mean = 138.8 characters (median = 149 characters, word count mean = 23.8 words). Spam messages cluster tightly against the 160-character carrier limit to maximize information payload.

---

## 4. METHODOLOGY & SYSTEM ARCHITECTURE

The classification framework follows a standardized five-stage architecture:

```
[ Raw SMS/Email Corpus ]
           │
           ▼
[ NLP Preprocessing Pipeline ]
  ├─ Case Normalization (lower)
  ├─ URL & Email Regex Stripping
  ├─ Punctuation & Digit Removal
  ├─ Tokenization (word-level)
  ├─ Stop-word Removal (NLTK English)
  └─ Morphological Stemming (Porter)
           │
           ▼
[ Stratified Train-Test Split (80/20, Seed=42) ]
  ├─ Training Set: 4,448 instances (3,851 Ham, 597 Spam)
  └─ Testing Set:  1,113 instances (963 Ham, 150 Spam)
           │
           ▼
[ TF-IDF Feature Representation ]
  ├─ Sublinear Term Frequency: 1 + log(tf)
  ├─ Smooth IDF: log((1+N)/(1+df)) + 1
  ├─ N-gram Spectrum: Unigrams + Bigrams (1, 2)
  └─ L2 Normalization (Strictly fitted on Train split)
           │
           ▼
[ 4-Classifier Battle Arena ]
  ├─ Decision Tree (CART, Gini)
  ├─ K-Nearest Neighbours (Cosine, k=5)
  ├─ Linear Support Vector Machine (C=1.0)
  └─ Multinomial Naive Bayes (Alpha=1.0)
           │
           ▼
[ Evaluation & What-If Diagnostics ]
  ├─ Accuracy, Precision, Recall, F1, ROC-AUC
  ├─ Confusion Matrices (TN, FP, FN, TP)
  ├─ Computational Latency Profiling
  └─ What-If Experiments (k, max_features, n-grams)
```

---

## 5. TEXT PREPROCESSING PIPELINE
Raw messages are passed through six sequential functional stages:
1. **Lowercasing:** Converts text to lowercase to unify token identities.
2. **Regex URL and Email Cleansing:** Matches patterns `https?://\S+|www\.\S+` and email addresses, replacing them with whitespace.
3. **Punctuation and Numeric Normalization:** Uses ASCII translation tables to strip punctuation and non-alphabetic noise.
4. **Tokenization:** Converts character strings into lexical word units.
5. **Stop-word Removal:** Discards 179 high-frequency non-discriminative English syntax tokens (`is`, `the`, `at`, `on`, `which`).
6. **Porter Stemming:** Algorithmic suffix stripping to reduce inflectional forms to roots (`winner`, `winning` -> `win`).

**Demonstration Transformation:**
* *Input:* `"WINNER!! As a valued customer, claim your FREE cash prize of £500 now at http://spam.xyz or call 0800!"`
* *Output:* `"winner valu custom claim free cash prize call"`

---

## 6. TF-IDF FEATURE ENGINEERING & DATA LEAKAGE PREVENTION

### 6.1 Mathematical Principles
1. **Term Frequency (TF):**
   $$TF(t, d) = 1 + \log(f_{t, d}) \quad \forall f_{t, d} > 0$$
2. **Inverse Document Frequency (IDF):**
   $$IDF(t, D) = \log\left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$
3. **Composite TF-IDF & $L_2$ Normalization:**
   $$TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)$$
   $$\vec{v}_{\text{norm}} = \frac{\vec{v}}{\|\vec{v}\|_2} = \frac{\vec{v}}{\sqrt{\sum_{i=1}^m v_i^2}}$$

### 6.2 Strict Data Leakage Prevention
Data leakage occurs when information from outside the training partition enters feature extraction. To guarantee complete academic rigor:
* The `TfidfVectorizer` is **fitted exclusively on the 4,448 training instances (`X_train`)**.
* The 1,113 test samples (`X_test`) are transformed using the fixed vocabulary and precomputed IDF weights.
* Zero test document statistics influence vocabulary selection or term weighting.

---

## 7. THE FOUR CLASSIFIERS: THEORETICAL FOUNDATIONS

### 7.1 Decision Tree (CART)
Recursively partitions feature space using orthogonal axis-aligned splits. The decision criterion minimizes the Gini Impurity:
$$I_G(p) = 1 - \sum_{i=1}^C p_i^2$$
*Pros:* Highly interpretable rule paths.  
*Cons:* Extremely susceptible to overfitting in high-dimensional, sparse text vectors.

### 7.2 K-Nearest Neighbours (KNN)
An instance-based lazy learner that stores training vectors in memory. Given a test vector $\vec{x}$, it retrieves the $k$ nearest training vectors using Cosine Distance:
$$\text{Cosine Distance}(\vec{u}, \vec{v}) = 1 - \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$$
Classification is determined by majority voting.  
*Pros:* Non-parametric, zero training time.  
*Cons:* Inference scales with $O(N \cdot |V|)$; highly sensitive to majority class dominance in imbalanced data.

### 7.3 Linear Support Vector Machine (SVM)
Constructs a maximum-margin separating hyperplane in the 3,000-dimensional TF-IDF space by solving:
$$\min_{\vec{w}, b, \xi} \frac{1}{2} \|\vec{w}\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(\vec{w}^T \vec{x}_i + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
*Pros:* Robust in high-dimensional sparse spaces; $L_2$ weight regularization prevents overfitting; exceptional generalization.  
*Cons:* Quadratic training optimization complexity.

### 7.4 Multinomial Naive Bayes (MNB)
A probabilistic generative classifier that estimates the posterior class probability using Bayes' Theorem:
$$P(c \mid \vec{d}) \propto P(c) \prod_{i=1}^m P(t_i \mid c)^{f_i}$$
with Laplace smoothing ($\alpha = 1.0$):
$$P(t \mid c) = \frac{\text{count}(t, c) + \alpha}{\sum_{t'} \text{count}(t', c) + \alpha |V|}$$
*Pros:* Linear time complexity, sub-millisecond inference, excellent calibrated precision.  
*Cons:* Independence assumption ignores word co-occurrence correlations.

---

## 8. EXPERIMENTAL RESULTS & COMPARATIVE ANALYSIS

All models were evaluated on the identical holdout test split of **1,113 messages (963 Ham, 150 Spam)**.

### 8.1 Empirical Performance Table

| Classifier | Accuracy | Spam Precision | Spam Recall | Spam F1-Score | Macro F1 | ROC-AUC | Train Time (s) | Latency / Sample (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | **98.20%** | **98.51%** | **88.00%** | **0.9296** | **0.9596** | **0.9880** | 1.4489s | 0.046 ms |
| **Multinomial NB** | **97.39%** | **99.19%** | **81.33%** | **0.8938** | **0.9395** | **0.9854** | **0.0017s** | **0.0002 ms** |
| **KNN (k=5)** | **97.12%** | **98.36%** | **80.00%** | **0.8824** | **0.9331** | **0.9634** | 0.0009s | 0.047 ms |
| **Decision Tree** | **96.68%** | **88.97%** | **86.00%** | **0.8746** | **0.9278** | **0.9217** | 0.1176s | 0.001 ms |

---

## 9. CONFUSION MATRIX & ERROR PROFILE ANALYSIS

| Classifier | True Negatives (TN) | False Positives (FP) | False Negatives (FN) | True Positives (TP) | FP Rate (Cost) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear SVM** | 961 | 2 | 18 | 132 | **0.21% (Very Low)** |
| **Multinomial NB** | **962** | **1** | 28 | 122 | **0.10% (Lowest)** |
| **KNN (k=5)** | 961 | 2 | 30 | 120 | 0.21% (Very Low) |
| **Decision Tree** | 947 | **16** | 21 | 129 | **1.66% (High)** |

### 9.1 Asymmetric Cost Discussion
In spam filtering, **False Positives represent a catastrophic failure mode**. If an OTP verification code, bank fraud alert, or job notification is misclassified as spam, the user suffers direct harm.
* **Multinomial Naive Bayes** achieved the safest false positive rate, misclassifying only 1 legitimate message out of 963.
* **Linear SVM** committed only 2 False Positives while successfully intercepting 132 out of 150 spam messages (88% recall).
* **Decision Tree** failed severely with 16 False Positives—8 times worse than SVM—due to brittle tree splits on isolated words.

---

## 10. WHAT-IF SENSITIVITY EXPERIMENTS

### 10.1 What-If A: KNN Sensitivity to Neighborhood Parameter $k$
We evaluated KNN over $k \in [1, 3, 5, 7, 9, 11, 15, 21, 31]$:
* At $k=1$, Spam Precision is 96.06% and Recall is 81.33% (F1 = 0.8809).
* At $k=5$, Accuracy peaks at 97.12% with Spam Precision 98.36%.
* As $k$ increases to 21 and 31, **Spam Recall collapses down to 68.00%** (F1 = 0.8031).
* **Scientific Explanation:** In an 86.6% majority Ham distribution, expanding the voting sphere introduces overwhelming numbers of Ham samples. Minority spam points are out-voted, causing severe recall degradation.

### 10.2 What-If B: TF-IDF Vocabulary Capacity (`max_features`)
We evaluated feature limits from 200 to Unlimited (30,508 features):
* At 200 features: Both models underfit (SVM F1: 0.8990, NB F1: 0.8736).
* Between 1,000–2,000 features: Both models achieve optimal balance (SVM F1 peaks at **0.9412**, NB F1 at **0.9214**).
* At Unlimited features (30,508): Naive Bayes recall collapses to **70.00%** because ultra-rare tail n-grams distort prior/likelihood products, while Linear SVM remains stable (F1: 0.9315) due to $L_2$ weight penalty.

### 10.3 What-If C: N-Gram Range Dynamics
* **Bigram Only (2,2):** Recall drops to 72.67% (F1: 0.8289) due to extreme text sparsity.
* **Unigram + Bigram (1,2):** Achieves peak precision (98.51%) by capturing critical phishing collocations (`free cash`, `claim prize`, `call now`).

---

## 11. STREAMLIT APPLICATION & INTERACTIVE DEPLOYMENT
A modern Streamlit dashboard (`app.py`) was constructed featuring:
1. **Executive Dashboard:** Student details, course info, system KPIs.
2. **Exploratory Data Viewer:** Real-time distribution charts and message length histograms.
3. **Preprocessing Pipeline Simulator:** Step-by-step intermediate token inspect visualizer.
4. **TF-IDF Lab:** Formula breakdown and top weighted n-grams.
5. **The 4-Classifier Battle Arena:** Comparison tables, radar metrics, and confusion matrix visualizers.
6. **What-If Sandbox:** Interactive charts for $k$-value and feature capacity exploration.
7. **Live Prediction Playground:** Real-time classification with probability gauges, latency timer, and multi-model consensus voting.
8. **Security Notice:** Explicit disclaimer that statistical heuristics do not replace defense-in-depth cybersecurity controls.

---

## 12. CONCLUSION, LIMITATIONS & FUTURE ENHANCEMENTS

### 12.1 Conclusion & Technical Defence
* **Defensible Champion:** Linear Support Vector Machine is the superior architecture for short-text spam classification, balancing high Accuracy (98.20%), top Recall (88.00%), and peak F1-Score (0.9296) with low False Positive risk.
* **Efficiency Runner-Up:** Multinomial Naive Bayes is optimal for resource-constrained edge gateways requiring sub-microsecond latency and 99.19% precision.

### 12.2 Limitations & Threats to Validity
1. **Adversarial Evasions:** Statistical n-gram models can be fooled by zero-width unicode characters, character homoglyphs (e.g., Cyrillic 'а' replacing Latin 'a'), or image-embedded spam.
2. **Short Context:** Lacks conversation thread history or sender reputation metrics.

### 12.3 Future Enhancements
* Incorporating contextual transformer architectures (DistilBERT or RoBERTa).
* Deploying character-level convolutional embeddings to resist adversarial homoglyph perturbations.
* Integrating semi-supervised active learning loops for continuous adaptation to zero-day spam campaigns.

---

## 13. REFERENCES
1. Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: New collection and results. *Information Sciences*, 181(11), 2422-2438.
2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
3. Joachims, T. (1998). Text categorization with Support Vector Machines: Learning with many relevant features. *European Conference on Machine Learning (ECML)*, 137-142.
4. McCallum, A., & Nigam, K. (1998). A comparison of event models for Naive Bayes text classification. *AAAI-98 Workshop on Learning for Text Categorization*, 752, 41-48.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research (JMLR)*, 12, 2825-2830.

