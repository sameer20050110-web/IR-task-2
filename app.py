"""
app.py
------
Streamlit Interactive Dashboard for Email and SMS Spam Detection.
CLA-2 Task 2 - Text Classification Battle: Compare, Analyze and Defend.

Author: Mohammed Sameer P
Register Number: RA2411026050199
Institution: SRM Institute of Science and Technology, Tiruchirappalli
Course: 21CSE377T - Information Retrieval
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocessing import preprocess_step_by_step, preprocess_text
from prediction import SpamDetector, MODEL_FILES

# Page configuration
st.set_page_config(
    page_title="SMS Spam Detection | SRM IST",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished academic appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1a365d;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4a5568;
        margin-bottom: 1.5rem;
    }
    .student-badge {
        background: linear-gradient(135deg, #0d3b66, #001e3d);
        color: white;
        padding: 14px 20px;
        border-radius: 10px;
        font-size: 0.95rem;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e3a8a;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .spam-alert {
        background-color: #fef2f2;
        border-left: 5px solid #ef4444;
        padding: 15px 20px;
        border-radius: 6px;
        color: #991b1b;
        font-weight: bold;
    }
    .ham-alert {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        padding: 15px 20px;
        border-radius: 6px;
        color: #166534;
        font-weight: bold;
    }
    .disclaimer-box {
        background-color: #fffbeb;
        border: 1px solid #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #92400e;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Helper cached data loaders
@st.cache_data
def load_metrics_df():
    path = os.path.join(BASE_DIR, "results", "metrics_comparison.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_raw_dataset():
    path = os.path.join(BASE_DIR, "data", "sms_spam_collection.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['char_length'] = df['message'].astype(str).apply(len)
        df['word_count'] = df['message'].astype(str).apply(lambda s: len(s.split()))
        return df
    return None

@st.cache_data
def load_what_if_data():
    k_path = os.path.join(BASE_DIR, "results", "what_if_knn_k.csv")
    feat_path = os.path.join(BASE_DIR, "results", "what_if_tfidf_features.csv")
    ng_path = os.path.join(BASE_DIR, "results", "what_if_ngram.csv")
    return {
        "knn": pd.read_csv(k_path) if os.path.exists(k_path) else None,
        "features": pd.read_csv(feat_path) if os.path.exists(feat_path) else None,
        "ngram": pd.read_csv(ng_path) if os.path.exists(ng_path) else None
    }

@st.cache_data
def load_confusion_matrices():
    path = os.path.join(BASE_DIR, "results", "confusion_matrices.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/shield.png", width=70)
    st.markdown("### **SRM Institute of Science and Technology**")
    st.markdown("**Department of Computer Science & Engineering**<br>Tiruchirappalli Campus", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("#### 👤 **Student Information**")
    st.markdown("**Name:** Mohammed Sameer P")
    st.markdown("**Reg No:** `RA2411026050199`")
    st.markdown("**Course:** 21CSE377T – Information Retrieval")
    st.markdown("**Assignment:** CLA-2 Task 2")
    st.markdown("**Max Marks:** 30")
    st.markdown("---")
    
    navigation = st.radio(
        "Navigation Menu:",
        [
            "🏠 Project Overview",
            "📊 Dataset Exploration",
            "🧹 Preprocessing Pipeline",
            "🧮 TF-IDF Formulation",
            "⚔️ 4-Classifier Battle",
            "🔬 What-If Experiments",
            "🛡️ Live Spam Predictor",
            "📚 Viva Reference Guide"
        ]
    )
    
    st.markdown("---")
    st.caption("Information Retrieval | Academic Project | 2026")


# ==========================================
# PAGE 1: PROJECT OVERVIEW
# ==========================================
if navigation == "🏠 Project Overview":
    st.markdown('<div class="main-header">Email and SMS Spam Detection System</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">CLA-2 Task 2: Text Classification Battle — Compare, Analyze and Defend</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="student-badge">
        <strong>Student:</strong> Mohammed Sameer P &nbsp;|&nbsp; 
        <strong>Register No:</strong> RA2411026050199 &nbsp;|&nbsp; 
        <strong>Course:</strong> 21CSE377T – Information Retrieval &nbsp;|&nbsp; 
        <strong>Institution:</strong> SRM IST Tiruchirappalli
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card"><div class="metric-val">5,574</div><div class="metric-label">SMS Messages</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><div class="metric-val">4</div><div class="metric-label">ML Classifiers</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-val">98.20%</div><div class="metric-label">Top Test Accuracy</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-val">0.9296</div><div class="metric-label">Best Spam F1-Score</div></div>', unsafe_allow_html=True)
        
    st.markdown("### 🎯 Project Objectives & Problem Statement")
    st.write("""
    Unsolicited mobile communications and electronic spam represent significant vectors for financial fraud, identity theft, 
    and network congestion. The primary objective of this project is to develop, evaluate, and defend an end-to-end 
    Information Retrieval and Supervised Text Classification framework capable of categorizing text communications into:
    * **Ham (Legitimate)**: Non-malicious conversational or transactional messages.
    * **Spam (Unsolicited / Malicious)**: Promotional lures, cash lottery scams, phishing URLs, or automated broadcast spam.
    """)
    
    st.markdown("### 🏛️ System Architecture Workflow")
    st.markdown("""
    ```
    ┌───────────────────────┐      ┌─────────────────────────┐      ┌──────────────────────────┐
    │ Raw SMS / Email Text  │ ───> │ Multi-Stage NLP Filter  │ ───> │ TF-IDF Feature Extractor │
    │ (UCI Benchmark Corpus)│      │ Lower, Strip, Token, Stem│      │ (Train-fit only; no leak)│
    └───────────────────────┘      └─────────────────────────┘      └──────────────────────────┘
                                                                                 │
                                   ┌─────────────────────────────────────────────┘
                                   ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────┐
    │                                THE 4-CLASSIFIER BATTLE ARENA                             │
    │  1. Decision Tree (Gini)               │  2. K-Nearest Neighbours (Cosine metric, k=5)   │
    │  3. Support Vector Machine (Linear C=1)│  4. Multinomial Naive Bayes (Laplace alpha=1.0) │
    └──────────────────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
    ┌──────────────────────────────────────────────────────────────────────────────────────────┐
    │                             RIGOROUS PERFORMANCE DEFENCE                                │
    │   • Stratified Test Set (1,113 msgs)   • Confusion Matrix (TN, FP, FN, TP)               │
    │   • Precision-Recall Trade-offs        • What-If Sensitivity Experiments (k, Vocab, Ngram)│
    └──────────────────────────────────────────────────────────────────────────────────────────┘
    ```
    """)
    
    st.markdown("### ⚖️ Academic Integrity & Dataset Clarification")
    st.info("""
    **Dataset Scope:** While the project title encompasses *Email and SMS Spam Detection*, the empirical investigation 
    is rigorously conducted on the gold-standard **UCI SMS Spam Collection**. SMS text represents an exceptionally 
    challenging benchmark for Information Retrieval due to extreme sparsity, short lengths (average ~15 words), 
    colloquial abbreviations ('u', 'ur', 'txt', 'free'), and absence of rich email metadata (such as DKIM, SPF, or MIME headers). 
    Every metric reported in this project is calculated from genuine model evaluations on real data. No metrics are fabricated.
    """)


# ==========================================
# PAGE 2: DATASET EXPLORATION
# ==========================================
elif navigation == "📊 Dataset Exploration":
    st.markdown('<div class="main-header">Dataset Exploration & Class Distribution</div>', unsafe_allow_html=True)
    df = load_raw_dataset()
    
    if df is not None:
        total = len(df)
        counts = df['label'].value_counts()
        ham_count = counts.get('ham', 0)
        spam_count = counts.get('spam', 0)
        ham_pct = (ham_count / total) * 100
        spam_pct = (spam_count / total) * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Messages", f"{total:,}")
        with col2:
            st.metric("Legitimate (Ham)", f"{ham_count:,} ({ham_pct:.1f}%)")
        with col3:
            st.metric("Spam Messages", f"{spam_count:,} ({spam_pct:.1f}%)")
            
        st.markdown("### 📈 Class Imbalance Analysis")
        st.write("""
        The dataset displays a pronounced **class imbalance ratio of ~6.5 : 1** (86.6% Ham vs 13.4% Spam). 
        This is a critical property in Information Retrieval:
        * A trivial dummy classifier predicting 'Ham' on 100% of messages would attain an impressive **86.6% accuracy** 
          while delivering **0.0% spam recall**, leaving users completely vulnerable.
        * Therefore, **Precision, Recall, and F1-score on the Spam class** are the decisive evaluation metrics.
        """)
        
        tab1, tab2 = st.tabs(["📊 Distribution Charts", "🔍 Raw Data Explorer"])
        
        with tab1:
            col_a, col_b = st.columns(2)
            with col_a:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.bar(['HAM', 'SPAM'], [ham_count, spam_count], color=['#2b5c8f', '#d9534f'], edgecolor='black', width=0.5)
                ax.set_ylabel("Message Count")
                ax.set_title("Class Frequency Distribution")
                for p in ax.patches:
                    ax.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 3), textcoords='offset points')
                st.pyplot(fig)
                
            with col_b:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.pie([ham_count, spam_count], labels=[f"Ham ({ham_pct:.1f}%)", f"Spam ({spam_pct:.1f}%)"],
                       autopct='%1.1f%%', colors=['#2b5c8f', '#d9534f'], explode=(0, 0.1), startangle=140,
                       wedgeprops={'edgecolor': 'black'})
                ax.set_title("Class Imbalance Ratio")
                st.pyplot(fig)
                
            st.markdown("#### 📏 Message Length Distribution (Ham vs Spam)")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(data=df, x='char_length', hue='label', bins=50, kde=True,
                         palette={'ham': '#2b5c8f', 'spam': '#d9534f'}, ax=ax, element='step')
            ax.set_xlim(0, 300)
            ax.set_title("Character Length: Spam messages are significantly longer and concentrated near 160 characters (SMS maximum limit)")
            st.pyplot(fig)
            
        with tab2:
            st.dataframe(df[['label', 'char_length', 'word_count', 'message']].head(50), use_container_width=True)
    else:
        st.error("Dataset not found. Please run data/download_dataset.py.")


# ==========================================
# PAGE 3: PREPROCESSING PIPELINE
# ==========================================
elif navigation == "🧹 Preprocessing Pipeline":
    st.markdown('<div class="main-header">Text Preprocessing & Normalization</div>', unsafe_allow_html=True)
    st.write("""
    Text preprocessing transforms unstructured, noisy, colloquial messages into clean, tokenized, 
    and morphologically normalized units suitable for vector space indexing.
    """)
    
    st.markdown("""
    ### ⚙️ Preprocessing Stages
    1. **Case Normalization:** Converts characters to lowercase to unify token identities (e.g., `FREE`, `Free`, `free` -> `free`).
    2. **URL & Email Stripping:** Removes hyperlinks and email patterns which introduce high feature sparsity.
    3. **Punctuation & Symbol Removal:** Strips exclamation points, currency symbols, and delimiters using ASCII translation tables.
    4. **Tokenization:** Splits text sequences into lexical word units.
    5. **Stop-word Removal:** Filters out high-frequency non-informative syntactic tokens (`is`, `the`, `at`, `which`).
    6. **Stemming (Porter Stemmer):** Reduces inflectional morphological variants to root stems (`winner`, `winning` -> `winner`/`win`).
    """)
    
    st.markdown("### 🧪 Live Preprocessing Demonstration")
    sample_text = st.text_area(
        "Enter or paste any sample message to observe step-by-step transformations:",
        value="URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt CLAIM to 81010 now! http://spam.xyz"
    )
    
    if sample_text:
        steps = preprocess_step_by_step(sample_text)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Original Message:**")
            st.info(steps['original'])
            st.markdown(f"*Original Character Length:* `{steps['original_len']}`")
        with c2:
            st.markdown("**Final Cleaned Output (Fed into Vectorizer):**")
            st.success(steps['final_output'])
            st.markdown(f"*Cleaned Character Length:* `{steps['cleaned_len']}`")
            
        st.markdown("#### 🔍 Step-by-Step Intermediate Representations")
        step_df = pd.DataFrame([
            {"Stage": "1. Lowercase", "Result": steps['step1_lowercased']},
            {"Stage": "2. Strip URLs/Emails", "Result": steps['step2_no_urls_emails']},
            {"Stage": "3. Strip Punctuation/Digits", "Result": steps['step3_no_punct']},
            {"Stage": "4. Tokenized Words", "Result": ", ".join(steps['step4_tokens'])},
            {"Stage": "5. Removed Stopwords", "Result": ", ".join(steps['step5_no_stopwords'])},
            {"Stage": "6. Porter Stemming", "Result": ", ".join(steps['step6_stemmed'])},
        ])
        st.table(step_df)
        
        st.markdown(f"**Identified Stopwords Filtered Out:** `{', '.join(steps['removed_tokens']) if steps['removed_tokens'] else 'None'}`")


# ==========================================
# PAGE 4: TF-IDF FORMULATION
# ==========================================
elif navigation == "🧮 TF-IDF Formulation":
    st.markdown('<div class="main-header">TF-IDF Feature Representation</div>', unsafe_allow_html=True)
    st.write("""
    Machine learning algorithms require numerical input. The **Term Frequency - Inverse Document Frequency (TF-IDF)**
    weighting scheme reflects how important a word is to a message within the entire training corpus.
    """)
    
    st.markdown("### 📐 Mathematical Formulation")
    st.markdown(r"""
    #### 1. Term Frequency ($TF$)
    Measures the frequency of term $t$ in message $d$:
    $$TF(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}} \quad \text{or with sublinear scaling:} \quad 1 + \log(f_{t, d})$$
    
    #### 2. Inverse Document Frequency ($IDF$)
    Down-weights terms that appear ubiquitously across all documents (e.g., generic words):
    $$IDF(t, D) = \log\left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$
    *(where $|D|$ is the total number of training documents, using Scikit-Learn smooth IDF).*
    
    #### 3. Composite TF-IDF Weight
    $$TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)$$
    Followed by $L_2$ Euclidean normalization so that longer messages do not artificially dominate:
    $$v_{\text{norm}} = \frac{v}{\|v\|_2} = \frac{v}{\sqrt{\sum_i v_i^2}}$$
    """)
    
    st.markdown("### 🛡️ Prevention of Data Leakage")
    st.warning("""
    **Zero Data Leakage Guarantee:**
    In this system, the `TfidfVectorizer` is **fit exclusively on the training partition** (`X_train`). 
    The testing partition (`X_test`) is transformed using the pre-computed vocabulary and IDF parameters derived strictly from training data. 
    This ensures test metrics reflect genuine out-of-sample generalization.
    """)
    
    st.markdown("### 🏆 Top Discriminative Keywords by TF-IDF Weight")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🚨 Common High-Weight Spam Tokens")
        spam_kw = pd.DataFrame({
            "Token / N-gram": ["free", "claim", "prize", "urgent", "txt", "call now", "cash", "winner", "award", "service"],
            "Category": ["Lure / Incentive", "Action", "Lure", "Urgency", "Protocol", "Call to Action", "Lure", "Lure", "Lure", "Service"],
            "Discrimination Power": ["Very High", "Very High", "High", "High", "High", "Very High", "High", "High", "Medium", "Medium"]
        })
        st.table(spam_kw)
        
    with col2:
        st.markdown("#### 💬 Common High-Weight Ham Tokens")
        ham_kw = pd.DataFrame({
            "Token / N-gram": ["go", "come", "got", "ok", "later", "love", "good", "meet", "home", "time"],
            "Category": ["Conversational", "Conversational", "Informative", "Acknowledgment", "Temporal", "Personal", "Affective", "Social", "Location", "Temporal"],
            "Discrimination Power": ["High", "High", "Medium", "High", "Medium", "High", "Medium", "High", "Medium", "Medium"]
        })
        st.table(ham_kw)


# ==========================================
# PAGE 5: 4-CLASSIFIER BATTLE
# ==========================================
elif navigation == "⚔️ 4-Classifier Battle":
    st.markdown('<div class="main-header">The 4-Classifier Battle Arena</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Empirical Comparison of Decision Tree, KNN, SVM, and Naive Bayes on 1,113 Test Messages</div>', unsafe_allow_html=True)
    
    df_metrics = load_metrics_df()
    cms = load_confusion_matrices()
    
    if df_metrics is not None:
        st.markdown("### 📊 Overall Performance Comparison Table")
        display_df = df_metrics[[
            'Model', 'Accuracy', 'Spam Precision', 'Spam Recall', 'Spam F1-Score',
            'Macro F1-Score', 'ROC-AUC', 'Training Time (s)', 'Latency per Sample (ms)'
        ]].copy()
        st.dataframe(display_df.style.highlight_max(subset=['Accuracy', 'Spam Precision', 'Spam Recall', 'Spam F1-Score'], color='#dcfce7'), use_container_width=True)
        
        st.markdown("### 🏆 Key Empirical Findings & Battle Analysis")
        st.markdown("""
        1. **Linear Support Vector Machine (SVM) — The Overall Champion:**
           * Achieves the highest overall **Accuracy (98.20%)**, the highest **Spam Recall (88.00%)**, and the highest **Spam F1-Score (0.9296)**.
           * **Why it excels:** In high-dimensional, sparse TF-IDF spaces (3,000 features), text classes are almost linearly separable. 
             SVM finds the maximum-margin hyperplane, providing superior generalization without overfitting.
        2. **Multinomial Naive Bayes — The Precision & Speed Champion:**
           * Achieves the highest **Spam Precision (99.19%)** with only **1 False Positive** across 963 legitimate test messages.
           * Delivers near-zero inference latency (**0.0002 seconds**, ~0.000 ms per sample).
           * **Why it excels:** The multinomial generative model naturally models word frequency counts with Laplace smoothing.
        3. **K-Nearest Neighbours (KNN, k=5):**
           * Attains solid **97.12% accuracy**, but suffers from higher inference latency (**52.1 ms**) because it must calculate cosine distances against all 4,448 training instances.
           * Suffers in Spam Recall (80.00%) due to majority-class neighborhood dominance.
        4. **Decision Tree:**
           * Lowest overall performance (Accuracy: 96.68%, Spam F1: 0.8746) and suffered the highest number of **False Positives (16 FP)**.
           * **Why it struggles:** Orthogonal axis-aligned decision boundaries in high-dimensional sparse spaces create fragmented, overfit decision leaves based on idiosyncratic word occurrences.
        """)
        
        st.markdown("### 🎯 Confusion Matrices (Interactive Inspection)")
        selected_model = st.selectbox("Select Classifier to view detailed Confusion Matrix & Error Profile:", df_metrics['Model'].tolist())
        
        if cms and selected_model in cms:
            data = cms[selected_model]
            cm = np.array(data['matrix'])
            
            c1, c2 = st.columns([1, 1.2])
            with c1:
                fig, ax = plt.subplots(figsize=(5, 4))
                annot = np.array([
                    [f"TN\n{cm[0,0]}", f"FP\n{cm[0,1]}"],
                    [f"FN\n{cm[1,0]}", f"TP\n{cm[1,1]}"]
                ])
                sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', ax=ax, cbar=False,
                            xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                            annot_kws={"fontsize": 13, "fontweight": "bold"})
                ax.set_xlabel("Predicted Label", fontweight='bold')
                ax.set_ylabel("True Label", fontweight='bold')
                ax.set_title(f"Confusion Matrix: {selected_model}")
                st.pyplot(fig)
                
            with c2:
                st.markdown(f"#### Error Diagnostics: **{selected_model}**")
                st.write(f"* **True Negatives (TN):** `{data['TN']}` legitimate messages correctly allowed.")
                st.write(f"* **False Positives (FP):** `{data['FP']}` legitimate messages wrongly flagged as spam *(High Cost)*.")
                st.write(f"* **False Negatives (FN):** `{data['FN']}` spam messages that slipped past the filter into inbox.")
                st.write(f"* **True Positives (TP):** `{data['TP']}` spam messages successfully blocked.")
                
                if data['FP'] > 5:
                    st.warning(f"⚠️ High False Positive count ({data['FP']})! This model risks sending important user communications to the junk folder.")
                else:
                    st.success(f"✅ Exceptional False Positive control ({data['FP']} FP)! Users rarely miss genuine messages.")
    else:
        st.error("Model metrics not loaded. Please train models first.")


# ==========================================
# PAGE 6: WHAT-IF EXPERIMENTS
# ==========================================
elif navigation == "🔬 What-If Experiments":
    st.markdown('<div class="main-header">What-If Sensitivity Experiments</div>', unsafe_allow_html=True)
    st.write("""
    Scientific validation requires probing model stability and behavior under controlled parameter variations. 
    Three empirical experiments were conducted and evaluated on the identical test split.
    """)
    
    what_if = load_what_if_data()
    
    exp_choice = st.radio(
        "Choose What-If Experiment:",
        [
            "Experiment A: KNN k-Parameter Sensitivity (Neighborhood Size)",
            "Experiment B: TF-IDF Vocabulary Capacity (max_features)",
            "Experiment C: N-Gram Spectrum (Unigram vs Bigram vs Trigram)"
        ]
    )
    
    if exp_choice.startswith("Experiment A") and what_if['knn'] is not None:
        st.markdown("### 🔬 What-If A: KNN Performance across k in [1, 3, 5, 7, 9, 11, 15, 21, 31]")
        df_k = what_if['knn']
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(df_k['k_neighbors'], df_k['Accuracy'], marker='o', label='Accuracy', color='#2b5c8f')
            ax.plot(df_k['k_neighbors'], df_k['Spam Precision'], marker='s', label='Spam Precision', color='#27ae60')
            ax.plot(df_k['k_neighbors'], df_k['Spam Recall'], marker='^', label='Spam Recall', color='#d9534f')
            ax.plot(df_k['k_neighbors'], df_k['Spam F1-Score'], marker='d', label='Spam F1', linestyle='--', color='#8e44ad')
            ax.set_xlabel("k (Nearest Neighbours)")
            ax.set_ylabel("Metric Score")
            ax.set_title("KNN Sensitivity to k: Notice the Collapse in Spam Recall at High k")
            ax.set_xticks(df_k['k_neighbors'])
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)
            
        with c2:
            st.dataframe(df_k[['k_neighbors', 'Accuracy', 'Spam Precision', 'Spam Recall', 'Spam F1-Score']], use_container_width=True)
            
        st.markdown("""
        **Scientific Explanation:**
        * At small $k$ ($k=1, 3$), KNN achieves high recall (81-83%) but is vulnerable to local noise and lower precision.
        * As $k$ increases to 21 and 31, **Spam Recall plummets precipitously from 83.3% down to 68.0%**.
        * **Root Cause:** Because the dataset is 86.6% Ham, expanding the spherical neighborhood brings an overwhelming number of 
          majority Ham instances into the voting pool, effectively smothering minority Spam points from obtaining majority vote.
        """)
        
    elif exp_choice.startswith("Experiment B") and what_if['features'] is not None:
        st.markdown("### 🔬 What-If B: Effect of TF-IDF max_features on Linear SVM vs Naive Bayes")
        df_feat = what_if['features']
        
        c1, c2 = st.columns([1.5, 1])
        with c1:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(df_feat['Actual Vocab Size'], df_feat['SVM Spam F1'], marker='o', label='SVM Spam F1', color='#2980b9')
            ax.plot(df_feat['Actual Vocab Size'], df_feat['NB Spam F1'], marker='s', label='NB Spam F1', color='#e74c3c')
            ax.plot(df_feat['Actual Vocab Size'], df_feat['SVM Accuracy'], marker='^', linestyle=':', label='SVM Accuracy', color='#34495e')
            ax.plot(df_feat['Actual Vocab Size'], df_feat['NB Accuracy'], marker='v', linestyle=':', label='NB Accuracy', color='#e67e22')
            ax.set_xlabel("Vocabulary Dimension")
            ax.set_ylabel("Score")
            ax.set_title("TF-IDF Feature Capacity Trade-off")
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)
            
        with c2:
            st.dataframe(df_feat[['Max Features Setting', 'Actual Vocab Size', 'SVM Spam F1', 'NB Spam F1']], use_container_width=True)
            
        st.markdown("""
        **Scientific Explanation:**
        * At **200 features**, both models underfit (F1 ~0.87-0.89) due to insufficient expressive capacity.
        * At **1,000 to 2,000 features**, both SVM and Naive Bayes reach their optimal peak performance.
        * At **Unlimited features (30,508 features)**, Naive Bayes recall collapses to **70.0%** because ultra-sparse tail n-grams with 
          frequency of 1 distort conditional probability estimations, whereas Linear SVM remains robust (F1: 0.9315) due to its L2-norm regularization.
        """)
        
    elif exp_choice.startswith("Experiment C") and what_if['ngram'] is not None:
        st.markdown("### 🔬 What-If C: Impact of N-Gram Ranges on Feature Discrimination")
        df_ng = what_if['ngram']
        st.dataframe(df_ng, use_container_width=True)
        
        st.markdown("""
        **Scientific Explanation:**
        * **Bigrams Only (2,2)** yields poor performance (Recall 72.67%, F1 0.8289). SMS messages are concise and informal, 
          so exact consecutive word pairs are too sparse.
        * **Unigrams + Bigrams (1,2)** provides optimal balance: unigrams preserve vocabulary coverage while bigrams capture 
          vital co-occurring phishing cues (e.g., `free cash`, `claim prize`, `urgent reply`).
        """)


# ==========================================
# PAGE 7: LIVE SPAM PREDICTOR
# ==========================================
elif navigation == "🛡️ Live Spam Predictor":
    st.markdown('<div class="main-header">Live SMS & Email Spam Predictor</div>', unsafe_allow_html=True)
    st.write("Test single messages through any trained classifier or execute a multi-model consensus battle.")
    
    # Presets
    preset = st.selectbox(
        "Load a preset benchmark test message or type custom text below:",
        [
            "Custom User Message...",
            "Preset 1 (Aggressive Spam): WINNER! You have won a £1000 cash prize or iPad. Call 09050000327 to claim your award now!",
            "Preset 2 (Legitimate Personal Ham): Hey, are we still meeting at the library at 5 PM for the information retrieval group study?",
            "Preset 3 (Urgent Phishing Scam): URGENT: Your mobile account has been suspended due to unpaid bill. Click http://verify-secure.me to avoid fee.",
            "Preset 4 (Transactional Ham): Your appointment with Dr. Henderson is confirmed for tomorrow at 10:30 AM. Reply CANCEL to reschedule."
        ]
    )
    
    if preset.startswith("Preset 1"):
        default_val = "WINNER! You have won a £1000 cash prize or iPad. Call 09050000327 to claim your award now!"
    elif preset.startswith("Preset 2"):
        default_val = "Hey, are we still meeting at the library at 5 PM for the information retrieval group study?"
    elif preset.startswith("Preset 3"):
        default_val = "URGENT: Your mobile account has been suspended due to unpaid bill. Click http://verify-secure.me to avoid fee."
    elif preset.startswith("Preset 4"):
        default_val = "Your appointment with Dr. Henderson is confirmed for tomorrow at 10:30 AM. Reply CANCEL to reschedule."
    else:
        default_val = "Claim your exclusive free bonus voucher worth $500 today by replying WIN!"
        
    user_input = st.text_area("Message Content:", value=default_val, height=110)
    
    col_ctrl1, col_ctrl2 = st.columns([1, 1])
    with col_ctrl1:
        chosen_classifier = st.selectbox("Select Active Classifier:", list(MODEL_FILES.keys()))
    with col_ctrl2:
        compare_all = st.checkbox("⚔️ Run All 4 Models Simultaneously (Consensus Mode)", value=True)
        
    if st.button("🚀 Analyze Message", type="primary"):
        if not user_input.strip():
            st.warning("Please enter a valid message string.")
        else:
            if not compare_all:
                detector = SpamDetector(chosen_classifier)
                res = detector.predict(user_input)
                
                if res['is_spam']:
                    st.markdown(f"""
                    <div class="spam-alert">
                        🚨 PREDICTION: SPAM MESSAGE DETECTED<br>
                        <span style="font-size: 0.9rem; font-weight: normal;">
                            Model: {res['model_used']} | Confidence: {res['confidence_percentage']}% | Latency: {res['latency_ms']} ms
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="ham-alert">
                        ✅ PREDICTION: HAM (LEGITIMATE MESSAGE)<br>
                        <span style="font-size: 0.9rem; font-weight: normal;">
                            Model: {res['model_used']} | Confidence: {res['confidence_percentage']}% | Latency: {res['latency_ms']} ms
                        </span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.markdown("#### 🔍 Processing Breakdown:")
                st.write(f"* **Normalized / Stemmed Text:** `{res['cleaned_text']}`")
                if res['probabilities']:
                    st.json(res['probabilities'])
            else:
                detector = SpamDetector("Support Vector Machine")
                multi_res = detector.predict_all_models(user_input)
                
                st.markdown(f"### Consensus Decision: **{multi_res['consensus']}** ({multi_res['spam_votes']} models flagged as Spam)")
                
                cols = st.columns(4)
                for idx, (m_name, m_res) in enumerate(multi_res['model_predictions'].items()):
                    with cols[idx]:
                        if m_res['is_spam']:
                            st.error(f"**{m_name}**\n\n🚨 **SPAM**\n\nConf: {m_res['confidence_percentage']}%\n\nTime: {m_res['latency_ms']}ms")
                        else:
                            st.success(f"**{m_name}**\n\n✅ **HAM**\n\nConf: {m_res['confidence_percentage']}%\n\nTime: {m_res['latency_ms']}ms")
                            
                st.markdown(f"**Preprocessed Feature String:** `{multi_res['model_predictions']['Support Vector Machine']['cleaned_text']}`")

    st.markdown("""
    <div class="disclaimer-box">
        <strong>⚠️ Security & Operational Disclaimer:</strong><br>
        Machine learning text classification is an Information Retrieval heuristic based on statistical n-gram patterns. 
        It provides actionable threat intelligence but does NOT constitute a complete cybersecurity guarantee against 
        sophisticated adversarial attacks, homoglyph substitutions, zero-day spear phishing, or evolving spam strategies.
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# PAGE 8: VIVA REFERENCE GUIDE
# ==========================================
elif navigation == "📚 Viva Reference Guide":
    st.markdown('<div class="main-header">Technical Defence & Viva Guide</div>', unsafe_allow_html=True)
    st.write("Comprehensive technical answers designed to defend CLA-2 Task 2 with maximum marks.")
    
    questions = [
        ("Q1: Why is this dataset called SMS Spam Collection instead of an email dataset?",
         "The dataset is composed of 5,574 authentic SMS messages gathered from university corpora (NUS SMS Corpus), the Grumbletext UK consumer site, and Caroline Tag's mobile thesis. SMS messages are strictly length-constrained (160 characters), contain mobile jargon ('u', 'ur', 'txt'), and lack email MIME/header protocols. It is a recognized IR benchmark for short-text classification."),
         
        ("Q2: Why is Accuracy deceptive in this project?",
         "Because of the 86.6% Ham to 13.4% Spam class imbalance. A zero-intelligence dummy model that blindly classifies every message as Ham achieves 86.6% accuracy, but has 0% Spam Recall. In spam filtering, Spam Precision and Spam Recall are far more informative."),
         
        ("Q3: Which is more harmful in spam detection: False Positives or False Negatives?",
         "False Positives (FP) are significantly more catastrophic. A False Positive occurs when a legitimate message (e.g., job offer, OTP password, medical confirmation) is wrongly sent to the spam folder and missed by the user. A False Negative merely results in nuisance spam appearing in the inbox."),
         
        ("Q4: How did you prevent data leakage during TF-IDF vectorization?",
         "By strictly fitting the TfidfVectorizer solely on the training partition (X_train). The inverse document frequencies (IDF) and vocabulary dictionary were calculated exclusively from training messages. The test set (X_test) was strictly transformed, guaranteeing no future test information leaked into training."),
         
        ("Q5: Why did Linear SVM outperform Decision Tree and KNN?",
         "Text vectorization produces a high-dimensional (3,000 features), highly sparse TF-IDF space. In high-dimensional spaces, text classes are nearly linearly separable. Linear SVM maximizes the geometric margin between support vectors and resists overfitting via L2 regularization, whereas Decision Trees overfit to noisy single words, and KNN suffers from majority class neighborhood dominance.")
    ]
    
    for q, a in questions:
        with st.expander(f"📌 {q}"):
            st.write(a)
            
    st.info("💡 A complete 25-Question Viva preparation guide is saved in docs/VIVA_QUESTIONS.md.")

