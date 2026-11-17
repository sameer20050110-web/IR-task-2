"""
model_training.py
-----------------
Model Training and Serialization for SMS Spam Classification.
Trains and compares 4 Machine Learning Classifiers:
1. Decision Tree
2. K-Nearest Neighbours (KNN)
3. Support Vector Machine (SVM)
4. Multinomial Naive Bayes

Author: Mohammed Sameer P (RA2411026050199)
Course: 21CSE377T - Information Retrieval
Assignment: CLA-2 Task 2
"""

import os
import time
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Any

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

from preprocessing import preprocess_text

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "sms_spam_collection.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_and_preprocess_dataset(data_path: str = DATA_PATH) -> pd.DataFrame:
    """Load dataset, handle nulls, clean labels, and apply text preprocessing."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run download_dataset.py first.")
    
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['label', 'message']).copy()
    
    # Standardize labels: ham -> 0, spam -> 1
    df['label_raw'] = df['label'].astype(str).str.strip().str.lower()
    df['target'] = df['label_raw'].map({'ham': 0, 'spam': 1})
    df = df.dropna(subset=['target']).copy()
    df['target'] = df['target'].astype(int)
    
    print(f"[*] Preprocessing {len(df)} messages...")
    df['cleaned_message'] = df['message'].apply(lambda x: preprocess_text(str(x), use_stemming=True))
    
    # Filter out any rare rows where message became completely empty
    df = df[df['cleaned_message'].str.strip() != ""].reset_index(drop=True)
    print(f"[+] Remaining valid records after preprocessing: {len(df)}")
    return df


def get_classifiers() -> Dict[str, Any]:
    """Define the four required classifiers with deterministic hyperparameters."""
    return {
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            metric="cosine",
            weights="uniform"
        ),
        "SVM": SVC(
            kernel="linear",
            C=1.0,
            probability=True,
            random_state=RANDOM_STATE
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=1.0
        )
    }


def train_and_evaluate_all():
    """
    Executes the training and evaluation workflow:
    1. Stratified split (80/20) with seed 42
    2. Fit TF-IDF Vectorizer strictly on train split
    3. Fit each classifier pipeline
    4. Compute actual Accuracy, Precision, Recall, F1, Latency, Confusion Matrix
    5. Save models, metrics CSV, and detailed reports
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)
    
    df = load_and_preprocess_dataset()
    
    X = np.array(df['cleaned_message'].tolist(), dtype=object)
    y = np.array(df['target'].tolist(), dtype=int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    
    print(f"[*] Training samples: {len(X_train)} (Ham: {sum(y_train == 0)}, Spam: {sum(y_train == 1)})")
    print(f"[*] Testing samples:  {len(X_test)} (Ham: {sum(y_test == 0)}, Spam: {sum(y_test == 1)})")
    
    # Save train/test test data for evaluation scripts and what-if scripts
    np.save(os.path.join(RESULTS_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(RESULTS_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(RESULTS_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(RESULTS_DIR, "y_test.npy"), y_test)
    
    classifiers = get_classifiers()
    
    metrics_summary = []
    classification_reports = {}
    confusion_matrices = {}
    
    # We will fit a shared TF-IDF vectorizer representation strictly on training data
    tfidf = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    
    t0_vec = time.time()
    X_train_vec = tfidf.fit_transform(X_train)
    vec_fit_time = time.time() - t0_vec
    X_test_vec = tfidf.transform(X_test)
    
    # Save standalone TF-IDF vectorizer
    joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
    print(f"[+] TF-IDF Vectorizer fit and saved. Vocabulary size: {len(tfidf.vocabulary_)} features.")
    
    for name, clf in classifiers.items():
        print(f"\n---> Training {name}...")
        
        # Train timing
        t0 = time.time()
        clf.fit(X_train_vec, y_train)
        train_time = time.time() - t0
        
        # Test inference timing
        t0_inf = time.time()
        y_pred = clf.predict(X_test_vec)
        test_time = time.time() - t0_inf
        
        # Probabilities for ROC-AUC
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test_vec)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        elif hasattr(clf, "decision_function"):
            y_proba = clf.decision_function(X_test_vec)
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            roc_auc = float('nan')
            
        acc = accuracy_score(y_test, y_pred)
        prec_spam = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec_spam = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1_spam = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        
        prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        clf_report = classification_report(
            y_test, y_pred,
            target_names=['Ham (0)', 'Spam (1)'],
            output_dict=True,
            zero_division=0
        )
        
        # Save pipeline containing both vectorizer and model for end-to-end inference
        pipeline = Pipeline([
            ('tfidf', tfidf),
            ('classifier', clf)
        ])
        model_filename = name.lower().replace(" ", "_").replace("-", "_") + ".joblib"
        joblib.dump(pipeline, os.path.join(MODELS_DIR, model_filename))
        
        metrics_summary.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Spam Precision": round(prec_spam, 4),
            "Spam Recall": round(rec_spam, 4),
            "Spam F1-Score": round(f1_spam, 4),
            "Macro Precision": round(prec_macro, 4),
            "Macro Recall": round(rec_macro, 4),
            "Macro F1-Score": round(f1_macro, 4),
            "ROC-AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else "N/A",
            "True Negatives (TN)": int(tn),
            "False Positives (FP)": int(fp),
            "False Negatives (FN)": int(fn),
            "True Positives (TP)": int(tp),
            "Training Time (s)": round(train_time, 4),
            "Inference Time (s)": round(test_time, 4),
            "Latency per Sample (ms)": round((test_time / len(X_test)) * 1000, 3)
        })
        
        classification_reports[name] = clf_report
        confusion_matrices[name] = {
            "TN": int(tn), "FP": int(fp),
            "FN": int(fn), "TP": int(tp),
            "matrix": cm.tolist()
        }
        
        print(f"    Accuracy:      {acc:.4f}")
        print(f"    Spam Precision:{prec_spam:.4f}")
        print(f"    Spam Recall:   {rec_spam:.4f}")
        print(f"    Spam F1-Score: {f1_spam:.4f}")
        print(f"    Confusion: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print(f"    Train Time: {train_time:.4f}s | Test Time: {test_time:.4f}s")

    # Save summary dataframe
    df_metrics = pd.DataFrame(metrics_summary)
    metrics_csv_path = os.path.join(RESULTS_DIR, "metrics_comparison.csv")
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"\n[+] Saved metrics summary to: {metrics_csv_path}")
    
    # Save JSON reports
    reports_path = os.path.join(RESULTS_DIR, "classification_reports.json")
    with open(reports_path, "w", encoding="utf-8") as f:
        json.dump(classification_reports, f, indent=4)
        
    cm_path = os.path.join(RESULTS_DIR, "confusion_matrices.json")
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump(confusion_matrices, f, indent=4)
        
    print(f"[+] Saved classification reports to: {reports_path}")
    print(f"[+] Saved confusion matrices to: {cm_path}")
    
    return df_metrics


if __name__ == "__main__":
    train_and_evaluate_all()
