"""
what_if_analysis.py
-------------------
Systematic empirical what-if experiments investigating hyperparameter
sensitivities, feature representations, and class imbalance dynamics.

Experiments:
1. What-If A: Sensitivity of KNN to neighbor count k (k in [1, 3, 5, 7, 9, 11, 15, 21, 31])
2. What-If B: Effect of TF-IDF vocabulary capacity (max_features in [200, 500, 1000, 2000, 3000, 5000, None])
3. What-If C: Impact of n-gram ranges (Unigram (1,1), Unigram+Bigram (1,2), Bigram (2,2), Trigram (1,3))

Author: Mohammed Sameer P (RA2411026050199)
Course: 21CSE377T - Information Retrieval
Assignment: CLA-2 Task 2
"""

import os
import time
import json
import pandas as pd
import numpy as np

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
RANDOM_STATE = 42


def load_splits():
    """Load previously saved train and test splits."""
    x_train_path = os.path.join(RESULTS_DIR, "X_train.npy")
    x_test_path = os.path.join(RESULTS_DIR, "X_test.npy")
    y_train_path = os.path.join(RESULTS_DIR, "y_train.npy")
    y_test_path = os.path.join(RESULTS_DIR, "y_test.npy")

    if not all(os.path.exists(p) for p in [x_train_path, x_test_path, y_train_path, y_test_path]):
        raise FileNotFoundError("Splits not found. Please run model_training.py first.")

    X_train = np.load(x_train_path, allow_pickle=True)
    X_test = np.load(x_test_path, allow_pickle=True)
    y_train = np.load(y_train_path, allow_pickle=True)
    y_test = np.load(y_test_path, allow_pickle=True)

    return X_train, X_test, y_train, y_test


def experiment_knn_k_sensitivity(X_train, X_test, y_train, y_test):
    """
    Experiment A: Evaluate KNN across multiple values of k.
    Demonstrates how majority-class voting dominates as k increases in imbalanced sets.
    """
    print("\n[+] Running What-If Experiment A: KNN k-Sensitivity Analysis...")
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    k_values = [1, 3, 5, 7, 9, 11, 15, 21, 31]
    results = []

    for k in k_values:
        t0 = time.time()
        knn = KNeighborsClassifier(n_neighbors=k, metric="cosine")
        knn.fit(X_train_vec, y_train)
        fit_time = time.time() - t0

        t0_inf = time.time()
        y_pred = knn.predict(X_test_vec)
        inf_time = time.time() - t0_inf

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)

        results.append({
            "k_neighbors": k,
            "Accuracy": round(acc, 4),
            "Spam Precision": round(prec, 4),
            "Spam Recall": round(rec, 4),
            "Spam F1-Score": round(f1, 4),
            "Training Time (s)": round(fit_time, 4),
            "Inference Time (s)": round(inf_time, 4)
        })

    df_k = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "what_if_knn_k.csv")
    df_k.to_csv(out_path, index=False)
    print(f"[+] Saved KNN What-If results to: {out_path}")
    return df_k


def experiment_tfidf_features(X_train, X_test, y_train, y_test):
    """
    Experiment B: Evaluate impact of TF-IDF vocabulary size (max_features).
    Tests MultinomialNB and Linear SVM across varied feature constraints.
    """
    print("\n[+] Running What-If Experiment B: TF-IDF max_features Spectrum...")
    feature_limits = [200, 500, 1000, 2000, 3000, 5000, None]
    results = []

    for feat in feature_limits:
        label_feat = feat if feat is not None else "Unlimited (All)"
        tfidf = TfidfVectorizer(max_features=feat, ngram_range=(1, 2), sublinear_tf=True)
        X_tr = tfidf.fit_transform(X_train)
        X_te = tfidf.transform(X_test)
        vocab_size = X_tr.shape[1]

        # Test SVM
        svm = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
        svm.fit(X_tr, y_train)
        svm_preds = svm.predict(X_te)
        
        # Test Naive Bayes
        nb = MultinomialNB(alpha=1.0)
        nb.fit(X_tr, y_train)
        nb_preds = nb.predict(X_te)

        results.append({
            "Max Features Setting": str(label_feat),
            "Actual Vocab Size": vocab_size,
            "SVM Accuracy": round(accuracy_score(y_test, svm_preds), 4),
            "SVM Spam Precision": round(precision_score(y_test, svm_preds, pos_label=1, zero_division=0), 4),
            "SVM Spam Recall": round(recall_score(y_test, svm_preds, pos_label=1, zero_division=0), 4),
            "SVM Spam F1": round(f1_score(y_test, svm_preds, pos_label=1, zero_division=0), 4),
            "NB Accuracy": round(accuracy_score(y_test, nb_preds), 4),
            "NB Spam Precision": round(precision_score(y_test, nb_preds, pos_label=1, zero_division=0), 4),
            "NB Spam Recall": round(recall_score(y_test, nb_preds, pos_label=1, zero_division=0), 4),
            "NB Spam F1": round(f1_score(y_test, nb_preds, pos_label=1, zero_division=0), 4)
        })

    df_feats = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "what_if_tfidf_features.csv")
    df_feats.to_csv(out_path, index=False)
    print(f"[+] Saved TF-IDF Features What-If results to: {out_path}")
    return df_feats


def experiment_ngram_ranges(X_train, X_test, y_train, y_test):
    """
    Experiment C: N-gram range effect (Unigrams vs Bigrams vs Trigrams).
    """
    print("\n[+] Running What-If Experiment C: N-Gram Range Impact...")
    ngram_configs = [
        ("Unigram (1,1)", (1, 1)),
        ("Unigram + Bigram (1,2)", (1, 2)),
        ("Bigram Only (2,2)", (2, 2)),
        ("Unigram + Bigram + Trigram (1,3)", (1, 3))
    ]
    results = []

    for name, ng_range in ngram_configs:
        tfidf = TfidfVectorizer(max_features=3000, ngram_range=ng_range, sublinear_tf=True)
        X_tr = tfidf.fit_transform(X_train)
        X_te = tfidf.transform(X_test)

        svm = SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE)
        svm.fit(X_tr, y_train)
        preds = svm.predict(X_te)

        results.append({
            "N-Gram Scheme": name,
            "Accuracy": round(accuracy_score(y_test, preds), 4),
            "Spam Precision": round(precision_score(y_test, preds, pos_label=1, zero_division=0), 4),
            "Spam Recall": round(recall_score(y_test, preds, pos_label=1, zero_division=0), 4),
            "Spam F1-Score": round(f1_score(y_test, preds, pos_label=1, zero_division=0), 4)
        })

    df_ng = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "what_if_ngram.csv")
    df_ng.to_csv(out_path, index=False)
    print(f"[+] Saved N-Gram What-If results to: {out_path}")
    return df_ng


def run_all_what_if_experiments():
    X_train, X_test, y_train, y_test = load_splits()
    df_k = experiment_knn_k_sensitivity(X_train, X_test, y_train, y_test)
    df_feat = experiment_tfidf_features(X_train, X_test, y_train, y_test)
    df_ng = experiment_ngram_ranges(X_train, X_test, y_train, y_test)
    print("[+] All What-If Experiments completed successfully.")


if __name__ == "__main__":
    run_all_what_if_experiments()

