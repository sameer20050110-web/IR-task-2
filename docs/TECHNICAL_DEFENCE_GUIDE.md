# QUICK TECHNICAL DEFENCE CHEAT SHEET
## 21CSE377T – Information Retrieval | CLA-2 Task 2
**Student:** Mohammed Sameer P (RA2411026050199) | **Max Marks:** 30

---

### 1. ONE-SENTENCE ELEVATOR PITCH
> *"We built an Information Retrieval and Supervised Text Classification framework evaluating Decision Tree, KNN, Linear SVM, and Multinomial Naive Bayes on 5,574 authentic UCI SMS messages, proving that Linear SVM achieves the highest F1-score (0.9296) and Accuracy (98.20%) while Multinomial Naive Bayes delivers near-zero latency (0.2µs) and 99.19% precision."*

---

### 2. THE THREE NUMBERS YOU MUST REMEMBER

| Metric | Champion: Linear SVM | Precision Runner-Up: Naive Bayes | Worst: Decision Tree |
| :--- | :--- | :--- | :--- |
| **Accuracy** | **98.20%** | 97.39% | 96.68% |
| **Spam F1-Score** | **0.9296** | 0.8938 | 0.8746 |
| **False Positives (FP)** | **2** (out of 963) | **1** (out of 963) | **16** (8x worse!) |
| **Spam Recall** | **88.00%** (132/150) | 81.33% (122/150) | 86.00% (129/150) |

---

### 3. THE 4 MODELS IN 15 SECONDS EACH

1. **Linear SVM (The Champion):**
   * *Why it won:* TF-IDF feature space is high-dimensional (3,000 features) and sparse. Text is virtually linearly separable. Linear SVM finds the maximum geometric margin with L2 regularization, preventing overfitting.
2. **Multinomial Naive Bayes (Speed & Precision Winner):**
   * *Why it excels:* Generative probabilistic model with Laplace smoothing ($\alpha=1.0$). Best precision (99.19%, only 1 FP) and blazing fast (0.0002 ms per message).
3. **K-Nearest Neighbours (KNN, k=5):**
   * *Why it struggles:* Lazy learner. Predicts by majority vote across 5 closest vectors using Cosine distance. Inference takes 52 ms because it compares against all 4,448 training instances. Sensitive to class imbalance.
4. **Decision Tree (CART):**
   * *Why it failed:* Recursive axis-aligned orthogonal splits overfit to isolated words in sparse vectors, leading to 16 False Positives.

---

### 4. WHAT-IF EXPERIMENTAL DEFENCE (INSTANT ANSWERS)

* **Question:** *"What happens when you increase $k$ in KNN?"*
  * **Answer:** *"Spam Recall collapses from 83% to 68%! Because Ham is 86.6% of the data, larger $k$ spheres pull in mostly Ham neighbors, drowning out Spam votes."*
* **Question:** *"What is the optimal TF-IDF vocabulary size?"*
  * **Answer:** *"Between 1,000 and 2,000 features. At 200 features models underfit; at 30,500 features Naive Bayes recall drops to 70% due to sparse tail noise."*
* **Question:** *"Why include Bigrams (1,2)?"*
  * **Answer:** *"Unigrams give vocabulary coverage, but Bigrams capture critical spam phrases like 'free cash', 'claim prize', 'call now' that single words miss."*

---

### 5. ARCHITECTURAL INTEGRITY (HOW YOU PREVENTED DATA LEAKAGE)
* `train_test_split(stratify=y, test_size=0.20, random_state=42)`
* `TfidfVectorizer` is fit **ONLY on `X_train`**.
* `X_test` is strictly transformed using the frozen vocabulary and IDF weights from training.
* Vectorizer and classifier are encapsulated in a single `sklearn.pipeline.Pipeline`.

