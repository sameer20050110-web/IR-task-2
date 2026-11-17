# COMPREHENSIVE VIVA VOCE QUESTIONS & DEFENCE GUIDE
## Course: 21CSE377T – Information Retrieval | Assignment: CLA-2 Task 2
### Title: Email and SMS Spam Detection Using Machine Learning
**Student:** Mohammed Sameer P  
**Register Number:** RA2411026050199  
**Institution:** SRM Institute of Science and Technology, Tiruchirappalli  
**Maximum Marks:** 30  

---

### SECTION 1: CORE INFORMATION RETRIEVAL & DOMAIN CONCEPTS

#### Q1: What is the primary difference between a Spam message and a Ham message?
**Answer:**
* **Ham:** A legitimate, non-malicious, authorized communication (e.g., personal chats, two-factor OTP authentication codes, flight bookings, meeting reminders).
* **Spam:** An unsolicited, bulk, often malicious or deceptive electronic message sent without recipient consent (e.g., lottery scams, fake banking alerts, aggressive marketing lures, phishing hyperlinks).

#### Q2: Why is this project tested on the SMS Spam Collection instead of an email dataset, despite the title mentioning Email and SMS?
**Answer:**
The project methodology applies universally to text classification. However, the empirical benchmark is the **UCI SMS Spam Collection (5,574 messages)** because SMS presents an inherently tougher, higher-entropy Information Retrieval challenge. Unlike emails that feature rich metadata (such as IP headers, DKIM signatures, SPF records, and MIME structures), SMS messages are constrained to 160 characters, rely on informal slang ('u', 'ur', 'txt'), contain intentional spelling evasions, and suffer from high term sparsity. Defending a model on SMS guarantees robustness in short-text environments.

#### Q3: Where does the UCI SMS Spam Collection originate from and what is its class distribution?
**Answer:**
The dataset was compiled by Almeida, Gómez Hidalgo, and Yamakami from four public corpora:
1. The Singapore National University (NUS) SMS Corpus (3,375 legitimate messages).
2. The Grumbletext UK consumer reports website (425 spam messages).
3. Caroline Tag's doctoral thesis SMS corpus (450 legitimate messages).
4. The SMS Spam Corpus v.0.1 Big (1,002 ham and 322 spam messages).
**Total:** 5,574 messages (4,827 Ham, 747 Spam). The class ratio is approximately 86.6% Ham to 13.4% Spam.

#### Q4: What is class imbalance and why is standard Accuracy misleading in this problem?
**Answer:**
Class imbalance occurs when one class substantially outnumbers another (here, Ham is ~6.5 times larger than Spam).
Standard Accuracy is misleading because a naive "Zero-Rule" classifier that predicts "Ham" for 100% of messages will achieve an **86.60% Accuracy**, yet its Spam Recall will be **0.0%**. It will fail to intercept even a single spam message, rendering it completely useless in production. Therefore, Precision, Recall, and the F1-Score for the Spam class are the essential evaluative criteria.

---

### SECTION 2: TEXT PREPROCESSING & NORMALIZATION

#### Q5: What are the sequence of steps implemented in your text preprocessing pipeline?
**Answer:**
1. **Case Normalization:** Converts all characters to lowercase to unify token representations (`WIN`, `Win`, `win` -> `win`).
2. **URL and Email Stripping:** Removes hyperlinked web URLs (`http://...`) and email addresses using regex patterns.
3. **Punctuation and Symbol Removal:** Strips ASCII punctuation, dollar/pound symbols, and formatting noise.
4. **Tokenization:** Splits character streams into discrete lexical tokens.
5. **Stop-word Removal:** Removes high-frequency, low-discrimination English words (`is`, `the`, `at`, `on`, `which`) using NLTK corpora.
6. **Porter Stemming:** Strips suffixes to reduce inflected words to morphological roots (`winner`, `winning` -> `win`).

#### Q6: Why did you choose Stemming instead of Lemmatization?
**Answer:**
Porter Stemming is a fast, rule-based algorithmic heuristic that operates without needing part-of-speech (POS) tagging or large grammatical lookup dictionaries. In short, noisy SMS texts where grammatical structure is fragmented or slang is pervasive, POS taggers frequently fail or introduce substantial latency. Stemming provides the highest operational throughput while grouping related spam stems (`valu`, `custom`, `claim`) effectively.

#### Q7: Provide an exact before-and-after example of your preprocessing pipeline.
**Answer:**
* **Raw Message:** `"WINNER!! As a valued customer, claim your FREE cash prize of £500 now at http://spam.xyz or call 0800!"`
* **Cleaned Output:** `"winner valu custom claim free cash prize call"`
* **Tokens eliminated:** `as`, `a`, `your`, `of`, `now`, `at`, `or`, `500`, URL, punctuation, and casing.

---

### SECTION 3: TF-IDF & FEATURE ENGINEERING

#### Q8: What is TF-IDF and what are its mathematical formulas?
**Answer:**
TF-IDF stands for **Term Frequency - Inverse Document Frequency**. It weights terms based on their local prevalence in a document and global rarity across the corpus.
1. **Term Frequency (TF):**
   $$TF(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}} \quad \text{or with sublinear scaling:} \quad 1 + \log(f_{t, d})$$
2. **Inverse Document Frequency (IDF):**
   $$IDF(t, D) = \log\left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$
3. **Composite TF-IDF:**
   $$TF\text{-}IDF(t, d, D) = TF(t, d) \times IDF(t, D)$$
Followed by Euclidean $L_2$ normalization so document length does not distort vector magnitudes:
$$\vec{v}_{\text{norm}} = \frac{\vec{v}}{\|\vec{v}\|_2}$$

#### Q9: How did you prevent data leakage during TF-IDF vectorization?
**Answer:**
Data leakage occurs if the feature extractor computes statistics (such as vocabulary dictionary or inverse document frequencies) using both training and testing samples.
In this project:
* The `TfidfVectorizer` is **fit exclusively on `X_train`** (the 4,448 training messages).
* The test set `X_test` (1,113 messages) is strictly transformed using the frozen vocabulary and IDF weights computed solely from `X_train`.
* In production, the vectorizer is bundled inside a Scikit-Learn `Pipeline`.

#### Q10: What are N-grams, and why did you use `ngram_range=(1, 2)`?
**Answer:**
An N-gram is a contiguous sequence of $N$ items from text. Unigrams ($N=1$) are single words (`free`), while Bigrams ($N=2$) capture two-word combinations (`free cash`, `call now`).
We utilized `ngram_range=(1, 2)` because unigrams ensure broad lexical coverage while bigrams capture key phishing collocations that carry far higher spam probability than the words would individually.

---

### SECTION 4: THE FOUR MACHINE LEARNING CLASSIFIERS

#### Q11: Explain how the Decision Tree operates for text classification and why it underperformed.
**Answer:**
* **Mechanism:** Recursively splits training instances based on feature presence/absence to minimize Gini Impurity ($Gini = 1 - \sum p_i^2$).
* **Reason for Underperformance:** Text vectorization produces a high-dimensional (3,000 features), highly sparse space. Decision trees make orthogonal, single-feature splits. In sparse text, this causes the tree to create deeply fragmented leaves over idiosyncratic words, leading to high variance, overfitting, and the highest False Positive count (16 FPs).

#### Q12: How does K-Nearest Neighbours (KNN) work, and why did you use Cosine distance instead of Euclidean distance?
**Answer:**
* **Mechanism:** A lazy, instance-based learner that stores training vectors and classifies a query message by finding the $k$ closest neighbors and taking a majority vote.
* **Metric Choice:** We used Cosine distance ($1 - \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$). In high-dimensional sparse text, Euclidean distance suffers from the "curse of dimensionality" where distances between all vectors converge. Cosine distance measures angular orientation, completely disregarding document length variations.

#### Q13: Why did Linear SVM emerge as the overall champion in your battle?
**Answer:**
* **Mechanism:** Linear Support Vector Machine solves a convex quadratic optimization problem to find the maximum-margin hyperplane separating classes:
  $$\min \frac{1}{2} \|\vec{w}\|^2 + C \sum \xi_i \quad \text{s.t.} \quad y_i(\vec{w}^T \vec{x}_i + b) \ge 1 - \xi_i$$
* **Reason for Victory:** In 3,000-dimensional sparse TF-IDF space, text classes are almost linearly separable (Joachims, 1998). Linear SVM maximizes margin separation while applying $L_2$ regularization on the weight vector $\vec{w}$, which prevents overfitting even when features outnumber samples. It achieved **98.20% Accuracy** and the highest **Spam F1-score (0.9296)**.

#### Q14: How does Multinomial Naive Bayes work and what is Laplace Smoothing?
**Answer:**
* **Mechanism:** Applies Bayes' Theorem under the conditional independence assumption:
  $$P(\text{Spam} \mid \vec{d}) \propto P(\text{Spam}) \prod_{i=1}^n P(t_i \mid \text{Spam})^{f_i}$$
* **Laplace Smoothing ($\alpha=1.0$):** Adds a pseudocount of 1 to every word count:
  $$P(t \mid c) = \frac{\text{count}(t, c) + \alpha}{\sum_{t'} \text{count}(t', c) + \alpha |V|}$$
  This prevents the "zero-frequency problem" where a single unseen word in a test message would otherwise cause the entire class probability product to become zero.

---

### SECTION 5: EVALUATION, CONFUSION MATRICES & WHAT-IF ANALYSIS

#### Q15: In spam detection, which error is more catastrophic: False Positive or False Negative?
**Answer:**
**False Positives (FP) are significantly more catastrophic.**
* **False Positive (FP):** A legitimate message (Ham) is erroneously flagged as Spam and redirected to the junk folder or deleted. The user misses critical communications (e.g., job offer, flight ticket, banking OTP).
* **False Negative (FN):** A Spam message slips into the primary inbox. The user experiences minor inconvenience.
Therefore, a commercial filter prioritizes extremely high Precision (minimizing FP) over raw Recall.

#### Q16: Walk through the exact Confusion Matrix numbers for your Linear SVM on the 1,113 test samples.
**Answer:**
* **True Negatives (TN):** 961 (Legitimate messages correctly identified as Ham)
* **False Positives (FP):** 2 (Legitimate messages wrongly flagged as Spam)
* **False Negatives (FN):** 18 (Spam messages that escaped into the inbox)
* **True Positives (TP):** 132 (Spam messages correctly blocked)
* Total Ham in test set: $961 + 2 = 963$
* Total Spam in test set: $18 + 132 = 150$

#### Q17: Which model had the highest Spam Precision, and what was the trade-off?
**Answer:**
**Multinomial Naive Bayes** achieved the highest Spam Precision at **99.19%**, producing only **1 False Positive** out of 963 legitimate messages. The trade-off was slightly lower Recall (**81.33%**, missing 28 spam messages) compared to SVM (**88.00%**, missing only 18 spam messages).

#### Q18: What happened in your What-If experiment when you increased $k$ in KNN from 3 to 31?
**Answer:**
Spam Recall **collapsed dramatically from 83.33% down to 68.00%**.
**Reason:** Because Ham represents 86.6% of the dataset, expanding the neighborhood sphere ($k$) pulls in an overwhelming proportion of majority-class Ham vectors. Minority-class Spam queries fail to gather enough votes to achieve a majority, so the model increasingly defaults to predicting Ham.

#### Q19: What did your TF-IDF feature capacity What-If experiment reveal?
**Answer:**
* At **200 features**, models underfit (F1 ~0.87) due to inadequate vocabulary coverage.
* Between **1,000 and 2,000 features**, models reach peak performance (SVM F1: 0.9412).
* At **Unlimited features (30,508)**, Naive Bayes recall degrades to **70.0%** because ultra-rare tail n-grams with frequency of 1 corrupt probability estimates, whereas Linear SVM remains resilient due to $L_2$ regularization.

#### Q20: Compare the computational complexity and runtime latency between Naive Bayes and KNN.
**Answer:**
* **Multinomial Naive Bayes:** Training time is $O(N \cdot |V|)$ (linear pass, taking 0.0017 seconds). Inference is $O(|d|)$ (taking 0.0002 milliseconds per query).
* **KNN:** Training time is $O(1)$ (0.0009 seconds, just stores data). However, inference is $O(N \cdot |V|)$ per query because it must compute cosine distance against all 4,448 training instances. This required 52 milliseconds per batch, making KNN unacceptable for real-time telecom SMS gateways.

---

### SECTION 6: IMPLEMENTATION & DEFENCE SUMMARY

#### Q21: What libraries and tools did you use to build the solution?
**Answer:**
* **Core Language:** Python 3.14
* **NLP & Text Processing:** NLTK (PorterStemmer, tokenization, stopwords) and regular expressions
* **Machine Learning & Pipeline:** Scikit-Learn (`TfidfVectorizer`, `Pipeline`, `DecisionTreeClassifier`, `KNeighborsClassifier`, `SVC`, `MultinomialNB`, `metrics`)
* **Data Processing & Math:** Pandas and NumPy
* **Visualization:** Matplotlib and Seaborn
* **Model Serialization:** Joblib
* **Interactive Dashboard:** Streamlit
* **Documentation & Reporting:** Python-docx, Python-pptx, and Jupyter (`nbformat`)

#### Q22: Can this machine learning model guarantee complete protection against all spam?
**Answer:**
**No.** Machine learning classifiers are statistical pattern-matching systems based on training data distributions. They cannot provide an absolute security guarantee because:
1. Spammers continuously evolve adversarial tactics (e.g., zero-width spaces, homoglyph character substitutions such as Cyrillic 'a' for Latin 'a', image-based spam).
2. Concept drift occurs as new slang and scam themes emerge.
3. In production, ML classifiers must be augmented by IP rate-limiters, phone number reputation lists, DNS/SPF checks, and heuristic rule engines.

#### Q23: How would you scale this system to handle millions of SMS per second in a telecom infrastructure?
**Answer:**
1. Deploy **Multinomial Naive Bayes or Linear SVM with quantized weights** in an optimized C++/Rust inference runtime (e.g., ONNX Runtime or TensorRT).
2. Use feature hashing (HashingVectorizer) to fix memory footprint without dictionary lookups.
3. Implement a multi-tiered filtering cascade: quick blacklist/whitelist lookup -> lightweight Naive Bayes filter -> deep SVM analysis only for borderline ambiguous cases.

#### Q24: What is stratified sampling and why did you use it?
**Answer:**
Stratified sampling (`stratify=y` in `train_test_split`) guarantees that the train and test subsets maintain the identical class distribution ratio (86.6% Ham and 13.4% Spam) as the original corpus. Without stratification, random sampling could result in an unrepresentative test split with too few spam instances, introducing high evaluation variance.

#### Q25: Concluding Defence: Summarize why you recommend Linear SVM for this task.
**Answer:**
Linear SVM provides the optimal trade-off in the high-dimensional sparse TF-IDF text space:
1. Highest Test Accuracy (98.20%).
2. Highest Spam Recall (88.00%), catching 132 out of 150 spam messages.
3. Highest Spam F1-Score (0.9296).
4. Exceptional False Positive control (only 2 FPs out of 963 legitimate messages).
5. Sub-millisecond inference per message, making it fully viable for production deployment.

