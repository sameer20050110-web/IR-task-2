# 14-SLIDE PRESENTATION OUTLINE & SPEAKER NOTES
## Course: 21CSE377T - Information Retrieval | CLA-2 Task 2
### Student: Mohammed Sameer P (RA2411026050199) | SRM IST Tiruchirappalli

---

### Slide 1: Title Slide
**Speaker Script / Notes:**
> Respected faculty and evaluation panel, good day. My name is Mohammed Sameer P, Register Number RA2411026050199. Today, I present my technical defence for CLA-2 Task 2 in 21CSE377T Information Retrieval. The project title is Email and SMS Spam Detection Using Machine Learning. In this study, we empirically evaluate four fundamental text classifiers on the authentic UCI SMS Spam Collection benchmark.

---

### Slide 2: Project Overview
**Speaker Script / Notes:**
> On this slide, we outline our project objectives. Spam detection is not merely an academic exercise; unsolicited messages account for billions in economic fraud annually. Our methodology spans five phases: dataset profiling, NLP preprocessing, TF-IDF feature engineering, a four-classifier comparative battle, and empirical what-if sensitivity analysis.

---

### Slide 3: Problem Statement
**Speaker Script / Notes:**
> Here we highlight the fundamental problem statement. In information retrieval and security, we face an asymmetric cost matrix. If our filter commits a False Positive, a user misses an urgent job offer or bank OTP. Furthermore, SMS messages provide very few words per document, creating extreme matrix sparsity.

---

### Slide 4: Dataset Profiling
**Speaker Script / Notes:**
> Let us discuss our dataset. We obtained the official SMS Spam Collection from the UCI Machine Learning Repository. It contains 5,574 messages: 4,827 legitimate messages and 747 spam messages. Notice the high class imbalance: 86.6% to 13.4%. This realistic distribution is preserved during our stratified sampling.

---

### Slide 5: Preprocessing
**Speaker Script / Notes:**
> On this slide, we detail our NLP preprocessing pipeline. Raw SMS text is extremely noisy. We pass every message through six standardized stages: lowercasing, regex URL stripping, punctuation removal, tokenization, stop-word filtering, and Porter stemming. As demonstrated, the raw 101-character message is distilled into 7 high-information stems.

---

### Slide 6: TF-IDF Formulation
**Speaker Script / Notes:**
> Turning to feature engineering, we utilize Term Frequency - Inverse Document Frequency. Term Frequency captures local importance within a single SMS, while Inverse Document Frequency down-weights words that appear everywhere. Crucially, to prevent data leakage, our vectorizer is fit strictly on X_train. The test set is only transformed.

---

### Slide 7: Classifiers Theory
**Speaker Script / Notes:**
> Here we introduce the four competing classifiers. We selected four distinct machine learning paradigms: a tree-based model (Decision Tree), an instance-based lazy learner (KNN), a margin-based geometric classifier (Linear SVM), and a probabilistic generative baseline (Multinomial Naive Bayes). All models are evaluated under identical conditions.

---

### Slide 8: Evaluation Framework
**Speaker Script / Notes:**
> On slide 8, we explain our evaluation framework and defend why Accuracy is misleading. Because Ham constitutes 86.6% of the data, a model that classifies everything as Ham gets 86.6% accuracy while doing zero work. Therefore, we focus on Spam Precision, Spam Recall, and their harmonic mean, the F1-Score.

---

### Slide 9: Empirical Results
**Speaker Script / Notes:**
> Here are our authentic empirical test results on 1,113 test samples. Notice that Linear SVM is the clear overall victor with 98.20% accuracy and 0.9296 F1-score. Multinomial Naive Bayes achieved an extraordinary 99.19% precision, producing only 1 False Positive out of 963 legitimate messages.

---

### Slide 10: Confusion Matrix Deep Dive
**Speaker Script / Notes:**
> Slide 10 breaks down the confusion matrix errors. In spam detection, False Positives are catastrophic. Decision Tree committed 16 False Positives, meaning 16 real messages were dropped. In contrast, Naive Bayes committed only 1 False Positive, and SVM committed only 2 False Positives while maintaining 88% recall.

---

### Slide 11: Computational Latency
**Speaker Script / Notes:**
> On slide 11, we examine runtime computational complexity. Naive Bayes is blazing fast, training in under 2 milliseconds and predicting in fractions of a microsecond. In contrast, while KNN trains in zero time, its inference requires computing cosine distance against all 4,448 training vectors, making it impractical for high-throughput telecom gateways.

---

### Slide 12: What-If Experiments
**Speaker Script / Notes:**
> Here we present our What-If sensitivity experiments. We conducted three experiments: First, we varied KNN neighbors from 1 to 31 and discovered that large k destroys Spam Recall due to class imbalance. Second, we tested TF-IDF vocabulary limits, showing that 2,000 features provides the sweet spot. Third, combining Unigrams with Bigrams provides the best discriminative power for phrases like 'free prize'.

---

### Slide 13: Conclusions & Limitations
**Speaker Script / Notes:**
> In conclusion, Linear SVM is our definitively defended champion, delivering 98.20% accuracy and 0.9296 F1. Multinomial Naive Bayes is our speed and precision runner-up. Importantly, we note that statistical classifiers are threat indicators and must be complemented by cybersecurity layers in production.

---

### Slide 14: References & Defense
**Speaker Script / Notes:**
> Thank you very much for your time and evaluation. I am now ready to defend my project and answer any technical questions.

---

