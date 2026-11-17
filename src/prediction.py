"""
prediction.py
-------------
Inference Engine and Live Prediction Module for SMS Spam Detection.

Author: Mohammed Sameer P (RA2411026050199)
Course: 21CSE377T - Information Retrieval
Assignment: CLA-2 Task 2
"""

import os
import time
import joblib
from typing import Dict, Any, List, Union
from preprocessing import preprocess_text, preprocess_step_by_step

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "Support Vector Machine": "svm.joblib",
    "Multinomial Naive Bayes": "multinomial_naive_bayes.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbours": "knn.joblib"
}


class SpamDetector:
    """Production-grade wrapper for loaded model pipelines."""
    def __init__(self, model_name: str = "Support Vector Machine"):
        self.model_name = model_name
        self.pipeline = self._load_model(model_name)

    def _load_model(self, name: str):
        if name not in MODEL_FILES:
            available = list(MODEL_FILES.keys())
            raise ValueError(f"Unknown model '{name}'. Choose from: {available}")
        
        path = os.path.join(MODELS_DIR, MODEL_FILES[name])
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file '{path}' not found. Please run model_training.py first."
            )
        return joblib.load(path)

    def predict(self, raw_message: str) -> Dict[str, Any]:
        """
        Predict whether a single message is Spam or Ham.
        
        Returns:
            Dict containing prediction label, numeric code, confidence score,
            probabilities (if supported), inference latency, and preprocessing steps.
        """
        t0 = time.perf_counter()
        steps = preprocess_step_by_step(raw_message)
        cleaned_text = steps["final_output"]

        # Handle edge case where text is completely empty after cleaning
        if not cleaned_text.strip():
            cleaned_text = "empty_message"

        # Predict using pipeline
        pred_label_code = int(self.pipeline.predict([cleaned_text])[0])
        
        confidence = 0.5
        probabilities = None
        
        clf = self.pipeline.named_steps.get('classifier', None)
        if hasattr(self.pipeline, "predict_proba"):
            probs = self.pipeline.predict_proba([cleaned_text])[0]
            confidence = float(probs[pred_label_code])
            probabilities = {
                "ham_prob": round(float(probs[0]), 4),
                "spam_prob": round(float(probs[1]), 4)
            }
        elif hasattr(clf, "predict_proba"):
            tfidf = self.pipeline.named_steps['tfidf']
            vec = tfidf.transform([cleaned_text])
            probs = clf.predict_proba(vec)[0]
            confidence = float(probs[pred_label_code])
            probabilities = {
                "ham_prob": round(float(probs[0]), 4),
                "spam_prob": round(float(probs[1]), 4)
            }
        elif hasattr(clf, "decision_function"):
            tfidf = self.pipeline.named_steps['tfidf']
            vec = tfidf.transform([cleaned_text])
            score = float(clf.decision_function(vec)[0])
            confidence = 1.0 / (1.0 + float(pow(2.71828, -abs(score))))
            probabilities = {
                "decision_margin": round(score, 4)
            }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        label_str = "Spam" if pred_label_code == 1 else "Ham (Legitimate)"

        return {
            "model_used": self.model_name,
            "raw_message": raw_message,
            "cleaned_text": cleaned_text,
            "prediction": label_str,
            "prediction_code": pred_label_code,
            "is_spam": bool(pred_label_code == 1),
            "confidence": round(confidence, 4),
            "confidence_percentage": round(confidence * 100, 2),
            "probabilities": probabilities,
            "latency_ms": round(elapsed_ms, 3),
            "preprocessing_breakdown": steps,
            "security_disclaimer": (
                "Notice: This prediction is generated using statistical Information Retrieval "
                "and Machine Learning techniques. It does not constitute a guaranteed security "
                "barrier against zero-day phishing attacks or sophisticated adversarial evasions."
            )
        }

    def predict_all_models(self, raw_message: str) -> Dict[str, Any]:
        """Run the message through all 4 classifiers simultaneously for comparative analysis."""
        results = {}
        for name in MODEL_FILES.keys():
            detector = SpamDetector(name)
            results[name] = detector.predict(raw_message)
            
        # Consensus voting
        votes = [res['prediction_code'] for res in results.values()]
        spam_votes = sum(votes)
        consensus = "Spam" if spam_votes >= 2 else "Ham (Legitimate)"
        
        return {
            "message": raw_message,
            "consensus": consensus,
            "spam_votes": f"{spam_votes}/4",
            "model_predictions": results
        }


if __name__ == "__main__":
    sample_ham = "Hey, are we still meeting at the library at 5 PM for the project discussion?"
    sample_spam = "CONGRATULATIONS! You have won £1,000 cash or a £2,000 holiday prize! Call 09050000327 to claim your prize now. T&Cs apply."
    
    print("\n--- TEST: SPAM DETECTOR INFERENCE ---")
    detector = SpamDetector("Support Vector Machine")
    
    print("\n[Evaluating Ham Sample]:")
    res_ham = detector.predict(sample_ham)
    print(f"Prediction: {res_ham['prediction']} (Confidence: {res_ham['confidence_percentage']}%)")
    print(f"Latency: {res_ham['latency_ms']} ms")
    
    print("\n[Evaluating Spam Sample]:")
    res_spam = detector.predict(sample_spam)
    print(f"Prediction: {res_spam['prediction']} (Confidence: {res_spam['confidence_percentage']}%)")
    print(f"Latency: {res_spam['latency_ms']} ms")

