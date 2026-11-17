"""
preprocessing.py
----------------
Text Preprocessing Pipeline for SMS Spam Detection.

Author: Mohammed Sameer P (RA2411026050199)
Course: 21CSE377T - Information Retrieval
Assignment: CLA-2 Task 2 - Text Classification Battle
Institution: SRM Institute of Science and Technology, Tiruchirappalli
"""

import os
import re
import string
from typing import Dict, List, Any
import nltk
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin

# Append local workspace nltk_data path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_NLTK_DIR = os.path.join(BASE_DIR, "nltk_data")
if os.path.exists(LOCAL_NLTK_DIR) and LOCAL_NLTK_DIR not in nltk.data.path:
    nltk.data.path.insert(0, LOCAL_NLTK_DIR)

# Comprehensive standard English stopwords list as robust fallback
STANDARD_ENGLISH_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm',
    'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven',
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"
}

# Try loading NLTK stopwords, falling back gracefully to standard set
try:
    from nltk.corpus import stopwords
    _STOPWORDS = set(stopwords.words('english'))
except Exception:
    _STOPWORDS = STANDARD_ENGLISH_STOPWORDS

try:
    _STEMMER = PorterStemmer()
except Exception:
    _STEMMER = PorterStemmer()


def clean_text_basic(text: str) -> str:
    """
    Standardize text: convert to lower case, normalize whitespace,
    and remove non-ASCII characters.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[\r\n\t]+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip()


def remove_urls_and_emails(text: str) -> str:
    """Remove URLs, hyperlinks, and email addresses."""
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' ', text)
    return text


def remove_punctuation_and_digits(text: str, keep_numbers: bool = False) -> str:
    """Remove punctuation and optionally digits."""
    # Replace punctuation characters with space
    text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    if not keep_numbers:
        text = re.sub(r'\b\d+\b', ' ', text)
    return text


def tokenize(text: str) -> List[str]:
    """Tokenize string into individual word tokens."""
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = re.findall(r'\b\w+\b', text)
    return [t for t in tokens if len(t) > 1]


def remove_stopwords(tokens: List[str], custom_stopwords: set = None) -> List[str]:
    """Remove English stop words from token list."""
    sw = custom_stopwords if custom_stopwords is not None else _STOPWORDS
    return [token for token in tokens if token not in sw]


def stem_tokens(tokens: List[str]) -> List[str]:
    """Apply Porter Stemmer to reduce words to root morphological forms."""
    return [_STEMMER.stem(token) for token in tokens]


def preprocess_text(text: str, use_stemming: bool = True) -> str:
    """
    Complete end-to-end preprocessing pipeline returning clean string.
    
    1. Lowercase normalization
    2. URL and email removal
    3. Punctuation removal
    4. Tokenization
    5. Stopword filtering
    6. Stemming (optional, default True)
    7. Reconstruction into normalized string
    """
    text = clean_text_basic(text)
    text = remove_urls_and_emails(text)
    text = remove_punctuation_and_digits(text, keep_numbers=False)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    if use_stemming:
        tokens = stem_tokens(tokens)
    return " ".join(tokens)


def preprocess_step_by_step(text: str) -> Dict[str, Any]:
    """
    Demonstrate granular transformation steps for educational walkthroughs,
    interactive Streamlit visualizations, and viva demonstrations.
    """
    original = str(text)
    step1_lower = clean_text_basic(original)
    step2_no_urls = remove_urls_and_emails(step1_lower)
    step3_no_punct = remove_punctuation_and_digits(step2_no_urls, keep_numbers=False)
    step4_tokens = tokenize(step3_no_punct)
    step5_no_stopwords = remove_stopwords(step4_tokens)
    step6_stemmed = stem_tokens(step5_no_stopwords)
    final_output = " ".join(step6_stemmed)

    return {
        "original": original,
        "step1_lowercased": step1_lower,
        "step2_no_urls_emails": step2_no_urls,
        "step3_no_punct": step3_no_punct,
        "step4_tokens": step4_tokens,
        "step5_no_stopwords": step5_no_stopwords,
        "step6_stemmed": step6_stemmed,
        "final_output": final_output,
        "original_len": len(original),
        "cleaned_len": len(final_output),
        "removed_tokens": [t for t in step4_tokens if t in _STOPWORDS]
    }


class TextPreprocessor(BaseEstimator, TransformerMixin):
    """
    Scikit-Learn compatible transformer to seamlessly integrate into
    sklearn.pipeline.Pipeline without data leakage.
    """
    def __init__(self, use_stemming: bool = True):
        self.use_stemming = use_stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "tolist"):
            X = X.tolist()
        return [preprocess_text(str(x), use_stemming=self.use_stemming) for x in X]


if __name__ == "__main__":
    sample = "WINNER!! As a valued customer, claim your FREE cash prize of 500 now at http://spam.xyz or call 0800!"
    print("--- SAMPLE PREPROCESSING DEMO ---")
    steps = preprocess_step_by_step(sample)
    for k, v in steps.items():
        print(f"{k}: {v}")
