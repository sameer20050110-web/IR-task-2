"""
generate_presentation.py
------------------------
Generates a 14-slide professional widescreen presentation (16:9)
using python-pptx for CLA-2 Task 2.

Author: Mohammed Sameer P (RA2411026050199)
Course: 21CSE377T - Information Retrieval
Institution: SRM Institute of Science and Technology, Tiruchirappalli
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PPTX = os.path.join(BASE_DIR, "SMS_Spam_Detection_Presentation.pptx")
OUTLINE_MD = os.path.join(BASE_DIR, "SLIDES_OUTLINE.md")

# Academic Color Palette
COLOR_PRIMARY = RGBColor(10, 37, 64)      # Deep Navy
COLOR_SECONDARY = RGBColor(0, 119, 182)   # Cerulean Blue
COLOR_ACCENT = RGBColor(220, 53, 69)      # Crimson Red (Spam)
COLOR_SUCCESS = RGBColor(40, 167, 69)     # Forest Green (Ham)
COLOR_TEXT_DARK = RGBColor(33, 37, 41)    # Charcoal
COLOR_BG_LIGHT = RGBColor(248, 249, 250)  # Off-white
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_MUTED = RGBColor(108, 117, 125)     # Gray


def set_shape_flat(shape, fill_color, line_color=None):
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    if line_color:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()


def add_header(slide, title_text, category_text="21CSE377T: INFORMATION RETRIEVAL | CLA-2 TASK 2"):
    # Category Tracker
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.35))
    tf_c = cat_box.text_frame
    tf_c.word_wrap = True
    p_c = tf_c.paragraphs[0]
    p_c.text = category_text.upper()
    p_c.font.size = Pt(10)
    p_c.font.bold = True
    p_c.font.color.rgb = COLOR_SECONDARY

    # Title
    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.5), Inches(0.8))
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY


def add_notes(slide, notes_text):
    notes_slide = slide.notes_slide
    text_frame = notes_slide.notes_text_frame
    text_frame.text = notes_text


def create_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    slides_info = []

    # ==========================================
    # SLIDE 1: TITLE SLIDE
    # ==========================================
    slide1 = prs.slides.add_slide(blank_layout)
    # Background accent card
    bg = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    set_shape_flat(bg, COLOR_PRIMARY)

    # Accent Stripe
    stripe = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.15), Inches(5.0))
    set_shape_flat(stripe, COLOR_SECONDARY)

    tb = slide1.shapes.add_textbox(Inches(1.2), Inches(1.3), Inches(11.0), Inches(4.8))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = "SRM INSTITUTE OF SCIENCE AND TECHNOLOGY, TIRUCHIRAPPALLI"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = RGBColor(144, 224, 239)

    p1 = tf.add_paragraph()
    p1.text = "EMAIL AND SMS SPAM DETECTION USING MACHINE LEARNING"
    p1.font.size = Pt(28)
    p1.font.bold = True
    p1.font.color.rgb = COLOR_WHITE
    p1.space_before = Pt(14)

    p2 = tf.add_paragraph()
    p2.text = "CLA-2 Task 2: Text Classification Battle – Compare, Analyze and Defend"
    p2.font.size = Pt(17)
    p2.font.color.rgb = RGBColor(200, 225, 255)
    p2.space_before = Pt(8)

    p3 = tf.add_paragraph()
    p3.text = "\nSTUDENT CREDENTIALS:\n" \
              "• Name: Mohammed Sameer P\n" \
              "• Register Number: RA2411026050199\n" \
              "• Course: 21CSE377T – Information Retrieval | Assignment Marks: 30\n" \
              "• Department of Computer Science & Engineering"
    p3.font.size = Pt(13)
    p3.font.color.rgb = RGBColor(220, 230, 242)
    p3.space_before = Pt(20)

    notes1 = "Respected faculty and evaluation panel, good day. My name is Mohammed Sameer P, Register Number RA2411026050199. " \
             "Today, I present my technical defence for CLA-2 Task 2 in 21CSE377T Information Retrieval. " \
             "The project title is Email and SMS Spam Detection Using Machine Learning. " \
             "In this study, we empirically evaluate four fundamental text classifiers on the authentic UCI SMS Spam Collection benchmark."
    add_notes(slide1, notes1)
    slides_info.append(("Slide 1: Title Slide", notes1))

    # ==========================================
    # SLIDE 2: PROJECT OVERVIEW & OBJECTIVES
    # ==========================================
    slide2 = prs.slides.add_slide(blank_layout)
    add_header(slide2, "Project Overview & Executive Summary")

    tb2 = slide2.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p = tf2.paragraphs[0]
    p.text = "Core Objectives & Deliverables:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets = [
        "1. Real-World Spam Identification: Categorize unstructured textual communications into Legitimate (Ham) or Malicious (Spam).",
        "2. Rigorous NLP Preprocessing: Implement lowercasing, tokenization, URL/punctuation stripping, stop-word removal, and Porter stemming.",
        "3. Mathematical Feature Representation: Convert text tokens into numerical vectors via TF-IDF, fitting strictly on training partitions to prevent data leakage.",
        "4. The 4-Classifier Battle: Execute an identical stratified 80/20 train-test split across Decision Tree, K-Nearest Neighbours, Linear SVM, and Multinomial Naive Bayes.",
        "5. Class Imbalance & Diagnostic Evaluation: Analyze why accuracy is misleading when 86.6% of data is Ham, focusing on Spam Precision, Recall, F1, and Confusion Matrices.",
        "6. Empirical What-If Analysis: Investigate KNN neighbor sensitivity (k), TF-IDF vocabulary limits, and N-gram range dynamics.",
        "7. Interactive Streamlit Interface: Provide real-time classification, multi-model consensus voting, and educational processing breakdowns."
    ]
    for b in bullets:
        p_b = tf2.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(13)
        p_b.space_before = Pt(8)
        p_b.font.color.rgb = COLOR_TEXT_DARK

    notes2 = "On this slide, we outline our project objectives. Spam detection is not merely an academic exercise; unsolicited messages account for " \
             "billions in economic fraud annually. Our methodology spans five phases: dataset profiling, NLP preprocessing, TF-IDF feature engineering, " \
             "a four-classifier comparative battle, and empirical what-if sensitivity analysis."
    add_notes(slide2, notes2)
    slides_info.append(("Slide 2: Project Overview", notes2))

    # ==========================================
    # SLIDE 3: PROBLEM STATEMENT
    # ==========================================
    slide3 = prs.slides.add_slide(blank_layout)
    add_header(slide3, "Problem Statement: The Challenge of SMS & Email Spam")

    tb3 = slide3.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf3 = tb3.text_frame
    tf3.word_wrap = True

    p = tf3.paragraphs[0]
    p.text = "Why Spam Detection is a Hard Information Retrieval Problem:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets3 = [
        "• Asymmetric Error Costs: In spam detection, a False Positive (classifying an important legitimate email or bank OTP as spam) is catastrophic. A False Negative (spam in inbox) is merely a nuisance.",
        "• Short-Text Sparsity in SMS: Unlike long emails with rich MIME headers, SMS messages average only 15 words and under 160 characters, providing very few lexical co-occurrences.",
        "• Colloquial Lingo & Intentional Obfuscation: Users and spammers use heavy abbreviations ('u', 'ur', 'txt', 'winr'), leetspeak ('fr33'), and grammatical variations.",
        "• High Class Imbalance: Legitimate messages naturally outnumber spam messages (~6.5 to 1), causing naive classifiers to predict only the majority class."
    ]
    for b in bullets3:
        p_b = tf3.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(14)
        p_b.space_before = Pt(12)
        p_b.font.color.rgb = COLOR_TEXT_DARK

    notes3 = "Here we highlight the fundamental problem statement. In information retrieval and security, we face an asymmetric cost matrix. " \
             "If our filter commits a False Positive, a user misses an urgent job offer or bank OTP. " \
             "Furthermore, SMS messages provide very few words per document, creating extreme matrix sparsity."
    add_notes(slide3, notes3)
    slides_info.append(("Slide 3: Problem Statement", notes3))

    # ==========================================
    # SLIDE 4: DATASET PROFILING
    # ==========================================
    slide4 = prs.slides.add_slide(blank_layout)
    add_header(slide4, "Dataset Profiling: The UCI SMS Spam Collection")

    tb4 = slide4.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf4 = tb4.text_frame
    tf4.word_wrap = True

    p = tf4.paragraphs[0]
    p.text = "Authentic Dataset Characteristics (No Fabricated Data):"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets4 = [
        "• Source: Official UCI Machine Learning Repository (Almeida, Gómez Hidalgo, and Yamakami).",
        "• Provenance: 5,574 authentic English SMS messages collected from the NUS SMS Corpus, Grumbletext UK consumer reports, and mobile research corpora.",
        "• Class Distribution: 4,827 Ham messages (86.60%) vs. 747 Spam messages (13.40%).",
        "• Data Quality & Verification: Zero missing or corrupted records; validated against UCI ground truth.",
        "• SMS vs Email Justification: Although the project title addresses Email and SMS, empirical testing is conducted on SMS because it offers a tougher short-text challenge without relying on email header artifacts."
    ]
    for b in bullets4:
        p_b = tf4.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(13.5)
        p_b.space_before = Pt(10)
        p_b.font.color.rgb = COLOR_TEXT_DARK

    notes4 = "Let us discuss our dataset. We obtained the official SMS Spam Collection from the UCI Machine Learning Repository. " \
             "It contains 5,574 messages: 4,827 legitimate messages and 747 spam messages. " \
             "Notice the high class imbalance: 86.6% to 13.4%. This realistic distribution is preserved during our stratified sampling."
    add_notes(slide4, notes4)
    slides_info.append(("Slide 4: Dataset Profiling", notes4))

    # ==========================================
    # SLIDE 5: PREPROCESSING PIPELINE
    # ==========================================
    slide5 = prs.slides.add_slide(blank_layout)
    add_header(slide5, "Natural Language Text Preprocessing Pipeline")

    tb5 = slide5.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf5 = tb5.text_frame
    tf5.word_wrap = True

    p = tf5.paragraphs[0]
    p.text = "Six-Stage Lexical Normalization Architecture:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets5 = [
        "1. Lowercasing: Maps uppercase variations to common forms ('FREE', 'Free' -> 'free').",
        "2. Hyperlink & Email Stripping: Uses regular expressions to eliminate web URLs and emails.",
        "3. Punctuation & Digit Removal: Strips currency signs, exclamation marks, and numeric strings.",
        "4. Word Tokenization: Segments continuous text streams into distinct lexical tokens.",
        "5. Stop-word Removal: Filters non-discriminative English syntax tokens ('is', 'the', 'at', 'which').",
        "6. Porter Stemming: Morphemically reduces words to linguistic root stems ('winner', 'winning' -> 'win').",
        "\nConcrete Before-and-After Example:",
        "  • Before: 'WINNER!! As a valued customer, claim your FREE cash prize of 500 now at http://spam.xyz!'",
        "  • After:  'winner valu custom claim free cash prize'"
    ]
    for b in bullets5:
        p_b = tf5.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(12.5)
        p_b.space_before = Pt(6)
        p_b.font.color.rgb = COLOR_TEXT_DARK

    notes5 = "On this slide, we detail our NLP preprocessing pipeline. Raw SMS text is extremely noisy. " \
             "We pass every message through six standardized stages: lowercasing, regex URL stripping, punctuation removal, " \
             "tokenization, stop-word filtering, and Porter stemming. " \
             "As demonstrated, the raw 101-character message is distilled into 7 high-information stems."
    add_notes(slide5, notes5)
    slides_info.append(("Slide 5: Preprocessing", notes5))

    # ==========================================
    # SLIDE 6: TF-IDF FORMULATION
    # ==========================================
    slide6 = prs.slides.add_slide(blank_layout)
    add_header(slide6, "TF-IDF Feature Representation & Mathematical Foundations")

    tb6 = slide6.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf6 = tb6.text_frame
    tf6.word_wrap = True

    p = tf6.paragraphs[0]
    p.text = "Mathematical Formulation of TF-IDF Weighting:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets6 = [
        "• Term Frequency (TF): Quantifies how frequently term t occurs in document d:",
        "    TF(t, d) = f(t, d) / ∑ f(t', d)    (with sublinear log-scaling: 1 + log(f(t, d)))",
        "• Inverse Document Frequency (IDF): Penalizes terms appearing universally across the corpus:",
        "    IDF(t, D) = log[ (1 + |D|) / (1 + |{d ∈ D : t ∈ d}|) ] + 1",
        "• Composite TF-IDF Weight: Term frequency scaled by global rarity:",
        "    TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)",
        "• L2 Vector Normalization: Ensures variable message lengths do not skew cosine distance calculations:",
        "    v_norm = v / ||v||_2",
        "• Data Leakage Prevention: TF-IDF is fit STRICTLY on training split (X_train). Test data (X_test) is transformed using pre-computed training vocabulary and IDF weights."
    ]
    for b in bullets6:
        p_b = tf6.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(12.5)
        p_b.space_before = Pt(6)
        p_b.font.color.rgb = COLOR_TEXT_DARK

    notes6 = "Turning to feature engineering, we utilize Term Frequency - Inverse Document Frequency. " \
             "Term Frequency captures local importance within a single SMS, while Inverse Document Frequency down-weights words " \
             "that appear everywhere. " \
             "Crucially, to prevent data leakage, our vectorizer is fit strictly on X_train. The test set is only transformed."
    add_notes(slide6, notes6)
    slides_info.append(("Slide 6: TF-IDF Formulation", notes6))

    # ==========================================
    # SLIDE 7: THE 4 CLASSIFIERS THEORY
    # ==========================================
    slide7 = prs.slides.add_slide(blank_layout)
    add_header(slide7, "The Four Classifiers: Algorithmic Foundations")

    tb7 = slide7.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf7 = tb7.text_frame
    tf7.word_wrap = True

    p = tf7.paragraphs[0]
    p.text = "Theoretical Foundations of the Competing Models:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets7 = [
        "1. Decision Tree (CART): Non-parametric greedy tree induction using Gini Impurity: Gini(D) = 1 - ∑ p_i^2. Evaluates recursive orthogonal axis-aligned feature splits.",
        "2. K-Nearest Neighbours (KNN): Instance-based lazy learner using Cosine Distance: dist(u, v) = 1 - (u·v)/(||u|| ||v||). Classifies via majority vote across k=5 closest vectors.",
        "3. Linear Support Vector Machine (SVM): Maximum-margin classifier: min 1/2 ||w||^2 + C ∑ ξ_i subject to y_i(w^T x_i + b) >= 1 - ξ_i. Finds the optimal separating hyperplane in high dimensions.",
        "4. Multinomial Naive Bayes (MNB): Probabilistic generative model using Bayes Theorem: P(c|d) ∝ P(c) ∏ P(t_i|c)^f_i with Laplace smoothing: P(t|c) = (count(t, c) + α) / (∑ count(t', c) + α|V|)."
    ]
    for b in bullets7:
        p_b = tf7.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(13)
        p_b.space_before = Pt(10)
        p_b.font.color.rgb = COLOR_TEXT_DARK

    notes7 = "Here we introduce the four competing classifiers. We selected four distinct machine learning paradigms: " \
             "a tree-based model (Decision Tree), an instance-based lazy learner (KNN), a margin-based geometric classifier (Linear SVM), " \
             "and a probabilistic generative baseline (Multinomial Naive Bayes). All models are evaluated under identical conditions."
    add_notes(slide7, notes7)
    slides_info.append(("Slide 7: Classifiers Theory", notes7))

    # ==========================================
    # SLIDE 8: EVALUATION FRAMEWORK
    # ==========================================
    slide8 = prs.slides.add_slide(blank_layout)
    add_header(slide8, "Evaluation Framework & The Class Imbalance Challenge")

    tb8 = slide8.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf8 = tb8.text_frame
    tf8.word_wrap = True

    p = tf8.paragraphs[0]
    p.text = "Why Standard Accuracy is Insufficient:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets8 = [
        "• The Accuracy Trap: In an 86.6% Ham dataset, a dummy zero-rule model predicting Ham on 100% of cases achieves 86.60% Accuracy, but has 0% Spam Recall, failing completely.",
        "• Spam Precision (TP / (TP + FP)): Measures the percentage of flagged messages that are genuinely spam. Critical to minimize False Positives.",
        "• Spam Recall (TP / (TP + FN)): Measures the percentage of actual spam caught by the filter. Critical to prevent spam infiltration.",
        "• Spam F1-Score: The harmonic mean of Precision and Recall: F1 = 2 × (P × R) / (P + R). Represents the gold-standard single benchmark metric.",
        "• Experimental Protocol: Stratified 80/20 train-test split (4,448 training instances, 1,113 testing instances) with fixed random seed (42)."
    ]
    for b in bullets8:
        p_b = tf8.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(13)
        p_b.space_before = Pt(10)
        p_b.font.color.rgb = COLOR_TEXT_DARK

    notes8 = "On slide 8, we explain our evaluation framework and defend why Accuracy is misleading. " \
             "Because Ham constitutes 86.6% of the data, a model that classifies everything as Ham gets 86.6% accuracy while doing zero work. " \
             "Therefore, we focus on Spam Precision, Spam Recall, and their harmonic mean, the F1-Score."
    add_notes(slide8, notes8)
    slides_info.append(("Slide 8: Evaluation Framework", notes8))

    # ==========================================
    # SLIDE 9: EMPIRICAL BATTLE RESULTS
    # ==========================================
    slide9 = prs.slides.add_slide(blank_layout)
    add_header(slide9, "The Classification Battle: Actual Test Results")

    # Table of actual test results
    rows, cols = 5, 8
    left, top, width, height = Inches(0.8), Inches(1.6), Inches(11.7), Inches(3.2)
    table_shape = slide9.shapes.add_table(rows, cols, left, top, width, height)
    table = table_shape.table

    headers = ["Classifier", "Accuracy", "Spam Precision", "Spam Recall", "Spam F1", "TN", "FP", "FN"]
    for j, h in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_PRIMARY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(11)
            p.font.bold = True
            p.font.color.rgb = COLOR_WHITE
            p.alignment = PP_ALIGN.CENTER

    data_rows = [
        ["Decision Tree", "96.68%", "88.97%", "86.00%", "0.8746", "947", "16", "21"],
        ["KNN (k=5)", "97.12%", "98.36%", "80.00%", "0.8824", "961", "2", "30"],
        ["Linear SVM", "98.20%", "98.51%", "88.00%", "0.9296", "961", "2", "18"],
        ["Multinomial NB", "97.39%", "99.19%", "81.33%", "0.8938", "962", "1", "28"]
    ]

    for i, row in enumerate(data_rows):
        for j, val in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            if row[0] == "Linear SVM":
                cell.fill.fore_color.rgb = RGBColor(220, 245, 220)
            else:
                cell.fill.fore_color.rgb = COLOR_BG_LIGHT if (i % 2 == 0) else COLOR_WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(11)
                p.alignment = PP_ALIGN.CENTER

    tb9 = slide9.shapes.add_textbox(Inches(0.8), Inches(5.1), Inches(11.7), Inches(1.8))
    tf9 = tb9.text_frame
    tf9.word_wrap = True
    p9 = tf9.paragraphs[0]
    p9.text = "Key Empirical Takeaways (1,113 Test Messages: 963 Ham, 150 Spam):"
    p9.font.bold = True
    p9.font.size = Pt(13)
    p9.font.color.rgb = COLOR_PRIMARY

    bullets9 = [
        "• Linear SVM is the Overall Winner: Highest Accuracy (98.20%), highest Recall (88.00%), and highest F1-Score (0.9296).",
        "• Multinomial Naive Bayes is the Precision Winner: 99.19% Precision with only 1 False Positive out of 963 legitimate test messages.",
        "• Decision Tree has the Highest Error Rate: 16 False Positives due to noisy axis-aligned splits on sparse keywords."
    ]
    for b in bullets9:
        p_b = tf9.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(12)
        p_b.space_before = Pt(3)

    notes9 = "Here are our authentic empirical test results on 1,113 test samples. " \
             "Notice that Linear SVM is the clear overall victor with 98.20% accuracy and 0.9296 F1-score. " \
             "Multinomial Naive Bayes achieved an extraordinary 99.19% precision, producing only 1 False Positive out of 963 legitimate messages."
    add_notes(slide9, notes9)
    slides_info.append(("Slide 9: Empirical Results", notes9))

    # ==========================================
    # SLIDE 10: CONFUSION MATRIX ANALYSIS
    # ==========================================
    slide10 = prs.slides.add_slide(blank_layout)
    add_header(slide10, "Confusion Matrix Analysis: Cost of FP vs. FN")

    tb10 = slide10.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf10 = tb10.text_frame
    tf10.word_wrap = True

    p = tf10.paragraphs[0]
    p.text = "In-Depth Diagnostic Error Analysis (Test Set = 963 Ham + 150 Spam):"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets10 = [
        "• True Negatives (TN - Ham classified as Ham): Multinomial NB: 962/963 | SVM & KNN: 961/963 | Decision Tree: 947/963.",
        "• False Positives (FP - Catastrophic Error):",
        "    - Multinomial NB: Only 1 FP! (0.1% false alarm rate)",
        "    - Linear SVM & KNN: 2 FPs (0.2% false alarm rate)",
        "    - Decision Tree: 16 FPs (1.66% false alarm rate) - 8x to 16x higher user disruption!",
        "• False Negatives (FN - Spam leaked to Inbox):",
        "    - Linear SVM: 18 FNs (Caught 132/150 spam messages, 88.0% recall)",
        "    - Decision Tree: 21 FNs (Caught 129/150 spam messages, 86.0% recall)",
        "    - Multinomial NB: 28 FNs (Caught 122/150 spam messages, 81.33% recall)",
        "    - KNN: 30 FNs (Caught 120/150 spam messages, 80.0% recall)",
        "• Architectural Verdict: Linear SVM offers the most balanced real-world deployment profile, catching 88% of spam while protecting 99.8% of legitimate emails."
    ]
    for b in bullets10:
        p_b = tf10.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(13)
        p_b.space_before = Pt(8)
        p_b.font.color.rgb = COLOR_TEXT_DARK

    notes10 = "Slide 10 breaks down the confusion matrix errors. In spam detection, False Positives are catastrophic. " \
              "Decision Tree committed 16 False Positives, meaning 16 real messages were dropped. " \
              "In contrast, Naive Bayes committed only 1 False Positive, and SVM committed only 2 False Positives while maintaining 88% recall."
    add_notes(slide10, notes10)
    slides_info.append(("Slide 10: Confusion Matrix Deep Dive", notes10))

    # ==========================================
    # SLIDE 11: LATENCY & SPEED BENCHMARK
    # ==========================================
    slide11 = prs.slides.add_slide(blank_layout)
    add_header(slide11, "Computational Complexity & Latency Benchmarks")

    tb11 = slide11.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf11 = tb11.text_frame
    tf11.word_wrap = True

    p = tf11.paragraphs[0]
    p.text = "Training Time vs Inference Latency per Sample:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets11 = [
        "• Multinomial Naive Bayes: Ultra-Low Latency Champion",
        "    - Training Time: 0.0017 seconds | Per-sample inference latency: ~0.0002 ms",
        "    - Complexity: O(N × |V|) linear single-pass feature count accumulation.",
        "• Linear Support Vector Machine (SVM):",
        "    - Training Time: 1.4489 seconds | Per-sample inference latency: 0.046 ms",
        "    - Complexity: Quadratic in sample count during quadratic programming optimization, but inference is a fast inner product w^T x.",
        "• K-Nearest Neighbours (KNN): The Lazy Learning Bottleneck",
        "    - Training Time: 0.0009 seconds (zero eager training).",
        "    - Inference Time: 0.0521 seconds batch (~0.047 ms/sample). Latency scales linearly with training set size O(N × d) per query!",
        "• Decision Tree: Fast training (0.1176s) and fast traversal O(depth), but vulnerable to high variance."
    ]
    for b in bullets11:
        p_b = tf11.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(13)
        p_b.space_before = Pt(8)
        p_b.font.color.rgb = COLOR_TEXT_DARK

    notes11 = "On slide 11, we examine runtime computational complexity. Naive Bayes is blazing fast, training in under 2 milliseconds and predicting " \
              "in fractions of a microsecond. In contrast, while KNN trains in zero time, its inference requires computing cosine distance against all " \
              "4,448 training vectors, making it impractical for high-throughput telecom gateways."
    add_notes(slide11, notes11)
    slides_info.append(("Slide 11: Computational Latency", notes11))

    # ==========================================
    # SLIDE 12: WHAT-IF EXPERIMENTS
    # ==========================================
    slide12 = prs.slides.add_slide(blank_layout)
    add_header(slide12, "Empirical What-If Sensitivity Experiments")

    tb12 = slide12.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf12 = tb12.text_frame
    tf12.word_wrap = True

    p = tf12.paragraphs[0]
    p.text = "Investigating Hyperparameter Dynamics & Feature Spaces:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets12 = [
        "1. What-If A: KNN k-Sensitivity across [1, 3, 5, 7, 9, 11, 15, 21, 31]:",
        "    - Finding: As k increases from 3 to 31, Spam Recall collapses from 83.33% down to 68.00%!",
        "    - Explanation: High k causes majority Ham instances to invade the local neighborhood, drowning out minority Spam votes.",
        "2. What-If B: TF-IDF Vocabulary Capacity (max_features from 200 to Unlimited 30,508):",
        "    - Finding: At 200 features, models underfit (F1 0.87-0.89). Optimal performance occurs between 1,000-2,000 features (SVM F1: 0.9412).",
        "    - Explanation: At 30,500 features, Naive Bayes recall collapses to 70.0% due to zero-frequency tail distortion, whereas SVM remains stable.",
        "3. What-If C: N-Gram Range Effect (Unigrams vs Bigrams vs Trigrams):",
        "    - Finding: Bigrams-only (2,2) degrades F1 to 0.8289 due to extreme text sparsity. Combined Unigram+Bigram (1,2) achieves peak precision (98.51%)."
    ]
    for b in bullets12:
        p_b = tf12.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(12)
        p_b.space_before = Pt(6)
        p_b.font.color.rgb = COLOR_TEXT_DARK

    notes12 = "Here we present our What-If sensitivity experiments. We conducted three experiments: " \
              "First, we varied KNN neighbors from 1 to 31 and discovered that large k destroys Spam Recall due to class imbalance. " \
              "Second, we tested TF-IDF vocabulary limits, showing that 2,000 features provides the sweet spot. " \
              "Third, combining Unigrams with Bigrams provides the best discriminative power for phrases like 'free prize'."
    add_notes(slide12, notes12)
    slides_info.append(("Slide 12: What-If Experiments", notes12))

    # ==========================================
    # SLIDE 13: CONCLUSIONS & LIMITATIONS
    # ==========================================
    slide13 = prs.slides.add_slide(blank_layout)
    add_header(slide13, "Conclusions, Real-World Limitations & Defence")

    tb13 = slide13.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf13 = tb13.text_frame
    tf13.word_wrap = True

    p = tf13.paragraphs[0]
    p.text = "Summary of Defensible Findings & Constraints:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets13 = [
        "• Champion Model: Linear Support Vector Machine (SVM) is the most defensible architecture, balancing 98.20% accuracy with 0.9296 F1 and low False Positive risk.",
        "• Production Alternative: Multinomial Naive Bayes is ideal for high-throughput edge devices requiring sub-millisecond inference and near-zero false alarms (99.19% precision).",
        "• Real-World Constraints & Security Disclaimer:",
        "    - Statistical models are not immune to adversarial evasions (e.g., zero-width spaces, homoglyph character substitutions like Cyrillic 'a').",
        "    - SMS spam filters must be coupled with rate limiters, reputation databases, and sender verification protocols in actual production.",
        "• Future Work: Transformer-based embeddings (e.g., DistilBERT/RoBERTa) and active learning feedback loops."
    ]
    for b in bullets13:
        p_b = tf13.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(13)
        p_b.space_before = Pt(10)
        p_b.font.color.rgb = COLOR_TEXT_DARK

    notes13 = "In conclusion, Linear SVM is our definitively defended champion, delivering 98.20% accuracy and 0.9296 F1. " \
              "Multinomial Naive Bayes is our speed and precision runner-up. " \
              "Importantly, we note that statistical classifiers are threat indicators and must be complemented by cybersecurity layers in production."
    add_notes(slide13, notes13)
    slides_info.append(("Slide 13: Conclusions & Limitations", notes13))

    # ==========================================
    # SLIDE 14: REFERENCES & VIVA Q&A
    # ==========================================
    slide14 = prs.slides.add_slide(blank_layout)
    add_header(slide14, "References & Technical Defence Q&A")

    tb14 = slide14.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
    tf14 = tb14.text_frame
    tf14.word_wrap = True

    p = tf14.paragraphs[0]
    p.text = "Academic References & Acknowledgments:"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = COLOR_PRIMARY

    bullets14 = [
        "1. Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). 'Contributions to the study of SMS spam filtering: New collection and results.' Information Sciences, 181(11), 2422-2438.",
        "2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.",
        "3. Joachims, T. (1998). 'Text categorization with Support Vector Machines: Learning with many relevant features.' ECML.",
        "4. Pedregosa, F., et al. (2011). 'Scikit-learn: Machine Learning in Python.' JMLR, 12, 2825-2830.",
        "\nStudent: Mohammed Sameer P (RA2411026050199)",
        "Course: 21CSE377T – Information Retrieval | SRM IST Tiruchirappalli",
        "\nThank you. I am now open to technical questions and defence."
    ]
    for b in bullets14:
        p_b = tf14.add_paragraph()
        p_b.text = b
        p_b.font.size = Pt(12.5)
        p_b.space_before = Pt(6)
        p_b.font.color.rgb = COLOR_TEXT_DARK

    notes14 = "Thank you very much for your time and evaluation. I am now ready to defend my project and answer any technical questions."
    add_notes(slide14, notes14)
    slides_info.append(("Slide 14: References & Defense", notes14))

    # Save presentation
    prs.save(OUTPUT_PPTX)
    print(f"[+] Saved PowerPoint presentation to: {OUTPUT_PPTX}")

    # Write Markdown Outline
    with open(OUTLINE_MD, "w", encoding="utf-8") as f:
        f.write("# 14-SLIDE PRESENTATION OUTLINE & SPEAKER NOTES\n")
        f.write("## Course: 21CSE377T - Information Retrieval | CLA-2 Task 2\n")
        f.write("### Student: Mohammed Sameer P (RA2411026050199) | SRM IST Tiruchirappalli\n\n")
        f.write("---\n\n")
        for title, note in slides_info:
            f.write(f"### {title}\n")
            f.write(f"**Speaker Script / Notes:**\n> {note}\n\n---\n\n")

    print(f"[+] Saved presentation outline and speaker notes to: {OUTLINE_MD}")


if __name__ == "__main__":
    create_deck()

