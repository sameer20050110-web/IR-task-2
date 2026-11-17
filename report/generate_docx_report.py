"""
generate_docx_report.py
-----------------------
Builds ACADEMIC_REPORT.docx with formatting, tables, and colors.
"""

import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCX_PATH = os.path.join(BASE_DIR, "report", "ACADEMIC_REPORT.docx")

def set_cell_background(cell, fill_hex):
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = parse_xml(
        f'<w:tcMar {nsdecls("w")}>'
        f'<w:top w:w="{top}" w:type="dxa"/>'
        f'<w:bottom w:w="{bottom}" w:type="dxa"/>'
        f'<w:left w:w="{left}" w:type="dxa"/>'
        f'<w:right w:w="{right}" w:type="dxa"/>'
        f'</w:tcMar>'
    )
    tcPr.append(tcMar)

def generate_docx():
    doc = Document()
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # Styles
    doc.styles['Normal'].font.name = 'Calibri'
    doc.styles['Normal'].font.size = Pt(11)

    # Title
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_inst = p_inst.add_run("SRM INSTITUTE OF SCIENCE AND TECHNOLOGY\n")
    r_inst.bold = True
    r_inst.font.size = Pt(14)
    r_inst.font.color.rgb = RGBColor(10, 37, 64)
    r_dept = p_inst.add_run("DEPARTMENT OF COMPUTER SCIENCE & ENGINEERING — TIRUCHIRAPPALLI CAMPUS\n")
    r_dept.font.size = Pt(11)
    r_dept.font.color.rgb = RGBColor(70, 80, 95)

    doc.add_paragraph()

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_ass = p_title.add_run("CONTINUOUS LEARNING ASSESSMENT – II (CLA-2 TASK 2)\n")
    r_ass.bold = True
    r_ass.font.size = Pt(13)
    r_ass.font.color.rgb = RGBColor(0, 119, 182)
    
    r_proj = p_title.add_run("EMAIL AND SMS SPAM DETECTION USING MACHINE LEARNING\n")
    r_proj.bold = True
    r_proj.font.size = Pt(20)
    r_proj.font.color.rgb = RGBColor(10, 37, 64)

    r_sub = p_title.add_run("TEXT CLASSIFICATION BATTLE: COMPARE, ANALYZE AND DEFEND\nCourse: 21CSE377T – Information Retrieval | Maximum Marks: 30\n")
    r_sub.font.size = Pt(12)
    r_sub.font.color.rgb = RGBColor(100, 110, 125)

    doc.add_paragraph()

    # Credentials Box
    tbl_cred = doc.add_table(rows=5, cols=2)
    tbl_cred.alignment = WD_TABLE_ALIGNMENT.CENTER
    cred_data = [
        ("Student Name:", "Mohammed Sameer P"),
        ("Register Number:", "RA2411026050199"),
        ("Course Code & Title:", "21CSE377T – Information Retrieval"),
        ("Degree & Department:", "B.Tech Computer Science and Engineering"),
        ("Institution:", "SRM Institute of Science and Technology, Tiruchirappalli")
    ]
    for idx, (label, val) in enumerate(cred_data):
        row = tbl_cred.rows[idx]
        c0, c1 = row.cells[0], row.cells[1]
        c0.text = label
        c1.text = val
        c0.paragraphs[0].runs[0].bold = True
        set_cell_background(c0, "F0F4F8")
        set_cell_background(c1, "FAFCFF")
        set_cell_margins(c0, 60, 60, 100, 100)
        set_cell_margins(c1, 60, 60, 100, 100)

    doc.add_page_break()

    def add_h1(text):
        h = doc.add_heading(text, level=1)
        h.runs[0].font.color.rgb = RGBColor(10, 37, 64)
        h.runs[0].font.size = Pt(15)
        h.runs[0].bold = True

    add_h1("ABSTRACT")
    doc.add_paragraph(
        "Unsolicited electronic messaging, encompassing email spam and mobile Short Message Service (SMS) spam, "
        "imposes massive operational costs on communication infrastructure, threatens privacy, and exposes consumers to "
        "phishing and financial fraud. This project designs, evaluates, and defends a comprehensive Information Retrieval (IR) "
        "and Supervised Text Classification framework to distinguish legitimate messages (Ham) from unsolicited intrusions (Spam). "
        "Using the gold-standard UCI SMS Spam Collection benchmark (5,574 authentic messages), we apply a multi-tier Natural Language "
        "Processing (NLP) pipeline combined with Term Frequency-Inverse Document Frequency (TF-IDF) feature engineering fitted with strict data "
        "leakage prevention. We orchestrate an empirical battle across four machine learning algorithms: Decision Tree (CART), "
        "K-Nearest Neighbours (KNN), Linear Support Vector Machine (SVM), and Multinomial Naive Bayes (MNB) under an identical stratified 80/20 "
        "train-test split. The empirical results demonstrate that Linear SVM is the definitive champion, achieving 98.20% Accuracy, 0.9296 Spam F1-score, "
        "and an 88.00% Spam Recall with exceptional False Positive control (only 2 out of 963 legitimate messages). Multinomial Naive Bayes achieved "
        "the highest Precision (99.19%) with sub-microsecond latency. Extensive What-If sensitivity experiments confirm that KNN recall collapses as k expands "
        "due to class imbalance, while TF-IDF feature truncation between 1,000–2,000 n-grams optimizes discriminative stability."
    )

    add_h1("1. INTRODUCTION & PROBLEM FORMULATION")
    doc.add_paragraph(
        "Information Retrieval (IR) has transitioned from traditional keyword-matching indexing toward semantic categorization and predictive text classification. "
        "Unsolicited mobile messaging (SMS spam) has emerged as an especially perilous threat vector due to mobile devices' personal nature, high open rates "
        "(often exceeding 95% within 3 minutes), and the severe lack of built-in transport security metadata compared to desktop email protocols."
    )
    doc.add_paragraph(
        "Text classification in mobile messaging operates under stringent constraints: (1) SMS messages are bounded by telecom protocols to 160 characters, "
        "yielding an average document length of merely 15 to 20 words; (2) Users and malicious adversaries employ phonetic abbreviations ('u', 'ur', 'txt'), "
        "non-standard orthography, and character substitutions; and (3) The vast majority of mobile traffic consists of legitimate interpersonal conversations, "
        "with spam representing a minority fraction (~13.4%). This investigation systematically addresses these challenges."
    )

    add_h1("2. DATASET BENCHMARK & QUALITY VERIFICATION")
    doc.add_paragraph(
        "The empirical investigation is conducted on the SMS Spam Collection v.1, sourced from the UCI Machine Learning Repository. "
        "The corpus contains 5,574 authentic messages (4,827 Ham and 747 Spam) collected from the National University of Singapore (NUS) corpus, "
        "Grumbletext UK consumer complaints, Caroline Tag's doctoral mobile corpus, and prior academic baselines. No synthetic data was fabricated."
    )

    add_h1("3. THE 4-CLASSIFIER BATTLE RESULTS")
    doc.add_paragraph(
        "All four classifiers were evaluated on an identical holdout test set of 1,113 messages (963 Ham and 150 Spam). "
        "The vectorizer and models were fit strictly on training partitions to guarantee zero data leakage."
    )

    # Metrics Table
    tbl = doc.add_table(rows=5, cols=8)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    headers = ["Model", "Accuracy", "Spam Prec", "Spam Rec", "Spam F1", "TN", "FP", "FN"]
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.text = h
        cell.paragraphs[0].runs[0].bold = True
        cell.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
        set_cell_background(cell, "0A2540")
        set_cell_margins(cell, 80, 80, 100, 100)

    rows_data = [
        ["Linear SVM", "98.20%", "98.51%", "88.00%", "0.9296", "961", "2", "18"],
        ["Multinomial NB", "97.39%", "99.19%", "81.33%", "0.8938", "962", "1", "28"],
        ["KNN (k=5)", "97.12%", "98.36%", "80.00%", "0.8824", "961", "2", "30"],
        ["Decision Tree", "96.68%", "88.97%", "86.00%", "0.8746", "947", "16", "21"]
    ]
    for i, row in enumerate(rows_data):
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.text = val
            bg_col = "E8F5E9" if row[0] == "Linear SVM" else ("F8F9FA" if i % 2 == 0 else "FFFFFF")
            set_cell_background(cell, bg_col)
            set_cell_margins(cell, 60, 60, 80, 80)

    doc.add_paragraph()

    add_h1("4. DIAGNOSTIC ERROR & CONFUSION MATRIX ANALYSIS")
    doc.add_paragraph(
        "In spam classification, False Positives (FP) are catastrophic because legitimate messages (e.g., bank OTPs, job offers) are missed by the user. "
        "Decision Tree exhibited the worst error profile with 16 False Positives (1.66% false alarm rate). In sharp contrast, Multinomial Naive Bayes "
        "achieved the lowest False Positive rate (only 1 FP out of 963 messages), while Linear SVM achieved an exceptional balance of only 2 False Positives "
        "and the lowest False Negative rate (catching 132 out of 150 spam messages, 88.0% recall)."
    )

    add_h1("5. WHAT-IF EXPERIMENTAL FINDINGS")
    doc.add_paragraph(
        "1. KNN k-Sensitivity: Expanding k from 3 to 31 caused Spam Recall to collapse from 83.33% down to 68.00% because majority Ham instances "
        "overwhelm local neighborhood voting in an imbalanced 86.6% Ham distribution.\n"
        "2. TF-IDF Vocabulary Capacity: 200 features led to underfitting (F1 ~0.87), while 1,000–2,000 features reached optimal balance (SVM F1: 0.9412). "
        "At 30,500 features, Naive Bayes recall degraded to 70.0% due to sparse tail noise, whereas SVM remained resilient due to L2 regularization.\n"
        "3. N-gram Schemes: Unigram+Bigram (1,2) achieved peak precision (98.51%) by capturing critical phishing phrases ('free cash', 'claim prize')."
    )

    add_h1("6. CONCLUSION & DEFENCE VERDICT")
    doc.add_paragraph(
        "Linear Support Vector Machine (SVM) is the definitively defended champion model for SMS spam detection, delivering 98.20% accuracy, "
        "0.9296 F1-score, and optimal False Positive protection. Multinomial Naive Bayes serves as an ultra-fast precision alternative (0.2 microseconds latency, 99.19% precision). "
        "Statistical machine learning filters provide robust threat indicators, though production deployments must be reinforced with cybersecurity defense-in-depth controls."
    )

    add_h1("7. REFERENCES")
    refs = [
        "1. Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). 'Contributions to the study of SMS spam filtering: New collection and results.' Information Sciences, 181(11), 2422-2438.",
        "2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). Introduction to Information Retrieval. Cambridge University Press.",
        "3. Joachims, T. (1998). 'Text categorization with Support Vector Machines: Learning with many relevant features.' ECML.",
        "4. Pedregosa, F., et al. (2011). 'Scikit-learn: Machine Learning in Python.' JMLR, 12, 2825-2830."
    ]
    for r in refs:
        doc.add_paragraph(r)

    doc.save(DOCX_PATH)
    print(f"[+] Saved Word Academic Report to: {DOCX_PATH}")

if __name__ == "__main__":
    generate_docx()

