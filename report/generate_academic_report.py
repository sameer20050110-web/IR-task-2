"""
generate_academic_report.py
---------------------------
Entry point to regenerate academic report artifacts.
"""

from generate_docx_report import generate_docx

if __name__ == "__main__":
    generate_docx()
    print("[+] Academic report generation complete.")
