# SMS Spam Collection Dataset Reference
## 21CSE377T – Information Retrieval | CLA-2 Task 2
**Student:** Mohammed Sameer P (RA2411026050199)  
**Institution:** SRM Institute of Science and Technology, Tiruchirappalli  

---

### 1. Dataset Summary & Citation
* **Dataset Name:** SMS Spam Collection v.1
* **Source:** UCI Machine Learning Repository
* **Direct Archive:** `https://archive.ics.uci.edu/dataset/228/sms+spam+collection`
* **Authors:** Tiago A. Almeida, José María Gómez Hidalgo, Akebo Yamakami
* **Primary Reference:**
  > Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam filtering: New collection and results. *Information Sciences*, 181(11), 2422-2438.

---

### 2. Composition & Data Origins
The collection consolidates 5,574 authentic English SMS messages gathered from four legitimate sources:
1. **NUS SMS Corpus:** 3,375 legitimate SMS messages gathered for research at the National University of Singapore.
2. **Grumbletext:** 425 spam messages collected from the UK consumer grievance and mobile advocacy website.
3. **Caroline Tag's PhD Thesis:** 450 personal legitimate messages collected with user consent for linguistic analysis.
4. **SMS Spam Corpus v.0.1 Big:** 1,002 legitimate and 322 spam messages from prior academic baselines.

---

### 3. Class Distribution & Quality Metrics
* **Total Instances:** 5,574
* **Ham (Legitimate):** 4,827 (86.60%)
* **Spam (Unsolicited):** 747 (13.40%)
* **Class Imbalance Ratio:** ~6.46 to 1
* **Missing or Corrupted Values:** 0
* **Data Format:** Tab-separated text file (`label \t message \n`)

---

### 4. SMS vs Email Domain Clarification
* **SMS Constraints:** 160-character length limit per SMS septet, heavy phonetic contractions ('u', 'ur', 'txt', 'winr'), and lack of transport headers.
* **Email Attributes:** Unlimited document length, HTML styling, MIME boundaries, DKIM/SPF domain verification, and client IP headers.
* **Justification:** SMS spam classification is a stricter test of NLP Information Retrieval because classifiers must make accurate decisions relying purely on short, noisy lexical content.

