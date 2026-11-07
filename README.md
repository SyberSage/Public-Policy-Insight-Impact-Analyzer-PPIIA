# 📜 Public Policy Insight & Impact Analyzer (PPIIA)

An NLP and LLM-powered web application that turns long **government bills** into plain-language summaries, industry impact reports, timelines, dependency graphs and short, medium and long-term impact assessments.

The application is built with **Python and Streamlit**. You load a bill (a PDF, a web link or pasted text), and the app cleans and structures the text with classic NLP, extracts topics, and then uses a large language model (Google Gemini) to produce a readable, single-page analysis report with a side navigation menu.

## 🚀 Live Demo

**Streamlit App:** https://your-app-name.streamlit.app/

## 📌 Project Overview

Bills are long, technical and hard for ordinary readers to follow. **PPIIA** answers the questions a citizen, student or analyst actually asks: *What does this bill mean for me? Which industries gain or lose? What does it change, and when? What happens in the next 1, 5 and 10 years?*

The project covers the complete pipeline:

* 🌐 Document Ingestion (PDF upload, PDF or HTML links, pasted text)
* 🧹 NLP Text Preprocessing & Section Structuring
* 🔑 Topic Extraction & Legislative Context Mapping
* 🕸️ Chronology, Knowledge Graph & Section Dependency Graph
* 🏭 Industry Impact Analysis
* 📈 Short, Medium and Long-Term Impact Assessment
* 📝 LLM-Powered Summarization
* 🖥️ Interactive Streamlit Dashboard with a downloadable report

## ✨ Features

* Four ways to load a bill: **upload a PDF**, **fetch from a URL** (a direct PDF link, or a bill page from which PDF links are listed), **paste text**, or try a built-in **sample bill**
* **Plain-language summary** shown as simple bullet points: "What this means for citizens"
* **Industry Impact Report**: sectors that benefit, sectors facing restrictions and new business opportunities, classified against a fixed list of sectors
* **Timeline of Bill History** shown as dated event cards
* **Knowledge graph** linking the bill to the Acts, committees and authorities it mentions, plus a **dependency graph** showing which sections refer to other sections
* **Impact Assessment** across three horizons: short term (0–1 year), medium term (1–5 years) and long term (more than 5 years)
* **Positives and Negatives** in simple bullet points
* **Topics & legislative context**: bill type, ministry, policy area, named topics, stakeholders and related laws
* **Document overview**: word and section counts, clause-type chart, keywords, topic groups, dates and time limits
* The bill's **title is detected automatically** from the text, or read by the model when the text has none
* **Single-page report** with a left sidebar menu that jumps to each section and highlights the one you are reading
* **Markdown report download** with all sections
* Handles **long bills** by condensing them in stages before analysis
* Resilient to free-tier limits: paced requests, automatic retries and fallback models when the main model is busy
* API key kept out of the code (Streamlit secrets or a local `.env` file)

## 🧠 AI & NLP Techniques

PPIIA combines classic, explainable NLP with an LLM, and gives the LLM exact facts found by rules as hints.

| Stage | Technique |
| ----- | --------- |
| Text cleaning | Unicode normalization, removal of repeated headers/footers and page numbers, hyphenation repair |
| Structuring | Regex-based section splitting (numbered clauses, with a paragraph fallback) |
| Legal NLP | Pattern matching for referenced Acts, dates, time limits and section cross-references; counts of obligations (*shall*), prohibitions (*shall not*) and permissions (*may*) |
| Text classification | Rule-based clause-type classifier (Definition, Penalty, Finance, Power / authority and more); LLM classification of sectors against a fixed list |
| Topic extraction | TF-IDF keywords and NMF topic modelling over the bill's sections |
| Summarization | Map-reduce condensing for long bills, then LLM summaries |
| Structured extraction | The LLM answers in JSON that is validated with Pydantic schemas, with one automatic retry on invalid output |
| Knowledge graphs | NetworkX graphs of the bill, Acts, committees, authorities and section dependencies, drawn with Graphviz |

### Language Model

| Component | Details |
| --------- | ------- |
| Model provider | Google Gemini API through its OpenAI-compatible endpoint |
| Default model | `gemini-3.8-flash` (free-tier Flash model; change it with `LLM_MODEL`) |
| Client | OpenAI Python SDK pointed at the Gemini endpoint |

No model is trained or stored in this repository. All language-model work is done by the API.

## 🔄 Workflow

```text
Bill (PDF / URL / HTML page / pasted text)
   ↓
Document Ingestion
   ↓
NLP Preprocessing (cleaning, sections, clause types, Acts, dates, dependencies)
   ↓
Topic Extraction (TF-IDF + NMF)
   ↓
Condense Long Bills (map-reduce)
   ↓
LLM Analysis (context, summary, industry impact, timeline & graph, impact horizons, pros & cons)
   ↓
Schema Validation (Pydantic)
   ↓
Streamlit Dashboard + Markdown Report
```

## 🛠️ Technologies Used

### Programming

* Python

### Web Application

* Streamlit
* Graphviz (graph drawing through Streamlit)

### Document Ingestion

* pypdf (PDF text extraction)
* Requests
* Beautiful Soup 4 (HTML pages)

### Natural Language Processing & Machine Learning

* Regular expressions (legal pattern extraction)
* Scikit-learn (TF-IDF, NMF)
* NumPy
* Pandas

### Graphs & Validation

* NetworkX
* Pydantic

### Large Language Model

* Google Gemini API
* OpenAI Python SDK (used as an OpenAI-compatible client)

### Development Tools

* Jupyter Notebook
* VS Code
* GitHub

## 📂 Project Structure

```text
PPIIA/
│
├── app.py                    # Streamlit dashboard (user interface only)
├── ppiia_core.py             # All logic: ingestion, NLP, topics, LLM calls, graphs, report
├── ppiia_pipeline.ipynb      # Step-by-step notebook with each cell explained
│
├── requirements.txt
├── .gitignore
│
├── .streamlit/
│   └── secrets.toml          # your API key (local only, never uploaded)
├── .env                      # optional alternative to secrets.toml (local only)
│
└── README.md
```

`app.py` imports `ppiia_core.py`, so both files must be in the same folder.

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

### 2. Navigate to the project directory

```bash
cd your-repository-name
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Mac/Linux:**

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Add your API key

See [API Key Setup](#-api-key-setup) below.

### 7. Run the Streamlit application

```bash
streamlit run app.py
```

If your terminal says `streamlit` is not recognised, run it through Python instead:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 🔑 API Key Setup

The app needs a free **Gemini API key** from https://aistudio.google.com/apikey (no payment card is required).

Save it **once** in either of these local files, and the app loads it automatically. Never commit either file to GitHub.

**Option A: `.streamlit/secrets.toml`**

```toml
LLM_API_KEY = "your-gemini-api-key"
```

**Option B: `.env`** (in the same folder as `app.py`, no quotes needed)

```text
LLM_API_KEY=your-gemini-api-key
```

Add both files to `.gitignore`:

```text
.streamlit/secrets.toml
.env
```

If no key is found, the sidebar shows a password box where you can paste one for the current session.

### Optional settings

| Setting | Purpose | Default |
| ------- | ------- | ------- |
| `LLM_API_KEY` | Your Gemini API key | none (required) |
| `LLM_MODEL` | Model name to use | `gemini-3.8-flash` |
| `LLM_BASE_URL` | API endpoint | Gemini's OpenAI-compatible endpoint |
| `LLM_MIN_GAP` | Seconds to pause between model calls (environment variable) | `4` |

Google renames its models often. If the default stops working, set `LLM_MODEL` to a model listed in Google AI Studio. When the main model is busy or over quota, the app automatically retries and then switches to fallback models.

## ▶️ How to Use

1. Choose how to load the bill: upload a PDF, give a link, paste text or use the sample.
2. Optionally type the bill's name; otherwise it is detected from the text.
3. Click **Analyze bill**. A short bill takes about one to three minutes on the free tier, and a long bill takes longer. The app shows an estimate and live progress.
4. Read the report as one continuous page, and use the sidebar menu to jump between sections.
5. Download the full report as a Markdown file from the sidebar.
6. Use **Analyze a different bill** at the top of the report to start again.

## 📦 Requirements

Example `requirements.txt`:

```text
streamlit>=1.40
openai>=1.50
pydantic>=2.7
pypdf>=4.0
networkx>=3.2
requests>=2.31
beautifulsoup4>=4.12
scikit-learn>=1.4
pandas>=2.0
```

To run the notebook as well, also install:

```text
jupyter
matplotlib
python-dotenv
```

## 🌐 Deployment

The application can be deployed using **Streamlit Community Cloud**.

Basic deployment steps:

1. Push `app.py`, `ppiia_core.py` and `requirements.txt` to GitHub (do **not** upload `secrets.toml` or `.env`).
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Select this repository.
5. Select `app.py` as the main file.
6. Open **Advanced settings → Secrets** and paste:

```toml
LLM_API_KEY = "your-gemini-api-key"
```

7. Deploy the application.

Visitors never see the key. It stays on the server, and their requests count towards your free quota.

## ⚠️ Disclaimer

This project is intended **for educational and demonstration purposes only**.

* All analysis is generated by an AI model and may contain mistakes. It is **not legal advice** and should not be used as the sole basis for real-world decisions.
* The **impact assessment is a basic inference** from the bill's provisions, not a forecast.
* The **timeline and graphs use only references that appear in the bill text.** Wider legislative history is not inferred.
* The text of each analysed bill is **sent to the Gemini API**. Do not analyse confidential documents. On the free tier, Google may use submitted content to improve its products.
* **Scanned PDFs** contain no text and need OCR first.
* Government portals may load pages with JavaScript or block automated downloads. If a link cannot be fetched, download the PDF in your browser and upload it instead.
* Free-tier API limits apply, and very long bills are analysed only up to a size limit (the app tells you when text is cut off).
* Section detection works best on bills with numbered sections.

## 👨‍💻 Author

Yoga Prabhu

Data Science | Machine Learning | Python

Portfolio: https://sybersage.github.io/Portfolio/

GitHub: https://github.com/your-username

LinkedIn: https://www.linkedin.com/in/yoga-prabhu

---

⭐ If you found this project useful, consider giving the repository a star!
