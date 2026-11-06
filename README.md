# PPIIA: Public Policy Insight & Impact Analyzer

Load a bill (PDF, PDF link, bill web page or pasted text) and get topics, a plain-language summary, an industry
impact report, a chronology and dependency graph, a short/medium/long-term impact assessment, and pros/cons.

## Files

| File | What it is |
|---|---|
| `ppiia_core.py` | All the logic (ingestion, NLP preprocessing, topics, LLM calls, graphs, report). One source of truth. |
| `ppiia_pipeline.ipynb` | Step-by-step notebook with every cell explained. Same logic as the core module. |
| `app.py` | Streamlit dashboard (user interface only). |
| `langflow_design.md` | The pipeline designed as a Langflow workflow. |
| `requirements.txt`, `secrets.toml.example` | Dependencies and the secrets template. |

## Requirement coverage

| Required topic | Where it is built |
|---|---|
| Web scraping / document ingestion (PDF, HTML bills from Sansad portals) | `fetch_url`, `find_pdf_links`, `pdf_to_pages`, `html_to_text`; notebook cells 3 to 4; dashboard "From a URL" |
| NLP-based text preprocessing | `clean_pages`, `split_sections`, `classify_clause`, `modal_counts`, `find_acts`, `find_dates`, `find_deadlines`; notebook cells 5 to 10 |
| Topic extraction & legislative context mapping | `extract_topics` (TF-IDF + NMF), then the LLM `context` task; notebook cells 11, 12, 19 |
| Chronology & dependency graph generation | `section_dependencies`, `build_graph`, `section_graph`, LLM `chronology` task; notebook cells 22 to 23 |
| Domain / industry impact analysis | LLM `industry` task with a fixed sector list; notebook cell 21 |
| Short / medium / long-term impact assessment | LLM `impact` task; notebook cell 24 |
| LLM-powered summarization | map-reduce `build_digest` for long bills, `summary` and `proscons` tasks; notebook cells 17 to 20, 25 |
| Streamlit dashboard | `app.py` |
| Langflow workflow design | `langflow_design.md` |
| Tags: Transformers, Text Classification | rule-based clause classifier plus an optional zero-shot classifier and summarizer (notebook cell 27) |

## Get a free Gemini API key (no card)

1. Go to https://aistudio.google.com/apikey and sign in with a Google account.
2. Create an API key and copy it.
3. Create a file named `.env` next to the notebook with `LLM_API_KEY=your_key` (add `.env` to `.gitignore`).

Google renames models often. The default is `gemini-3.8-flash`; if it stops working, run notebook cell 16 to list the
models your key can use and set `LLM_MODEL` (app) or `MODEL` (notebook) to one of them.

## Run it

**Notebook (VS Code or Colab):** open `ppiia_pipeline.ipynb` and run the cells from top to bottom. Cells 2 to 12 need no
key. In Colab, upload `ppiia_pipeline.ipynb` only (it is self-contained) and save your key as a Colab secret named `LLM_API_KEY`.

**Dashboard locally:**
```
pip install -r requirements.txt
streamlit run app.py
```

## Deploy free on Streamlit Community Cloud

1. Put `app.py`, `ppiia_core.py` and `requirements.txt` in a GitHub repository (never commit your key).
2. On https://share.streamlit.io create an app from that repo with `app.py` as the main file.
3. In the app's Secrets settings paste the contents of `secrets.toml.example` with your real key.

## Known limits

- **Free-tier limits:** Gemini's free tier has per-minute and per-day request caps that Google does not publish in its
  docs (see https://aistudio.google.com/rate-limit). The code pauses between calls and retries, so a run takes a minute or two.
- **Sansad portals:** the bill listing pages may load their content with JavaScript and may block cloud servers. If
  fetching a link fails, download the PDF in your browser and use "Upload PDF".
- **Scanned PDFs** contain no text and need OCR first.
- **Graphs and timeline** use only references that appear in the bill text. Wider legislative history is not inferred.
- The impact assessment is a basic AI inference, not a forecast. All output is AI-generated and not legal advice.
