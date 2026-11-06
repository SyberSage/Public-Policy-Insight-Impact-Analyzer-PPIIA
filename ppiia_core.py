"""PPIIA core logic: ingestion -> NLP preprocessing -> topics -> LLM analysis -> graphs -> report.

This file is the single source of truth. The Streamlit app imports it, and the notebook cells are
generated from the "# ==== stage ====" blocks below, so both always behave the same way.
"""

# ==== settings ====
import io
import json
import math
import os
import re
import textwrap
import time
import traceback
import unicodedata
from collections import Counter
from typing import List, Optional
from urllib.parse import urljoin

import networkx as nx
import numpy as np
import requests
from bs4 import BeautifulSoup
from openai import (APIConnectionError, APITimeoutError, InternalServerError, NotFoundError, OpenAI,
                    OpenAIError, RateLimitError)
from pydantic import BaseModel, Field, ValidationError, field_validator
from pypdf import PdfReader
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"  # Gemini, OpenAI-compatible
DEFAULT_MODEL = "gemini-3.8-flash"   # a free-tier Flash model; Google renames models often, so check your list
MIN_GAP_SECONDS = float(os.getenv("LLM_MIN_GAP", "4"))  # pause between calls to respect free requests-per-minute limits
# If the main model is overloaded (503) or out of quota (429), these free models are tried in turn.
FALLBACK_MODELS = ["gemini-3.6-flash", "gemini-3.5-flash-lite"]
BUSY_ERRORS = (RateLimitError, InternalServerError, APIConnectionError, APITimeoutError)
SINGLE_PASS_CHARS = 60_000           # bills shorter than this are analysed in one piece
CHUNK_CHARS, CHUNK_OVERLAP, MAX_CHUNKS = 80_000, 500, 12   # big chunks = fewer calls (Gemini reads ~1M tokens at once)
JSON_TOKENS, MAP_TOKENS = 8_000, 4_000
MAX_DOWNLOAD_BYTES = 30 * 1024 * 1024
SECTORS = [
    "Agriculture", "Banking & Finance", "Healthcare & Pharma", "IT & Digital", "Manufacturing",
    "Energy & Utilities", "Education", "Transport & Logistics", "Real Estate & Construction",
    "Retail & E-commerce", "Telecom & Media", "Environment & Climate", "Defence & Security",
    "Tourism & Hospitality", "MSMEs & Startups", "Other",
]


# ==== ingestion ====
HEADERS = {"User-Agent": "Mozilla/5.0 (PPIIA student research project)"}


def pdf_to_pages(source) -> List[str]:
    """Text of every page of a PDF (a file path, an uploaded file or a BytesIO)."""
    reader = PdfReader(source)
    if reader.is_encrypted:
        reader.decrypt("")
    return [(page.extract_text() or "") for page in reader.pages]


def html_to_text(html: str) -> str:
    """Visible text of an HTML page, without scripts, menus and footers."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript"]):
        tag.decompose()
    body = soup.find("main") or soup.find("article") or soup.body or soup
    return re.sub(r"\n\s*\n+", "\n\n", body.get_text("\n")).strip()


def fetch_url(url: str) -> dict:
    """Download a bill from a URL (a PDF or an HTML page) and return its pages plus some metadata."""
    note = ""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, stream=True)
    except requests.exceptions.SSLError:  # some government sites have broken certificates
        import urllib3
        urllib3.disable_warnings()
        resp = requests.get(url, headers=HEADERS, timeout=30, stream=True, verify=False)
        note = "The site's SSL certificate could not be verified, so the page was fetched without verification."
    resp.raise_for_status()
    parts, size = [], 0
    for part in resp.iter_content(65_536):
        parts.append(part)
        size += len(part)
        if size > MAX_DOWNLOAD_BYTES:
            raise ValueError("The file is larger than 30 MB.")
    data = b"".join(parts)
    if "pdf" in resp.headers.get("Content-Type", "").lower() or data[:5] == b"%PDF-":
        return {"pages": pdf_to_pages(io.BytesIO(data)), "kind": "pdf", "source": url, "note": note, "html": ""}
    try:
        html = data.decode("utf-8")
    except UnicodeDecodeError:
        html = data.decode(resp.encoding or "latin-1", errors="ignore")
    return {"pages": [html_to_text(html)], "kind": "html", "source": url, "note": note, "html": html}


def find_pdf_links(html: str, base_url: str) -> List[tuple]:
    """(label, absolute URL) for every link on an HTML page that points to a PDF file."""
    soup = BeautifulSoup(html, "html.parser")
    found = {}
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"].strip())
        if ".pdf" in href.lower():
            found.setdefault(href, a.get_text(" ", strip=True)[:70] or href.rsplit("/", 1)[-1])
    return [(label, href) for href, label in found.items()]


SAMPLE_BILL = """The Digital Wellness in Schools Bill, 2026 (FICTIONAL sample for testing only)
1. Short title and commencement. This Act may be called the Digital Wellness in Schools Act, 2026 and shall come into force on 1 April 2027.
2. Definitions. 'Educational institution' means any school recognised under the Right to Education Act, 2009. 'Ed-tech provider' means any company offering digital learning services to students under 18.
3. Screen-time limits. Every educational institution shall limit non-instructional screen use to two hours a day for students below Class 8.
4. Data protection for minors. Ed-tech providers shall not collect biometric data of students or show targeted advertising to them. Violations shall attract a penalty of up to ten lakh rupees, subject to section 7.
5. Teacher training. The Central Government shall fund training of teachers in digital safety through the Ministry of Education within two years of commencement.
6. Grants. Small ed-tech startups with turnover below five crore rupees shall receive a compliance grant of up to two lakh rupees, on the conditions in section 2.
7. Oversight. A Standing Committee on Education shall review implementation annually. This Act amends section 12 of the Information Technology Act, 2000 and repeals the School Digital Devices Circular, 2019.
8. Reports. Institutions shall file an annual compliance report with the State Education Department."""


# ==== clean ====
PAGE_NO_RE = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.I)


def clean_pages(pages: List[str]) -> str:
    """NLP step 1: normalise unicode, drop running headers/footers and page numbers, repair line breaks."""
    pages = [unicodedata.normalize("NFKC", p) for p in pages]
    repeated = set()
    if len(pages) >= 5:  # a line printed on 40% or more of the pages is a running header/footer
        counts = Counter(line for p in pages for line in {l.strip() for l in p.splitlines() if l.strip()})
        repeated = {l for l, c in counts.items() if c >= 0.4 * len(pages) and len(l) < 120}
    kept = []
    for p in pages:
        lines = [l for l in p.splitlines() if l.strip() not in repeated and not PAGE_NO_RE.match(l.strip())]
        kept.append("\n".join(lines))
    text = "\n".join(kept)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)   # re-join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)             # collapse repeated spaces
    text = re.sub(r"\n{3,}", "\n\n", text)          # collapse blank lines
    return text.strip()


# ==== sections ====
SECTION_RE = re.compile(r"^[ \t]*(\d{1,3}[A-Z]?)\.[ \t]+(\S.*)$", re.M)
CLAUSE_RULES = [  # the first rule that matches decides the clause type
    ("Short title / commencement", r"\bmay be called\b|\bcome into force\b|\bshort title\b"),
    ("Definition", r"\bmeans\b|\bunless the context otherwise requires\b|\bshall have the meaning\b"),
    ("Penalty", r"\bpenalt\w*|\bfine\b|\bimprisonment\b|\bpunish\w*|\boffence\w*|\bliable\b"),
    ("Amendment / repeal", r"\bamend\w*|\brepeal\w*|\bomitted\b|\bsubstituted\b|\binserted\b"),
    ("Finance", r"\bfund\b|\bgrant\w*|\brupees\b|\btax\w*|\blevy\b|\bfees?\b|\bcess\b|\bappropriat\w*"),
    ("Power / authority", r"\bmay,? by notification\b|\bempower\w*|\bconstitut\w*|\bappoint\w*|\bauthority\b|\bcommittee\b"),
    ("Procedure", r"\bappeal\w*|\bapplication\b|\btribunal\b|\bhearing\b|\binquiry\b|\bnotice\b|\breport\w*"),
    ("Obligation", r"\bshall\b|\bmust\b"),
]
MODAL_RULES = {  # legal "modal" words show whether a clause obliges, forbids or allows something
    "Obligation": r"\bshall(?!\s+not)\b|\bmust(?!\s+not)\b",
    "Prohibition": r"\bshall\s+not\b|\bmust\s+not\b|\bno\s+person\b|\bprohibit\w*",
    "Permission": r"\bmay\b",
}


def split_sections(text: str) -> List[dict]:
    """NLP step 2: split a bill into its numbered sections (falls back to groups of paragraphs)."""
    keep, last = [], 0
    for m in SECTION_RE.finditer(text):
        number = int(re.match(r"\d+", m.group(1)).group())
        if number >= last:  # numbers must not go backwards, which filters out stray list items
            keep.append(m)
            last = number
    if len(keep) >= 2:
        sections = []
        preamble = text[: keep[0].start()].strip()
        if len(preamble) > 150:
            sections.append({"id": "Preamble", "title": "Preamble / objects", "text": preamble})
        for i, m in enumerate(keep):
            end = keep[i + 1].start() if i + 1 < len(keep) else len(text)
            title = re.split(r"[.\u2014:]", m.group(2))[0].strip()[:80]
            sections.append({"id": m.group(1), "title": title or f"Section {m.group(1)}", "text": text[m.start():end].strip()})
        return sections
    parts, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        if current and len(current) + len(para) > 1_500:
            parts.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        parts.append(current.strip())
    return [{"id": str(i), "title": f"Part {i}", "text": t} for i, t in enumerate(parts, 1)]


def classify_clause(text: str) -> str:
    """Rule-based text classification of one section into a clause type."""
    low = text.lower()
    for label, pattern in CLAUSE_RULES:
        if re.search(pattern, low):
            return label
    return "General"


def modal_counts(text: str) -> dict:
    """Count obligation / prohibition / permission words (regulation analytics)."""
    low = text.lower()
    return {name: len(re.findall(pattern, low)) for name, pattern in MODAL_RULES.items()}


# ==== entities ====
_W = r"(?:[A-Z][\w&'\-]*|of|and|for|the|to|on|in)"
ACT_RE = re.compile(rf"\b[A-Z][\w&'\-]*(?:\s+{_W}){{0,8}}\s+(?:Act|Code)\s*,?\s*(?:1[89]|20)\d{{2}}\b")
TITLE_RE = re.compile(rf"\b[A-Z][\w&'\-]*(?:\s+{_W}){{0,10}}\s+Bill\s*,?\s*(?:1[89]|20)\d{{2}}\b")
LEADING_RE = re.compile(r"^(?:(?:Under|The|This|Every|In|Subject|Notwithstanding|Section|Sections|Of|And|By|For|To|On|Any|All|As|Such)\s+)+")
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
DATE_RE = re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+(?:{MONTHS}),?\s+\d{{4}}\b|\b(?:{MONTHS})\s+\d{{1,2}},?\s+\d{{4}}\b")
DEADLINE_RE = re.compile(
    r"\bwithin\s+(?:a\s+period\s+of\s+)?(?:[\w-]+\s+){1,5}?(?:days?|months?|years?)\b"
    r"|\b(?:annual|annually|every year|each year|quarterly|monthly)\b", re.I)
SECTION_REF_RE = re.compile(r"\b[Ss]ections?\s+(\d{1,3}[A-Z]?)\b")
OTHER_LAW_RE = re.compile(r"\s*(?:\(\w+\)\s*)*(?:and\s+\d+[A-Z]?\s+)?of\s+(?:the\s+)?[A-Z]")


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def find_acts(text: str) -> List[str]:
    """Legal NLP: names of other Acts and Codes the bill refers to."""
    acts = []
    for m in ACT_RE.finditer(text):
        name = re.sub(r"\s+,", ",", LEADING_RE.sub("", m.group(0))).strip()
        if name and name not in acts:
            acts.append(name)
    return acts


def guess_title(text: str) -> str:
    m = TITLE_RE.search(text[:3000])
    return LEADING_RE.sub("", m.group(0)).strip() if m else "This Bill"


def find_dates(text: str) -> List[str]:
    return list(dict.fromkeys(m.group(0) for m in DATE_RE.finditer(text)))


def find_deadlines(text: str) -> List[str]:
    return list(dict.fromkeys(m.group(0).lower() for m in DEADLINE_RE.finditer(text)))


def section_dependencies(sections: List[dict]) -> List[tuple]:
    """Internal dependency edges: (section, section it points to), e.g. "subject to section 7"."""
    ids = {s["id"] for s in sections}
    edges = []
    for s in sections:
        for m in SECTION_REF_RE.finditer(s["text"]):
            target = m.group(1)
            if OTHER_LAW_RE.match(s["text"][m.end(): m.end() + 60]):
                continue  # "section 12 of the XYZ Act" points to a different law, not to this bill
            if target in ids and target != s["id"] and (s["id"], target) not in edges:
                edges.append((s["id"], target))
    return edges


# ==== preprocess ====
def preprocess(pages: List[str], title: str = "") -> dict:
    """Run every preprocessing step and collect the results in one dictionary."""
    text = clean_pages(pages)
    sections = split_sections(text)
    for s in sections:
        s["type"] = classify_clause(s["text"])
        s["modals"] = modal_counts(s["text"])
        s["words"] = len(s["text"].split())
    bill_title = title.strip() or guess_title(text)
    own_name = _key(bill_title.replace("Bill", "Act"))
    return {
        "title": bill_title,
        "text": text,
        "pages": len(pages),
        "sections": sections,
        "acts": [a for a in find_acts(text) if _key(a) != own_name],  # the bill's own name is not an outside Act
        "dates": find_dates(text),
        "deadlines": find_deadlines(text),
        "dependencies": section_dependencies(sections),
    }


# ==== topics ====
LEGAL_STOPWORDS = {
    "shall", "act", "acts", "bill", "section", "sections", "subsection", "clause", "provision", "provisions",
    "provided", "thereof", "therein", "herein", "hereby", "said", "notwithstanding", "accordance", "pursuant",
    "respect", "purposes", "purpose", "made", "make", "given", "means", "include", "includes", "including",
    "person", "persons", "may", "any", "under", "such", "unless", "context", "otherwise", "required",
}
STOP_WORDS = list(set(ENGLISH_STOP_WORDS) | LEGAL_STOPWORDS)


def simple_keywords(sections: List[dict], n: int = 15) -> List[str]:
    """Pure-Python TF-IDF done by hand: the safety net used when scikit-learn cannot run."""
    stop = set(STOP_WORDS)
    docs = [Counter(w for w in re.findall(r"[a-z][a-z\-]{2,}", s["text"].lower()) if w not in stop) for s in sections]
    df = Counter(w for d in docs for w in d)
    scores = Counter()
    for d in docs:
        for w, c in d.items():
            scores[w] += c * (math.log((1 + len(docs)) / (1 + df[w])) + 1)
    return [w for w, _ in scores.most_common(n)]


def _sklearn_topics(kept: List[dict], result: dict, n_topics: int, n_words: int) -> dict:
    try:
        vec = TfidfVectorizer(stop_words=STOP_WORDS, ngram_range=(1, 2), max_df=0.9,
                              token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]{2,}\b")
        X = vec.fit_transform([s["text"] for s in kept])
    except ValueError as e:
        if "empty vocabulary" in str(e) or "no terms remain" in str(e):  # nothing left after removing stop words
            return result
        raise
    terms = vec.get_feature_names_out()
    scores = np.asarray(X.sum(axis=0)).ravel()
    result["keywords"] = [str(terms[i]) for i in np.argsort(-scores)[:15]]
    k = min(n_topics, len(kept) - 1)
    if k >= 2:
        # A dense, C-ordered copy and the pure-NumPy "mu" solver avoid a C-contiguity error that some
        # scikit-learn / NumPy version combinations raise inside NMF's compiled solver.
        dense = np.ascontiguousarray(X.toarray(), dtype=np.float64)
        try:
            nmf = NMF(n_components=k, init="nndsvda", solver="mu", random_state=0, max_iter=500)
            weights = nmf.fit_transform(dense)
        except (ValueError, TypeError) as e:  # topic modelling is a bonus: keep the keywords if it fails
            result["note"] = f"Topic modelling (NMF) was skipped ({type(e).__name__}: {e}). Showing keywords only."
            return result
        result["topics"] = [[str(terms[i]) for i in np.argsort(-comp)[:n_words]] for comp in nmf.components_]
        result["section_topic"] = {s["id"]: int(np.argmax(w)) for s, w in zip(kept, weights)}
    return result


def extract_topics(sections: List[dict], n_topics: int = 4, n_words: int = 6) -> dict:
    """Topic extraction: TF-IDF keywords plus NMF topic modelling over the bill's sections.
    Topics are a bonus, so a library problem is reported in result["note"] instead of stopping the run."""
    result = {"keywords": [], "topics": [], "section_topic": {}, "note": ""}
    kept = [s for s in sections if len(s["text"]) > 80]
    if len(kept) < 2:
        return result
    try:
        return _sklearn_topics(kept, result, n_topics, n_words)
    except Exception as e:  # compiled-library problems must never stop the analysis
        where = traceback.extract_tb(e.__traceback__)[-1]
        result.update(topics=[], section_topic={}, keywords=simple_keywords(kept),
                      note=f"Topic modelling was skipped ({type(e).__name__}: {e} at {os.path.basename(where.filename)}"
                           f":{where.lineno} in {where.name}). Showing simple keyword counts instead.")
        return result


def build_hints(bill: dict, topics: dict) -> str:
    """Rule-based findings handed to the LLM as hints (hybrid classical NLP + LLM)."""
    lines = []
    if bill["acts"]:
        lines.append("Acts mentioned: " + "; ".join(bill["acts"]))
    if bill["dates"]:
        lines.append("Dates: " + "; ".join(bill["dates"]))
    if bill["deadlines"]:
        lines.append("Time limits: " + "; ".join(bill["deadlines"]))
    if topics["keywords"]:
        lines.append("Top keywords (TF-IDF): " + ", ".join(topics["keywords"][:12]))
    for i, words in enumerate(topics["topics"], 1):
        lines.append(f"Topic {i} words (NMF): " + ", ".join(words))
    counts = Counter(s["type"] for s in bill["sections"])
    lines.append("Clause types: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    return "\n".join(lines)


# ==== schemas ====
def _clean(items):
    return [s.strip() for s in items if isinstance(s, str) and s.strip()]


def _text(v):
    return "" if v is None else str(v)


class PlainSummary(BaseModel):
    lines: List[str]

    @field_validator("lines")
    @classmethod
    def check_lines(cls, v):
        v = _clean(v)
        if len(v) < 8:
            raise ValueError("lines must contain 10 to 20 plain-language lines")
        return v[:20]


class Sector(BaseModel):
    name: str
    reason: str


class IndustryImpact(BaseModel):
    benefiting: List[Sector] = Field(default_factory=list)
    restricted: List[Sector] = Field(default_factory=list)
    opportunities: List[str] = Field(default_factory=list)


class Topic(BaseModel):
    name: str
    keywords: List[str] = Field(default_factory=list)
    description: str = ""


class LegislativeContext(BaseModel):
    bill_type: str = ""
    ministry: str = ""
    policy_area: str = ""
    topics: List[Topic] = Field(default_factory=list)
    stakeholders: List[str] = Field(default_factory=list)
    related_laws: List[str] = Field(default_factory=list)

    @field_validator("bill_type", "ministry", "policy_area", mode="before")
    @classmethod
    def none_to_text(cls, v):
        return _text(v)


class Relation(BaseModel):
    source: str
    relation: str
    target: str
    year: Optional[str] = None

    @field_validator("year", mode="before")
    @classmethod
    def year_to_str(cls, v):
        return None if v in (None, "", "null") else str(v)


class Event(BaseModel):
    when: str
    event: str
    title: str = ""

    @field_validator("when", "title", mode="before")
    @classmethod
    def to_text(cls, v):
        return _text(v)


class Chronology(BaseModel):
    relations: List[Relation] = Field(default_factory=list)
    timeline: List[Event] = Field(default_factory=list)


class ImpactAssessment(BaseModel):
    short_term: List[str] = Field(default_factory=list)
    medium_term: List[str] = Field(default_factory=list)
    long_term: List[str] = Field(default_factory=list)

    @field_validator("short_term", "medium_term", "long_term")
    @classmethod
    def clean(cls, v):
        return _clean(v)


class ProsCons(BaseModel):
    positives: List[str] = Field(default_factory=list)
    negatives: List[str] = Field(default_factory=list)

    @field_validator("positives", "negatives")
    @classmethod
    def clean(cls, v):
        return _clean(v)


# ==== prompts ====
SYSTEM_PROMPT = (
    "You are a careful public-policy analyst. Use ONLY the bill text provided; if the text does not say "
    "something, do not invent it. Hints from rule-based NLP may be noisy: use them only when the text supports them. "
    "Write for an educated general reader in clear, neutral language."
)
MAP_PROMPT = (
    "Condense this part of a bill into factual notes (max 500 words): key provisions, definitions, who must do "
    "what, penalties or incentives, dates and deadlines, and any other Acts, amendments, committees or ministries "
    "it mentions. Add no outside information.\n\nTEXT:\n"
)
LABELS = {
    "context": "Topics & legislative context",
    "summary": "Plain-language summary",
    "industry": "Industry impact",
    "chronology": "Chronology & dependency graph",
    "impact": "Impact assessment",
    "proscons": "Positives & negatives",
}
TASKS = {  # key: (schema, instruction, JSON shape)
    "context": (
        LegislativeContext,
        "Describe the bill's legislative context: bill_type (new Act, amendment, repeal or other), the responsible "
        "ministry or department if the text names one, the main policy_area, 3 to 5 topics (each with a short name, "
        "3 to 6 keywords and a one-sentence description; you may refine the hinted topic words), the key stakeholders, "
        "and the related_laws the text mentions.",
        '{"bill_type": "", "ministry": "", "policy_area": "", "topics": [{"name": "", "keywords": [""], '
        '"description": ""}], "stakeholders": [""], "related_laws": [""]}',
    ),
    "summary": (
        PlainSummary,
        "Explain what this bill means for a normal person in 10 to 20 short lines. Each line is one separate bullet "
        "point (do not number the lines). No jargon or legal citations. Cover what changes in daily life, who is "
        "affected, and what to expect.",
        '{"lines": ["...", "..."]}',
    ),
    "industry": (
        IndustryImpact,
        "Classify the bill's impact on industries. Use these sector names where possible: " + ", ".join(SECTORS)
        + ". 'benefiting' = sectors that gain; 'restricted' = sectors facing new rules, limits or costs; "
        "'opportunities' = new business opportunities it creates. Give a one-sentence reason per sector and include "
        "only sectors the text supports.",
        '{"benefiting": [{"name": "", "reason": ""}], "restricted": [{"name": "", "reason": ""}], "opportunities": [""]}',
    ),
    "chronology": (
        Chronology,
        "Build the dependency graph and timeline. 'relations': links between this bill and earlier Acts, amendments, "
        "repeals, committees, ministries and authorities that the text EXPLICITLY mentions; use the bill's own title as "
        "a node and short verbs for 'relation' (amends, repeals, replaces, refers to, examined by, administered by, "
        "introduced in). 'timeline': dated or time-bound events in the text (introduction, committee referral, "
        "commencement, deadlines, review dates), each with 'when' exactly as written, a short 'title' (3 to 6 words) "
        "and a one-sentence 'event' description. Do not add history that is not in the text. 'year' may be null.",
        '{"relations": [{"source": "", "relation": "", "target": "", "year": null}], '
        '"timeline": [{"when": "", "title": "", "event": ""}]}',
    ),
    "impact": (
        ImpactAssessment,
        "Give a basic inference of the likely impact in three horizons: short_term (0-1 year), medium_term "
        "(1-5 years), long_term (more than 5 years). 2 to 4 bullets each, grounded in the provisions, with cautious "
        "wording ('likely', 'may').",
        '{"short_term": [""], "medium_term": [""], "long_term": [""]}',
    ),
    "proscons": (
        ProsCons,
        "List the main positives and negatives of this bill as short bullets (3 to 6 each). Be balanced and base them "
        "on the text.",
        '{"positives": [""], "negatives": [""]}',
    ),
}


# ==== llm ====
_last_call = [0.0]
_status = [lambda message: None]


def set_status_hook(func) -> None:
    """Register a function that receives live status messages (retries, model switches)."""
    _status[0] = func


def make_client(base_url: str, api_key: str) -> OpenAI:
    # A 2-minute timeout and no hidden retries: chat() below does the retrying and reports it.
    return OpenAI(base_url=base_url, api_key=api_key, timeout=120, max_retries=0)


def chat(client, model: str, user: str, max_tokens: int) -> str:
    """One model call. Waits between calls. If Gemini is busy (503) or over quota (429) it retries with
    growing pauses and, after two failures, switches to the next fallback model."""
    models = [model] + [m for m in FALLBACK_MODELS if m != model]
    for attempt in range(6):
        current = models[min(attempt // 2, len(models) - 1)]
        pause = MIN_GAP_SECONDS - (time.time() - _last_call[0])
        if pause > 0:
            time.sleep(pause)
        try:
            r = client.chat.completions.create(
                model=current, temperature=0.2, max_tokens=max_tokens,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
            )
            _last_call[0] = time.time()
            return (r.choices[0].message.content or "").strip()
        except NotFoundError:
            if current == model:
                raise  # the main model name is wrong: tell the user
            # an unknown fallback name is simply skipped
        except BUSY_ERRORS as e:
            _last_call[0] = time.time()
            if attempt == 5:
                raise
            wait = min(5 * 2 ** attempt, 60)  # 5, 10, 20, 40, 60 seconds
            nxt = models[min((attempt + 1) // 2, len(models) - 1)]
            _status[0](f"Gemini is busy ({type(e).__name__}). Retrying in {wait}s with {nxt} (attempt {attempt + 2} of 6)...")
            time.sleep(wait)
    raise OpenAIError("No model answered. Check the model names in your settings.")


def parse_json(text: str) -> dict:
    """Pull the JSON object out of a model reply (it may be wrapped in a code fence)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the model reply")
    return json.loads(text[start: end + 1])


def ask_json(client, model, schema, task, shape, context, hints=""):
    """Ask for JSON, validate it with pydantic, and retry once with the error message if it is invalid."""
    prompt = f"{task}\n\nReturn ONLY a JSON object shaped like:\n{shape}\n"
    if hints:
        prompt += f"\nHINTS FROM RULE-BASED NLP (may be noisy):\n{hints}\n"
    prompt += f'\nBILL TEXT:\n"""\n{context}\n"""'
    error = None
    for _ in range(2):
        msg = prompt if error is None else f"{prompt}\n\nYour previous reply was invalid ({error}). Return valid JSON only."
        try:
            return schema.model_validate(parse_json(chat(client, model, msg, JSON_TOKENS)))
        except (ValueError, ValidationError) as e:
            error = str(e)[:200]
    raise ValueError(f"model did not return valid JSON for {schema.__name__}: {error}")


def run_task(client, model, key, digest, hints=""):
    schema, task, shape = TASKS[key]
    return ask_json(client, model, schema, task, shape, digest, hints)


# ==== digest ====
def chunk_text(text: str) -> List[str]:
    """Cut long text into overlapping pieces, preferring to cut at a line break."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + CHUNK_CHARS, len(text))
        if end < len(text):
            cut = text.rfind("\n", start + CHUNK_CHARS // 2, end)
            end = cut if cut != -1 else end
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def estimate_calls(n_chars: int) -> int:
    """How many model calls analysing this much text will need."""
    if n_chars <= SINGLE_PASS_CHARS:
        return len(TASKS)
    parts = min(MAX_CHUNKS, -(-(n_chars - CHUNK_OVERLAP) // (CHUNK_CHARS - CHUNK_OVERLAP)))
    return parts + len(TASKS)


def condense_parts(client, model, parts, label, progress, lo, hi):
    notes = []
    for i, part in enumerate(parts, 1):
        progress(lo + (hi - lo) * (i - 1) / len(parts), f"{label} {i} of {len(parts)}")
        notes.append(chat(client, model, MAP_PROMPT + part, MAP_TOKENS))
    return "\n\n".join(notes)


def build_digest(client, model, text, progress):
    """Short bills pass through unchanged. Long bills are condensed part by part (map step), then the
    notes are condensed again until they fit one prompt (reduce step)."""
    if len(text) <= SINGLE_PASS_CHARS:
        return text, False
    chunks = chunk_text(text)
    digest = condense_parts(client, model, chunks[:MAX_CHUNKS], "Reading part", progress, 0.0, 0.8)
    for _ in range(2):
        if len(digest) <= SINGLE_PASS_CHARS:
            break
        digest = condense_parts(client, model, chunk_text(digest), "Condensing notes", progress, 0.8, 1.0)
    return digest[:SINGLE_PASS_CHARS], len(chunks) > MAX_CHUNKS


# ==== graphs ====
NODE_COLORS = {"bill": "#fde68a", "act": "#bfdbfe", "committee": "#bbf7d0", "authority": "#fed7aa",
               "section": "#e5e7eb", "other": "#f3f4f6"}


def node_kind(name: str) -> str:
    low = name.lower()
    if re.search(r"\bbill\b|\bthis act\b", low):
        return "bill"
    if re.search(r"\b(act|code|ordinance|circular|rules?|regulations?)\b", low):
        return "act"
    if re.search(r"\b(committee|commission|council|board)\b", low):
        return "committee"
    if re.search(r"\b(ministry|department|authority|government|tribunal|court|agency|parliament|lok sabha|rajya sabha)\b", low):
        return "authority"
    return "other"


def build_graph(bill: dict, chronology=None) -> nx.DiGraph:
    """Knowledge / dependency graph: the bill, the Acts it touches, committees and authorities."""
    g, seen, title = nx.DiGraph(), {}, bill["title"]

    def canon(name: str) -> str:  # one node per real-world thing, even if spelled slightly differently
        name = name.strip()
        return title if node_kind(name) == "bill" else seen.setdefault(_key(name), name)

    g.add_node(title, kind="bill")
    for act in bill["acts"]:
        a = canon(act)
        g.add_node(a, kind="act")
        g.add_edge(title, a, label="refers to")
    for r in (chronology.relations if chronology else []):
        src, dst = canon(r.source), canon(r.target)
        for n in (src, dst):
            if n not in g:
                g.add_node(n, kind=node_kind(n))
        label = r.relation + (f" ({r.year})" if r.year else "")
        if g.has_edge(src, dst):
            if label not in g[src][dst]["label"]:
                g[src][dst]["label"] += "; " + label
        else:
            g.add_edge(src, dst, label=label)
    return g


def section_graph(bill: dict) -> nx.DiGraph:
    """Internal dependency graph: which section points to which other section."""
    g = nx.DiGraph()
    names = {s["id"]: f"Sec {s['id']}: {s['title'][:28]}" for s in bill["sections"]}
    for a, b in bill["dependencies"]:
        g.add_node(names[a], kind="section")
        g.add_node(names[b], kind="section")
        g.add_edge(names[a], names[b], label="refers to")
    return g


def graph_to_dot(g: nx.DiGraph, rankdir: str = "LR") -> str:
    """Graphviz DOT text, which Streamlit can draw directly."""
    esc = lambda s: str(s).replace("\\", "/").replace('"', "'")
    wrap = lambda s: "\\n".join(textwrap.wrap(esc(s), 26)) or esc(s)
    lines = ["digraph G {", f"rankdir={rankdir};", 'node [shape=box, style="rounded,filled", fontsize=10];', "edge [fontsize=9];"]
    for n, d in g.nodes(data=True):
        lines.append(f'"{esc(n)}" [label="{wrap(n)}", fillcolor="{NODE_COLORS.get(d.get("kind", "other"), "#f3f4f6")}"];')
    for u, v, d in g.edges(data=True):
        lines.append(f'"{esc(u)}" -> "{esc(v)}" [label="{esc(d.get("label", ""))}"];')
    return "\n".join(lines + ["}"])


def draw_graph(g: nx.DiGraph, title: str = "") -> None:
    """Draw a graph with matplotlib (used in the notebook)."""
    import matplotlib.pyplot as plt
    if g.number_of_nodes() == 0:
        print("Nothing to draw.")
        return
    plt.figure(figsize=(11, 6))
    pos = nx.spring_layout(g, seed=42, k=1.5)
    colors = [NODE_COLORS.get(d.get("kind", "other"), "#f3f4f6") for _, d in g.nodes(data=True)]
    labels = {n: textwrap.fill(str(n), 18) for n in g.nodes}
    nx.draw(g, pos, labels=labels, node_color=colors, node_size=3600, font_size=7, arrows=True, edgecolors="#9ca3af")
    nx.draw_networkx_edge_labels(g, pos, edge_labels=nx.get_edge_attributes(g, "label"), font_size=7)
    plt.title(title)
    plt.axis("off")
    plt.show()


# ==== report ====
def to_markdown(bill: dict, topics: dict, results: dict) -> str:
    """Assemble everything into one Markdown report."""
    counts = Counter(s["type"] for s in bill["sections"])
    out = [f"# PPIIA report: {bill['title']}", "", "## Document overview",
           f"- {bill['pages']} page(s), {len(bill['text'].split()):,} words, {len(bill['sections'])} sections",
           "- Clause types: " + ", ".join(f"{k} ({v})" for k, v in counts.most_common()),
           "- Other Acts referenced: " + ("; ".join(bill["acts"]) or "none found"),
           "- Dates: " + ("; ".join(bill["dates"]) or "none found") + " | Time limits: " + ("; ".join(bill["deadlines"]) or "none found"),
           "- Top keywords: " + (", ".join(topics["keywords"][:10]) or "n/a"), ""]
    c, s, i, ch, im, pc = (results.get(k) for k in ("context", "summary", "industry", "chronology", "impact", "proscons"))
    if c:
        out += ["## Topics & legislative context",
                f"- Type: {c.bill_type or 'n/a'} | Ministry: {c.ministry or 'n/a'} | Policy area: {c.policy_area or 'n/a'}"]
        out += [f"- **{t.name}** ({', '.join(t.keywords)}): {t.description}" for t in c.topics]
        out += ["- Stakeholders: " + (", ".join(c.stakeholders) or "n/a"), "- Related laws: " + (", ".join(c.related_laws) or "n/a"), ""]
    if s:
        out += ["## What this means for citizens"] + [f"- {line}" for line in s.lines] + [""]
    if i:
        out += ["## Industry impact", "**Sectors that benefit**"] + [f"- {x.name}: {x.reason}" for x in i.benefiting]
        out += ["**Sectors facing restrictions**"] + [f"- {x.name}: {x.reason}" for x in i.restricted]
        out += ["**New opportunities**"] + [f"- {x}" for x in i.opportunities] + [""]
    if ch:
        out += ["## Chronology & dependency graph", "**Relations**"] + [
            f"- {r.source} -- {r.relation} --> {r.target}" + (f" ({r.year})" if r.year else "") for r in ch.relations]
        out += ["**Timeline**"] + [f"- {e.when}: " + (f"**{e.title}**. " if e.title else "") + e.event for e in ch.timeline]
        out += ["**Section dependencies**"] + [f"- Section {a} refers to section {b}" for a, b in bill["dependencies"]] + [""]
    if im:
        out += ["## Impact assessment"]
        for name, items in (("Short term (0-1 year)", im.short_term), ("Medium term (1-5 years)", im.medium_term),
                            ("Long term (more than 5 years)", im.long_term)):
            out += [f"**{name}**"] + [f"- {x}" for x in items]
        out.append("")
    if pc:
        out += ["## Positives & negatives", "**Positives**"] + [f"- {x}" for x in pc.positives]
        out += ["**Negatives**"] + [f"- {x}" for x in pc.negatives]
    out += ["", "_AI-generated analysis for learning purposes; not legal advice._"]
    return "\n".join(out)


# ==== pipeline ====
def analyse(client, model, bill: dict, topics: dict, progress):
    """Condense long bills, then run every LLM task. Returns (results, errors, truncated)."""
    text = bill["text"]
    share = 0.5 if len(text) > SINGLE_PASS_CHARS else 0.0
    digest, truncated = build_digest(client, model, text, lambda f, m: progress(f * share, m))
    hints = build_hints(bill, topics)
    results, errors = {}, {}
    for i, key in enumerate(TASKS):
        progress(share + (1 - share) * i / len(TASKS), f"Analysing: {LABELS[key]}")
        try:
            results[key] = run_task(client, model, key, digest, hints)
        except ValueError as e:
            results[key], errors[key] = None, str(e)
    progress(1.0, "Done")
    return results, errors, truncated
