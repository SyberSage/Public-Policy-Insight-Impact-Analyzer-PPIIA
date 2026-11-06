"""Public Policy Insight & Impact Analyzer (PPIIA): Streamlit dashboard.

All the real work lives in ppiia_core.py; this file is only the user interface.
Run locally with:  python -m streamlit run app.py
"""
import html
import io
import os
from collections import Counter

import pandas as pd
import streamlit as st
from openai import OpenAIError

from ppiia_core import (
    CHUNK_CHARS, DEFAULT_BASE_URL, DEFAULT_MODEL, LABELS, MAX_CHUNKS, SAMPLE_BILL, analyse, build_graph,
    estimate_calls, extract_topics, fetch_url, find_pdf_links, graph_to_dot, make_client, pdf_to_pages,
    preprocess, section_graph, set_status_hook, to_markdown,
)

# ------------------------------------------------------------------ settings
def _env_file_values() -> dict:
    """Values from a .env file next to this script (KEY=value per line), if there is one."""
    try:
        folder = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        folder = os.getcwd()
    values, path = {}, os.path.join(folder, ".env")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_secret(name: str, default: str = "") -> str:
    """Look in Streamlit secrets first, then environment variables, then a .env file."""
    try:
        return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name) or _env_file_values().get(name, default)


@st.cache_resource(show_spinner=False)
def get_client(base_url: str, api_key: str):
    return make_client(base_url, api_key)


@st.cache_data(show_spinner=False)
def pages_from_pdf(data: bytes):
    return pdf_to_pages(io.BytesIO(data))


# -------------------------------------------------------------------- styling
CSS = """
<style>
html, [data-testid="stMain"], [data-testid="stAppViewContainer"] { scroll-behavior: smooth; }
.stApp { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { background: #ffffff; border-radius: 16px; box-shadow: 0 12px 40px rgba(0,0,0,.22); padding: 2rem 2.4rem 1.6rem; max-width: 980px; margin: 1rem auto 2rem; }
[data-testid="stMainBlockContainer"] :is(h1,h2,h3,h4,p,li,label,summary):not(button *) { color: #2d3748; }
[data-testid="stMainBlockContainer"] [data-testid="stCaptionContainer"] * { color: #718096; }
[data-testid="stMainBlockContainer"] button[kind="primary"], [data-testid="stMainBlockContainer"] [data-testid="stBaseButton-primary"] { background: linear-gradient(135deg, #667eea, #764ba2); border: none; }
[data-testid="stMainBlockContainer"] button[kind="primary"] p, [data-testid="stMainBlockContainer"] [data-testid="stBaseButton-primary"] p { color: #ffffff !important; }
[data-testid="stMainBlockContainer"] [id] { scroll-margin-top: 1.2rem; }
[data-testid="stMainBlockContainer"] [data-testid="stHeading"] h3 { border-left: 4px solid; border-image: linear-gradient(180deg, #667eea, #764ba2) 1; padding: 0 0 0 12px; font-size: 1.18rem; font-weight: 700; margin: 1.6rem 0 .5rem; }
[data-testid="stHeaderActionElements"] { display: none !important; }
.rpt-title { text-align: center; font-weight: 800; font-size: 1.7rem; margin: .2rem 0 .4rem; background: linear-gradient(90deg, #4c51bf, #764ba2); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.lead { text-align: center; color: #718096 !important; margin-bottom: 1.2rem; }
.sub { font-weight: 700; font-size: .98rem; color: #2d3748; margin: 1rem 0 .2rem; }
ul.pp { margin: .2rem 0 .6rem 1.1rem; padding: 0; }
ul.pp li { font-size: .93rem; line-height: 1.6; color: #2d3748; margin: .1rem 0; }
.muted, .small { font-size: .88rem; color: #718096 !important; margin: .2rem 0 .6rem; }
.chips { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: .4rem 0 .8rem; }
.chip { background: linear-gradient(135deg, #eef2ff 0%, #f5eeff 100%); border-radius: 10px; padding: .6rem .8rem; }
.chip span { display: block; font-size: .74rem; color: #718096; text-transform: uppercase; letter-spacing: .04em; }
.chip b { font-size: 1rem; color: #2d3748; }
.tl { display: grid; grid-template-columns: 110px 1fr; gap: 10px; margin: .5rem 0; align-items: start; }
.tl .yr { color: #667eea; font-weight: 700; font-size: .85rem; padding-top: .55rem; }
.tl .card { background: linear-gradient(135deg, #eef2ff 0%, #f5eeff 100%); border-radius: 8px; padding: .6rem .85rem; }
.tl .card b { display: block; font-size: .93rem; color: #2d3748; }
.tl .card span { font-size: .83rem; color: #4a5568; }
.foot { text-align: center; color: #a0aec0; font-size: .78rem; margin-top: 1.8rem; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #f7f8ff 0%, #ece9fe 100%); border-right: 1px solid #e0e4ff; }
[data-testid="stSidebar"] :is(p,label,h1,h2,h3):not(button *) { color: #2d3748; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * { color: #718096; }
.brand { font-weight: 800; font-size: 1.35rem; margin: .2rem 0 .2rem; background: linear-gradient(90deg, #4c51bf, #9f7aea); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; }
.brand small { display: block; font-weight: 400; font-size: .78rem; -webkit-text-fill-color: #718096; }
.navhead { font-size: .72rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: #8b93c9; margin: 1.2rem 0 .4rem; }
.navlist a { display: block; padding: .5rem .75rem; margin: .1rem 0; border-radius: 10px; color: #3730a3 !important; font-weight: 600; font-size: .93rem; text-decoration: none !important; }
.navlist a:hover { background: linear-gradient(135deg, #e4e7ff, #f0e9ff); }
.navlist a.active { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #ffffff !important; box-shadow: 0 4px 12px rgba(102,126,234,.35); }
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button { width: 100%; justify-content: flex-start; background: linear-gradient(135deg, #ffffff, #eef2ff); border: 1px solid #c7d2fe; border-radius: 10px; }
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button p { color: #3730a3; font-weight: 600; }
[data-testid="stSidebar"] [data-testid="stDownloadButton"] > button:hover { background: linear-gradient(135deg, #eef2ff, #e9e3ff); border-color: #a5b4fc; }
</style>
"""
FOOTER = '<div class="foot">Generated using AI • For informational purposes only</div>'

# Highlights the menu item of the section you are reading. It runs once in the browser page: every 0.3 s and on
# every scroll it finds the last section heading that has reached the top of the screen.
SPY_JS = """
<script>
(function () {
  var w = (window.parent && window.parent !== window) ? window.parent : window;
  if (w.__ppiiaSpy) { return; }
  w.__ppiiaSpy = true;
  var d = w.document;
  function update() {
    var links = d.querySelectorAll('.navlist a[href^="#"]');
    if (!links.length) { return; }
    var current = links[0];
    for (var i = 0; i < links.length; i++) {
      var el = d.getElementById(links[i].getAttribute('href').slice(1));
      if (el && el.getBoundingClientRect().top <= 90) { current = links[i]; }
    }
    var s = d.querySelector('[data-testid="stMain"]') || d.querySelector('section.main') || d.scrollingElement;
    if (s && s.scrollTop > 0 && s.scrollTop + s.clientHeight >= s.scrollHeight - 4) { current = links[links.length - 1]; }
    for (var j = 0; j < links.length; j++) { links[j].classList.toggle('active', links[j] === current); }
  }
  d.addEventListener('scroll', update, true);
  w.addEventListener('resize', update);
  w.setInterval(update, 300);
  update();
})();
</script>
"""


def enable_scroll_highlight() -> None:
    """Turn on the 'you are here' highlight in the sidebar menu."""
    try:
        st.html(SPY_JS, unsafe_allow_javascript=True)
    except TypeError:  # older Streamlit: run the same script inside a hidden iframe instead
        import streamlit.components.v1 as components
        components.html(SPY_JS, height=0)


def esc(text) -> str:
    """Make any text safe to place inside HTML (also stops $ being read as a formula)."""
    return html.escape(" ".join(str(text).split())).replace("$", "&#36;")


def heading(title: str, icon: str, anchor: str) -> None:
    """A section heading. Streamlit's own anchor setting is what lets the sidebar links jump here."""
    st.subheader(f"{icon} {title}", anchor=anchor)


def sub(title: str, icon: str) -> str:
    return f'<div class="sub">{icon} {esc(title)}</div>'


def ul(items, raw: bool = False, empty: str = "Nothing identified in the text.") -> str:
    """A bullet list. Items are escaped unless raw=True (then they must already be safe HTML)."""
    if not items:
        return f'<p class="muted">{esc(empty)}</p>'
    return '<ul class="pp">' + "".join(f"<li>{i if raw else esc(i)}</li>" for i in items) + "</ul>"


def chips(pairs) -> str:
    return '<div class="chips">' + "".join(f'<div class="chip"><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v in pairs) + "</div>"


def show(html_text: str) -> None:
    st.markdown(html_text, unsafe_allow_html=True)


# ----------------------------------------------------------------- input step
def choose_input():
    """Return the pages of the chosen bill, or None while nothing has been loaded yet."""
    method = st.radio("How do you want to load the bill?", ["Upload PDF", "From a URL (Sansad / PDF / HTML)", "Paste text",
                                                          "Try the sample bill"], horizontal=True, key="method")
    if method == "Upload PDF":
        f = st.file_uploader("Bill PDF", type=["pdf"], key="pdf_file")
        if f:
            try:
                return pages_from_pdf(f.getvalue())
            except Exception as e:
                st.error(f"Could not read that PDF: {e}")
    elif method.startswith("From a URL"):
        url = st.text_input("Bill page or PDF link", placeholder="https://sansad.in/...", key="bill_url")
        if st.button("Fetch", key="fetch_btn") and url:
            try:
                doc = fetch_url(url.strip())
                st.session_state["doc"] = doc
                st.session_state["links"] = find_pdf_links(doc["html"], url) if doc["kind"] == "html" else []
            except Exception as e:
                st.error(f"Could not fetch that link: {e}")
        links = st.session_state.get("links", [])
        if links:
            st.info(f"This page has {len(links)} PDF link(s). Pick one, or analyse the page text instead.")
            choice = st.selectbox("PDF links found", links, format_func=lambda x: x[0], key="pdf_choice")
            if st.button("Use the selected PDF", key="use_pdf_btn"):
                try:
                    st.session_state["doc"] = fetch_url(choice[1])
                    st.session_state["links"] = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not fetch that PDF: {e}")
        doc = st.session_state.get("doc")
        if doc:
            st.success(f"Loaded a {doc['kind'].upper()} from {doc['source']} ({sum(len(p) for p in doc['pages']):,} characters)")
            if doc["note"]:
                st.warning(doc["note"])
            return doc["pages"]
    elif method == "Paste text":
        text = st.text_area("Paste the bill text", height=220, key="pasted")
        if len(text.strip()) >= 200:
            return [text]
    else:
        st.caption("A short fictional bill, handy for testing the dashboard.")
        return [SAMPLE_BILL]
    return None


def input_panel(api_key: str, base_url: str, model: str) -> None:
    """The 'load a bill' controls and the Analyze button. Saves the finished report in session_state."""
    pages = choose_input()
    title = st.text_input("Bill name (optional, otherwise guessed from the text)", key="bill_title")
    if not st.button("🔍 Analyze bill", type="primary", disabled=not pages, key="analyze_btn"):
        return
    if not api_key:
        st.error("No API key found. Add it to .streamlit/secrets.toml or a .env file, or paste it in the sidebar.")
        st.stop()
    bill = preprocess(pages, title)
    if len(bill["text"]) < 200:
        st.error("Not enough text found. Scanned PDFs need OCR first; try pasting the text instead.")
        st.stop()
    topics = extract_topics(bill["sections"])
    calls = estimate_calls(len(bill["text"]))
    st.info(f"This bill needs about {calls} model calls, roughly {max(1, calls * 15 // 60)}-{max(2, calls * 30 // 60)} "
            "minutes on the free tier. Keep this tab open.")
    bar = st.progress(0.0, text="Starting...")
    state = {"f": 0.0}

    def on_progress(fraction, message):
        state["f"] = min(fraction, 1.0)
        bar.progress(state["f"], text=message)

    set_status_hook(lambda message: bar.progress(state["f"], text=message))
    try:
        results, errors, truncated = analyse(get_client(base_url, api_key), model, bill, topics, on_progress)
    except OpenAIError as e:
        bar.empty()
        st.error(f"The model API is busy or returned an error: {e}. Wait a minute and try again, or set a different LLM_MODEL.")
        st.stop()
    bar.empty()
    st.session_state["report"] = {"bill": bill, "topics": topics, "results": results, "errors": errors, "truncated": truncated}
    st.rerun()


# ------------------------------------------------------------- report sections
def show_summary(d, report):
    heading("What this means for citizens", "👥", "summary")
    show(ul(d.lines))


def show_industry(d, report):
    heading("Industry Impact Report", "🏭", "industry")
    show(sub("Benefited Sectors", "✅") + ul([f"<b>{esc(s.name)}</b> – {esc(s.reason)}" for s in d.benefiting], raw=True)
         + sub("Restricted Sectors", "⚠️") + ul([f"<b>{esc(s.name)}</b> – {esc(s.reason)}" for s in d.restricted], raw=True)
         + sub("New Business Opportunities", "🚀") + ul(d.opportunities))


def timeline_html(events) -> str:
    rows = []
    for e in events:
        title = f"<b>{esc(e.title)}</b>" if e.title else ""
        rows.append(f'<div class="tl"><div class="yr">{esc(e.when)}</div><div class="card">{title}<span>{esc(e.event)}</span></div></div>')
    return "".join(rows)


def show_chronology(d, report):
    bill = report["bill"]
    heading("Timeline of Bill History", "🕐", "timeline")
    show(timeline_html(d.timeline) if d.timeline else '<p class="muted">No dated or time-bound events found in the text.</p>')
    heading("Knowledge Graph", "🕸️", "graph")
    g = build_graph(bill, d)
    if g.number_of_edges() == 0:
        st.info("No links to other Acts, committees or authorities were found.")
    else:
        st.graphviz_chart(graph_to_dot(g))
        st.caption("Yellow = this bill, blue = Acts, green = committees, orange = authorities. Built only from references in the text.")
    heading("Dependency Graph Between Sections", "🔗", "dependencies")
    sg = section_graph(bill)
    if sg.number_of_edges() == 0:
        st.caption("No section refers to another section.")
    else:
        st.graphviz_chart(graph_to_dot(sg))


def show_impact(d, report):
    heading("Impact Assessment", "📊", "impact")
    show(sub("Short Term (0–1 year)", "⏱") + ul(d.short_term) + sub("Medium Term (1–5 years)", "📈") + ul(d.medium_term)
         + sub("Long Term (>5 years)", "🚀") + ul(d.long_term)
         + '<p class="muted">Basic AI inference from the bill\'s provisions, not a forecast.</p>')


def show_proscons(d, report):
    heading("Positives and Negatives", "⚖️", "proscons")
    show(sub("Positives", "✅") + ul(d.positives) + sub("Negatives", "⚠️") + ul(d.negatives))


def show_context(d, report):
    heading("Topics & Legislative Context", "🧭", "context")
    show(chips([("Bill type", d.bill_type or "n/a"), ("Ministry", d.ministry or "n/a"), ("Policy area", d.policy_area or "n/a")])
         + sub("Topics", "🏷️")
         + ul([f"<b>{esc(t.name)}</b> ({esc(', '.join(t.keywords))}) – {esc(t.description)}" for t in d.topics], raw=True)
         + sub("Stakeholders", "👤") + ul(d.stakeholders) + sub("Related Laws", "📚") + ul(d.related_laws))


def show_overview(report):
    bill, topics = report["bill"], report["topics"]
    heading("Document Overview", "📑", "overview")
    if topics.get("note"):
        st.warning(topics["note"])
    show(chips([("Pages", bill["pages"]), ("Words", f"{len(bill['text'].split()):,}"), ("Sections", len(bill["sections"])),
                ("Other Acts", len(bill["acts"]))]) + sub("Clause types (rule-based classification)", "🏷️"))
    st.bar_chart(pd.Series(Counter(s["type"] for s in bill["sections"]), name="sections"))
    show(sub("Other Acts referenced", "📚") + ul(bill["acts"], empty="No other Acts found.")
         + sub("Top keywords (TF-IDF)", "🔑")
         + f'<p class="small">{esc(", ".join(topics["keywords"]) or "Not enough text for keyword extraction.")}</p>'
         + sub("Topic groups (NMF topic modelling)", "🧩")
         + ul([f"<b>Topic {i}:</b> {esc(', '.join(w))}" for i, w in enumerate(topics["topics"], 1)], raw=True,
              empty="No topic groups were produced.")
         + sub("Dates and time limits", "📅") + ul(bill["dates"] + bill["deadlines"], empty="None found."))


def show_sections(report):
    heading("Section Details", "🔎", "sections")
    rows = [{"Section": s["id"], "Title": s["title"], "Clause type": s["type"], "Words": s["words"],
             "Obligations": s["modals"]["Obligation"], "Prohibitions": s["modals"]["Prohibition"],
             "Permissions": s["modals"]["Permission"]} for s in report["bill"]["sections"]]
    st.dataframe(rows, hide_index=True)
    st.caption("Obligation = shall/must, Prohibition = shall not/no person, Permission = may.")


# (anchor, menu label, icon, key of the LLM result or None). The report shows them in this order and
# the sidebar links jump to the anchors.
SECTIONS = [
    ("summary", "Summary", "👥", "summary"),
    ("industry", "Industry Impact", "🏭", "industry"),
    ("timeline", "Timeline & Graph", "🕐", "chronology"),
    ("impact", "Impact Assessment", "📊", "impact"),
    ("proscons", "Positives & Negatives", "⚖️", "proscons"),
    ("context", "Topics & Context", "🧭", "context"),
    ("overview", "Document Overview", "📑", None),
    ("sections", "Section Details", "🔎", None),
]
VIEWS = {"summary": show_summary, "industry": show_industry, "chronology": show_chronology,
         "impact": show_impact, "proscons": show_proscons, "context": show_context}


def sidebar_nav(report) -> None:
    links = "".join(f'<a href="#{anchor}" target="_self">{icon} {esc(label)}</a>' for anchor, label, icon, _ in SECTIONS)
    show(f'<div class="navhead">Jump to</div><nav class="navlist">{links}</nav><div class="navhead">Export</div>')
    st.download_button("⬇️  Download report (.md)", to_markdown(report["bill"], report["topics"], report["results"]),
                       file_name="ppiia_report.md", key="download_btn")


def show_report(report) -> None:
    """The whole report as one continuous page; the sidebar links scroll to each section."""
    show(f'<div class="rpt-title">📜 {esc(report["bill"]["title"])} – AI Analysis</div>')
    if report["truncated"]:
        st.warning(f"This bill is very long, so only the first {MAX_CHUNKS * CHUNK_CHARS:,} characters were analysed.")
    for anchor, label, icon, key in SECTIONS:
        if key is None:
            (show_overview if anchor == "overview" else show_sections)(report)
            continue
        data = report["results"].get(key)
        if data is None:
            heading(label, icon, anchor)
            st.warning(f"{LABELS[key]} could not be generated ({report['errors'].get(key, 'unknown error')}). "
                       "Open 'Analyze a different bill' and run it again.")
        else:
            VIEWS[key](data, report)
    show(FOOTER)


# ------------------------------------------------------------------------ app
def main() -> None:
    st.set_page_config(page_title="PPIIA - Policy Insight & Impact Analyzer", page_icon="📜", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)

    # Base URL and model come from secrets only, so a visitor can never redirect the owner's key.
    base_url = get_secret("LLM_BASE_URL", DEFAULT_BASE_URL)
    model = get_secret("LLM_MODEL", DEFAULT_MODEL)
    api_key = get_secret("LLM_API_KEY") or get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY")
    report = st.session_state.get("report")

    with st.sidebar:
        show('<div class="brand">📜 PPIIA<small>Policy Insight & Impact Analyzer</small></div>')
        if report:
            sidebar_nav(report)
            enable_scroll_highlight()
            st.divider()
        if api_key:
            st.caption("🔒 API key loaded")
        else:
            api_key = st.text_input("Gemini API key", type="password", help="Free at aistudio.google.com/apikey")
            st.caption("Tip: save it once in .streamlit/secrets.toml so you never type it again.")
        st.caption(f"Model: `{model}`")

    if report:
        with st.expander("➕ Analyze a different bill"):
            input_panel(api_key, base_url, model)
        show_report(report)
    else:
        show('<div class="rpt-title">📜 Public Policy Insight & Impact Analyzer</div>'
             '<p class="lead">Load a bill to get a plain-language summary, industry impact, a timeline, an impact '
             'assessment and pros and cons.</p>')
        input_panel(api_key, base_url, model)
        show(FOOTER)


if __name__ == "__main__":
    main()