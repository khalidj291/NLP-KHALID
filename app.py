"""
Resume Analyzer & Job Matcher
==============================
A Streamlit application that analyzes resumes against job descriptions
using NLP techniques: tokenization, stop-word removal, lemmatization,
N-grams, POS tagging, chunking, NER, and TF-IDF cosine similarity.

Run with:  streamlit run app.py
"""

import streamlit as st
import pdfplumber
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from io import BytesIO

# Local modules
from nlp_pipeline import (
    ensure_nltk_data,
    clean_text,
    preprocess,
    extract_all_ngrams,
    pos_tagging,
    extract_noun_phrases_nltk,
    extract_noun_phrases_spacy,
    extract_entities_nltk,
    extract_entities_spacy,
    extract_skills,
    keyword_frequency,
)
from matcher import analyze_match, generate_suggestions
from skills_database import ALL_SKILLS, categorize_skill

# ─── Page Configuration ──────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Analyzer & Job Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Styles ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }

    /* ── Header ── */
    .main-header {
        text-align: center;
        padding: 2rem 1rem 1rem;
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #a0aec0;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(12px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(102,126,234,0.15);
    }
    .glass-card h3 {
        color: #e2e8f0;
        margin-top: 0;
        font-weight: 600;
    }

    /* ── Metric Cards ── */
    .metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
    .metric-card {
        flex: 1;
        min-width: 140px;
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 0.3rem;
    }

    /* ── Skill Tags ── */
    .skill-tag {
        display: inline-block;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 500;
        margin: 0.2rem;
        transition: transform 0.15s;
    }
    .skill-tag:hover { transform: scale(1.05); }
    .skill-matched {
        background: rgba(72,187,120,0.15);
        color: #68d391;
        border: 1px solid rgba(72,187,120,0.3);
    }
    .skill-missing {
        background: rgba(245,101,101,0.15);
        color: #fc8181;
        border: 1px solid rgba(245,101,101,0.3);
    }
    .skill-extra {
        background: rgba(99,179,237,0.15);
        color: #90cdf4;
        border: 1px solid rgba(99,179,237,0.3);
    }

    /* ── Suggestion Box ── */
    .suggestion-box {
        background: rgba(102,126,234,0.08);
        border-left: 4px solid #667eea;
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        color: #e2e8f0;
        font-size: 0.95rem;
    }

    /* ── Entity Tag ── */
    .entity-tag {
        display: inline-block;
        padding: 0.3rem 0.7rem;
        border-radius: 8px;
        font-size: 0.8rem;
        margin: 0.15rem;
        font-weight: 500;
    }
    .ent-PERSON   { background: rgba(237,137,54,0.2); color: #fbd38d; border: 1px solid rgba(237,137,54,0.3); }
    .ent-ORG      { background: rgba(129,140,248,0.2); color: #a5b4fc; border: 1px solid rgba(129,140,248,0.3); }
    .ent-GPE, .ent-LOC { background: rgba(52,211,153,0.2); color: #6ee7b7; border: 1px solid rgba(52,211,153,0.3); }
    .ent-DATE     { background: rgba(244,114,182,0.2); color: #f9a8d4; border: 1px solid rgba(244,114,182,0.3); }
    .ent-MONEY    { background: rgba(251,191,36,0.2); color: #fcd34d; border: 1px solid rgba(251,191,36,0.3); }
    .ent-default  { background: rgba(148,163,184,0.2); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.3); }

    /* ── NLP Pipeline Step ── */
    .pipeline-step {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
    }
    .pipeline-step h4 {
        color: #a78bfa;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    .pipeline-step code {
        font-size: 0.82rem;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(15,12,41,0.95);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #e2e8f0;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        padding: 0.6rem 1.2rem;
        color: #a0aec0;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(102,126,234,0.2) !important;
        color: #667eea !important;
        border-color: rgba(102,126,234,0.4) !important;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 3px; }

    /* ── Hide Streamlit Branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Ensure NLP resources ────────────────────────────────────────────
ensure_nltk_data()


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from an uploaded PDF using pdfplumber."""
    text = ""
    with pdfplumber.open(BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


def render_gauge(score: float, title: str = "Match Score"):
    """Render a Plotly gauge chart for the match score."""
    if score >= 75:
        bar_color = "#48bb78"
    elif score >= 50:
        bar_color = "#ecc94b"
    elif score >= 30:
        bar_color = "#ed8936"
    else:
        bar_color = "#f56565"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={"suffix": "%", "font": {"size": 52, "color": "#e2e8f0", "family": "Inter"}},
        title={"text": title, "font": {"size": 18, "color": "#a0aec0", "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#4a5568",
                     "tickfont": {"color": "#a0aec0"}},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "rgba(245,101,101,0.1)"},
                {"range": [30, 50], "color": "rgba(237,137,54,0.1)"},
                {"range": [50, 75], "color": "rgba(236,201,75,0.1)"},
                {"range": [75, 100], "color": "rgba(72,187,120,0.1)"},
            ],
            "threshold": {
                "line": {"color": "#e2e8f0", "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
        font={"family": "Inter"},
    )
    return fig


def render_keyword_chart(keywords, title="Top Keywords"):
    """Horizontal bar chart for keyword frequencies."""
    if not keywords:
        return None
    words, counts = zip(*keywords)
    df = pd.DataFrame({"Keyword": words, "Frequency": counts})
    df = df.sort_values("Frequency", ascending=True)

    fig = px.bar(
        df, x="Frequency", y="Keyword", orientation="h",
        color="Frequency",
        color_continuous_scale=["#667eea", "#764ba2", "#f093fb"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter", "color": "#a0aec0"},
        xaxis={"gridcolor": "rgba(255,255,255,0.05)", "title": ""},
        yaxis={"title": ""},
        coloraxis_showscale=False,
        height=max(300, len(keywords) * 28),
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
    )
    fig.update_traces(
        marker_line_width=0,
        textposition="outside",
        textfont={"color": "#e2e8f0"},
    )
    return fig


def render_skills_html(skills, css_class):
    """Generate HTML tags for a set of skills."""
    if not skills:
        return "<p style='color:#718096;font-size:0.9rem;'>None</p>"
    html = ""
    for skill in sorted(skills):
        html += f'<span class="skill-tag {css_class}">{skill}</span>'
    return html


def render_entities_html(entities_dict):
    """Generate coloured entity tags."""
    if not entities_dict:
        return "<p style='color:#718096;font-size:0.9rem;'>No entities detected</p>"
    html = ""
    for label, ents in entities_dict.items():
        css = f"ent-{label}" if label in ("PERSON", "ORG", "GPE", "LOC", "DATE", "MONEY") else "ent-default"
        for e in ents:
            html += f'<span class="entity-tag {css}">{e} <small>({label})</small></span>'
    return html


# =====================================================================
# MAIN APP
# =====================================================================

def main():
    # ── Header ──
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Resume Analyzer & Job Matcher</h1>
        <p>Upload your resume and paste a job description to get an AI-powered match analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 📄 Upload Resume")
        upload_method = st.radio(
            "Input method",
            ["Upload PDF", "Paste Text"],
            horizontal=True,
            label_visibility="collapsed",
        )

        resume_text = ""
        if upload_method == "Upload PDF":
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=["pdf"],
                help="Upload your resume in PDF format",
            )
            if uploaded_file is not None:
                with st.spinner("Extracting text from PDF…"):
                    resume_text = extract_text_from_pdf(uploaded_file)
                st.success(f"Extracted {len(resume_text.split())} words")
        else:
            resume_text = st.text_area(
                "Paste your resume text",
                height=250,
                placeholder="Paste the full text of your resume here…",
            )

        st.markdown("---")
        st.markdown("## 💼 Job Description")
        job_text = st.text_area(
            "Paste the job description",
            height=250,
            placeholder="Paste the job description you want to match against…",
        )

        st.markdown("---")
        analyze_btn = st.button(
            "🔍  Analyze Match",
            use_container_width=True,
            type="primary",
        )

        st.markdown("---")
        st.markdown(
            "<p style='text-align:center;color:#718096;font-size:0.75rem;'>"
            "Built with spaCy · NLTK · scikit-learn · Streamlit</p>",
            unsafe_allow_html=True,
        )

    # ── Analysis ──
    if analyze_btn:
        if not resume_text.strip():
            st.error("⚠️  Please provide your resume text or upload a PDF.")
            return
        if not job_text.strip():
            st.error("⚠️  Please paste a job description.")
            return

        with st.spinner("🔬 Running NLP pipeline…"):
            result = analyze_match(resume_text, job_text)
            suggestions = generate_suggestions(result)
            resume_entities = extract_entities_spacy(resume_text)
            resume_nouns = extract_noun_phrases_spacy(resume_text)

        # ──────── RESULTS ────────

        # ── Score Cards Row ──
        score = result["match_score"]
        matched_count = len(result["matched_skills"])
        missing_count = len(result["missing_skills"])
        extra_count = len(result["resume_only_skills"])

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-value" style="color:{'#48bb78' if score>=70 else '#ecc94b' if score>=40 else '#f56565'};">
                    {score}%
                </div>
                <div class="metric-label">Overall Match</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:#48bb78;">{matched_count}</div>
                <div class="metric-label">Matched Skills</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:#f56565;">{missing_count}</div>
                <div class="metric-label">Missing Skills</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:#90cdf4;">{extra_count}</div>
                <div class="metric-label">Extra Skills</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="color:#d6bcfa;">{result['tfidf_score']}%</div>
                <div class="metric-label">TF-IDF Score</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Tabs ──
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Match Overview",
            "🏷️ Skills Detail",
            "🔤 NLP Pipeline",
            "🏛️ NER & Entities",
            "📈 Keywords",
        ])

        # ────── Tab 1: Match Overview ──────
        with tab1:
            col_gauge, col_suggest = st.columns([1, 1])

            with col_gauge:
                st.plotly_chart(render_gauge(score), use_container_width=True)

                # Sub-scores
                st.markdown(f"""
                <div class="glass-card">
                    <h3>Score Breakdown</h3>
                    <p style="color:#cbd5e1;">
                        <strong>Skill Overlap:</strong> {result['skill_score']}% — 
                        {matched_count} of {len(result['job_skills'])} required skills matched<br>
                        <strong>TF-IDF Similarity:</strong> {result['tfidf_score']}% — 
                        Measures overall language overlap<br><br>
                        <em style="color:#a0aec0;">Final score = 60% skill overlap + 40% TF-IDF similarity</em>
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with col_suggest:
                st.markdown('<div class="glass-card"><h3>💡 Improvement Suggestions</h3>', unsafe_allow_html=True)
                for s in suggestions:
                    st.markdown(f'<div class="suggestion-box">{s}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # ────── Tab 2: Skills Detail ──────
        with tab2:
            col_m, col_x = st.columns(2)

            with col_m:
                st.markdown(f"""
                <div class="glass-card">
                    <h3>✅ Matched Skills ({matched_count})</h3>
                    {render_skills_html(result['matched_skills'], 'skill-matched')}
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="glass-card">
                    <h3>❌ Missing Skills ({missing_count})</h3>
                    {render_skills_html(result['missing_skills'], 'skill-missing')}
                </div>
                """, unsafe_allow_html=True)

            with col_x:
                st.markdown(f"""
                <div class="glass-card">
                    <h3>🔵 Extra Skills on Resume ({extra_count})</h3>
                    <p style="color:#718096;font-size:0.85rem;">
                        Skills found on your resume but not required by the job
                    </p>
                    {render_skills_html(result['resume_only_skills'], 'skill-extra')}
                </div>
                """, unsafe_allow_html=True)

                # Category breakdown
                st.markdown('<div class="glass-card"><h3>📂 By Category</h3>', unsafe_allow_html=True)
                for cat, skills in result["matched_categories"].items():
                    st.markdown(
                        f"<p style='color:#68d391;font-weight:600;margin-bottom:0.2rem;'>{cat}</p>"
                        + render_skills_html(skills, 'skill-matched'),
                        unsafe_allow_html=True,
                    )
                for cat, skills in result["missing_categories"].items():
                    st.markdown(
                        f"<p style='color:#fc8181;font-weight:600;margin-bottom:0.2rem;'>{cat} (missing)</p>"
                        + render_skills_html(skills, 'skill-missing'),
                        unsafe_allow_html=True,
                    )
                st.markdown('</div>', unsafe_allow_html=True)

        # ────── Tab 3: NLP Pipeline Steps ──────
        with tab3:
            st.markdown('<div class="glass-card"><h3>🔬 NLP Processing Pipeline</h3></div>', unsafe_allow_html=True)

            # Sample from resume (first 500 chars)
            sample = resume_text[:500]

            # Step 1 — Tokenization
            from nlp_pipeline import tokenize, remove_stopwords, lemmatize, clean_text as _clean
            sample_clean = _clean(sample)
            tokens = tokenize(sample_clean)
            st.markdown(f"""
            <div class="pipeline-step">
                <h4>Step 1 · Tokenization</h4>
                <p style="color:#a0aec0;font-size:0.85rem;">Split text into individual tokens using NLTK <code>word_tokenize</code></p>
                <p style="color:#cbd5e1;font-size:0.85rem;"><strong>Tokens ({len(tokens)}):</strong></p>
                <code style="color:#90cdf4;">{' · '.join(tokens[:40])}{'…' if len(tokens)>40 else ''}</code>
            </div>
            """, unsafe_allow_html=True)

            # Step 2 — Stop-word removal
            filtered = remove_stopwords(tokens)
            st.markdown(f"""
            <div class="pipeline-step">
                <h4>Step 2 · Stop-word Removal</h4>
                <p style="color:#a0aec0;font-size:0.85rem;">Removed {len(tokens)-len(filtered)} stop words and punctuation</p>
                <code style="color:#68d391;">{' · '.join(filtered[:40])}{'…' if len(filtered)>40 else ''}</code>
            </div>
            """, unsafe_allow_html=True)

            # Step 3 — Lemmatization
            lemmas = lemmatize(filtered)
            st.markdown(f"""
            <div class="pipeline-step">
                <h4>Step 3 · Lemmatization</h4>
                <p style="color:#a0aec0;font-size:0.85rem;">Reduced words to base forms using WordNet lemmatizer</p>
                <code style="color:#fbd38d;">{' · '.join(lemmas[:40])}{'…' if len(lemmas)>40 else ''}</code>
            </div>
            """, unsafe_allow_html=True)

            # Step 4 — N-grams
            from nlp_pipeline import extract_ngrams
            bigrams = extract_ngrams(lemmas, 2)
            trigrams = extract_ngrams(lemmas, 3)
            st.markdown(f"""
            <div class="pipeline-step">
                <h4>Step 4 · N-gram Extraction</h4>
                <p style="color:#a0aec0;font-size:0.85rem;">Bigrams: {len(bigrams)} · Trigrams: {len(trigrams)}</p>
                <p style="color:#cbd5e1;font-size:0.83rem;"><strong>Sample Bigrams:</strong></p>
                <code style="color:#a5b4fc;">{' · '.join(bigrams[:15])}</code>
                <p style="color:#cbd5e1;font-size:0.83rem;margin-top:0.5rem;"><strong>Sample Trigrams:</strong></p>
                <code style="color:#f9a8d4;">{' · '.join(trigrams[:10])}</code>
            </div>
            """, unsafe_allow_html=True)

            # Step 5 — POS Tagging
            pos_tags = pos_tagging(sample)
            tag_sample = ", ".join([f"{w}/{t}" for w, t in pos_tags[:25]])
            st.markdown(f"""
            <div class="pipeline-step">
                <h4>Step 5 · POS Tagging</h4>
                <p style="color:#a0aec0;font-size:0.85rem;">Part-of-speech tags assigned using NLTK's averaged perceptron tagger</p>
                <code style="color:#fca5a5;">{tag_sample}…</code>
            </div>
            """, unsafe_allow_html=True)

            # Step 6 — Noun Phrase Chunking
            noun_phrases = extract_noun_phrases_spacy(sample)
            st.markdown(f"""
            <div class="pipeline-step">
                <h4>Step 6 · Noun Phrase Chunking</h4>
                <p style="color:#a0aec0;font-size:0.85rem;">Extracted {len(noun_phrases)} noun phrases using spaCy chunker</p>
                <code style="color:#86efac;">{' · '.join(noun_phrases[:20])}</code>
            </div>
            """, unsafe_allow_html=True)

        # ────── Tab 4: NER ──────
        with tab4:
            col_ner1, col_ner2 = st.columns(2)

            with col_ner1:
                st.markdown(f"""
                <div class="glass-card">
                    <h3>🏛️ Resume Entities (spaCy)</h3>
                    {render_entities_html(resume_entities)}
                </div>
                """, unsafe_allow_html=True)

                # NLTK entities
                nltk_entities = extract_entities_nltk(resume_text[:2000])
                st.markdown(f"""
                <div class="glass-card">
                    <h3>🏛️ Resume Entities (NLTK)</h3>
                    {render_entities_html(nltk_entities)}
                </div>
                """, unsafe_allow_html=True)

            with col_ner2:
                job_entities = extract_entities_spacy(job_text)
                st.markdown(f"""
                <div class="glass-card">
                    <h3>💼 Job Description Entities (spaCy)</h3>
                    {render_entities_html(job_entities)}
                </div>
                """, unsafe_allow_html=True)

                # Noun phrases
                job_nouns = extract_noun_phrases_spacy(job_text)
                st.markdown(f"""
                <div class="glass-card">
                    <h3>📝 Key Noun Phrases (Job)</h3>
                    <p style="color:#a0aec0;font-size:0.82rem;">Extracted via spaCy noun chunking</p>
                    <code style="color:#d6bcfa;">{' · '.join(job_nouns[:25])}</code>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="glass-card">
                    <h3>📝 Key Noun Phrases (Resume)</h3>
                    <p style="color:#a0aec0;font-size:0.82rem;">Extracted via spaCy noun chunking</p>
                    <code style="color:#86efac;">{' · '.join(resume_nouns[:25])}</code>
                </div>
                """, unsafe_allow_html=True)

        # ────── Tab 5: Keywords ──────
        with tab5:
            col_k1, col_k2 = st.columns(2)

            with col_k1:
                st.markdown('<div class="glass-card"><h3>💼 Job Description — Top Keywords</h3></div>',
                            unsafe_allow_html=True)
                fig_jk = render_keyword_chart(result["job_keywords"])
                if fig_jk:
                    st.plotly_chart(fig_jk, use_container_width=True)

            with col_k2:
                st.markdown('<div class="glass-card"><h3>📄 Resume — Top Keywords</h3></div>',
                            unsafe_allow_html=True)
                fig_rk = render_keyword_chart(result["resume_keywords"])
                if fig_rk:
                    st.plotly_chart(fig_rk, use_container_width=True)

            # Overlap visualization
            st.markdown('<div class="glass-card"><h3>🔄 Keyword Overlap</h3></div>', unsafe_allow_html=True)
            job_kw_set = {w for w, _ in result["job_keywords"]}
            resume_kw_set = {w for w, _ in result["resume_keywords"]}
            common_kw = job_kw_set & resume_kw_set
            only_job = job_kw_set - resume_kw_set
            only_resume = resume_kw_set - job_kw_set

            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""
                <div style="text-align:center;">
                    <p style="color:#ecc94b;font-weight:600;">Common ({len(common_kw)})</p>
                    {''.join(f'<span class="skill-tag skill-matched">{k}</span>' for k in sorted(common_kw))}
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style="text-align:center;">
                    <p style="color:#fc8181;font-weight:600;">Only in Job ({len(only_job)})</p>
                    {''.join(f'<span class="skill-tag skill-missing">{k}</span>' for k in sorted(only_job))}
                </div>
                """, unsafe_allow_html=True)
            with c3:
                st.markdown(f"""
                <div style="text-align:center;">
                    <p style="color:#90cdf4;font-weight:600;">Only in Resume ({len(only_resume)})</p>
                    {''.join(f'<span class="skill-tag skill-extra">{k}</span>' for k in sorted(only_resume))}
                </div>
                """, unsafe_allow_html=True)

    else:
        # ── Landing State ──
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem;">
            <div style="font-size:5rem; margin-bottom:1rem;">📄 ➜ 🔍 ➜ 🎯</div>
            <h2 style="color:#e2e8f0; font-weight:600;">How It Works</h2>
            <p style="color:#a0aec0; font-size:1.05rem; max-width:600px; margin:0 auto 2rem;">
                Upload your resume and paste a job description in the sidebar,
                then hit <strong>Analyze Match</strong> to see how well you fit.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Feature cards
        features = [
            ("🔤", "Tokenization & Preprocessing", "Text cleaning, stop-word removal, and lemmatization"),
            ("📊", "N-gram Analysis", "Bigram and trigram extraction to capture multi-word skills"),
            ("🏷️", "POS Tagging & Chunking", "Part-of-speech tagging and noun phrase extraction"),
            ("🏛️", "Named Entity Recognition", "Detect names, organizations, locations, and dates"),
            ("📐", "TF-IDF & Cosine Similarity", "Measure semantic overlap between resume and job text"),
            ("✅", "Skill Matching", "Match against 300+ skills across technical, soft, and certifications"),
        ]

        cols = st.columns(3)
        for i, (icon, title, desc) in enumerate(features):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="glass-card" style="text-align:center; min-height:160px;">
                    <div style="font-size:2.5rem; margin-bottom:0.5rem;">{icon}</div>
                    <h3 style="font-size:1rem; margin-bottom:0.3rem;">{title}</h3>
                    <p style="color:#a0aec0; font-size:0.85rem;">{desc}</p>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
