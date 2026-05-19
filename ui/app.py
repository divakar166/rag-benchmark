import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# Config 
API_BASE = "http://localhost:8000"
TIMEOUT_INGEST = 600
TIMEOUT_QUERY = 180
TIMEOUT_EVAL = 1800

STRATEGY_META = {
    "naive":        {"color": "#94a3b8", "label": "Naive RAG"},
    "semantic":     {"color": "#3b82f6", "label": "Semantic"},
    "hierarchical": {"color": "#8b5cf6", "label": "Hierarchical"},
    "hybrid":       {"color": "#f97316", "label": "Hybrid"},
    "hyde":         {"color": "#22c55e", "label": "HyDE"},
}

# Page config 
st.set_page_config(
    page_title="RAG Benchmark",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header */
.dash-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.dash-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(59,130,246,0.08) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(139,92,246,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.dash-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
    letter-spacing: -0.5px;
}
.dash-subtitle {
    font-size: 0.85rem;
    color: #64748b;
    margin: 6px 0 0;
    font-weight: 400;
    letter-spacing: 0.3px;
}

/* Strategy cards */
.strategy-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 20px;
    height: 100%;
    transition: border-color 0.2s;
}
.strategy-card:hover { border-color: #334155; }
.strategy-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.strategy-answer {
    font-size: 0.88rem;
    line-height: 1.65;
    color: #cbd5e1;
}
.metric-pill {
    display: inline-block;
    background: #1e293b;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: #94a3b8;
    margin: 3px 3px 0 0;
    font-family: 'Space Mono', monospace;
}
.hypothesis-box {
    background: #0a1628;
    border-left: 3px solid #22c55e;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin-top: 12px;
    font-size: 0.78rem;
    color: #86efac;
    font-style: italic;
    line-height: 1.6;
}
.source-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 0;
    border-bottom: 1px solid #1e293b;
    font-size: 0.75rem;
    color: #64748b;
}
.score-bar-outer {
    flex: 1;
    background: #1e293b;
    border-radius: 3px;
    height: 6px;
    overflow: hidden;
}
.score-bar-inner {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
}

/* Status badges */
.badge-ok   { color: #22c55e; font-weight: 600; }
.badge-warn { color: #f59e0b; font-weight: 600; }
.badge-err  { color: #ef4444; font-weight: 600; }

/* Metric cards in eval */
.ragas-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.ragas-score {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.ragas-label {
    font-size: 0.72rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 6px;
}

/* Section dividers */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #475569;
    margin: 20px 0 10px;
}

/* Streamlit overrides */
div[data-testid="stTabs"] button {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.5px;
}
div.stButton > button {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)


# Helpers 
def api_get(path: str, timeout: int = 10) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, **kwargs) -> tuple[dict | None, int]:
    try:
        r = requests.post(f"{API_BASE}{path}", **kwargs)
        return r.json(), r.status_code
    except requests.exceptions.ConnectionError:
        return None, 0
    except Exception as e:
        return {"detail": str(e)}, 500


def backend_online() -> bool:
    return api_get("/health") is not None


def score_color(score: float | None) -> str:
    if score is None:
        return "#475569"
    if score >= 0.85:
        return "#22c55e"
    if score >= 0.70:
        return "#f59e0b"
    return "#ef4444"


def fmt_score(score: float | None) -> str:
    return f"{score:.3f}" if score is not None else "N/A"


def fmt_latency(ms: float) -> str:
    if ms >= 1000:
        return f"{ms/1000:.1f}s"
    return f"{ms:.0f}ms"


def load_latest_eval(results_dir: str = "core/evaluation/results") -> dict | None:
    """Load the most recent eval_summary JSON from the results directory."""
    p = Path(results_dir)
    if not p.exists():
        return None
    files = sorted(p.glob("eval_summary_*.json"), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


#  Sidebar 
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
                letter-spacing:2px;color:#475569;text-transform:uppercase;
                margin-bottom:16px;">
        RAG Benchmark
    </div>
    """, unsafe_allow_html=True)

    # Backend status
    online = backend_online()
    if online:
        status_html = """
        <span style="color:#22c55e; font-weight:bold;">
            Backend online
        </span>
        """
    else:
        status_html = """
        <span style="color:#ef4444; font-weight:bold;">
            Backend offline
        </span>
        """
    st.markdown(status_html, unsafe_allow_html=True)

    st.markdown('<div class="section-label">Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=5,
                       help="Number of chunks retrieved per query")

    # Collection status
    st.markdown('<div class="section-label">Collections</div>', unsafe_allow_html=True)
    if st.button("↻ Refresh", width='stretch'):
        st.rerun()

    if online:
        col_data = api_get("/collections")
        if col_data:
            for col in col_data.get("collections", []):
                count = col.get("points_count", 0)
                note  = col.get("note", "")
                label = col.get("strategy", "")

                if note:
                    st.markdown(f"**{label}** — *{note}*", unsafe_allow_html=False)
                
                elif count and count > 0:
                    st.markdown(
                        f"<span style='color:#22c55e;'><strong>{label}</strong></span> — "
                        f"<strong>{count:,}</strong> chunks", 
                        unsafe_allow_html=True
                    )
                
                else:
                    st.markdown(
                        f"<span style='color:#64748b;'>{label} — empty</span>", 
                        unsafe_allow_html=True
                    )

    # st.markdown('<div class="section-label">Strategy Guide</div>', unsafe_allow_html=True)
    # for name, meta in STRATEGY_META.items():
    #     st.caption(f"{meta['icon']} **{meta['label']}**")


#  Header 
st.markdown("""
<div class="dash-header">
    <p class="dash-title">RAG Strategy Benchmarker</p>
    <p class="dash-subtitle">
        Compare Naive · Semantic · Hierarchical · Hybrid · HyDE — side by side
    </p>
</div>
""", unsafe_allow_html=True)

#  Tabs 
tab_query, tab_ingest, tab_eval, tab_status = st.tabs([
    "Query & Compare",
    "Ingest PDF",
    "Evaluation Results",
    "System Status",
])


# TAB 1 — QUERY & COMPARE
with tab_query:
    st.markdown('<div class="section-label">Ask a question</div>', unsafe_allow_html=True)

    col_q, col_opts = st.columns([3, 1])
    with col_q:
        query_text = st.text_area(
            "Question",
            placeholder="e.g. What are special methods in Python and how do they work?",
            height=120,
            label_visibility="collapsed",
        )
    with col_opts:
        strategy_choice = st.selectbox(
            "Strategies",
            options=["all"] + list(STRATEGY_META.keys()),
            format_func=lambda x: "All strategies" if x == "all"
                                  else f"{STRATEGY_META[x]['label']}",
        )
        run_btn = st.button(
            "Run Query",
            type="primary",
            width='stretch',
            disabled=not online or not query_text.strip(),
        )

    if run_btn and query_text.strip():
        with st.spinner("Retrieving and generating answers…"):
            payload = {"query": query_text, "strategy": strategy_choice, "top_k": top_k}
            data, status = api_post("/query", json=payload, timeout=TIMEOUT_QUERY)

        if status != 200 or data is None:
            st.error(f"Query failed: {data.get('detail', 'Unknown error') if data else 'Backend unreachable'}")
        else:
            results = data.get("results", [])

            # Latency chart 
            if len(results) > 1:
                st.markdown('<div class="section-label">Latency comparison</div>',
                            unsafe_allow_html=True)
                lat_fig = go.Figure()
                for r in results:
                    meta = STRATEGY_META.get(r["strategy"], {})
                    lat_fig.add_trace(go.Bar(
                        x=[meta.get("label", r["strategy"])],
                        y=[r["latency_ms"]],
                        marker_color=meta.get("color", "#94a3b8"),
                        text=[fmt_latency(r["latency_ms"])],
                        textposition="outside",
                        name=r["strategy"],
                    ))
                lat_fig.update_layout(
                    showlegend=False,
                    height=220,
                    margin=dict(t=10, b=10, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8", size=11),
                    yaxis=dict(title="ms", gridcolor="#1e293b", showgrid=True),
                    xaxis=dict(gridcolor="#1e293b"),
                    bargap=0.35,
                )
                st.plotly_chart(lat_fig, width='stretch')

            #  Retrieval score radar (multi-strategy only) 
            if len(results) > 1:
                st.markdown('<div class="section-label">Top retrieval score per strategy</div>',
                            unsafe_allow_html=True)
                score_fig = go.Figure()
                for r in results:
                    meta = STRATEGY_META.get(r["strategy"], {})
                    scores = r.get("retrieval_scores", [])
                    if scores:
                        score_fig.add_trace(go.Bar(
                            x=[meta.get("label", r["strategy"])],
                            y=[max(scores)],
                            marker_color=meta.get("color", "#94a3b8"),
                            text=[f"{max(scores):.3f}"],
                            textposition="outside",
                            name=r["strategy"],
                        ))
                score_fig.update_layout(
                    showlegend=False,
                    height=200,
                    margin=dict(t=10, b=10, l=0, r=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8", size=11),
                    yaxis=dict(title="score", gridcolor="#1e293b",
                               range=[0, 1], showgrid=True),
                    xaxis=dict(gridcolor="#1e293b"),
                    bargap=0.35,
                )
                st.plotly_chart(score_fig, width='stretch')

            #  Side-by-side answer cards 
            st.markdown('<div class="section-label">Answers</div>', unsafe_allow_html=True)

            n = len(results)
            cols = st.columns(min(n, 3))

            for i, result in enumerate(results):
                meta = STRATEGY_META.get(result["strategy"], {})
                col = cols[i % len(cols)]

                with col:
                    # Strategy header
                    st.markdown(f"""
                    <div class="strategy-name" style="color:{meta.get('color','#94a3b8')}">
                        {meta.get('label', result['strategy'])}
                    </div>
                    """, unsafe_allow_html=True)

                    # Metric pills
                    total_tok = result["prompt_tokens"] + result["completion_tokens"]
                    st.markdown(f"""
                    <div>
                        <span class="metric-pill">{fmt_latency(result['latency_ms'])}</span>
                        <span class="metric-pill">{result['chunks_used']} chunks</span>
                        <span class="metric-pill">{total_tok} tokens</span>
                    </div>
                    """, unsafe_allow_html=True)

                    # Answer
                    st.markdown(f"""
                    <div class="strategy-answer" style="margin-top:12px;">
                        {result['answer']}
                    </div>
                    """, unsafe_allow_html=True)

                    # HyDE hypothesis
                    if result.get("extra", {}).get("hypothesis"):
                        hyp = result["extra"]["hypothesis"]
                        hyp_lat = result["extra"].get("hypothesis_latency_ms", 0)
                        st.markdown(f"""
                        <div class="hypothesis-box">
                            <strong style="font-style:normal;color:#4ade80;">
                                Hypothesis ({fmt_latency(hyp_lat)})
                            </strong><br>
                            {hyp}
                        </div>
                        """, unsafe_allow_html=True)

                    # Hierarchical meta
                    if result.get("extra", {}).get("child_chunks_searched") is not None:
                        cc = result["extra"]["child_chunks_searched"]
                        pc = result["extra"]["parent_chunks_fetched"]
                        st.markdown(f"""
                        <div style="margin-top:8px;">
                            <span class="metric-pill">{cc} children searched</span>
                            <span class="metric-pill">{pc} parents returned</span>
                        </div>
                        """, unsafe_allow_html=True)

                    # Sources expander
                    if result["sources"]:
                        with st.expander("Sources & scores"):
                            for j, src in enumerate(result["sources"]):
                                score = src.get("score", 0)
                                bar_w = int(score * 100)
                                st.markdown(f"""
                                <div class="source-row">
                                    <span style="width:18px;color:#475569;">{j+1}</span>
                                    <span style="flex:1;">p.{src.get('page','?')}</span>
                                    <div class="score-bar-outer">
                                        <div class="score-bar-inner" style="width:{bar_w}%"></div>
                                    </div>
                                    <span style="width:44px;text-align:right;
                                                 font-family:'Space Mono',monospace;">
                                        {score:.3f}
                                    </span>
                                </div>
                                """, unsafe_allow_html=True)

                    # Separator between rows of cards if > 3 strategies
                    if n > 3 and (i + 1) % 3 == 0 and i + 1 < n:
                        st.markdown("---")


# TAB 2 — INGEST
with tab_ingest:
    st.markdown('<div class="section-label">Upload & ingest a PDF</div>',
                unsafe_allow_html=True)

    st.info(
        "Each strategy gets its own Qdrant collection. "
        "Run **all** to ingest once for every strategy — "
        "takes ~2-5 minutes for a chapter-sized PDF.",
        icon="ℹ",
    )

    uploaded = st.file_uploader("Choose a PDF", type=["pdf"])

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        ingest_strategy = st.selectbox(
            "Strategy",
            options=["all", "naive", "semantic", "hierarchical", "hybrid"],
            help="'all' runs every strategy sequentially",
        )
    with col_b:
        recreate = st.checkbox(
            "Recreate collections",
            value=False,
            help="Drop existing data and start fresh",
        )
    with col_c:
        st.markdown("<br>", unsafe_allow_html=True)
        ingest_btn = st.button(
            "🚀 Ingest",
            type="primary",
            width='stretch',
            disabled=not online or uploaded is None,
        )

    if ingest_btn and uploaded:
        progress = st.progress(0, text="Uploading…")
        with st.spinner("Ingesting — this may take a few minutes…"):
            data, status = api_post(
                "/ingest",
                files={"file": (uploaded.name, uploaded, "application/pdf")},
                data={"strategy": ingest_strategy, "recreate": str(recreate).lower()},
                timeout=TIMEOUT_INGEST,
            )

        progress.progress(100, text="Done")

        if status != 200 or data is None:
            st.error(f"Ingestion failed: {data.get('detail', 'Unknown') if data else 'No response'}")
        else:
            st.success(f"✅ Ingested **{uploaded.name}** successfully")
            summaries = data.get("summaries", [])

            for summary in summaries:
                strategy = summary.get("strategy", "?")
                meta = STRATEGY_META.get(strategy, {})
                icon = meta.get("icon", "⬜")
                label = meta.get("label", strategy)

                with st.expander(f"{icon} {label}", expanded=True):
                    if "note" in summary:
                        st.info(summary["note"])
                        continue

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Pages", summary.get("pages_loaded", "—"))
                    c2.metric("Chunks", summary.get("chunks_created",
                                summary.get("parent_chunks", "—")))
                    c3.metric("Vectors", summary.get("vectors_stored", "—"))
                    c4.metric("Collection",
                              summary.get("collection",
                                          str(summary.get("collections", "—"))))


# TAB 3 — EVALUATION RESULTS
with tab_eval:
    st.markdown('<div class="section-label">RAGAS evaluation results</div>',
                unsafe_allow_html=True)

    eval_data = load_latest_eval()

    if eval_data is None:
        st.warning(
            "No evaluation results found. Run the benchmark first:\n\n"
        )
    else:
        ts = eval_data.get("timestamp", "unknown")
        q_set = eval_data.get("question_set", "unknown")
        top_k_used = eval_data.get("top_k", "?")
        strategies_data = eval_data.get("strategies", {})

        st.caption(f"Results from `{ts}` · question set: `{q_set}` · top-k: {top_k_used}")

        #  RAGAS score overview table 
        METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        METRIC_LABELS = {
            "faithfulness":      "Faithfulness",
            "answer_relevancy":  "Ans. Relevancy",
            "context_precision": "Ctx. Precision",
            "context_recall":    "Ctx. Recall",
        }

        rows = []
        for strat, data in strategies_data.items():
            ragas = data.get("ragas", {})
            row = {
                "Strategy": STRATEGY_META.get(strat, {}).get("label", strat),
                **{METRIC_LABELS[m]: ragas.get(m) for m in METRICS},
                "Avg Latency": data.get("latency_avg_ms"),
                "Avg Tokens": data.get("tokens_avg"),
                "N": data.get("questions_evaluated", 0),
            }
            rows.append((strat, row))

        # Sort by faithfulness
        rows.sort(key=lambda x: x[1].get("Faithfulness") or 0, reverse=True)

        #  Radar / grouped bar chart 
        st.markdown('<div class="section-label">Score comparison</div>',
                    unsafe_allow_html=True)

        has_scores = any(
            r[1].get("Faithfulness") is not None
            for r in rows
        )

        if has_scores:
            bar_fig = go.Figure()
            for strat, row in rows:
                meta = STRATEGY_META.get(strat, {})
                y_vals = [row.get(METRIC_LABELS[m]) for m in METRICS]
                bar_fig.add_trace(go.Bar(
                    name=meta.get("label", strat),
                    x=[METRIC_LABELS[m] for m in METRICS],
                    y=y_vals,
                    marker_color=meta.get("color", "#94a3b8"),
                    text=[f"{v:.3f}" if v is not None else "N/A" for v in y_vals],
                    textposition="outside",
                ))
            bar_fig.update_layout(
                barmode="group",
                height=360,
                margin=dict(t=20, b=20, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=11),
                yaxis=dict(range=[0, 1.15], gridcolor="#1e293b",
                           title="score", showgrid=True),
                xaxis=dict(gridcolor="#1e293b"),
                legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h",
                            yanchor="bottom", y=1.02),
                bargap=0.15,
                bargroupgap=0.05,
            )
            st.plotly_chart(bar_fig, width='stretch')
        else:
            st.info("Scores are N/A — run evaluation with a working RAGAS setup first.")

        #  Per-strategy score cards 
        st.markdown('<div class="section-label">Per-strategy breakdown</div>',
                    unsafe_allow_html=True)

        for strat, row in rows:
            meta = STRATEGY_META.get(strat, {})
            with st.expander(
                f"{meta.get('label', strat)}  "
                f"· {row['N']} questions  "
                f"· avg {fmt_latency(row['Avg Latency'] or 0)}",
                expanded=(strat == rows[0][0]),
            ):
                c1, c2, c3, c4 = st.columns(4)
                for col, metric in zip([c1, c2, c3, c4], METRICS):
                    score = row.get(METRIC_LABELS[metric])
                    color = score_color(score)
                    col.markdown(f"""
                    <div class="ragas-card">
                        <div class="ragas-score" style="color:{color}">
                            {fmt_score(score)}
                        </div>
                        <div class="ragas-label">{METRIC_LABELS[metric]}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-top:12px;display:flex;gap:12px;padding-bottom:5px;">
                    <span class="metric-pill">avg {fmt_latency(row['Avg Latency'] or 0)}</span>
                    <span class="metric-pill">avg {int(row['Avg Tokens'] or 0)} tokens</span>
                    <span class="metric-pill">{row['N']} questions</span>
                </div>
                """, unsafe_allow_html=True)

        #  Latency vs Quality scatter 
        if has_scores:
            st.markdown('<div class="section-label">Latency vs faithfulness tradeoff</div>',
                        unsafe_allow_html=True)
            scatter_fig = go.Figure()
            for strat, row in rows:
                meta = STRATEGY_META.get(strat, {})
                faith = row.get("Faithfulness")
                lat   = row.get("Avg Latency")
                if faith is not None and lat is not None:
                    scatter_fig.add_trace(go.Scatter(
                        x=[lat],
                        y=[faith],
                        mode="markers+text",
                        name=meta.get("label", strat),
                        text=[meta.get("label", strat)],
                        textposition="top center",
                        marker=dict(
                            size=18,
                            color=meta.get("color", "#94a3b8"),
                            line=dict(width=2, color="#0f172a"),
                        ),
                        textfont=dict(size=10, color="#94a3b8"),
                    ))
            scatter_fig.update_layout(
                height=320,
                showlegend=False,
                margin=dict(t=20, b=20, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=11),
                xaxis=dict(title="Avg Latency (ms)", gridcolor="#1e293b", showgrid=True),
                yaxis=dict(title="Faithfulness", range=[0, 1.1],
                           gridcolor="#1e293b", showgrid=True),
                annotations=[dict(
                    x=0.02, y=0.98, xref="paper", yref="paper",
                    text="↑ better quality", showarrow=False,
                    font=dict(size=10, color="#475569"),
                )],
            )
            st.plotly_chart(scatter_fig, width='stretch')
            st.caption("Top-left = fast AND faithful. Top-right = faithful but slow.")

        #  Load CSV for per-question drill-down 
        st.markdown('<div class="section-label">Per-question drill-down</div>',
                    unsafe_allow_html=True)
        csv_files = sorted(Path("core/evaluation/results").glob("eval_per_question_*.csv"),
                           reverse=True) if Path("core/evaluation/results").exists() else []
        if csv_files:
            df = pd.read_csv(csv_files[0])
            filter_strat = st.selectbox(
                "Filter by strategy",
                options=["all"] + list(df["strategy"].unique()),
            )
            display_df = df if filter_strat == "all" else df[df["strategy"] == filter_strat]
            cols_to_show = ["strategy", "user_input", "response", "latency_ms",
                            "prompt_tokens", "completion_tokens", "chunks_used"]
            cols_to_show = [c for c in cols_to_show if c in display_df.columns]
            st.dataframe(display_df[cols_to_show], width='stretch', height=300)

            csv_bytes = display_df.to_csv(index=False).encode()
            st.download_button(
                "Download CSV",
                data=csv_bytes,
                file_name=f"rag_eval_{filter_strat}.csv",
                mime="text/csv",
            )
        else:
            st.caption("No per-question CSV found yet.")


# TAB 4 — SYSTEM STATUS
with tab_status:
    st.markdown('<div class="section-label">System health</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        health = api_get("/health")
        if health:
            st.success(f"API online — v{health.get('version', '?')}")
        else:
            st.error("API offline")

    with c2:
        col_data = api_get("/collections")
        if col_data:
            total = sum(
                c.get("points_count", 0)
                for c in col_data.get("collections", [])
                if isinstance(c.get("points_count"), int)
            )
            st.info(f"Qdrant reachable — {total:,} total vectors")
        else:
            st.warning("Qdrant status unknown")

    #  Collections table 
    st.markdown('<div class="section-label">Qdrant collections</div>',
                unsafe_allow_html=True)

    if col_data:
        col_rows = []
        for c in col_data.get("collections", []):
            col_rows.append({
                "Strategy": c.get("strategy", "—"),
                "Collection": c.get("name", c.get("note", "—")),
                "Chunks": c.get("points_count", 0),
                "Status": c.get("status", "shared" if c.get("note") else "empty"),
            })
        st.dataframe(pd.DataFrame(col_rows), width='stretch', hide_index=True)

    # Strategy descriptions
    st.markdown('<div class="section-label">Strategy reference</div>',
                unsafe_allow_html=True)

    strat_data = api_get("/strategies")
    if strat_data:
        descriptions = strat_data.get("descriptions", {})
        for name, desc in descriptions.items():
            meta = STRATEGY_META.get(name, {})
            st.markdown(f"""
            <div style="padding:10px 0;border-bottom:1px solid #1e293b;">
                <span style="font-family:'Space Mono',monospace;font-size:0.78rem;
                             color:{meta.get('color','#94a3b8')};">
                    {meta.get('label', name)}
                </span>
                <span style="font-size:0.82rem;color:#64748b;margin-left:12px;">
                    {desc}
                </span>
            </div>
            """, unsafe_allow_html=True)