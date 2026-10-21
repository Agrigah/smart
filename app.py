"""Lab 02 - Smart Keyboard V1 - English light theme with animations.

Run:  streamlit run app.py        (from the 02_smart_keyboard_v1 folder)
"""
import json
import os

import pandas as pd
import streamlit as st

from src.pipeline import (BOS, EOS, UNK, InterpolatedModel, NGramModel, backoff_top_k,
                          load_documents, split_by_document, top_k_accuracy)

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "data", "corpus.csv")
RESULTS = os.path.join(HERE, "results")

MODEL_CHOICES = [
    "Trigram",
    "Bigram",
    "Unigram",
    "Back-off (3 → 2 → 1)",
    "Interpolated (0.6 / 0.3 / 0.1)",
]
K_VALUES = [0.0, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

st.set_page_config(
    page_title="Smart Keyboard V1",
    page_icon="⌨️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ light theme + animations
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

    /* ---------- base ---------- */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
        color: #0f172a;
    }

    /* animated top ribbon */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0; left: 0; right: 0;
        height: 5px;
        background: linear-gradient(90deg, #6366f1, #ec4899, #f59e0b, #10b981, #6366f1);
        background-size: 300% 100%;
        animation: ribbon 6s linear infinite;
        z-index: 999999;
    }
    @keyframes ribbon {
        0%   { background-position: 0% 50%; }
        100% { background-position: 300% 50%; }
    }

    /* sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
        border-right: 1px solid #e2e8f0;
    }
    [data-testid="stSidebar"] * { color: #1e293b; }

    /* ---------- hero header ---------- */
    .hero {
        background: linear-gradient(135deg, #eef2ff 0%, #fce7f3 50%, #fef3c7 100%);
        padding: 32px 36px;
        border-radius: 24px;
        margin-bottom: 24px;
        border: 1px solid #e0e7ff;
        box-shadow: 0 20px 60px -30px rgba(99,102,241,.45);
        position: relative;
        overflow: hidden;
        animation: heroIn .8s ease-out;
    }
    @keyframes heroIn {
        from { opacity: 0; transform: translateY(-14px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .hero::before {
        content: "";
        position: absolute;
        inset: -50%;
        background: conic-gradient(from 0deg, transparent, rgba(139,92,246,.15), transparent 30%);
        animation: rotate 12s linear infinite;
        pointer-events: none;
    }
    @keyframes rotate { to { transform: rotate(360deg); } }

    .hero-inner { position: relative; z-index: 1; }
    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 900;
        letter-spacing: -.03em;
        background: linear-gradient(90deg, #4f46e5, #db2777, #f59e0b, #4f46e5);
        background-size: 300% 100%;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 5s linear infinite;
    }
    @keyframes shimmer {
        0%   { background-position: 0% 50%; }
        100% { background-position: 300% 50%; }
    }
    .hero p {
        margin: 8px 0 0 0;
        font-size: 1.05rem;
        color: #475569;
        font-weight: 500;
    }
    .hero .kbd-emoji {
        display: inline-block;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0) rotate(-4deg); }
        50%      { transform: translateY(-8px) rotate(4deg); }
    }

    /* ---------- tabs ---------- */
    [data-baseweb="tab-list"] {
        gap: 8px;
        background: #ffffff;
        padding: 8px;
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 16px -8px rgba(15,23,42,.08);
    }
    [data-baseweb="tab"] {
        border-radius: 12px !important;
        padding: 10px 18px !important;
        font-weight: 600;
        color: #64748b !important;
        transition: all .25s ease;
    }
    [data-baseweb="tab"]:hover { color: #4f46e5 !important; background: #eef2ff; }
    [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: #fff !important;
        box-shadow: 0 6px 18px -8px rgba(99,102,241,.6);
    }
    [data-testid="stTabPanel"] {
        animation: fadeUp .45s ease-out;
    }
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ---------- buttons / suggestion chips ---------- */
    div[data-testid="stButton"] > button {
        border-radius: 16px;
        border: 1px solid #e2e8f0;
        background: #ffffff;
        color: #1e293b;
        font-weight: 600;
        transition: all .22s cubic-bezier(.34,1.56,.64,1);
        padding: 14px 18px;
        box-shadow: 0 2px 8px -4px rgba(15,23,42,.08);
    }
    div[data-testid="stButton"] > button:hover {
        border-color: #8b5cf6;
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 12px 28px -12px rgba(139,92,246,.55);
        background: #faf5ff;
    }
    div[data-testid="stButton"] > button:active {
        transform: translateY(-1px) scale(.99);
    }
    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border: none;
        color: #fff;
        box-shadow: 0 8px 24px -10px rgba(99,102,241,.65);
        animation: pulse 2.4s ease-in-out infinite;
    }
    @keyframes pulse {
        0%, 100% { box-shadow: 0 8px 24px -10px rgba(99,102,241,.65); }
        50%      { box-shadow: 0 12px 32px -8px  rgba(139,92,246,.85); }
    }

    /* ---------- metric cards ---------- */
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 16px 18px;
        box-shadow: 0 4px 16px -10px rgba(15,23,42,.12);
        transition: all .25s ease;
        animation: fadeUp .5s ease-out both;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 14px 30px -14px rgba(99,102,241,.4);
        border-color: #c7d2fe;
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #4f46e5 !important; font-weight: 800; }

    /* stagger metric cards */
    [data-testid="column"]:nth-child(1) [data-testid="stMetric"] { animation-delay: .05s; }
    [data-testid="column"]:nth-child(2) [data-testid="stMetric"] { animation-delay: .12s; }
    [data-testid="column"]:nth-child(3) [data-testid="stMetric"] { animation-delay: .19s; }
    [data-testid="column"]:nth-child(4) [data-testid="stMetric"] { animation-delay: .26s; }

    /* ---------- text input ---------- */
    .stTextInput input {
        background: #ffffff !important;
        border: 2px solid #e0e7ff !important;
        border-radius: 16px !important;
        color: #0f172a !important;
        font-size: 1.15rem !important;
        padding: 16px 18px !important;
        transition: all .25s ease !important;
        box-shadow: 0 4px 16px -10px rgba(99,102,241,.25) !important;
    }
    .stTextInput input:focus {
        border-color: #8b5cf6 !important;
        box-shadow: 0 0 0 4px rgba(139,92,246,.18),
                    0 8px 24px -12px rgba(139,92,246,.45) !important;
        transform: translateY(-1px);
    }

    /* ---------- dataframes ---------- */
    [data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 16px -12px rgba(15,23,42,.15);
        animation: fadeUp .5s ease-out;
    }

    /* alerts */
    [data-testid="stAlert"] {
        border-radius: 14px;
        animation: fadeUp .4s ease-out;
    }

    /* misc */
    hr { border-color: #e2e8f0 !important; }
    h2, h3 { color: #0f172a !important; letter-spacing: -.01em; }
    .stCaption, [data-testid="stCaptionContainer"] { color: #64748b !important; }

    /* bar chart container */
    [data-testid="stArrowVegaLiteChart"] {
        background: #ffffff;
        border-radius: 16px;
        padding: 12px;
        border: 1px solid #e2e8f0;
        animation: fadeUp .5s ease-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ data & models
@st.cache_data
def get_data():
    docs = load_documents(CORPUS)
    train, test = split_by_document(docs)
    return docs, train, test


@st.cache_resource
def get_models(k: float, use_full_corpus: bool):
    docs, train, _ = get_data()
    sents = [s for d in docs.values() for s in d] if use_full_corpus else train
    tri, bi, uni = (NGramModel(sents, n, k) for n in (3, 2, 1))
    interp = InterpolatedModel(sents, k=max(k, 1e-9))
    return {"tri": tri, "bi": bi, "uni": uni, "interp": interp}


def suggest(models, choice, words, n):
    if choice == "Trigram":
        return models["tri"].top_k(words, n), "3-gram"
    if choice == "Bigram":
        return models["bi"].top_k(words, n), "2-gram"
    if choice == "Unigram":
        return models["uni"].top_k(words, n), "1-gram"
    if choice.startswith("Back-off"):
        return backoff_top_k(models["tri"], models["bi"], models["uni"], words, n)
    return models["interp"].top_k(words, n), "interpolated"


def get_model_object(models, choice):
    return {"Trigram": models["tri"], "Bigram": models["bi"], "Unigram": models["uni"]}.get(
        choice, models["interp"] if choice.startswith("Interp") else None)


def tokenize(text):
    return text.lower().replace(".", " . ").split()


def label(w):
    return "⏎ end of sentence" if w == EOS else w


# ------------------------------------------------------------------ keyboard state
if "text" not in st.session_state:
    st.session_state.text = "the happy girls"


def add_word(w):
    if w == EOS:
        st.session_state.text = (st.session_state.text.strip() + " .").strip()
    else:
        st.session_state.text = (st.session_state.text.strip() + " " + w).strip()


def backspace():
    st.session_state.text = " ".join(st.session_state.text.split()[:-1])


def clear():
    st.session_state.text = ""


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    choice = st.selectbox("Model", MODEL_CHOICES, index=0)
    k = st.select_slider(
        "Add-k smoothing  (k)",
        options=K_VALUES,
        value=0.1,
        help="k = 0: no smoothing (probability 0 for unseen sequences). "
             "Large k: too much mass given to impossible sequences.",
    )
    n_sugg = st.slider("Number of suggestions", 1, 8, 3)
    full = st.checkbox(
        "Train on the full corpus",
        value=False,
        help="By default: only the 80% training documents "
             "(the remaining 20% are used for testing).",
    )
    st.divider()
    st.caption(
        "Synthetic corpus (1,200 sentences, 61 words). "
        "Probabilities are model estimates, not real typing statistics."
    )

if k == 0 and choice.startswith("Interp"):
    st.sidebar.info("The interpolated model uses k = 1e-9 instead of 0.")

docs, train, test = get_data()
models = get_models(k, full)
words = tokenize(st.session_state.text)
unknown = [w for w in words if w not in models["uni"].vocab]

# ------------------------------------------------------------------ hero
st.markdown(
    """
    <div class="hero">
      <div class="hero-inner">
        <h1><span class="kbd-emoji">⌨️</span> Smart Keyboard V1</h1>
        <p>Next-word suggestions powered by n-gram models — add-k smoothing, back-off &amp; interpolation.</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_kb, tab_cmp, tab_hood, tab_eval, tab_data = st.tabs(
    ["⌨️ Keyboard", "🔀 Compare models", "🔍 Under the hood", "📊 Evaluation", "🗂️ Data"]
)

# ================================================================== 1. KEYBOARD
with tab_kb:
    st.text_input(
        "Start typing a sentence:",
        key="text",
        placeholder="the happy girls …",
    )
    if unknown:
        st.warning(f"Out-of-vocabulary words (treated as `<unk>`): {', '.join(unknown)}")

    sugg, order = suggest(models, choice, words, n_sugg)
    if not words:
        st.info("No word entered yet — here is how sentences typically start.")
    st.subheader(f"Suggestions · active model: {order}")

    if sugg and sugg[0][1] == 0:
        st.error(
            "All probabilities are zero: unseen context and k = 0. "
            "Increase k or switch to back-off."
        )

    cols = st.columns(len(sugg)) if sugg else []
    for i, (col, (w, p)) in enumerate(zip(cols, sugg)):
        with col:
            st.button(
                f"{label(w)}",
                key=f"sg{i}",
                on_click=add_word,
                args=(w,),
                use_container_width=True,
                type="primary" if i == 0 else "secondary",
            )
            st.caption(f"probability: **{p:.1%}**")

    b1, b2, _ = st.columns([1, 1, 6])
    b1.button("⌫ Delete last", on_click=backspace, use_container_width=True)
    b2.button("🗑️ Clear all", on_click=clear, use_container_width=True)

    st.markdown("**Full distribution (top 10)**")
    top10, _ = suggest(models, choice, words, 10)
    if top10:
        df = pd.DataFrame(
            {"word": [label(w) for w, _ in top10],
             "probability": [p for _, p in top10]}
        ).set_index("word")
        st.bar_chart(df, horizontal=True, height=300)
    st.caption("Click a suggestion to append it to the text, just like a real phone keyboard.")

# ================================================================== 2. COMPARE
with tab_cmp:
    st.markdown(f"Context: **{' '.join(words) or '<start of sentence>'}**")
    rows = []
    for name in MODEL_CHOICES:
        s, o = suggest(models, name, words, n_sugg)
        rows.append({
            "Model": name,
            "Order used": o,
            **{f"#{i + 1}": f"{label(w)} ({p:.1%})" for i, (w, p) in enumerate(s)},
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    seen = {
        n: models[key].seen_context(words)
        for n, key in (("trigram", "tri"), ("bigram", "bi"))
    }
    st.write(
        "Context seen during training: "
        f"trigram **{'yes' if seen['trigram'] else 'no'}**, "
        f"bigram **{'yes' if seen['bigram'] else 'no'}**."
    )
    if not seen["trigram"]:
        st.info(
            "The trigram has never seen these last two words — this is exactly the case "
            "where back-off and interpolation help."
        )

# ================================================================== 3. UNDER THE HOOD
with tab_hood:
    st.markdown(
        "How the probability is computed:  "
        r"$P(w\mid c)=\dfrac{\text{count}(c,w)+k}{\text{count}(c)+k\cdot V}$"
    )
    n_order = st.radio(
        "Model order",
        [3, 2, 1],
        horizontal=True,
        format_func=lambda n: {3: "Trigram", 2: "Bigram", 1: "Unigram"}[n],
    )
    m = {3: models["tri"], 2: models["bi"], 1: models["uni"]}[n_order]
    ctx = m.context(words)
    counts = m.ctx_counts.get(ctx)
    total = m.ctx_totals.get(ctx, 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Context used", " ".join(ctx) if ctx else "(none)")
    c2.metric("count(context)", total)
    c3.metric("Vocabulary size V", m.V)
    c4.metric("k", k)
    if counts is None:
        st.warning(
            "This context never appears in the training corpus: all counts are 0, "
            "so the probability comes only from smoothing "
            "(1/V with k > 0, or 0 with k = 0)."
        )

    cand = sorted(m.vocab - {UNK}, key=lambda w: (-(counts[w] if counts else 0), w))[:10]
    st.dataframe(
        pd.DataFrame([
            {
                "next word": label(w),
                "count(context, word)": (counts[w] if counts else 0),
                "(count + k)": (counts[w] if counts else 0) + k,
                "(total + k·V)": round(total + k * m.V, 3),
                "P(word | context)": round(m.prob(ctx, w), 4),
            }
            for w in cand
        ]),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(f"`{BOS}` = start of sentence, `{EOS}` = end of sentence, `{UNK}` = unknown word.")

# ================================================================== 4. EVALUATION
with tab_eval:
    st.subheader("Lab results (results/metrics.json)")
    mpath = os.path.join(RESULTS, "metrics.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            met = json.load(f)
        df = pd.DataFrame(met["rows"])
        df["perplexity"] = df["perplexity"].astype(float)
        df = df.rename(
            columns={
                "model": "Model",
                "perplexity": "Perplexity",
                "zero_prob_share": "Tokens with 0 prob.",
                "top3": "Top-3 accuracy",
            }
        )
        st.dataframe(
            df.style.format(
                {"Perplexity": "{:.2f}", "Tokens with 0 prob.": "{:.1%}", "Top-3 accuracy": "{:.1%}"}
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption(
            f"Hard back-off (k=0.1): top-3 accuracy = {met['backoff_top3']:.1%}  ·  "
            f"{met['unseen_trigram_contexts']} test positions with an unseen trigram context."
        )

        st.markdown("**Perplexity vs. k**")
        sweep = pd.DataFrame({
            "k": [str(x) for x in met["k_sweep"]["k"]],
            "bigram": met["k_sweep"]["2"],
            "trigram": met["k_sweep"]["3"],
        }).set_index("k")
        st.line_chart(sweep)
        st.caption(
            "Too little or too much smoothing both hurt perplexity: "
            "the minimum is around k ≈ 0.01 – 0.1."
        )
    else:
        st.info("results/metrics.json not found: run `python -m src.evaluate`.")

    st.divider()
    st.subheader("Evaluate the live model & k selected in the sidebar")
    if full:
        st.warning(
            "“Train on the full corpus” is enabled: the test set is no longer independent "
            "(data leakage). Uncheck it for an honest evaluation."
        )
    if st.button("▶️ Run evaluation on test documents"):
        obj = get_model_object(models, choice)
        with st.spinner("Computing…"):
            if obj is not None:
                ppl, zero = obj.perplexity(test)
                acc, misses = top_k_accuracy(obj, test, 3)
            else:
                ppl, zero = float("nan"), 0.0
                hits = total = 0
                misses = []
                for s in test:
                    for i, w in enumerate(s + [EOS]):
                        g = [
                            x for x, _ in backoff_top_k(
                                models["tri"], models["bi"], models["uni"], s[:i], 3
                            )[0]
                        ]
                        total += 1
                        if (w if w in models["uni"].vocab else UNK) in g:
                            hits += 1
                        elif len(misses) < 200:
                            misses.append((" ".join(s[:i]) or "<start>", w, g))
                acc = hits / total
        e1, e2, e3 = st.columns(3)
        e1.metric(
            "Perplexity (test)",
            "n/a (back-off)" if ppl != ppl else ("∞" if ppl == float("inf") else f"{ppl:.2f}"),
        )
        e2.metric("Tokens with 0 probability", f"{zero:.1%}")
        e3.metric("Top-3 accuracy", f"{acc:.1%}")
        if misses:
            st.markdown("**Sample errors (context → true word | suggestions)**")
            st.dataframe(
                pd.DataFrame(misses[:15], columns=["context", "true word", "suggestions"]),
                hide_index=True,
                use_container_width=True,
            )

    png = os.path.join(RESULTS, "ngram_results.png")
    if os.path.exists(png):
        with st.expander("Lab figure (ngram_results.png)"):
            st.image(png)
    fpath = os.path.join(RESULTS, "failures.txt")
    if os.path.exists(fpath):
        with st.expander("Failure analysis (failures.txt)"):
            st.code(open(fpath, encoding="utf-8").read())

# ================================================================== 5. DATA
with tab_data:
    n_tr, n_te = sum(len(s) for s in train), sum(len(s) for s in test)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Documents", len(docs))
    d2.metric("Sentences (train / test)", f"{len(train)} / {len(test)}")
    d3.metric("Tokens (train / test)", f"{n_tr} / {n_te}")
    d4.metric("Vocabulary", models["uni"].V - 2)
    st.info(
        "**Document-level split**: the last 20% of documents form the test set. "
        "Splitting by sentence would leak vocabulary from test documents "
        "and artificially inflate the scores."
    )
    st.dataframe(pd.read_csv(CORPUS).head(100), hide_index=True, use_container_width=True)