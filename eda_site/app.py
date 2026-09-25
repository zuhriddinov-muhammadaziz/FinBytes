"""Public-facing, aggregate-only EDA website for the WIUT FinTech task."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SUMMARY = Path(__file__).parent / "assets" / "eda_summary.json"
SECTIONS = [
    "1. Problem overview", "2. Dataset structure", "3. Data quality and temporal relationship",
    "4. Target distribution", "5. Transaction activity over time", "6. Incoming vs outgoing behavior",
    "7. Transaction-type behavior", "8. Amount distributions", "9. Recent activity / temporal behavior",
    "10. Escalated vs dismissed comparisons", "11. Key EDA observations",
    "12. Feature engineering decisions motivated by EDA", "13. Modeling approach and validation methodology",
    "14. Final conclusion",
]
COLORS = ["#2166ac", "#d6604d", "#4d9221", "#762a83", "#e08214", "#4393c3"]

st.set_page_config(page_title="WIUT FinTech | Exploratory Data Analysis", page_icon="📊", layout="wide")
st.markdown("""
<style>
.stApp {background:#f4f6f8;color:#17212b}
[data-testid="stHeader"] {background:rgba(244,246,248,.92)}
.block-container {max-width:1280px;padding:1.7rem 2.5rem 3.25rem}
section[data-testid="stSidebar"] {background:#111b27;border-right:1px solid #253342}
section[data-testid="stSidebar"] * {color:#e5edf4}
section[data-testid="stSidebar"] [data-baseweb="select"] div {background:#1a2735;border-color:#344454}
section[data-testid="stSidebar"] hr {border-color:#344454}
.hero {position:relative;overflow:hidden;background:linear-gradient(118deg,#142232 0%,#20384a 62%,#345448 100%);color:white;border-radius:24px;padding:2.4rem 2.7rem;margin-bottom:1.6rem;box-shadow:0 20px 45px rgba(21,39,54,.14)}
.hero:after {content:"";position:absolute;width:280px;height:280px;border:1px solid rgba(211,244,118,.23);border-radius:50%;right:-80px;top:-145px;box-shadow:0 0 0 28px rgba(211,244,118,.045),0 0 0 58px rgba(211,244,118,.035)}
.eyebrow {font-size:.72rem;letter-spacing:.19em;color:#d3f476;font-weight:700;margin-bottom:1.15rem}
.hero h1 {font-size:clamp(2.3rem,5vw,4.1rem);line-height:1.03;letter-spacing:-.045em;margin:0 0 .8rem;color:white;font-weight:650}
.hero p {font-size:1.03rem;margin:.2rem 0;color:#d3dce3;max-width:690px}
.hero-tag {display:inline-block;margin-top:1.2rem;border:1px solid rgba(211,244,118,.35);border-radius:99px;padding:.35rem .72rem;font-size:.75rem;color:#e7f4ce;background:rgba(211,244,118,.08)}
.callout {background:#fff;border:1px solid #e3e8ec;border-left:4px solid #8fb83f;padding:1rem 1.1rem;border-radius:12px;margin:.65rem 0 1rem;color:#334252}
.small-note {color:#71808d;font-size:.85rem}
h1,h2,h3 {letter-spacing:-.025em}
h2 {color:#20384a;margin-top:.5rem}
[data-testid="stMetric"] {background:#fff;border:1px solid #e5e9ed;border-radius:16px;padding:1rem 1.15rem;box-shadow:0 6px 18px rgba(24,40,52,.035)}
[data-testid="stMetricLabel"] {color:#71808d}
[data-testid="stMetricValue"] {color:#172b3a}
[data-testid="stPlotlyChart"] {background:#fff;border:1px solid #e6eaee;border-radius:16px;padding:.35rem;box-shadow:0 6px 20px rgba(24,40,52,.035)}
[data-testid="stDataFrame"] {border:1px solid #e5e9ed;border-radius:14px;overflow:hidden}
.stAlert {border-radius:14px}
@media(max-width:700px){.block-container{padding:1rem 1rem 2rem}.hero{padding:1.7rem;border-radius:18px}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_summary():
    if not SUMMARY.exists():
        return None
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def chart(data, x, y, *, kind="bar", title=None, xlabel=None, ylabel=None, color=None):
    df = pd.DataFrame(data)
    if df.empty:
        st.info("This view is not available because the source data did not include the required field.")
        return
    if kind == "line":
        fig = px.line(df, x=x, y=y, markers=True, title=title, color=color)
    elif kind == "hist":
        fig = px.bar(df, x=x, y=y, title=title, color_discrete_sequence=COLORS)
    else:
        fig = px.bar(df, x=x, y=y, title=title, color=color, color_discrete_sequence=COLORS)
    fig.update_layout(template="plotly_white", height=390, margin=dict(l=18, r=18, t=58, b=18),
                      xaxis_title=xlabel, yaxis_title=ylabel, legend_title_text="",
                      paper_bgcolor="rgba(255,255,255,0)", plot_bgcolor="rgba(255,255,255,0)",
                      font=dict(family="Inter, sans-serif", color="#263b4b"),
                      title_font=dict(size=17, color="#20384a"))
    fig.update_xaxes(showgrid=False, linecolor="#dbe2e7")
    fig.update_yaxes(gridcolor="#edf0f2", zerolinecolor="#dbe2e7")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def show(s):
    d, q, t, a, act, recent = s["dataset"], s["quality"], s["target"], s["amounts"], s["activity"], s["recent"]
    selected = st.session_state.section
    title = selected.split(". ", 1)[1]
    st.markdown(f"## {selected}")
    if selected.startswith("1."):
        st.write("This project studies whether a financial signal should be escalated for review or dismissed. The goal of this site is to explain the data, its limitations, and the analysis choices. It does not score signals or serve a prediction model.")
        st.info("Competition data are synthetic and transformed/localized for the hackathon. They do not contain real customer or transaction records from the Central Bank of Uzbekistan or a commercial bank.")
        st.markdown('<div class="callout"><b>Analysis boundary:</b> descriptive comparisons help characterize this dataset; they do not establish why a signal was escalated or prove that a pattern will generalize.</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Labeled signals", f"{d['train_signals']:,}")
        c2.metric("Unlabeled signals", f"{d['test_signals']:,}")
        c3.metric("Train signal dates", f"{d['train_date_min']} – {d['train_date_max']}")
        st.caption("Transaction behavior is summarized only from transactions dated on or before each linked signal date.")
    elif selected.startswith("2."):
        st.write("The training set pairs signal identifiers and dates with a binary escalation label. The test set has the same signal fields but no label. Transaction records link to signals through the shared identifier.")
        st.markdown("**Training signal fields:** " + ", ".join(f"`{x}`" for x in d["train_signal_columns"]))
        st.markdown("**Test signal fields:** " + ", ".join(f"`{x}`" for x in d["test_signal_columns"]))
        st.markdown("**Transaction fields:** " + ", ".join(f"`{x}`" for x in d["transaction_columns"]))
        c1, c2 = st.columns(2)
        c1.metric("Training transactions", f"{d['train_transaction_rows']:,}")
        c2.metric("Test transactions", f"{d['test_transaction_rows']:,}")
        st.caption("The website contains aggregate summaries only; it does not expose signal-level records or identifiers.")
    elif selected.startswith("3."):
        st.write("The signal identifier should be unique within each signal table. Transaction records are linked to training signals to assess coverage and compare transaction timestamps with signal dates.")
        a1, a2, a3 = st.columns(3)
        a1.metric("Duplicate training IDs", f"{q['train_signal_duplicate_ids']:,}")
        a2.metric("Duplicate test IDs", f"{q['test_signal_duplicate_ids']:,}")
        a3.metric("Transactions with unmatched IDs", f"{q['transaction_signal_ids_unmatched']:,}")
        b1, b2, b3 = st.columns(3)
        b1.metric("Transactions missing date", f"{q['transaction_missing_date']:,}")
        b2.metric("Dated and linked transactions", f"{q['linked_dated_transactions']:,}")
        b3.metric("After signal date", f"{q['future_transactions']:,} ({q['future_transaction_percent']}%)")
        st.markdown('<div class="callout"><b>Temporal rule:</b> transaction behavior sections use dated transactions at or before the signal date. Future-dated records are excluded from those summaries to avoid using information unavailable at signal time.</div>', unsafe_allow_html=True)
        quality = pd.DataFrame([{"Field": k, "Missing": v["nulls"], "Missing (%)": v["null_percent"], "Distinct values": v["distinct"]} for k, v in q["transaction_column_quality"].items()])
        st.dataframe(quality, hide_index=True, width="stretch")
    elif selected.startswith("4."):
        st.write("The label is binary: 0 denotes dismissed and 1 denotes escalated. Monthly rates are descriptive and can be noisy when a month contains few signals.")
        chart(t["counts"], "label", "count", title="Training label counts", ylabel="Signals")
        chart(t["monthly"], "month", "escalation_rate", kind="line", title="Escalation share by signal month", ylabel="Escalation share")
    elif selected.startswith("5."):
        st.write("Monthly transaction totals describe the timing of the historical records retained under the signal-date rule. Signals may have very different numbers of linked transactions.")
        chart(act["transactions_by_month"], "month", "count", title="Eligible training transactions by month", ylabel="Transactions")
        stats = act["transactions_per_signal"]
        st.write(f"Across {d['train_signals']:,} training signals: mean **{stats['mean']:.2f}**, median **{stats['median']:.0f}**, 90th percentile **{stats['p90']:.0f}** transactions; **{stats['zero_signal_percent']:.1f}%** have no eligible transaction history.")
        chart(act["transaction_count_distribution"], "bin", "count", title="Transactions per signal (upper tail clipped for readability)", ylabel="Signals")
    elif selected.startswith("6."):
        st.write("Direction is shown as recorded in the source transaction data. The values are transaction counts after temporal filtering; they do not measure net cash flow.")
        chart(act["directions"], "category", "count", title="Transaction records by direction", ylabel="Transactions")
    elif selected.startswith("7."):
        st.write("Transaction-type mix gives context for feature design. Only the most frequent categories are shown; category labels follow the dataset.")
        chart(act["types"], "category", "count", title="Most frequent transaction types", ylabel="Transactions")
    elif selected.startswith("8."):
        st.write("The dataset contains an amount index rather than a currency-denominated amount. The histogram clips the display range at the 99th percentile for readability; summary figures below retain the observed values.")
        if a["column"]:
            c1, c2, c3 = st.columns(3)
            suffix = " index units" if a.get("is_index") else ""
            c1.metric("Mean amount index", f"{a['mean']:,.2f}{suffix}")
            c2.metric("Median amount index", f"{a['median']:,.2f}{suffix}")
            c3.metric("95th percentile", f"{a['p95']:,.2f}{suffix}")
            chart(a["distribution"], "bin", "count", title="Transaction amount index distribution (upper tail clipped)", ylabel="Transactions")
            col1, col2 = st.columns(2)
            with col1: chart(a["amount_per_signal_distribution"], "bin", "count", title="Total eligible amount per signal", ylabel="Signals")
            with col2: chart(a["mean_per_signal_distribution"], "bin", "count", title="Mean eligible transaction amount per signal", ylabel="Signals")
        else:
            st.info("An amount field was not identified in the transaction schema.")
    elif selected.startswith("9."):
        st.write("The view below shows the latest 90 calendar dates with training signals, highlighting uneven sample density. Feature engineering additionally considers recency and multiple fixed lookback windows.")
        chart(recent["daily"], "date", "signals", kind="line", title="Daily training signal arrivals (latest 90 active dates)", ylabel="Signals")
        st.caption(f"Training signal dates span {recent['date_range_days']:,} days. The latest month represented contains {recent['latest_month_signals']:,} signals.")
    elif selected.startswith("10."):
        st.write("These label-group summaries describe associations within the labeled training data. Differences are unadjusted and should not be read as causal or as proof of predictive value.")
        rows = []
        for key, label in (("dismissed", "Dismissed (0)"), ("escalated", "Escalated (1)")):
            v = s["target_comparison"][key]
            rows.append({"Group": label, "Signals": v["signals"], "Mean transactions / signal": v["transactions_per_signal_mean"], "Median transactions / signal": v["transactions_per_signal_median"], "Mean total amount index / signal": v["amount_per_signal_mean"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        chart([{"Group": "Dismissed", "Mean transactions": s["target_comparison"]["dismissed"]["transactions_per_signal_mean"]}, {"Group": "Escalated", "Mean transactions": s["target_comparison"]["escalated"]["transactions_per_signal_mean"]}], "Group", "Mean transactions", title="Mean eligible transaction count by label")
    elif selected.startswith("11."):
        st.write("The main observations below are calculated directly from the project data and aggregate artifact.")
        stats = act["transactions_per_signal"]
        st.markdown(f"- The training set has **{d['train_signals']:,}** labeled signals and the test set has **{d['test_signals']:,}** unlabeled signals.")
        st.markdown(f"- The median eligible transaction history is **{stats['median']:.0f}** transactions per signal; **{stats['zero_signal_percent']:.1f}%** have none.")
        st.markdown(f"- **{q['future_transaction_percent']:.2f}%** of linked, dated training transactions occur after their signal date and are excluded from historical behavior summaries.")
        st.markdown(f"- The recorded signal date range overlaps broadly across train ({d['train_date_min']} to {d['train_date_max']}) and test ({d['test_date_min']} to {d['test_date_max']}); random splitting alone would not represent future deployment well.")
        positive = next((x for x in t["counts"] if x["class"] == 1), {"count": 0})
        st.markdown(f"- Escalated cases are **{positive['count']:,} / {d['train_signals']:,}** training signals ({positive['count'] / max(d['train_signals'], 1):.1%}); class imbalance makes ROC-AUC a suitable ranking metric.")
        compare = s["target_comparison"]
        st.markdown(f"- Mean eligible transaction counts differ between dismissed (**{compare['dismissed']['transactions_per_signal_mean']:.1f}**) and escalated (**{compare['escalated']['transactions_per_signal_mean']:.1f}**) signals; these are descriptive associations, not causal effects.")
        model = s["method"].get("model_evaluation", {})
        if model.get("oof_roc_auc") is not None:
            st.markdown(f"- The final reproducible model, **{model['model']}**, reached OOF ROC-AUC **{model['oof_roc_auc']:.4f}** on {model['oof_rows']:,} forward-validation rows.")
    elif selected.startswith("12."):
        st.write("EDA motivates compact signal-level summaries. The final model uses features built without labels and only from transactions available at signal time.")
        st.markdown("- **History and mix:** transaction count, active days, incoming/outgoing counts and shares, and counts for the four transaction types.")
        st.markdown("- **Amount profile:** sum, mean, standard deviation, median, minimum, maximum, and 75th/90th percentiles of the standardized amount index.")
        st.markdown("- **Recency:** transaction counts and amount sums for 7-, 30-, and 90-day windows, days since the last eligible transaction, and history span.")
        st.markdown("- **Signal calendar:** month and weekday are retained as context, with performance checked on forward-only date blocks.")
        st.markdown("- **Feature selection:** the controlled sprint tested longer windows, velocity ratios, gaps, sequence summaries, subsets, model settings and blends. Added feature groups did not beat the simpler 28-feature shallow XGBoost candidate.")
        st.markdown("- **Safety:** future transactions are excluded; missing-history counts and zero denominators are handled explicitly; no target encoding of unique signal IDs is used.")
        st.caption("The label-group comparisons in this report inform questions, but are not inputs to feature creation.")
    elif selected.startswith("13."):
        model = s["method"].get("model_evaluation", {})
        st.write("Model candidates were compared on the same five expanding chronological folds. Each fold trains on earlier signals and validates on a later date block; an initial warm-up block is unscored.")
        st.markdown("- **Selected model:** shallow XGBoost; 500 trees, depth 2, learning rate 0.025, minimum child weight 15, subsample/column sample 0.8, L2 regularization 5, seed 42.")
        st.markdown(f"- **Validation:** {model.get('fold_count', 0)} forward-only folds and {model.get('oof_rows', 0):,} OOF rows; **OOF ROC-AUC {model.get('oof_roc_auc', 0):.4f}** across {model.get('feature_count', 0)} features.")
        if model.get("fold_auc"):
            st.markdown("- **Fold ROC-AUC:** " + ", ".join(f"fold {row['fold']}: {row['roc_auc']:.4f}" for row in model["fold_auc"]) + ".")
        if model.get("baseline_models"):
            comparison = ", ".join(f"{row['model']}: {row['oof_roc_auc']:.4f}" for row in model["baseline_models"])
            st.markdown("- **Baseline comparison:** " + comparison + ".")
        st.markdown("- Model and feature selection used training OOF data only; the hidden test labels and leaderboard were not used.")
        st.caption("OOF performance is an internal validation estimate, not a hidden-test or leaderboard score.")
    else:
        model = s["method"].get("model_evaluation", {})
        score_text = f" OOF ROC-AUC was {model['oof_roc_auc']:.4f} under forward-only validation." if model.get("oof_roc_auc") is not None else ""
        st.write("The strongest validated approach combines strict as-of-signal transaction history with a compact set of interpretable count, amount, mix, recency and calendar features." + score_text)
        st.markdown("The controlled experiments favored a shallow XGBoost model on 28 features over more complex feature expansions and blends. This is a validation result, not a guarantee of hidden-test performance.")
        st.markdown('<div class="callout"><b>Conclusion:</b> this site presents aggregate EDA and modeling decisions. The competition dataset is synthetic; the website does not expose records or serve predictions.</div>', unsafe_allow_html=True)


summary = load_summary()
st.markdown('<div class="hero"><div class="eyebrow">FinBytes · RESEARCH BRIEF</div><h1>Signals, seen in time.</h1><p>An evidence-led tour of alert outcomes, transaction histories, and the modeling choices shaped by the data.</p><span class="hero-tag">Synthetic competition data · Aggregate analysis</span></div>', unsafe_allow_html=True)
if summary is None:
    st.error("The precomputed aggregate summary is missing. From the project root, run: python eda_site/build_summary.py")
    st.stop()
st.sidebar.markdown('<div style="font-size:.72rem;letter-spacing:.16em;text-transform:uppercase;color:#d3f476;font-weight:700;margin:.3rem 0 1rem">FIELD NOTES / 2025</div>', unsafe_allow_html=True)
st.sidebar.title("Explore the analysis")
st.sidebar.caption("14 chapters · descriptive EDA")
st.sidebar.selectbox("Section", SECTIONS, key="section")
st.sidebar.markdown("---")
st.sidebar.caption("All transaction behavior views use records available at signal time. No individual signal records are shown.")
show(summary)
st.markdown("---")
st.markdown('<p class="small-note">Exploratory analysis only · Aggregate data · No predictions are served</p>', unsafe_allow_html=True)
