"""
Interactive demo + results overview for the dispute pipeline.

Two tabs:
  - "Overview" -- a pitch-ready summary built ONLY from files this repo's
    own scripts already produce (results/ml/*, results/robustness_check.csv,
    results/audit_trail/*). Nothing on this tab is a mock or a hardcoded
    number; every chart reads a CSV/JSON on disk and will change the next
    time you rerun evaluate.py / train_and_compare.py / pipeline.py. If a
    file is missing, the section says which script to run instead of
    silently faking data.
  - "Live decision demo" -- the original single-dispute drill-down, reusing
    DisputePipeline from pipeline.py directly so this can never drift out
    of sync with the actual decision system.

Run from the project root (same directory as model_artifacts/):
    streamlit run streamlit_app.py
"""
import glob
import json
import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pipeline import DisputePipeline, REASON_RELEVANT_FIELDS, KNOWN_REASON_CODES

st.set_page_config(page_title="CaseWise -- Chargeback Evidence Responder", layout="wide")

# ---------------------------------------------------------------------------
# Palette -- matches .streamlit/config.toml. Defined once here so every
# Plotly chart below uses the same colors as the native Streamlit widgets.
# ---------------------------------------------------------------------------
BG = "#0B0B0D"
CARD_BG = "#16171A"
CARD_BORDER = "#26272B"
TEXT_PRIMARY = "#F2F2F0"
TEXT_MUTED = "#9A9A9E"
ORANGE = "#E8944A"
ORANGE_SOFT = "rgba(232,148,74,0.18)"
GREEN = "#3ECF8E"
GREEN_SOFT = "rgba(62,207,142,0.15)"
RED = "#E5484D"
RED_SOFT = "rgba(229,72,77,0.15)"
GRAY_BAR = "#4A4B50"

DECISION_COLOR = {
    'submit': '#1a7f37',
    'submit_with_caveat': '#9a6700',
    'flag_ungrounded_narrative': '#9a6700',
    'flag_insufficient': '#57606a',
    'manual_review': '#cf222e',
    'missed_deadline': '#cf222e',
}
DECISION_LABEL = {
    'submit': 'SUBMIT',
    'submit_with_caveat': 'SUBMIT (with caveat)',
    'flag_ungrounded_narrative': 'FLAGGED -- ungrounded narrative',
    'flag_insufficient': 'FLAG AS INSUFFICIENT',
    'manual_review': 'ROUTE TO MANUAL REVIEW',
    'missed_deadline': 'MISSED DEADLINE -- action rejected',
}

# ---------------------------------------------------------------------------
# CSS -- only styles elements this file draws itself (cards, pills, hero
# banner). Native Streamlit widgets are themed via .streamlit/config.toml,
# not fought with CSS overrides here.
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.hero {{
    background: radial-gradient(circle at 15% 20%, rgba(232,148,74,0.16), transparent 45%),
                linear-gradient(135deg, #1a1a1d 0%, #0d0d0f 70%);
    border: 1px solid {CARD_BORDER};
    border-radius: 20px;
    padding: 34px 38px;
    margin-bottom: 22px;
}}
.hero h1 {{ color: {TEXT_PRIMARY}; font-size: 30px; font-weight: 800; margin: 0 0 8px 0; }}
.hero p {{ color: {TEXT_MUTED}; font-size: 15px; margin: 0; max-width: 700px; }}

.kpi-card {{
    background: {CARD_BG}; border: 1px solid {CARD_BORDER}; border-radius: 16px;
    padding: 18px 20px; height: 108px; display: flex; flex-direction: column; justify-content: space-between;
}}
.kpi-label {{ color: {TEXT_MUTED}; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }}
.kpi-value {{ color: {TEXT_PRIMARY}; font-size: 26px; font-weight: 800; }}
.badge {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 700; }}
.badge-green {{ background: {GREEN_SOFT}; color: {GREEN}; }}
.badge-orange {{ background: {ORANGE_SOFT}; color: {ORANGE}; }}
.badge-red {{ background: {RED_SOFT}; color: {RED}; }}

.panel {{
    background: {CARD_BG}; border: 1px solid {CARD_BORDER}; border-radius: 16px;
    padding: 20px 22px; margin-bottom: 18px;
}}
.panel-title {{ color: {TEXT_PRIMARY}; font-size: 15px; font-weight: 700; margin-bottom: 2px; }}
.panel-sub {{ color: {TEXT_MUTED}; font-size: 12.5px; margin-bottom: 14px; }}

.list-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 2px; border-bottom: 1px solid {CARD_BORDER};
}}
.list-row:last-child {{ border-bottom: none; }}
.list-id {{ color: {TEXT_PRIMARY}; font-weight: 600; font-size: 13.5px; }}
.list-reason {{ color: {TEXT_MUTED}; font-size: 12px; }}
.pill {{ padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; color: white; }}
</style>
""", unsafe_allow_html=True)


def plotly_dark_layout(fig, height=320):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, family="Inter", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=TEXT_PRIMARY)),
        hoverlabel=dict(bgcolor=CARD_BG, font_color=TEXT_PRIMARY, bordercolor=CARD_BORDER),
    )
    fig.update_xaxes(gridcolor=CARD_BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=CARD_BORDER, zeroline=False)
    return fig


def kpi_card(label, value, badge_text=None, badge_class="badge-green"):
    badge_html = f'<span class="badge {badge_class}">{badge_text}</span>' if badge_text else ""
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div style="display:flex; align-items:baseline; justify-content:space-between;">
            <div class="kpi-value">{value}</div>{badge_html}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loaders -- every one of these reads a real file this repo's own
# scripts produce. None of them fabricate a fallback number; missing files
# are surfaced as "run this script first", not papered over.
# ---------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    return DisputePipeline()


@st.cache_data
def load_sample_disputes():
    return pd.read_csv('disputes_10k.csv')


@st.cache_data
def load_json_if_exists(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_csv_if_exists(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_latest_audit_trail():
    candidates = sorted(glob.glob('results/audit_trail/*/audit_trail.json'))
    if not candidates:
        return None, None
    latest = candidates[-1]
    with open(latest) as f:
        return json.load(f), latest


def missing_file_notice(path, script):
    st.info(f"`{path}` not found yet -- run `python {script}` first to populate this section.")


pipeline = load_pipeline()
df = load_sample_disputes()

summary = load_json_if_exists('results/ml/summary.json')
fp_sensitivity = load_csv_if_exists('results/ml/fp_cost_sensitivity.csv')
augmentation_comparison = load_csv_if_exists('results/ml/final_comparison.csv')
heuristic_vs_ml = load_csv_if_exists('results/ml/heuristic_vs_ml.csv')
robustness = load_csv_if_exists('results/robustness_check.csv')
audit_entries, audit_path = load_latest_audit_trail()

st.markdown(f"""
<div class="hero">
    <h1>CaseWise -- turn chargeback risk into measurable savings</h1>
    <p>Razorpay AI Buildathon, Track 02: AI Risk Manager. Every number and chart
    on this page is read live from this repo's own <code>results/</code> files --
    rerun <code>evaluate.py</code>, <code>train_and_compare.py</code>, or
    <code>pipeline.py</code> and this page reflects the new run, not a fixed demo.</p>
</div>
""", unsafe_allow_html=True)

tab_overview, tab_demo = st.tabs(["Overview", "Live decision demo"])

# ===========================================================================
# TAB 1 -- OVERVIEW
# ===========================================================================
with tab_overview:
    if summary is None:
        missing_file_notice('results/ml/summary.json', 'evaluate.py')
    else:
        per_reason_all = pd.DataFrame(summary['per_reason'])
        overall = per_reason_all[per_reason_all['reason'] == 'OVERALL'].iloc[0]
        per_reason = per_reason_all[per_reason_all['reason'] != 'OVERALL'].copy()

        naive_best = min(summary['cost_flag_everything_insufficient_inr'],
                          summary['cost_submit_everything_inr'])
        savings = naive_best - summary['total_cost_inr']

        # ---- KPI row -------------------------------------------------------
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("Gate savings vs. naive policy", f"Rs.{savings:,.0f}",
                     "vs. best fixed rule", "badge-green")
        with c2:
            badge = None
            delta_pp = 0.0
            if heuristic_vs_ml is not None:
                heur_p = heuristic_vs_ml.loc[heuristic_vs_ml['method'].str.contains('heuristic'), 'precision'].iloc[0]
                delta_pp = (overall['precision'] - heur_p) * 100
                badge = f"{delta_pp:+.1f}pp vs heuristic"
            kpi_card("Overall precision", f"{overall['precision']:.1%}", badge,
                     "badge-green" if delta_pp >= 0 else "badge-red")
        with c3:
            kpi_card("Overall recall", f"{overall['recall']:.1%}", f"F1 {overall['f1']:.3f}", "badge-orange")
        with c4:
            kpi_card("Held-out disputes evaluated", f"{int(overall['n']):,}",
                     f"{int(overall['n_positive'])} actually won", "badge-orange")

        st.write("")

        # ---- Chart row 1: FP-cost sensitivity + per-reason bars ------------
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">FP-cost sensitivity</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="panel-sub">How precision/F1 shift if a false submission is '
                        f'assumed cheaper or more expensive than the chosen default '
                        f'(Rs.{summary["fp_cost_default_inr"]:.0f})</div>', unsafe_allow_html=True)
            if fp_sensitivity is None:
                missing_file_notice('results/ml/fp_cost_sensitivity.csv', 'evaluate.py')
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fp_sensitivity['assumed_fp_cost_inr'], y=fp_sensitivity['f1'],
                    mode='lines+markers', name='F1', line=dict(color=ORANGE, width=3),
                    fill='tozeroy', fillcolor=ORANGE_SOFT,
                ))
                fig.add_trace(go.Scatter(
                    x=fp_sensitivity['assumed_fp_cost_inr'], y=fp_sensitivity['precision'],
                    mode='lines+markers', name='Precision', line=dict(color=GREEN, width=2, dash='dot'),
                ))
                fig.add_vline(x=summary['fp_cost_default_inr'], line_dash="dash",
                              line_color=TEXT_MUTED, annotation_text="chosen default",
                              annotation_font_color=TEXT_MUTED)
                fig.update_xaxes(title_text="Assumed FP cost (Rs.)")
                fig.update_yaxes(title_text="Score", tickformat=".0%")
                st.plotly_chart(plotly_dark_layout(fig), width='stretch', config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Precision &amp; recall by reason code</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-sub">subscription_cancelled uses a 3x FP-cost multiplier '
                        '(Section 4.4); not_as_described is left unadjusted -- too few positive '
                        'cases to trust a specific value</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(x=per_reason['reason'], y=per_reason['precision'],
                                  name='Precision', marker_color=ORANGE))
            fig.add_trace(go.Bar(x=per_reason['reason'], y=per_reason['recall'],
                                  name='Recall', marker_color=GREEN))
            fig.update_layout(barmode='group')
            fig.update_yaxes(title_text="Score", tickformat=".0%")
            fig.update_xaxes(tickangle=-20)
            st.plotly_chart(plotly_dark_layout(fig), width='stretch', config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        # ---- Chart row 2: augmentation comparison + heuristic vs ML --------
        col_c, col_d = st.columns(2)

        with col_c:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">SynthEdge vs. SMOTE vs. baseline</div>', unsafe_allow_html=True)
            st.markdown('<div class="panel-sub">Overall recall by augmentation method (single-split '
                        'result; see the robustness expander below for the 5-seed version)</div>',
                        unsafe_allow_html=True)
            if augmentation_comparison is None:
                missing_file_notice('results/ml/final_comparison.csv', 'train_and_compare.py')
            else:
                labels = augmentation_comparison['label']
                colors = [ORANGE if 'synthedge' in l.lower() else GRAY_BAR for l in labels]
                fig = go.Figure(go.Bar(
                    x=labels, y=augmentation_comparison['recall'],
                    marker_color=colors, text=[f"{v:.1%}" for v in augmentation_comparison['recall']],
                    textposition='outside',
                ))
                fig.update_yaxes(title_text="Recall", tickformat=".0%")
                fig.update_xaxes(tickangle=-10)
                st.plotly_chart(plotly_dark_layout(fig), width='stretch', config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col_d:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">ML gate vs. a simple &gt;=50% evidence heuristic</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="panel-sub">Same field set, same cost function -- a fair comparison, '
                        'not a strawman (Section 4.5)</div>', unsafe_allow_html=True)
            if heuristic_vs_ml is None:
                missing_file_notice('results/ml/heuristic_vs_ml.csv', 'heuristic_baseline.py')
            else:
                metrics = ['precision', 'recall', 'f1']
                heur_row = heuristic_vs_ml[heuristic_vs_ml['method'].str.contains('heuristic')].iloc[0]
                ml_row = heuristic_vs_ml[heuristic_vs_ml['method'].str.contains('ML')].iloc[0]
                fig = go.Figure()
                fig.add_trace(go.Bar(x=metrics, y=[heur_row[m] for m in metrics],
                                      name='Heuristic', marker_color=GRAY_BAR))
                fig.add_trace(go.Bar(x=metrics, y=[ml_row[m] for m in metrics],
                                      name='ML gate', marker_color=ORANGE))
                fig.update_layout(barmode='group')
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(plotly_dark_layout(fig, height=260), width='stretch',
                                config={'displayModeBar': False})
                cost_delta = heur_row['total_cost_inr'] - ml_row['total_cost_inr']
                pct = cost_delta / heur_row['total_cost_inr'] * 100
                st.markdown(
                    f'<span class="badge badge-green">Rs.{cost_delta:,.0f} cheaper '
                    f'({pct:.0f}%) than the heuristic on this test set</span>',
                    unsafe_allow_html=True,
                )
            st.markdown('</div>', unsafe_allow_html=True)

        # ---- Optional: robustness across seeds ------------------------------
        if robustness is not None:
            with st.expander("Robustness across 5 random seeds (backs the SynthEdge-vs-SMOTE claim above)"):
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=robustness['seed'], y=robustness['baseline_overall'],
                                          mode='lines+markers', name='Baseline', line=dict(color=GRAY_BAR)))
                fig.add_trace(go.Scatter(x=robustness['seed'], y=robustness['smote_overall'],
                                          mode='lines+markers', name='SMOTE', line=dict(color=RED)))
                fig.add_trace(go.Scatter(x=robustness['seed'], y=robustness['synthedge_overall'],
                                          mode='lines+markers', name='SynthEdge', line=dict(color=ORANGE, width=3)))
                fig.update_xaxes(title_text="Seed", type='category')
                fig.update_yaxes(title_text="Overall recall", tickformat=".0%")
                st.plotly_chart(plotly_dark_layout(fig, height=280), width='stretch',
                                config={'displayModeBar': False})
                st.caption("SMOTE's overall recall looks competitive here only because it comes almost "
                          "entirely from majority-adjacent reason codes -- see README Section 4.1 for "
                          "why the per-reason breakdown (not shown here) is what actually matters.")

        # ---- Recent decisions -------------------------------------------------
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Recent decisions</div>', unsafe_allow_html=True)
        if audit_entries is None:
            st.markdown('<div class="panel-sub"></div>', unsafe_allow_html=True)
            missing_file_notice('results/audit_trail/*/audit_trail.json', 'pipeline.py')
        else:
            st.markdown(f'<div class="panel-sub">From the latest run: <code>{audit_path}</code></div>',
                        unsafe_allow_html=True)
            for entry in audit_entries:
                stages = entry.get('stages', {})
                reason = stages.get('reason_classification', {}).get('reason_code', 'unknown')
                decision = entry.get('final_decision', 'unknown')
                color = DECISION_COLOR.get(decision, GRAY_BAR)
                label = DECISION_LABEL.get(decision, decision)
                scoring = stages.get('scoring')
                wp = scoring.get('win_probability') if scoring else None
                wp_text = f"{wp:.1%} win prob." if wp is not None else "not scored"
                st.markdown(f"""
                <div class="list-row">
                    <div>
                        <span class="list-id">{entry.get('dispute_id', '-')}</span>
                        &nbsp;&nbsp;<span class="list-reason">{reason}</span>
                    </div>
                    <div style="display:flex; align-items:center; gap:14px;">
                        <span class="list-reason">{wp_text}</span>
                        <span class="pill" style="background:{color};">{label}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ===========================================================================
# TAB 2 -- LIVE DECISION DEMO (original single-dispute drill-down)
# ===========================================================================
with tab_demo:
    st.caption(
        "Every decision below is produced by the real pipeline (pipeline.py), not a mock."
    )

    st.sidebar.header("Dispute input")
    input_mode = st.sidebar.radio("Source", ["Pick a real dispute", "Build a custom dispute"])

    if input_mode == "Pick a real dispute":
        idx = st.sidebar.selectbox(
            "Sample dispute (from disputes_10k.csv)",
            options=df.sample(n=30, random_state=1).index,
            format_func=lambda i: f"{df.loc[i, 'dispute_id']} -- {df.loc[i, 'dispute_reason_code']} -- Rs.{df.loc[i, 'transaction_amount_inr']:.0f}",
        )
        dispute = df.loc[idx].to_dict()
        st.sidebar.caption("Ground truth (for your reference only -- not shown to the pipeline):")
        st.sidebar.write(f"Actually won: **{bool(dispute.get('won'))}**")

    else:
        reason_code = st.sidebar.selectbox("Reason code", sorted(KNOWN_REASON_CODES))
        amount = st.sidebar.slider("Transaction amount (INR)", 100, 20000, 1500, step=100)
        dispute = {
            'dispute_id': 'DSP-CUSTOM-001',
            'transaction_id': 'TXN-CUSTOM-001',
            'dispute_reason_code': reason_code,
            'transaction_amount_inr': amount,
            'merchant_category': 'retail',
            'days_since_transaction': 30,
            'customer_account_age_days': 365,
            'previous_dispute_count': 0,
            'customer_communication_count': 1,
            'customer_communication_sentiment': 0.0,
            'evidence_completeness_score': 0.5,
        }
        st.sidebar.caption("Reason-relevant evidence:")
        for field in REASON_RELEVANT_FIELDS.get(reason_code, []):
            dispute[field] = int(st.sidebar.checkbox(field.replace('_', ' '), value=False))

        st.sidebar.divider()
        days_left = st.sidebar.slider(
            "Days until response deadline (respond_by)", -5, 30, 10,
            help="Negative = deadline already passed. Matches Razorpay's real API field name.",
        )
        dispute['respond_by'] = (datetime.now(timezone.utc) + timedelta(days=days_left)).timestamp()
        if days_left < 0:
            st.sidebar.warning(f"Deadline passed {abs(days_left)} day(s) ago -- gate will be skipped entirely.")

    st.sidebar.divider()
    if st.sidebar.button("Run a deliberately malformed dispute instead"):
        dispute = dict(dispute)
        dispute['dispute_id'] = 'DSP-MALFORMED-DEMO'
        dispute['dispute_reason_code'] = 'chargeback_10_4_visa_new_code'

    # ---- run the real pipeline ----
    result = pipeline.process(dispute)

    # ---- decision banner (status display, not a button -- shows what the
    # pipeline already decided; there's nothing to click here) ----
    decision = result['final_decision']
    color = DECISION_COLOR.get(decision, GRAY_BAR)
    label = DECISION_LABEL.get(decision, decision)
    st.markdown(
        f"<div style='padding:14px 18px;border-radius:10px;background:{CARD_BG};"
        f"border:1px solid {CARD_BORDER};border-left:4px solid {color};'>"
        f"<div style='color:{TEXT_MUTED};font-size:11px;text-transform:uppercase;"
        f"letter-spacing:0.05em;font-weight:600;margin-bottom:4px;'>Pipeline decision</div>"
        f"<div style='color:{TEXT_PRIMARY};font-size:19px;font-weight:700;'>"
        f"<span style='display:inline-block;width:9px;height:9px;border-radius:50%;"
        f"background:{color};margin-right:9px;'></span>{label}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Pipeline stages</div>', unsafe_allow_html=True)
        stages = result['stages']

        st.markdown(f"**1. Ingestion** -- {stages.get('ingestion', {}).get('status', '-')}")

        rc = stages.get('reason_classification')
        if rc is not None:
            st.markdown(f"**2. Reason validation** -- `{rc.get('reason_code', '-')}` "
                        f"({'valid' if rc.get('valid') else 'UNRECOGNIZED'})")
        else:
            st.caption("2. Reason validation -- not reached (stopped at an earlier stage)")

        if 'evidence_retrieval' in stages:
            ev = stages['evidence_retrieval']
            st.markdown("**3. Evidence retrieval** -- relevant fields for this reason code:")
            st.json(ev['evidence'], expanded=False)

        gate = stages.get('decision_gate')
        if 'scoring' in stages and gate is not None and 'breakeven_probability' in gate:
            wp = stages['scoring']['win_probability']
            be = gate['breakeven_probability']
            st.markdown(f"**4. Win probability vs. break-even**")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=wp * 100,
                number={'suffix': '%', 'font': {'color': TEXT_PRIMARY}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': TEXT_MUTED},
                    'bar': {'color': ORANGE},
                    'bgcolor': CARD_BG,
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, be * 100], 'color': 'rgba(229,72,77,0.20)'},
                        {'range': [be * 100, 100], 'color': 'rgba(62,207,142,0.20)'},
                    ],
                    'threshold': {'line': {'color': GREEN, 'width': 3}, 'thickness': 0.85, 'value': be * 100},
                },
            ))
            st.plotly_chart(plotly_dark_layout(fig, height=200), width='stretch',
                            config={'displayModeBar': False})
            st.caption(f"Green line = break-even win probability ({be:.1%}) for this dispute's amount. "
                      f"Orange bar clearing it means the expected value of trying beats the expected "
                      f"cost of a wasted submission.")
        elif 'scoring' in stages:
            wp = stages['scoring']['win_probability']
            st.markdown(f"**4. Win probability (calibrated)** -- {wp:.1%}")
            st.progress(min(max(wp, 0.0), 1.0))

        if gate is not None:
            st.markdown("**5. Decision gate**")
            if 'breakeven_probability' in gate:
                st.markdown(f"- Expected cost of flagging (missed win): Rs.{gate['expected_fn_cost_inr']:,.0f}")
                st.markdown(f"- Expected cost of submitting (wasted try): Rs.{gate['expected_fp_cost_inr']:,.0f}")
            if gate.get('explanation'):
                st.info(gate['explanation'])
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        if 'razorpay_decline_resolution_path' in result:
            st.caption(f"If not won: {result['razorpay_decline_resolution_path']}")

        if 'packet_drafting' in stages:
            st.markdown('<div class="panel-title">Drafted evidence packet</div>', unsafe_allow_html=True)
            packet = stages['packet_drafting']
            st.markdown(f"**Narrative:** {packet['narrative']}")
            grounding = packet['grounding_check']
            if grounding['grounded']:
                st.success("Grounding check passed -- every number in the narrative traces to real evidence.")
            else:
                st.error(f"Grounding check FAILED -- ungrounded numbers: {grounding['ungrounded_numbers']}")

            schema = packet.get('razorpay_evidence_schema')
            if schema:
                st.markdown("**Mapped to Razorpay's real Contest API evidence fields:**")
                if schema['evidence_categories_present']:
                    for razorpay_field, source_fields in schema['evidence_categories_present'].items():
                        st.markdown(f"- `{razorpay_field}` -- from: {', '.join(source_fields)}")
                else:
                    st.caption("No evidence fields mapped to a submittable category.")
                st.caption(
                    f"explanation_letter: {len(schema['explanation_letter'])}/1000 chars"
                    + (" -- TRUNCATED to fit Razorpay's real limit" if schema['explanation_letter_truncated'] else "")
                )
        elif 'error' in stages:
            st.markdown('<div class="panel-title">Handled failure</div>', unsafe_allow_html=True)
            st.warning(f"{stages['error']['type']}: {stages['error']['message']}")
        else:
            st.markdown('<div class="panel-title">No packet drafted</div>', unsafe_allow_html=True)
            st.caption("This dispute either did not clear the gate or never reached it "
                      "(e.g. an unrecognized reason code), so no evidence packet was generated.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-title">Full audit trail</div>', unsafe_allow_html=True)
        st.json(result, expanded=False)
        st.markdown('</div>', unsafe_allow_html=True)