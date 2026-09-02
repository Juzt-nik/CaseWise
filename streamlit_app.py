"""
Day 6/7 -- interactive demo of the dispute pipeline. Reuses DisputePipeline
from pipeline.py directly rather than reimplementing any of its logic, so
this dashboard can never drift out of sync with the actual decision system.

Run from the project root (same directory as model_artifacts/):
    streamlit run streamlit_app.py
"""
import json
from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st

from pipeline import DisputePipeline, REASON_RELEVANT_FIELDS, KNOWN_REASON_CODES

st.set_page_config(page_title="Chargeback Evidence Responder", layout="wide")

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


@st.cache_resource
def load_pipeline():
    return DisputePipeline()


@st.cache_data
def load_sample_disputes():
    return pd.read_csv('disputes_10k.csv')


st.title("Chargeback Evidence Responder + Win-Probability Gate")
st.caption(
    "Razorpay AI Buildathon -- Track 02: AI Risk Manager. "
    "Every decision below is produced by the real pipeline (pipeline.py), not a mock."
)

pipeline = load_pipeline()
df = load_sample_disputes()

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

# ---- decision banner ----
decision = result['final_decision']
color = DECISION_COLOR.get(decision, '#57606a')
label = DECISION_LABEL.get(decision, decision)
st.markdown(
    f"<div style='padding:16px;border-radius:8px;background:{color};color:white;"
    f"font-size:22px;font-weight:600;'>{label}</div>",
    unsafe_allow_html=True,
)
st.write("")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Pipeline stages")
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

    if 'scoring' in stages:
        wp = stages['scoring']['win_probability']
        st.markdown(f"**4. Win probability (calibrated)** -- {wp:.1%}")
        st.progress(min(max(wp, 0.0), 1.0))

    if 'decision_gate' in stages:
        gate = stages['decision_gate']
        st.markdown("**5. Decision gate**")
        if 'breakeven_probability' in gate:
            be = gate['breakeven_probability']
            st.markdown(f"- Break-even win probability for this amount: **{be:.1%}**")
            st.markdown(f"- Expected cost of flagging (missed win): Rs.{gate['expected_fn_cost_inr']:,.0f}")
            st.markdown(f"- Expected cost of submitting (wasted try): Rs.{gate['expected_fp_cost_inr']:,.0f}")
        if gate.get('explanation'):
            st.info(gate['explanation'])

with col2:
    if 'razorpay_decline_resolution_path' in result:
        st.caption(f"If not won: {result['razorpay_decline_resolution_path']}")

    if 'packet_drafting' in stages:
        st.subheader("Drafted evidence packet")
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
                f"explanation_letter: {len(schema['explanation_letter'])}/"
                f"{1000} chars"
                + (" -- TRUNCATED to fit Razorpay's real limit" if schema['explanation_letter_truncated'] else "")
            )
    elif 'error' in stages:
        st.subheader("Handled failure")
        st.warning(f"{stages['error']['type']}: {stages['error']['message']}")
    else:
        st.subheader("No packet drafted")
        st.caption("This dispute either did not clear the gate or never reached it "
                    "(e.g. an unrecognized reason code), so no evidence packet was generated.")

    st.subheader("Full audit trail")
    st.json(result, expanded=False)
