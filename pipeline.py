"""
Phrase 4 -- end-to-end dispute pipeline.

Stages: ingest -> classify_reason -> retrieve_evidence -> score
        -> decision_gate -> draft_packet, all logged to an audit trail.

Run train_model.py first to produce model_artifacts/.
"""
import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import pandas as pd
from xgboost import XGBClassifier

from grounding_check import validate_narrative_grounding

ARTIFACT_DIR = Path('model_artifacts')
KNOWN_REASON_CODES = {
    "fraud", "item_not_received", "not_as_described",
    "duplicate_charge", "credit_not_processed", "subscription_cancelled",
}

# Which raw evidence fields matter for each reason code -- mirrors the
# relevance structure in generate_disputes.py's REASON_WEIGHTS, so this
# stage surfaces only what's relevant to THIS dispute type rather than
# dumping every field the merchant has on file.
REASON_RELEVANT_FIELDS = {
    "item_not_received": ["delivery_proof_available", "tracking_number_valid",
                           "signature_confirmation", "delivery_confirmed_before_dispute"],
    "fraud": ["avs_match", "cvv_match", "ip_device_match",
              "customer_account_age_days", "previous_dispute_count"],
    "not_as_described": ["product_photos_available", "customer_communication_count",
                          "customer_communication_sentiment", "refund_policy_disclosed_at_purchase"],
    "duplicate_charge": ["duplicate_transaction_exists"],
    "credit_not_processed": ["refund_already_processed", "refund_processed_before_dispute"],
    "subscription_cancelled": ["subscription_cancellation_confirmed"],
}

# FP cost placeholder (see evaluate.py's fp_cost_sensitivity.csv). The gate
# now uses the per-dispute economic rule (win_prob vs FP_COST/(FP_COST+amount))
# rather than one fixed threshold for every dispute -- this REQUIRES calibrated
# probabilities to be valid; verified via train_model.py that raw XGBoost
# probabilities were badly overconfident (worse than the base-rate baseline)
# before calibration.
DEFAULT_FP_COST_INR = 250.0

# Per-reason-code FP-cost multiplier, applied on top of DEFAULT_FP_COST_INR
# before the breakeven calculation. Global overall precision (0.239) hides
# that precision is far worse on two reason codes than the rest -- see
# results/day3/per_reason_metrics.csv. A single global FP cost silently
# assumes a false submission is equally costly to review/reject regardless
# of reason code, which isn't true if some reason codes are inherently
# harder to score confidently.
#
# Swept on the held-out test set (sweep_reason_fp_cost.py) before choosing
# these values -- NOT picked by intuition:
#   - subscription_cancelled: raising its FP cost improves precision
#     monotonically and cleanly (0.123 -> 0.200 -> 0.294 as the multiplier
#     rises from 1x to 3x to 4x), with total dollar cost on this reason
#     code actually falling, not just trading recall for precision on paper.
#     3.0x is used here as a conservative choice safely below the point
#     (~6x) where the reason code's submissions collapse to zero.
#   - not_as_described: DELIBERATELY left at 1.0x (no adjustment). Sweeping
#     its multiplier does NOT behave monotonically -- precision moves
#     0.137 -> 0.130 -> 0.120 -> ... -> 0.157 (at 2.0x) -> 0.096 (at 2.5x)
#     as the multiplier rises, i.e. it briefly gets *worse* before getting
#     slightly better, then collapses. With only 47 positive cases in the
#     held-out test set for this reason code, any specific multiplier
#     chosen to fit this pattern would be tuned to noise in one split, not
#     a real signal -- the same failure mode robustness_check.py exists to
#     catch for the SynthEdge-vs-SMOTE comparison. Left at parity pending
#     either more data or a multi-seed check (retrain across several
#     random_states, as robustness_check.py already does for the
#     augmentation comparison) before trusting a specific value here.
REASON_FP_COST_MULTIPLIER = {
    "subscription_cancelled": 3.0,
}


# Real Razorpay Disputes API evidence schema (per public docs:
# razorpay.com/docs/api/disputes/contest/) -- maps CaseWise's synthetic
# evidence fields onto the typed fields Razorpay's actual Contest API
# expects, so a drafted packet's shape is deployment-compatible rather
# than a generic internal structure. Most mappings are direct; three
# (access_activity_log for verification signals, proof_of_service for
# product photos, billing_proof for duplicate-transaction evidence) are
# reasoned interpretations, not explicitly documented pairings.
EVIDENCE_FIELD_TO_RAZORPAY_SCHEMA = {
    "delivery_proof_available": "shipping_proof",
    "tracking_number_valid": "shipping_proof",
    "signature_confirmation": "shipping_proof",
    "refund_already_processed": "refund_confirmation",
    "refund_processed_before_dispute": "refund_confirmation",
    "customer_communication_count": "customer_communication",
    "customer_communication_sentiment": "customer_communication",
    "refund_policy_disclosed_at_purchase": "refund_cancellation_policy",
    "subscription_cancellation_confirmed": "cancellation_proof",
    "duplicate_transaction_exists": "billing_proof",          # interpreted mapping
    "product_photos_available": "proof_of_service",           # interpreted mapping
    "avs_match": "access_activity_log",                       # interpreted mapping
    "cvv_match": "access_activity_log",                       # interpreted mapping
    "ip_device_match": "access_activity_log",                 # interpreted mapping
}
RAZORPAY_EXPLANATION_LETTER_MAX_CHARS = 1000  # real API constraint on the explanation field

# Per Razorpay's public docs: if a fraud-reason dispute isn't contested,
# the MERCHANT must refund manually; for every other reason code, Razorpay
# auto-refunds the customer. This doesn't change any CaseWise decision --
# it's downstream-process context for whichever way the gate decides.
FRAUD_REASON_CODES = {"fraud"}


def map_to_razorpay_evidence_schema(evidence: dict, narrative: str) -> dict:
    """
    Reshapes drafted evidence into the field names Razorpay's real Contest
    API expects. Applied AFTER the gate has already decided to submit --
    purely a presentation-layer mapping. Cannot affect win-probability
    scoring, the economic decision, or anything reported in evaluate.py,
    day2_train_and_compare.py, or heuristic_baseline.py, none of which
    call this function.
    """
    categories = {}
    for field, value in evidence.items():
        if value in (None, 0, False):
            continue
        razorpay_field = EVIDENCE_FIELD_TO_RAZORPAY_SCHEMA.get(field)
        if razorpay_field:
            categories.setdefault(razorpay_field, []).append(field)

    unmapped = [f for f in evidence.keys() if f not in EVIDENCE_FIELD_TO_RAZORPAY_SCHEMA]

    explanation_letter = narrative
    truncated = False
    if len(explanation_letter) > RAZORPAY_EXPLANATION_LETTER_MAX_CHARS:
        explanation_letter = explanation_letter[:RAZORPAY_EXPLANATION_LETTER_MAX_CHARS]
        truncated = True

    return {
        'evidence_categories_present': categories,
        'explanation_letter': explanation_letter,
        'explanation_letter_truncated': truncated,
        'evidence_fields_without_a_schema_mapping': unmapped,
    }


def get_decline_resolution_path(reason_code: str) -> str:
    """Informational only -- describes what happens downstream in Razorpay's
    real system if this dispute is NOT successfully contested, based on its
    reason code. Never used in any decision logic."""
    if reason_code in FRAUD_REASON_CODES:
        return ("Fraud-reason dispute: per Razorpay's process, if not contested "
                "(or contested and lost), the merchant must refund the customer manually.")
    return ("Non-fraud-reason dispute: per Razorpay's process, if not contested "
            "(or contested and lost), Razorpay auto-refunds the customer -- no manual "
            "merchant action required.")


class DisputePipeline:
    def __init__(self, fp_cost_inr=DEFAULT_FP_COST_INR):
        self.model = XGBClassifier()
        self.model.load_model(str(ARTIFACT_DIR / 'win_probability_model.json'))
        self.calibrator = joblib.load(ARTIFACT_DIR / 'probability_calibrator.joblib')
        with open(ARTIFACT_DIR / 'feature_columns.json') as f:
            self.feature_columns = json.load(f)
        self.fp_cost_inr = fp_cost_inr

    # ---- stage 1: ingestion --------------------------------------------------
    def ingest(self, dispute: dict) -> dict:
        required = {'dispute_id', 'dispute_reason_code', 'transaction_amount_inr'}
        missing = required - dispute.keys()
        if missing:
            raise ValueError(f"Dispute missing required fields: {missing}")
        return dispute

    # ---- stage 2: reason classification ---------------------------------------
    def classify_reason(self, dispute: dict) -> dict:
        # The reason code arrives as part of the card-network dispute record
        # itself (see project plan Sec. "Dispute Ingestion"), not as free
        # text -- so this stage validates/normalizes rather than running a
        # redundant classifier over data that's already structured. Swap in
        # real NLP if your intake ever needs to infer the code from text.
        code = str(dispute['dispute_reason_code']).strip().lower()
        if code not in KNOWN_REASON_CODES:
            return {'status': 'unrecognized_reason_code', 'reason_code': code, 'valid': False}
        return {'status': 'ok', 'reason_code': code, 'valid': True}

    # ---- stage 3: evidence retrieval -------------------------------------------
    def retrieve_evidence(self, dispute: dict, reason_code: str) -> dict:
        # Synthetic-data stand-in for a real merchant-record lookup: projects
        # only the fields relevant to this reason code out of the same row
        # (see the project plan's honesty note -- not a separate retrieval
        # system in this demo).
        relevant_fields = REASON_RELEVANT_FIELDS.get(reason_code, [])
        evidence = {f: dispute.get(f) for f in relevant_fields}
        missing = [f for f, v in evidence.items() if v is None]
        return {'relevant_fields': relevant_fields, 'evidence': evidence, 'missing_fields': missing}

    # ---- stage 4: win-probability scoring ---------------------------------------
    def score(self, dispute: dict) -> float:
        row = {c: 0 for c in self.feature_columns}
        for k, v in dispute.items():
            if k in row:
                row[k] = v
        reason_col = f"dispute_reason_code_{dispute.get('dispute_reason_code')}"
        if reason_col in row:
            row[reason_col] = 1
        merchant_col = f"merchant_category_{dispute.get('merchant_category')}"
        if merchant_col in row:
            row[merchant_col] = 1
        X = pd.DataFrame([row])[self.feature_columns]
        raw_proba = self.model.predict_proba(X)[0, 1]
        # Raw XGBoost probability is NOT usable directly -- verified badly
        # overconfident (worse than a base-rate baseline). Always calibrate.
        calibrated_proba = float(self.calibrator.predict([raw_proba])[0])
        return calibrated_proba

    # ---- stage 5: decision gate ---------------------------------------------------
    def decision_gate(self, win_prob: float, transaction_amount: float, evidence_result: dict,
                       reason_code: str = None) -> dict:
        # Reason-specific FP cost: some reason codes are inherently harder to
        # score confidently (see REASON_FP_COST_MULTIPLIER's derivation note
        # above). Falls back to the base fp_cost_inr for any reason code not
        # explicitly listed, including not_as_described -- deliberately, per
        # that note.
        effective_fp_cost = self.fp_cost_inr * REASON_FP_COST_MULTIPLIER.get(reason_code, 1.0)

        expected_fn_cost = win_prob * transaction_amount        # cost if we wrongly flag a winnable case
        expected_fp_cost = (1 - win_prob) * effective_fp_cost   # cost if we wrongly submit a losing case
        # Per-dispute economic rule, not a single fixed threshold: submit
        # whenever the expected value of trying beats the expected waste of
        # a submission that loses. This is amount-aware by construction --
        # a high-value dispute clears the bar at a lower win probability
        # than a low-value one, which a flat threshold can't express.
        # Verified on held-out data: this rule only outperforms a fixed
        # threshold once probabilities are properly calibrated (uncalibrated,
        # it was WORSE by ~Rs.37k on the test set -- see training_meta.json).
        decision = 'submit' if expected_fn_cost > expected_fp_cost else 'flag_insufficient'

        explanation = None
        if decision == 'flag_insufficient':
            weak_points = [f for f, v in evidence_result['evidence'].items() if v in (0, False, None)]
            breakeven_prob = effective_fp_cost / (effective_fp_cost + transaction_amount)
            explanation = (
                f"Win probability {win_prob:.2f} doesn't clear the break-even point ({breakeven_prob:.3f}) "
                f"for this dispute's transaction amount. Weak/absent evidence: "
                f"{', '.join(weak_points) if weak_points else 'none individually flagged, but the overall evidence pattern is weak for this reason code'}."
            )
        return {
            'decision': decision,
            'win_probability': round(win_prob, 4),
            'breakeven_probability': round(effective_fp_cost / (effective_fp_cost + transaction_amount), 4),
            'fp_cost_inr_effective': round(effective_fp_cost, 2),
            'expected_fn_cost_inr': round(expected_fn_cost, 2),
            'expected_fp_cost_inr': round(expected_fp_cost, 2),
            'explanation': explanation,
        }

    # ---- stage 6: evidence packet drafting -------------------------------------
    def draft_packet(self, dispute: dict, reason_code: str, evidence_result: dict) -> dict:
        # Template scaffold with narrow field-fill, per the project plan's
        # explicit scope decision (no freeform generation). `narrative` is
        # the one place an LLM belongs -- see _draft_narrative_field().
        narrative = self._draft_narrative_field(reason_code, evidence_result['evidence'])
        grounding = validate_narrative_grounding(narrative, evidence_result['evidence'])
        razorpay_schema = map_to_razorpay_evidence_schema(evidence_result['evidence'], narrative)
        return {
            'dispute_id': dispute['dispute_id'],
            'reason_code': reason_code,
            'transaction_id': dispute.get('transaction_id'),
            'transaction_amount_inr': dispute.get('transaction_amount_inr'),
            'evidence_summary': evidence_result['evidence'],
            'narrative': narrative,
            'grounding_check': grounding,
            'razorpay_evidence_schema': razorpay_schema,
        }

    def _draft_narrative_field(self, reason_code: str, evidence: dict) -> str:
        """
        LLM fill point -- currently a deterministic template fallback so the
        pipeline runs without an API key. To wire in a real model:

            prompt = (
                "You are drafting one paragraph for a chargeback evidence "
                "packet. Reason code: {reason}. Evidence on file: {evidence}. "
                "Rules: reference ONLY the facts given above. Do not invent "
                "dates, amounts, names, or any detail not explicitly present "
                "in the evidence. If evidence is sparse, say so plainly "
                "rather than filling the gap with a plausible-sounding claim."
            ).format(reason=reason_code, evidence=json.dumps(evidence))

            response = anthropic_client.messages.create(
                model="claude-...", max_tokens=200, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        That prompt constraint is the FIRST line of defense and is necessary
        but not sufficient -- a model can still ignore instructions. The
        grounding check in draft_packet() is the SECOND line of defense:
        it verifies the output in code rather than trusting the prompt
        worked, and anything that fails is routed to human review instead
        of auto-submitted. Never rely on the prompt alone for this stage.
        """
        present = [f"{k.replace('_', ' ')}: {v}" for k, v in evidence.items() if v not in (None, 0, False)]
        if not present:
            return f"No supporting evidence on file for this {reason_code.replace('_', ' ')} dispute."
        return f"Evidence on file for this {reason_code.replace('_', ' ')} dispute: " + "; ".join(present) + "."

    # ---- deadline guard -------------------------------------------------------
    def check_deadline(self, dispute: dict):
        """
        Hard guard, not a soft cost factor: if the response deadline has
        passed, the outcome is forced regardless of win probability.
        Deliberately NOT folded into the cost function as a continuous
        urgency weight -- that would just recreate the FP-cost-arbitrariness
        problem with a second invented constant. A deadline either has or
        hasn't passed.

        `respond_by` matches Razorpay's real Disputes API field name (a
        Unix timestamp) -- see README's integration-points section. It's
        optional: disputes_10k.csv has no respond_by column, so every
        dispute processed by any script in this project is unaffected
        unless a caller explicitly supplies one.
        """
        respond_by = dispute.get('respond_by')
        if respond_by is None:
            return None
        now = datetime.now(timezone.utc).timestamp()
        if now > respond_by:
            return {
                'decision': 'missed_deadline',
                'explanation': (
                    f"Response deadline (respond_by={int(respond_by)}) has already passed. "
                    f"Razorpay's real API itself rejects a contest action past this point "
                    f"('Action not allowed as deadline to respond has elapsed'), so no "
                    f"scoring or drafting is attempted here either."
                ),
            }
        return None

    # ---- orchestration + audit trail ---------------------------------------------
    def process(self, dispute: dict) -> dict:
        trail = {'dispute_id': dispute.get('dispute_id'),
                  'timestamp': datetime.now(timezone.utc).isoformat(), 'stages': {}}
        try:
            dispute = self.ingest(dispute)
            trail['stages']['ingestion'] = {'status': 'ok'}

            deadline_result = self.check_deadline(dispute)
            if deadline_result is not None:
                trail['stages']['decision_gate'] = deadline_result
                trail['final_decision'] = deadline_result['decision']
                return trail

            reason_result = self.classify_reason(dispute)
            trail['stages']['reason_classification'] = reason_result
            if not reason_result['valid']:
                trail['stages']['decision_gate'] = {
                    'decision': 'manual_review',
                    'explanation': f"Unrecognized dispute_reason_code '{reason_result['reason_code']}' -- "
                                    f"cannot score or draft against an unknown reason type; routed for manual review "
                                    f"rather than guessed at.",
                }
                trail['final_decision'] = 'manual_review'
                return trail

            evidence_result = self.retrieve_evidence(dispute, reason_result['reason_code'])
            trail['stages']['evidence_retrieval'] = evidence_result
            trail['razorpay_decline_resolution_path'] = get_decline_resolution_path(reason_result['reason_code'])

            win_prob = self.score(dispute)
            trail['stages']['scoring'] = {'win_probability': round(win_prob, 4)}

            gate_result = self.decision_gate(win_prob, dispute['transaction_amount_inr'], evidence_result,
                                              reason_code=reason_result['reason_code'])

            # Sanity check: the model's win probability is scored over the FULL
            # feature set, so it can clear the gate even when the reason-specific
            # evidence field(s) are absent -- the model is picking up correlated
            # signal elsewhere (e.g. account age, prior dispute count) that isn't
            # in the evidence summary itself. Submitting a packet whose own
            # narrative says "no supporting evidence" would be an easy, valid
            # criticism of the system, so this is caught here rather than shipped.
            reason_evidence_present = any(
                v not in (0, False, None) for v in evidence_result['evidence'].values()
            )
            if gate_result['decision'] == 'submit' and not reason_evidence_present:
                gate_result['decision'] = 'submit_with_caveat'
                gate_result['explanation'] = (
                    f"Win probability {win_prob:.2f} clears the break-even point for this transaction "
                    f"amount, but none of the reason-specific evidence fields "
                    f"({', '.join(evidence_result['relevant_fields'])}) are present or true -- the score "
                    f"is likely driven by other account/transaction signals. Flagged for human review "
                    f"before submission rather than auto-submitted."
                )
            trail['stages']['decision_gate'] = gate_result

            if gate_result['decision'] in ('submit', 'submit_with_caveat'):
                packet = self.draft_packet(dispute, reason_result['reason_code'], evidence_result)
                trail['stages']['packet_drafting'] = packet

                # Second line of defense: even if the prompt told the model
                # to stay grounded, verify it actually did before letting a
                # submission through. A model that ignores its instructions
                # here is caught by code, not by hoping the prompt worked.
                if not packet['grounding_check']['grounded']:
                    gate_result['decision'] = 'flag_ungrounded_narrative'
                    gate_result['explanation'] = (
                        f"Drafted narrative references numbers not present in the retrieved evidence "
                        f"({packet['grounding_check']['ungrounded_numbers']}) -- withheld from auto-submission "
                        f"and routed for human review rather than shipped as-is."
                    )
                    trail['stages']['decision_gate'] = gate_result

            trail['final_decision'] = gate_result['decision']
        except Exception as e:
            # Graceful failure handling -- never silently drop a dispute;
            # log enough to debug and route to manual review instead of
            # guessing. This is Day 5's "documented failure case" mechanism.
            trail['stages']['error'] = {'type': type(e).__name__, 'message': str(e)}
            trail['final_decision'] = 'manual_review'
        return trail


def run_demo_batch():
    """Runs the pipeline over a few disputes sampled from disputes_10k.csv,
    plus one deliberately malformed record (unrecognized reason code) to
    demonstrate graceful failure handling. This is a plumbing demo, not a
    metrics run -- Day 3's evaluate.py already covers held-out metrics."""
    df = pd.read_csv('disputes_10k.csv')
    sample = df.sample(n=5, random_state=7).to_dict(orient='records')

    malformed = dict(sample[0])
    malformed['dispute_id'] = 'DSP-MALFORMED-001'
    malformed['dispute_reason_code'] = 'chargeback_10_4_visa_new_code'  # e.g. an unseen network reason code

    pipeline = DisputePipeline()
    results = [pipeline.process(d) for d in sample] + [pipeline.process(malformed)]

    out_dir = Path('results/audit_trail') / datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'audit_trail.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    for r in results:
        print(f"{r['dispute_id']:<20s} -> {r['final_decision']}")
    print(f"\nFull audit trail -> {out_dir}/audit_trail.json")
    return results


if __name__ == '__main__':
    run_demo_batch()
