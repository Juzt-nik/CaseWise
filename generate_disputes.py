"""
generate_disputes.py

Synthetic chargeback dispute dataset generator for the Razorpay AI Buildathon
(Track 02 — AI Risk Manager, Chargeback Evidence Responder).

Design goals
------------
1. Realistic class imbalance: most disputes have weak/ambiguous evidence.
   Clear-cut winnable cases are a minority — this is what makes the
   win-probability classifier a genuine imbalanced-learning problem
   (i.e. why SynthEdge vs SMOTE is a meaningful comparison here, not
   a contrived one).

2. Reason-specific evidence relevance: not all evidence matters for every
   dispute type. A signature confirmation matters for "item not received"
   but is irrelevant for "duplicate charge". The label is generated from
   a reason-specific weighted combination of only the RELEVANT evidence
   fields, plus noise — so a classifier has to learn conditional
   relevance, not just "more evidence = more likely to win".

3. Deterministic-ish reasons kept mostly deterministic: "duplicate_charge"
   is close to a factual check (does a matching transaction exist?) in
   real life, so its noise term is much smaller than e.g. "not_as_described"
   which is inherently more subjective/ambiguous.

Usage
-----
    python generate_disputes.py --n 2000 --seed 42 --out disputes.csv
"""

import argparse
import numpy as np
import pandas as pd

REASON_CODES = [
    "fraud",
    "item_not_received",
    "not_as_described",
    "duplicate_charge",
    "credit_not_processed",
    "subscription_cancelled",
]

# Realistic-ish base rates for how often each reason occurs among disputes.
REASON_PROBS = [0.22, 0.28, 0.18, 0.10, 0.14, 0.08]

MERCHANT_CATEGORIES = [
    "retail",
    "digital_goods",
    "travel",
    "food_delivery",
    "subscription",
    "electronics",
]

# ---------------------------------------------------------------------------
# Reason-specific evidence relevance weights.
# Each dict maps evidence_field -> weight in the win-probability logit.
# Only fields relevant to that reason get non-zero weight; everything else
# is present in the dataset as noise/context for that row but should NOT
# drive the label for that reason (this is what makes it a "learn what's
# relevant" problem rather than a simple sum).
# ---------------------------------------------------------------------------
REASON_WEIGHTS = {
    "item_not_received": {
        "delivery_proof_available": 1.8,
        "tracking_number_valid": 1.5,
        "signature_confirmation": 1.2,
        "delivery_confirmed_before_dispute": -2.0,  # if confirmed delivered, hard to win
        "customer_communication_count": 0.05,
    },
    "fraud": {
        "avs_match": 1.6,
        "cvv_match": 1.6,
        "ip_device_match": 1.4,
        "customer_account_age_norm": 0.8,  # normalized 0-1, see row_fields
        "previous_dispute_count": -0.5,
    },
    "not_as_described": {
        "product_photos_available": 1.3,
        "customer_communication_count": 0.15,
        "customer_communication_sentiment": 0.9,
        "refund_policy_disclosed_at_purchase": 0.8,
    },
    "duplicate_charge": {
        "duplicate_transaction_exists": 3.5,  # near-deterministic
    },
    "credit_not_processed": {
        "refund_already_processed": 2.5,
        "refund_processed_before_dispute": 1.5,
    },
    "subscription_cancelled": {
        "subscription_cancellation_confirmed": 2.2,
        "customer_communication_count": 0.1,
    },
}

# Noise scale per reason: how much unmodeled ambiguity there is in outcomes.
# Lower = closer to deterministic given the evidence.
REASON_NOISE = {
    "item_not_received": 0.9,
    "fraud": 1.0,
    "not_as_described": 1.4,   # most subjective
    "duplicate_charge": 0.3,   # closest to a factual check
    "credit_not_processed": 0.6,
    "subscription_cancelled": 0.8,
}

# Reason-specific intercepts (base difficulty of winning this reason type
# even with decent evidence) — calibrated so the OVERALL win rate lands
# around 20-25%, matching real-world imbalance.
# Calibrated via iterative feedback search against target win rates
# (item_not_received=0.16, fraud=0.20, not_as_described=0.12,
#  duplicate_charge=0.30, credit_not_processed=0.18, subscription_cancelled=0.14)
# so the overall win rate lands ~18%, matching realistic dispute-outcome
# imbalance (most disputes are weak/ambiguous; clear wins are the minority).
REASON_INTERCEPT = {
    "item_not_received": -3.289,
    "fraud": -4.035,
    "not_as_described": -3.757,
    "duplicate_charge": -2.213,
    "credit_not_processed": -2.770,
    "subscription_cancelled": -3.033,
}


def _bool_skewed(rng, p_true, n):
    """Bernoulli draw where p_true controls how often evidence is PRESENT.
    Most evidence fields are skewed toward absent/weak, mirroring real
    merchant record-keeping quality."""
    return rng.random(n) < p_true


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    reason = rng.choice(REASON_CODES, size=n, p=REASON_PROBS)
    merchant_category = rng.choice(MERCHANT_CATEGORIES, size=n)

    transaction_amount = np.round(rng.lognormal(mean=7.2, sigma=1.0, size=n), 2)  # INR
    days_since_transaction = rng.integers(1, 121, size=n)
    customer_account_age_days = rng.integers(0, 2000, size=n)
    previous_dispute_count = rng.poisson(0.4, size=n)

    # Evidence fields — deliberately skewed toward weak/absent (this is the
    # source of class imbalance: winnable evidence is the minority case).
    delivery_proof_available = _bool_skewed(rng, 0.35, n)
    tracking_number_valid = _bool_skewed(rng, 0.30, n)
    signature_confirmation = _bool_skewed(rng, 0.15, n)
    delivery_confirmed_before_dispute = _bool_skewed(rng, 0.20, n)

    avs_match = _bool_skewed(rng, 0.40, n)
    cvv_match = _bool_skewed(rng, 0.45, n)
    ip_device_match = _bool_skewed(rng, 0.30, n)

    customer_communication_count = rng.poisson(1.5, size=n)
    customer_communication_sentiment = np.round(rng.normal(0.0, 0.5, size=n), 2)
    customer_communication_sentiment = np.clip(customer_communication_sentiment, -1, 1)

    refund_policy_disclosed_at_purchase = _bool_skewed(rng, 0.55, n)
    product_photos_available = _bool_skewed(rng, 0.25, n)

    duplicate_transaction_exists = _bool_skewed(rng, 0.30, n)
    refund_already_processed = _bool_skewed(rng, 0.20, n)
    refund_processed_before_dispute = _bool_skewed(rng, 0.15, n)
    subscription_cancellation_confirmed = _bool_skewed(rng, 0.25, n)

    row_fields = {
        "delivery_proof_available": delivery_proof_available.astype(float),
        "tracking_number_valid": tracking_number_valid.astype(float),
        "signature_confirmation": signature_confirmation.astype(float),
        "delivery_confirmed_before_dispute": delivery_confirmed_before_dispute.astype(float),
        "avs_match": avs_match.astype(float),
        "cvv_match": cvv_match.astype(float),
        "ip_device_match": ip_device_match.astype(float),
        "customer_account_age_norm": customer_account_age_days.astype(float) / 2000.0,
        "previous_dispute_count": previous_dispute_count.astype(float),
        "customer_communication_count": customer_communication_count.astype(float),
        "customer_communication_sentiment": customer_communication_sentiment.astype(float),
        "refund_policy_disclosed_at_purchase": refund_policy_disclosed_at_purchase.astype(float),
        "product_photos_available": product_photos_available.astype(float),
        "duplicate_transaction_exists": duplicate_transaction_exists.astype(float),
        "refund_already_processed": refund_already_processed.astype(float),
        "refund_processed_before_dispute": refund_processed_before_dispute.astype(float),
        "subscription_cancellation_confirmed": subscription_cancellation_confirmed.astype(float),
    }

    # --- compute reason-specific win logit ---
    logit = np.zeros(n)
    noise = np.zeros(n)
    intercept = np.zeros(n)
    for r in REASON_CODES:
        mask = reason == r
        if not mask.any():
            continue
        weights = REASON_WEIGHTS[r]
        r_logit = np.full(mask.sum(), REASON_INTERCEPT[r])
        for field, w in weights.items():
            r_logit = r_logit + w * row_fields[field][mask]
        logit[mask] = r_logit
        noise[mask] = REASON_NOISE[r]

    logit_noisy = logit + rng.normal(0, 1, size=n) * noise
    win_prob = 1 / (1 + np.exp(-logit_noisy))
    won = (rng.random(n) < win_prob).astype(int)

    # Evidence completeness score: simple composite over ALL evidence
    # fields present (independent of relevance) — this is a feature the
    # classifier can use, but by construction is NOT sufficient on its own
    # to predict the outcome (relevance depends on reason).
    evidence_cols = list(row_fields.keys())
    bool_like = [
        "delivery_proof_available", "tracking_number_valid", "signature_confirmation",
        "avs_match", "cvv_match", "ip_device_match",
        "refund_policy_disclosed_at_purchase", "product_photos_available",
        "duplicate_transaction_exists", "refund_already_processed",
        "refund_processed_before_dispute", "subscription_cancellation_confirmed",
    ]
    evidence_completeness_score = np.round(
        sum(row_fields[c] for c in bool_like) / len(bool_like), 3
    )

    df = pd.DataFrame({
        "dispute_id": [f"DSP{100000+i}" for i in range(n)],
        "transaction_id": [f"TXN{200000+i}" for i in range(n)],
        "transaction_amount_inr": transaction_amount,
        "days_since_transaction": days_since_transaction,
        "dispute_reason_code": reason,
        "merchant_category": merchant_category,
        "customer_account_age_days": customer_account_age_days,
        "previous_dispute_count": previous_dispute_count,
        **{k: v.astype(int) if set(np.unique(v)) <= {0.0, 1.0} else v for k, v in row_fields.items()},
        "evidence_completeness_score": evidence_completeness_score,
        "win_probability_true": np.round(win_prob, 4),  # ground-truth prob, kept for eval/debugging
        "won": won,
    })

    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic chargeback dispute dataset")
    parser.add_argument("--n", type=int, default=2000, help="number of dispute records")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--out", type=str, default="disputes.csv", help="output CSV path")
    args = parser.parse_args()

    df = generate(args.n, args.seed)
    df.to_csv(args.out, index=False)

    print(f"Generated {len(df)} records -> {args.out}")
    print(f"Overall win rate: {df['won'].mean():.3f}")
    print("\nWin rate by reason code:")
    print(df.groupby("dispute_reason_code")["won"].agg(["mean", "count"]))


if __name__ == "__main__":
    main()
