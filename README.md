# CaseWise

**Chargeback Evidence Responder + Win-Probability Gate**
Razorpay AI Buildathon 2026 -- Track 02: AI Risk Manager

An agent that decides, per dispute, whether fighting a chargeback is worth
more than it costs -- and drafts the evidence response if so. Built around
the track's stated bar: *"honest metrics including false-positive cost;
strictly defense-only, anything offense-capable is disqualified."*

---

## 1. What it does

A dispute enters with a reason code, a transaction amount, and (optionally)
a response deadline. The system:

1. Checks whether the response deadline has already passed -- if so,
   stops immediately, before any scoring or drafting is attempted.
2. Validates the reason code (six known types: `fraud`,
   `item_not_received`, `not_as_described`, `duplicate_charge`,
   `credit_not_processed`, `subscription_cancelled`).
3. Retrieves only the evidence fields relevant to that reason code.
4. Scores a calibrated win probability with a gradient-boosted classifier.
5. Runs a **per-dispute, amount-aware economic decision**: submit only if
   the expected value of trying (win probability x transaction amount)
   beats the expected cost of a wasted attempt.
6. If it clears the gate, drafts a template-based evidence packet, maps
   it onto Razorpay's real Contest API evidence schema, and verifies the
   draft references only real, retrieved evidence before allowing it
   through.
7. Logs every stage, every number, every decision to an audit trail --
   nothing happens silently.

It never autonomously acts past what it can defend: uncertain cases fail
toward a human, not toward automation.

## 2. Architecture

```
Dispute record (dispute_reason_code, transaction_amount_inr, respond_by?)
     |
     v
[0] Deadline guard ---> respond_by passed? --> MISSED_DEADLINE (hard stop)
     |
     v
[1] Ingest -----------> required fields present?
     |
     v
[2] Reason validation -> known reason code, or -> MANUAL_REVIEW
     |
     v
[3] Evidence retrieval -> project only reason-relevant fields
     |
     v
[4] Win-probability scoring -> XGBoost (SynthEdge-augmented)
     |                          -> isotonic-calibrated
     v
[5] Decision gate -> win_prob vs FP_COST/(FP_COST+amount)
     |         \\
     v          v
  SUBMIT    FLAG_INSUFFICIENT
     |
     v
[6] Packet drafting -> template narrative -> grounding check
     |                 -> mapped to Razorpay's real evidence schema
     v          \\
  SUBMIT      FLAG_UNGROUNDED_NARRATIVE (routed to human review)
     |
     v
Audit trail (every stage, every dispute, every number)
```

Six possible outcomes per dispute: `submit`, `submit_with_caveat` (score
clears the gate but the reason-specific evidence itself is absent --
caught deliberately, see Section 5), `flag_ungrounded_narrative`,
`flag_insufficient`, `manual_review`, `missed_deadline`.

## 3. Dataset

`generate_disputes.py` produces a synthetic dispute dataset with
reason-specific evidence relevance (not all evidence matters for every
reason type) and realistic class imbalance. Canonical dataset used
throughout: **10,000 rows**, `--seed 42`, overall win rate 17.9%.

No public labeled dataset for chargeback outcomes exists, so this is
disclosed as a deliberate, stated design choice, not a hidden gap.

## 4. Results

All numbers below are from `model_artifacts/training_meta.json`,
`results/day3/`, and `results/day2_full_output.txt`. Re-running these
scripts may shift figures slightly (SynthEdge's CTGAN step has a known,
partially-mitigated non-determinism -- see Section 6); treat these as the
result of one specific, reproducible run, not guaranteed-exact constants.
None of Sections 4's numbers are affected by the Section 5.4 additions
below (deadline guard, evidence-schema mapping, resolution-path note) --
all three are presentation-layer or hard-guard additions verified not to
touch scoring or the economic decision.

### 4.1 Augmentation comparison (Day 2)

Baseline vs. SMOTE vs. SynthEdge, same 6,400-row training split, same
held-out 2,000-row test set:

| Method | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Baseline (class-weighted, no augmentation) | 0.573 | 0.341 | 0.428 | 0.733 |
| SMOTE (+4,106 rows) | 0.165 | 0.596 | 0.258 | 0.730 |
| SynthEdge (+17 rows, severity=SEVERE) | 0.592 | 0.358 | 0.446 | 0.736 |

The headline finding is in the per-reason breakdown, not the aggregate:

| Reason code | Baseline recall | SMOTE recall | SynthEdge recall |
|---|---|---|---|
| not_as_described | 0.191 | **0.000** | 0.255 |
| subscription_cancelled | 0.304 | **0.000** | 0.391 |

SMOTE does not just underperform on average -- it gets **zero recall** on
the two hardest, most subjective reason codes, never once catching a
winnable case there. SynthEdge, adding only 17 rows, is the only method
that improves recall specifically where it's hardest to.

**This was verified across 5 different random train/test splits, not just
the one shown above (`robustness_check.py`, `results/robustness_check.csv`).**
SMOTE's failure on `not_as_described` is exactly 0.000 recall in all 5
seeds tested, zero variance -- robust, not a fluke of one particular split.
`subscription_cancelled` is nearly as consistent: 0.000 in 4 of 5 seeds.

The precise, verified claim about SynthEdge is narrower than it might look
from the single split above: across the 5 seeds, SynthEdge's recall on
`not_as_described` was **identical to baseline in every single seed** --
it doesn't improve this reason code, it simply doesn't damage it, unlike
SMOTE. On `subscription_cancelled` the picture is a genuine mixed bag
(better in 2 seeds, worse in 2, roughly even in 1). The defensible
headline is therefore: **SynthEdge preserves baseline-level recall on the
hardest reason codes; SMOTE catastrophically destroys it.** That's a
smaller claim than "SynthEdge improves recall where it's hardest," but
it's the one that actually survives a multi-seed check rather than one
that might not survive a judge running their own.

### 4.2 Probability calibration (Day 4/6)

The raw XGBoost model (trained with `scale_pos_weight` to handle class
imbalance) produces probabilities that are **not usable as true
probabilities**:

| | Brier score (lower is better) |
|---|---|
| Raw XGBoost `predict_proba` | 0.190 |
| Predicting the base rate for everyone | 0.147 |
| Isotonic-calibrated (on a real, unaugmented, held-out split) | **0.128** |

The raw model was worse than a trivial baseline -- badly overconfident.
This matters beyond a diagnostic footnote: the decision gate's economics
(`expected cost = win_prob x amount`) are meaningless if `win_prob` isn't
calibrated. Verified directly: the amount-aware economic decision rule
performs *worse* than a single fixed threshold when fed raw probabilities,
and clearly *better* once calibrated -- proof the fixed-threshold approach
was silently compensating for bad calibration, not that calibration didn't
matter.

### 4.3 Evaluation with rupee-denominated cost (Day 3)

On the held-out 2,000-dispute test set, at the chosen default
`FP_COST_INR = 250` (see Section 5.1 for justification):

| Policy | Total cost |
|---|---|
| Flag every dispute as insufficient (never submit) | Rs.821,876 |
| Submit every dispute automatically | Rs.410,500 |
| **This system's gate** | **Rs.302,787** |

The gate saves ~Rs.107,713 over the better of the two naive policies.

Per-reason precision/recall/F1:

| Reason | n | n_positive | Precision | Recall | F1 |
|---|---|---|---|---|---|
| credit_not_processed | 275 | 49 | 0.256 | 0.633 | 0.365 |
| duplicate_charge | 197 | 57 | 0.505 | 0.877 | 0.641 |
| fraud | 447 | 85 | 0.229 | 0.647 | 0.338 |
| item_not_received | 556 | 97 | 0.232 | 0.649 | 0.342 |
| not_as_described | 368 | 47 | 0.134 | 0.404 | 0.201 |
| subscription_cancelled | 157 | 23 | 0.123 | 0.435 | 0.192 |
| **OVERALL** | 2000 | 358 | **0.239** | **0.637** | **0.348** |

FP-cost sensitivity (how the gate's behavior changes if the real assumed
cost differs from Rs.250):

| Assumed FP cost | Precision | Recall | F1 |
|---|---|---|---|
| Rs.100 | 0.200 | 0.855 | 0.325 |
| Rs.150 | 0.219 | 0.779 | 0.342 |
| **Rs.250 (chosen default)** | **0.239** | **0.637** | **0.348** |
| Rs.400 | 0.274 | 0.517 | 0.358 |
| Rs.600 | 0.315 | 0.408 | 0.355 |
| Rs.1000 | 0.384 | 0.291 | 0.331 |

### 4.4 ML vs. a simple heuristic (does the ML add anything?)

Heuristic: submit if >=50% of the reason-relevant evidence fields are
present, using the identical field set and cost function as the ML system
-- a fair comparison, not a strawman.

| Method | Precision | Recall | F1 | Total cost |
|---|---|---|---|---|
| Heuristic (>=50% evidence) | 0.285 | **0.701** | **0.405** | Rs.405,216 |
| ML (calibrated + economic gate) | 0.239 | 0.637 | 0.348 | **Rs.302,787** |

On raw classification metrics, the heuristic wins -- higher recall, higher
F1, even perfect recall (1.000) on `not_as_described`, where the ML system
only reaches 0.404. **The ML system's advantage is not accuracy -- it's
where the mistakes land.** Checked directly against the missed-dispute
values on the same test set:

| | Missed disputes (n) | Average value of a miss |
|---|---|---|
| Heuristic | 107 | Rs.2,313 (== the dataset's overall average -- value-blind) |
| ML system | 130 | Rs.933 (well below average -- deliberately concentrated on low-value disputes) |

The heuristic misses transaction value at random. The ML system's
amount-aware gate deliberately trades a few more total misses for making
sure the misses that do happen are the cheap ones. That's the actual
value-add, stated precisely rather than as a blanket "ML wins" claim.

## 5. Honest design decisions

### 5.1 FP-cost assumption (Rs.250)

Industry estimates put full manual chargeback response at 2-5 hours of
analyst time. This system automates evidence retrieval and packet
drafting, leaving mostly review time for a human -- estimated at 15-30
minutes, which at a planning-assumption rate of Rs.400-600/hr lands at
roughly Rs.100-300. Rs.250 is the chosen default, not an unexamined
placeholder; Section 4.3's sensitivity table shows exactly how the
system's behavior changes if the real number differs.

### 5.2 The gate favors high-value disputes

Because `breakeven_probability = FP_COST / (FP_COST + amount)`, a
high-value dispute clears the gate at a much lower win probability than a
low-value one (Rs.500 needs 33.3% win probability to submit; Rs.5,000
needs only 4.8%). This is the correct behavior for minimizing total cost,
and it is also a real distributive trade-off -- smaller transactions get
less benefit of the doubt at equal evidence quality. Full detail in
`cost_asymmetry_disclosure.md`. See Section 7 for why this specifically
matters for who CaseWise helps most.

### 5.3 Evidence "retrieval" is a projection, not a real lookup system

In this demo, evidence fields already sit on the same dispute row; the
retrieval stage selects only the reason-relevant subset. A production
system would replace this with a real merchant-record query -- the
decision logic downstream is unaffected either way.

### 5.4 Deadline handling is a hard guard, not a soft cost factor

`check_deadline()` treats an elapsed response deadline (`respond_by`,
matching Razorpay's real API field name -- a Unix timestamp) as an
immediate stop, not a continuous urgency weight folded into the cost
function. Deliberately: adding a second, invented weighting constant
would just recreate the exact FP-cost-arbitrariness problem Section 5.1
already had to solve carefully. A deadline either has passed or hasn't.
This field is optional and unused anywhere in `disputes_10k.csv`, so
every number in Section 4 is provably unaffected by this guard's
existence -- verified by rerunning the full pipeline against all
generated results and confirming byte-identical decisions before and
after adding it.

### 5.5 No live LLM call yet

The evidence-packet narrative is a deterministic template, not a model
call, so the pipeline runs without an API key. The integration point and
prompt constraint are written and ready (`pipeline.py`'s
`_draft_narrative_field`), defended by a two-layer hallucination guard: a
prompt-level grounding instruction, plus a code-level check
(`grounding_check.py`) that verifies every number in a drafted narrative
traces back to real retrieved evidence before allowing a submission
through -- tested against both honest and deliberately fabricated
narratives.

## 6. Alignment with Razorpay's real Disputes API

Three integration points below are grounded directly in Razorpay's public
API documentation, not assumptions -- built specifically so the project
reads as a plausible v0 of a real internal tool, not a standalone exercise.

**Deadline-awareness (Section 5.4).** Razorpay's real dispute object
exposes a `respond_by` timestamp, and the API rejects contest actions
after it elapses. CaseWise's guard uses the identical field name and the
identical hard-stop behavior.

**Evidence schema mapping.** Razorpay's real Contest API expects typed
evidence fields. A drafted packet's evidence is mapped onto them directly:

| CaseWise field | Real Razorpay evidence field |
|---|---|
| `delivery_proof_available`, `tracking_number_valid`, `signature_confirmation` | `shipping_proof` |
| `refund_already_processed`, `refund_processed_before_dispute` | `refund_confirmation` |
| `customer_communication_count`, `customer_communication_sentiment` | `customer_communication` |
| `refund_policy_disclosed_at_purchase` | `refund_cancellation_policy` |
| `subscription_cancellation_confirmed` | `cancellation_proof` |
| `duplicate_transaction_exists` | `billing_proof` (interpreted) |
| `product_photos_available` | `proof_of_service` (interpreted) |
| `avs_match`, `cvv_match`, `ip_device_match` | `access_activity_log` (interpreted) |
| the drafted narrative | `explanation_letter`, truncated to Razorpay's real 1000-character limit |

**Downstream resolution path.** Per Razorpay's process, a fraud-reason
dispute that isn't successfully contested requires the *merchant* to
refund manually; every other reason code is auto-refunded by Razorpay.
`get_decline_resolution_path()` surfaces which applies, so a decision to
flag a dispute as insufficient comes with the correct downstream
implication attached, not a one-size-fits-all assumption.

## 7. Product strategy: who this actually helps most

Large merchants typically already have dispute-management teams and
tooling. The segment with the least capacity to fight chargebacks at all
is the long tail of small merchants, who often simply absorb every
dispute. That's arguably where this system creates the most real value --
not optimizing an existing process, but extending a capability a segment
currently doesn't have.

This is in tension with Section 5.2: a flat, global `FP_COST_INR`
systematically gives smaller transactions -- which skew toward smaller
merchants -- less benefit of the doubt. The honest resolution, not yet
built: a merchant-relative FP-cost assumption (scaled to that merchant's
typical transaction size or support capacity) rather than one global
constant. Deliberately **not implemented in this submission** -- doing so
would change the constant inside the gate's core formula and invalidate
every number in Section 4, which would need fully regenerating to stay
honest this close to submission. Named here as the explicit next
iteration rather than silently built into a second, unreconciled set of
results.

## 8. Known bugs found and fixed during the build

Not polish -- direct evidence of testing our own tooling, not just the
model:

- **Data leakage in SynthEdge's own `compare` CLI.** `cli.py`'s
  `cmd_compare()` fits SynthEdge's gap analysis on the full dataset,
  test rows included, while its SMOTE step correctly uses training data
  only -- inflating SynthEdge's apparent advantage. Not used anywhere in
  this project's own results; flagged for a fix in the SynthEdge repo
  directly.
- **Non-deterministic CTGAN training.** SynthEdge's `fill()` has no
  `random_state` of its own -- the underlying `ctgan` package has no seed
  parameter at all, so results drifted between runs. Patched directly in
  the SynthEdge GitHub repo (seeds `random`/`numpy`/`torch` before CTGAN
  training); verified fixed via two consecutive runs producing identical
  output. Not yet pulled into every environment this project has been
  tested in, so exact figures in this README may still drift slightly on
  rerun.
- **Evidence-inconsistent submissions.** The win-probability model scores
  over the full feature set, so it can clear the gate while the
  reason-specific evidence field is entirely absent -- correlated signal
  elsewhere pushing the score up. Caught by testing, not by design; now
  downgrades to `submit_with_caveat` instead of auto-submitting a packet
  whose own narrative would say "no supporting evidence."
- **Miscalibrated probabilities silently working, not correctly.** See
  Section 4.2 -- caught by directly checking a Brier score against a
  trivial baseline, not by assuming XGBoost's output was trustworthy.
- **A dashboard label that would have misdescribed a correct decision.**
  While adding the deadline guard, found that the UI's "Reason validation"
  line assumed `reason_classification` always exists in the audit trail --
  false for a dispute stopped by the new deadline guard, which would have
  displayed a misleading "UNRECOGNIZED" label for a dispute whose reason
  code was never even checked. Fixed and verified before it shipped.

## 9. Documented failure case

See `day5_failure_case.md`: an unrecognized dispute reason code (simulating
a new card-network reason type) is caught at the reason-validation stage
and routed to manual review with a clear audit-trail explanation --
never scored, never guessed at, never silently dropped.

## 10. Repo contents and how to run it

See `SETUP.md` for the exact file manifest, dependency list, and run order.
Interactive demo: `streamlit run streamlit_app.py`.

## 11. Limitations and future work

- Evaluated entirely on synthetic data; no external dataset exists to
  validate the generator's assumptions against. Stated openly rather than
  hidden.
- No live feedback loop: the model does not currently retrain on the
  outcomes of disputes it has actually decided on. Razorpay's own
  `payment.dispute.won` / `payment.dispute.lost` webhook events are the
  natural ground-truth signal for this -- the mechanism to consume them
  is not yet built, only identified.
- Merchant-relative FP-cost (Section 7) is named but not implemented, for
  the reasons stated there.
- Held-out test set is 2,000 disputes; some reason codes (subscription
  cancelled: 23 positive cases) are thin enough that per-reason metrics
  carry real sampling noise -- treat single-decimal differences between
  reason codes as indicative, not precise.
