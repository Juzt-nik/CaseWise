# Documented Failure Case

**Dispute:** `DSP-MALFORMED-001`
**Failure type:** Unrecognized dispute reason code
**Outcome:** Gracefully routed to manual review -- no crash, no silent misclassification

## What happened

The pipeline was fed a dispute carrying `dispute_reason_code = "chargeback_10_4_visa_new_code"` --
a value outside the six reason codes the system was trained on (`fraud`,
`item_not_received`, `not_as_described`, `duplicate_charge`,
`credit_not_processed`, `subscription_cancelled`). This simulates a real
scenario the system will eventually face: a card network introducing a new
dispute reason code, or a malformed/unexpected value reaching the pipeline
from upstream.

## How it was handled

The `classify_reason` stage validates the incoming code against the known
set *before* any evidence retrieval or scoring happens. On a match failure,
the pipeline does not:
- guess a reason category and proceed anyway, or
- fall through to the win-probability model with a default/placeholder reason code, or
- crash the batch and drop the dispute silently.

Instead it short-circuits immediately to `final_decision = "manual_review"`,
with a plain-language explanation logged in the audit trail:

> Unrecognized dispute_reason_code 'chargeback_10_4_visa_new_code' -- cannot
> score or draft against an unknown reason type; routed for manual review
> rather than guessed at.

The full audit trail entry (`results/audit_trail/audit_trail.json`) shows the
`ingestion` stage completed normally and the `reason_classification` stage
is where the failure was caught -- evidence retrieval, scoring, and packet
drafting never ran for this dispute, since none of those stages are
meaningful without a valid reason code.

## Why this matters for the track's evaluation bar

This is a deliberate design choice, not an oversight: a system that guesses
at an unfamiliar reason code and scores it anyway risks either wasting a
response window on a malformed submission or silently mis-handling a
legitimate dispute. Failing closed -- flagging for human review instead of
acting on uncertain input -- is the conservative, defense-only behavior the
track's evaluation bar asks for.

## A second example of the same pattern

This is the deliberately-constructed failure case for Phrase 5, but it isn't
the only place the pipeline fails closed rather than guessing. `pipeline.py`
also includes a deadline guard (`check_deadline`): if a dispute's response
deadline (`respond_by`) has already passed, the pipeline stops immediately
with a `missed_deadline` outcome, before any scoring or drafting is
attempted -- the same "stop and route rather than proceed on uncertain or
inapplicable input" philosophy, applied to a second, independent failure
mode. Between the two, the pipeline currently recognizes six distinct
outcomes (`submit`, `submit_with_caveat`, `flag_ungrounded_narrative`,
`flag_insufficient`, `manual_review`, `missed_deadline`) -- every one of
them an explicit state with a logged reason, never a silent default.
