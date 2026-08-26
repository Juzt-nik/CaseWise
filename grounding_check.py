"""
Hallucination guard for the LLM-drafted narrative field.

Extracts every numeric token from a drafted narrative and checks it
traces back to something actually present in the evidence dict --
catches invented dates, amounts, counts, or IDs the model wasn't given.
This is NOT a complete grounding system (a qualitative fabrication with
no numbers in it, e.g. an invented claim of tone or intent, slips past
a purely numeric check) -- it's a cheap, high-value net for the most
common and most dangerous failure mode: fabricated specifics that look
authoritative in a submission a card network will actually read.
"""
import re


def validate_narrative_grounding(narrative: str, evidence: dict) -> dict:
    # Require a digit on both sides of a decimal point so a sentence-ending
    # period after a number ("...before dispute: 1.") isn't swept in as part
    # of the number itself. Leading '-' handled so negative values (e.g.
    # customer_communication_sentiment) compare correctly instead of having
    # their sign silently dropped.
    narrative_numbers = set(re.findall(r'-?\d+(?:\.\d+)?', narrative))

    evidence_numbers = set()
    for v in evidence.values():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            evidence_numbers.add(str(v))
            if isinstance(v, float) and v == int(v):
                evidence_numbers.add(str(int(v)))

    # 0/1 are always safe -- they're how boolean evidence fields render
    # ("delivery_proof_available: 1"), not fabricated specifics.
    ungrounded = narrative_numbers - evidence_numbers - {'0', '1'}

    return {
        'grounded': len(ungrounded) == 0,
        'ungrounded_numbers': sorted(ungrounded),
        'evidence_numbers_available': sorted(evidence_numbers),
    }


if __name__ == '__main__':
    evidence = {
        'delivery_proof_available': 1,
        'tracking_number_valid': 1,
        'signature_confirmation': 0,
        'delivery_confirmed_before_dispute': 0,
    }

    # Case 1: a legitimate, fully-grounded narrative (what the template
    # version already produces, and what a well-behaved LLM should too)
    honest = "Delivery proof and a valid tracking number are on file for this dispute."
    r1 = validate_narrative_grounding(honest, evidence)
    print("Honest narrative  ->", r1)
    assert r1['grounded'], "FAILED: should be grounded"

    # Case 2: a hallucinated narrative -- invents a delivery date and a
    # signature timestamp that were never in the evidence dict at all
    hallucinated = ("Package was delivered on March 15 and signed for at "
                     "14:32, confirming receipt beyond doubt.")
    r2 = validate_narrative_grounding(hallucinated, evidence)
    print("Hallucinated narrative ->", r2)
    assert not r2['grounded'], "FAILED: should have caught the fabrication"

    # Case 3: sentence-ending period after a number shouldn't be swallowed
    # into the number itself -- this exact pattern was a real false positive
    # caught while wiring this into the pipeline.
    trailing_period = "Evidence on file: delivery confirmed before dispute: 1."
    r3 = validate_narrative_grounding(trailing_period, evidence)
    print("Trailing-period narrative ->", r3)
    assert r3['grounded'], "FAILED: sentence-ending period should not count as part of the number"

    # Case 4: negative evidence values (e.g. sentiment scores) must compare
    # with their sign intact, not silently stripped
    signed_evidence = {'customer_communication_sentiment': -0.35}
    signed_narrative = "Customer communication sentiment on file: -0.35."
    r4 = validate_narrative_grounding(signed_narrative, signed_evidence)
    print("Negative-value narrative ->", r4)
    assert r4['grounded'], "FAILED: negative value should match with sign intact"

    print("\nAll checks passed -- catches fabrication without false-flagging honest, "
          "evidence-grounded prose (including punctuation and negative-value edge cases).")
