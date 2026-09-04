# Known Trade-off: The Gate Favors High-Value Disputes

## What happens

The decision gate submits evidence whenever the expected value of trying
beats the expected cost of a wasted submission:

```
submit if:  win_probability > FP_COST / (FP_COST + transaction_amount)
```

Because `transaction_amount` sits in the denominator, the break-even win
probability required to trigger a submission **falls as transaction amount
rises** -- a high-value dispute clears the gate at a much lower win
probability than a low-value one.

Concretely, at the system's `FP_COST_INR = 250` default:

| Transaction amount | Win probability needed to submit |
|---|---|
| Rs.500 | 33.3% |
| Rs.1,500 | 14.3% |
| Rs.5,000 | 4.8% |

Two disputes with the **identical win probability of 15%** get opposite
decisions purely because of transaction size: the Rs.1,500 dispute clears
the gate and gets submitted, the Rs.500 dispute does not.

## Why this is deliberate, not a bug

This is the correct behavior for the stated objective -- minimizing total
rupee cost across all disputes. A missed high-value win costs far more than
a missed low-value one, so the system is right to take more chances on
larger disputes. Removing this behavior would mean optimizing for something
other than the cost function the whole gate is built around.

## The trade-off worth naming explicitly

Holding evidence quality constant, **smaller transactions systematically
get less benefit of the doubt than larger ones.** A customer or merchant
with a marginal case on a Rs.500 transaction is less likely to see that
dispute fought than the same evidence quality attached to a Rs.5,000
transaction. This is an economically rational policy, but it is also a
distributive choice -- it concentrates the system's effort on high-value
disputes, which is exactly what a cost-minimizing operator would want and
exactly what a fairness-minded reviewer would want disclosed rather than
discovered.

This connects directly to a product-strategy question worth stating
plainly (see `README.md` Section 7): the merchants with the least capacity
to fight chargebacks on their own are often smaller merchants, whose
transactions skew toward the lower end of this table. A flat, global
`FP_COST_INR` therefore gives the least benefit of the doubt to exactly the
segment this system could help most. The honest resolution -- a
merchant-relative FP-cost assumption instead of one global constant -- is
named as a next iteration, not built in this submission, because
implementing it would change the gate's core formula and require fully
regenerating every number in `README.md` Section 4 to stay honest.

## Interaction with the deadline guard

The amount-based asymmetry above only applies to disputes that still have
time to respond. `pipeline.py`'s deadline guard (`check_deadline`) runs
*before* the economic gate and, if the response deadline has already
passed, short-circuits straight to a `missed_deadline` outcome regardless
of win probability or transaction amount. A missed deadline overrides the
cost-asymmetry logic entirely rather than interacting with it.

## What we are not claiming

We are not claiming this trade-off is wrong, and we are not proposing to
silently remove it -- a flat, amount-blind threshold was tested directly
against it (see `heuristic_baseline.py`) and produced worse total cost by
concentrating its own misses randomly across transaction values rather than
deliberately on the low end. The point of this document is disclosure, not
apology: anyone deploying or auditing this system should know the trade-off
exists and why, rather than infer it from behavior after the fact.
