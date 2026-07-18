# Fail fast on invalid loss evaluations

The base optimizer requires every successful loss call to return one finite real
scalar. If the loss raises an exception, `qoc` propagates it without converting
it into solver failure or a penalty. If the loss returns a complex, non-scalar,
NaN, or infinite value, `qoc` raises a validation error and stops the solve.

`qoc` never guesses that a failed simulation represents a bad physical
candidate and never silently substitutes a large objective value. A study that
intentionally treats a known class of candidates as penalized must catch that
condition inside its own loss and explicitly return a finite penalty.
