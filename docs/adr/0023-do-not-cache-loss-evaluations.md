# Do not cache loss evaluations

`qoc.minimize` calls the supplied loss whenever the selected solver requests an
evaluation, even if the same packed coordinates were evaluated earlier. It
does not memoize scalar losses or deduplicate callback events.

Only the study knows whether its simulation is deterministic, stochastic,
stateful, or safe to reuse. A study with an expensive deterministic loss may
wrap that loss in its own explicitly keyed cache; a noisy study may deliberately
reevaluate identical parameters. QOC preserves those semantics by default.
