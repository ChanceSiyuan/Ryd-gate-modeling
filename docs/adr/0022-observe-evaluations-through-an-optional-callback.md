# Observe evaluations through an optional callback

`qoc.minimize` accepts an optional callback that is invoked after every
successful finite scalar-loss evaluation. The callback receives a QOC-owned
event containing the evaluation index, named candidate parameters, that
candidate's loss, and the best loss observed so far.

The optimizer does not retain or return the complete parameter trajectory by
default. A study that needs convergence plots, external logging, or
checkpointing implements those policies in its callback. The event contains no
simulator result or physical diagnostic; a study may capture its own state if
it needs additional research-specific records.
