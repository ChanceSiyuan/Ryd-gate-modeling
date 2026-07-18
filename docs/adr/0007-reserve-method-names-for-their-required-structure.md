# Reserve QOC method names for their required structure

`qoc` will distinguish black-box optimization, generic gradient optimization, GRAPE, and direct trajectory optimization by the structure each method actually uses. A solver consuming a supplied or finite-difference loss gradient is not GRAPE; GRAPE must construct time-slice gradients through forward trajectory and backward costate propagation, while a direct method must optimize intermediate states and enforce local dynamics constraints.
