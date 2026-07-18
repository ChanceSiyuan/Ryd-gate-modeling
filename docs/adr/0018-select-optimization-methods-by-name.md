# Select optimization methods by stable names

The base interface selects an optimization implementation through a stable
lower-case method name and an optional plain options mapping:

```python
qoc.minimize(loss, x0, method="nelder-mead", options={...})
```

The first release supports `"nelder-mead"`, `"powell"`, and `"l-bfgs-b"`.
Callers do not construct public solver classes. `qoc` validates the method name,
dispatches to its implementation, and keeps any underlying numerical-library
objects private. Method-specific controls remain in the selected method's
options mapping rather than widening the common `minimize` interface.
