# Legacy policy-gradient prototype

`original/` preserves the initial PyTorch experiment exactly as it existed
before the supported `satrl` package was introduced. It is retained for audit
reproducibility, not as a supported solver.

The same seven files remain at the repository root to preserve their historical
paths. This directory is the explicit, stable snapshot used by the audit tools.

Two blocking behaviors are intentionally still present in that snapshot:

- preprocessing applies a two-dimensional mask and can flatten a CNF matrix;
- the policy receives a constant all-ones vector instead of the input formula.

The repository's `benchmark/` directory measures the observable behavior of
that historical prototype. New code should import `satrl` or use the `satrl`
command-line interface.
