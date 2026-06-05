# Vendored third-party code

## PyTreeNet

- **Package:** `pytreenet`
- **Version:** 1.0.0
- **Used by:** `ryd_gate.backends.ttn` (tree tensor-network TDVP backend)
- **License:** see `pytreenet/LICENSE`

The source tree under `pytreenet/` is an unmodified copy of the upstream
release. Only the wheel metadata (`*.dist-info/`) was dropped, since it is build
artifact rather than source. Access it through
`ryd_gate._vendor.import_pytreenet()`, which falls back to an installed
`pytreenet` if one is available.
