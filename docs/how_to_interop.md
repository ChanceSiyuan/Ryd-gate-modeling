# How-To: Serialization and Pulser Interop

## Frozen v1 payloads

Every product object serializes to a dict tagged `ryd-gate/<kind>/v1`. The
payload shapes are frozen as JSON Schema files shipped inside the package
(`ryd_gate/schemas/*.v1.schema.json`):

```python
from ryd_gate.core.serialization import load_json_schema, validate_json_schema

schema = load_json_schema("sequence")          # via importlib.resources
issues = validate_json_schema(seq.to_dict(), "sequence")   # [] when valid
```

Schema validation needs the optional `jsonschema` dependency
(`pip install "ryd-gate[schema]"`); without it, `validate_json_schema`
returns one typed `serialization.jsonschema_missing` issue instead of
raising.

## Pulser abstract representation (narrow subset)

The bridge works on Pulser's JSON abstract representation **without
importing Pulser**:

```python
from ryd_gate.interop import from_pulser_abstract_repr, to_pulser_abstract_repr

seq = from_pulser_abstract_repr(payload)       # dict or JSON string -> native Sequence
payload2 = to_pulser_abstract_repr(seq)        # native Sequence -> subset payload
```

Supported subset (see the [Capability Matrix](capability_matrix.md) for the
generated table): registers with explicit qubit ids, layout trap → qubit
mappings (landing on `RegisterLayout.define_register`), the global Rydberg
channel, constant/ramp/blackman/interpolated/custom waveforms, zero-phase
pulses, delays, and `ground-rydberg` measurement. Units match natively
(integer ns, rad/µs).

Everything else raises a typed, path-aware `PulserInteropError` — nothing is
silently dropped:

```python
from ryd_gate.interop import PulserInteropError

try:
    from_pulser_abstract_repr(payload_with_eom)
except PulserInteropError as err:
    err.code        # e.g. "pulser.eom_not_supported"
    err.path        # e.g. ("operations", "3")
    err.construct   # e.g. "enable_eom_mode"
```

NoiseModel fields with matching semantics bridge both ways
(`noise_from_pulser_abstract_repr` / `noise_to_pulser_abstract_repr`);
microscopic extensions (decay, position, local RIN) refuse to export rather
than emit a lossy payload.

## Observable schedules

`ObservableConfig` is the serializable "what to record, when" object for
streaming TN measurement:

```python
from ryd_gate import ObservableConfig

config = ObservableConfig(("sum_nr",), every_ns=200)        # or times_ns=(...)
result = simulate_sequence(seq, backend="mps", observables=config,
                           backend_options={"chi_max": 64, "dt": 2.5e-7})
result.raw.metadata["obs"]["sum_nr"]
```

The exact backend keeps final-state semantics and ignores the streaming
schedule; backends outside the sequence path raise typed errors.
