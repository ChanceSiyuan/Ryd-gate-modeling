"""Regenerate ``precompile_payload.json`` for the TNQS sysimage build.

The precompile workload drives the *real* kernel code path, so its payload must
match the schema the production Python bridge emits (``build_tnqs_payload``).
Generating it from the actual builder guarantees every ``runtime`` key the kernel
reads via ``_property_or`` is present.

Run from the repo root:

    uv run python src/ryd_gate/backends/itensor/julia/sysimage_build/gen_precompile_payload.py
"""

from __future__ import annotations

import json
from pathlib import Path

from ryd_gate.backends.itensor.backend import _jsonable
from ryd_gate.backends.itensor.tnqs_backend import build_tnqs_payload
from ryd_gate.backends.tn_common.compiler import TNEvolutionIR
from ryd_gate.backends.tn_common.lattice_spec import create_tn_lattice_spec
from ryd_gate.backends.tn_common.simulate import _protocol_context
from ryd_gate.protocols.lattice_dynamics import TFIMQuenchProtocol


def main() -> None:
    # Smallest non-trivial case: 2x2 with nearest-neighbour vdW pairs, two Trotter
    # steps, both observables, CPU. This exercises local + pair gate layers,
    # apply_gates, truncate, and bp measurement during the precompile trace.
    spec = create_tn_lattice_spec(2, 2, V_nn=4.0, interaction_mode="nn")
    proto = TFIMQuenchProtocol(hx=1.0, t_gate=0.2)
    params = proto.unpack_params([], _protocol_context(spec))
    ir = TNEvolutionIR(
        spec=spec,
        protocol=proto,
        params=params,
        method="2dtn_bp",
        metadata={"compiler": "tn", "tn_spec": spec, "backend": "2dtn", "n_sites": spec.N},
    )
    payload = build_tnqs_payload(
        ir,
        initial_state="all_ground",
        t_eval=[0.0, 0.1, 0.2],
        observables=["sigma_z", "czz_centerline"],
        dt=0.1,
        chi_max=4,
        svd_min=1e-10,
        use_cuda=False,
        measurement_alg="bp",
        measurement_bond_dim=4,
        chi_2d_prime=4,
        normalize_tensors=False,
        eltype="ComplexF64",
    )

    out = Path(__file__).with_name("precompile_payload.json")
    out.write_text(json.dumps(_jsonable(payload), sort_keys=True, indent=2), encoding="utf-8")
    print(f"wrote {out} ({len(payload['schedule'])} steps, N={payload['lattice']['N']})")


if __name__ == "__main__":
    main()
