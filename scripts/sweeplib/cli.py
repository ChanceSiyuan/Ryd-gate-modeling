"""Shared CLI scaffold for the two-atom CZ max-leakage sweep scripts.

Both scripts share the same subparser skeleton — ``--output``, ``--spacing-um``
with a ``results/<family>/a{spacing:.1f}`` derived default, ``--workers`` /
``--batch-size`` accepting ``auto``, the production/audit tolerance flags and the
``--panels`` restriction.  Each script keeps its own subcommand wiring (which
``func`` each subcommand dispatches to, its metric list and docstring) and passes
its store-family root here so the derived default and the ``--output`` help stay
per-script.
"""
from __future__ import annotations

import os

# "auto" resolves to the pilot-benchmarked host default (up to 40 of the logical
# CPUs) and the acceptance-gated packing size, so the agreed production
# invocations parse verbatim.
WORKERS_AUTO = min(40, os.cpu_count() or 40)
BATCH_SIZE_AUTO = 48


def int_or_auto(default: int):
    """argparse type accepting an int or the literal ``auto`` -> ``default``."""

    def parse(value):
        return default if value == "auto" else int(value)

    parse.__name__ = "int-or-auto"
    return parse


def add_common_args(sp, family_root: str, *, compute: bool = False) -> None:
    """Add the shared ``--output``/``--spacing-um`` (+ compute flags) to ``sp``.

    ``family_root`` is the store-family directory under ``results/`` used for the
    derived-output default in the help text (``max_leakage_ode`` / ``max_leakage_297``).
    ``compute=True`` adds the pool/tolerance/``--panels`` flags shared by the
    pilot/run/audit/scatter subcommands.
    """
    sp.add_argument("--output", default=None,
                    help="scan store directory (default: "
                         f"results/{family_root}/a{{spacing:.1f}})")
    sp.add_argument("--spacing-um", type=float, default=3.0,
                    help="atom spacing in um (physics-hash relevant; also "
                         "selects the default store directory)")
    if compute:
        sp.add_argument("--workers", type=int_or_auto(WORKERS_AUTO), default=40)
        sp.add_argument("--batch-size", type=int_or_auto(BATCH_SIZE_AUTO), default=48,
                        help="max points packed per solve (acceptance-gated)")
        sp.add_argument("--point-timeout", type=float, default=3600.0,
                        help="wall-clock timeout per point (scaled by batch size)")
        sp.add_argument("--rtol", type=float, default=1e-9,
                        help="production relative tolerance")
        sp.add_argument("--atol", type=float, default=1e-12,
                        help="production absolute tolerance")
        sp.add_argument("--audit-rtol", type=float, default=1e-10)
        sp.add_argument("--audit-atol", type=float, default=1e-13)
        sp.add_argument("--panels", default=None, metavar="DI,TI[;DI,TI...]",
                        help="restrict to specific panels (smoke tests, reruns)")


def default_output(family_root: str, spacing_um: float) -> str:
    """The derived store directory ``results/<family_root>/a{spacing:.1f}``."""
    return os.path.join("results", family_root, f"a{spacing_um:.1f}")


def resolve_output(args, family_root: str):
    """Fill ``args.output`` from the spacing-derived default when unset."""
    if args.output is None:
        args.output = default_output(family_root, args.spacing_um)
    return args
