"""Compat locks for the two live Rydberg-sweep stores.

This is the regression gate every later refactor task must keep green: it pins
the physics/pulse/model provenance hashes and the on-disk store schema of both
``scripts/max_leakage_ode_sweep.py`` and ``scripts/max_leakage_297_sweep.py``
against the current, unmodified code.  A hash mismatch here is ALWAYS an
implementation bug in the refactor — never edit the literals below.

Both scripts are loaded the same way the two focused test files load them
(importlib from ``scripts/``, registered in ``sys.modules`` under the module
names ``max_leakage_ode_sweep`` / ``max_leakage_297_sweep``).  The loader reuses
an already-registered module so that running this file alongside
``test_max_leakage_ode_sweep.py`` / ``test_max_leakage_297_sweep.py`` in one
pytest session never re-executes either script.
"""

import importlib.util
import sys
from argparse import Namespace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    mod = sys.modules.get(name)
    if mod is not None:                       # already loaded by another test file
        return mod
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


mlso = _load("max_leakage_ode_sweep", "max_leakage_ode_sweep.py")
mls297 = _load("max_leakage_297_sweep", "max_leakage_297_sweep.py")


# ── locked provenance hashes (copied verbatim; a mismatch is an impl bug) ─────

ODE_PHYSICS = "d66867f0f1f9404203933778250f859bb672e6c3081d266b2acaa734b0d06f3c"
ODE_PULSE = "671a54574d9ad674f211086f11a20174d0734427c6dd2077d4ff4635752d4f3e"
ODE_MODEL = "2b0a443017c9769519b0e493bd5372d379eee95df8b7da0e0cfb658d508817df"

S297_PHYSICS = "a6653e742bd4592e499a56f7586c50f743049db8b97a38f5a297aa696e7897ca"
S297_PULSE = "7e8bb1b09a93508ab3fe0f17c1659ca29804ba7bcdf48d54e88cc5ed336de4e9"
S297_MODEL = "17dfcb524e1ddcadaaacf833d019b93e8f4edfb57f90c32fcbd028516f925cc3"

ODE_STORE = ROOT / "results" / "max_leakage_ode" / "a3.0"
S297_STORE = ROOT / "results" / "max_leakage_297" / "a3.0"


# ── fast: physics/pulse hashes from live code ────────────────────────────────


def test_ode_physics_and_pulse_hash_locks():
    assert mlso.ScanConfig().physics_hash() == ODE_PHYSICS
    assert mlso.pulse_hash() == ODE_PULSE


def test_297_physics_and_pulse_hash_locks():
    assert mls297.ScanConfig().physics_hash() == S297_PHYSICS
    assert mls297.pulse_hash() == S297_PULSE


# ── fast: live-store round-trips (skip when the store is absent) ──────────────


@pytest.mark.skipif(not ODE_STORE.exists(), reason=f"missing store {ODE_STORE}")
def test_ode_live_store_roundtrip():
    store = mlso.Store(str(ODE_STORE))
    manifest = store.load_manifest()
    assert manifest is not None
    assert manifest["physics_hash"] == ODE_PHYSICS
    assert manifest["model_hash"] == ODE_MODEL
    assert manifest["pulse_hash"] == ODE_PULSE
    records = store.load_records(manifest, include_states=False)
    assert sum(1 for r in records if r.status == "ok") > 0
    scatter = store.load_scatter_records(manifest)
    assert len(scatter) > 0


@pytest.mark.skipif(not S297_STORE.exists(), reason=f"missing store {S297_STORE}")
def test_297_live_store_roundtrip():
    store = mls297.Store(str(S297_STORE))
    manifest = store.load_manifest()
    assert manifest is not None
    assert manifest["physics_hash"] == S297_PHYSICS
    assert manifest["model_hash"] == S297_MODEL
    assert manifest["pulse_hash"] == S297_PULSE
    records = store.load_records(manifest, include_states=False)
    assert sum(1 for r in records if r.status == "ok") > 0
    # The 297 store may legitimately have no scatter rows yet; only pin that it
    # parses without error.
    assert isinstance(store.load_scatter_records(manifest), list)


# ── fast: cmd_status smoke on each live store ────────────────────────────────


@pytest.mark.skipif(not ODE_STORE.exists(), reason=f"missing store {ODE_STORE}")
def test_ode_cmd_status_smoke(capsys):
    mlso.cmd_status(Namespace(output=str(ODE_STORE)))
    assert "records:" in capsys.readouterr().out


@pytest.mark.skipif(not S297_STORE.exists(), reason=f"missing store {S297_STORE}")
def test_297_cmd_status_smoke(capsys):
    mls297.cmd_status(Namespace(output=str(S297_STORE)))
    assert "records:" in capsys.readouterr().out


# ── slow: full ARC model-hash locks (each builds 8 panels, ~minutes) ─────────


@pytest.mark.slow
def test_ode_warm_and_build_model_hash_lock():
    _ops_by_delta, _omega_1013, model_hash, _checks = mlso.warm_and_build(
        mlso.ScanConfig())
    assert model_hash == ODE_MODEL


@pytest.mark.slow
def test_297_warm_and_build_model_hash_lock():
    _ops_by_n, model_hash, _checks = mls297.warm_and_build(mls297.ScanConfig())
    assert model_hash == S297_MODEL
