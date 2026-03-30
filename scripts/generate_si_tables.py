"""Generate PDF error budget tables combining deterministic and MC results.

Loads pre-computed data from data/ directory, or generates it inline.
Produces one PDF per detuning sign.

Usage:
    uv run python scripts/generate_si_tables.py
    uv run python scripts/generate_si_tables.py --recompute
    uv run python scripts/generate_si_tables.py --generate-mc --n-mc-shots 100
"""
import argparse
import datetime
import json
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from ryd_gate.ideal_cz import CZGateSimulator, MonteCarloResult

SSS_12_STATES = [f"SSS-{i}" for i in range(12)]

X_TO_OUR_DARK = [
    -0.6990251940088914, 1.0294930712188455, 0.37642793463018853, 1.5710847832834478, 1.4454415553284314, 1.340639491094446
]
X_TO_OUR_BRIGHT = [
    0.6246672641243727, 1.2369507331752663, -0.470787497434612, 1.6547386752699043, 3.41960305947842, 1.3338111168065905
]


def save_deterministic_errors(errors, filepath):
    """Save deterministic error dict to a text file."""
    with open(filepath, "w") as f:
        f.write(f"# DeterministicErrors saved {datetime.datetime.now().isoformat()}\n")
        f.write(f"# fid_type = sss\n")
        f.write(f"# initial_states = SSS-0..SSS-11\n")
        f.write(json.dumps(errors, indent=2))
    print(f"  Saved {filepath}")


def load_deterministic_errors(filepath):
    """Load deterministic error dict from a text file."""
    text = Path(filepath).read_text()
    # Strip comment header lines
    json_lines = [line for line in text.splitlines() if not line.startswith("#")]
    return json.loads("\n".join(json_lines))


def compute_deterministic_errors(sign, x):
    """Compute deterministic error sources with 2-state average."""
    errors = {}

    # Rydberg decay
    print("  Rydberg decay...")
    sim_ryd = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=sign,
        enable_rydberg_decay=True,
    )
    infid_ryd = sim_ryd.gate_fidelity(x,fid_type="sss")
    budget_ryd = sim_ryd.error_budget(x,initial_states=SSS_12_STATES)
    errors["rydberg_decay"] = {
        "infidelity": infid_ryd,
        **budget_ryd["rydberg_decay"],
    }

    # Intermediate decay (full 0+1 scattering)
    print("  Intermediate decay (full)...")
    sim_mid = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=sign,
        enable_intermediate_decay=True,
    )
    infid_mid = sim_mid.gate_fidelity(x,fid_type="sss")
    budget_mid = sim_mid.error_budget(x,initial_states=SSS_12_STATES)
    bm = budget_mid["intermediate_decay"]
    errors["intermediate_decay"] = {
        "infidelity": infid_mid,
        "XYZ": bm["XYZ"],
        "AL": bm["AL"],
        "LG": bm["LG"],
    }

    # |0⟩ contribution: extra infidelity from enabling |0⟩ scattering.
    # enable_0_scattering toggles a ground-state light-shift decay term,
    # not intermediate-state population routing, so only the total infidelity
    # difference is meaningful (no XYZ/AL/LG decomposition).
    print("  Intermediate decay (no |0> scattering)...")
    sim_mid_no0 = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=sign,
        enable_intermediate_decay=True,
        enable_0_scattering=False,
    )
    infid_mid_no0 = sim_mid_no0.gate_fidelity(x,fid_type="sss")
    errors["scattering_0_extra_infidelity"] = max(0.0, infid_mid - infid_mid_no0)

    # Polarization leakage
    print("  Polarization leakage...")
    sim_pol = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=sign,
        enable_polarization_leakage=True,
    )
    infid_pol = sim_pol.gate_fidelity(x,fid_type="sss")
    budget_pol = sim_pol.error_budget(x,initial_states=SSS_12_STATES)
    errors["polarization_leakage"] = {
        "infidelity": infid_pol,
        **budget_pol["polarization_leakage"],
    }

    # All deterministic combined
    print("  All deterministic...")
    sim_all = CZGateSimulator(
        param_set="our", strategy="TO",
        blackmanflag=True, detuning_sign=sign,
        enable_rydberg_decay=True,
        enable_intermediate_decay=True,
        enable_polarization_leakage=True,
    )
    infid_all = sim_all.gate_fidelity(x,fid_type="sss")
    ba = sim_all.error_budget(x,initial_states=SSS_12_STATES)
    errors["all_deterministic"] = {
        "infidelity": infid_all,
        "XYZ": ba["rydberg_decay"]["XYZ"] + ba["intermediate_decay"]["XYZ"] + ba["polarization_leakage"]["XYZ"],
        "AL": ba["rydberg_decay"]["AL"] + ba["intermediate_decay"]["AL"] + ba["polarization_leakage"]["AL"],
        "LG": ba["rydberg_decay"]["LG"] + ba["intermediate_decay"]["LG"] + ba["polarization_leakage"]["LG"],
    }

    return errors


SIGMA_POS = (70e-9, 70e-9, 130e-9)  # meters
SIGMA_DETUNING = 130e3  # Hz


def load_mc_results(label):
    """Load MC results from data/ directory."""
    results = {}
    for key in ("dephasing", "position", "all"):
        path = f"data/mc_{label}_{key}.txt"
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} not found. Use --generate-mc to compute inline.")
        results[key] = MonteCarloResult.load_from_file(path)
    return results


def generate_mc_results(sign, label, x, n_shots=1000, seed=42):
    """Run MC simulations for dephasing, position, and all-combined."""
    Path("data").mkdir(exist_ok=True)

    scenarios = [
        ("dephasing", f"Dephasing ({SIGMA_DETUNING/1e3:.0f} kHz)", {
            "enable_rydberg_dephasing": True,
        }, {"sigma_detuning": SIGMA_DETUNING}),
        ("position", "Position error (70,70,130 nm)", {
            "enable_position_error": True,
        }, {"sigma_pos_xyz": SIGMA_POS}),
        ("all", "All errors combined", {
            "enable_rydberg_decay": True,
            "enable_intermediate_decay": True,
            "enable_polarization_leakage": True,
            "enable_rydberg_dephasing": True,
            "enable_position_error": True,
        }, {"sigma_detuning": SIGMA_DETUNING, "sigma_pos_xyz": SIGMA_POS}),
    ]

    results = {}
    for i, (key, desc, sim_kw, mc_kw) in enumerate(scenarios, 1):
        print(f"  [{i}/{len(scenarios)}] {desc}...")
        sim = CZGateSimulator(
            param_set="our", strategy="TO",
            blackmanflag=True, detuning_sign=sign, **sim_kw,
        )
        results[key] = sim.run_monte_carlo_simulation(
            x, n_shots=n_shots, seed=seed + i - 1,
            compute_branching=True, **mc_kw,
        )
        results[key].save_to_file(f"data/mc_{label}_{key}.txt")

    print(f"  Saved to data/mc_{label}_*.txt")
    return results


def build_table_rows(det, mc):
    """Build table rows as list of lists."""
    n = mc["dephasing"].n_shots

    def sem(std):
        return std / np.sqrt(n)

    def det_row(name, e):
        has_branching = e["XYZ"] is not None
        if has_branching:
            coherent = e["infidelity"] - (e["XYZ"] + e["AL"] + e["LG"])
            return [
                name,
                f"{e['infidelity']:.6e}",
                f"{e['XYZ']:.6e}",
                f"{e['AL']:.6e}",
                f"{e['LG']:.6e}",
                f"{coherent:.6e}" if abs(coherent) > 1e-15 else "0.000000e+00",
            ]
        else:
            return [
                name,
                f"{e['infidelity']:.6e}",
                "\u2014", "\u2014", "\u2014", "\u2014",
            ]

    def mc_row(name, r):
        return [
            name,
            f"{r.mean_infidelity:.6e} \u00b1 {sem(r.std_infidelity):.2e}",
            f"{r.mean_branch_XYZ:.6e}",
            f"{r.mean_branch_AL:.6e}",
            f"{r.mean_branch_LG:.6e}",
            f"{r.mean_branch_phase:.6e}",
        ]

    header = ["Error Source", "Infidelity", "XYZ", "AL", "LG", "Coh/Phase"]

    # Intermediate decay: the |0⟩ scattering adds ground-state population
    # loss that error_budget() cannot track (it only measures intermediate/
    # Rydberg populations). This inflates the coherent residual. Subtract
    # the |0⟩ extra infidelity so Coh/Phase reflects true coherent error.
    s0_extra = det["scattering_0_extra_infidelity"]
    e_mid = det["intermediate_decay"]
    mid_coherent = e_mid["infidelity"] - (e_mid["XYZ"] + e_mid["AL"] + e_mid["LG"]) - s0_extra
    mid_row = [
        "  Intermediate decay\u2020",
        f"{e_mid['infidelity']:.6e}",
        f"{e_mid['XYZ']:.6e}",
        f"{e_mid['AL']:.6e}",
        f"{e_mid['LG']:.6e}",
        f"{mid_coherent:.6e}" if abs(mid_coherent) > 1e-15 else "0.000000e+00",
    ]

    rows = [
        header,
        ["DETERMINISTIC (SSS)", "", "", "", "", ""],
        det_row("  Rydberg decay", det["rydberg_decay"]),
        mid_row,
        det_row("  Polarization leakage", det["polarization_leakage"]),
        det_row("  All deterministic", det["all_deterministic"]),
        ["", "", "", "", "", ""],
        ["STOCHASTIC (MC, avg)*", "", "", "", "", ""],
        mc_row("  Dephasing (130 kHz)", mc["dephasing"]),
        mc_row("  Position (70,70,130 nm)", mc["position"]),
        ["", "", "", "", "", ""],
        ["TOTAL (MC, avg)*", "", "", "", "", ""],
        mc_row("  All errors combined", mc["all"]),
    ]

    footnotes = [
        f"\u2020 Intermediate decay includes full |0\u27e9+|1\u27e9 scattering; "
        f"|0\u27e9 contribution to infidelity: {s0_extra:.6e} "
        f"(ground-state loss, excluded from Coh/Phase)",
        "* MC uses fid_type='average'; deterministic uses fid_type='sss'.",
    ]

    return rows, footnotes


def render_pdf(rows, title, output_path, footnotes=None):
    """Render table rows to a PDF using matplotlib."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.95)

    table = ax.table(cellText=rows, cellLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Header styling
    for col in range(6):
        cell = table[(0, col)]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(weight="bold", color="white")

    # Section header styling
    for i, row in enumerate(rows):
        if row[0] in ("DETERMINISTIC (SSS)", "STOCHASTIC (MC, avg)*", "TOTAL (MC, avg)*"):
            for col in range(6):
                cell = table[(i, col)]
                cell.set_facecolor("#D9E1F2")
                cell.set_text_props(weight="bold")

    for _, cell in table.get_celld().items():
        cell.set_edgecolor("gray")
        cell.set_linewidth(0.5)

    # Footnotes outside the table, bottom-left
    if footnotes:
        footnote_text = "\n".join(footnotes)
        fig.text(0.05, 0.02, footnote_text, fontsize=8, fontstyle="italic",
                 verticalalignment="bottom", wrap=True)

    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recompute", action="store_true",
                        help="Recompute deterministic errors instead of loading cached data")
    parser.add_argument("--generate-mc", action="store_true",
                        help="Generate MC data inline instead of loading from data/")
    parser.add_argument("--n-mc-shots", type=int, default=1000,
                        help="Number of MC shots when using --generate-mc (default: 1000)")
    parser.add_argument("--mc-seed", type=int, default=42,
                        help="RNG seed for MC generation (default: 42)")
    args = parser.parse_args()

    for sign, label, x in [
        # (-1, "bright", X_TO_OUR_BRIGHT),
        (1, "dark", X_TO_OUR_DARK),
    ]:
        print(f"\n{'='*60}")
        print(f"  {label.upper()} DETUNING")
        print(f"{'='*60}")

        det_path = f"data/det_{label}.json"
        if args.recompute or not Path(det_path).exists():
            print("\nComputing deterministic errors...")
            det = compute_deterministic_errors(sign, x)
            save_deterministic_errors(det, det_path)
        else:
            print(f"\nLoading deterministic errors from {det_path}")
            det = load_deterministic_errors(det_path)

        if args.generate_mc:
            print("\nGenerating MC results:")
            mc = generate_mc_results(sign, label, x,
                                     n_shots=args.n_mc_shots, seed=args.mc_seed)
        else:
            print("\nLoading MC results:")
            mc = load_mc_results(label)

        rows, footnotes = build_table_rows(det, mc)
        render_pdf(rows, f"Error Budget: {label.capitalize()} Detuning", f"scripts/SI_Tables_{label}.pdf", footnotes=footnotes)

    print("\nDone.")


if __name__ == "__main__":
    main()
