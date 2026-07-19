"""Per-case Heimburg-Jackson soliton excitability scoring.

Realises the soliton mechanism's downstream scalar contract
(:func:`ultraneuron.soliton_excitation`) on the canonical
run(s). For each case it reads the two per-case quantities the upstream
solvers already produce — the rescaled focal pressure amplitude
(``rescale_provenance.json``) and the Pennes focal temperature rise
(``bioheat/thermal_metrics.json``) — and scores

    ξ = SolitonModel().excitability(P_focal_pa, 37.0 + dT_focus_K).

To expose the model's falsifiable sign it also reports the isothermal
score (ξ at ΔT = 0) and the +1 K counterfactual, so the
heating-reduces-excitability direction is legible per case.

Select the case with ``HALLE_SIM_NAME`` (default ``halle_dacc``); pass
multiple comma-separated names to score several at once.

Output:
  data/soliton_score{_suffix}.npz   — per-case scores + operating point
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from ultraneuron import SolitonModel, soliton_excitation  # noqa: E402

BASELINE_T_C = 37.0


def _read_case(sim_name: str) -> dict:
    """Read focal pressure [Pa] and focal ΔT [K] for one canonical case."""
    sim_dir = REPO_ROOT / "data" / sim_name
    resc = json.loads((sim_dir / "rescale_provenance.json").read_text())
    # Prefer the post-rescale in-brain focal peak; fall back to the target.
    p_focal_pa = float(
        resc.get("rescaled_in_brain_peak_Pa")
        or resc.get("target_in_brain_peak_Pa")
    )
    thermal = json.loads((sim_dir / "bioheat" / "thermal_metrics.json").read_text())
    dT_focus_K = float(thermal["dT_focus_K"])
    return {"sim_name": sim_name, "P_focal_pa": p_focal_pa, "dT_focus_K": dT_focus_K}


def score_case(case: dict, model: SolitonModel | None = None) -> dict:
    """Score the soliton excitability for one case, with the sign probes."""
    model = model or SolitonModel()
    P = case["P_focal_pa"]
    dT = case["dT_focus_K"]
    xi = soliton_excitation(P, dT, baseline_T_C=BASELINE_T_C, model=model)
    xi_isothermal = soliton_excitation(P, 0.0, baseline_T_C=BASELINE_T_C, model=model)
    xi_plus1K = soliton_excitation(P, dT + 1.0, baseline_T_C=BASELINE_T_C, model=model)
    return {
        **case,
        "T_focus_C": BASELINE_T_C + dT,
        "excitability": xi,
        "excitability_isothermal": xi_isothermal,
        "excitability_plus1K": xi_plus1K,
        # Negative ⇒ heating reduces excitability (the falsifiable sign).
        "d_xi_per_K": xi_plus1K - xi,
    }


def main() -> None:
    names = [s.strip() for s in
             os.environ.get("HALLE_SIM_NAME", "halle_dacc").split(",") if s.strip()]
    model = SolitonModel()
    print(f"=== Soliton per-case scoring (T_m={model.T_m:.1f} °C, "
          f"σ_T={model.sigma_T:.1f} °C, c_p_max={model.c_p_max:.0f}) ===",
          flush=True)

    rows = []
    for name in names:
        try:
            case = _read_case(name)
        except FileNotFoundError as exc:
            print(f"  [skip] {name}: {exc}", flush=True)
            continue
        r = score_case(case, model)
        rows.append(r)
        print(f"  {name}: P_focal={r['P_focal_pa']/1e6:.3f} MPa, "
              f"ΔT_focus={r['dT_focus_K']*1e3:.0f} mK (T={r['T_focus_C']:.2f} °C) "
              f"→ ξ={r['excitability']:.3e}", flush=True)
        print(f"      isothermal ξ₀={r['excitability_isothermal']:.3e}; "
              f"+1 K ⇒ ξ={r['excitability_plus1K']:.3e} "
              f"(dξ/dK={r['d_xi_per_K']:+.2e}, "
              f"{'heating REDUCES' if r['d_xi_per_K'] < 0 else 'heating raises'} ξ)",
              flush=True)

    if not rows:
        raise SystemExit("No cases scored (missing rescale_provenance/thermal_metrics?).")

    suffix = "" if names == ["halle_dacc"] else "_" + "_".join(
        n.replace("halle_", "") for n in names)
    out = REPO_ROOT / "data" / f"soliton_score{suffix}.npz"
    np.savez_compressed(
        out,
        baseline_T_C=BASELINE_T_C,
        T_m=model.T_m, sigma_T=model.sigma_T, c_p_max=model.c_p_max,
        sim_names=np.array([r["sim_name"] for r in rows]),
        P_focal_pa=np.array([r["P_focal_pa"] for r in rows]),
        dT_focus_K=np.array([r["dT_focus_K"] for r in rows]),
        excitability=np.array([r["excitability"] for r in rows]),
        excitability_isothermal=np.array([r["excitability_isothermal"] for r in rows]),
        excitability_plus1K=np.array([r["excitability_plus1K"] for r in rows]),
        d_xi_per_K=np.array([r["d_xi_per_K"] for r in rows]),
    )
    print(f"Saved: {out}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
