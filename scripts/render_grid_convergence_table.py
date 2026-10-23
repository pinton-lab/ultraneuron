"""LaTeX table for the shear/neural grid-convergence check (JNE revision, R2-6).
Reads data/halle_dacc/grid_convergence/grid_convergence.json and writes
data/grid_convergence_table.tex (+ macros file).
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
d = json.loads((REPO / "data/halle_dacc/grid_convergence/grid_convergence.json").read_text())
for v in d.values():                                            # older entries lack the intensive measure
    v.setdefault("mean_spikes_per_firing_voxel", v["total_spikes"] / max(v["n_firing_voxels"], 1))
rows = sorted(d.values(), key=lambda r: -r["dx_mm"])          # coarse -> fine
ref = d.get("factor_1")


def pct(a, b):
    return f"{100 * (a / b - 1):+.0f}" if b else "--"


lines = [r"\begin{table}[tbp]", r"\begin{revblock}[R2]",
         r"\caption{\revtag{R2-6} Grid-convergence check of the elastic and neural stages on the canonical Halle\,/\,dACC field. The radiation-force field of the canonical run (1.232\,mm analysis grid) is sub-sampled to a 2$\times$ coarser grid and linearly interpolated to a 2$\times$ finer grid (original samples preserved); the shear solver is run to the end of the 20\,ms ON window on each grid with identical physics, a CFL-scaled time step and the per-step velocity damping rescaled to keep its e-folding time fixed, the equivalent strain is formed with the same central differences, and the canonical S2 neuron ($\bar g_{\mathrm{Piezo1}} = 0.10$\,mS/cm$^2$, $K_A\alpha = 250$\,N/m per unit strain) is run on each strain field. Percentages are relative to the 1.232\,mm reference row; the total spike count scales with the number of voxels and is therefore reported as the mean per firing voxel. The 1.232\,mm row reproduces the canonical run (Table~\ref{tab:yaakub_summary}) with the GPU shear solver to within 0.1\,\%. The acoustic field itself is resolved at 0.308\,mm (10 points per wavelength) and is decimated, not interpolated, to the analysis grid; that decimation is discussed in the text. \rev[E]{Abbreviations: dACC, dorsal anterior cingulate cortex; ON/OFF, sonication on/off windows of the pulsed protocol; GPU, graphics processing unit; CFL, Courant--Friedrichs--Lewy number; S1--S5, mechanism scenarios of Table~\ref{tab:scenarios}.}}",
         r"\label{tab:grid_convergence}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
         r"Spacing & Grid & $\Delta t_{\mathrm{shear}}$ & Peak $\varepsilon_{\mathrm{eq}}$ & $T_{\max}$ & Firing & Firing vol. & Iso-25\,\% & Spikes per \\", r"(mm) & & ($\mu$s) & & (mN/m) & voxels & ($10^3$\,mm$^3$) & ($10^3$\,mm$^3$) & firing voxel \\",
         r"\midrule"]
for r in rows:
    g = "$\\times$".join(str(x) for x in r["grid"])
    g = g.replace("$\\times$", "{\\times}"); g = "$" + g + "$"
    lines.append(f"{r['dx_mm']:.3f} & {g} & {r['shear_dt_us']:.1f} & ${r['peak_eps']*1e5:.2f}{{\\times}}10^{{-5}}$ ({pct(r['peak_eps'], ref['peak_eps'])}\\%) & {r['peak_tension_mNm']:.2f} & {r['n_firing_voxels']:,} & {r['firing_volume_mm3']/1e3:.1f} ({pct(r['firing_volume_mm3'], ref['firing_volume_mm3'])}\\%) & {r['iso25_volume_mm3']/1e3:.1f} ({pct(r['iso25_volume_mm3'], ref['iso25_volume_mm3'])}\\%) & {r['mean_spikes_per_firing_voxel']:.2f} ({pct(r['mean_spikes_per_firing_voxel'], ref['mean_spikes_per_firing_voxel'])}\\%) \\\\".replace(",", "\\,").replace("\\\\,", "\\,"))
lines += [r"\bottomrule", r"\end{tabular}", r"\end{revblock}", r"\end{table}"]
out = REPO / "data/grid_convergence_table.tex"
out.write_text("\n".join(lines) + "\n"); print("wrote", out)
mac = {}
if "factor_2" in d and ref:
    f = d["factor_2"]
    mac["GCfineEps"] = pct(f["peak_eps"], ref["peak_eps"]); mac["GCfineVol"] = pct(f["firing_volume_mm3"], ref["firing_volume_mm3"])
    mac["GCfineIso"] = pct(f["iso25_volume_mm3"], ref["iso25_volume_mm3"]); mac["GCfineSpk"] = pct(f["mean_spikes_per_firing_voxel"], ref["mean_spikes_per_firing_voxel"])
if "factor_0.5" in d and ref:
    c = d["factor_0.5"]
    mac["GCcoarseEps"] = pct(c["peak_eps"], ref["peak_eps"]); mac["GCcoarseVol"] = pct(c["firing_volume_mm3"], ref["firing_volume_mm3"])
    mac["GCcoarseIso"] = pct(c["iso25_volume_mm3"], ref["iso25_volume_mm3"]); mac["GCcoarseSpk"] = pct(c["mean_spikes_per_firing_voxel"], ref["mean_spikes_per_firing_voxel"])
(REPO / "data/grid_convergence_numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")
print(mac)
