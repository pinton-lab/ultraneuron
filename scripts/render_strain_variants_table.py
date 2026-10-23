"""LaTeX table for the strain-measure model-form check (JNE revision, R2-1).

Reads data/halle_dacc/strain_variants/summary.json and writes
data/strain_variants_table.tex (input by the
manuscript) so the numbers in the paper are generated, not transcribed.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
summary = json.loads((REPO / "data/halle_dacc/strain_variants/summary.json").read_text())
runs, diag = summary["runs"], summary["diagnostics"]
out = REPO / "data/strain_variants_table.tex"

label = {
    "vm": r"von Mises $\varepsilon_{\mathrm{eq}}$ (default)",
    "e1d": r"max.\ principal $\varepsilon_1$",
    "eAmaxd": r"max.\ areal $\varepsilon_A^{\max}$",
    "eAavgd": r"orient.-avg.\ areal $\langle\varepsilon_A^{+}\rangle$",
    "e1": r"max.\ principal $\varepsilon_1$",
    "eAmax": r"max.\ areal $\varepsilon_A^{\max}$",
    "eAavg": r"orient.-avg.\ areal $\langle\varepsilon_A^{+}\rangle$",
}
ref = runs["vm_raw"]


def row(key, meas, mode):
    h = runs[key]
    ratio = diag.get(f"{meas}_over_vm_focal_median", 1.0)
    dice = h.get("dice_vs_control", 1.0)
    dv = 100.0 * (h["iso25_volume_mm3"] / ref["iso25_volume_mm3"] - 1.0)
    ds = 100.0 * (h["total_spikes"] / ref["total_spikes"] - 1.0)
    dice_deep = h.get("dice_vs_control_excl_shell", dice)
    shell = h.get("n_firing_in_skull_shell", 0)
    return (f"{label[meas]} & {mode} & {ratio:.2f} & {h['peak_tension_mNm']:.2f} & "
            f"{h['n_firing_voxels']:,} ({shell:,}) & {h['total_spikes']:,} & "
            f"{h['iso25_volume_mm3']/1e3:.1f} ({dv:+.0f}\\%) & {ds:+.0f}\\% & {dice:.2f} / {dice_deep:.2f} \\\\")


lines = [
    r"\begin{table}[tbp]",
    r"\begin{revblock}[R2]",
    r"\caption{\revtag{R2-1} Model-form sensitivity of the canonical Halle\,/\,dACC firing map (S2, $\bar g_{\mathrm{Piezo1}} = 0.10$\,mS/cm$^2$, $K_A\alpha = 250$\,N/m) to the scalar strain measure that drives the membrane tension in Eq.~\ref{eq:tension}. All four measures are computed from the same end-of-pulse displacement field on the 1.23\,mm analysis grid; the field is solenoidal to $\mathrm{div}\,\mathbf u/\varepsilon_{\mathrm{eq}} = 7\times10^{-4}$ (\S\ref{sec:methods_upstream}), and forming the measures from the deviatoric part of the tensor instead changes every entry by less than 0.2\,\%. ``canonical'' keeps the canonical $K_A\alpha$; ``matched'' rescales the measure so that its focal peak equals the von Mises focal peak, isolating the difference in the \emph{shape} of the map from the overall scale that a calibration of $K_A\alpha$ would absorb. Ratio, median of measure\,/\,$\varepsilon_{\mathrm{eq}}$ over the focal region ($\varepsilon_{\mathrm{eq}} > 0.25\,\varepsilon_{\mathrm{eq}}^{\max}$); $T_{\max}$, focal peak membrane tension; firing voxels, total (of which within two voxels of the inner skull surface, in parentheses; zero throughout); iso-25\,\%, volume with firing rate $\ge 25\,\%$ of the peak (per cent change relative to the default map in parentheses); $\Delta$spikes, change in total spike count; Dice, overlap coefficient of the set of firing voxels with the default (von Mises) map ($2|A\cap B|/(|A|+|B|)$, 1 = identical), for all voxels / excluding the two-voxel skull shell. \rev[E]{Abbreviations: dACC, dorsal anterior cingulate cortex; S1--S5, mechanism scenarios of Table~\ref{tab:scenarios}.}}",
    r"\label{tab:strain_variants}",
    r"\scriptsize",
    r"\setlength{\tabcolsep}{2.5pt}",
    r"\begin{tabular}{@{}llcccccrc@{}}",
    r"\toprule",
    r"Strain measure & $K_A\alpha$ & Ratio & $T_{\max}$ & Firing voxels & Spikes & Iso-25\,\% & $\Delta$spikes & Dice \\",
    r"& & & (mN/m) & (in shell) & & ($10^3$\,mm$^3$) & & all\,/\,deep \\",
    r"\midrule",
    row("vm_raw", "vm", "--"),
]
for meas in ("e1", "eAmax", "eAavg"):
    lines.append(row(f"{meas}_raw", meas, "canonical"))
    lines.append(row(f"{meas}_matched", meas, "matched"))
lines += [r"\bottomrule", r"\end{tabular}", r"\end{revblock}", r"\end{table}"]
out.write_text("\n".join(lines) + "\n")
print("wrote", out)
print("\n".join(lines[10:-4]))
