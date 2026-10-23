"""Write data/sensitivity_numbers.tex (manuscript macros) from data/sensitivity_results.npz.

Run after ``scripts/run_sensitivity.py --canonical``, then
``docs/jne_submission_verhagen/tools/fix_minus.py`` (typesets the minus signs).

    python scripts/render_sensitivity_numbers.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
d = np.load(REPO / "data" / "sensitivity_results.npz")
names = [str(n) for n in d["names"]]

# macro stem -> substring identifying the S5 row
ROWS = {"Thalf": "Piezo1 $T_{1/2}$", "KAa": "K_A\\alpha", "Dend": "Dendrite Piezo1",
        "SKg": "SK $\\bar{g}$", "SKca": "SK $[\\mathrm{Ca}", "AIS": "AIS Na"}


def row(key: str) -> int:
    hits = [i for i, n in enumerate(names) if ROWS[key] in n]
    assert len(hits) == 1, (key, hits)
    return hits[0]


def thou(n: int) -> str:
    return f"{int(n):,}".replace(",", "\\,")


mac = {"SensBase": thou(int(d["baseline"]))}
for k in ROWS:
    i = row(k)
    mac[f"Sens{k}Lo"] = f"{float(d['delta_low_pct'][i]):+.0f}"
    mac[f"Sens{k}Hi"] = f"{float(d['delta_high_pct'][i]):+.0f}"
mac["SensMsixBase"] = str(int(d["m6_baseline"]))
m6 = np.concatenate([d["m6_delta_low_pct"], d["m6_delta_high_pct"]])
mac["SensMsixMaxAbs"] = f"{float(np.max(np.abs(m6))):.0f}"

out = REPO / "data" / "sensitivity_numbers.tex"
order = ["SensBase", "SensThalfLo", "SensThalfHi", "SensKAaLo", "SensKAaHi", "SensDendLo",
         "SensDendHi", "SensSKgLo", "SensSKgHi", "SensSKcaLo", "SensSKcaHi", "SensAISLo",
         "SensAISHi", "SensMsixBase", "SensMsixMaxAbs"]
out.write_text("\n".join(f"\\newcommand{{\\{k}}}{{{mac[k]}}}" for k in order) + "\n")
print(out.read_text())
