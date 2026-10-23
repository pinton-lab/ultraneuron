"""R2-7: is the post-hoc linear rescale of the Fullwave field legitimate?

Compares the canonical propagation map (run at p0 = 0.10 MPa, then multiplied by
s = 1.0868 post hoc) with a full nonlinear Fullwave re-run at source amplitude
s * p0 (data/halle_dacc_p0check/).  Reports, over brain voxels:
  * peak |p| of each field and the ratio,
  * relative L2 and max-abs differences of the per-voxel peak-pressure maps,
  * the same restricted to the -6 dB focal region,
  * harmonic content at the focal voxel (2nd/1st and 3rd/1st amplitude ratios) from the
    time trace of the re-run and of the rescaled field, and the shock-formation
    distance for reference.
Writes data/halle_dacc/p0check_comparison.json and docs/.../data/p0check_numbers.tex.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
can_dir, chk_dir = REPO / "data/halle_dacc", REPO / "data/halle_dacc_p0check"
prov = json.loads((can_dir / "rescale_provenance.json").read_text()); s_fac = float(prov["scale_factor_applied"])
c = np.load(can_dir / "c_ds.npy"); brain = c <= 1600
grid = np.load(can_dir / "grid_params.npz"); dt = float(grid["dx"]) / 1540.0 * 0.2 * int(grid["sampling_modulus_time"])  # CFL 0.2 -> dt_ac = cfl*dx/c0; times sampling modulus
# safer: dt from nt & duration in manifest
man = json.loads((can_dir / "validation_report/manifest.json").read_text())
dt_ac = float(man["grid"]["dt_s"]); dt = dt_ac * int(grid["sampling_modulus_time"])
f0 = float(grid["f0"])


def peak_map(path):
    pm = np.load(path, mmap_mode="r"); pk = np.zeros(pm.shape[1:], np.float32)
    for i in range(0, pm.shape[0], 40):
        pk = np.maximum(pk, np.abs(np.asarray(pm[i:i + 40])).max(axis=0))
    return pk, pm


pk_can, pm_can = peak_map(can_dir / "propagation_map.npy")
pk_chk, pm_chk = peak_map(chk_dir / "propagation_map.npy")
assert pk_can.shape == pk_chk.shape, (pk_can.shape, pk_chk.shape)
foc = np.unravel_index(int(np.argmax(np.where(brain, pk_can, 0))), pk_can.shape)
foc_chk = np.unravel_index(int(np.argmax(np.where(brain, pk_chk, 0))), pk_chk.shape)
m6 = brain & (pk_can >= 0.5 * pk_can[brain].max())
d = pk_chk - pk_can
out = {
    "scale_factor": s_fac, "source_p0_canonical_MPa": 0.10, "source_p0_rerun_MPa": 0.10 * s_fac,
    "peak_p_brain_rescaled_MPa": float(pk_can[brain].max() / 1e6), "peak_p_brain_rerun_MPa": float(pk_chk[brain].max() / 1e6),
    "peak_ratio_rerun_over_rescaled": float(pk_chk[brain].max() / pk_can[brain].max()),
    "focal_voxel_rescaled": [int(x) for x in foc], "focal_voxel_rerun": [int(x) for x in foc_chk],
    "rel_L2_diff_brain": float(np.linalg.norm(d[brain]) / np.linalg.norm(pk_can[brain])),
    "rel_L2_diff_focal_m6dB": float(np.linalg.norm(d[m6]) / np.linalg.norm(pk_can[m6])),
    "max_abs_rel_diff_focal_m6dB": float(np.abs(d[m6]).max() / pk_can[brain].max()),
    "median_ratio_focal_m6dB": float(np.median(pk_chk[m6] / pk_can[m6])),
    "n_focal_voxels_m6dB": int(m6.sum()),
}
# harmonic content at the focal voxel
def harmonics(pm, idx):
    tr = np.asarray(pm[:, idx[0], idx[1], idx[2]], dtype=np.float64); tr -= tr.mean()
    n = tr.size; w = np.hanning(n); F = np.abs(np.fft.rfft(tr * w)); fr = np.fft.rfftfreq(n, dt)
    def amp(fh):
        sel = (fr > 0.8 * fh) & (fr < 1.2 * fh); return float(F[sel].max()) if sel.any() else float("nan")
    a1, a2, a3 = amp(f0), amp(2 * f0), amp(3 * f0)
    return dict(f_nyquist_MHz=fr[-1] / 1e6, second_over_first=a2 / a1, third_over_first=a3 / a1)
out["harmonics_rerun_focal"] = harmonics(pm_chk, foc_chk)
out["harmonics_rescaled_focal"] = harmonics(pm_can, foc)
# shock distance at the focus (plane-wave estimate): x_sh = rho c^3 / (beta omega p)
beta = float(man["medium"]["beta_max"]); rho, c0 = 1000.0, 1540.0; p = pk_chk[brain].max()
out["shock_distance_m_at_focal_p"] = rho * c0 ** 3 / (beta * 2 * np.pi * f0 * p)
out["focal_axial_fwhm_mm"] = float(man["acoustic_outputs"]["transcranial"]["fwhm_x_mm"])
out["sigma_goldberg_like_over_fwhm"] = out["focal_axial_fwhm_mm"] * 1e-3 / out["shock_distance_m_at_focal_p"]
(can_dir / "p0check_comparison.json").write_text(json.dumps(out, indent=1, default=float))
print(json.dumps(out, indent=1, default=float))
mac = {"PCpeakRatio": f"{out['peak_ratio_rerun_over_rescaled']:.4f}", "PCrelLtwoBrain": f"{100*out['rel_L2_diff_brain']:.2f}",
       "PCrelLtwoFocal": f"{100*out['rel_L2_diff_focal_m6dB']:.2f}", "PCmaxAbsFocal": f"{100*out['max_abs_rel_diff_focal_m6dB']:.2f}",
       "PCsecondHarm": f"{100*out['harmonics_rerun_focal']['second_over_first']:.1f}", "PCthirdHarm": f"{100*out['harmonics_rerun_focal']['third_over_first']:.2f}",
       "PCshockDist": f"{out['shock_distance_m_at_focal_p']*100:.0f}", "PCpeakRerun": f"{out['peak_p_brain_rerun_MPa']:.3f}", "PCpeakRescaled": f"{out['peak_p_brain_rescaled_MPa']:.3f}"}
(REPO / "data/p0check_numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")
print(mac)
