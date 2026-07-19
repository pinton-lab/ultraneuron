"""Free-field surface-pressure calibration for the CTX-500 / Yaakub 2023 run.

Drives the angular-spectrum solver with a unit-amplitude (p0 = 1 Pa) CTX-500
bowl in homogeneous water, finds the simulated free-field focal peak, and
linearly inverts to the surface drive that reproduces the paper's reported
free-field focal pressure:

    p0_surface = p_target / cal_focal_peak

This is the *free-field* surface drive. A transcranial run incurs ~50%
skull-transmission loss on top of this; whether to apply that derating to the
canonical halle_bowl / halle_caudate p0_pa is a separate decision (see the
README of this PR / the discussion in chat).

Reference implementation: run_freefield_sweep.py in the neuromod_parameters
repository (sister-repo human sweep).

CTX-500 / NeuroFUS spec (Yaakub SN et al., Brain Stimulation 2023):
  ROC          = 63 mm
  outer radius = 32 mm   (aperture diameter 64 mm)
  inner radius = 0
  n_elements   = 4       (annular)
  f0           = 500 kHz
  reported free-field focal peak = 1.01 MPa
"""
from __future__ import annotations

import os
import sys

import numpy as np

from ultraneuron.paths import angular_spectrum_dir

ASM_DIR = str(angular_spectrum_dir())
sys.path.insert(0, ASM_DIR)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from angular_spectrum_solver import (  # noqa: E402
    SolverParams,
    angular_spectrum_solve,
    make_bowl_source_planes,
)


# --- CTX-500 / Yaakub 2023 geometry ---------------------------------------
ROC_M = 63e-3
OUTER_RADIUS_M = 32e-3
INNER_RADIUS_M = 0.0
N_ELEMENTS = 4
F0_HZ = 500e3
C0 = 1540.0
RHO0 = 1000.0

P_FREEFIELD_TARGET_MPA = 1.01  # Yaakub SN et al., Brain Stim. 2023

# Sanity-check window for the calibrated surface drive
P0_SANITY_LO_KPA = 20.0
P0_SANITY_HI_KPA = 200.0


def _focal_fwhm(pI, xaxis, yaxis, zaxis, foc_idx, nX, nY):
    """Axial / lateral FWHM (mm) of the intensity beam at the focus."""
    on_axis = pI[nX // 2, nY // 2, :]
    peak = float(on_axis[foc_idx])
    if peak <= 0:
        return 0.0, 0.0
    half = peak / 2
    z = zaxis
    pl, pr = on_axis[:foc_idx], on_axis[foc_idx:]
    z_lo = z[:foc_idx][np.argmax(pl >= half)] if np.any(pl >= half) else z[0]
    below_right = pr < half
    z_hi = z[foc_idx:][np.argmax(below_right)] if np.any(below_right) else z[-1]
    fwhm_axial_mm = float((z_hi - z_lo) * 1e3)

    lat = pI[:, nY // 2, foc_idx]
    if lat.max() <= 0:
        return fwhm_axial_mm, 0.0
    above = np.where(lat >= lat.max() / 2)[0]
    fwhm_lateral_mm = (
        float((xaxis[above[-1]] - xaxis[above[0]]) * 1e3)
        if len(above) >= 2 else 0.0
    )
    return fwhm_axial_mm, fwhm_lateral_mm


def calibrate():
    aperture_m = 2 * OUTER_RADIUS_M
    focal_m = ROC_M

    # Domain: lateral 1.7 × aperture; depth 1.6 × focal + 25 mm post-focal margin
    domain_lat = max(60e-3, 1.7 * aperture_m)
    domain_depth = 1.6 * focal_m + 25e-3

    # Grid: lambda/5 first, drop to lambda/3 then lambda/2 if it gets too big.
    lam = C0 / F0_HZ
    NX_MAX = 513
    for ppwl in (5.0, 3.0, 2.0):
        dX = lam / ppwl
        nX = int(round(domain_lat / dX))
        if nX % 2 == 0:
            nX += 1
        if nX <= NX_MAX:
            break
    if nX > NX_MAX:
        nX = NX_MAX
        dX = domain_lat / (nX - 1)
    dT = dX / (5 * C0)
    nY = nX
    nT = max(401, int(np.ceil(8 / (F0_HZ * dT))))
    nT = nT if nT % 2 else nT + 1
    xaxis = (np.arange(nX) - (nX - 1) / 2) * dX
    yaxis = (np.arange(nY) - (nY - 1) / 2) * dX
    taxis = (np.arange(nT) - nT / 2) * dT
    dZ = dX

    print(
        f"Grid: {nX}x{nY}x{nT}  dX={dX*1e3:.3f} mm  dT={dT*1e9:.1f} ns  "
        f"lat={domain_lat*1e3:.0f} mm  depth={domain_depth*1e3:.0f} mm  "
        f"ppwl={lam/dX:.2f}"
    )

    # Unit-amplitude (1 Pa) CTX-500 bowl
    src_planes_unit, bowl_depth = make_bowl_source_planes(
        xaxis, yaxis, taxis, F0_HZ, C0, p0=1.0,
        radius=OUTER_RADIUS_M, roc=ROC_M, dZ=dZ,
        focus=ROC_M, inner_radius=INNER_RADIUS_M,
        n_elements=N_ELEMENTS,
    )
    apa_unit = src_planes_unit[0][1]

    params = SolverParams(
        dX=dX, dY=dX, dT=dT, c0=C0, rho0=RHO0, beta=0.0,
        alpha0=0.5, attenPow=1, f0=F0_HZ, propDist=domain_depth,
        boundaryFactor=0.15, useSplitStep=True,
        useAdaptiveFiltering=True, useTVD=False,
        boundaryProfile="wendland",
        useFreqWeightedBoundary=True,
        useSuperAbsorbing=True, superAbsorbingStrength=0.5,
        dZmin=dZ,
        phaseScreens=None,
        sourcePlanes=src_planes_unit[1:],
        diagnostic=False,
        preflight=False,
    )

    _, pnp_u, ppp_u, pI_u, _, zaxis, _ = angular_spectrum_solve(
        apa_unit, params, verbose=False,
    )

    # Find free-field focal peak in the far field (z > 0.4*ROC)
    far_start = int(np.argmax(zaxis > 0.4 * ROC_M))
    pnp_max_u = np.maximum(pnp_u, ppp_u)
    flat = int(np.argmax(pnp_max_u[:, :, far_start:]))
    _, _, iz = np.unravel_index(flat, pnp_max_u[:, :, far_start:].shape)
    iz_global = iz + far_start
    cal_focal_peak = float(pnp_max_u[:, :, far_start:].max())
    cal_focal_z_mm = float(zaxis[iz_global] * 1e3)

    p_target_Pa = P_FREEFIELD_TARGET_MPA * 1e6
    p0_surface = p_target_Pa / cal_focal_peak

    # Linear: scale unit-source intensity / pressure rather than re-running
    pI = pI_u * (p0_surface ** 2)
    pnp = pnp_max_u * p0_surface

    on_axis_pI = pI[nX // 2, nY // 2, :]
    foc_idx = int(far_start + np.argmax(on_axis_pI[far_start:]))
    foc_z_mm = float(zaxis[foc_idx] * 1e3)
    sim_focal_peak_MPa = float(pnp.max() * 1e-6)
    fwhm_axial_mm, fwhm_lateral_mm = _focal_fwhm(
        pI, xaxis, yaxis, zaxis, foc_idx, nX, nY,
    )

    print()
    print("=== CTX-500 free-field calibration ===")
    print(f"  Target free-field focal peak : {P_FREEFIELD_TARGET_MPA:.3f} MPa "
          f"(Yaakub SN et al., Brain Stim. 2023)")
    print(f"  Bowl ROC / outer radius      : {ROC_M*1e3:.1f} / "
          f"{OUTER_RADIUS_M*1e3:.1f} mm  ({N_ELEMENTS}-element annular)")
    print(f"  Unit-source focal peak (p0=1): {cal_focal_peak:.4e} Pa "
          f"at z = {cal_focal_z_mm:.1f} mm")
    print(f"  Free-field focal gain        : {cal_focal_peak:.2f}× "
          f"(unit-source peak per 1 Pa drive)")
    print()
    print(f"  >>> p0_surface = {p0_surface:.2f} Pa "
          f"= {p0_surface*1e-3:.2f} kPa "
          f"= {p0_surface*1e-6:.4f} MPa <<<")
    print()
    print(f"  Sanity (re-scaled at calibrated p0):")
    print(f"    focal peak     : {sim_focal_peak_MPa:.3f} MPa "
          f"(should match target {P_FREEFIELD_TARGET_MPA:.3f} MPa)")
    print(f"    focal depth    : {foc_z_mm:.1f} mm (geom. focus = "
          f"{ROC_M*1e3:.1f} mm)")
    print(f"    FWHM axial     : {fwhm_axial_mm:.1f} mm")
    print(f"    FWHM lateral   : {fwhm_lateral_mm:.1f} mm")

    p0_kPa = p0_surface * 1e-3
    if not (P0_SANITY_LO_KPA <= p0_kPa <= P0_SANITY_HI_KPA):
        print()
        print(
            f"  WARNING: calibrated p0_surface = {p0_kPa:.1f} kPa is OUTSIDE "
            f"the sanity window [{P0_SANITY_LO_KPA:.0f}, "
            f"{P0_SANITY_HI_KPA:.0f}] kPa. "
            f"Investigate before applying to configs."
        )

    return {
        "p0_surface_Pa": float(p0_surface),
        "p0_surface_MPa": float(p0_surface * 1e-6),
        "cal_focal_peak_per_unit_drive_Pa": cal_focal_peak,
        "sim_focal_peak_MPa": sim_focal_peak_MPa,
        "focal_depth_mm": foc_z_mm,
        "fwhm_axial_mm": fwhm_axial_mm,
        "fwhm_lateral_mm": fwhm_lateral_mm,
    }


if __name__ == "__main__":
    calibrate()
