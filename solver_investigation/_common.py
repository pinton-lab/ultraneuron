"""Shared loader for the solver investigation.

Reproduces the run_neural_sim.py front-end (strain -> tension, threshold,
S2 neuron build) and caches the eligible tension vector so integrator A/B
tests do not re-run the FDTD front-end each time.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from ultraneuron import (  # noqa: E402
    PIEZO1,
    TREK1,
    MulticompartmentMechanoHHNeuron,
)
from ultraneuron.channels import strain_to_membrane_tension  # noqa: E402

# Canonical run_neural_sim.py defaults (S2, halle_dacc).
K_A = 0.25
COUPLING_FACTOR = 1000.0
TENSION_THRESHOLD = 0.5
PIEZO1_G = 0.10
TREK1_G = 0.04

SIM_DIR = REPO / "data" / "halle_dacc"
CACHE = HERE / "eligible_tensions.npy"


def _strain_from_disp(disp, dx, dy, dz):
    ux, uy, uz = disp[..., 0], disp[..., 1], disp[..., 2]
    duxdx, duxdy, duxdz = np.gradient(ux, dx, dy, dz, edge_order=1)
    duydx, duydy, duydz = np.gradient(uy, dx, dy, dz, edge_order=1)
    duzdx, duzdy, duzdz = np.gradient(uz, dx, dy, dz, edge_order=1)
    e_xx, e_yy, e_zz = duxdx, duydy, duzdz
    e_xy = 0.5 * (duxdy + duydx)
    e_xz = 0.5 * (duxdz + duzdx)
    e_yz = 0.5 * (duydz + duzdy)
    em = (e_xx + e_yy + e_zz) / 3.0
    dxx, dyy, dzz = e_xx - em, e_yy - em, e_zz - em
    e2 = dxx * dxx + dyy * dyy + dzz * dzz + 2.0 * (e_xy ** 2 + e_xz ** 2 + e_yz ** 2)
    return np.sqrt((2.0 / 3.0) * e2).astype(np.float32)


def build_neuron(scenario="S2"):
    p1 = deepcopy(PIEZO1)
    p1.g_density = PIEZO1_G
    t1 = deepcopy(TREK1)
    t1.g_density = TREK1_G
    return MulticompartmentMechanoHHNeuron.cortical_pyramidal_3comp(
        mechano_channels=[p1, t1], temperature=37.0,
    )


def eligible_tensions():
    """Return the (N,) above-threshold tension vector, cached on disk."""
    if CACHE.exists():
        return np.load(CACHE)
    disp = np.load(SIM_DIR / "displacement.npy").astype(np.float32)
    c_map = np.load(SIM_DIR / "c_ds.npy").astype(np.float32)
    grid = np.load(SIM_DIR / "grid_params.npz")
    dx = float(grid["dx"]) * int(grid["mod_x"])
    dy = float(grid["dy"]) * int(grid["mod_y"])
    dz = float(grid["dz"]) * int(grid["mod_z"])
    strain = _strain_from_disp(disp, dx, dy, dz)
    skull = c_map > 1600.0
    strain[skull] = 0.0
    tension_3d = strain_to_membrane_tension(strain, K_A, COUPLING_FACTOR)
    mask = tension_3d >= TENSION_THRESHOLD
    T = np.asarray(tension_3d[mask], dtype=np.float64)
    np.save(CACHE, T)
    return T


if __name__ == "__main__":
    T = eligible_tensions()
    print(f"eligible voxels: {T.size}")
    print(f"tension range: {T.min():.3f} .. {T.max():.3f} mN/m")
