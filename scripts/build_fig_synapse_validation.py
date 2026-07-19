"""M6 mechanosensitive synaptic / voltage-gated transmission validation figure.

Three literature-grounded gain factors, each computed directly from the
synapse primitives (reproducible; the same scenarios are asserted in
tests/test_synapse.py):

  (a) Presynaptic release modulation (Tyler 2008/2012): Pr(T) rises from
      the resting Pr_baseline and saturates at Pr_max.
  (b) Post-synaptic NMDA stretch enhancement (Maneshi 2017): f_stretch(T)
      = 1 at zero tension, saturating to 1 + alpha_NMDA (current doubling).
  (c) Voltage-gated Na+ activation shift (Kubanek 2016): nav_current_gain(T)
      = 1 at zero tension, monotone increasing.

Each factor is exactly 1 / baseline at zero tension (the no-op control:
the pathway reduces to the unmodified neuron with no ultrasound), and
monotone non-decreasing in tension.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
import sys

sys.path.insert(0, str(REPO_ROOT))

from ultraneuron.neuron import nav_current_gain  # noqa: E402
from ultraneuron.nmda import NMDAReceptor  # noqa: E402
from ultraneuron.synapse import MechanoSynapse  # noqa: E402

OUT_DIR = REPO_ROOT / "figures"


def main() -> None:
    T = np.linspace(0.0, 15.0, 300)

    syn = MechanoSynapse()
    Pr = np.asarray(syn.release_probability(T), dtype=np.float64)

    nmda = NMDAReceptor(alpha_NMDA=1.0)
    f_stretch = np.asarray(nmda.stretch_factor(T), dtype=np.float64)
    f_off = np.asarray(NMDAReceptor(alpha_NMDA=0.0).stretch_factor(T), dtype=np.float64)

    nav_gain = np.array([nav_current_gain(float(t)) for t in T])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    # (a) presynaptic release (Tyler)
    ax = axes[0]
    ax.plot(T, Pr, color="tab:blue", lw=2)
    ax.axhline(syn.Pr_baseline, color="grey", ls=":", lw=1,
               label=f"$P_r(0) = {syn.Pr_baseline:.2f}$ (baseline)")
    ax.axhline(syn.Pr_max, color="grey", ls="--", lw=1,
               label=f"$P_{{r,\\max}} = {syn.Pr_max:.2f}$")
    ax.set_xlabel("presynaptic tension (mN/m)")
    ax.set_ylabel("release probability $P_r$")
    ax.set_title("(a) Presynaptic release\n(Tyler 2008/2012)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    # (b) NMDA stretch (Maneshi)
    ax = axes[1]
    ax.plot(T, f_stretch, color="tab:green", lw=2, label=r"$\alpha_{\mathrm{NMDA}}=1$")
    ax.plot(T, f_off, color="tab:gray", lw=2, ls="--", label=r"$\alpha_{\mathrm{NMDA}}=0$ (disabled)")
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.axhline(1.0 + nmda.alpha_NMDA, color="grey", ls="--", lw=1,
               label=r"$1+\alpha_{\mathrm{NMDA}}$")
    ax.set_xlabel("membrane tension (mN/m)")
    ax.set_ylabel("NMDA stretch factor $f_{\\mathrm{stretch}}$")
    ax.set_title("(b) NMDA stretch enhancement\n(Maneshi 2017)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    # (c) Nav activation shift (Kubanek)
    ax = axes[2]
    ax.plot(T, nav_gain, color="tab:red", lw=2)
    ax.axhline(1.0, color="grey", ls=":", lw=1, label="unity at $T=0$ (no-op)")
    ax.set_xlabel("membrane tension (mN/m)")
    ax.set_ylabel("Na$^+$ current gain")
    ax.set_title("(c) Voltage-gated Na$^+$ shift\n(Kubanek 2016)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    for ext in ("pdf", "png"):
        out = OUT_DIR / f"fig_synapse_validation.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)

    print(f"  (a) Pr(0)={Pr[0]:.3f} (baseline {syn.Pr_baseline}), Pr(15)={Pr[-1]:.3f} "
          f"(max {syn.Pr_max})")
    print(f"  (b) f_stretch(0)={f_stretch[0]:.3f}, f_stretch(15)={f_stretch[-1]:.3f}, "
          f"sat=1+alpha={1.0 + nmda.alpha_NMDA:.2f}; alpha=0 flat at "
          f"{f_off.min():.3f}-{f_off.max():.3f}")
    print(f"  (c) nav_gain(0)={nav_gain[0]:.3f}, nav_gain(15)={nav_gain[-1]:.3f}, "
          f"monotone={bool(np.all(np.diff(nav_gain) >= -1e-12))}")


if __name__ == "__main__":
    main()
