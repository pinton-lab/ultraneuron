"""Per-case multi-page simulation reports (LaTeX output).

Produces a single PDF that walks through one simulation case across
all pipeline phases (acoustic → thermal → mechanical → firing →
spatiotemporal composite).  The PDF is rendered by ``pdflatex`` from
a self-contained LaTeX template populated with case-specific scalar
metrics and figure paths.

The first concrete adapter is :func:`build_tips_report`, targeting the
TIPS curved-bowl theta-burst dataset that ships with the manuscript
(``tips_thermal_neuromod/thermal_neuromod_results.npz`` upstream).  The
generic walker for arbitrary ``validation_results/<case>/`` trees is a
follow-up.
"""

from __future__ import annotations

import os
import shutil
import string
import subprocess
import tempfile
from typing import Any

import numpy as np


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _fmt(x: Any, spec: str = ".4g") -> str:
    """Format a scalar / array-zero-d for table display."""
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return str(x)


def _tex_escape(s: str) -> str:
    """Escape characters with special meaning in LaTeX text mode."""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def _run_pdflatex(tex_path: str, work_dir: str) -> None:
    """Run pdflatex twice in *work_dir*; raise RuntimeError on failure."""
    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode",
             "-halt-on-error", os.path.basename(tex_path)],
            cwd=work_dir, capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            log_tail = (result.stdout or "")[-2000:]
            raise RuntimeError(
                f"pdflatex failed (rc={result.returncode}):\n{log_tail}"
            )


# ---------------------------------------------------------------------
# TIPS-specific report (LaTeX)
# ---------------------------------------------------------------------


_LATEX_TEMPLATE = r"""\documentclass[9pt]{extarticle}
\usepackage[letterpaper,margin=0.4in,top=0.35in,bottom=0.4in]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{array}
\usepackage{xcolor}
\usepackage{titlesec}
\usepackage{enumitem}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{textcomp}

\definecolor{accAcoustic}{HTML}{1F77B4}
\definecolor{accThermal}{HTML}{D62728}
\definecolor{accMechanical}{HTML}{2CA02C}
\definecolor{accNeural}{HTML}{9467BD}
\definecolor{accNeutral}{HTML}{555555}
\definecolor{rulegrey}{HTML}{888888}

\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}
\setlength{\fboxsep}{2pt}
\renewcommand{\arraystretch}{0.95}
\pagestyle{empty}

\newcolumntype{R}{>{\raggedleft\arraybackslash}X}
\newcolumntype{L}{>{\raggedright\arraybackslash}X}

% Compact section header used between blocks
\newcommand{\blockhead}[2]{%
  \par\smallskip\noindent\textbf{\color{#1}#2}\par\vspace{1pt}%
  \hrule height 0.3pt\vspace{2pt}}

\begin{document}

%% --------------------- Title strip ---------------------
\noindent\begin{minipage}[t]{0.62\linewidth}
{\Large\bfseries Simulation Report}\\
{\footnotesize\color{accNeutral}${case_title}}
\end{minipage}\hfill
\begin{minipage}[t]{0.36\linewidth}\raggedleft
{\footnotesize\color{accNeutral}${paper}\\
acoustic $\to$ thermal $\to$ mechanical $\to$ firing}
\end{minipage}
\par\vspace{2pt}\hrule height 0.4pt\vspace{4pt}

%% --------------------- Headline KPIs ---------------------
\begin{tabularx}{\linewidth}{@{}*{4}{L}@{}}
{\scriptsize\color{accNeutral}$\Delta T_{\mathrm{peak}}$} &
{\scriptsize\color{accNeutral}Tension$_{\mathrm{peak}}$} &
{\scriptsize\color{accNeutral}Firing voxels (thermal)} &
{\scriptsize\color{accNeutral}Total spikes (thermal)}\\[-1pt]
{\large\bfseries ${peak_dT}\,K} &
{\large\bfseries ${peak_tension}\,mN/m} &
{\large\bfseries ${n_firing_thermal}} &
{\large\bfseries ${total_spikes_thermal}}\\[-1pt]
{\scriptsize\color{accNeutral}${fda_note}} &
{\scriptsize\color{accNeutral}${tension_x_thresh}$\times$ threshold} &
{\scriptsize\color{accNeutral}$\Delta$ vs iso: ${d_firing}} &
{\scriptsize\color{accNeutral}$\Delta$ vs iso: ${d_spikes}}\\
\end{tabularx}
\vspace{2pt}\hrule height 0.3pt

%% --------------------- Input parameters ---------------------
\blockhead{accNeutral}{Input parameters}
\renewcommand{\arraystretch}{0.92}
\begin{tabularx}{\linewidth}{@{}LR@{\hspace{4pt}}|@{\hspace{4pt}}LR@{\hspace{4pt}}|@{\hspace{4pt}}LR@{\hspace{4pt}}|@{\hspace{4pt}}LR@{}}
\multicolumn{2}{@{}l}{\textbf{\color{accAcoustic}\scriptsize Acoustic / protocol}} &
\multicolumn{2}{l}{\textbf{\color{accMechanical}\scriptsize Mechanical}} &
\multicolumn{2}{l}{\textbf{\color{accNeural}\scriptsize Channel / neuron}} &
\multicolumn{2}{l@{}}{\textbf{\color{accNeutral}\scriptsize Numerical / grid}}\\
\addlinespace[1pt]\hline\addlinespace[1pt]
{\scriptsize Source}        & {\scriptsize\ttfamily TIPS bowl} &
{\scriptsize Coupling $\alpha$} & {\scriptsize\ttfamily ${alpha}} &
{\scriptsize Channel}       & {\scriptsize\ttfamily ${channel}} &
{\scriptsize Grid}          & {\scriptsize\ttfamily ${grid}}\\
{\scriptsize $f_0$}         & {\scriptsize\ttfamily 650\,kHz} &
{\scriptsize $K_A$}         & {\scriptsize\ttfamily 0.25\,N/m} &
{\scriptsize $\bar g$}      & {\scriptsize\ttfamily ${g_piezo}\,mS/cm$^2$} &
{\scriptsize Voxels}        & {\scriptsize\ttfamily ${voxels}}\\
{\scriptsize Pulse}         & {\scriptsize\ttfamily theta-burst} &
{\scriptsize $T_{\mathrm{thresh}}$} & {\scriptsize\ttfamily ${t_thresh}\,mN/m} &
{\scriptsize $T_{1/2}$}     & {\scriptsize\ttfamily ${t_half}\,mN/m} &
{\scriptsize HH solver}     & {\scriptsize\ttfamily RK4}\\
{\scriptsize Duty}          & {\scriptsize\ttfamily ${duty}} &
{\scriptsize Map}           & {\scriptsize\ttfamily linear} &
{\scriptsize $\Delta A$}    & {\scriptsize\ttfamily ${delta_A}\,nm$^2$} &
{\scriptsize $\Delta t$}    & {\scriptsize\ttfamily 0.01\,ms}\\
{\scriptsize Free-field}    & {\scriptsize\ttfamily 400\,kPa} &
& &
{\scriptsize $\tau_0$/$E_{\mathrm{rev}}$}      & {\scriptsize\ttfamily ${tau_0}\,ms / ${e_rev}\,mV} &
{\scriptsize Bioheat}       & {\scriptsize\ttfamily Pennes FDTD}\\
& & & &
{\scriptsize Neuron}        & {\scriptsize\ttfamily Pospischil RS, 37\textdegree C} &
{\scriptsize $I_{\mathrm{stim}}$} & {\scriptsize\ttfamily ${i_stim}\,$\mu$A/cm$^2$}\\
\end{tabularx}

%% --------------------- Three-pathway figure row ---------------------
\blockhead{accNeutral}{Pathway maps}
\noindent
\begin{minipage}[t]{0.32\linewidth}
{\scriptsize\bfseries\color{accThermal}Thermal}\\[1pt]
\fbox{\includegraphics[width=\linewidth,height=1.55in,keepaspectratio]{${png_thermal}}}\\[2pt]
\renewcommand{\arraystretch}{0.95}
\begin{tabularx}{\linewidth}{@{}Lr@{}}
\hline
{\scriptsize Peak $\Delta T$ (K)} & {\scriptsize\ttfamily ${peak_dT}}\\
{\scriptsize $\Delta T$ in firing (K)} & {\scriptsize\ttfamily ${mean_dT_firing}}\\
{\scriptsize $\Delta T$ 99\% (K)} & {\scriptsize\ttfamily ${dT_q99}}\\
{\scriptsize FDA $\leq 1$\,K} & {\scriptsize\ttfamily ${fda_yn}}\\
\hline
\end{tabularx}
\end{minipage}\hfill
\begin{minipage}[t]{0.32\linewidth}
{\scriptsize\bfseries\color{accMechanical}Mechanical}\\[1pt]
\fbox{\includegraphics[width=\linewidth,height=1.55in,keepaspectratio]{${png_strain_tension}}}\\[2pt]
\renewcommand{\arraystretch}{0.95}
\begin{tabularx}{\linewidth}{@{}Lr@{}}
\hline
{\scriptsize Peak strain} & {\scriptsize\ttfamily ${peak_eps}}\\
{\scriptsize Peak tension (mN/m)} & {\scriptsize\ttfamily ${peak_tension}}\\
{\scriptsize Voxels $>$ thresh} & {\scriptsize\ttfamily ${n_over}}\\
{\scriptsize Vol $>$ thresh (\%)} & {\scriptsize\ttfamily ${pct_over}}\\
\hline
\end{tabularx}
\end{minipage}\hfill
\begin{minipage}[t]{0.32\linewidth}
{\scriptsize\bfseries\color{accNeural}Firing (iso vs thermal)}\\[1pt]
\fbox{\includegraphics[width=\linewidth,height=1.55in,keepaspectratio]{${png_firing}}}\\[2pt]
\renewcommand{\arraystretch}{0.95}
\begin{tabularx}{\linewidth}{@{}Lr@{}}
\hline
{\scriptsize Firing voxels (iso)}     & {\scriptsize\ttfamily ${n_firing_iso}}\\
{\scriptsize Firing voxels (thermal)} & {\scriptsize\ttfamily ${n_firing_thermal}}\\
{\scriptsize Spikes (iso)}            & {\scriptsize\ttfamily ${total_spikes_iso}}\\
{\scriptsize Spikes (thermal)}        & {\scriptsize\ttfamily ${total_spikes_thermal}}\\
\hline
\end{tabularx}
\end{minipage}

%% --------------------- Spike-count and snapshots ---------------------
\blockhead{accNeural}{Spike-count maps and spatiotemporal evolution}
\begin{minipage}[t]{0.42\linewidth}
\fbox{\includegraphics[width=\linewidth,height=1.7in,keepaspectratio]{${png_spikes}}}\\
{\scriptsize\color{accNeutral}Per-voxel spike counts: isothermal vs thermal-coupled.}
\end{minipage}\hfill
\begin{minipage}[t]{0.55\linewidth}
\fbox{\includegraphics[width=\linewidth,height=1.7in,keepaspectratio]{${png_snapshots}}}\\
{\scriptsize\color{accNeutral}Tension, $V_m$, and cumulative spikes at $t=3,8,20,40$\,ms across the theta-burst.}
\end{minipage}

\end{document}
"""


def build_tips_report(
    npz_path: str,
    png_dir: str,
    pdf_out: str,
    *,
    case_title: str = "TIPS curved-bowl theta-burst (Phase-2 framework)",
    paper: str = "Internal benchmark; no DOI",
) -> str:
    """Produce a 2-page LaTeX-rendered PDF report for the TIPS case.

    Parameters mirror the previous matplotlib version; output is now
    typeset by ``pdflatex`` so tables, fonts, and spacing are clean.
    """
    from ultraneuron.channel_library import PIEZO1

    data = np.load(npz_path, allow_pickle=True)
    threshold_tension = float(data["threshold_tension"])
    coupling_factor = float(data["coupling_factor"])
    duty_cycle = float(data["duty_cycle"])
    g_piezo = float(data["g_piezo"])
    i_stim = float(data["i_stim"])
    delta_T = data["delta_T"]
    tension_3d = data["tension_3d"]
    firing_iso = data["firing_map_iso"]
    firing_thermal = data["firing_map_thermal"]
    spike_iso = data["spike_count_iso"]
    spike_thermal = data["spike_count_thermal"]

    peak_dT_K = float(np.max(delta_T))
    peak_tension_mN_per_m = float(np.max(tension_3d))
    peak_eps_dev = float(np.max(np.asarray(data["deviatoric_strain"])))
    n_firing_iso = int(np.count_nonzero(firing_iso))
    n_firing_thermal = int(np.count_nonzero(firing_thermal))
    total_spikes_iso = int(spike_iso.sum())
    total_spikes_thermal = int(spike_thermal.sum())
    grid_shape = delta_T.shape
    n_voxels = grid_shape[0] * grid_shape[1] * grid_shape[2]
    over_thresh = tension_3d >= threshold_tension
    n_over = int(np.count_nonzero(over_thresh))
    mean_dT_firing = (
        float(delta_T[firing_iso].mean()) if n_firing_iso else 0.0
    )
    dT_q99 = float(np.quantile(delta_T, 0.99))

    d_firing = n_firing_thermal - n_firing_iso
    d_spikes = total_spikes_thermal - total_spikes_iso

    ctx: dict[str, str] = {
        "case_title": _tex_escape(case_title),
        "paper": _tex_escape(paper),
        "peak_dT": _fmt(peak_dT_K, ".3f"),
        "peak_tension": _fmt(peak_tension_mN_per_m, ".2f"),
        "peak_eps": _fmt(peak_eps_dev, ".3e"),
        "n_firing_iso": f"{n_firing_iso:,d}",
        "n_firing_thermal": f"{n_firing_thermal:,d}",
        "total_spikes_iso": f"{total_spikes_iso:,d}",
        "total_spikes_thermal": f"{total_spikes_thermal:,d}",
        "d_firing": f"{d_firing:+,d}",
        "d_spikes": f"{d_spikes:+,d}",
        "fda_note": "FDA-compliant ($\\leq 1$\\,K)" if peak_dT_K <= 1.0
                     else "exceeds 1\\,K",
        "fda_yn": "yes" if peak_dT_K <= 1.0 else "no",
        "tension_x_thresh": _fmt(peak_tension_mN_per_m / threshold_tension,
                                 ".1f"),
        "alpha": str(int(coupling_factor)),
        "channel": _tex_escape(PIEZO1.name),
        "g_piezo": _fmt(g_piezo, ".3f"),
        "t_half": _fmt(PIEZO1.T_half, ".2f"),
        "delta_A": _fmt(PIEZO1.delta_A, ".1f"),
        "tau_0": _fmt(PIEZO1.tau_0, ".1f"),
        "e_rev": _fmt(PIEZO1.E_rev, ".0f"),
        "t_thresh": _fmt(threshold_tension, ".2f"),
        "duty": _fmt(duty_cycle, ".4f"),
        "i_stim": _fmt(i_stim, ".2f"),
        "grid": "{}$\\times${}$\\times${}".format(*grid_shape),
        "voxels": f"{n_voxels / 1e6:.2f}\\,M",
        "mean_dT_firing": _fmt(mean_dT_firing, ".3f"),
        "dT_q99": _fmt(dT_q99, ".3f"),
        "n_over": f"{n_over:,d}",
        "pct_over": _fmt(100.0 * n_over / tension_3d.size, ".3f"),
        # Figure paths — relative to the work_dir we copy them into below
        "png_thermal": "fig_thermal.png",
        "png_strain_tension": "fig_strain_tension.png",
        "png_firing": "fig_firing.png",
        "png_spikes": "fig_spikes.png",
        "png_snapshots": "fig_snapshots.png",
    }

    # Render in a temp dir so build artefacts don't clutter the project.
    with tempfile.TemporaryDirectory(prefix="tips_report_") as work_dir:
        # Copy PNGs in under canonical names so the .tex doesn't carry
        # absolute paths (cleaner pdflatex behaviour and reproducibility).
        png_map = {
            "fig_thermal.png": "temperature_rise.png",
            "fig_strain_tension.png": "strain_tension.png",
            "fig_firing.png": "firing_comparison.png",
            "fig_spikes.png": "spike_count_comparison.png",
            "fig_snapshots.png": "firing_snapshots.png",
        }
        for dst, src in png_map.items():
            src_path = os.path.join(png_dir, src)
            if os.path.exists(src_path):
                shutil.copyfile(src_path, os.path.join(work_dir, dst))
            # If a PNG is missing pdflatex will fail loudly; preferable to
            # silent placeholders.

        tex_path = os.path.join(work_dir, "report.tex")
        with open(tex_path, "w") as fh:
            fh.write(string.Template(_LATEX_TEMPLATE).safe_substitute(ctx))

        _run_pdflatex(tex_path, work_dir)
        built = os.path.join(work_dir, "report.pdf")
        os.makedirs(os.path.dirname(os.path.abspath(pdf_out)) or ".",
                    exist_ok=True)
        shutil.copyfile(built, pdf_out)

    return pdf_out


# ---------------------------------------------------------------------
# CLI: ``python -m ultraneuron.io.report --tips``
# ---------------------------------------------------------------------


def _main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Build a multi-page simulation report PDF (LaTeX).",
    )
    p.add_argument("--tips", action="store_true",
                   help="Build the TIPS theta-burst case report.")
    p.add_argument(
        "--npz", default=None,
        help="Path to TIPS thermal_neuromod_results.npz.",
    )
    p.add_argument(
        "--png-dir", default=None,
        help="Directory containing the upstream-rendered PNGs.",
    )
    p.add_argument("--out", default=None,
                   help="Destination PDF path.")
    args = p.parse_args()

    if args.tips:
        out = args.out or os.path.join(
            os.path.dirname(__file__), "..", "..", "docs", "figures",
            "tips_report.pdf",
        )
        out = os.path.abspath(out)
        build_tips_report(args.npz, args.png_dir, out)
        print(f"Wrote {out}")
        return
    p.print_help()


if __name__ == "__main__":
    _main()
