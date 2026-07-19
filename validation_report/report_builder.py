"""Validation-report driver.

Usage:
    python -m validation_report.report_builder \\
        --config validation_report.configs.halle_bowl

Or with a path:
    python report_builder.py --config configs/halle_bowl.py

Sections built so far: acoustic_inputs (1). 2..5 to come.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import jinja2

THIS_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = THIS_DIR / "templates"
# validation_report/ is at validation_report/
# repo root is two levels up from there.
REPO_ROOT = THIS_DIR.parent


def _load_config(arg: str) -> ModuleType:
    """Accept a dotted module path OR a filesystem path to a .py config."""
    p = Path(arg)
    if p.exists() and p.suffix == ".py":
        spec = importlib.util.spec_from_file_location("vr_config", p)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load config from {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(arg)


_TEX_SPECIAL = {
    "\\": r"\textbackslash{}",
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _tex_escape(s):
    if s is None:
        return ""
    out = str(s)
    out = out.replace("\\", _TEX_SPECIAL["\\"])
    for ch, repl in _TEX_SPECIAL.items():
        if ch == "\\":
            continue
        out = out.replace(ch, repl)
    return out


def _map_fmt(seq, fmt):
    return [fmt % v for v in seq]


def _render_tex(ctx: dict, template_path: Path) -> str:
    # Custom comment delimiters so LaTeX patterns like {#1} in \newcommand
    # don't trigger Jinja's default {# ... #} comment parser.
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_path.parent)),
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=False,
        lstrip_blocks=False,
        comment_start_string="{##",
        comment_end_string="##}",
    )
    env.filters["tex_escape"] = _tex_escape
    env.filters["map_fmt"] = _map_fmt
    tmpl = env.get_template(template_path.name)
    return tmpl.render(ctx=ctx)


def _compile_pdf(tex_path: Path, timeout: int = 120) -> Path:
    cwd = tex_path.parent
    for _ in range(2):  # two passes for cross-refs / lastpage
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=cwd, capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            log = tex_path.with_suffix(".log")
            tail = ""
            if log.exists():
                tail = log.read_text(errors="replace").splitlines()[-60:]
                tail = "\n".join(tail)
            raise RuntimeError(
                f"pdflatex failed (rc={proc.returncode}).\n--- last log ---\n{tail}"
            )
    return tex_path.with_suffix(".pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a sim validation report.")
    ap.add_argument("--config", required=True,
                    help="Dotted module path or filesystem path to a config .py")
    ap.add_argument("--sections",
                    default="acoustic_inputs,acoustic_outputs,thermal_outputs,displacement_outputs,elastic_outputs,neural_outputs,astrocyte_outputs,synaptic_outputs",
                    help="Comma-separated list of section names. "
                         "Currently supported: acoustic_inputs, "
                         "acoustic_outputs, thermal_outputs, "
                         "displacement_outputs, elastic_outputs, "
                         "neural_outputs, astrocyte_outputs, "
                         "synaptic_outputs.")
    ap.add_argument("--with-animation", action="store_true",
                    help="Generate 3-D propagation animations for each --anim-field: "
                         "render an MP4 AND embed a napari frame sequence via "
                         "\\animategraphics. Expensive; needs a display + GPU. Off by default.")
    ap.add_argument("--anim-fields", default="pressure,shear,temperature,neural",
                    help="comma-separated fields to animate (missing inputs skipped)")
    ap.add_argument("--anim-frames", type=int, default=120,
                    help="frames per animation (shared by the MP4 and the embedded frames)")
    ap.add_argument("--anim-fps", type=int, default=10,
                    help="in-PDF \\animategraphics playback fps (independent of MP4 speed)")
    ap.add_argument("--anim-size", type=int, default=1000,
                    help="render size in px for the standalone MP4s (square)")
    ap.add_argument("--anim-embed-size", type=int, default=480,
                    help="downsampled px size of the frames embedded in the PDF "
                         "(keeps the MP4 sharp while the PDF stays reasonable)")
    ap.add_argument("--no-compile", action="store_true",
                    help="Render TEX but skip pdflatex.")
    ap.add_argument("--tex-only", action="store_true",
                    help="Skip all builders; re-render TEX + PDF from cached "
                         "manifest.json. Use this to iterate on the template "
                         "without recomputing figures.")
    args = ap.parse_args()

    cfg = _load_config(args.config)
    out_dir = Path(cfg.OUT_DIR)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline schematic (manuscript Fig 1) — copy alongside per-sim figures
    # so the report can reference it by basename via graphicspath.
    pipeline_src = REPO_ROOT / "figures" / "fig1_pipeline.pdf"
    if pipeline_src.exists():
        shutil.copy2(pipeline_src, fig_dir / pipeline_src.name)

    manifest_path = out_dir / "manifest.json"

    if args.tex_only:
        if not manifest_path.exists():
            raise SystemExit(
                f"--tex-only requires a prior full build; no manifest at "
                f"{manifest_path}"
            )
        ctx = json.loads(manifest_path.read_text())
    else:
        sections = [s.strip() for s in args.sections.split(",") if s.strip()]
        # Seed ctx from cached manifest so sections not in --sections keep
        # their prior context (avoids re-running e.g. the 90 s acoustic_inputs
        # NRRD/slab load when iterating on acoustic_outputs).
        ctx = (json.loads(manifest_path.read_text())
               if manifest_path.exists() else {})

        for sec in sections:
            if sec == "acoustic_inputs":
                from builders import acoustic_inputs
                sec_ctx = acoustic_inputs.build(cfg, fig_dir)
                ctx.update(sec_ctx)
            elif sec == "acoustic_outputs":
                from builders import acoustic_outputs
                sec_ctx = acoustic_outputs.build(cfg, fig_dir)
                ctx.update(sec_ctx)
            elif sec == "thermal_outputs":
                from builders import thermal_outputs
                sec_ctx = thermal_outputs.build(cfg, fig_dir)
                ctx.update(sec_ctx)
            elif sec == "displacement_outputs":
                from builders import displacement_outputs
                sec_ctx = displacement_outputs.build(cfg, fig_dir)
                ctx.update(sec_ctx)
            elif sec == "elastic_outputs":
                from builders import elastic_outputs
                sec_ctx = elastic_outputs.build(cfg, fig_dir)
                ctx.update(sec_ctx)
            elif sec == "neural_outputs":
                from builders import neural_outputs
                sec_ctx = neural_outputs.build(cfg, fig_dir)
                ctx.update(sec_ctx)
            elif sec == "astrocyte_outputs":
                from builders import astrocyte_outputs
                sec_ctx = astrocyte_outputs.build(cfg, fig_dir)
                ctx.update(sec_ctx)
            elif sec == "synaptic_outputs":
                from builders import synaptic_outputs
                sec_ctx = synaptic_outputs.build(cfg, fig_dir)
                ctx.update(sec_ctx)
            else:
                raise SystemExit(f"Unknown section: {sec}")

        if args.with_animation:
            from builders import wave_animation
            fields = [f.strip() for f in args.anim_fields.split(",") if f.strip()]
            res = wave_animation.build(
                cfg, fig_dir, fields=fields,
                n_frames=args.anim_frames, fps=args.anim_fps,
                size=args.anim_size, embed_size=args.anim_embed_size,
            )
            # Merge per-field so re-rendering a subset keeps the others' animations.
            ctx.setdefault("animations", {}).update(res.get("animations", {}))
        else:
            # Flag is authoritative on a normal build: drop any stale animation
            # context seeded from a prior manifest so we don't embed frames that
            # may no longer exist. (--tex-only deliberately keeps the manifest.)
            ctx.pop("animations", None)
            ctx.pop("wave_animation", None)

        (out_dir / "manifest.json").write_text(
            json.dumps(ctx, indent=2, default=str)
        )

    # The template references ctx.animations.<field> (via the anim_fig macro);
    # ensure the key always exists so non-animation builds (and old manifests
    # under --tex-only) don't hit Jinja's Undefined attribute error.
    ctx.setdefault("animations", {})

    # Render tex
    tex_src = _render_tex(ctx, TEMPLATE_DIR / "report.tex.j2")
    tex_path = out_dir / f"validation_report_{cfg.SIM_NAME}.tex"
    tex_path.write_text(tex_src)

    # pdflatex compiles relative to the tex file's directory; the figures live in
    # ./figures/ — the template references them by basename, so copy / symlink.
    # Easiest: move them into out_dir directly. We already saved to fig_dir; the
    # template strings include just the basename, so update graphicspath:
    # cleaner — patch tex by inserting graphicspath after \begin{document}? No,
    # simpler still: the template calls \includegraphics{<basename>}. Pre-compose
    # graphicspath via a TEX header insert.
    tex_with_path = tex_src.replace(
        "\\begin{document}",
        "\\graphicspath{{figures/}}\n\\begin{document}",
    )
    tex_path.write_text(tex_with_path)

    print(f"Wrote {tex_path}")
    if args.no_compile:
        return

    # Embedding animation frames makes pdflatex much slower; scale with the
    # total embedded frame count (each \animategraphics inlines all its frames).
    total_anim_frames = sum(a.get("n_frames", 0)
                            for a in (ctx.get("animations") or {}).values()
                            if a.get("available"))
    compile_timeout = max(120, 120 + total_anim_frames * 4) if total_anim_frames else 120
    pdf_path = _compile_pdf(tex_path, timeout=compile_timeout)
    print(f"Wrote {pdf_path}")
    # Clean aux files
    for ext in (".aux", ".log", ".out", ".toc"):
        f = tex_path.with_suffix(ext)
        if f.exists():
            f.unlink()


if __name__ == "__main__":
    sys.path.insert(0, str(THIS_DIR))
    main()
