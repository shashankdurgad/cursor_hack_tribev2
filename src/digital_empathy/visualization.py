"""
visualization.py — Brain surface heatmap renderer.

Produces a 2×2 grid PNG showing predicted BOLD activations mapped onto the
fsaverage5 cortical surface, with PFC regions highlighted for the human
reviewer (and for the hackathon demo).

Layout of the 2×2 grid:
  ┌─────────────────┬─────────────────┐
  │  Left  Lateral  │  Right Lateral  │
  ├─────────────────┼─────────────────┤
  │  Left  Medial   │  Right Medial   │
  └─────────────────┴─────────────────┘

Color scheme:
  - Base activation map: "cold_hot" diverging colormap (blue = deactivation,
    red/yellow = strong activation)
  - PFC ROI boundary: overlaid as a semi-transparent orange mask so the
    cognitive-strain region is immediately visible

Rendering backend:
  We use nilearn's plot_surf_stat_map() which is a pure matplotlib/OpenGL
  solution — no display required when using matplotlib's Agg backend.
  This is critical for a headless MCP server.

Output:
  A PNG at output/<video_stem>_brain_heatmap.png
  Returns the absolute path string so the MCP response can include it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Nilearn surface plotting config
# ---------------------------------------------------------------------------

# Views to render — (hemi, view) pairs matching the 2×2 grid
_SURFACE_VIEWS: list[tuple[str, str]] = [
    ("left",  "lateral"),
    ("right", "lateral"),
    ("left",  "medial"),
    ("right", "medial"),
]

# Matplotlib colormap for the activation stat map
_CMAP: str = "cold_hot"

# Symmetrical color range — z-score BOLD activations typically span ±2
_VMAX: float = 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_brain_heatmap(
    activations: np.ndarray,
    video_path: str,
    output_dir: str | Path = "./output",
    pfc_mask: np.ndarray | None = None,
) -> str:
    """Render a 2×2 brain surface heatmap PNG and return its absolute path.

    Parameters
    ----------
    activations:
        Raw TRIBE v2 output — shape (T, 20_484).  We compute the temporal
        mean before projecting onto the surface.
    video_path:
        Source video path (used only to derive the output filename).
    output_dir:
        Directory where the PNG is saved.  Created if it doesn't exist.
    pfc_mask:
        Optional boolean array of shape (20_484,).  If provided, PFC vertices
        are overlaid with an orange boundary for visual emphasis.

    Returns
    -------
    str
        Absolute path to the saved PNG file.
    """
    # ------------------------------------------------------------------
    # Force headless matplotlib backend BEFORE importing pyplot
    # (must happen before any pyplot import in this process)
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")  # non-interactive, no display needed
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    try:
        from nilearn import datasets as nl_datasets
        from nilearn import plotting as nl_plotting
        from nilearn import surface as nl_surface
    except ImportError as exc:
        raise ImportError(
            "nilearn is required for brain visualization.  "
            "Install it with:  pip install nilearn"
        ) from exc

    # ------------------------------------------------------------------
    # Prepare the stat map (temporal mean over all TRs)
    # ------------------------------------------------------------------
    if activations.ndim != 2 or activations.shape[1] != 20_484:
        raise ValueError(
            f"activations must be shape (T, 20_484), got {activations.shape}"
        )

    mean_act = activations.mean(axis=0).astype(np.float32)  # (20_484,)
    # Split into hemispheres (nilearn expects per-hemi arrays for surface plotting)
    left_data  = mean_act[:10_242]   # (10_242,)
    right_data = mean_act[10_242:]   # (10_242,)
    hemi_data  = {"left": left_data, "right": right_data}

    # ------------------------------------------------------------------
    # Fetch fsaverage5 surface mesh (cached by nilearn after first download)
    # ------------------------------------------------------------------
    logger.info("Fetching fsaverage5 surface mesh for visualization …")
    fsaverage = nl_datasets.fetch_surf_fsaverage(mesh="fsaverage5")

    # Mesh keys in the nilearn fsaverage dict:
    #   infl_left / infl_right  — inflated surface (good for medial view)
    #   pial_left / pial_right  — pial surface (realistic shape)
    meshes = {
        "left":  {"lateral": fsaverage.infl_left,  "medial": fsaverage.infl_left},
        "right": {"lateral": fsaverage.infl_right, "medial": fsaverage.infl_right},
    }
    bg_maps = {
        "left":  fsaverage.sulc_left,
        "right": fsaverage.sulc_right,
    }

    # ------------------------------------------------------------------
    # Build PFC ROI overlay texture (if mask provided)
    # ------------------------------------------------------------------
    pfc_overlay: dict[str, np.ndarray | None] = {"left": None, "right": None}
    if pfc_mask is not None:
        pfc_left  = pfc_mask[:10_242].astype(np.float32)
        pfc_right = pfc_mask[10_242:].astype(np.float32)
        # Convert bool → 0/1; non-zero will be drawn as a separate color layer
        pfc_overlay = {"left": pfc_left, "right": pfc_right}

    # ------------------------------------------------------------------
    # Compose the 2×2 figure
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16, 10), facecolor="#1a1a2e")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.05, wspace=0.02)

    titles = [
        "Left Hemisphere — Lateral",
        "Right Hemisphere — Lateral",
        "Left Hemisphere — Medial",
        "Right Hemisphere — Medial",
    ]

    for idx, (hemi, view) in enumerate(_SURFACE_VIEWS):
        ax = fig.add_subplot(gs[idx // 2, idx % 2], projection="3d")
        ax.set_facecolor("#1a1a2e")

        # Render the activation stat map
        nl_plotting.plot_surf_stat_map(
            surf_mesh=meshes[hemi][view],
            stat_map=hemi_data[hemi],
            hemi=hemi,
            view=view,
            bg_map=bg_maps[hemi],
            bg_on_data=True,
            colorbar=(idx == 1),          # only one colorbar (top-right panel)
            cmap=_CMAP,
            vmax=_VMAX,
            symmetric_cbar=True,
            axes=ax,
            figure=fig,
            title=None,                   # we add our own styled title
        )

        # Overlay PFC mask in semi-transparent orange
        if pfc_overlay[hemi] is not None and pfc_overlay[hemi].sum() > 0:
            nl_plotting.plot_surf_roi(
                surf_mesh=meshes[hemi][view],
                roi_map=pfc_overlay[hemi],
                hemi=hemi,
                view=view,
                bg_map=bg_maps[hemi],
                bg_on_data=True,
                alpha=0.35,
                cmap="Oranges",
                axes=ax,
                figure=fig,
            )

        # Panel title
        ax.set_title(
            titles[idx],
            color="#e0e0e0",
            fontsize=11,
            pad=4,
        )

    # ------------------------------------------------------------------
    # Figure-level title and annotation
    # ------------------------------------------------------------------
    video_stem = Path(video_path).stem
    fig.suptitle(
        f"TRIBE v2 — Predicted Neural Activation\n{video_stem}",
        color="#ffffff",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Subtle legend annotation
    fig.text(
        0.5, 0.01,
        "Red/Yellow = high activation  |  Blue = deactivation  |  "
        "Orange overlay = Prefrontal Cortex (cognitive load region)",
        ha="center",
        color="#aaaaaa",
        fontsize=9,
    )

    # ------------------------------------------------------------------
    # Save to disk
    # ------------------------------------------------------------------
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_stem}_brain_heatmap.png"

    fig.savefig(
        str(out_path),
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    logger.info("Brain heatmap saved → %s", out_path)
    return str(out_path)
