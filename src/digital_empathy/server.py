"""
server.py — Digital Empathy MCP Server entry point.

This module defines:
  - The FastMCP application instance with lifespan model loading
  - The `evaluate_ui_friction` tool that agents call
  - The `main()` entry point for running via stdio transport

Connecting to this server
-------------------------
Add to claude_desktop_config.json (adjust paths):

    {
      "mcpServers": {
        "digital-empathy": {
          "command": "uv",
          "args": [
            "--directory",
            "/absolute/path/to/cursor_hack_tribev2",
            "run",
            "python", "-m", "digital_empathy.server"
          ]
        }
      }
    }

Or run directly:
    cd cursor_hack_tribev2
    uv run python -m digital_empathy.server

Architecture
------------
The TRIBE v2 model is loaded ONCE at server startup via the `lifespan`
async context manager.  Each tool call accesses the already-loaded model
through FastMCP's `ctx.request_context.lifespan_context`.

IMPORTANT: In stdio transport mode, stdout is the JSON-RPC wire.
All diagnostic output goes through `ctx.*` logging methods or to stderr.
Never use print() without file=sys.stderr.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from mcp.server.fastmcp import Context, FastMCP

from .brain_regions import BrainMasks, load_brain_masks
from .inference import TribeInferenceEngine
from .scoring import compute_friction_score
from .visualization import render_brain_heatmap

# ---------------------------------------------------------------------------
# Logging — stderr only (stdout is the MCP JSON-RPC channel)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directory for heatmap PNGs (relative to working directory at startup)
# ---------------------------------------------------------------------------
OUTPUT_DIR: Path = Path("./output").resolve()


# ---------------------------------------------------------------------------
# Lifespan context — holds everything that survives across tool calls
# ---------------------------------------------------------------------------

@dataclass
class AppContext:
    """Shared application state loaded once at server startup."""
    engine: TribeInferenceEngine    # TRIBE v2 loaded into MPS memory
    masks: BrainMasks               # fsaverage5 PFC + visual cortex masks


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Load all heavy resources once at startup; release on shutdown.

    This runs BEFORE the server begins accepting MCP requests, so the
    first tool call does not pay the model-loading cost.

    Yields
    ------
    AppContext
        Accessible inside every tool via ctx.request_context.lifespan_context
    """
    logger.info("=" * 60)
    logger.info("Digital Empathy MCP Server — starting up")
    logger.info("=" * 60)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", OUTPUT_DIR)

    # Load brain region masks (fast — downloads atlas once, then cached)
    logger.info("Loading brain region masks (Destrieux atlas on fsaverage5) …")
    masks = load_brain_masks()
    logger.info(
        "Masks loaded — PFC: %d vertices, Visual: %d vertices",
        masks.n_pfc_vertices,
        masks.n_visual_vertices,
    )

    # Load TRIBE v2 (slow on first run — downloads ~10 GB of weights)
    logger.info("Loading TRIBE v2 into memory (device=mps) …")
    engine = await TribeInferenceEngine.create(cache_dir="./cache")

    logger.info("Server ready — accepting tool calls.")
    logger.info("=" * 60)

    try:
        yield AppContext(engine=engine, masks=masks)
    finally:
        # Shutdown — release GPU memory
        logger.info("Server shutting down — releasing model memory …")
        await engine.unload()
        logger.info("Shutdown complete.")


# ---------------------------------------------------------------------------
# FastMCP application
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="digital-empathy",
    instructions=(
        "Analyzes screen recordings of UI interactions using Meta's TRIBE v2 "
        "brain model to predict cognitive friction. "
        "Call evaluate_ui_friction with paths to .mp4 recordings to receive "
        "a Cognitive Friction Score (1-10) and a brain activation heatmap PNG."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Tool: evaluate_ui_friction
# ---------------------------------------------------------------------------

@mcp.tool()
async def evaluate_ui_friction(
    video_paths: list[str],
    ctx: Context,
) -> str:
    """Analyze UI screen recordings and return Cognitive Friction Scores.

    This tool uses Meta's TRIBE v2 foundation model to predict human neural
    activity from a video of someone interacting with a UI.  It isolates the
    Prefrontal Cortex (cognitive strain) and Visual Cortex (baseline attention)
    signals to produce a 1-10 Friction Score.

    A score of 1 means the UI feels effortless; 10 means critical cognitive
    overload.  Use this signal to decide whether to refactor the UI.

    Args:
        video_paths: List of absolute paths to .mp4 screen recordings.
                     Each file should be a real human testing the UI.
                     The agent should NOT generate synthetic videos.

    Returns:
        A JSON string containing one result object per video:
        {
          "video_path": "/path/to/recording.mp4",
          "score": 6.2,
          "label": "Strained",
          "explanation": "The UI is creating measurable strain ...",
          "heatmap_path": "/path/to/output/recording_brain_heatmap.png",
          "debug": {
            "pfc_mean_activation": 0.42,
            "visual_mean_activation": 0.18,
            "cognitive_load_ratio": 2.33
          }
        }
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not video_paths:
        return json.dumps({"error": "video_paths must not be empty."})

    invalid = [p for p in video_paths if not Path(p).exists()]
    if invalid:
        return json.dumps({
            "error": "The following video files were not found.",
            "missing_files": invalid,
        })

    # Resolve app context (engine + masks loaded at startup)
    app: AppContext = ctx.request_context.lifespan_context

    results: list[dict] = []

    for i, video_path in enumerate(video_paths):
        await ctx.report_progress(
            progress=i,
            total=len(video_paths),
        )
        await ctx.info(
            f"Processing [{i + 1}/{len(video_paths)}]: {Path(video_path).name}"
        )

        try:
            # ----------------------------------------------------------
            # Phase 1: Run TRIBE v2 inference
            # ----------------------------------------------------------
            await ctx.info("Running TRIBE v2 neural prediction …")
            prediction = app.engine.predict_from_video(video_path)
            await ctx.info(
                f"Prediction complete — {prediction.n_timesteps} TR windows, "
                f"{prediction.n_vertices} cortical vertices."
            )

            # ----------------------------------------------------------
            # Phase 2: Compute friction score
            # ----------------------------------------------------------
            await ctx.info("Computing Cognitive Friction Score …")
            friction = compute_friction_score(
                activations=prediction.activations,
                masks=app.masks,
            )
            await ctx.info(
                f"Score: {friction.score:.1f}/10 — {friction.label}"
            )

            # ----------------------------------------------------------
            # Phase 3: Render brain heatmap
            # ----------------------------------------------------------
            await ctx.info("Rendering brain surface heatmap …")
            heatmap_path = render_brain_heatmap(
                activations=prediction.activations,
                video_path=video_path,
                output_dir=OUTPUT_DIR,
                pfc_mask=app.masks.pfc,
            )
            await ctx.info(f"Heatmap saved → {heatmap_path}")

            # ----------------------------------------------------------
            # Assemble result
            # ----------------------------------------------------------
            result = {
                "video_path": video_path,
                "heatmap_path": heatmap_path,
                **friction.to_dict(),
            }
            results.append(result)

        except FileNotFoundError as exc:
            await ctx.warning(f"File not found: {exc}")
            results.append({
                "video_path": video_path,
                "error": str(exc),
            })
        except ValueError as exc:
            await ctx.warning(f"Invalid input: {exc}")
            results.append({
                "video_path": video_path,
                "error": f"Invalid input: {exc}",
            })
        except Exception as exc:  # noqa: BLE001
            await ctx.error(f"Unexpected error processing {video_path}: {exc}")
            results.append({
                "video_path": video_path,
                "error": f"Prediction failed: {exc}",
            })

    await ctx.report_progress(
        progress=len(video_paths),
        total=len(video_paths),
    )

    return json.dumps(
        {"results": results},
        indent=2,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the MCP server over stdio transport.

    Called by the `digital-empathy` console script defined in pyproject.toml,
    or directly via:  python -m digital_empathy.server
    """
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
