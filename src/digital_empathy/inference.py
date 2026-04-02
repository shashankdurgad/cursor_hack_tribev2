"""
inference.py — TRIBE v2 model wrapper.

Responsibilities:
  - Load `facebook/tribev2` from HuggingFace Hub into unified memory once.
  - Accept a video file path, build the events DataFrame, run prediction.
  - Return the raw prediction array (n_timesteps × 20 484 cortical vertices)
    for downstream scoring and visualization.

TRIBE v2 API quick-reference (from facebookresearch/tribev2):
  TribeModel.from_pretrained(repo, cache_folder, device)
  model.get_events_dataframe(video_path=...) → pd.DataFrame
  model.predict(events=df, verbose=True)     → (np.ndarray, list[dict])
      preds shape: (n_timesteps, 20_484)   — fsaverage5 z-scored BOLD signal
      segments:    list of segment metadata dicts

Device strategy on Apple Silicon:
  We pass device="mps" explicitly so PyTorch uses the Metal Performance Shaders
  GPU backend built into the M-series chip.  If MPS is unavailable (e.g., CI),
  we fall back to CPU automatically.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Logging — MUST go to stderr; stdout is the MCP JSON-RPC wire
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_device() -> str:
    """Pick the best available device.

    Priority: MPS (Apple Silicon GPU) → CPU
    We skip CUDA because this project targets M-series Macs.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("MPS backend available — using Apple Silicon GPU.")
        return "mps"
    logger.warning("MPS not available — falling back to CPU.")
    return "cpu"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Holds everything returned from a single TRIBE v2 prediction run."""

    # Path to the source video that was analyzed
    video_path: str

    # Raw predicted BOLD activations — shape (T, 20_484)
    # T = number of 1-second TR windows; 20_484 = fsaverage5 vertices
    activations: np.ndarray

    # Segment metadata from TRIBE (start, end, features used, etc.)
    segments: list[dict[str, Any]] = field(default_factory=list)

    @property
    def n_timesteps(self) -> int:
        return self.activations.shape[0]

    @property
    def n_vertices(self) -> int:
        return self.activations.shape[1]


# ---------------------------------------------------------------------------
# Main engine class
# ---------------------------------------------------------------------------

class TribeInferenceEngine:
    """Loads TRIBE v2 once and exposes a simple predict() interface.

    Designed to be instantiated once inside the FastMCP lifespan context
    manager so the expensive model load happens only at server startup.

    Usage::

        engine = await TribeInferenceEngine.create()
        result = engine.predict_from_video("/path/to/recording.mp4")
        await engine.unload()
    """

    # HuggingFace repo identifier for TRIBE v2
    HF_REPO: str = "facebook/tribev2"

    def __init__(self, model: Any, device: str) -> None:
        # model is a TribeModel instance (imported lazily below)
        self._model = model
        self._device = device
        logger.info(
            "TribeInferenceEngine ready on device=%s  "
            "(model=%s)",
            device,
            self.HF_REPO,
        )

    # ------------------------------------------------------------------
    # Factory — async so we can run inside an async lifespan context
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        cache_dir: str | Path = "./cache",
    ) -> "TribeInferenceEngine":
        """Load TRIBE v2 weights into memory.

        This is the expensive step (~30 s on first run, cached thereafter).
        Call this once inside the FastMCP lifespan `async with` block.

        Parameters
        ----------
        cache_dir:
            Directory where HuggingFace Hub stores the downloaded weights.
            Defaults to ./cache relative to the working directory.
        """
        device = _select_device()
        cache_path = str(Path(cache_dir).resolve())

        logger.info(
            "Loading TRIBE v2 from HuggingFace Hub (repo=%s, cache=%s, device=%s) …",
            cls.HF_REPO,
            cache_path,
            device,
        )

        # Lazy import: TribeModel lives in the tribev2 package which is
        # installed separately from source (pip install -e path/to/tribev2).
        # We import here rather than at module level so the rest of the server
        # can boot even if the package isn't installed (useful for testing).
        try:
            from tribev2.demo_utils import TribeModel  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "tribev2 package not found.  Install it with:\n"
                "  git clone https://github.com/facebookresearch/tribev2\n"
                "  pip install -e ./tribev2"
            ) from exc

        # Load the full model in float16 to fit comfortably in 48 GB unified
        # memory while still being numerically accurate.
        #
        # Note: TribeModel.from_pretrained internally calls
        #   transformers.AutoModel.from_pretrained(..., torch_dtype=torch.float16)
        # for its sub-models, then moves everything to `device`.
        model = TribeModel.from_pretrained(
            cls.HF_REPO,
            cache_folder=cache_path,
            device=device,
        )

        logger.info("TRIBE v2 loaded successfully.")
        return cls(model=model, device=device)

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict_from_video(self, video_path: str | Path) -> PredictionResult:
        """Run TRIBE v2 on a single .mp4 recording.

        Steps:
          1. Build events DataFrame from the video file.
          2. Run the full multimodal prediction pipeline.
          3. Return raw activations + segment metadata.

        Parameters
        ----------
        video_path:
            Absolute or relative path to the .mp4 recording.
            The file must be readable by ffmpeg (installed separately).

        Returns
        -------
        PredictionResult
            .activations — np.ndarray of shape (T, 20_484)
            .segments    — list of segment dicts from TRIBE
        """
        path = Path(video_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        if path.suffix.lower() not in {".mp4", ".avi", ".mkv", ".mov", ".webm"}:
            raise ValueError(f"Unsupported video format: {path.suffix}")

        logger.info("Building events DataFrame from %s …", path)
        events_df = self._model.get_events_dataframe(video_path=str(path))
        logger.info("Events DataFrame built — %d rows.", len(events_df))

        logger.info("Running TRIBE v2 prediction (this may take a minute) …")
        preds, segments = self._model.predict(events=events_df, verbose=True)
        # preds: np.ndarray[float32], shape (n_timesteps, 20_484)
        # segments: list[dict] — one entry per temporal segment

        logger.info(
            "Prediction complete — activations shape: %s",
            preds.shape,
        )

        return PredictionResult(
            video_path=str(path),
            activations=preds.astype(np.float32),
            segments=list(segments),
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    async def unload(self) -> None:
        """Release model weights from memory.

        Called by the FastMCP lifespan context manager on server shutdown.
        On MPS, we explicitly free the device cache to return memory to the OS.
        """
        logger.info("Unloading TRIBE v2 model …")
        del self._model
        self._model = None  # type: ignore[assignment]

        if self._device == "mps":
            torch.mps.empty_cache()
        elif self._device == "cuda":
            torch.cuda.empty_cache()

        logger.info("Model unloaded; memory released.")
