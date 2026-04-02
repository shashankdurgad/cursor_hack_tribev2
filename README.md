# Digital Empathy — MCP Server

> **"What if your AI coding agent could feel cognitive friction?"**

An MCP (Model Context Protocol) server that exposes Meta's **TRIBE v2** — a multimodal foundation model that predicts human neural activity — as a callable tool for AI coding agents.

Instead of guessing whether a UI "feels right," agents can now get a **mathematically grounded Cognitive Friction Score** derived from predicted brain activity.

---

## The Problem

AI coding agents write functional code. They cannot evaluate UX.
They don't know if a button placement causes confusion, if a form flow is exhausting, or if an error message triggers cognitive overload. They have no sensory feedback loop for human experience.

## The Solution

```
Human records a screen → Agent calls evaluate_ui_friction() → TRIBE v2 predicts brain activity → Score + heatmap returned
```

1. A **human tests the UI** and records a `.mp4` of the interaction
2. The **agent calls this MCP server** with the video path
3. **TRIBE v2** predicts z-scored fMRI BOLD activations across 20,484 cortical vertices
4. We isolate the **Prefrontal Cortex** (cognitive strain) and **Visual Cortex** (baseline attention)
5. The **Cognitive Friction Score** (1–10) and a **brain heatmap PNG** are returned

The agent reads the score and decides: polish or refactor.

---

## Score Interpretation

| Score | Label | Meaning |
|-------|-------|---------|
| 1.0 – 2.5 | **Effortless** | Intuitive UI; minimal cognitive overhead |
| 2.5 – 4.0 | **Comfortable** | Slight attention needed; well-designed flow |
| 4.0 – 5.5 | **Moderate** | Noticeable friction; review navigation or labels |
| 5.5 – 7.0 | **Strained** | Confusing patterns; audit layout and CTAs |
| 7.0 – 8.5 | **High Friction** | Users are struggling; significant refactor needed |
| 8.5 – 10.0 | **Critical** | Severe cognitive overload; fundamental redesign required |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AI Coding Agent                          │
│                    (Cursor / Claude Code)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │ MCP tool call
                           │ evaluate_ui_friction(video_paths)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Digital Empathy MCP Server                     │
│                        (server.py)                              │
│                                                                 │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────┐  │
│  │ inference.py │──▶│  scoring.py  │──▶│  visualization.py   │  │
│  │             │   │              │   │                     │  │
│  │  TribeModel  │   │ CLR → 1-10   │   │  nilearn surface    │  │
│  │  .predict()  │   │  sigmoid     │   │  plot → PNG         │  │
│  └─────────────┘   └──────────────┘   └─────────────────────┘  │
│         ▲                  ▲                                    │
│         │                  │                                    │
│  ┌──────┴──────┐   ┌───────┴────────┐                          │
│  │  .mp4 file  │   │ brain_regions  │                          │
│  │  (human     │   │ Destrieux atlas│                          │
│  │   recorded) │   │ PFC + V1/V2    │                          │
│  └─────────────┘   └────────────────┘                          │
└─────────────────────────────────────────────────────────────────┘
                           │
                           ▼
            JSON: { score, label, explanation,
                    heatmap_path, debug }
```

---

## Hardware Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4) with **≥ 48 GB Unified Memory**
- TRIBE v2 runs in **float16** on the **MPS backend** — no quantization needed
- ~10 GB disk for model weights (cached after first download)

---

## Project Structure

```
cursor_hack_tribev2/
├── pyproject.toml                 # Dependencies and build config
├── README.md                      # This file
├── src/
│   └── digital_empathy/
│       ├── __init__.py
│       ├── server.py              # FastMCP server — tools, lifespan, entry point
│       ├── inference.py           # TribeInferenceEngine (load / predict / unload)
│       ├── brain_regions.py       # Destrieux atlas masks for PFC & visual cortex
│       ├── scoring.py             # Friction score: CLR → sigmoid → 1-10
│       └── visualization.py       # 2×2 brain surface heatmap → PNG
├── output/                        # Generated heatmap PNGs (gitignored)
├── tests/
│   └── test_scoring.py
└── cache/                         # HuggingFace model weights (gitignored)
```

---

## Installation

### 1. Clone and install TRIBE v2 from source

TRIBE v2 is not on PyPI — install directly from the Meta research repo:

```bash
git clone https://github.com/facebookresearch/tribev2
pip install -e ./tribev2
```

> **HuggingFace gated access required.** TRIBE v2 uses LLaMA 3.2-3B internally.
> You must accept the license at [facebook/tribev2](https://huggingface.co/facebook/tribev2)
> and log in before running:
> ```bash
> huggingface-cli login
> ```

### 2. Install this package

```bash
pip install -e .
```

Or install dependencies manually:

```bash
pip install "mcp[cli]" torch torchvision torchaudio \
            transformers huggingface-hub \
            nilearn nibabel \
            matplotlib pyvista \
            pandas numpy Pillow
```

### 3. Verify PyTorch MPS is available

```python
import torch
print(torch.backends.mps.is_available())  # should print True
```

---

## Running the Server

### Standalone (for testing)

```bash
cd cursor_hack_tribev2
python -m digital_empathy.server
```

On first boot you'll see the model downloading (~10 GB). Subsequent starts load from `./cache` in seconds.

### With Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "digital-empathy": {
      "command": "python",
      "args": ["-m", "digital_empathy.server"],
      "cwd": "/absolute/path/to/cursor_hack_tribev2"
    }
  }
}
```

### With `mcp dev` (interactive inspector)

```bash
mcp dev src/digital_empathy/server.py
```

---

## Using the Tool

Once connected, call:

```
evaluate_ui_friction(video_paths=["/path/to/recording.mp4"])
```

**Example response:**

```json
{
  "results": [
    {
      "video_path": "/recordings/checkout_flow.mp4",
      "score": 6.8,
      "label": "Strained",
      "explanation": "The UI is creating measurable strain. High prefrontal activation suggests users are holding multiple mental models simultaneously. Audit for confusing layouts, ambiguous CTAs, or information overload.",
      "heatmap_path": "/output/checkout_flow_brain_heatmap.png",
      "debug": {
        "pfc_mean_activation": 0.4821,
        "visual_mean_activation": 0.1923,
        "cognitive_load_ratio": 2.507
      }
    }
  ]
}
```

The agent reads this, opens the heatmap to visually confirm where activation is concentrated, and decides which UI components to refactor.

---

## The Science

**TRIBE v2** is trained on the Courtois-Neuromod fMRI dataset — thousands of hours of human participants watching videos inside an MRI scanner. The model learns to predict where the brain lights up in response to any audiovisual stimulus.

We tap into two regions:

| Region | Role in scoring | Atlas labels |
|--------|----------------|-------------|
| **Prefrontal Cortex (PFC)** | Cognitive strain, working memory, decision-making | `frontal`, `orbital`, `cingulate_ant` |
| **Visual Cortex (V1/V2)** | Baseline sensory processing — normalization anchor | `occipital`, `calcarine`, `cuneus`, `lingual` |

The **Cognitive Load Ratio** = PFC activation / (Visual activation + ε)

A high ratio means the brain is doing hard thinking relative to simple visual processing — the signature of a confusing interface.

---

## License

TRIBE v2 model weights: [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Meta)
This server code: MIT
