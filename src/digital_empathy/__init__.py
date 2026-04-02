"""
digital_empathy — MCP server that wraps Meta's TRIBE v2 brain model
to produce a "Cognitive Friction Score" for UI screen recordings.

Package layout:
    server.py        — FastMCP entry point, tool definitions, lifespan
    inference.py     — TribeInferenceEngine (load / predict / unload)
    brain_regions.py — fsaverage5 vertex masks for PFC & visual cortex
    scoring.py       — friction score calculation (1-10 scale)
    visualization.py — brain surface heatmap renderer → PNG
"""
