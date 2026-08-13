#!/usr/bin/env python3
"""Gallileo-4D: Frozen Backbone Ensemble for Dynamic 4D Reconstruction.

3rd Place at PhysAI Dynamic 4D Reconstruction Challenge (ECCV 2026)
Final Score: 0.58356 APD

Usage:
    # Single component inference
    python -m gallileo4d.run --checkpoint /path/to/4rc --benchmark_root /path/to/data
    
    # Full ensemble (reproduces 0.58356)
    python -m gallileo4d.run --checkpoint /path/to/4rc --benchmark_root /path/to/data --ensemble
"""

from gallileo4d.pipeline import main

if __name__ == "__main__":
    main()
