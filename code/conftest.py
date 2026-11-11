"""Ensure pytest resolves top-level packages (models/ dataset/ training/
calibration/ config/ entry/) from the repository root.

With no pytest configuration file, a minimal sys.path injection ensures both
``pytest`` and ``python -m pytest`` can resolve top-level packages.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
