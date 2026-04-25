"""NSGA-III based class-level pre-clustering for microxpert.

This package borrows the multi-objective optimisation idea from MONO2REST
but operates on **class-level** nodes (instead of method-level) so that the
output plugs directly into microxpert's ``ServiceClusterConfig`` schema.

Pipeline (entry point: :func:`nsga3.main`):

1. Load ir-a.json and build a class graph (nodes = classes, weighted edges
   = aggregated method calls + field uses + inheritance + imports).
2. Build a "rich" textual descriptor for every class
   (class + method + field + annotation simple names) and embed with SBERT.
3. Run a custom NSGA-III GA over class assignments using three objectives:
   - minimise structural coupling
   - maximise structural cohesion
   - maximise semantic similarity inside clusters
4. Emit ``clusters.json`` compatible with microxpert's
   ``ServiceClusterConfig`` schema.
"""

from .main import NSGA3Pipeline, run

__all__ = ["NSGA3Pipeline", "run"]
