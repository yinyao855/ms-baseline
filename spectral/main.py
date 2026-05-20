"""
Spectral Clustering baseline — class-level graph partitioning that accepts K
directly. Replaces Louvain in the RQ1 K-controlled comparison because Louvain
only optimises modularity and does not natively accept a target cluster count.

Algorithm:
  1. Build a symmetric weighted adjacency from the class-level call graph
     (weight = method-call count, treated as edge similarity).
  2. Add a small uniform self-loop / floor (1e-6) so isolated classes are
     reachable in the affinity matrix without sklearn warnings.
  3. Run sklearn.cluster.SpectralClustering with affinity='precomputed' and
     n_clusters=K. Normalized cut produces K-balanced partitions naturally
     without the god-cluster pattern that Girvan-Newman exhibits.

Usage::

    python -m spectral.main -i data/petclinic/ir-a.json -k 4
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.cluster import SpectralClustering

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.ir_parser import IrAProject


def build_affinity_matrix(project: IrAProject) -> np.ndarray:
    """Symmetric affinity (N, N) from the class-level call graph.

    affinity[i][j] = call_weight(i -> j) + call_weight(j -> i). A small floor
    is added on the diagonal and off-diagonal so the graph stays connected
    enough for Spectral's normalised-cut Laplacian.
    """
    fqns = project.class_fqns
    n = len(fqns)
    idx = {fqn: i for i, fqn in enumerate(fqns)}
    a = np.zeros((n, n), dtype=np.float64)
    for src, targets in project.class_call_weights.items():
        i = idx.get(src)
        if i is None:
            continue
        for dst, w in targets.items():
            j = idx.get(dst)
            if j is None or i == j:
                continue
            a[i][j] += float(w)
    # symmetrize and add tiny floor so disconnected components survive
    a = a + a.T
    a += 1e-6
    np.fill_diagonal(a, 0.0)
    return a


def run_spectral(
    project: IrAProject,
    num_clusters: int = 4,
    seed: int = 42,
) -> Dict[str, int]:
    """Spectral clustering on the precomputed call-graph affinity.

    Returns {class_fqn: cluster_id}. Cluster IDs are 0..K-1 and the partition
    is guaranteed to have exactly K clusters (assuming N >= K).
    """
    affinity = build_affinity_matrix(project)
    k = min(num_clusters, project.num_classes)
    if k <= 1:
        return {fqn: 0 for fqn in project.class_fqns}

    model = SpectralClustering(
        n_clusters=k,
        affinity="precomputed",
        assign_labels="kmeans",
        random_state=seed,
        n_init=10,
    )
    labels = model.fit_predict(affinity)
    return {fqn: int(labels[i]) for i, fqn in enumerate(project.class_fqns)}


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Spectral Clustering baseline (normalized-cut graph partitioning)")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to ir-a.json")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: result/spectral/<project>)")
    parser.add_argument("--clusters", "-k", type=int, default=4,
                        help="Number of clusters (target partition count)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project = IrAProject(args.input)
    print(f"[Spectral] Loaded {project.num_classes} classes")
    print(f"[Spectral] Target K = {args.clusters}")

    class_to_cluster = run_spectral(
        project,
        num_clusters=args.clusters,
        seed=args.seed,
    )
    actual_k = len(set(class_to_cluster.values()))
    print(f"[Spectral] Clustered into {actual_k} clusters (target k={args.clusters})")

    from collections import Counter
    cluster_sizes = Counter(class_to_cluster.values())
    for cid in sorted(cluster_sizes.keys()):
        print(f"  cluster {cid}: {cluster_sizes[cid]} classes")

    if args.output is None:
        project_name = Path(args.input).resolve().parent.name
        args.output = str(Path(__file__).resolve().parent.parent / "result" / "spectral" / project_name)
    os.makedirs(args.output, exist_ok=True)

    result = project.build_clusters_json(class_to_cluster, algorithm="Spectral")
    out_path = os.path.join(args.output, "clusters.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
