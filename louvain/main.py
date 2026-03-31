"""
Louvain baseline — community detection on the class-level call graph.

Uses networkx's Louvain community detection algorithm to partition classes
into clusters based on structural coupling (method invocation relationships).

Usage::

    python -m louvain.main -i data/petclinic/ir-a.json
    python -m louvain.main -i data/daytrader/ir-a.json --resolution 1.5
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict

import networkx as nx
from networkx.algorithms.community import louvain_communities

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.ir_parser import IrAProject


def run_louvain(project: IrAProject, resolution: float = 1.0, seed: int = 42) -> Dict[str, int]:
    """Run Louvain community detection on the class-level call graph.

    Returns:
        {class_fqn: cluster_id}
    """
    G = nx.Graph()
    G.add_nodes_from(project.class_fqns)

    # Add weighted edges (undirected: merge both directions)
    edge_weights: Dict[tuple, int] = {}
    for src, targets in project.class_call_weights.items():
        for dst, w in targets.items():
            key = tuple(sorted([src, dst]))
            edge_weights[key] = edge_weights.get(key, 0) + w

    for (u, v), w in edge_weights.items():
        G.add_edge(u, v, weight=w)

    communities = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)

    class_to_cluster: Dict[str, int] = {}
    for cid, members in enumerate(communities):
        for fqn in members:
            class_to_cluster[fqn] = cid

    return class_to_cluster


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Louvain community detection baseline")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to ir-a.json (e.g. data/petclinic/ir-a.json)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: result/louvain/<project>)")
    parser.add_argument("--resolution", "-r", type=float, default=1.0,
                        help="Louvain resolution parameter (higher → more clusters)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project = IrAProject(args.input)
    print(f"[Louvain] Loaded {project.num_classes} classes, "
          f"{sum(len(v) for v in project.class_call_weights.values())} call edges")

    class_to_cluster = run_louvain(project, resolution=args.resolution, seed=args.seed)
    num_clusters = len(set(class_to_cluster.values()))
    print(f"[Louvain] Detected {num_clusters} communities (resolution={args.resolution})")

    # Print cluster summary
    from collections import Counter
    cluster_sizes = Counter(class_to_cluster.values())
    for cid in sorted(cluster_sizes.keys()):
        members = [fqn for fqn, c in class_to_cluster.items() if c == cid]
        print(f"  cluster {cid}: {cluster_sizes[cid]} classes")
        for fqn in sorted(members):
            print(f"    - {fqn.rsplit('.', 1)[-1]}")

    # Save output
    if args.output is None:
        project_name = Path(args.input).resolve().parent.name
        args.output = str(Path(__file__).resolve().parent.parent / "result" / "louvain" / project_name)
    os.makedirs(args.output, exist_ok=True)

    result = project.build_clusters_json(class_to_cluster, algorithm="Louvain")
    out_path = os.path.join(args.output, "clusters.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
