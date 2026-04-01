"""
Service Cutter baseline — multi-criteria weighted graph clustering.

Implements the Service Cutter approach (Gysel et al., ESOCC 2016) adapted
for static code analysis via IR-A.  Builds an undirected weighted graph from
multiple coupling criteria and clusters it with either Girvan-Newman
(deterministic, requires K) or Leung's Epidemic Label Propagation
(non-deterministic, auto-determines K).

Usage::

    python -m service_cutter.main -i data/petclinic/ir-a.json
    python -m service_cutter.main -i data/petclinic/ir-a.json --algorithm leung
    python -m service_cutter.main -i data/petclinic/ir-a.json -k 5
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Dict

import networkx as nx
from networkx.algorithms.community import girvan_newman, label_propagation_communities

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.ir_parser import IrAProject
from service_cutter.coupling import CouplingScorer, DEFAULT_PRIORITIES


def run_service_cutter(
    project: IrAProject,
    algorithm: str = "girvan_newman",
    num_clusters: int = 7,
    priorities: Dict[str, float] | None = None,
    seed: int = 42,
) -> Dict[str, int]:
    """Run Service Cutter clustering on the class-level weighted graph.

    Args:
        project: Parsed IR-A project.
        algorithm: "girvan_newman" or "leung".
        num_clusters: Target cluster count (Girvan-Newman only).
        priorities: Coupling criteria priority weights.
        seed: Random seed (Leung only).

    Returns:
        {class_fqn: cluster_id}
    """
    scorer = CouplingScorer(project, priorities)
    G = scorer.build_weighted_graph()

    if algorithm == "leung":
        return _cluster_leung(G, project.class_fqns)
    else:
        return _cluster_girvan_newman(G, project.class_fqns, num_clusters)


def _cluster_girvan_newman(
    G: nx.Graph,
    class_fqns: list[str],
    num_clusters: int,
) -> Dict[str, int]:
    """Girvan-Newman: iteratively remove highest-betweenness edges.

    Deterministic; requires the desired number of clusters as input.
    """
    k = min(num_clusters, len(class_fqns))

    if G.number_of_edges() == 0:
        return {fqn: i % k for i, fqn in enumerate(class_fqns)}

    comp = girvan_newman(G)
    for communities in islice(comp, None):
        if len(communities) >= k:
            break

    class_to_cluster: Dict[str, int] = {}
    assigned = set()
    for cid, members in enumerate(communities):
        for fqn in members:
            class_to_cluster[fqn] = cid
            assigned.add(fqn)

    # Assign any isolated nodes not covered by the algorithm
    next_cid = len(communities)
    for fqn in class_fqns:
        if fqn not in assigned:
            class_to_cluster[fqn] = next_cid
            next_cid += 1

    return class_to_cluster


def _cluster_leung(
    G: nx.Graph,
    class_fqns: list[str],
) -> Dict[str, int]:
    """Leung's Epidemic Label Propagation (via networkx).

    Non-deterministic; automatically determines the number of clusters.
    """
    if G.number_of_edges() == 0:
        return {fqn: i for i, fqn in enumerate(class_fqns)}

    communities = label_propagation_communities(G)

    class_to_cluster: Dict[str, int] = {}
    assigned = set()
    for cid, members in enumerate(communities):
        for fqn in members:
            class_to_cluster[fqn] = cid
            assigned.add(fqn)

    next_cid = cid + 1 if class_to_cluster else 0
    for fqn in class_fqns:
        if fqn not in assigned:
            class_to_cluster[fqn] = next_cid
            next_cid += 1

    return class_to_cluster


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Service Cutter baseline (multi-criteria weighted graph clustering)")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to ir-a.json (e.g. data/petclinic/ir-a.json)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: result/service_cutter/<project>)")
    parser.add_argument("--algorithm", "-a", default="girvan_newman",
                        choices=["girvan_newman", "leung"],
                        help="Clustering algorithm (default: girvan_newman)")
    parser.add_argument("--clusters", "-k", type=int, default=7,
                        help="Number of clusters (Girvan-Newman only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project = IrAProject(args.input)
    num_edges = sum(len(v) for v in project.class_call_weights.values())
    print(f"[ServiceCutter] Loaded {project.num_classes} classes, {num_edges} call edges")
    print(f"[ServiceCutter] Algorithm: {args.algorithm}")

    class_to_cluster = run_service_cutter(
        project,
        algorithm=args.algorithm,
        num_clusters=args.clusters,
        seed=args.seed,
    )
    num_clusters = len(set(class_to_cluster.values()))
    print(f"[ServiceCutter] Clustered into {num_clusters} clusters")

    cluster_sizes = Counter(class_to_cluster.values())
    for cid in sorted(cluster_sizes.keys()):
        members = [fqn for fqn, c in class_to_cluster.items() if c == cid]
        print(f"  cluster {cid}: {cluster_sizes[cid]} classes")
        for fqn in sorted(members):
            print(f"    - {fqn.rsplit('.', 1)[-1]}")

    if args.output is None:
        project_name = Path(args.input).resolve().parent.name
        args.output = str(
            Path(__file__).resolve().parent.parent / "result" / "service_cutter" / project_name
        )
    os.makedirs(args.output, exist_ok=True)

    algo_label = "ServiceCutter-GN" if args.algorithm == "girvan_newman" else "ServiceCutter-Leung"
    result = project.build_clusters_json(class_to_cluster, algorithm=algo_label)
    out_path = os.path.join(args.output, "clusters.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
