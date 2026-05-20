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


def _weighted_most_valuable_edge(G):
    """Select the edge with highest weighted betweenness centrality.

    The paper's Gephi-based Girvan-Newman uses edge weights in the
    betweenness computation.  In networkx, weight is treated as "distance"
    for shortest-path calculation: a high coupling score (=large weight)
    makes an edge "expensive" to traverse, so intra-community edges are
    avoided in shortest paths while low-coupling bridge edges accumulate
    high betweenness — exactly what Girvan-Newman needs to cut.
    """
    betweenness = nx.edge_betweenness_centrality(G, weight="weight")
    return max(betweenness, key=betweenness.get)


def _cluster_girvan_newman(
    G: nx.Graph,
    class_fqns: list[str],
    num_clusters: int,
) -> Dict[str, int]:
    """Girvan-Newman: iteratively remove highest-betweenness edges.

    Deterministic; requires the desired number of clusters as input.
    Uses weighted betweenness centrality per the paper's GephiSolver.
    """
    k = min(num_clusters, len(class_fqns))

    if G.number_of_edges() == 0:
        return {fqn: i % k for i, fqn in enumerate(class_fqns)}

    comp = girvan_newman(G, most_valuable_edge=_weighted_most_valuable_edge)
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

    # Girvan-Newman often yields more clusters than requested when the
    # initial graph has many connected components (PetClinic: 8 components
    # for K=4 target). Merge the smallest cluster into the closest cluster
    # (highest inter-cluster edge weight) until we hit K, so `-k` is real.
    class_to_cluster = _merge_down_to_k(class_to_cluster, G, k)
    return class_to_cluster


def _merge_down_to_k(
    class_to_cluster: Dict[str, int],
    G: nx.Graph,
    target_k: int,
) -> Dict[str, int]:
    """Reconcile the cluster count to target_k.

    Two-phase strategy that avoids the god-cluster trap of the original
    Girvan-Newman cuts:
      1. SPLIT phase: while any cluster holds more than 2x its fair share
         (i.e. > (N / target_k) * 2 classes), split it by running a fresh
         Girvan-Newman cut on the cluster's induced subgraph until the largest
         resulting child cluster falls below the fair-share ceiling.
      2. MERGE phase: merge the smallest cluster into its highest-coupling
         neighbour until cluster count == target_k.

    This keeps SC deterministic and still K-faithful, while making sure no
    single cluster trivially absorbs 50%+ of the classes (which would let SC
    inflate SM via the m_k^2 normalisation in the structural modularity).
    """
    if target_k <= 0:
        return class_to_cluster

    total_n = len(class_to_cluster)
    fair_share = total_n / target_k
    god_ceiling = fair_share * 2

    # 1. SPLIT god-clusters before merging anything.
    class_to_cluster = _split_god_clusters(class_to_cluster, G, god_ceiling, target_k)

    # 2. MERGE smallest -> its highest-coupling neighbour, idempotent if <= target_k.
    # NOTE: candidate target clusters are forbidden to exceed god_ceiling AFTER
    # the merge — otherwise the merge phase trivially undoes the SPLIT phase
    # by re-routing every small cluster back into the freshly cracked god.
    while True:
        members: Dict[int, list[str]] = {}
        for fqn, cid in class_to_cluster.items():
            members.setdefault(cid, []).append(fqn)
        if len(members) <= target_k:
            return class_to_cluster

        coupling: Dict[int, Dict[int, float]] = {cid: {} for cid in members}
        for u, v, data in G.edges(data=True):
            cu = class_to_cluster.get(u)
            cv = class_to_cluster.get(v)
            if cu is None or cv is None or cu == cv:
                continue
            w = float(data.get("weight", 1))
            coupling[cu][cv] = coupling[cu].get(cv, 0.0) + w
            coupling[cv][cu] = coupling[cv].get(cu, 0.0) + w

        smallest_cid = min(members, key=lambda c: (len(members[c]), c))
        small_size = len(members[smallest_cid])

        def can_absorb(cid: int) -> bool:
            return (len(members[cid]) + small_size) <= god_ceiling

        # Prefer the highest-coupling neighbour that can absorb without
        # creating a god-cluster. If none can, relax and accept any
        # neighbour (we still need to reach target_k).
        neighbours = coupling[smallest_cid]
        candidates = [(c, w) for c, w in neighbours.items() if c != smallest_cid and can_absorb(c)]
        if candidates:
            target_cid = max(candidates, key=lambda kv: kv[1])[0]
        elif neighbours:
            target_cid = max(neighbours.items(), key=lambda kv: kv[1])[0]
        else:
            other_cids = [c for c in members if c != smallest_cid]
            absorbable = [c for c in other_cids if can_absorb(c)]
            pool = absorbable if absorbable else other_cids
            target_cid = max(pool, key=lambda c: len(members[c]))

        for fqn in members[smallest_cid]:
            class_to_cluster[fqn] = target_cid


def _split_god_clusters(
    class_to_cluster: Dict[str, int],
    G: nx.Graph,
    god_ceiling: float,
    target_k: int,
) -> Dict[str, int]:
    """If any cluster holds more than `god_ceiling` classes, split it by
    running Girvan-Newman on its induced subgraph until its largest child
    falls under the ceiling (or the subgraph is too small to split further).
    """
    next_cid = max(class_to_cluster.values(), default=-1) + 1
    while True:
        members: Dict[int, list[str]] = {}
        for fqn, cid in class_to_cluster.items():
            members.setdefault(cid, []).append(fqn)

        # Find the largest cluster exceeding the god-ceiling.
        oversized = sorted(
            (cid for cid, ms in members.items() if len(ms) > god_ceiling),
            key=lambda c: -len(members[c]),
        )
        if not oversized:
            return class_to_cluster

        big_cid = oversized[0]
        big_members = members[big_cid]
        # Hard upper cap on total clusters during the split phase. Generous so
        # we never short-circuit when the initial Girvan-Newman cut already
        # produced many trivial stubs alongside a big god (PetClinic raw GN
        # for K=4 yields 8 clusters [16, 3, 1, 1, 1, 1, 1, 1] — splitting the
        # 16-class god still leaves room before this cap).
        if len(members) >= max(target_k * 4, 16):
            return class_to_cluster

        sub = G.subgraph(big_members).copy()
        if sub.number_of_edges() == 0 or sub.number_of_nodes() < 4:
            return class_to_cluster

        # One Girvan-Newman cut on the subgraph
        try:
            comp_iter = girvan_newman(sub, most_valuable_edge=_weighted_most_valuable_edge)
            first_split = next(comp_iter)
        except (StopIteration, ZeroDivisionError):
            return class_to_cluster

        if len(first_split) < 2:
            return class_to_cluster

        # Reassign: largest child keeps big_cid, others get new cids
        ordered = sorted(first_split, key=lambda s: -len(s))
        keep, others = ordered[0], ordered[1:]
        for fqn in keep:
            class_to_cluster[fqn] = big_cid
        for child in others:
            for fqn in child:
                class_to_cluster[fqn] = next_cid
            next_cid += 1


def _cluster_leung(
    G: nx.Graph,
    class_fqns: list[str],
) -> Dict[str, int]:
    """Leung's Epidemic Label Propagation (via networkx).

    Non-deterministic; automatically determines the number of clusters.
    """
    if G.number_of_edges() == 0:
        return {fqn: i for i, fqn in enumerate(class_fqns)}

    communities = label_propagation_communities(G, weight="weight")

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
