"""
Clustering quality metrics for microservice decomposition baselines.

Four metrics (all computed on the class-level call graph):

- **ICP** (Inter-partition Call Percentage): median ratio of cross-partition
  call pairs — lower is better.
- **SM** (Structural Modularity): avg cohesion − avg coupling — higher is better.
- **IFN** (Interface Number): avg number of classes per partition that are
  called from other partitions — lower is better.
- **NED** (Non-Extreme Distribution): fraction of classes in non-extreme
  partitions — higher is better.

Usage::

    # Evaluate a single baseline result
    python -m common.metrics -i data/petclinic/ir-a.json -c result/louvain/petclinic/clusters.json

    # Evaluate all baselines on all projects
    python -m common.metrics --all
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np

from .ir_parser import IrAProject


# ---------------------------------------------------------------------------
# Build networkx graph from IrAProject
# ---------------------------------------------------------------------------

def build_call_graph(project: IrAProject) -> nx.DiGraph:
    """Build a directed call graph from IrAProject's class_call_weights."""
    G = nx.DiGraph()
    G.add_nodes_from(project.class_fqns)
    for src, targets in project.class_call_weights.items():
        for dst, weight in targets.items():
            G.add_edge(src, dst, weight=weight)
    return G


def load_partitions(clusters_json_path: str) -> Dict[str, List[str]]:
    """Load a clusters.json file and return {partition_id: [class_fqn, ...]}."""
    with open(clusters_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    partitions: Dict[str, List[str]] = {}
    for entry in data.get("clusters", []):
        pid = str(entry.get("id", entry.get("name", "")))
        partitions[pid] = entry.get("classes", [])
    return partitions


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def cal_icp(G: nx.DiGraph, partitions: Dict[str, List[str]]) -> float:
    """Inter-partition Call Percentage (lower is better).

    Median of cross-partition call-pair ratios.
    """
    class_to_part = {}
    for pid, classes in partitions.items():
        for cls in classes:
            class_to_part[cls] = pid

    sub_nodes = set(class_to_part.keys())
    subG = G.subgraph(sub_nodes)

    cross_calls: Dict[Tuple[str, str], int] = defaultdict(int)
    total_cross = 0
    for u, v in subG.edges():
        pu, pv = class_to_part.get(u), class_to_part.get(v)
        if pu and pv and pu != pv:
            cross_calls[(pu, pv)] += 1
            total_cross += 1

    if total_cross == 0:
        return 0.0

    ratios = [cnt / total_cross for cnt in cross_calls.values()]
    return float(np.median(ratios))


def cal_sm(G: nx.DiGraph, partitions: Dict[str, List[str]]) -> float:
    """Structural Modularity (higher is better).

    SM = avg_cohesion − avg_coupling
    """
    class_to_part = {}
    part_sizes: Dict[str, int] = {}
    for pid, classes in partitions.items():
        part_sizes[pid] = len(classes)
        for cls in classes:
            class_to_part[cls] = pid

    sub_nodes = set(class_to_part.keys())
    subG = G.subgraph(sub_nodes)
    pids = list(partitions.keys())
    pid_idx = {p: i for i, p in enumerate(pids)}
    M = len(pids)

    # Cohesion: for each partition, intra_edges / size^2
    intra_edges: Dict[str, int] = defaultdict(int)
    inter_edges = np.zeros((M, M), dtype=int)
    for u, v in subG.edges():
        pu, pv = class_to_part.get(u), class_to_part.get(v)
        if not pu or not pv:
            continue
        if pu == pv:
            intra_edges[pu] += 1
        else:
            inter_edges[pid_idx[pu]][pid_idx[pv]] += 1

    scoh_list = []
    for pid in pids:
        m_i = part_sizes[pid]
        if m_i <= 1:
            scoh_list.append(0.0)
        else:
            scoh_list.append(intra_edges[pid] / (m_i ** 2))
    avg_scoh = float(np.mean(scoh_list))

    # Coupling: for each pair, (edges_ij + edges_ji) / (2 * (size_i + size_j))
    scop_list = []
    for i in range(M):
        for j in range(i + 1, M):
            m1, m2 = part_sizes[pids[i]], part_sizes[pids[j]]
            denom = 2 * (m1 + m2)
            if denom == 0:
                scop_list.append(0.0)
            else:
                scop_list.append((inter_edges[i][j] + inter_edges[j][i]) / denom)
    num_pairs = M * (M - 1) / 2
    avg_scop = float(np.sum(scop_list) / num_pairs) if num_pairs > 0 else 0.0

    return avg_scoh - avg_scop


def cal_ifn(G: nx.DiGraph, partitions: Dict[str, List[str]]) -> float:
    """Average Interface Number per partition (lower is better).

    A class is an "interface" if it is called from another partition.
    """
    class_to_part = {}
    for pid, classes in partitions.items():
        for cls in classes:
            class_to_part[cls] = pid

    sub_nodes = set(class_to_part.keys())
    subG = G.subgraph(sub_nodes)

    part_interfaces: Dict[str, set] = {pid: set() for pid in partitions}
    for u, v in subG.edges():
        pu, pv = class_to_part.get(u), class_to_part.get(v)
        if pu and pv and pu != pv:
            part_interfaces[pv].add(v)

    ifn_list = [len(ifaces) for ifaces in part_interfaces.values()]
    return float(np.mean(ifn_list))


def cal_ned(partitions: Dict[str, List[str]],
            min_threshold: int = 5, max_threshold: int = 20) -> float:
    """Non-Extreme Distribution (higher is better).

    Fraction of classes in partitions with size in [min_threshold, max_threshold].
    """
    total = sum(len(c) for c in partitions.values())
    if total == 0:
        return 0.0

    non_extreme = sum(
        len(c) for c in partitions.values()
        if min_threshold <= len(c) <= max_threshold
    )
    return non_extreme / total


# ---------------------------------------------------------------------------
# Evaluate helper
# ---------------------------------------------------------------------------

def evaluate(
    ir_a_path: str,
    clusters_json_path: str,
    ned_min: int = 5,
    ned_max: int = 20,
) -> Dict[str, float]:
    """Compute all four metrics for a given ir-a.json + clusters.json pair."""
    project = IrAProject(ir_a_path)
    G = build_call_graph(project)
    partitions = load_partitions(clusters_json_path)

    return {
        "ICP": cal_icp(G, partitions),
        "SM": cal_sm(G, partitions),
        "IFN": cal_ifn(G, partitions),
        "NED": cal_ned(partitions, min_threshold=ned_min, max_threshold=ned_max),
        "num_clusters": len(partitions),
        "num_classes": project.num_classes,
    }


def format_metrics(metrics: Dict[str, float]) -> str:
    """Format metrics dict as a readable string."""
    return (
        f"  Clusters:  {int(metrics['num_clusters']):>3d}   "
        f"Classes: {int(metrics['num_classes']):>3d}\n"
        f"  ICP:       {metrics['ICP']:.4f}   (lower is better)\n"
        f"  SM:        {metrics['SM']:.4f}   (higher is better)\n"
        f"  IFN:       {metrics['IFN']:.4f}   (lower is better)\n"
        f"  NED:       {metrics['NED']:.4f}   (higher is better)"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate microservice decomposition quality metrics")
    parser.add_argument("--input", "-i", help="Path to ir-a.json")
    parser.add_argument("--clusters", "-c", help="Path to clusters.json")
    parser.add_argument("--all", action="store_true",
                        help="Evaluate all baselines on all projects under data/ and result/")
    parser.add_argument("--ned-min", type=int, default=5,
                        help="NED minimum partition size threshold")
    parser.add_argument("--ned-max", type=int, default=20,
                        help="NED maximum partition size threshold")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent

    if args.all:
        _run_all(base_dir, args.ned_min, args.ned_max)
    elif args.input and args.clusters:
        metrics = evaluate(args.input, args.clusters, args.ned_min, args.ned_max)
        print(format_metrics(metrics))
    else:
        parser.error("Provide --input and --clusters, or use --all")


def _run_all(base_dir: Path, ned_min: int, ned_max: int):
    """Discover and evaluate all baseline × project combinations."""
    data_dir = base_dir / "data"
    result_dir = base_dir / "result"

    projects = sorted(p.name for p in data_dir.iterdir() if p.is_dir()) if data_dir.exists() else []
    baselines = sorted(b.name for b in result_dir.iterdir() if b.is_dir()) if result_dir.exists() else []

    if not projects or not baselines:
        print("No projects in data/ or no results in result/. Run baselines first.")
        return

    # Collect all results for table output
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for project in projects:
        ir_a_path = data_dir / project / "ir-a.json"
        if not ir_a_path.exists():
            continue

        print(f"\n{'=' * 70}")
        print(f"  Project: {project}")
        print(f"{'=' * 70}")

        all_results[project] = {}
        for baseline in baselines:
            clusters_path = result_dir / baseline / project / "clusters.json"
            if not clusters_path.exists():
                continue

            metrics = evaluate(str(ir_a_path), str(clusters_path), ned_min, ned_max)
            all_results[project][baseline] = metrics

            print(f"\n  [{baseline}]")
            print(format_metrics(metrics))

    # Print summary comparison table
    if all_results:
        _print_summary_table(all_results)


def _print_summary_table(all_results: Dict[str, Dict[str, Dict[str, float]]]):
    """Print a compact comparison table across all projects and baselines."""
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY TABLE")
    print(f"{'=' * 70}\n")

    # Gather all baseline names
    all_baselines = sorted({b for proj in all_results.values() for b in proj})
    metric_names = ["ICP", "SM", "IFN", "NED"]
    directions = {"ICP": "↓", "SM": "↑", "IFN": "↓", "NED": "↑"}

    for project, baselines in sorted(all_results.items()):
        print(f"  {project}:")

        # Header
        header = f"    {'Baseline':<15s} {'K':>3s}"
        for m in metric_names:
            header += f"  {m+directions[m]:>8s}"
        print(header)
        print(f"    {'-' * (15 + 3 + 10 * len(metric_names))}")

        for baseline in all_baselines:
            if baseline not in baselines:
                continue
            metrics = baselines[baseline]
            row = f"    {baseline:<15s} {int(metrics['num_clusters']):>3d}"
            for m in metric_names:
                row += f"  {metrics[m]:>8.4f}"
            print(row)
        print()


if __name__ == "__main__":
    main()
