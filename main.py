"""
ms-baseline — run all baselines on all projects, then evaluate metrics.

Usage::

    python main.py                     # run all baselines + metrics
    python main.py --metrics-only      # only evaluate existing results
    python main.py -k 7               # specify cluster count for kmeans/mono2rest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULT_DIR = BASE_DIR / "result"

PROJECTS = ["petclinic", "jpetstore", "daytrader", "plants"]


def run_louvain(ir_a_path: str, output_dir: str, resolution: float = 1.0):
    from common.ir_parser import IrAProject
    from louvain.main import run_louvain as _run

    project = IrAProject(ir_a_path)
    class_to_cluster = _run(project, resolution=resolution)
    num_clusters = len(set(class_to_cluster.values()))
    print(f"    {num_clusters} clusters detected")

    os.makedirs(output_dir, exist_ok=True)
    result = project.build_clusters_json(class_to_cluster, algorithm="Louvain")
    with open(os.path.join(output_dir, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def run_kmeans(ir_a_path: str, output_dir: str, num_clusters: int = 7):
    from common.ir_parser import IrAProject
    from kmeans.main import run_kmeans as _run

    project = IrAProject(ir_a_path)
    class_to_cluster = _run(project, num_clusters=num_clusters)
    print(f"    {len(set(class_to_cluster.values()))} clusters")

    os.makedirs(output_dir, exist_ok=True)
    result = project.build_clusters_json(class_to_cluster, algorithm="KMeans")
    with open(os.path.join(output_dir, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def run_mono2rest(ir_a_path: str, output_dir: str, num_clusters: int = 7,
                  generations: int = 100, population: int = 100):
    from mono2rest.main import MONO2REST

    config = {
        "num_clusters": num_clusters,
        "max_generations": generations,
        "population_size": population,
    }
    mono = MONO2REST(config)
    result = mono.run(ir_a_path)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "mono2rest_result.json"), "w", encoding="utf-8") as f:
        json.dump(result["method_level"], f, indent=2, ensure_ascii=False)
    with open(os.path.join(output_dir, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(result["clusters_json"], f, indent=2, ensure_ascii=False)


def run_all_baselines(num_clusters: int, resolution: float,
                      generations: int, population: int,
                      skip_baselines: list[str] | None = None):
    skip = set(skip_baselines or [])

    for project_name in PROJECTS:
        ir_a_path = str(DATA_DIR / project_name / "ir-a.json")
        if not os.path.exists(ir_a_path):
            print(f"[SKIP] {project_name}: ir-a.json not found")
            continue

        print(f"\n{'=' * 60}")
        print(f"  Project: {project_name}")
        print(f"{'=' * 60}")

        if "louvain" not in skip:
            print(f"\n  [Louvain] running ...")
            out = str(RESULT_DIR / "louvain" / project_name)
            try:
                run_louvain(ir_a_path, out, resolution=resolution)
            except Exception as e:
                print(f"    ERROR: {e}")

        if "kmeans" not in skip:
            print(f"\n  [KMeans] running (k={num_clusters}) ...")
            out = str(RESULT_DIR / "kmeans" / project_name)
            try:
                run_kmeans(ir_a_path, out, num_clusters=num_clusters)
            except Exception as e:
                print(f"    ERROR: {e}")

        if "mono2rest" not in skip:
            print(f"\n  [MONO2REST] running (k={num_clusters}, gen={generations}, pop={population}) ...")
            out = str(RESULT_DIR / "mono2rest" / project_name)
            try:
                run_mono2rest(ir_a_path, out, num_clusters=num_clusters,
                              generations=generations, population=population)
            except Exception as e:
                print(f"    ERROR: {e}")


def run_metrics():
    from common.metrics import evaluate, format_metrics

    all_results: dict[str, dict[str, dict[str, float]]] = {}

    for project_name in PROJECTS:
        ir_a_path = str(DATA_DIR / project_name / "ir-a.json")
        if not os.path.exists(ir_a_path):
            continue

        all_results[project_name] = {}
        baselines = sorted(
            b.name for b in RESULT_DIR.iterdir() if b.is_dir()
        ) if RESULT_DIR.exists() else []

        for baseline in baselines:
            clusters_path = RESULT_DIR / baseline / project_name / "clusters.json"
            if not clusters_path.exists():
                continue
            metrics = evaluate(ir_a_path, str(clusters_path))
            all_results[project_name][baseline] = metrics

    # Print per-project details
    for project_name, baselines in sorted(all_results.items()):
        print(f"\n{'=' * 60}")
        print(f"  Project: {project_name}")
        print(f"{'=' * 60}")
        for baseline, metrics in sorted(baselines.items()):
            print(f"\n  [{baseline}]")
            print(format_metrics(metrics))

    # Print summary table
    if all_results:
        _print_summary(all_results)


def _print_summary(all_results: dict[str, dict[str, dict[str, float]]]):
    print(f"\n\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}\n")

    all_baselines = sorted({b for proj in all_results.values() for b in proj})
    metric_names = ["ICP", "SM", "IFN", "NED"]
    directions = {"ICP": "↓", "SM": "↑", "IFN": "↓", "NED": "↑"}

    for project_name, baselines in sorted(all_results.items()):
        print(f"  {project_name}:")
        header = f"    {'Baseline':<15s} {'K':>3s}"
        for m in metric_names:
            header += f"  {m + directions[m]:>8s}"
        print(header)
        print(f"    {'-' * 55}")

        for baseline in all_baselines:
            if baseline not in baselines:
                continue
            metrics = baselines[baseline]
            row = f"    {baseline:<15s} {int(metrics['num_clusters']):>3d}"
            for m in metric_names:
                row += f"  {metrics[m]:>8.4f}"
            print(row)
        print()


def main():
    parser = argparse.ArgumentParser(description="Run all ms-baseline experiments")
    parser.add_argument("--metrics-only", action="store_true",
                        help="Skip running baselines, only evaluate existing results")
    parser.add_argument("-k", "--clusters", type=int, default=7,
                        help="Number of clusters for KMeans and MONO2REST")
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Louvain resolution parameter")
    parser.add_argument("--generations", "-g", type=int, default=100,
                        help="NSGA-III generations for MONO2REST")
    parser.add_argument("--population", "-p", type=int, default=100,
                        help="Population size for MONO2REST")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["louvain", "kmeans", "mono2rest"],
                        help="Skip specific baselines")
    args = parser.parse_args()

    if not args.metrics_only:
        print("=" * 70)
        print("  RUNNING ALL BASELINES")
        print("=" * 70)
        run_all_baselines(
            num_clusters=args.clusters,
            resolution=args.resolution,
            generations=args.generations,
            population=args.population,
            skip_baselines=args.skip,
        )

    print("\n\n" + "=" * 70)
    print("  EVALUATING METRICS")
    print("=" * 70)
    run_metrics()


if __name__ == "__main__":
    main()
