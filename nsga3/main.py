"""CLI entry point for the class-level NSGA-III pre-clustering.

Usage::

    uv run python -m nsga3.main \\
        --input  data/projects/m-petclinic/ir-a.json \\
        --output data/projects/m-petclinic/nsga3-clusters.json \\
        --base-package org.springframework.samples.petclinic
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from typing import Dict, List, Tuple

from .data_models import ClusterResult
from .ir_adapter import IrAClassAdapter
from .nsga_clustering import NSGAClassClustering
from .semantic_embedder import ClassEmbedder


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

DEFAULT_K_FLOOR = 2
DEFAULT_K_CEIL = 12


def heuristic_k(n: int) -> int:
    """K = round(sqrt(N / 2)), clamped to [2, 12]."""
    if n < 2:
        return 1
    raw = round(math.sqrt(n / 2.0))
    return max(DEFAULT_K_FLOOR, min(DEFAULT_K_CEIL, raw))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class NSGA3Pipeline:
    def __init__(
        self,
        *,
        sbert_model: str = "bert-base-nli-mean-tokens",
        population_size: int = 80,
        max_generations: int = 120,
        seed: int | None = 42,
        min_cluster_size: int = 2,
    ):
        self.sbert_model = sbert_model
        self.population_size = population_size
        self.max_generations = max_generations
        self.seed = seed
        self.min_cluster_size = min_cluster_size

    def run(
        self,
        ir_a_path: str,
        *,
        base_package: str | None = None,
        k: int | None = None,
    ) -> Dict:
        print(f"[nsga3] loading {ir_a_path}")
        adapter = IrAClassAdapter(ir_a_path, base_package=base_package)
        n = adapter.n_classes
        if n == 0:
            raise RuntimeError(
                f"no in-project classes found under base package "
                f"{adapter.base_package!r}"
            )

        target_k = k or heuristic_k(n)
        print(
            f"[nsga3] {n} classes  basePackage={adapter.base_package!r}  "
            f"K={target_k}"
        )

        print("[nsga3] embedding classes with SBERT …")
        embedder = ClassEmbedder(model_name=self.sbert_model)
        vecs = embedder.embed(adapter.nodes)
        sim = embedder.cosine_matrix(vecs)
        print(f"[nsga3] similarity matrix: {sim.shape}")

        print("[nsga3] running NSGA-III GA …")
        clusterer = NSGAClassClustering(
            population_size=self.population_size,
            max_generations=self.max_generations,
            seed=self.seed,
            min_cluster_size=self.min_cluster_size,
        )
        clusters = clusterer.cluster(adapter.nodes, adapter.adj, sim, target_k)
        for c in clusters:
            print(
                f"  cluster {c.cluster_id}: {len(c.classes)} classes  "
                f"coupling={c.coupling:.3f}  cohesion={c.cohesion:.3f}  "
                f"semSim={c.semantic_similarity:.3f}"
            )

        named = _name_clusters(clusters, adapter.base_package)
        return {
            "basePackage": adapter.base_package,
            "projectId": adapter.project_id,
            "algorithm": "nsga3",
            "k": target_k,
            "clusters": [c.to_dict() for c in named],
            "sharedClasses": [],
        }


# ---------------------------------------------------------------------------
# Cluster naming
# ---------------------------------------------------------------------------

_SUFFIXES = (
    "Controller",
    "RestController",
    "Resource",
    "Service",
    "ServiceImpl",
    "Repository",
    "Dao",
    "DAO",
    "Mapper",
    "Manager",
    "Facade",
    "Helper",
    "Util",
    "Utils",
    "Bean",
    "Entity",
    "DTO",
    "Dto",
    "Vo",
    "VO",
)


def _strip_suffix(name: str) -> str:
    for suf in _SUFFIXES:
        if name.endswith(suf) and len(name) > len(suf):
            return name[: -len(suf)]
    return name


def _kebab(name: str) -> str:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1-\2", name)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", s)
    return s.lower().strip("-") or "service"


def _name_clusters(
    clusters: List[ClusterResult], base_package: str
) -> List[ClusterResult]:
    """Assign a unique kebab-case ``<domain>-service`` name per cluster."""
    used: Counter = Counter()
    named: List[ClusterResult] = []
    for c in clusters:
        # Pick the most common domain token from class simple names.
        simple = [fqn.rsplit(".", 1)[-1] for fqn in c.classes]
        domains = [_strip_suffix(s) for s in simple if _strip_suffix(s)]
        if not domains:
            domain = f"cluster-{c.cluster_id}"
        else:
            domain = Counter(domains).most_common(1)[0][0]
        kebab = _kebab(domain)
        if kebab.endswith("-service"):
            kebab = kebab[: -len("-service")]
        base = f"{kebab}-service"
        used[base] += 1
        suffix = "" if used[base] == 1 else f"-{used[base]}"
        c.name = base + suffix
        c.reason = (
            f"NSGA-III cluster {c.cluster_id}: {len(c.classes)} classes "
            f"(coupling={c.coupling:.3f}, cohesion={c.cohesion:.3f}, "
            f"semSim={c.semantic_similarity:.3f})"
        )
        named.append(c)
    return named


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="nsga3",
        description="Class-level NSGA-III microservice pre-clustering",
    )
    parser.add_argument("--input", "-i", required=True, help="path to ir-a.json")
    parser.add_argument(
        "--output", "-o", required=True, help="path to write clusters.json"
    )
    parser.add_argument(
        "--base-package", "-b", default=None,
        help="base Java package (auto-detected from ir-a.json if omitted)",
    )
    parser.add_argument(
        "--k", "-k", type=int, default=None,
        help="target number of clusters (default: round(sqrt(N/2)))",
    )
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--generations", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-cluster-size", type=int, default=2,
        help="dissolve clusters smaller than this into their best-connected "
             "neighbour after the GA finishes (set to 1 to disable)",
    )
    parser.add_argument(
        "--sbert-model",
        default="bert-base-nli-mean-tokens",
        help="SBERT model name (huggingface id)",
    )
    args = parser.parse_args(argv)

    pipeline = NSGA3Pipeline(
        sbert_model=args.sbert_model,
        population_size=args.population,
        max_generations=args.generations,
        seed=args.seed,
        min_cluster_size=args.min_cluster_size,
    )
    result = pipeline.run(
        args.input, base_package=args.base_package, k=args.k
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[nsga3] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(run())
