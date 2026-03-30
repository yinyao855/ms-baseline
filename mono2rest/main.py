"""
MONO2REST main entry point.

Usage::

    python -m mono2rest.main --input ir-a.json --clusters 7 --output out/
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from typing import Dict, List

from .ir_adapter import IrAAdapter
from .semantic_embedder import SemanticEmbedder
from .nsga_clustering import NSGAClustering
from .rest_api_generator import RESTAPIGenerator
from .data_models import Cluster, Method, RESTEndpoint


class MONO2REST:
    """Two-phase framework: microservice identification → REST API generation."""

    def __init__(self, config: Dict | None = None):
        self.config = config or {}

    def run(self, ir_a_path: str) -> Dict:
        print("=" * 60)
        print("MONO2REST — starting")
        print("=" * 60)

        # ---- Load IR-A ----
        print("\n[Load] Parsing ir-a.json …")
        adapter = IrAAdapter(ir_a_path)
        methods = adapter.methods
        call_graph = adapter.call_graph
        print(f"  {len(methods)} methods, {sum(len(v) for v in call_graph.values())} call edges")

        # ---- Phase 1: Microservice identification ----
        print("\n[Phase 1] Microservice identification")

        print("  [1/3] Generating SBERT embeddings …")
        model_name = self.config.get("sbert_model", "bert-base-nli-mean-tokens")
        embedder = SemanticEmbedder(model_name=model_name)
        embeddings = embedder.embed_methods(methods)
        sim_matrix = embedder.build_similarity_matrix(methods, embeddings)
        print(f"    similarity matrix: {sim_matrix.shape}")

        print("  [2/3] Running NSGA-III clustering …")
        num_clusters = self.config.get("num_clusters", 7)
        clustering = NSGAClustering(
            population_size=self.config.get("population_size", 100),
            max_generations=self.config.get("max_generations", 100),
            crossover_rate=self.config.get("crossover_rate", 0.8),
            mutation_rate=self.config.get("mutation_rate", 0.1),
        )
        clusters = clustering.cluster(
            methods=methods,
            call_graph=call_graph,
            sim_matrix=sim_matrix,
            num_clusters=num_clusters,
        )
        print(f"  [3/3] Identified {len(clusters)} clusters")
        for c in clusters:
            print(f"    cluster {c.cluster_id}: {len(c.methods)} methods  "
                  f"coupling={c.coupling:.3f}  cohesion={c.cohesion:.3f}  "
                  f"semSim={c.semantic_similarity:.3f}")

        # ---- Phase 2: REST API generation ----
        print("\n[Phase 2] REST API generation")
        api_gen = RESTAPIGenerator(embedder=embedder)

        print("  [1/3] Selecting exposed methods …")
        exposed = api_gen.filter_exposed_methods(clusters, call_graph)
        print(f"    {len(exposed)} methods to expose")

        print("  [2/3] Assigning HTTP methods …")
        assignments = api_gen.assign_http_methods(exposed)

        print("  [3/3] Generating URIs …")
        endpoints = api_gen.generate_endpoints(assignments, clusters)
        for ep in endpoints[:8]:
            print(f"    {ep.http_method.value:6s} {ep.uri}")
        if len(endpoints) > 8:
            print(f"    … and {len(endpoints) - 8} more")

        # ---- Build result ----
        result = self._build_result(clusters, endpoints, methods, adapter)

        print("\n" + "=" * 60)
        print("MONO2REST — done")
        print("=" * 60)
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        clusters: List[Cluster],
        endpoints: List[RESTEndpoint],
        methods: List[Method],
        adapter: IrAAdapter,
    ) -> Dict:
        # Method-level result
        method_result = {
            "clusters": [c.to_dict() for c in clusters],
            "rest_endpoints": [ep.to_dict() for ep in endpoints],
            "summary": {
                "total_methods": len(methods),
                "total_clusters": len(clusters),
                "total_rest_endpoints": len(endpoints),
            },
        }

        # Class-level clusters.json (compatible with ServiceClusterConfig)
        clusters_json = _method_to_class_clusters(clusters, adapter)

        return {
            "method_level": method_result,
            "clusters_json": clusters_json,
        }


def _method_to_class_clusters(
    clusters: List[Cluster], adapter: IrAAdapter
) -> Dict:
    """Convert method-level clustering into class-level ServiceClusterConfig."""
    # class_fqn → {cluster_id: count}
    class_votes: Dict[str, Counter] = defaultdict(Counter)
    for c in clusters:
        for m in c.methods:
            class_votes[m.class_fqn][c.cluster_id] += 1

    # Assign each class to its majority cluster
    class_to_cluster: Dict[str, int] = {}
    shared_classes = []
    for fqn, votes in class_votes.items():
        most_common = votes.most_common()
        winner_id = most_common[0][0]
        class_to_cluster[fqn] = winner_id
        if len(most_common) > 1:
            total = sum(votes.values())
            top_share = most_common[0][1] / total
            if top_share < 0.8:
                shared_classes.append({
                    "fqn": fqn,
                    "strategy": "SPLIT",
                    "reason": f"Methods split across clusters: {dict(votes)}",
                    "detail": "; ".join(
                        f"cluster-{cid}: {cnt} methods"
                        for cid, cnt in most_common
                    ),
                })

    # Build cluster entries
    cluster_entries: Dict[int, List[str]] = defaultdict(list)
    for fqn, cid in class_to_cluster.items():
        cluster_entries[cid].append(fqn)

    entries = []
    for cid in sorted(cluster_entries.keys()):
        classes = sorted(cluster_entries[cid])
        # Derive a name from the most common simple class name
        simple_names = [c.rsplit(".", 1)[-1] for c in classes]
        dominant = Counter(simple_names).most_common(1)[0][0]
        from .rest_api_generator import _strip_suffix_and_kebab
        name = _strip_suffix_and_kebab(dominant) + "-service"
        entries.append({
            "id": cid - 1,
            "name": name,
            "reason": f"NSGA-III cluster {cid}",
            "classes": classes,
        })

    return {
        "relativePath": adapter.raw.get("projectId", ""),
        "clusters": entries,
        "sharedClasses": shared_classes,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MONO2REST baseline")
    parser.add_argument("--input", "-i", required=True, help="Path to ir-a.json")
    parser.add_argument("--clusters", "-k", type=int, default=7, help="Number of target clusters")
    parser.add_argument("--output", "-o", default=".", help="Output directory")
    parser.add_argument("--generations", "-g", type=int, default=100, help="NSGA-III generations")
    parser.add_argument("--population", "-p", type=int, default=100, help="Population size")
    args = parser.parse_args()

    config = {
        "num_clusters": args.clusters,
        "max_generations": args.generations,
        "population_size": args.population,
    }
    mono = MONO2REST(config)
    result = mono.run(args.input)

    os.makedirs(args.output, exist_ok=True)

    # Save method-level result
    with open(os.path.join(args.output, "mono2rest_result.json"), "w", encoding="utf-8") as f:
        json.dump(result["method_level"], f, indent=2, ensure_ascii=False)

    # Save class-level clusters.json
    with open(os.path.join(args.output, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(result["clusters_json"], f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
