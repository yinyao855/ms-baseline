"""
KMeans baseline — class-level clustering based on SBERT semantic embeddings.

Builds a feature vector for each class from its method-level text descriptions,
then runs scikit-learn's KMeans to partition classes into K clusters.

Usage::

    python -m kmeans.main -i data/petclinic/ir-a.json -k 7
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.ir_parser import IrAProject


def _get_embedder():
    """Create a SemanticEmbedder from the mono2rest package (reuse existing code)."""
    try:
        from mono2rest.semantic_embedder import SemanticEmbedder
        return SemanticEmbedder()
    except ImportError:
        return None


def _hash_embed(text: str, dim: int = 384) -> np.ndarray:
    import hashlib
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed)
    vec = rng.randn(dim).astype(np.float32)
    vec /= np.linalg.norm(vec) + 1e-9
    return vec


def build_class_embeddings(project: IrAProject) -> np.ndarray:
    """Build an (N_classes, dim) embedding matrix for all project classes."""
    texts = [project.class_texts[fqn] for fqn in project.class_fqns]

    embedder = _get_embedder()
    if embedder is not None:
        vectors = embedder.embed_texts(texts)
    else:
        print("[WARN] Using hash-based fallback embeddings")
        vectors = np.array([_hash_embed(t) for t in texts])

    return vectors


def run_kmeans(
    project: IrAProject,
    num_clusters: int = 7,
    seed: int = 42,
) -> Dict[str, int]:
    """Run KMeans clustering on class-level semantic embeddings.

    Returns:
        {class_fqn: cluster_id}
    """
    embeddings = build_class_embeddings(project)
    k = min(num_clusters, project.num_classes)

    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(embeddings)

    return {fqn: int(labels[i]) for i, fqn in enumerate(project.class_fqns)}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="KMeans semantic clustering baseline")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to ir-a.json (e.g. data/petclinic/ir-a.json)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: result/kmeans/<project>)")
    parser.add_argument("--clusters", "-k", type=int, default=7,
                        help="Number of clusters")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project = IrAProject(args.input)
    print(f"[KMeans] Loaded {project.num_classes} classes")

    class_to_cluster = run_kmeans(
        project,
        num_clusters=args.clusters,
        seed=args.seed,
    )
    num_clusters = len(set(class_to_cluster.values()))
    print(f"[KMeans] Clustered into {num_clusters} clusters (k={args.clusters})")

    from collections import Counter
    cluster_sizes = Counter(class_to_cluster.values())
    for cid in sorted(cluster_sizes.keys()):
        members = [fqn for fqn, c in class_to_cluster.items() if c == cid]
        print(f"  cluster {cid}: {cluster_sizes[cid]} classes")
        for fqn in sorted(members):
            print(f"    - {fqn.rsplit('.', 1)[-1]}")

    if args.output is None:
        project_name = Path(args.input).resolve().parent.name
        args.output = str(Path(__file__).resolve().parent.parent / "result" / "kmeans" / project_name)
    os.makedirs(args.output, exist_ok=True)

    result = project.build_clusters_json(class_to_cluster, algorithm="KMeans")
    out_path = os.path.join(args.output, "clusters.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
