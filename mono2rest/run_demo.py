"""
MONO2REST demo — runs the full pipeline on the PetClinic ir-a.json.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mono2rest.main import MONO2REST


def main():
    import argparse
    parser = argparse.ArgumentParser(description="MONO2REST demo")
    data_dir = Path(__file__).resolve().parent.parent / "data"
    parser.add_argument(
        "--input", "-i",
        default=str(data_dir / "petclinic" / "ir-a.json"),
        help="Path to ir-a.json (default: data/petclinic/ir-a.json)",
    )
    parser.add_argument("--clusters", "-k", type=int, default=7)
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: result/mono2rest/<project>)")
    parser.add_argument("--generations", "-g", type=int, default=100)
    parser.add_argument("--population", "-p", type=int, default=100)
    args = parser.parse_args()

    config = {
        "num_clusters": args.clusters,
        "max_generations": args.generations,
        "population_size": args.population,
    }
    mono = MONO2REST(config)
    result = mono.run(args.input)

    if args.output is None:
        project_name = Path(args.input).resolve().parent.name
        args.output = str(Path(__file__).resolve().parent.parent / "result" / "mono2rest" / project_name)
    os.makedirs(args.output, exist_ok=True)

    with open(os.path.join(args.output, "mono2rest_result.json"), "w", encoding="utf-8") as f:
        json.dump(result["method_level"], f, indent=2, ensure_ascii=False)

    with open(os.path.join(args.output, "clusters.json"), "w", encoding="utf-8") as f:
        json.dump(result["clusters_json"], f, indent=2, ensure_ascii=False)

    summary = result["method_level"]["summary"]
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Total methods:       {summary['total_methods']}")
    print(f"  Identified clusters: {summary['total_clusters']}")
    print(f"  REST endpoints:      {summary['total_rest_endpoints']}")

    print("\nCluster breakdown:")
    for c in result["method_level"]["clusters"]:
        print(f"  cluster {c['cluster_id']}: {len(c['methods'])} methods  "
              f"coupling={c['metrics']['coupling']:.3f}  "
              f"cohesion={c['metrics']['cohesion']:.3f}  "
              f"semSim={c['metrics']['semantic_similarity']:.3f}")

    print("\nREST endpoints (first 15):")
    for i, ep in enumerate(result["method_level"]["rest_endpoints"][:15]):
        print(f"  {i+1:>2}. {ep['http_method']:6s} {ep['uri']}")
        print(f"      → {ep['method']['class_name']}.{ep['method']['name']}")

    print(f"\nClass-level clusters.json has {len(result['clusters_json']['clusters'])} clusters")
    if result["clusters_json"]["sharedClasses"]:
        print(f"  {len(result['clusters_json']['sharedClasses'])} shared classes detected")

    print(f"\nResults saved to: {args.output}/")


if __name__ == "__main__":
    main()
