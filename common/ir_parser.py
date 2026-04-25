"""
Shared IR-A parser — provides class-level data structures for all baselines.

Reads ir-a.json and exposes:
  - class_fqns: list of all project class FQNs
  - class_call_graph: class-to-class weighted directed call count
  - class_texts: text descriptions for embedding (method names + class name)
  - build_clusters_json(): format output as clusters.json
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


class IrAProject:
    """Class-level view of a monolithic Java project parsed from ir-a.json."""

    def __init__(self, ir_a_path: str):
        from .schema_migration import coerce_legacy_method_schema

        with open(ir_a_path, "r", encoding="utf-8") as f:
            self.raw: dict = json.load(f)
        # ir-a.json switched to a structured method schema in v1.0.0; back-fill
        # the legacy field names so kmeans/louvain/service_cutter keep working.
        coerce_legacy_method_schema(self.raw)

        self.project_id: str = self.raw.get("projectId", "")
        self.base_package = self._infer_base_package()

        self._interface_to_impls: Dict[str, List[str]] = defaultdict(list)

        self.class_fqns: List[str] = []
        self.class_methods: Dict[str, List[dict]] = defaultdict(list)
        self.class_texts: Dict[str, str] = {}

        # class FQN → class FQN → call weight
        self.class_call_weights: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

        self._build()

    @property
    def num_classes(self) -> int:
        return len(self.class_fqns)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _infer_base_package(self) -> str:
        packages = [
            c.get("packageName", "")
            for c in self.raw.get("classes", [])
            if c.get("packageName")
        ]
        if not packages:
            return ""
        common = packages[0]
        for pkg in packages[1:]:
            while not pkg.startswith(common):
                dot = common.rfind(".")
                if dot < 0:
                    common = ""
                    break
                common = common[:dot]
        return common

    def _build(self):
        class_map: Dict[str, dict] = {}
        for cls in self.raw.get("classes", []):
            fqn = cls.get("fqn", "")
            if not fqn or not fqn.startswith(self.base_package):
                continue
            class_map[fqn] = cls
            for iface in cls.get("implementsTypes", []):
                raw_iface = iface.split("<")[0].strip()
                if raw_iface.startswith(self.base_package):
                    self._interface_to_impls[raw_iface].append(fqn)

        self.class_fqns = sorted(class_map.keys())

        # Extract methods per class and build text descriptions
        for fqn in self.class_fqns:
            cls = class_map[fqn]
            simple_name = fqn.rsplit(".", 1)[-1] if "." in fqn else fqn
            text_tokens: List[str] = _split_camel(simple_name)

            for m in cls.get("methods", []):
                method_name = m.get("name", "")
                if not method_name:
                    continue
                top_class = simple_name.split("$")[-1]
                if method_name == simple_name or method_name == top_class:
                    continue
                self.class_methods[fqn].append(m)
                text_tokens.extend(_split_camel(method_name))
                text_tokens.extend(m.get("parameterNames", []))

            self.class_texts[fqn] = " ".join(text_tokens)

        # Build class-level call graph
        fqn_set = set(self.class_fqns)
        for caller_fqn in self.class_fqns:
            for m in self.class_methods[caller_fqn]:
                for ref in m.get("invokedMethods", []):
                    dot = ref.rfind(".")
                    if dot < 0:
                        continue
                    target_fqn = ref[:dot]
                    if not target_fqn.startswith(self.base_package):
                        continue

                    resolved: List[str] = []
                    if target_fqn in fqn_set:
                        resolved.append(target_fqn)
                    elif target_fqn in self._interface_to_impls:
                        resolved.extend(
                            impl for impl in self._interface_to_impls[target_fqn]
                            if impl in fqn_set
                        )

                    for callee_fqn in resolved:
                        if callee_fqn != caller_fqn:
                            self.class_call_weights[caller_fqn][callee_fqn] += 1

    # ------------------------------------------------------------------
    # Output formatting
    # ------------------------------------------------------------------

    def build_clusters_json(
        self,
        class_to_cluster: Dict[str, int],
        algorithm: str = "unknown",
    ) -> dict:
        """Format clustering result as a simplified clusters.json."""
        cluster_classes: Dict[int, List[str]] = defaultdict(list)
        for fqn, cid in class_to_cluster.items():
            cluster_classes[cid].append(fqn)

        entries = []
        for cid in sorted(cluster_classes.keys()):
            classes = sorted(cluster_classes[cid])
            simple_names = [c.rsplit(".", 1)[-1] for c in classes]
            dominant = Counter(simple_names).most_common(1)[0][0]
            name = _camel_to_kebab(dominant) + "-service"
            entries.append({
                "id": cid,
                "name": name,
                "reason": f"{algorithm} cluster {cid}",
                "classes": classes,
            })

        return {
            "relativePath": self.project_id,
            "clusters": entries,
        }


# ---------------------------------------------------------------------------
# String utilities
# ---------------------------------------------------------------------------

def _split_camel(name: str) -> List[str]:
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    parts = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", parts)
    return [w.lower() for w in parts.split()]


def _camel_to_kebab(text: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1-\2", text)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", s)
    return s.lower()
