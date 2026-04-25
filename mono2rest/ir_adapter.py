"""
IR-A adapter — parses ir-a.json and builds Method objects + method-level call graph.

The call graph is a directed graph where each node is a Method and each edge
represents a method invocation extracted from the ``invokedMethods`` field in
IR-A's MethodInfo.  Interface-to-implementation resolution is handled so that
calls targeting an interface method are redirected to the concrete
implementation when one exists.
"""
from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

from .data_models import Method


class IrAAdapter:
    """Parse an ir-a.json file and expose methods + call graph."""

    def __init__(self, ir_a_path: str, *, base_package: str | None = None):
        # Lazy import keeps mono2rest installable without the full ms-baseline
        # package layout when used standalone.
        from common.schema_migration import coerce_legacy_method_schema

        with open(ir_a_path, "r", encoding="utf-8") as f:
            self.raw: dict = json.load(f)
        coerce_legacy_method_schema(self.raw)

        self.base_package = base_package or self._infer_base_package()

        # interface FQN → list of implementing class FQNs
        self._interface_to_impls: Dict[str, List[str]] = defaultdict(list)
        # classFqn → ClassInfo dict
        self._class_map: Dict[str, dict] = {}

        self.methods: List[Method] = []
        self.method_index: Dict[str, Method] = {}        # method.id → Method
        self.call_graph: Dict[str, Set[str]] = {}         # caller_id → {callee_ids}

        self._build()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_adjacency_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """Return (adj_matrix, ordered_method_ids).

        adj_matrix[i][j] == 1  ⟺  method i calls method j.
        """
        ids = [m.id for m in self.methods]
        idx = {mid: i for i, mid in enumerate(ids)}
        n = len(ids)
        mat = np.zeros((n, n), dtype=np.int8)
        for caller, callees in self.call_graph.items():
            if caller not in idx:
                continue
            for callee in callees:
                if callee in idx:
                    mat[idx[caller]][idx[callee]] = 1
        return mat, ids

    # ------------------------------------------------------------------
    # Internal build pipeline
    # ------------------------------------------------------------------

    def _infer_base_package(self) -> str:
        """Heuristic: the longest common package prefix of all classes."""
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
        # 1. Index classes and build interface→impl mapping
        for cls in self.raw.get("classes", []):
            fqn = cls.get("fqn", "")
            if not fqn:
                continue
            self._class_map[fqn] = cls
            for iface in cls.get("implementsTypes", []):
                # Strip generic parameters: "JpaRepository<Owner, Integer>" → needs raw FQN
                raw_iface = iface.split("<")[0].strip()
                if raw_iface.startswith(self.base_package):
                    self._interface_to_impls[raw_iface].append(fqn)

        # 2. Extract methods
        self._extract_methods()

        # 3. Build call graph
        self._build_call_graph()

    def _extract_methods(self):
        """Create Method objects for every method in every project class."""
        seen_ids: Dict[str, int] = {}  # for handling overloads

        for cls in self.raw.get("classes", []):
            fqn = cls.get("fqn", "")
            if not fqn or not fqn.startswith(self.base_package):
                continue

            pkg = cls.get("packageName", "")
            simple_class = fqn.rsplit(".", 1)[-1] if "." in fqn else fqn

            for m in cls.get("methods", []):
                method_name = m.get("name", "")
                if not method_name:
                    continue

                # Skip constructors (name == simple class name)
                if method_name == simple_class or method_name == simple_class.split("$")[-1]:
                    continue

                # Build unique ID
                base_id = f"{fqn}#{method_name}"
                if base_id in seen_ids:
                    seen_ids[base_id] += 1
                    method_id = f"{base_id}#{seen_ids[base_id]}"
                else:
                    seen_ids[base_id] = 0
                    method_id = base_id

                # Build text description for SBERT
                text_parts = _split_camel(method_name)
                text_parts.extend(m.get("parameterNames", []))
                text_parts.append(simple_class)
                text_desc = " ".join(text_parts)

                method = Method(
                    id=method_id,
                    name=method_name,
                    class_name=simple_class,
                    class_fqn=fqn,
                    package_name=pkg,
                    return_type=m.get("returnType", "void"),
                    parameter_types=m.get("parameterTypes", []),
                    parameter_names=m.get("parameterNames", []),
                    annotations=m.get("annotations", []),
                    invoked_methods=m.get("invokedMethods", []),
                    text_description=text_desc,
                )
                self.methods.append(method)
                self.method_index[method.id] = method

    def _build_call_graph(self):
        """Build method→method directed call graph from invokedMethods."""
        # Pre-build lookup: (classFqn, methodName) → list of Method ids
        target_lookup: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for m in self.methods:
            target_lookup[(m.class_fqn, m.name)].append(m.id)

        for m in self.methods:
            callees: Set[str] = set()
            for ref in m.invoked_methods:
                # ref format: "com.example.SomeClass.methodName"
                dot = ref.rfind(".")
                if dot < 0:
                    continue
                target_class_fqn = ref[:dot]
                target_method_name = ref[dot + 1:]

                # Only keep calls within the project's base package
                if not target_class_fqn.startswith(self.base_package):
                    continue

                # Direct match
                candidates = target_lookup.get((target_class_fqn, target_method_name), [])

                # Interface → implementation fallback
                if not candidates and target_class_fqn in self._interface_to_impls:
                    for impl_fqn in self._interface_to_impls[target_class_fqn]:
                        candidates = target_lookup.get((impl_fqn, target_method_name), [])
                        if candidates:
                            break

                for cid in candidates:
                    if cid != m.id:  # skip self-recursive
                        callees.add(cid)

            if callees:
                self.call_graph[m.id] = callees


def _split_camel(name: str) -> List[str]:
    """Split a camelCase or PascalCase identifier into lowercase words."""
    import re
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    parts = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", parts)
    return [w.lower() for w in parts.split()]
