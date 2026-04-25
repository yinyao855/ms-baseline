"""Class-level adapter for ir-a.json.

Builds:
* ``nodes``: list of :class:`ClassNode` (one per project class)
* ``adj``  : NxN weighted adjacency matrix where the weight of edge (i, j)
  blends method calls / field uses / inheritance / import references.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .data_models import ClassNode


# ---------------------------------------------------------------------------
# Edge weights — borrowed from microxpert's louvain implementation so that
# downstream metrics stay comparable across algorithms.
# ---------------------------------------------------------------------------
W_METHOD_CALL = 1.0
W_FIELD_USE = 0.6
W_INHERITS = 0.5
W_IMPORT_REF = 0.2


class IrAClassAdapter:
    """Parse ir-a.json into a class graph + textual descriptors."""

    def __init__(self, ir_a_path: str, *, base_package: str | None = None):
        from common.schema_migration import coerce_legacy_method_schema

        with open(ir_a_path, "r", encoding="utf-8") as f:
            self.raw: dict = json.load(f)
        coerce_legacy_method_schema(self.raw)

        self.base_package = base_package or self._infer_base_package()

        self.nodes: List[ClassNode] = []
        self.fqn_to_idx: Dict[str, int] = {}
        self.adj: np.ndarray = np.zeros((0, 0), dtype=np.float32)

        self._build()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def project_id(self) -> str:
        return self.raw.get("projectId", "")

    @property
    def n_classes(self) -> int:
        return len(self.nodes)

    # ------------------------------------------------------------------
    # Build pipeline
    # ------------------------------------------------------------------

    def _infer_base_package(self) -> str:
        """Longest common package prefix of all classes."""
        packages = [
            c.get("packageName", "")
            for c in self.raw.get("classes", [])
            if c.get("packageName")
        ]
        if not packages:
            return ""
        common = packages[0]
        for pkg in packages[1:]:
            while common and not pkg.startswith(common):
                dot = common.rfind(".")
                common = common[:dot] if dot >= 0 else ""
        return common

    def _build(self) -> None:
        self._extract_nodes()
        if not self.nodes:
            return
        self._build_adjacency()

    def _extract_nodes(self) -> None:
        for cls in self.raw.get("classes", []):
            fqn = cls.get("fqn", "")
            if not fqn or not fqn.startswith(self.base_package):
                continue

            simple = fqn.rsplit(".", 1)[-1]
            method_names = [
                m.get("name", "") for m in cls.get("methods", []) if m.get("name")
            ]
            field_names = [
                f.get("name", "") for f in cls.get("fields", []) if f.get("name")
            ]
            annotations = [
                _strip_annotation(a) for a in cls.get("annotations", [])
            ]

            text = _build_class_text(simple, method_names, field_names, annotations)
            node = ClassNode(
                fqn=fqn,
                simple_name=simple,
                package_name=cls.get("packageName", ""),
                annotations=annotations,
                method_names=method_names,
                field_names=field_names,
                text_description=text,
            )
            self.fqn_to_idx[fqn] = len(self.nodes)
            self.nodes.append(node)

    def _build_adjacency(self) -> None:
        n = len(self.nodes)
        adj = np.zeros((n, n), dtype=np.float32)

        # interface FQN → list of implementing class FQNs (for late binding)
        iface_to_impls: Dict[str, List[str]] = defaultdict(list)
        for cls in self.raw.get("classes", []):
            fqn = cls.get("fqn", "")
            if not fqn:
                continue
            for iface in cls.get("implementsTypes", []):
                raw_iface = iface.split("<")[0].strip()
                if raw_iface in self.fqn_to_idx:
                    iface_to_impls[raw_iface].append(fqn)

        def add(src_fqn: str, tgt_fqn: str, w: float) -> None:
            i = self.fqn_to_idx.get(src_fqn)
            j = self.fqn_to_idx.get(tgt_fqn)
            if i is None or j is None or i == j:
                return
            adj[i, j] += w
            adj[j, i] += w  # treat as undirected for clustering objectives

        for cls in self.raw.get("classes", []):
            fqn = cls.get("fqn", "")
            if fqn not in self.fqn_to_idx:
                continue

            for m in cls.get("methods", []):
                for ref in m.get("invokedMethods", []):
                    target = ref.rsplit(".", 1)[0] if "." in ref else ""
                    if not target:
                        continue
                    if target in self.fqn_to_idx:
                        add(fqn, target, W_METHOD_CALL)
                    elif target in iface_to_impls:
                        for impl in iface_to_impls[target]:
                            add(fqn, impl, W_METHOD_CALL)

            for fld in cls.get("fields", []):
                ftype = fld.get("type", "")
                target = ftype.split("<")[0].strip()
                if target in self.fqn_to_idx:
                    add(fqn, target, W_FIELD_USE)
                elif target in iface_to_impls:
                    for impl in iface_to_impls[target]:
                        add(fqn, impl, W_FIELD_USE)

            ext = (cls.get("extendsType") or "").split("<")[0].strip()
            if ext in self.fqn_to_idx:
                add(fqn, ext, W_INHERITS)
            for iface in cls.get("implementsTypes", []):
                tgt = iface.split("<")[0].strip()
                if tgt in self.fqn_to_idx:
                    add(fqn, tgt, W_INHERITS)

            for imp in cls.get("imports", []):
                if imp in self.fqn_to_idx:
                    add(fqn, imp, W_IMPORT_REF)

        self.adj = adj


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CAMEL = re.compile(r"([a-z])([A-Z])")
_CAMEL2 = re.compile(r"([A-Z]+)([A-Z][a-z])")


def _split_camel(name: str) -> List[str]:
    s = _CAMEL.sub(r"\1 \2", name)
    s = _CAMEL2.sub(r"\1 \2", s)
    return [w.lower() for w in s.split() if w]


def _strip_annotation(raw: str) -> str:
    """``@Service`` / ``org.springframework.stereotype.Service`` → ``Service``."""
    s = raw.lstrip("@").strip()
    return s.rsplit(".", 1)[-1]


def _build_class_text(
    simple: str,
    methods: List[str],
    fields: List[str],
    annotations: List[str],
) -> str:
    """Concatenate camel-split tokens for SBERT input."""
    tokens: List[str] = []
    tokens.extend(_split_camel(simple))
    for m in methods:
        tokens.extend(_split_camel(m))
    for f in fields:
        tokens.extend(_split_camel(f))
    # Spring stereotypes carry strong domain semantics; keep them as full words.
    for a in annotations:
        tokens.append(a.lower())
    return " ".join(tokens)
