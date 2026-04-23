"""
Coupling criteria extraction and weighted graph construction for Service Cutter.

Implements three coupling criteria from the Service Cutter paper
(Gysel et al., ESOCC 2016) adapted to work with IR-A static analysis data:

  - CC-1 Identity & Lifecycle Commonality (Cohesive Group Scorer)
  - CC-2 Semantic Proximity (Semantic Proximity Scorer)
  - CC-9 Consistency Constraint (Cohesive Group Scorer)

Each scorer produces a dict of (classA, classB) → raw_score.
Scores are combined via:  weight(A, B) = Σ(score_cc × priority_cc)
Edges with weight ≤ 0 are removed (per paper's deleteNegativeEdges).
"""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Set, Tuple

import networkx as nx

from common.ir_parser import IrAProject

Pair = Tuple[str, str]

# Paper T-shirt size priorities: IGNORE=0, XS=1, S=2, M=4, L=7, XL=8, XXL=9
DEFAULT_PRIORITIES = {
    "identity_lifecycle": 5,   # M
    "semantic_proximity": 7,   # L
    "consistency_constraint": 4,  # M
}

COHESIVE_GROUP_SCORE = 10.0


class CouplingScorer:
    """Extract coupling criteria from IrAProject and build a weighted graph."""

    def __init__(self, project: IrAProject, priorities: Dict[str, float] | None = None):
        self.project = project
        self.priorities = priorities or dict(DEFAULT_PRIORITIES)

        self._fqn_set: Set[str] = set(project.class_fqns)
        self._class_map: Dict[str, dict] = {}
        self._entity_fqns: Set[str] = set()
        for cls in project.raw.get("classes", []):
            fqn = cls.get("fqn", "")
            if fqn in self._fqn_set:
                self._class_map[fqn] = cls
                if cls.get("isEntity"):
                    self._entity_fqns.add(fqn)

    # ------------------------------------------------------------------
    # CC-1: Identity & Lifecycle Commonality  (Cohesive Group Scorer)
    # ------------------------------------------------------------------

    def score_identity_lifecycle(self) -> Dict[Pair, float]:
        """Classes referencing the same entity type form a cohesive group.

        For each entity class, find all project classes that reference it
        via field types, method parameter types, or return types.  Every pair
        within such a group receives +10 (Cohesive Group Scorer).
        """
        entity_referrers: Dict[str, Set[str]] = defaultdict(set)

        for fqn in self.project.class_fqns:
            cls = self._class_map.get(fqn)
            if cls is None:
                continue

            # The entity itself is part of its own lifecycle group
            if fqn in self._entity_fqns:
                entity_referrers[fqn].add(fqn)

            referenced = set()
            for field in cls.get("fields", []):
                referenced.add(_strip_generics(field.get("type", "")))
            for method in cls.get("methods", []):
                referenced.add(_strip_generics(method.get("returnType", "")))
                for pt in method.get("parameterTypes", []):
                    referenced.add(_strip_generics(pt))

            for entity_fqn in self._entity_fqns:
                if entity_fqn in referenced:
                    entity_referrers[entity_fqn].add(fqn)

        scores: Dict[Pair, float] = defaultdict(float)
        for _entity, group in entity_referrers.items():
            members = sorted(group & self._fqn_set)
            for a, b in combinations(members, 2):
                key = _ordered_pair(a, b)
                scores[key] = max(scores[key], COHESIVE_GROUP_SCORE)
        return dict(scores)

    # ------------------------------------------------------------------
    # CC-2: Semantic Proximity  (Semantic Proximity Scorer)
    # ------------------------------------------------------------------

    def score_semantic_proximity(self) -> Dict[Pair, float]:
        """Joint access counting from method invocations, normalized to 0-10.

        For each method in each class, collect all distinct project classes it
        calls.  Every pair of callees co-accessed in the same method gets a
        score increment.  Direct caller→callee edges also contribute.
        Scores are normalized per the paper's top-10% algorithm.
        """
        raw_scores: Dict[Pair, float] = defaultdict(float)

        for caller_fqn in self.project.class_fqns:
            for method in self.project.class_methods.get(caller_fqn, []):
                callees: Set[str] = set()
                for ref in method.get("invokedMethods", []):
                    dot = ref.rfind(".")
                    if dot < 0:
                        continue
                    target = ref[:dot]
                    resolved = self._resolve_target(target)
                    callees.update(resolved)

                callees.discard(caller_fqn)
                callees &= self._fqn_set

                # Co-access: pairs of callees within the same method
                for a, b in combinations(sorted(callees), 2):
                    raw_scores[_ordered_pair(a, b)] += 3.0  # READ-like co-access

                # Direct caller-callee coupling
                for callee in callees:
                    raw_scores[_ordered_pair(caller_fqn, callee)] += 1.0

        return _normalize_semantic_proximity(dict(raw_scores))

    # ------------------------------------------------------------------
    # CC-9: Consistency Constraint  (Cohesive Group Scorer)
    # ------------------------------------------------------------------

    def score_consistency_constraint(self) -> Dict[Pair, float]:
        """Classes in the same inheritance hierarchy form a cohesive group.

        Builds transitive closure of the inheritance tree: if A extends B
        extends C, then {A, B, C} all belong to the same cohesive group.
        Every pair within a group receives +10 (Cohesive Group Scorer).
        """
        # Build child → parent mapping
        parent_of: Dict[str, str] = {}
        for fqn in self.project.class_fqns:
            cls = self._class_map.get(fqn)
            if cls is None:
                continue
            parent = cls.get("extendsType")
            if parent and parent in self._fqn_set:
                parent_of[fqn] = parent

        # Find the root of each class's inheritance chain
        def find_root(fqn: str) -> str:
            visited: Set[str] = set()
            cur = fqn
            while cur in parent_of and cur not in visited:
                visited.add(cur)
                cur = parent_of[cur]
            return cur

        # Group classes by their inheritance root
        hierarchy: Dict[str, Set[str]] = defaultdict(set)
        for fqn in parent_of:
            root = find_root(fqn)
            hierarchy[root].add(root)
            hierarchy[root].add(fqn)

        scores: Dict[Pair, float] = defaultdict(float)
        for _root, group in hierarchy.items():
            members = sorted(group & self._fqn_set)
            if len(members) < 2:
                continue
            for a, b in combinations(members, 2):
                key = _ordered_pair(a, b)
                scores[key] = max(scores[key], COHESIVE_GROUP_SCORE)
        return dict(scores)

    # ------------------------------------------------------------------
    # Weighted graph construction
    # ------------------------------------------------------------------

    def build_weighted_graph(self) -> nx.Graph:
        """Combine all criteria scores × priorities into a single weighted graph.

        Edge weight = Σ(score_cc × priority_cc) for each coupling criterion.
        Edges with weight ≤ 0 are removed (paper: deleteNegativeEdges).
        """
        criteria = [
            ("identity_lifecycle", self.score_identity_lifecycle()),
            ("semantic_proximity", self.score_semantic_proximity()),
            ("consistency_constraint", self.score_consistency_constraint()),
        ]

        merged: Dict[Pair, float] = defaultdict(float)
        for name, scores in criteria:
            priority = self.priorities.get(name, 1.0)
            for pair, score in scores.items():
                merged[pair] += score * priority

        G = nx.Graph()
        G.add_nodes_from(self.project.class_fqns)

        for (u, v), weight in merged.items():
            if weight > 0:
                G.add_edge(u, v, weight=weight)

        return G

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_target(self, target_fqn: str) -> List[str]:
        """Resolve a target FQN, handling interface → implementation mapping."""
        if target_fqn in self._fqn_set:
            return [target_fqn]
        if target_fqn in self.project._interface_to_impls:
            return [
                impl for impl in self.project._interface_to_impls[target_fqn]
                if impl in self._fqn_set
            ]
        return []


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _ordered_pair(a: str, b: str) -> Pair:
    return (a, b) if a <= b else (b, a)


def _strip_generics(type_str: str) -> str:
    """Remove generic parameters: 'List<Foo>' → 'Foo', 'Map<K,V>' → 'K'."""
    idx = type_str.find("<")
    if idx < 0:
        return type_str
    inner = type_str[idx + 1:]
    inner = inner.rstrip(">").split(",")[0].strip()
    return inner if inner else type_str[:idx]


def _normalize_semantic_proximity(raw: Dict[Pair, float]) -> Dict[Pair, float]:
    """Normalize scores to 0-10 using the paper's top-10% algorithm.

    Reproduces SemanticProximityCriterionScorer.normalizeResult():
      - Sort scores descending
      - referenceValue = score at the 10th-percentile position
      - divisor = referenceValue / 10
      - Each score = min(10, score / divisor)
    """
    if not raw:
        return {}

    values = sorted(raw.values(), reverse=True)
    ten_pct = max(1, int(len(values) * 0.1))
    reference_value = values[ten_pct - 1]
    if reference_value <= 0:
        reference_value = values[0]
    divisor = reference_value / 10.0
    if divisor <= 0:
        return {k: 10.0 for k in raw}

    return {k: min(10.0, v / divisor) for k, v in raw.items()}
