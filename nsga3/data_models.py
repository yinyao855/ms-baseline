"""Lightweight data models for the class-level NSGA-III pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ClassNode:
    """A single Java class extracted from ir-a.json."""

    fqn: str
    simple_name: str
    package_name: str
    annotations: List[str] = field(default_factory=list)
    method_names: List[str] = field(default_factory=list)
    field_names: List[str] = field(default_factory=list)
    text_description: str = ""

    def __hash__(self) -> int:
        return hash(self.fqn)

    def to_dict(self) -> Dict:
        return {
            "fqn": self.fqn,
            "simple_name": self.simple_name,
            "package_name": self.package_name,
        }


@dataclass
class ClusterResult:
    """One cluster produced by NSGA-III."""

    cluster_id: int
    name: str
    classes: List[str]
    coupling: float = 0.0
    cohesion: float = 0.0
    semantic_similarity: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.cluster_id,
            "name": self.name,
            "reason": self.reason,
            "classes": self.classes,
            "metrics": {
                "coupling": float(self.coupling),
                "cohesion": float(self.cohesion),
                "semantic_similarity": float(self.semantic_similarity),
            },
        }
