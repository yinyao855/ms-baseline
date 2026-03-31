"""
Data models for MONO2REST pipeline.
"""
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class HTTPMethod(str, Enum):
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"


@dataclass
class Method:
    """A single Java method extracted from IR-A."""
    id: str                          # unique: "classFqn#methodName" (or #methodName#n for overloads)
    name: str                        # simple method name
    class_name: str                  # simple class name
    class_fqn: str                   # fully-qualified class name
    package_name: str
    return_type: str
    parameter_types: List[str] = field(default_factory=list)
    parameter_names: List[str] = field(default_factory=list)
    annotations: List[str] = field(default_factory=list)
    invoked_methods: List[str] = field(default_factory=list)  # "classFqn.methodName" refs
    text_description: str = ""       # concatenated text for SBERT embedding

    def get_full_name(self) -> str:
        return f"{self.class_fqn}.{self.name}"

    def get_signature_text(self) -> str:
        """Build 'returnType methodName(paramTypes)' for zero-shot classification."""
        params = ", ".join(self.parameter_types) if self.parameter_types else ""
        return f"{self.return_type} {self.name}({params})"

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Method):
            return self.id == other.id
        return False

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "class_name": self.class_name,
            "class_fqn": self.class_fqn,
            "package_name": self.package_name,
            "return_type": self.return_type,
            "parameter_types": self.parameter_types,
            "parameter_names": self.parameter_names,
        }


@dataclass
class Cluster:
    """A microservice candidate: a group of methods."""
    cluster_id: int
    methods: List[Method] = field(default_factory=list)
    coupling: float = 0.0
    cohesion: float = 0.0
    semantic_similarity: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "cluster_id": self.cluster_id,
            "methods": [m.to_dict() for m in self.methods],
            "metrics": {
                "coupling": float(self.coupling),
                "cohesion": float(self.cohesion),
                "semantic_similarity": float(self.semantic_similarity),
            },
        }


@dataclass
class RESTEndpoint:
    """A generated REST API endpoint."""
    uri: str
    http_method: HTTPMethod
    method: Method
    cluster_id: int
    description: str = ""

    def to_dict(self) -> Dict:
        return {
            "uri": self.uri,
            "http_method": self.http_method.value,
            "method": self.method.to_dict(),
            "cluster_id": self.cluster_id,
            "description": self.description,
        }
