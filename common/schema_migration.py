"""Schema migration between ir-a.json revisions.

The microxpert extractor switched in v1.0.0+ from the legacy method shape::

    {
      "parameterTypes": ["int", "String"],
      "parameterNames": ["id", "name"],
      "invokedMethods": ["com.example.Foo.bar", ...],
      "invokedTypes":   ["com.example.Foo", ...],
    }

to a structured shape::

    {
      "parameters":    [{"name": "id", "type": "int"}, ...],
      "calledMethods": [{"targetType": "com.example.Foo",
                         "methodName": "bar"}, ...],
    }

Every baseline adapter (kmeans / louvain / mono2rest / nsga3 /
service_cutter) was written against the legacy field names. Rather than
audit each one, this module **back-fills** the legacy fields in place so
the rest of the pipeline keeps working. The migration is idempotent
(running it twice is a no-op).
"""
from __future__ import annotations

from typing import Any, Dict, List


def coerce_legacy_method_schema(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Mutate ``raw`` (an ir-a.json dict) so every method carries both the
    new and the legacy field names. Returns the same dict for chaining.

    The function tolerates partial inputs (missing classes / methods) so
    it is safe to call on hand-crafted fixtures during tests.
    """
    for cls in raw.get("classes", []) or []:
        for method in cls.get("methods", []) or []:
            _normalise_parameters(method)
            _normalise_called_methods(method)
    return raw


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_parameters(method: Dict[str, Any]) -> None:
    """Ensure both ``parameters`` and ``parameterTypes/Names`` are present."""
    has_new = "parameters" in method
    has_legacy = "parameterTypes" in method or "parameterNames" in method

    if has_new and not has_legacy:
        params = method.get("parameters") or []
        method["parameterTypes"] = [p.get("type", "") for p in params]
        method["parameterNames"] = [p.get("name", "") for p in params]
    elif has_legacy and not has_new:
        types: List[str] = method.get("parameterTypes") or []
        names: List[str] = method.get("parameterNames") or []
        # Pad the shorter list so we never index out of range.
        n = max(len(types), len(names))
        method["parameters"] = [
            {"name": names[i] if i < len(names) else "",
             "type": types[i] if i < len(types) else ""}
            for i in range(n)
        ]


def _normalise_called_methods(method: Dict[str, Any]) -> None:
    """Ensure both ``calledMethods`` and ``invokedMethods/Types`` are present."""
    has_new = "calledMethods" in method
    has_legacy = "invokedMethods" in method or "invokedTypes" in method

    if has_new and not has_legacy:
        called = method.get("calledMethods") or []
        invoked: List[str] = []
        invoked_types: List[str] = []
        for c in called:
            target = c.get("targetType", "")
            name = c.get("methodName", "")
            if target:
                invoked.append(f"{target}.{name}" if name else target)
                invoked_types.append(target)
        method["invokedMethods"] = invoked
        method["invokedTypes"] = invoked_types
    elif has_legacy and not has_new:
        invoked: List[str] = method.get("invokedMethods") or []
        called = []
        for ref in invoked:
            dot = ref.rfind(".")
            if dot < 0:
                called.append({"targetType": ref, "methodName": ""})
            else:
                called.append({"targetType": ref[:dot], "methodName": ref[dot + 1:]})
        method["calledMethods"] = called
