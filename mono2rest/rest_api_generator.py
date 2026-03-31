"""
REST API generation module — Phase 2 of the MONO2REST pipeline.

1. **Exposed method selection** (paper §III-B, Fig. 7):
   Methods called by another cluster must be wrapped by an API endpoint.

2. **HTTP method assignment** (paper §III-B, Fig. 8):
   Zero-Shot Classification with ``facebook/bart-large-mnli``.

3. **URI generation** (paper §III-B, Fig. 9):
   Tree-based URI construction with NLTK POS tagging for verb removal
   and SBERT for class-name semantic grouping.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .data_models import Cluster, HTTPMethod, Method, RESTEndpoint


# ---------------------------------------------------------------------------
# Zero-Shot HTTP classifier (lazy-loaded singleton)
# ---------------------------------------------------------------------------

_zs_classifier = None


def _get_classifier():
    global _zs_classifier
    if _zs_classifier is not None:
        return _zs_classifier
    try:
        from transformers import pipeline
        _zs_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
    except Exception as e:
        print(f"[WARN] Could not load zero-shot classifier: {e}")
        _zs_classifier = None
    return _zs_classifier


# ---------------------------------------------------------------------------
# NLTK POS tagger (lazy-loaded)
# ---------------------------------------------------------------------------

_nltk_ready = False


def _ensure_nltk():
    global _nltk_ready
    if _nltk_ready:
        return
    try:
        import nltk
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        _nltk_ready = True
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class RESTAPIGenerator:
    def __init__(self, embedder=None, llm_classifier=None):
        """
        Args:
            embedder: a ``SemanticEmbedder`` instance (reused from Phase 1) for
                      class-name grouping.  If *None*, grouping falls back to
                      simple string matching.
            llm_classifier: an ``LLMHttpClassifier`` instance for HTTP method
                      assignment via LLM API.  If *None*, uses local zero-shot
                      classifier or heuristic fallback.
        """
        self.embedder = embedder
        self.llm_classifier = llm_classifier

    # ------------------------------------------------------------------
    # Step 1: select methods to expose (paper §III-B, Fig. 7)
    # ------------------------------------------------------------------

    def filter_exposed_methods(
        self,
        clusters: List[Cluster],
        call_graph: Dict[str, Set[str]],
    ) -> List[Tuple[Method, int]]:
        """Return [(method, cluster_id)] for every method called from outside its cluster."""
        cluster_of: Dict[str, int] = {}
        for c in clusters:
            for m in c.methods:
                cluster_of[m.id] = c.cluster_id

        exposed: List[Tuple[Method, int]] = []
        seen: Set[str] = set()
        for c in clusters:
            for m in c.methods:
                if m.id in seen:
                    continue
                for caller_id, callees in call_graph.items():
                    if m.id in callees and cluster_of.get(caller_id, c.cluster_id) != c.cluster_id:
                        exposed.append((m, c.cluster_id))
                        seen.add(m.id)
                        break
        return exposed

    # ------------------------------------------------------------------
    # Step 2: HTTP method assignment (paper §III-B, Fig. 8)
    # ------------------------------------------------------------------

    def assign_http_methods(
        self, exposed: List[Tuple[Method, int]]
    ) -> List[Tuple[Method, int, HTTPMethod]]:
        """Assign an HTTP verb to each exposed method.

        Backend priority: LLM API → local zero-shot classifier → heuristic.
        """
        methods_only = [m for m, _ in exposed]
        cids = [cid for _, cid in exposed]

        if self.llm_classifier is not None:
            # LLM API batch classification
            http_list = self.llm_classifier.classify_batch(methods_only)
            return [(m, c, h) for m, c, h in zip(methods_only, cids, http_list)]

        classifier = _get_classifier()
        results: List[Tuple[Method, int, HTTPMethod]] = []
        for method, cid in exposed:
            if classifier is not None:
                http = self._classify_zero_shot(classifier, method)
            else:
                http = self._classify_heuristic(method)
            results.append((method, cid, http))
        return results

    @staticmethod
    def _classify_zero_shot(classifier, method: Method) -> HTTPMethod:
        text = method.get_signature_text()
        labels = ["GET", "POST", "PUT", "DELETE"]
        try:
            out = classifier(text, candidate_labels=labels)
            top = out["labels"][0]
            return HTTPMethod(top)
        except Exception:
            return RESTAPIGenerator._classify_heuristic(method)

    @staticmethod
    def _classify_heuristic(method: Method) -> HTTPMethod:
        name_lower = method.name.lower()
        for verb, http in [
            ("get", HTTPMethod.GET), ("find", HTTPMethod.GET),
            ("fetch", HTTPMethod.GET), ("retrieve", HTTPMethod.GET),
            ("list", HTTPMethod.GET), ("search", HTTPMethod.GET),
            ("query", HTTPMethod.GET), ("load", HTTPMethod.GET),
            ("create", HTTPMethod.POST), ("add", HTTPMethod.POST),
            ("save", HTTPMethod.POST), ("insert", HTTPMethod.POST),
            ("submit", HTTPMethod.POST), ("register", HTTPMethod.POST),
            ("update", HTTPMethod.PUT), ("modify", HTTPMethod.PUT),
            ("edit", HTTPMethod.PUT), ("set", HTTPMethod.PUT),
            ("delete", HTTPMethod.DELETE), ("remove", HTTPMethod.DELETE),
        ]:
            if verb in name_lower:
                return http
        if method.return_type.lower() in ("void", "boolean"):
            return HTTPMethod.POST
        return HTTPMethod.GET

    # ------------------------------------------------------------------
    # Step 3: URI generation (paper §III-B, Fig. 9)
    # ------------------------------------------------------------------

    def generate_endpoints(
        self,
        assignments: List[Tuple[Method, int, HTTPMethod]],
        clusters: List[Cluster],
    ) -> List[RESTEndpoint]:
        """Build REST endpoints with proper URIs."""
        # Group by cluster
        cluster_map: Dict[int, List[Tuple[Method, HTTPMethod]]] = defaultdict(list)
        for method, cid, http in assignments:
            cluster_map[cid].append((method, http))

        cluster_name_map = self._build_cluster_names(clusters)
        endpoints: List[RESTEndpoint] = []

        for cid, items in cluster_map.items():
            cluster_root = cluster_name_map.get(cid, f"cluster-{cid}")

            # Group class names within this cluster for semantic merging
            class_names = list({m.class_name for m, _ in items})
            class_label = self._merge_class_names(class_names)

            for method, http in items:
                method_segment = self._process_method_name(method.name)
                uri_parts = [cluster_root]
                if class_label:
                    uri_parts.append(class_label)
                if method_segment and method_segment != class_label:
                    uri_parts.append(method_segment)
                uri = "/" + "/".join(uri_parts)

                # Append path parameter for ID-like single params
                if len(method.parameter_names) == 1:
                    pname = method.parameter_names[0].lower()
                    if "id" in pname or pname in ("key", "identifier"):
                        uri += f"/{{{method.parameter_names[0]}}}"

                endpoints.append(RESTEndpoint(
                    uri=uri,
                    http_method=http,
                    method=method,
                    cluster_id=cid,
                    description=f"{method.class_name}.{method.name}",
                ))
        return endpoints

    # ---- URI helpers ----

    @staticmethod
    def _build_cluster_names(clusters: List[Cluster]) -> Dict[int, str]:
        """Derive a kebab-case name for each cluster from its dominant class name."""
        result: Dict[int, str] = {}
        for c in clusters:
            counts: Dict[str, int] = defaultdict(int)
            for m in c.methods:
                # Normalise inner classes: "PetResource$PetRequest" → "PetResource"
                top_class = m.class_name.split("$")[0]
                counts[top_class] += 1
            if counts:
                dominant = max(counts, key=counts.get)
                result[c.cluster_id] = _strip_suffix_and_kebab(dominant)
            else:
                result[c.cluster_id] = f"cluster-{c.cluster_id}"
        return result

    def _merge_class_names(self, names: List[str]) -> str:
        """Merge semantically similar class names into a single URI segment (paper step 3)."""
        if len(names) == 0:
            return ""
        if len(names) == 1:
            return _strip_suffix_and_kebab(names[0])

        if self.embedder is not None:
            vecs = self.embedder.embed_texts([_strip_suffix(n) for n in names])
            # Pick the name whose embedding is closest to the centroid
            centroid = vecs.mean(axis=0)
            sims = vecs @ centroid
            best = names[int(sims.argmax())]
            return _strip_suffix_and_kebab(best)

        # Fallback: longest common prefix
        prefix = names[0]
        for n in names[1:]:
            while not n.startswith(prefix) and prefix:
                prefix = prefix[:-1]
        return _camel_to_kebab(prefix).lower() if prefix else _strip_suffix_and_kebab(names[0])

    @staticmethod
    def _process_method_name(name: str) -> str:
        """CamelCase split → POS tag → remove verbs → lowercase-hyphen (paper step 2)."""
        words = _split_camel(name)
        if len(words) <= 1:
            return _camel_to_kebab(name)

        _ensure_nltk()
        try:
            import nltk
            tagged = nltk.pos_tag(words)
            # Remove leading verbs (VB*)
            non_verb = [w for w, pos in tagged if not pos.startswith("VB")]
            if non_verb:
                words = non_verb
        except Exception:
            pass

        return "-".join(w.lower() for w in words)


# ---------------------------------------------------------------------------
# String utilities
# ---------------------------------------------------------------------------

_SUFFIXES = (
    "Controller", "Service", "ServiceImpl", "Repository", "DAO",
    "Manager", "Handler", "Resource", "Impl", "Request", "Response",
)


def _strip_suffix(name: str) -> str:
    # Handle inner classes: "PetResource$PetRequest" → "PetRequest"
    if "$" in name:
        name = name.rsplit("$", 1)[-1]
    for s in _SUFFIXES:
        if name.endswith(s):
            return name[: -len(s)] or name
    return name


def _strip_suffix_and_kebab(name: str) -> str:
    return _camel_to_kebab(_strip_suffix(name))


def _camel_to_kebab(text: str) -> str:
    s = re.sub(r"([a-z])([A-Z])", r"\1-\2", text)
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1-\2", s)
    return s.lower()


def _split_camel(name: str) -> List[str]:
    parts = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    parts = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", parts)
    return parts.split()
