"""NSGA-III three-objective clustering at the **class** level.

Objectives (all minimised internally):
    f1 = total coupling      = Σ_k  E_external(k) / (E_internal(k) + E_external(k))
    f2 = -avg cohesion       = -(1/K) Σ_k  min(E_internal(k) / V(k), 1.0)
    f3 = -avg semantic sim   = -(1/K) Σ_k  pair_avg_sim(k)

The genetic operators follow MONO2REST Fig.5 (cluster injection crossover
+ random transfer mutation), with a simple non-dominated sort + crowding
distance selection (NSGA-II style — this is a faithful and self-contained
substitute for NSGA-III's reference-point selection at the small population
sizes we use here).
"""
from __future__ import annotations

import random
from typing import List, Tuple

import numpy as np

from .data_models import ClusterResult, ClassNode


# ---------------------------------------------------------------------------
# Problem
# ---------------------------------------------------------------------------


class ClassClusteringProblem:
    """Encapsulates the three NSGA objectives over weighted class graphs."""

    def __init__(self, adj: np.ndarray, sim: np.ndarray, k: int):
        self.adj = adj.astype(np.float32)
        self.sim = sim.astype(np.float32)
        self.n = adj.shape[0]
        self.k = k
        self._row_sum = self.adj.sum(axis=1)

    def evaluate(self, pop: np.ndarray) -> np.ndarray:
        out = np.empty((pop.shape[0], 3), dtype=np.float64)
        for i in range(pop.shape[0]):
            out[i] = self._eval_individual(pop[i])
        return out

    def _eval_individual(self, x: np.ndarray) -> Tuple[float, float, float]:
        K = self.k
        adj = self.adj
        sim = self.sim

        total_coupling = 0.0
        total_cohesion = 0.0
        total_semsim = 0.0
        n_nonempty = 0

        for k in range(K):
            mk = x == k
            v = int(mk.sum())
            if v == 0:
                continue
            n_nonempty += 1

            inner = float(adj[np.ix_(mk, mk)].sum())
            outer = float(self._row_sum[mk].sum()) - inner
            denom = inner + outer
            total_coupling += outer / denom if denom > 0 else 0.0
            total_cohesion += min(inner / v, 1.0)

            if v >= 2:
                csim = sim[np.ix_(mk, mk)]
                semsim = float(csim.sum() - np.trace(csim)) / v
            else:
                semsim = 1.0
            total_semsim += semsim

        avg_coh = total_cohesion / n_nonempty if n_nonempty > 0 else 0.0
        avg_sem = total_semsim / n_nonempty if n_nonempty > 0 else 0.0
        return (total_coupling, -avg_coh, -avg_sem)


# ---------------------------------------------------------------------------
# GA operators (paper Fig. 5 + repair)
# ---------------------------------------------------------------------------


def _crossover(p1: np.ndarray, p2: np.ndarray, K: int) -> np.ndarray:
    child = p2.copy()
    donor = random.randint(0, K - 1)
    child[p1 == donor] = donor
    _repair(child, K)
    return child


def _mutate(x: np.ndarray, K: int) -> np.ndarray:
    child = x.copy()
    if K < 2 or len(child) == 0:
        return child
    idx = random.randint(0, len(child) - 1)
    old = int(child[idx])
    new = random.randint(0, K - 2)
    if new >= old:
        new += 1
    child[idx] = new
    _repair(child, K)
    return child


def _repair(x: np.ndarray, K: int) -> None:
    """Ensure every cluster id in [0, K) has at least one member."""
    present = set(int(v) for v in np.unique(x))
    missing = [k for k in range(K) if k not in present]
    if not missing:
        return
    for k in missing:
        counts = np.bincount(x, minlength=K)
        largest = int(np.argmax(counts))
        candidates = np.where(x == largest)[0]
        if len(candidates) > 1:
            victim = int(random.choice(candidates.tolist()))
            x[victim] = k


# ---------------------------------------------------------------------------
# Selection — non-dominated sort + crowding distance
# ---------------------------------------------------------------------------


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def _non_dominated_sort(F: np.ndarray) -> List[List[int]]:
    n = F.shape[0]
    dom_count = np.zeros(n, dtype=int)
    dominated_by: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(F[i], F[j]):
                dominated_by[i].append(j)
                dom_count[j] += 1
            elif _dominates(F[j], F[i]):
                dominated_by[j].append(i)
                dom_count[i] += 1
    fronts: List[List[int]] = []
    current = [i for i in range(n) if dom_count[i] == 0]
    while current:
        fronts.append(current)
        nxt: List[int] = []
        for i in current:
            for j in dominated_by[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        current = nxt
    return fronts


def _crowding(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    dist = np.zeros(n)
    for obj in range(m):
        order = np.argsort(F[:, obj])
        dist[order[0]] = dist[order[-1]] = np.inf
        rng = F[order[-1], obj] - F[order[0], obj]
        if rng == 0:
            continue
        for i in range(1, n - 1):
            dist[order[i]] += (F[order[i + 1], obj] - F[order[i - 1], obj]) / rng
    return dist


def _select(F: np.ndarray, target: int) -> np.ndarray:
    fronts = _non_dominated_sort(F)
    chosen: List[int] = []
    for front in fronts:
        if len(chosen) + len(front) <= target:
            chosen.extend(front)
            continue
        remaining = target - len(chosen)
        cd = _crowding(F[front])
        order = np.argsort(-cd)
        chosen.extend([front[order[i]] for i in range(remaining)])
        break
    return np.array(chosen)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


class NSGAClassClustering:
    def __init__(
        self,
        population_size: int = 80,
        max_generations: int = 120,
        crossover_rate: float = 0.85,
        mutation_rate: float = 0.15,
        seed: int | None = 42,
        min_cluster_size: int = 2,
    ):
        self.pop_size = population_size
        self.max_gen = max_generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate
        self.min_cluster_size = min_cluster_size
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    # ------------------------------------------------------------------

    def cluster(
        self,
        nodes: List[ClassNode],
        adj: np.ndarray,
        sim: np.ndarray,
        k: int,
    ) -> List[ClusterResult]:
        n = len(nodes)
        if n == 0:
            return []
        k = max(1, min(k, n))

        problem = ClassClusteringProblem(adj, sim, k)

        pop = np.array(
            [self._seed_individual(n, k, sim) for _ in range(self.pop_size)]
        )
        for i in range(self.pop_size):
            _repair(pop[i], k)
        F = problem.evaluate(pop)

        for gen in range(self.max_gen):
            offspring: List[np.ndarray] = []
            while len(offspring) < self.pop_size:
                i1, i2 = random.sample(range(self.pop_size), 2)
                if random.random() < self.cx_rate:
                    child = _crossover(pop[i1], pop[i2], k)
                else:
                    child = pop[i1].copy()
                if random.random() < self.mut_rate:
                    child = _mutate(child, k)
                offspring.append(child)
            offs = np.array(offspring)
            F_off = problem.evaluate(offs)

            combined = np.vstack([pop, offs])
            F_comb = np.vstack([F, F_off])
            sel = _select(F_comb, self.pop_size)
            pop = combined[sel]
            F = F_comb[sel]

            if gen % 30 == 0 or gen == self.max_gen - 1:
                best = int(np.argmin(F[:, 0] - F[:, 1] - F[:, 2]))
                c, ch, ss = F[best]
                print(
                    f"    gen {gen:>4d}: coupling={c:.4f}  "
                    f"cohesion={-ch:.4f}  semSim={-ss:.4f}"
                )

        best_idx = self._pick_best(F)
        assignment = pop[best_idx].copy()
        assignment = self._merge_orphans(
            assignment, adj, sim, self.min_cluster_size
        )
        return self._materialise(assignment, nodes, adj, sim, k)

    # ------------------------------------------------------------------

    @staticmethod
    def _seed_individual(n: int, k: int, sim: np.ndarray) -> np.ndarray:
        """Random init biased by semantic similarity to give the GA a head start.

        We pick K random "centroids" and assign every other class to the
        centroid it is most similar to. This dramatically reduces the
        warm-up time vs pure random init on small graphs.
        """
        if k <= 1 or n <= k:
            return np.random.randint(0, max(1, k), size=n)
        centroids = np.random.choice(n, size=k, replace=False)
        sim_to_centroids = sim[:, centroids]
        return np.argmax(sim_to_centroids, axis=1).astype(np.int64)

    @staticmethod
    def _pick_best(F: np.ndarray) -> int:
        F_norm = F.copy()
        for j in range(F.shape[1]):
            lo, hi = F_norm[:, j].min(), F_norm[:, j].max()
            if hi - lo > 0:
                F_norm[:, j] = (F_norm[:, j] - lo) / (hi - lo)
            else:
                F_norm[:, j] = 0.0
        return int(np.argmin(F_norm.sum(axis=1)))

    @staticmethod
    def _merge_orphans(
        assignment: np.ndarray,
        adj: np.ndarray,
        sim: np.ndarray,
        min_size: int,
    ) -> np.ndarray:
        """Dissolve clusters smaller than ``min_size`` into their best neighbour.

        For each member of an orphan cluster pick the host cluster that
        maximises (graph_link_strength + 0.25 * semantic_similarity). The
        graph term dominates so we honour structural coupling first, then
        use semantics as a tie-breaker for fully isolated nodes.
        """
        if min_size <= 1:
            return assignment

        result = assignment.copy()
        # Iterate until a fixed point: merging an orphan can re-shape the
        # remaining clusters, but we cap to a few rounds for safety.
        for _ in range(4):
            counts = np.bincount(result, minlength=int(result.max()) + 1)
            small = [c for c, n in enumerate(counts) if 0 < n < min_size]
            if not small:
                break

            # Cluster ids that are *not* orphans — eligible host pool.
            hosts = [c for c, n in enumerate(counts) if n >= min_size]
            if not hosts:
                # Everyone is small; merge into the largest cluster.
                hosts = [int(np.argmax(counts))]

            for orphan_id in small:
                members = np.where(result == orphan_id)[0]
                for node in members:
                    best_host = hosts[0]
                    best_score = -np.inf
                    for h in hosts:
                        host_mask = result == h
                        link = float(adj[node, host_mask].sum())
                        sem = float(sim[node, host_mask].mean()) if host_mask.any() else 0.0
                        score = link + 0.25 * sem
                        if score > best_score:
                            best_score = score
                            best_host = h
                    result[node] = best_host
        return result

    @staticmethod
    def _materialise(
        assignment: np.ndarray,
        nodes: List[ClassNode],
        adj: np.ndarray,
        sim: np.ndarray,
        k: int,
    ) -> List[ClusterResult]:
        out: List[ClusterResult] = []
        for cid in range(k):
            mask = assignment == cid
            members = [nodes[i].fqn for i in range(len(nodes)) if mask[i]]
            if not members:
                continue
            v = int(mask.sum())
            inner = float(adj[np.ix_(mask, mask)].sum())
            outer = float(adj[mask, :].sum()) - inner
            denom = inner + outer
            coupling = outer / denom if denom > 0 else 0.0
            cohesion = min(inner / v, 1.0)
            if v >= 2:
                csim = sim[np.ix_(mask, mask)]
                semsim = float(csim.sum() - np.trace(csim)) / v
            else:
                semsim = 1.0
            out.append(
                ClusterResult(
                    cluster_id=cid,
                    name="",
                    classes=sorted(members),
                    coupling=coupling,
                    cohesion=cohesion,
                    semantic_similarity=semsim,
                )
            )
        # Renumber so the IDs are 0..K-1 contiguous.
        for new_id, c in enumerate(out):
            c.cluster_id = new_id
        return out
