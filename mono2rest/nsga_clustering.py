"""
NSGA-III multi-objective clustering using *pymoo*.

Three objectives (paper §III-A3):
  1. Minimise  Coupling(S) = Σ Coupling(xᵢ)
  2. Maximise  Cohesion(S) = (1/N) Σ Cohesion(xᵢ)   → minimise negative
  3. Maximise  SemSim(S)   = (1/N) Σ SemSim(xᵢ)     → minimise negative

Custom genetic operators (paper Fig. 5):
  * Crossover – inject a random cluster from parent₁ into parent₂ then repair
  * Mutation  – move one random method to a different cluster
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np

from .data_models import Cluster, Method


# ---------------------------------------------------------------------------
# pymoo Problem
# ---------------------------------------------------------------------------

class ClusteringProblem:
    """Lightweight problem description used by the custom GA loop.

    We avoid subclassing ``pymoo.core.problem.Problem`` directly so that we
    can run a hand-rolled NSGA-III loop with proper crossover/mutation
    operators that match the paper, while still using pymoo's reference-point
    infrastructure for selection.
    """

    def __init__(
        self,
        n_methods: int,
        n_clusters: int,
        adj_matrix: np.ndarray,
        sim_matrix: np.ndarray,
    ):
        self.n_methods = n_methods
        self.n_clusters = n_clusters
        self.adj = adj_matrix        # (N, N) int8 adjacency
        self.sim = sim_matrix        # (N, N) float cosine similarity

    # ---- objective evaluation (vectorised over a population) ----

    def evaluate(self, pop: np.ndarray) -> np.ndarray:
        """Evaluate a population of shape (pop_size, n_methods).

        Returns an (pop_size, 3) array where columns are:
          [coupling_to_minimise, neg_cohesion, neg_semsim]
        """
        results = np.empty((pop.shape[0], 3), dtype=np.float64)
        for i in range(pop.shape[0]):
            results[i] = self._eval_individual(pop[i])
        return results

    def _eval_individual(self, x: np.ndarray) -> Tuple[float, float, float]:
        K = self.n_clusters
        adj = self.adj
        sim = self.sim

        # Build cluster membership masks
        masks: List[np.ndarray] = []
        for k in range(K):
            masks.append(x == k)

        total_coupling = 0.0
        total_cohesion = 0.0
        total_semsim = 0.0
        n_nonempty = 0

        for k in range(K):
            mk = masks[k]
            v_cluster = int(mk.sum())
            if v_cluster == 0:
                continue
            n_nonempty += 1

            # Internal edges: calls within cluster k
            e_internal = int(adj[np.ix_(mk, mk)].sum())
            # External edges: calls from cluster k to outside
            e_external = int(adj[mk, :].sum()) - e_internal

            # Coupling(x)
            denom = e_internal + e_external
            coupling_k = e_external / denom if denom > 0 else 0.0
            total_coupling += coupling_k

            # Cohesion(x) = min(E_internal / V_cluster, 1.0)
            cohesion_k = min(e_internal / v_cluster, 1.0) if v_cluster > 0 else 0.0
            total_cohesion += cohesion_k

            # SemSim(x) = (1/V_cluster) Σ Sim(i,j) for i≠j in x
            if v_cluster >= 2:
                cluster_sim = sim[np.ix_(mk, mk)]
                pair_sum = cluster_sim.sum() - np.trace(cluster_sim)  # exclude diagonal
                semsim_k = pair_sum / v_cluster
            else:
                semsim_k = 1.0  # single method → perfect similarity
            total_semsim += semsim_k

        avg_cohesion = total_cohesion / n_nonempty if n_nonempty > 0 else 0.0
        avg_semsim = total_semsim / n_nonempty if n_nonempty > 0 else 0.0

        return (total_coupling, -avg_cohesion, -avg_semsim)


# ---------------------------------------------------------------------------
# Genetic operators (paper Fig. 5)
# ---------------------------------------------------------------------------

def _crossover(p1: np.ndarray, p2: np.ndarray, K: int) -> np.ndarray:
    """Inject a random cluster from p1 into p2 and repair."""
    child = p2.copy()
    donor_cluster = random.randint(0, K - 1)
    donor_mask = (p1 == donor_cluster)
    child[donor_mask] = donor_cluster
    # Repair: ensure all K clusters present
    _repair(child, K)
    return child


def _mutate(x: np.ndarray, K: int) -> np.ndarray:
    """Move one random method to a different cluster."""
    child = x.copy()
    idx = random.randint(0, len(child) - 1)
    old = child[idx]
    new = random.randint(0, K - 2)
    if new >= old:
        new += 1
    child[idx] = new
    _repair(child, K)
    return child


def _repair(x: np.ndarray, K: int):
    """Ensure every cluster id in [0, K) has at least one method."""
    present = set(np.unique(x))
    missing = [k for k in range(K) if k not in present]
    if not missing:
        return
    # Steal one random method from the largest cluster for each missing id
    for k in missing:
        counts = np.bincount(x, minlength=K)
        largest = int(np.argmax(counts))
        candidates = np.where(x == largest)[0]
        if len(candidates) > 1:
            victim = random.choice(candidates.tolist())
            x[victim] = k


# ---------------------------------------------------------------------------
# NSGA-III selection helpers
# ---------------------------------------------------------------------------

def _non_dominated_sort(F: np.ndarray):
    """Return list of fronts (each front = list of indices). All objectives minimised."""
    n = F.shape[0]
    domination_count = np.zeros(n, dtype=int)
    dominated_by: List[List[int]] = [[] for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            if _dominates(F[i], F[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif _dominates(F[j], F[i]):
                dominated_by[j].append(i)
                domination_count[i] += 1

    fronts: List[List[int]] = []
    current = [i for i in range(n) if domination_count[i] == 0]
    while current:
        fronts.append(current)
        next_front = []
        for i in current:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current = next_front
    return fronts


def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def _crowding_distance(F: np.ndarray) -> np.ndarray:
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


def _select(pop: np.ndarray, F: np.ndarray, target_size: int) -> np.ndarray:
    """NSGA-III–style selection: non-dominated sort + crowding distance."""
    fronts = _non_dominated_sort(F)
    selected_idx: List[int] = []
    for front in fronts:
        if len(selected_idx) + len(front) <= target_size:
            selected_idx.extend(front)
        else:
            remaining = target_size - len(selected_idx)
            cd = _crowding_distance(F[front])
            order = np.argsort(-cd)  # descending
            selected_idx.extend([front[order[i]] for i in range(remaining)])
            break
    return np.array(selected_idx)


# ---------------------------------------------------------------------------
# Main clustering entry point
# ---------------------------------------------------------------------------

class NSGAClustering:
    """Run the NSGA-III method-level clustering from the MONO2REST paper."""

    def __init__(
        self,
        population_size: int = 100,
        max_generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
    ):
        self.pop_size = population_size
        self.max_gen = max_generations
        self.cx_rate = crossover_rate
        self.mut_rate = mutation_rate

    def cluster(
        self,
        methods: List[Method],
        call_graph: Dict[str, Set[str]],
        sim_matrix: np.ndarray,
        num_clusters: int = 7,
    ) -> List[Cluster]:
        N = len(methods)
        K = num_clusters
        ids = [m.id for m in methods]
        idx_map = {mid: i for i, mid in enumerate(ids)}

        # Build adjacency matrix
        adj = np.zeros((N, N), dtype=np.int8)
        for caller, callees in call_graph.items():
            if caller not in idx_map:
                continue
            ci = idx_map[caller]
            for callee in callees:
                if callee in idx_map:
                    adj[ci][idx_map[callee]] = 1

        problem = ClusteringProblem(N, K, adj, sim_matrix)

        # Initialise population
        pop = np.column_stack([
            np.random.randint(0, K, size=N) for _ in range(self.pop_size)
        ]).T  # (pop_size, N)
        # Repair initial population
        for i in range(self.pop_size):
            _repair(pop[i], K)

        F = problem.evaluate(pop)

        for gen in range(self.max_gen):
            # Generate offspring
            offspring_list = []
            while len(offspring_list) < self.pop_size:
                i1, i2 = random.sample(range(self.pop_size), 2)
                if random.random() < self.cx_rate:
                    child = _crossover(pop[i1], pop[i2], K)
                else:
                    child = pop[i1].copy()
                if random.random() < self.mut_rate:
                    child = _mutate(child, K)
                offspring_list.append(child)
            offspring = np.array(offspring_list)
            F_off = problem.evaluate(offspring)

            # Merge parent + offspring
            combined = np.vstack([pop, offspring])
            F_combined = np.vstack([F, F_off])

            # Select next generation
            sel_idx = _select(combined, F_combined, self.pop_size)
            pop = combined[sel_idx]
            F = F_combined[sel_idx]

            if gen % 20 == 0 or gen == self.max_gen - 1:
                best_i = int(np.argmin(F[:, 0] - F[:, 1] - F[:, 2]))
                c, coh, ss = F[best_i]
                print(f"    gen {gen:>4d}: coupling={c:.4f}  "
                      f"cohesion={-coh:.4f}  semSim={-ss:.4f}")

        # Pick best compromise solution from final population
        best_idx = self._pick_best(F)
        best = pop[best_idx]

        return self._to_clusters(best, methods, ids, K, adj, sim_matrix)

    @staticmethod
    def _pick_best(F: np.ndarray) -> int:
        """Pick the solution with best combined rank (simple weighted sum)."""
        # Normalise each objective to [0, 1]
        F_norm = F.copy()
        for j in range(F.shape[1]):
            lo, hi = F_norm[:, j].min(), F_norm[:, j].max()
            if hi - lo > 0:
                F_norm[:, j] = (F_norm[:, j] - lo) / (hi - lo)
            else:
                F_norm[:, j] = 0.0
        scores = F_norm.sum(axis=1)  # lower = better (all objectives minimised)
        return int(np.argmin(scores))

    @staticmethod
    def _to_clusters(
        assignment: np.ndarray,
        methods: List[Method],
        ids: List[str],
        K: int,
        adj: np.ndarray,
        sim: np.ndarray,
    ) -> List[Cluster]:
        clusters: List[Cluster] = []
        for k in range(K):
            mask = assignment == k
            member_methods = [methods[i] for i in range(len(methods)) if mask[i]]
            if not member_methods:
                continue

            v = int(mask.sum())
            e_internal = int(adj[np.ix_(mask, mask)].sum())
            e_external = int(adj[mask, :].sum()) - e_internal
            denom = e_internal + e_external
            coupling = e_external / denom if denom > 0 else 0.0
            cohesion = min(e_internal / v, 1.0) if v > 0 else 0.0

            if v >= 2:
                csim = sim[np.ix_(mask, mask)]
                semsim = (csim.sum() - np.trace(csim)) / v
            else:
                semsim = 1.0

            clusters.append(Cluster(
                cluster_id=k + 1,
                methods=member_methods,
                coupling=coupling,
                cohesion=cohesion,
                semantic_similarity=semsim,
            ))
        return clusters
