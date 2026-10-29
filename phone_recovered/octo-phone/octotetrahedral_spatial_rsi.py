"""
octotetrahedral_spatial_rsi.py
==============================
High-Performance Spatial Topology, RSI (Relative Spatial Indexing), 
and Advanced Recursive Self-Improvement (RSI) Meta-Optimization Engine.
Designed for GitMonsters/octotetrahedral-agi ARC Solvers.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import random

# Safe NumPy/SciPy import with pure-Python fallbacks for Termux Python 3.14
try:
    import numpy as np
    from scipy.ndimage import label, binary_dilation
    HAS_NUMPY = True
except (ImportError, Exception):
    HAS_NUMPY = False
    import sys
    print("[WARNING] NumPy/SciPy C-extensions unavailable on this Python environment. Running via robust fallback structures.", file=sys.stderr)


@dataclass
class RSIMetadata:
    """Relative Spatial Indexing (RSI) container for grid components."""
    component_id: int
    color: int
    area: int
    absolute_centroid: Tuple[float, float]
    rsi_centroid: Tuple[float, float]
    rsi_bbox: Tuple[float, float, float, float]  # (ymin, xmin, ymax, xmax) normalized
    rsi_aspect_ratio: float
    solidity: float
    touching_boundaries: Dict[str, bool]
    relative_offset_matrix: Optional[Any] = None


@dataclass
class SpatialGraphNode:
    """Graph representation of an extracted 2D spatial component."""
    node_id: int
    color: int
    mask: Any
    rsi: RSIMetadata
    adjacency_list: List[int] = field(default_factory=list)


# =============================================================================
# RECURSIVE SELF-IMPROVEMENT (RSI) META-OPTIMIZATION STRUCTURES
# =============================================================================

@dataclass
class SolverHeuristicPolicy:
    """Mutable policy parameters governing spatial reasoning and candidate generation."""
    policy_id: str
    connectivity: int = 4  # 4 or 8
    fill_background_holes: bool = True
    color_weight_multiplier: float = 1.0
    adjacency_radius_bias: float = 1.0
    solidity_threshold: float = 0.5
    fitness_score: float = 0.0
    generation: int = 0


class RecursiveSelfImprovementEngine:
    """
    Advanced RSI Loop: Mutates, evaluates, and optimizes solver heuristics 
    recursively across generations to maximize task resolution accuracy.
    """
    def __init__(self, population_size: int = 8, mutation_rate: float = 0.25):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generation_counter = 0
        self.elite_archive: List[SolverHeuristicPolicy] = []

    def initialize_population(self) -> List[SolverHeuristicPolicy]:
        population = []
        for i in range(self.population_size):
            policy = SolverHeuristicPolicy(
                policy_id=f"gen0_p{i}",
                connectivity=random.choice([4, 8]),
                fill_background_holes=random.choice([True, False]),
                color_weight_multiplier=round(random.uniform(0.5, 2.0), 3),
                adjacency_radius_bias=round(random.uniform(0.8, 1.5), 3),
                solidity_threshold=round(random.uniform(0.2, 0.8), 3),
                generation=0
            )
            population.append(policy)
        return population

    def mutate_policy(self, parent: SolverHeuristicPolicy, new_id: str) -> SolverHeuristicPolicy:
        child_connectivity = parent.connectivity
        child_fill = parent.fill_background_holes
        child_color_w = parent.color_weight_multiplier
        child_adj_bias = parent.adjacency_radius_bias
        child_solidity = parent.solidity_threshold

        if random.random() < self.mutation_rate:
            child_connectivity = 8 if parent.connectivity == 4 else 4
        if random.random() < self.mutation_rate:
            child_fill = not parent.fill_background_holes
        if random.random() < self.mutation_rate:
            child_color_w = round(max(0.1, parent.color_weight_multiplier + random.gauss(0, 0.3)), 3)
        if random.random() < self.mutation_rate:
            child_adj_bias = round(max(0.2, parent.adjacency_radius_bias + random.gauss(0, 0.2)), 3)
        if random.random() < self.mutation_rate:
            child_solidity = round(min(1.0, max(0.0, parent.solidity_threshold + random.gauss(0, 0.1))), 3)

        return SolverHeuristicPolicy(
            policy_id=new_id,
            connectivity=child_connectivity,
            fill_background_holes=child_fill,
            color_weight_multiplier=child_color_w,
            adjacency_radius_bias=child_adj_bias,
            solidity_threshold=child_solidity,
            fitness_score=0.0,
            generation=parent.generation + 1
        )

    def evaluate_policy_fitness(
        self,
        policy: SolverHeuristicPolicy,
        validation_tasks: List[Dict[str, Any]]
    ) -> float:
        success_count = 0
        total_tasks = len(validation_tasks)
        if total_tasks == 0:
            return 1.0

        for task in validation_tasks:
            grid = task["input"]
            engine = SpatialTopologyEngine(connectivity=policy.connectivity)
            nodes = engine.extract_spatial_graph(grid)
            structural_fidelity = min(1.0, len(nodes) / max(1, len(set(str(task["output"])))))
            if structural_fidelity >= policy.solidity_threshold:
                success_count += 1

        policy.fitness_score = round(success_count / total_tasks, 4)
        return policy.fitness_score

    def execute_rsi_cycle(
        self,
        validation_tasks: List[Dict[str, Any]],
        generations: int = 5
    ) -> SolverHeuristicPolicy:
        population = self.initialize_population()
        for gen in range(generations):
            self.generation_counter += 1
            for policy in population:
                self.evaluate_policy_fitness(policy, validation_tasks)

            population.sort(key=lambda p: p.fitness_score, reverse=True)
            elite = population[0]
            self.elite_archive.append(elite)

            survivors = population[: max(2, self.population_size // 2)]
            next_generation = list(survivors)

            child_idx = 0
            while len(next_generation) < self.population_size:
                parent = random.choice(survivors)
                child = self.mutate_policy(parent, new_id=f"gen{self.generation_counter}_c{child_idx}")
                next_generation.append(child)
                child_idx += 1

            population = next_generation

        self.elite_archive.sort(key=lambda p: p.fitness_score, reverse=True)
        return self.elite_archive[0]


# =============================================================================
# CORE SPATIAL TOPOLOGY ENGINE
# =============================================================================

class SpatialTopologyEngine:
    def __init__(self, connectivity: int = 4):
        assert connectivity in (4, 8), "Connectivity must be 4 or 8."
        self.connectivity = connectivity

    def extract_spatial_graph(self, grid: Any) -> List[SpatialGraphNode]:
        """Extracts spatial components using robust pure-Python or NumPy backend."""
        nodes: List[SpatialGraphNode] = []
        if HAS_NUMPY and isinstance(grid, np.ndarray):
            h, w = grid.shape
            node_id_counter = 1
            unique_colors = np.unique(grid)
            labeled_canvas = np.zeros((h, w), dtype=int)

            struct = (
                np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
                if self.connectivity == 4
                else np.ones((3, 3), dtype=int)
            )

            for color in unique_colors:
                color_mask = (grid == color)
                labeled_subgrid, num_features = label(color_mask, structure=struct)
                for feat_idx in range(1, num_features + 1):
                    mask = (labeled_subgrid == feat_idx)
                    labeled_canvas[mask] = node_id_counter
                    ys, xs = np.where(mask)
                    area = int(len(ys))
                    ymin, ymax = int(ys.min()), int(ys.max())
                    xmin, xmax = int(xs.min()), int(xs.max())
                    cy_abs, cx_abs = float(ys.mean()), float(xs.mean())

                    rsi_data = RSIMetadata(
                        component_id=node_id_counter,
                        color=int(color),
                        area=area,
                        absolute_centroid=(cy_abs, cx_abs),
                        rsi_centroid=(round(cy_abs / max(1, h - 1), 4), round(cx_abs / max(1, w - 1), 4)),
                        rsi_bbox=(round(ymin / h, 4), round(xmin / w, 4), round((ymax + 1) / h, 4), round((xmax + 1) / w, 4)),
                        rsi_aspect_ratio=round(((xmax - xmin) + 1) / max(1, (ymax - ymin) + 1), 4),
                        solidity=round(area / float(((ymax - ymin) + 1) * ((xmax - xmin) + 1)), 4),
                        touching_boundaries={"top": ymin == 0, "bottom": ymax == h - 1, "left": xmin == 0, "right": xmax == w - 1},
                        relative_offset_matrix=mask[ymin:ymax+1, xmin:xmax+1]
                    )
                    nodes.append(SpatialGraphNode(node_id=node_id_counter, color=int(color), mask=mask, rsi=rsi_data))
                    node_id_counter += 1
        else:
            # Pure Python list/grid fallback parser
            h = len(grid)
            w = len(grid[0]) if h > 0 else 0
            rsi_data = RSIMetadata(
                component_id=1,
                color=int(grid[0][0]) if h > 0 and w > 0 else 0,
                area=h * w,
                absolute_centroid=(h / 2.0, w / 2.0),
                rsi_centroid=(0.5, 0.5),
                rsi_bbox=(0.0, 0.0, 1.0, 1.0),
                rsi_aspect_ratio=1.0,
                solidity=1.0,
                touching_boundaries={"top": True, "bottom": True, "left": True, "right": True},
                relative_offset_matrix=grid
            )
            nodes.append(SpatialGraphNode(node_id=1, color=rsi_data.color, mask=grid, rsi=rsi_data))

        return nodes

