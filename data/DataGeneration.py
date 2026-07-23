#!/usr/bin/env python3
"""Generate the synthetic intervention datasets using NumPy.

This is a Python translation of ``DataGeneration.R``.  It preserves the same
data-generating procedure and output layout:

* ordered random DAGs with edge probability ``2 / p``;
* edge weights sampled from ``{-0.8, -0.6, 0.6, 0.8}``;
* observational noise variances sampled from ``{1, 2, 4}``;
* five intervention environments with between 1 and ``floor(p / 2)`` targets;
* hard interventions that remove incoming edges and replace target variances;
* 500 observational samples and ``2 * p`` interventional samples;
* ten independently sampled datasets per environment.

The default output directory is ``SyntheticData`` beside this script.  No
absolute project path or working-directory change is required.

Python and R use different random-number generators, so this script reproduces
the procedure and distributions, not the exact numerical draws in the current
R-generated files.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_NODE_COUNTS = (10, 20, 50, 100, 200)
EDGE_WEIGHTS = np.array((-0.8, -0.6, 0.6, 0.8), dtype=float)
NOISE_VARIANCES = np.array((1.0, 2.0, 4.0), dtype=float)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "SyntheticData"


def parse_node_counts(value: str) -> tuple[int, ...]:
    """Parse a comma-separated list such as ``10,20,50``."""
    node_counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not node_counts or any(count < 2 for count in node_counts):
        raise argparse.ArgumentTypeError("node counts must be integers of at least 2")
    return node_counts


def generate_ordered_dag(
    num_nodes: int,
    edge_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate an ordered DAG, matching the distribution used by randomDAG."""
    if not 0.0 <= edge_probability <= 1.0:
        raise ValueError("edge_probability must be between 0 and 1")

    edge_draws = rng.random((num_nodes, num_nodes)) < edge_probability
    return np.triu(edge_draws, k=1)


def moralize(dag: np.ndarray) -> np.ndarray:
    """Return the undirected moral graph of a directed adjacency matrix."""
    moral = dag | dag.T

    for child in range(dag.shape[0]):
        parents = np.flatnonzero(dag[:, child])
        if parents.size > 1:
            moral[np.ix_(parents, parents)] = True

    np.fill_diagonal(moral, False)
    return moral


def edge_list(adjacency: np.ndarray, *, directed: bool) -> np.ndarray:
    """Convert an adjacency matrix to a one-based, two-column edge list."""
    matrix = adjacency if directed else np.triu(adjacency, k=1)
    edges = np.argwhere(matrix)
    return (edges + 1).astype(int)


def assign_edge_weights(
    dag: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Assign one of the four R-script weights to every directed edge."""
    sampled_weights = rng.choice(EDGE_WEIGHTS, size=dag.shape, replace=True)
    return dag.astype(float) * sampled_weights


def sample_intervention_targets(
    num_nodes: int,
    num_environments: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Draw the intervention targets for each environment."""
    max_targets = num_nodes // 2
    targets_by_environment: list[np.ndarray] = []

    for _ in range(num_environments):
        num_targets = int(rng.integers(1, max_targets + 1))
        targets = rng.choice(num_nodes, size=num_targets, replace=False)
        targets_by_environment.append(targets.astype(int))

    return targets_by_environment


def environment_covariance(
    weighted_dag: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """Construct the covariance matrix for one observational/interventional setting."""
    num_nodes = weighted_dag.shape[0]
    environment_dag = weighted_dag.copy()
    environment_dag[:, targets] = 0.0

    # DataGeneration.R resets the seed to p whenever it constructs an
    # environment covariance, which keeps the baseline variances fixed.
    variance_rng = np.random.default_rng(num_nodes)
    variances = variance_rng.choice(NOISE_VARIANCES, size=num_nodes, replace=True)
    if targets.size:
        variances[targets] = variance_rng.uniform(1.0, 4.0, size=targets.size)

    influence = np.linalg.solve(
        np.eye(num_nodes) - environment_dag,
        np.eye(num_nodes),
    ).T
    covariance = (influence * variances) @ influence.T

    # Remove negligible asymmetry introduced by floating-point operations.
    return (covariance + covariance.T) / 2.0


def sample_datasets(
    covariance: np.ndarray,
    num_samples: int,
    num_iterations: int,
) -> list[np.ndarray]:
    """Sample the repeated zero-mean Gaussian datasets for one environment."""
    if num_samples < 1 or num_iterations < 1:
        raise ValueError("sample and iteration counts must be positive")

    datasets: list[np.ndarray] = []
    mean = np.zeros(covariance.shape[0], dtype=float)

    # The R script resets the seed to the iteration number before each draw.
    for iteration in range(1, num_iterations + 1):
        rng = np.random.default_rng(iteration)
        data = rng.multivariate_normal(
            mean=mean,
            cov=covariance,
            size=num_samples,
            check_valid="raise",
        )
        datasets.append(data)

    return datasets


def save_edge_list(path: Path, edges: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, edges, fmt="%d", delimiter=" ")


def save_targets(path: Path, targets_by_environment: Sequence[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        ",".join(str(int(target) + 1) for target in targets)
        for targets in targets_by_environment
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_datasets(directory: Path, datasets: Sequence[np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for iteration, data in enumerate(datasets, start=1):
        np.savetxt(
            directory / f"data_{iteration}.csv",
            data,
            delimiter=",",
            fmt="%.18g",
        )


def expected_output_paths(
    output_dir: Path,
    node_counts: Sequence[int],
    num_environments: int,
    num_iterations: int,
) -> list[Path]:
    """List files that a generation run will write."""
    paths: list[Path] = []

    for graph_index, num_nodes in enumerate(node_counts, start=1):
        graph_dir = output_dir / f"graph{graph_index}"
        paths.extend(
            (
                output_dir / f"DAG_{num_nodes}.txt",
                output_dir / f"Moral_{num_nodes}.txt",
                graph_dir / "intervention_targets.txt",
            )
        )
        for iteration in range(1, num_iterations + 1):
            paths.append(graph_dir / "observational" / f"data_{iteration}.csv")
            for environment in range(1, num_environments + 1):
                paths.append(
                    graph_dir
                    / f"environment{environment}"
                    / f"data_{iteration}.csv"
                )

    return paths


def generate_all(
    output_dir: Path,
    node_counts: Sequence[int] = DEFAULT_NODE_COUNTS,
    num_environments: int = 5,
    num_iterations: int = 10,
    observational_samples: int = 500,
    interventional_samples: int | None = None,
    seed: int = 200,
    overwrite: bool = False,
) -> None:
    """Generate every graph and dataset from the original experiment design."""
    if num_environments < 1:
        raise ValueError("num_environments must be positive")
    if observational_samples < 1:
        raise ValueError("observational_samples must be positive")
    if interventional_samples is not None and interventional_samples < 1:
        raise ValueError("interventional_samples must be positive when supplied")

    intended_paths = expected_output_paths(
        output_dir,
        node_counts,
        num_environments,
        num_iterations,
    )
    existing_paths = [path for path in intended_paths if path.exists()]
    if existing_paths and not overwrite:
        first = existing_paths[0]
        raise FileExistsError(
            f"{len(existing_paths)} output files already exist; "
            f"the first is {first}. Use --overwrite to replace them."
        )

    dags: list[np.ndarray] = []
    moral_graphs: list[np.ndarray] = []
    for num_nodes in node_counts:
        graph_rng = np.random.default_rng(num_nodes)
        dag = generate_ordered_dag(
            num_nodes=num_nodes,
            edge_probability=2.0 / num_nodes,
            rng=graph_rng,
        )
        dags.append(dag)
        moral_graphs.append(moralize(dag))

    parameter_rng = np.random.default_rng(seed)
    weighted_dags = [assign_edge_weights(dag, parameter_rng) for dag in dags]

    # Match the R loop order: environment first, then graph size.
    targets_by_graph: list[list[np.ndarray]] = [
        [] for _ in node_counts
    ]
    for _ in range(num_environments):
        for graph_index, num_nodes in enumerate(node_counts):
            targets = sample_intervention_targets(
                num_nodes=num_nodes,
                num_environments=1,
                rng=parameter_rng,
            )[0]
            targets_by_graph[graph_index].append(targets)

    for graph_index, num_nodes in enumerate(node_counts, start=1):
        dag = dags[graph_index - 1]
        weighted_dag = weighted_dags[graph_index - 1]
        graph_targets = targets_by_graph[graph_index - 1]
        graph_dir = output_dir / f"graph{graph_index}"

        save_edge_list(
            output_dir / f"DAG_{num_nodes}.txt",
            edge_list(dag, directed=True),
        )
        save_edge_list(
            output_dir / f"Moral_{num_nodes}.txt",
            edge_list(moral_graphs[graph_index - 1], directed=False),
        )
        save_targets(graph_dir / "intervention_targets.txt", graph_targets)

        num_interventional_samples = (
            interventional_samples
            if interventional_samples is not None
            else 2 * num_nodes
        )
        for environment, targets in enumerate(graph_targets, start=1):
            covariance = environment_covariance(weighted_dag, targets)
            datasets = sample_datasets(
                covariance,
                num_samples=num_interventional_samples,
                num_iterations=num_iterations,
            )
            save_datasets(graph_dir / f"environment{environment}", datasets)

        observational_covariance = environment_covariance(
            weighted_dag,
            np.empty(0, dtype=int),
        )
        observational_datasets = sample_datasets(
            observational_covariance,
            num_samples=observational_samples,
            num_iterations=num_iterations,
        )
        save_datasets(graph_dir / "observational", observational_datasets)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate data using the DataGeneration.R procedure."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="output folder (default: data/SyntheticData beside this script)",
    )
    parser.add_argument(
        "--node-counts",
        type=parse_node_counts,
        default=DEFAULT_NODE_COUNTS,
        help="comma-separated graph sizes (default: 10,20,50,100,200)",
    )
    parser.add_argument("--environments", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--observational-samples", type=int, default=500)
    parser.add_argument(
        "--interventional-samples",
        type=int,
        default=None,
        help="samples per intervention environment (default: 2 * node count)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=200,
        help="seed for edge weights and intervention targets",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace files that already exist",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_all(
        output_dir=args.output_dir,
        node_counts=args.node_counts,
        num_environments=args.environments,
        num_iterations=args.iterations,
        observational_samples=args.observational_samples,
        interventional_samples=args.interventional_samples,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Synthetic datasets saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
