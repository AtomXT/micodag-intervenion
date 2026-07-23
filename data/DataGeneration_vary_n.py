#!/usr/bin/env python3
"""Generate a fixed-DAG experiment with varying sample size."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from DataGeneration import (
    assign_edge_weights,
    edge_list,
    environment_covariance,
    generate_ordered_dag,
    moralize,
    sample_datasets,
    sample_intervention_targets,
    save_datasets,
    save_edge_list,
    save_targets,
)


DEFAULT_OUTPUT = Path(__file__).parent / "Vary_n"


def parse_sample_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError("sample sizes must be positive integers")
    return sizes


def generate(
    output_dir: Path = DEFAULT_OUTPUT,
    num_nodes: int = 20,
    sample_sizes: tuple[int, ...] = (100, 500, 1000),
    num_environments: int = 5,
    num_iterations: int = 10,
    seed: int = 200,
    overwrite: bool = False,
) -> None:
    """Generate all sample sizes from one graph and one intervention design."""
    if num_nodes < 2 or num_environments < 1 or num_iterations < 1:
        raise ValueError("nodes, environments, and iterations must be positive")
    if output_dir.exists() and any(output_dir.rglob("*")) and not overwrite:
        raise FileExistsError(
            f"{output_dir} is not empty; use --overwrite to replace its files"
        )

    graph_rng = np.random.default_rng(num_nodes)
    dag = generate_ordered_dag(num_nodes, 2.0 / num_nodes, graph_rng)

    parameter_rng = np.random.default_rng(seed)
    weighted_dag = assign_edge_weights(dag, parameter_rng)
    targets = sample_intervention_targets(
        num_nodes,
        num_environments,
        parameter_rng,
    )

    save_edge_list(
        output_dir / f"DAG_{num_nodes}.txt",
        edge_list(dag, directed=True),
    )
    save_edge_list(
        output_dir / f"Moral_{num_nodes}.txt",
        edge_list(moralize(dag), directed=False),
    )
    save_targets(output_dir / "intervention_targets.txt", targets)

    covariances = [
        environment_covariance(weighted_dag, environment_targets)
        for environment_targets in targets
    ]
    observational_covariance = environment_covariance(
        weighted_dag,
        np.empty(0, dtype=int),
    )

    for sample_size in sample_sizes:
        sample_dir = output_dir / f"n_{sample_size}"
        save_datasets(
            sample_dir / "observational",
            sample_datasets(
                observational_covariance,
                sample_size,
                num_iterations,
            ),
        )
        for environment, covariance in enumerate(covariances, start=1):
            save_datasets(
                sample_dir / f"environment{environment}",
                sample_datasets(covariance, sample_size, num_iterations),
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fixed-p data for a sample-size experiment."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument(
        "--sample-sizes",
        type=parse_sample_sizes,
        default=(100, 500, 1000),
    )
    parser.add_argument("--environments", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate(
        output_dir=args.output_dir,
        num_nodes=args.nodes,
        sample_sizes=args.sample_sizes,
        num_environments=args.environments,
        num_iterations=args.iterations,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(f"Vary-n datasets saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
