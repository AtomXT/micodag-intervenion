#!/usr/bin/env python3
"""
Generate synthetic multi-environment datasets for intervention-aware DAG learning.

Each saved instance contains:
- a baseline weighted DAG B,
- its moral graph,
- one observational dataset,
- several interventional datasets,
- intervention masks,
- metadata as JSON.

The default settings mirror the current project design:
- random DAG with edge probability about 2 / p,
- edge weights sampled from {-0.8, -0.6, 0.6, 0.8},
- observational noise variances sampled from {1, 2, 4},
- one observational environment plus several hard-intervention environments,
- hard interventions remove incoming edges into targeted nodes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class InstanceMeta:
    num_nodes: int
    rep: int
    seed: int
    edge_prob: float
    num_edges: int
    num_intervention_envs: int
    obs_samples: int
    int_samples: int
    intervention_sizes: list[int]
    intervention_type: str


def parse_int_list(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def sample_random_dag(
    num_nodes: int,
    edge_prob: float,
    weight_values: list[float],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(num_nodes)
    B = np.zeros((num_nodes, num_nodes), dtype=float)

    for idx, parent in enumerate(order[:-1]):
        children = order[idx + 1 :]
        keep = rng.random(children.size) < edge_prob
        if np.any(keep):
            B[parent, children[keep]] = rng.choice(weight_values, size=int(keep.sum()))

    return B, order


def moral_graph(B: np.ndarray) -> np.ndarray:
    adjacency = B != 0
    moral = adjacency | adjacency.T

    for child in range(B.shape[0]):
        parents = np.flatnonzero(adjacency[:, child])
        if parents.size > 1:
            moral[np.ix_(parents, parents)] = True

    np.fill_diagonal(moral, False)
    return moral.astype(np.int8)


def sample_intervention_mask(
    num_nodes: int,
    num_envs: int,
    min_targets: int,
    max_targets: int,
    rng: np.random.Generator,
) -> np.ndarray:
    mask = np.zeros((num_envs, num_nodes), dtype=np.int8)

    for env in range(num_envs):
        size = int(rng.integers(min_targets, max_targets + 1))
        targets = rng.choice(num_nodes, size=size, replace=False)
        mask[env, targets] = 1

    return mask


def sample_environment(
    B: np.ndarray,
    order: np.ndarray,
    base_noise_vars: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    intervention_mask: np.ndarray | None = None,
    intervention_noise_low: float = 1.0,
    intervention_noise_high: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    env_B = B.copy()
    env_noise_vars = base_noise_vars.copy()

    if intervention_mask is not None:
        targets = intervention_mask.astype(bool)
        env_B[:, targets] = 0.0
        env_noise_vars[targets] = rng.uniform(
            intervention_noise_low,
            intervention_noise_high,
            size=int(targets.sum()),
        )

    eps = rng.normal(scale=np.sqrt(env_noise_vars), size=(n_samples, B.shape[0]))
    X = np.zeros_like(eps)

    for node in order:
        X[:, node] = eps[:, node] + X @ env_B[:, node]

    return X, env_noise_vars


def save_instance(
    out_path: Path,
    B: np.ndarray,
    moral: np.ndarray,
    order: np.ndarray,
    base_noise_vars: np.ndarray,
    observational_data: np.ndarray,
    intervention_mask: np.ndarray,
    interventional_data: list[np.ndarray],
    interventional_noise_vars: list[np.ndarray],
    meta: InstanceMeta,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, np.ndarray] = {
        "B": B.astype(np.float64),
        "moral": moral.astype(np.int8),
        "topological_order": order.astype(np.int64),
        "base_noise_vars": base_noise_vars.astype(np.float64),
        "observational_data": observational_data.astype(np.float64),
        "intervention_mask": intervention_mask.astype(np.int8),
        "meta_json": np.array(json.dumps(asdict(meta)), dtype=object),
    }

    for env_idx, data in enumerate(interventional_data, start=1):
        payload[f"environment_{env_idx:02d}_data"] = data.astype(np.float64)
        payload[f"environment_{env_idx:02d}_noise_vars"] = interventional_noise_vars[
            env_idx - 1
        ].astype(np.float64)

    np.savez_compressed(out_path, **payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./data/generated")
    parser.add_argument("--num_nodes_list", type=str, default="10,20,50,100,200")
    parser.add_argument("--reps", type=int, default=10)
    parser.add_argument("--num_intervention_envs", type=int, default=5)
    parser.add_argument("--obs_samples", type=int, default=500)
    parser.add_argument(
        "--int_samples",
        type=int,
        default=0,
        help="Samples per interventional environment. If 0, use round(2 * num_nodes).",
    )
    parser.add_argument(
        "--edge_prob",
        type=float,
        default=-1.0,
        help="If negative, use edge_scale / num_nodes.",
    )
    parser.add_argument("--edge_scale", type=float, default=2.0)
    parser.add_argument("--weight_values", type=str, default="-0.8,-0.6,0.6,0.8")
    parser.add_argument("--noise_values", type=str, default="1,2,4")
    parser.add_argument("--min_targets", type=int, default=1)
    parser.add_argument(
        "--max_targets",
        type=int,
        default=0,
        help="Maximum intervention targets per environment. If 0, use floor(num_nodes / 2).",
    )
    parser.add_argument("--intervention_noise_low", type=float, default=1.0)
    parser.add_argument("--intervention_noise_high", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    num_nodes_list = parse_int_list(args.num_nodes_list)
    weight_values = parse_float_list(args.weight_values)
    noise_values = parse_float_list(args.noise_values)
    out_dir = Path(args.out_dir)

    rng_master = np.random.default_rng(args.seed)
    manifest: list[dict[str, object]] = []

    for num_nodes in num_nodes_list:
        edge_prob = args.edge_prob if args.edge_prob >= 0 else args.edge_scale / num_nodes
        int_samples = args.int_samples if args.int_samples > 0 else int(round(2 * num_nodes))
        max_targets = args.max_targets if args.max_targets > 0 else max(1, num_nodes // 2)

        for rep in range(args.reps):
            seed = int(rng_master.integers(0, 2**31 - 1))
            rng = np.random.default_rng(seed)

            B, order = sample_random_dag(
                num_nodes=num_nodes,
                edge_prob=edge_prob,
                weight_values=weight_values,
                rng=rng,
            )
            moral = moral_graph(B)
            base_noise_vars = rng.choice(noise_values, size=num_nodes, replace=True).astype(float)
            intervention_mask = sample_intervention_mask(
                num_nodes=num_nodes,
                num_envs=args.num_intervention_envs,
                min_targets=args.min_targets,
                max_targets=max_targets,
                rng=rng,
            )

            observational_data, _ = sample_environment(
                B=B,
                order=order,
                base_noise_vars=base_noise_vars,
                n_samples=args.obs_samples,
                rng=rng,
            )

            interventional_data: list[np.ndarray] = []
            interventional_noise_vars: list[np.ndarray] = []
            for env_mask in intervention_mask:
                env_data, env_noise_vars = sample_environment(
                    B=B,
                    order=order,
                    base_noise_vars=base_noise_vars,
                    n_samples=int_samples,
                    rng=rng,
                    intervention_mask=env_mask,
                    intervention_noise_low=args.intervention_noise_low,
                    intervention_noise_high=args.intervention_noise_high,
                )
                interventional_data.append(env_data)
                interventional_noise_vars.append(env_noise_vars)

            meta = InstanceMeta(
                num_nodes=num_nodes,
                rep=rep,
                seed=seed,
                edge_prob=float(edge_prob),
                num_edges=int(np.count_nonzero(B)),
                num_intervention_envs=args.num_intervention_envs,
                obs_samples=args.obs_samples,
                int_samples=int_samples,
                intervention_sizes=intervention_mask.sum(axis=1).astype(int).tolist(),
                intervention_type="hard",
            )

            out_path = out_dir / f"m={num_nodes}" / f"instance_m{num_nodes}_rep{rep:02d}.npz"
            save_instance(
                out_path=out_path,
                B=B,
                moral=moral,
                order=order,
                base_noise_vars=base_noise_vars,
                observational_data=observational_data,
                intervention_mask=intervention_mask,
                interventional_data=interventional_data,
                interventional_noise_vars=interventional_noise_vars,
                meta=meta,
            )

            manifest.append({"path": str(out_path), **asdict(meta)})

    manifest_path = out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved {len(manifest)} instances.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
