"""CLI configuration for function-approximation experiments."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch

from experiments.common import (
    RECORDING_CHECKPOINTS,
    evaluation_checkpoints,
    resolve_seed_values,
)
from experiments.function_approximation.environments import (
    KNOWN_ENVIRONMENTS,
    experiment_defaults,
    make_environment,
    progress_configuration,
    recording_frame_stride,
)

ALGORITHMS = ("td_prediction", "sarsa", "q_learning")
DEFAULT_HIDDEN_SIZES = (64, 64)


@dataclass(frozen=True)
class ExperimentConfig:
    environment: str
    preset: str
    algorithm: str
    training_episodes: int
    evaluation_episodes: int
    validation_episodes: int
    seeds: int
    seed_values: tuple[int, ...]
    learning_rate: float
    minimum_learning_rate: float
    discount: float
    rollout_steps: int
    epsilon: float
    sarsa_final_epsilon: float
    optimizer: str
    hidden_sizes: tuple[int, ...]
    recording_mode: str
    recording_checkpoints: tuple[int, ...]
    checkpoints: tuple[int, ...]
    diagnostics: bool
    frame_stride: int
    observation_shape: list[int]
    original_observation_low: list[float] | None
    original_observation_high: list[float] | None
    observations_rescaled: bool
    observation_processing: str
    number_of_actions: int
    max_episode_steps: int | None
    progress_label: str
    progress_goal: float
    success_definition: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run semi-gradient prediction or control on flattenable-observation "
            "Gymnasium environments with discrete actions."
        )
    )
    parser.add_argument("--environment", "--env", "-e", default=KNOWN_ENVIRONMENTS[0])
    parser.add_argument(
        "--preset", "-p", choices=("quick", "tuning", "standard"), default="standard"
    )
    parser.add_argument(
        "--algorithm",
        choices=ALGORITHMS,
        required=True,
        help="single algorithm evaluated by this run",
    )
    parser.add_argument("--training-episodes", "--train", type=int)
    parser.add_argument("--evaluation-episodes", "--eval", type=int)
    parser.add_argument("--seeds", "-n", type=int)
    parser.add_argument(
        "--seed-base",
        type=int,
        help="first paired seed (default: 0 for quick/tuning; randomized for standard)",
    )
    parser.add_argument("--learning-rate", "--lr", type=float, default=0.001)
    parser.add_argument(
        "--minimum-learning-rate",
        "--lr-min",
        type=float,
        help="minimum cosine learning rate (default: 1%% of --lr)",
    )
    parser.add_argument(
        "--validation-episodes",
        "--val",
        type=int,
        help="fixed-seed episodes for best-checkpoint selection (default: up to 5)",
    )
    parser.add_argument("--discount", "--gamma", type=float, default=0.99)
    parser.add_argument(
        "--rollout-steps",
        "--rollout",
        "--n-steps",
        "--n-step",
        dest="rollout_steps",
        type=int,
        default=1,
        help="maximum transitions per semi-gradient TD rollout (default: 1)",
    )
    parser.add_argument("--epsilon", "--eps", type=float, default=0.1)
    parser.add_argument(
        "--sarsa-final-epsilon",
        "--sarsa-eps-final",
        "--final-epsilon",
        "--eps-final",
        type=float,
        help="final SARSA epsilon (default: min(0.01, --eps))",
    )
    parser.add_argument(
        "--optimizer",
        "--opt",
        "-o",
        choices=("sgd", "adam"),
        default="adam",
    )
    parser.add_argument(
        "--hidden-sizes",
        "--hidden",
        type=int,
        nargs="+",
        default=DEFAULT_HIDDEN_SIZES,
        metavar="N",
    )
    parser.add_argument(
        "--recordings",
        "--record",
        "-r",
        choices=("none", "final", "checkpoints"),
    )
    parser.add_argument(
        "--diagnostics",
        "--diag",
        action="store_true",
        help="write per-training-episode value-learning diagnostics",
    )
    return parser


def parse_config(argv: Sequence[str] | None = None) -> ExperimentConfig:
    parser = _parser()
    args = parser.parse_args(argv)
    defaults = experiment_defaults(args.environment, args.preset)
    training_episodes = (
        args.training_episodes
        if args.training_episodes is not None
        else defaults["training_episodes"]
    )
    evaluation_episodes = (
        args.evaluation_episodes
        if args.evaluation_episodes is not None
        else defaults["evaluation_episodes"]
    )
    seeds = args.seeds if args.seeds is not None else defaults["seeds"]
    validation_episodes = (
        min(5, evaluation_episodes)
        if args.validation_episodes is None
        else args.validation_episodes
    )
    if min(training_episodes, evaluation_episodes, validation_episodes, seeds) < 1:
        parser.error(
            "training, validation, evaluation episodes, and seeds must be positive"
        )
    try:
        seed_values = resolve_seed_values(seeds, args.preset, args.seed_base)
    except ValueError as error:
        parser.error(str(error))
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning rate must be finite and positive")
    minimum_learning_rate = (
        args.learning_rate * 0.01
        if args.minimum_learning_rate is None
        else args.minimum_learning_rate
    )
    if (
        not np.isfinite(minimum_learning_rate)
        or minimum_learning_rate < 0
        or minimum_learning_rate > args.learning_rate
    ):
        parser.error("minimum learning rate must be finite and between 0 and --lr")
    if not np.isfinite(args.discount) or not 0 <= args.discount <= 1:
        parser.error("discount must be finite and in [0, 1]")
    if args.rollout_steps < 1:
        parser.error("rollout steps must be positive")
    if not np.isfinite(args.epsilon) or not 0 <= args.epsilon <= 1:
        parser.error("epsilon must be finite and in [0, 1]")
    sarsa_final_epsilon = (
        min(0.01, args.epsilon)
        if args.sarsa_final_epsilon is None
        else args.sarsa_final_epsilon
    )
    if not np.isfinite(sarsa_final_epsilon) or not 0 <= sarsa_final_epsilon <= 1:
        parser.error("final SARSA epsilon must be finite and in [0, 1]")
    if sarsa_final_epsilon > args.epsilon:
        parser.error("final SARSA epsilon must not exceed initial epsilon")
    if any(hidden_size <= 0 for hidden_size in args.hidden_sizes):
        parser.error("hidden sizes must be positive")

    is_prediction = args.algorithm == "td_prediction"
    if is_prediction and args.diagnostics:
        parser.error("td_prediction does not provide training diagnostics")
    recording_mode = (
        "none"
        if is_prediction
        else {
            "quick": "none",
            "tuning": "none",
            "standard": "checkpoints",
        }[args.preset]
    )
    if args.recordings is not None and args.recordings != recording_mode:
        parser.error(f"{args.preset} mode requires --recordings {recording_mode}")
    recording_checkpoints = {
        "none": (),
        "final": (100,),
        "checkpoints": RECORDING_CHECKPOINTS,
    }[recording_mode]

    try:
        inspection_env = make_environment(args.environment)
    except (gym.error.Error, ValueError) as error:
        parser.error(str(error))
    assert isinstance(inspection_env.observation_space, gym.spaces.Box)
    assert isinstance(inspection_env.action_space, gym.spaces.Discrete)
    original_space = inspection_env.unwrapped.observation_space
    observations_rescaled = bool(
        isinstance(original_space, gym.spaces.Box)
        and np.all(np.isfinite(original_space.low))
        and np.all(np.isfinite(original_space.high))
    )
    original_low = (
        original_space.low.tolist()
        if isinstance(original_space, gym.spaces.Box)
        else None
    )
    original_high = (
        original_space.high.tolist()
        if isinstance(original_space, gym.spaces.Box)
        else None
    )
    observation_processing = (
        "rescaled to [-1, 1] then flattened"
        if observations_rescaled
        else (
            "flattened without scaling"
            if isinstance(original_space, gym.spaces.Box)
            else "flattened one-hot encoding"
        )
    )
    if (
        args.preset == "standard"
        and not is_prediction
        and "rgb_array" not in inspection_env.metadata.get("render_modes", [])
    ):
        inspection_env.close()
        parser.error("standard mode requires an environment with rgb_array rendering")
    max_episode_steps = (
        inspection_env.spec.max_episode_steps
        if inspection_env.spec is not None
        else None
    )
    progress_label, progress_goal, success_definition = progress_configuration(
        args.environment, max_episode_steps
    )
    config = ExperimentConfig(
        environment=args.environment,
        preset=args.preset,
        algorithm=args.algorithm,
        training_episodes=training_episodes,
        evaluation_episodes=evaluation_episodes,
        validation_episodes=validation_episodes,
        seeds=seeds,
        seed_values=seed_values,
        learning_rate=args.learning_rate,
        minimum_learning_rate=minimum_learning_rate,
        discount=args.discount,
        rollout_steps=args.rollout_steps,
        epsilon=args.epsilon,
        sarsa_final_epsilon=sarsa_final_epsilon,
        optimizer=args.optimizer,
        hidden_sizes=tuple(args.hidden_sizes),
        recording_mode=recording_mode,
        recording_checkpoints=recording_checkpoints,
        checkpoints=evaluation_checkpoints(args.preset),
        diagnostics=args.diagnostics,
        frame_stride=recording_frame_stride(args.environment),
        observation_shape=list(inspection_env.observation_space.shape),
        original_observation_low=original_low,
        original_observation_high=original_high,
        observations_rescaled=observations_rescaled,
        observation_processing=observation_processing,
        number_of_actions=int(inspection_env.action_space.n),
        max_episode_steps=max_episode_steps,
        progress_label=progress_label,
        progress_goal=progress_goal,
        success_definition=success_definition,
    )
    inspection_env.close()
    return config


def initial_metadata(config: ExperimentConfig) -> dict[str, object]:
    metadata: dict[str, object] = {
        "status": "running",
        "environment": config.environment,
        "preset": config.preset,
        "algorithm": config.algorithm,
        "training_episodes": config.training_episodes,
        "evaluation_episodes": config.evaluation_episodes,
        "seeds": config.seeds,
        "seed_values": list(config.seed_values),
        "observation_shape": config.observation_shape,
        "original_observation_low": config.original_observation_low,
        "original_observation_high": config.original_observation_high,
        "observation_scaling": ([-1.0, 1.0] if config.observations_rescaled else None),
        "observation_processing": config.observation_processing,
        "number_of_actions": config.number_of_actions,
        "hidden_sizes": list(config.hidden_sizes),
        "optimizer": "SGD" if config.optimizer == "sgd" else "Adam",
        "learning_rate": config.learning_rate,
        "minimum_learning_rate": config.minimum_learning_rate,
        "learning_rate_schedule": "cosine annealing once per training episode",
        "discount": config.discount,
        "rollout_steps": config.rollout_steps,
        "initial_epsilon": config.epsilon,
        "sarsa_final_epsilon": config.sarsa_final_epsilon,
        "sarsa_epsilon_schedule": (
            "constant"
            if config.sarsa_final_epsilon == config.epsilon
            else "linear over training episodes"
        ),
        "q_learning_final_epsilon": config.epsilon,
        "q_learning_epsilon_schedule": "constant",
        "max_episode_steps": config.max_episode_steps,
        "evaluation_checkpoints": list(config.checkpoints),
        "validation_episodes": config.validation_episodes,
        "validation_environment_seed": (
            "500000 + seed * 100000 + validation_episode; fixed across checkpoints"
        ),
        "validation_selection": (
            "lexicographic maximum of mean success, return, then progress"
        ),
        "reported_checkpoint_policy": (
            "best validation checkpoint at or before each training budget"
        ),
        "recording_checkpoints": list(config.recording_checkpoints),
        "recording_mode": config.recording_mode,
        "recording_environment_seed": (
            "2000000 + checkpoint" if config.recording_checkpoints else None
        ),
        "recording_frame_stride": (
            config.frame_stride if config.recording_checkpoints else None
        ),
        "training_diagnostics": config.diagnostics,
        "training_diagnostics_file": (
            "training_diagnostics.csv" if config.diagnostics else None
        ),
        "diagnostic_action_values": (
            "all action values before each update" if config.diagnostics else None
        ),
        "training_environment_seed": "seed * training_episodes + episode",
        "evaluation_environment_seed": (
            "1000000 + seed * 100000 + checkpoint * 1000 + evaluation_episode"
        ),
        "evaluation_policy": "frozen greedy with deterministic first-index tie break",
        "success_definition": config.success_definition,
        "progress_metric": config.progress_label,
        "progress_goal": config.progress_goal,
        "truncation_target": "bootstrap then stop interaction",
        "variability": "standard deviation across seed evaluation means",
        "figure_variability": "individual seed curves plus across-seed mean",
        "smoothing_window": 1,
        "versions": {
            "gymnasium": gym.__version__,
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }
    if config.algorithm == "td_prediction":
        metadata.update(
            {
                "initial_epsilon": None,
                "sarsa_final_epsilon": None,
                "sarsa_epsilon_schedule": None,
                "q_learning_final_epsilon": None,
                "q_learning_epsilon_schedule": None,
                "prediction_policy": "uniform random",
                "validation_episodes": None,
                "validation_environment_seed": None,
                "validation_selection": None,
                "reported_checkpoint_policy": "current estimator at each checkpoint",
                "evaluation_policy": "frozen value estimator under prediction policy",
                "prediction_target": "discounted Monte Carlo episode return",
                "success_definition": None,
                "progress_metric": None,
                "progress_goal": None,
            }
        )
    return metadata
