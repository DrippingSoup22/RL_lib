"""CLI configuration for policy-gradient experiments."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch

from experiments.common import (
    RECORDING_CHECKPOINTS,
    evaluation_checkpoints,
    resolve_seed_values,
)
from experiments.policy_gradient.environments import (
    KNOWN_ENVIRONMENTS,
    experiment_defaults,
    make_environment,
    recording_frame_stride,
    success_definition,
)

ALGORITHMS = ("reinforce", "reinforce_with_baseline", "a2c", "a3c", "ppo")
ALGORITHM_LABELS = {
    "reinforce": "REINFORCE",
    "reinforce_with_baseline": "REINFORCE + baseline",
    "a2c": "A2C",
    "a3c": "A3C",
    "ppo": "PPO",
}
PRESETS = ("quick", "tuning", "standard")
DEFAULT_HIDDEN_SIZES = (64, 64)
DEFAULT_A3C_WORKERS = min(4, max(1, (os.cpu_count() or 2) - 1))


@dataclass(frozen=True)
class ExperimentConfig:
    environment: str
    preset: str
    training_episodes: int
    evaluation_episodes: int
    seeds: int
    seed_values: tuple[int, ...]
    algorithm: str
    actor_learning_rate: float
    actor_minimum_learning_rate: float
    critic_learning_rate: float
    critic_minimum_learning_rate: float
    optimizer: str
    weight_decay: float
    max_gradient_norm: float | None
    warmup_episodes: int
    warmup_start_factor: float
    discount: float
    rollout_steps: int
    a3c_workers: int
    entropy_coefficient: float
    ppo_batch_episodes: int
    ppo_update_epochs: int
    ppo_minibatch_size: int
    ppo_clip_ratio: float
    gae_lambda: float
    hidden_sizes: tuple[int, ...]
    continuous_std: str
    initial_std: float
    observation_normalization: str
    reward_scale: float
    evaluation_policy: str
    diagnostics: bool
    recording_mode: str
    recording_checkpoints: tuple[int, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one policy-gradient algorithm on a Gymnasium environment."
    )
    experiment = parser.add_argument_group("experiment")
    experiment.add_argument(
        "--environment", "--env", "-e", default=KNOWN_ENVIRONMENTS[0]
    )
    experiment.add_argument("--preset", "-p", choices=PRESETS, default="standard")
    experiment.add_argument("--training-episodes", "--train", type=int)
    experiment.add_argument("--evaluation-episodes", "--eval", type=int)
    experiment.add_argument("--seeds", "-n", type=int)
    experiment.add_argument(
        "--seed-base",
        type=int,
        help="first paired seed (default: 0 for quick/tuning; randomized for standard)",
    )
    experiment.add_argument(
        "--algorithm",
        choices=ALGORITHMS,
        required=True,
        help="single algorithm evaluated by this run",
    )

    optimization = parser.add_argument_group("optimization")
    optimization.add_argument(
        "--actor-learning-rate",
        "--actor-lr",
        "--lr",
        type=float,
        default=0.003,
    )
    optimization.add_argument(
        "--actor-minimum-learning-rate",
        "--actor-lr-min",
        "--lr-min",
        type=float,
        help="minimum actor cosine learning rate (default: 1%% of --lr)",
    )
    optimization.add_argument(
        "--critic-learning-rate",
        "--critic-lr",
        type=float,
        default=0.01,
    )
    optimization.add_argument(
        "--critic-minimum-learning-rate",
        "--critic-lr-min",
        type=float,
        help="minimum critic cosine learning rate (default: 1%% of --critic-lr)",
    )
    optimization.add_argument(
        "--optimizer",
        "--opt",
        "-o",
        choices=("sgd", "adam", "adamw"),
        default="adamw",
    )
    optimization.add_argument(
        "--weight-decay",
        "--wd",
        type=float,
        default=0.0001,
        help="weight decay applied to actor and critic (default: 0.0001)",
    )
    optimization.add_argument(
        "--max-gradient-norm",
        "--max-grad-norm",
        type=float,
        help="clip actor and critic gradient norms independently; omitted disables",
    )
    optimization.add_argument(
        "--warmup-episodes",
        "--warmup",
        type=int,
        help="linear warmup episodes (default: 10%% of training; 0 disables)",
    )
    optimization.add_argument(
        "--warmup-start-factor",
        "--warmup-start",
        type=float,
        default=0.1,
        help="initial fraction of each peak learning rate (default: 0.1)",
    )
    optimization.add_argument("--discount", "--gamma", type=float, default=0.99)

    actor_critic = parser.add_argument_group("actor-critic")
    actor_critic.add_argument(
        "--rollout-steps",
        "--rollout",
        "--a2c-rollout-steps",
        dest="rollout_steps",
        type=int,
        default=5,
        help="maximum transitions per A2C/A3C update (default: 5)",
    )
    actor_critic.add_argument(
        "--workers",
        "--a3c-workers",
        type=int,
        default=DEFAULT_A3C_WORKERS,
        help=f"A3C worker processes (default on this machine: {DEFAULT_A3C_WORKERS})",
    )
    actor_critic.add_argument(
        "--entropy-coefficient",
        "--entropy",
        type=float,
        default=0.0,
        help="A2C/A3C/PPO entropy bonus coefficient (default: 0)",
    )

    ppo = parser.add_argument_group("PPO")
    ppo.add_argument(
        "--ppo-batch-episodes",
        "--batch-episodes",
        type=int,
        default=4,
        help="complete episodes collected before each PPO update (default: 4)",
    )
    ppo.add_argument(
        "--ppo-update-epochs",
        "--update-epochs",
        type=int,
        default=4,
        help="shuffled passes over each frozen PPO batch (default: 4)",
    )
    ppo.add_argument(
        "--ppo-minibatch-size",
        "--minibatch-size",
        type=int,
        default=64,
        help="transitions per PPO optimizer step (default: 64)",
    )
    ppo.add_argument(
        "--ppo-clip-ratio",
        "--clip-ratio",
        type=float,
        default=0.2,
        help="PPO probability-ratio clipping radius (default: 0.2)",
    )
    ppo.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE bias-variance parameter (default: 0.95)",
    )

    model = parser.add_argument_group("model")
    model.add_argument(
        "--hidden-sizes",
        "--hidden",
        type=int,
        nargs="+",
        default=DEFAULT_HIDDEN_SIZES,
        metavar="N",
    )
    model.add_argument(
        "--continuous-std",
        choices=("state-dependent", "global"),
        default="state-dependent",
        help="continuous policy standard-deviation parameterization",
    )
    model.add_argument(
        "--initial-std",
        type=float,
        default=1.0,
        help="initial continuous Gaussian standard deviation (default: 1)",
    )

    preprocessing = parser.add_argument_group("preprocessing")
    preprocessing.add_argument(
        "--observation-normalization",
        choices=("none", "bounds", "running"),
        default="none",
    )
    preprocessing.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="positive scale applied only to rewards used for learning",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--recordings",
        "--record",
        "-r",
        choices=("none", "final", "checkpoints"),
    )
    output.add_argument(
        "--diagnostics",
        "--diag",
        action="store_true",
        help="write compact per-training-episode diagnostics",
    )
    output.add_argument(
        "--evaluation-policy",
        choices=("stochastic", "deterministic"),
        default="stochastic",
        help="action selection used by frozen evaluation and recordings",
    )
    return parser


def parse_config(argv: Sequence[str] | None = None) -> ExperimentConfig:
    parser = build_parser()
    args = parser.parse_args(argv)
    defaults = experiment_defaults(args.environment, args.preset)
    training_episodes = (
        defaults["training_episodes"]
        if args.training_episodes is None
        else args.training_episodes
    )
    evaluation_episodes = (
        defaults["evaluation_episodes"]
        if args.evaluation_episodes is None
        else args.evaluation_episodes
    )
    seeds = defaults["seeds"] if args.seeds is None else args.seeds
    if min(training_episodes, evaluation_episodes, seeds) < 1:
        parser.error(
            "training episodes, evaluation episodes, and seeds must be positive"
        )
    try:
        seed_values = resolve_seed_values(seeds, args.preset, args.seed_base)
    except ValueError as error:
        parser.error(str(error))

    actor_learning_rate = args.actor_learning_rate
    critic_learning_rate = (
        actor_learning_rate
        if args.critic_learning_rate is None
        else args.critic_learning_rate
    )
    if not math.isfinite(actor_learning_rate) or actor_learning_rate <= 0:
        parser.error("actor learning rate must be finite and positive")
    if not math.isfinite(critic_learning_rate) or critic_learning_rate <= 0:
        parser.error("critic learning rate must be finite and positive")
    actor_minimum_learning_rate = (
        actor_learning_rate * 0.01
        if args.actor_minimum_learning_rate is None
        else args.actor_minimum_learning_rate
    )
    critic_minimum_learning_rate = (
        critic_learning_rate * 0.01
        if args.critic_minimum_learning_rate is None
        else args.critic_minimum_learning_rate
    )
    if (
        not math.isfinite(actor_minimum_learning_rate)
        or actor_minimum_learning_rate < 0
        or actor_minimum_learning_rate > actor_learning_rate
    ):
        parser.error("actor minimum learning rate must be between 0 and --lr")
    if (
        not math.isfinite(critic_minimum_learning_rate)
        or critic_minimum_learning_rate < 0
        or critic_minimum_learning_rate > critic_learning_rate
    ):
        parser.error("critic minimum learning rate must be between 0 and --critic-lr")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("weight decay must be finite and nonnegative")
    if args.max_gradient_norm is not None and (
        not math.isfinite(args.max_gradient_norm) or args.max_gradient_norm <= 0
    ):
        parser.error("maximum gradient norm must be finite and positive")
    warmup_episodes = (
        min(max(1, round(training_episodes * 0.1)), training_episodes - 1)
        if args.warmup_episodes is None and training_episodes > 1
        else (0 if args.warmup_episodes is None else args.warmup_episodes)
    )
    if not 0 <= warmup_episodes < training_episodes:
        parser.error("warmup episodes must be between 0 and training episodes - 1")
    if (
        not math.isfinite(args.warmup_start_factor)
        or not 0 < args.warmup_start_factor <= 1
    ):
        parser.error("warmup start factor must be finite and in (0, 1]")
    if not math.isfinite(args.discount) or not 0 <= args.discount <= 1:
        parser.error("discount must be finite and in [0, 1]")
    if args.rollout_steps < 1:
        parser.error("actor-critic rollout steps must be positive")
    if args.workers < 1:
        parser.error("A3C workers must be positive")
    if not math.isfinite(args.entropy_coefficient) or args.entropy_coefficient < 0:
        parser.error("entropy coefficient must be finite and nonnegative")
    if args.ppo_batch_episodes < 1:
        parser.error("PPO batch episodes must be positive")
    if args.ppo_update_epochs < 1:
        parser.error("PPO update epochs must be positive")
    if args.ppo_minibatch_size < 1:
        parser.error("PPO minibatch size must be positive")
    if not math.isfinite(args.ppo_clip_ratio) or not 0 < args.ppo_clip_ratio < 1:
        parser.error("PPO clip ratio must be finite and in (0, 1)")
    if not math.isfinite(args.gae_lambda) or not 0 <= args.gae_lambda <= 1:
        parser.error("GAE lambda must be finite and in [0, 1]")
    if any(hidden_size <= 0 for hidden_size in args.hidden_sizes):
        parser.error("hidden sizes must be positive")
    if not math.isfinite(args.initial_std) or args.initial_std <= 0:
        parser.error("initial standard deviation must be finite and positive")
    if not math.isfinite(args.reward_scale) or args.reward_scale <= 0:
        parser.error("reward scale must be finite and positive")
    if args.algorithm == "a3c" and args.observation_normalization == "running":
        parser.error("A3C supports 'none' or 'bounds' observation normalization")

    recording_mode = {
        "quick": "none",
        "tuning": "none",
        "standard": "checkpoints",
    }[args.preset]
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
    if args.observation_normalization == "bounds":
        assert isinstance(inspection_env.observation_space, gym.spaces.Box)
        if not np.all(np.isfinite(inspection_env.observation_space.low)) or not np.all(
            np.isfinite(inspection_env.observation_space.high)
        ):
            inspection_env.close()
            parser.error("bounds normalization requires finite observation bounds")
    if args.preset == "standard" and "rgb_array" not in inspection_env.metadata.get(
        "render_modes", []
    ):
        inspection_env.close()
        parser.error("standard mode requires an environment with rgb_array rendering")
    inspection_env.close()

    return ExperimentConfig(
        environment=args.environment,
        preset=args.preset,
        training_episodes=training_episodes,
        evaluation_episodes=evaluation_episodes,
        seeds=seeds,
        seed_values=seed_values,
        algorithm=args.algorithm,
        actor_learning_rate=actor_learning_rate,
        actor_minimum_learning_rate=actor_minimum_learning_rate,
        critic_learning_rate=critic_learning_rate,
        critic_minimum_learning_rate=critic_minimum_learning_rate,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        max_gradient_norm=args.max_gradient_norm,
        warmup_episodes=warmup_episodes,
        warmup_start_factor=args.warmup_start_factor,
        discount=args.discount,
        rollout_steps=args.rollout_steps,
        a3c_workers=args.workers,
        entropy_coefficient=args.entropy_coefficient,
        ppo_batch_episodes=args.ppo_batch_episodes,
        ppo_update_epochs=args.ppo_update_epochs,
        ppo_minibatch_size=args.ppo_minibatch_size,
        ppo_clip_ratio=args.ppo_clip_ratio,
        gae_lambda=args.gae_lambda,
        hidden_sizes=tuple(args.hidden_sizes),
        continuous_std=args.continuous_std,
        initial_std=args.initial_std,
        observation_normalization=args.observation_normalization,
        reward_scale=args.reward_scale,
        evaluation_policy=args.evaluation_policy,
        diagnostics=args.diagnostics,
        recording_mode=recording_mode,
        recording_checkpoints=recording_checkpoints,
    )


def _serialized_bounds(bounds: np.ndarray) -> list[float | str]:
    result: list[float | str] = []
    for value in bounds.reshape(-1):
        if np.isneginf(value):
            result.append("-Infinity")
        elif np.isposinf(value):
            result.append("Infinity")
        else:
            result.append(float(value))
    return result


def initial_metadata(config: ExperimentConfig) -> dict[str, object]:
    """Build complete reproducibility metadata before training begins."""
    inspection_env = make_environment(config.environment)
    assert isinstance(inspection_env.observation_space, gym.spaces.Box)
    assert isinstance(
        inspection_env.action_space,
        (gym.spaces.Discrete, gym.spaces.Box),
    )
    max_episode_steps = (
        inspection_env.spec.max_episode_steps
        if inspection_env.spec is not None
        else None
    )
    observation_shape = list(inspection_env.observation_space.shape)
    observation_low = _serialized_bounds(inspection_env.observation_space.low)
    observation_high = _serialized_bounds(inspection_env.observation_space.high)
    if isinstance(inspection_env.action_space, gym.spaces.Discrete):
        action_space_type = "categorical"
        number_of_actions: int | None = int(inspection_env.action_space.n)
        action_shape: list[int] | None = None
        action_low: list[float] | None = None
        action_high: list[float] | None = None
        evaluation_policy_description = (
            f"frozen {config.evaluation_policy} categorical actor"
        )
    else:
        action_space_type = "bounded continuous"
        number_of_actions = None
        action_shape = list(inspection_env.action_space.shape)
        action_low = [float(value) for value in inspection_env.action_space.low]
        action_high = [float(value) for value in inspection_env.action_space.high]
        evaluation_policy_description = (
            f"frozen {config.evaluation_policy} tanh-squashed Gaussian actor"
        )
    inspection_env.close()

    metadata = asdict(config)
    metadata.update(
        {
            "status": "running",
            "evaluation_checkpoints": list(evaluation_checkpoints(config.preset)),
            "recording_checkpoints": list(config.recording_checkpoints),
            "observation_shape": observation_shape,
            "observation_low": observation_low,
            "observation_high": observation_high,
            "observation_processing": {
                "none": "convert to float32 and flatten; no normalization",
                "bounds": "flatten and map finite observation bounds to [-1, 1]",
                "running": (
                    "flatten, normalize with running training mean/variance, and "
                    "clip to [-10, 10]; freeze statistics during evaluation"
                ),
            }[config.observation_normalization],
            "action_space_type": action_space_type,
            "number_of_actions": number_of_actions,
            "action_shape": action_shape,
            "action_low": action_low,
            "action_high": action_high,
            "max_episode_steps": max_episode_steps,
            "training_environment_seed": "seed * training_episodes + episode",
            "training_action_seed": "10000000 + seed * training_episodes + episode",
            "evaluation_environment_seed": (
                "1000000 + seed * 100000 + checkpoint * 1000 + evaluation_episode"
            ),
            "evaluation_action_seed": (
                "11000000 + seed * 100000 + checkpoint * 1000 + evaluation_episode"
            ),
            "recording_environment_seed": (
                "2000000 + checkpoint" if config.recording_checkpoints else None
            ),
            "recording_action_seed": (
                "12000000 + checkpoint" if config.recording_checkpoints else None
            ),
            "recording_frame_stride": (
                recording_frame_stride(config.environment)
                if config.recording_checkpoints
                else None
            ),
            "evaluation_policy_description": evaluation_policy_description,
            "training_reward_processing": (
                f"multiply rewards used for learning by {config.reward_scale:g}; "
                "report unscaled environment returns"
            ),
            "success_definition": success_definition(config.environment),
            "episode_stopping": "termination or truncation",
            "truncation_target": (
                "Monte Carlo trajectories end without bootstrap; A2C, A3C, and "
                "PPO bootstrap the final observation after truncation"
            ),
            "initialization": "fresh actor and optional critic for every seed trial",
            "paired_environment_seeds": True,
            "algorithm_update": {
                "reinforce": "one Monte Carlo policy update after each episode",
                "reinforce_with_baseline": (
                    "one Monte Carlo actor and critic update after each episode"
                ),
                "a2c": "actor and critic updates after each bounded n-step rollout",
                "a3c": (
                    "asynchronous worker actor and critic updates after each bounded "
                    "n-step rollout; a short lock protects each shared optimizer step"
                ),
                "ppo": (
                    "collect complete episodes without updating, calculate GAE per "
                    "episode, then reuse the frozen transition batch over shuffled "
                    "clipped-objective minibatch epochs"
                ),
            }[config.algorithm],
            "a3c_worker_lifecycle": (
                "spawn one worker pool per nonempty checkpoint segment and join it "
                "before frozen evaluation"
                if config.algorithm == "a3c"
                else None
            ),
            "learning_rate_schedule": (
                "independent linear warmup then cosine annealing for actor and critic; "
                "scheduled by completed training episode and fixed during each PPO "
                "batch"
            ),
            "training_diagnostics_file": (
                "training_diagnostics.csv" if config.diagnostics else None
            ),
            "training_diagnostics_definition": (
                "one row per completed training episode; A2C and A3C losses are "
                "transition-weighted means across their rollout updates; PPO repeats "
                "the batch mean minibatch losses for its collected episodes and "
                "evenly attributes minibatch optimizer steps across them"
                if config.diagnostics
                else None
            ),
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
    )
    return metadata
