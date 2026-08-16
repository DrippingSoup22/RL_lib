"""Compare Monte Carlo and actor-critic policy gradients on Gymnasium."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image

from experiments.common import (
    EVALUATION_CHECKPOINTS,
    RECORDING_CHECKPOINTS,
    annotated_frame,
    checkpoint_episode_target,
    create_run_directory,
    recording_title,
    write_csv,
    write_metadata,
)
from experiments.summary import SummaryMedia, SummaryTable, write_summary
from rl_lib.algorithms.policy_gradient import A2C, Reinforce, ReinforceWithBaseline
from rl_lib.data import Episode, EpisodeStep
from rl_lib.models import DiscretePolicyNetwork, StateValueNetwork

ENVIRONMENTS = ("CartPole-v1", "Acrobot-v1")
ALGORITHMS = ("reinforce", "reinforce_with_baseline", "a2c")
ALGORITHM_LABELS = {
    "reinforce": "REINFORCE",
    "reinforce_with_baseline": "REINFORCE + baseline",
    "a2c": "A2C",
}
PRESETS = ("quick", "tuning", "standard")
DEFAULT_HIDDEN_SIZES = (64, 64)
CSV_FIELDS = (
    "algorithm",
    "seed",
    "checkpoint",
    "evaluation_episode",
    "episode_return",
    "episode_length",
    "success",
    "terminated",
    "truncated",
)
DIAGNOSTIC_FIELDS = (
    "algorithm",
    "seed",
    "training_episode",
    "actor_learning_rate",
    "critic_learning_rate",
    "episode_return",
    "episode_length",
    "terminated",
    "truncated",
    "updates",
    "actor_loss",
    "critic_loss",
    "policy_entropy",
)
Agent = Reinforce | ReinforceWithBaseline | A2C
Observation = NDArray[np.float32]
ModelState = dict[str, torch.Tensor]


@dataclass(frozen=True)
class ExperimentConfig:
    environment: str
    preset: str
    training_episodes: int
    evaluation_episodes: int
    seeds: int
    algorithms: tuple[str, ...]
    actor_learning_rate: float
    actor_minimum_learning_rate: float
    critic_learning_rate: float
    critic_minimum_learning_rate: float
    optimizer: str
    weight_decay: float
    warmup_episodes: int
    warmup_start_factor: float
    discount: float
    a2c_rollout_steps: int
    entropy_coefficient: float
    hidden_sizes: tuple[int, ...]
    diagnostics: bool
    recording_mode: str
    recording_checkpoints: tuple[int, ...]


@dataclass(frozen=True)
class EvaluationResult:
    episode_return: float
    episode_length: int
    success: bool
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class TrainingEpisodeResult:
    episode_return: float
    episode_length: int
    terminated: bool
    truncated: bool
    updates: int
    actor_loss: float
    critic_loss: float | None
    policy_entropy: float | None


def make_environment(
    environment: str,
    *,
    render_mode: str | None = None,
    max_episode_steps: int | None = None,
) -> gym.Env:
    """Create an environment compatible with the discrete policy network."""
    if environment not in ENVIRONMENTS:
        raise ValueError(f"environment must be one of {ENVIRONMENTS}")

    env = gym.make(
        environment,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
    )
    if not isinstance(env.observation_space, gym.spaces.Box):
        env.close()
        raise ValueError("environment must have a continuous Box observation space")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        env.close()
        raise ValueError("environment must have a discrete action space")
    if env.action_space.start != 0:
        env.close()
        raise ValueError("environment actions must start at zero")
    return env


def make_optimizer(
    name: str,
    parameters: Iterable[torch.Tensor],
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Construct one of the optimizers exposed by the runner."""
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    raise ValueError("optimizer must be 'sgd', 'adam', or 'adamw'")


def make_agent(
    algorithm: str,
    env: gym.Env,
    *,
    actor_learning_rate: float,
    critic_learning_rate: float,
    optimizer_name: str,
    weight_decay: float,
    discount: float,
    entropy_coefficient: float,
    hidden_sizes: tuple[int, ...],
    seed: int,
) -> Agent:
    """Build one policy-gradient agent with reproducible initialization."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"algorithm must be one of {ALGORITHMS}")
    if not isinstance(env.observation_space, gym.spaces.Box):
        raise ValueError("environment must have a Box observation space")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("environment must have a Discrete action space")

    observation_size = int(np.prod(env.observation_space.shape))
    number_of_actions = int(env.action_space.n)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        actor_model = DiscretePolicyNetwork(
            observation_size,
            number_of_actions,
            hidden_sizes,
        )
        actor_optimizer = make_optimizer(
            optimizer_name,
            actor_model.parameters(),
            actor_learning_rate,
            weight_decay,
        )
        if algorithm == "reinforce":
            return Reinforce(actor_model, actor_optimizer, discount)

        critic_model = StateValueNetwork(observation_size, hidden_sizes)
        critic_optimizer = make_optimizer(
            optimizer_name,
            critic_model.parameters(),
            critic_learning_rate,
            weight_decay,
        )
        if algorithm == "reinforce_with_baseline":
            return ReinforceWithBaseline(
                actor_model,
                actor_optimizer,
                critic_model,
                critic_optimizer,
                discount,
            )
        return A2C(
            actor_model,
            actor_optimizer,
            critic_model,
            critic_optimizer,
            discount,
            entropy_coefficient,
        )


def make_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    training_episodes: int,
    minimum_learning_rate: float,
    warmup_episodes: int,
    warmup_start_factor: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Warm up linearly, then anneal to the minimum with one cosine cycle."""
    if len(optimizer.param_groups) != 1:
        raise ValueError("policy-gradient optimizers need one parameter group")
    peak_learning_rate = float(optimizer.param_groups[0]["lr"])
    minimum_factor = minimum_learning_rate / peak_learning_rate
    cosine_episodes = max(1, training_episodes - warmup_episodes)

    def learning_rate_factor(episode: int) -> float:
        if warmup_episodes and episode <= warmup_episodes:
            progress = episode / warmup_episodes
            return warmup_start_factor + progress * (1.0 - warmup_start_factor)

        progress = (episode - warmup_episodes) / cosine_episodes
        return minimum_factor + 0.5 * (1.0 - minimum_factor) * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)


def observation_array(observation: object, observation_size: int) -> Observation:
    """Convert a Gymnasium observation to one owned, flat float32 array."""
    result = np.asarray(observation, dtype=np.float32).reshape(-1)
    if result.shape != (observation_size,):
        raise ValueError("environment observation does not match the model input size")
    if not np.all(np.isfinite(result)):
        raise ValueError("environment observation must be finite")
    return result.copy()


def generate_episode(
    env: gym.Env,
    agent: Agent,
    *,
    environment_seed: int,
    action_seed: int,
) -> Episode[Observation]:
    """Sample one complete episode without changing the agent."""
    observation, _ = env.reset(seed=environment_seed)
    state = observation_array(observation, agent.actor_model.observation_size)
    steps: list[EpisodeStep[Observation]] = []
    terminated = truncated = False

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(action_seed)
        while not (terminated or truncated):
            action = agent.select_action(state)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            steps.append(EpisodeStep(state, action, float(reward)))
            state = observation_array(
                next_observation,
                agent.actor_model.observation_size,
            )

    return Episode(
        steps=tuple(steps),
        final_state=state,
        terminated=terminated,
        truncated=truncated,
    )


def mean_policy_entropy(
    agent: Agent,
    steps: Sequence[EpisodeStep[Observation]],
) -> float:
    """Measure categorical-policy entropy on rollout observations."""
    observations = torch.as_tensor(
        np.asarray([step.state for step in steps], dtype=np.float32)
    )
    with torch.no_grad():
        logits = agent.actor_model(observations)
        entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
    return float(entropy.item())


def train_a2c_episode(
    env: gym.Env,
    agent: A2C,
    *,
    rollout_steps: int,
    environment_seed: int,
    action_seed: int,
    collect_diagnostics: bool,
) -> TrainingEpisodeResult:
    """Train A2C during one episode using bounded n-step rollouts."""
    observation, _ = env.reset(seed=environment_seed)
    state = observation_array(observation, agent.actor_model.observation_size)
    rollout: list[EpisodeStep[Observation]] = []
    terminated = truncated = False
    episode_return = 0.0
    episode_length = 0
    updates = 0
    weighted_actor_loss = 0.0
    weighted_critic_loss = 0.0
    weighted_entropy = 0.0

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(action_seed)
        while not (terminated or truncated):
            action = agent.select_action(state)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = observation_array(
                next_observation,
                agent.actor_model.observation_size,
            )
            rollout.append(EpisodeStep(state, action, float(reward)))
            episode_return += float(reward)
            episode_length += 1
            state = next_state

            if len(rollout) < rollout_steps and not (terminated or truncated):
                continue

            rollout_size = len(rollout)
            if collect_diagnostics:
                weighted_entropy += mean_policy_entropy(agent, rollout) * rollout_size
            actor_loss, critic_loss = agent.update(
                tuple(rollout),
                state,
                terminated=terminated,
            )
            if not (math.isfinite(actor_loss) and math.isfinite(critic_loss)):
                raise RuntimeError("A2C produced a non-finite loss")
            weighted_actor_loss += actor_loss * rollout_size
            weighted_critic_loss += critic_loss * rollout_size
            updates += 1
            rollout.clear()

    return TrainingEpisodeResult(
        episode_return=episode_return,
        episode_length=episode_length,
        terminated=terminated,
        truncated=truncated,
        updates=updates,
        actor_loss=weighted_actor_loss / episode_length,
        critic_loss=weighted_critic_loss / episode_length,
        policy_entropy=(
            weighted_entropy / episode_length if collect_diagnostics else None
        ),
    )


def train_episode(
    env: gym.Env,
    agent: Agent,
    *,
    rollout_steps: int,
    environment_seed: int,
    action_seed: int,
    collect_diagnostics: bool,
) -> TrainingEpisodeResult:
    """Train one episode using the update cadence owned by the algorithm."""
    if isinstance(agent, A2C):
        return train_a2c_episode(
            env,
            agent,
            rollout_steps=rollout_steps,
            environment_seed=environment_seed,
            action_seed=action_seed,
            collect_diagnostics=collect_diagnostics,
        )

    episode = generate_episode(
        env,
        agent,
        environment_seed=environment_seed,
        action_seed=action_seed,
    )
    entropy = mean_policy_entropy(agent, episode.steps) if collect_diagnostics else None
    actor_loss = agent.update(episode)
    if not math.isfinite(actor_loss):
        raise RuntimeError("REINFORCE produced a non-finite loss")
    return TrainingEpisodeResult(
        episode_return=sum(step.reward for step in episode.steps),
        episode_length=len(episode.steps),
        terminated=episode.terminated,
        truncated=episode.truncated,
        updates=1,
        actor_loss=actor_loss,
        critic_loss=None,
        policy_entropy=entropy,
    )


def episode_succeeded(
    environment: str,
    *,
    terminated: bool,
    truncated: bool,
) -> bool:
    """Interpret Gymnasium stopping conditions using environment semantics."""
    if environment == "CartPole-v1":
        return truncated and not terminated
    if environment == "Acrobot-v1":
        return terminated
    raise ValueError(f"environment must be one of {ENVIRONMENTS}")


def evaluate_episode(
    env: gym.Env,
    environment: str,
    agent: Agent,
    *,
    environment_seed: int,
    action_seed: int,
) -> EvaluationResult:
    """Evaluate the frozen stochastic actor for one episode."""
    was_training = agent.actor_model.training
    agent.actor_model.eval()
    try:
        episode = generate_episode(
            env,
            agent,
            environment_seed=environment_seed,
            action_seed=action_seed,
        )
    finally:
        agent.actor_model.train(was_training)

    return EvaluationResult(
        episode_return=sum(step.reward for step in episode.steps),
        episode_length=len(episode.steps),
        success=episode_succeeded(
            environment,
            terminated=episode.terminated,
            truncated=episode.truncated,
        ),
        terminated=episode.terminated,
        truncated=episode.truncated,
    )


def rendered_frame(env: gym.Env, label: str) -> Image.Image:
    frame = env.render()
    if not isinstance(frame, np.ndarray):
        raise RuntimeError("environment did not return an RGB frame")
    return annotated_frame(frame, label)


def action_name(environment: str, action: int) -> str:
    if environment == "CartPole-v1":
        return ("push left", "push right")[action]
    return ("negative torque", "no torque", "positive torque")[action]


def record_evaluation(
    path: Path,
    environment: str,
    agent: Agent,
    *,
    environment_seed: int,
    action_seed: int,
    frame_stride: int,
) -> None:
    """Record one frozen stochastic-policy episode as an annotated GIF."""
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    env = make_environment(environment, render_mode="rgb_array")
    was_training = agent.actor_model.training
    agent.actor_model.eval()
    frames: list[Image.Image] = []
    durations: list[int] = []
    try:
        observation, _ = env.reset(seed=environment_seed)
        state = observation_array(observation, agent.actor_model.observation_size)
        frames.append(rendered_frame(env, "Frozen stochastic evaluation: start"))
        durations.append(100)
        terminated = truncated = False
        step = 0
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(action_seed)
            while not (terminated or truncated):
                action = agent.select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action)
                state = observation_array(
                    observation,
                    agent.actor_model.observation_size,
                )
                step += 1
                if step % frame_stride == 0 or terminated or truncated:
                    frames.append(
                        rendered_frame(
                            env,
                            f"Step {step}: {action_name(environment, action)}, "
                            f"reward {float(reward):g}",
                        )
                    )
                    durations.append(1_000 if terminated or truncated else 100)
    finally:
        agent.actor_model.train(was_training)
        env.close()

    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )


def experiment_defaults(environment: str, preset: str) -> dict[str, int]:
    """Return experiment budgets; explicit CLI values can override each one."""
    if preset == "quick":
        return {
            "training_episodes": 5,
            "evaluation_episodes": 2,
            "seeds": 1,
        }
    if preset == "tuning":
        return {
            "training_episodes": 1_000 if environment == "Acrobot-v1" else 500,
            "evaluation_episodes": 10,
            "seeds": 1,
        }
    return {
        "training_episodes": 2_000 if environment == "Acrobot-v1" else 1_000,
        "evaluation_episodes": 50,
        "seeds": 3,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    experiment = parser.add_argument_group("experiment")
    experiment.add_argument(
        "--environment",
        "--env",
        "-e",
        choices=ENVIRONMENTS,
        default=ENVIRONMENTS[0],
    )
    experiment.add_argument(
        "--preset",
        "-p",
        choices=PRESETS,
        default="standard",
    )
    experiment.add_argument("--training-episodes", "--train", type=int)
    experiment.add_argument("--evaluation-episodes", "--eval", type=int)
    experiment.add_argument("--seeds", "-n", type=int)
    experiment.add_argument(
        "--algorithms",
        "--algorithm",
        nargs="+",
        choices=ALGORITHMS,
        default=ALGORITHMS,
        help="algorithms to run in the given order (default: all)",
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

    a2c = parser.add_argument_group("A2C")
    a2c.add_argument(
        "--a2c-rollout-steps",
        "--rollout-steps",
        "--rollout",
        type=int,
        default=5,
        help="maximum transitions per A2C update (default: 5)",
    )
    a2c.add_argument(
        "--entropy-coefficient",
        "--entropy",
        type=float,
        default=0.0,
        help="A2C entropy bonus coefficient (default: 0)",
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
    algorithms = tuple(args.algorithms)
    if len(set(algorithms)) != len(algorithms):
        parser.error("algorithms must not contain duplicates")

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
    if args.a2c_rollout_steps < 1:
        parser.error("A2C rollout steps must be positive")
    if not math.isfinite(args.entropy_coefficient) or args.entropy_coefficient < 0:
        parser.error("entropy coefficient must be finite and nonnegative")
    if any(hidden_size <= 0 for hidden_size in args.hidden_sizes):
        parser.error("hidden sizes must be positive")

    recording_mode = args.recordings
    if recording_mode is None:
        recording_mode = {
            "quick": "final",
            "tuning": "none",
            "standard": "checkpoints",
        }[args.preset]
    recording_checkpoints = {
        "none": (),
        "final": (100,),
        "checkpoints": RECORDING_CHECKPOINTS,
    }[recording_mode]

    return ExperimentConfig(
        environment=args.environment,
        preset=args.preset,
        training_episodes=training_episodes,
        evaluation_episodes=evaluation_episodes,
        seeds=seeds,
        algorithms=algorithms,
        actor_learning_rate=actor_learning_rate,
        actor_minimum_learning_rate=actor_minimum_learning_rate,
        critic_learning_rate=critic_learning_rate,
        critic_minimum_learning_rate=critic_minimum_learning_rate,
        optimizer=args.optimizer,
        weight_decay=args.weight_decay,
        warmup_episodes=warmup_episodes,
        warmup_start_factor=args.warmup_start_factor,
        discount=args.discount,
        a2c_rollout_steps=args.a2c_rollout_steps,
        entropy_coefficient=args.entropy_coefficient,
        hidden_sizes=tuple(args.hidden_sizes),
        diagnostics=args.diagnostics,
        recording_mode=recording_mode,
        recording_checkpoints=recording_checkpoints,
    )


def evaluation_row(
    algorithm: str,
    seed: int,
    checkpoint: int,
    evaluation_episode: int,
    result: EvaluationResult,
) -> dict[str, object]:
    """Convert one frozen-policy evaluation to the raw metrics schema."""
    return {
        "algorithm": algorithm,
        "seed": seed,
        "checkpoint": checkpoint,
        "evaluation_episode": evaluation_episode,
        "episode_return": result.episode_return,
        "episode_length": result.episode_length,
        "success": int(result.success),
        "terminated": int(result.terminated),
        "truncated": int(result.truncated),
    }


def optimizer_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Return the learning rate from an optimizer with one parameter group."""
    if len(optimizer.param_groups) != 1:
        raise ValueError("policy-gradient optimizers need one parameter group")
    return float(optimizer.param_groups[0]["lr"])


def critic_optimizer(agent: Agent) -> torch.optim.Optimizer | None:
    """Return the critic optimizer used by baseline and actor-critic agents."""
    if isinstance(agent, (ReinforceWithBaseline, A2C)):
        return agent.critic_optimizer
    return None


def diagnostic_row(
    algorithm: str,
    seed: int,
    training_episode: int,
    actor_learning_rate: float,
    critic_learning_rate: float | None,
    result: TrainingEpisodeResult,
) -> dict[str, object]:
    """Convert one training episode to the optional diagnostic schema."""
    if result.policy_entropy is None:
        raise ValueError("training result does not contain diagnostics")
    return {
        "algorithm": algorithm,
        "seed": seed,
        "training_episode": training_episode,
        "actor_learning_rate": actor_learning_rate,
        "critic_learning_rate": (
            "" if critic_learning_rate is None else critic_learning_rate
        ),
        "episode_return": result.episode_return,
        "episode_length": result.episode_length,
        "terminated": int(result.terminated),
        "truncated": int(result.truncated),
        "updates": result.updates,
        "actor_loss": result.actor_loss,
        "critic_loss": "" if result.critic_loss is None else result.critic_loss,
        "policy_entropy": result.policy_entropy,
    }


def print_diagnostic_summary(rows: list[dict[str, object]]) -> None:
    """Print compact diagnostic means for one completed checkpoint segment."""
    if not rows:
        return

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in rows]))

    critic_losses = [
        float(row["critic_loss"]) for row in rows if row["critic_loss"] != ""
    ]
    critic_text = (
        f" | critic loss={float(np.mean(critic_losses)):.3g}" if critic_losses else ""
    )
    print(
        "    DIAG  | "
        f"actor loss={mean('actor_loss'):.3g}{critic_text} | "
        f"entropy={mean('policy_entropy'):.3g} | "
        f"updates/episode={mean('updates'):.2f}",
        flush=True,
    )


def print_experiment_header(config: ExperimentConfig) -> None:
    """Print the resolved settings most useful while watching a run."""
    optimizer_name = {
        "adam": "Adam",
        "adamw": "AdamW",
        "sgd": "SGD",
    }[config.optimizer]
    warmup = (
        f"{config.warmup_episodes} episodes from {config.warmup_start_factor:g}x peak"
        if config.warmup_episodes
        else "disabled"
    )

    print("", flush=True)
    print("POLICY-GRADIENT EXPERIMENT", flush=True)
    print(f"  Environment : {config.environment}", flush=True)
    print(f"  Preset      : {config.preset}", flush=True)
    print(
        f"  Training    : {config.training_episodes} episodes per seed",
        flush=True,
    )
    print(
        f"  Evaluation  : {config.evaluation_episodes} episodes per checkpoint",
        flush=True,
    )
    print(
        f"  Seeds       : {config.seeds} independent "
        f"{'run' if config.seeds == 1 else 'runs'}; fresh network each run",
        flush=True,
    )
    print(
        "  Algorithms  : "
        + ", ".join(ALGORITHM_LABELS[algorithm] for algorithm in config.algorithms),
        flush=True,
    )
    print(
        f"  Optimizer   : {optimizer_name}; weight decay={config.weight_decay:g}",
        flush=True,
    )
    print(
        f"  Actor LR    : {config.actor_learning_rate:g} -> "
        f"{config.actor_minimum_learning_rate:g}",
        flush=True,
    )
    print(
        f"  Critic LR   : {config.critic_learning_rate:g} -> "
        f"{config.critic_minimum_learning_rate:g}",
        flush=True,
    )
    print(f"  Warmup      : {warmup}", flush=True)
    print(
        f"  A2C         : rollout={config.a2c_rollout_steps} steps; "
        f"entropy coefficient={config.entropy_coefficient:g}",
        flush=True,
    )
    print(
        "  Diagnostics : "
        + ("training_diagnostics.csv" if config.diagnostics else "off"),
        flush=True,
    )


def print_final_evaluation(
    rows: list[dict[str, object]],
    algorithms: Sequence[str] = ALGORITHMS,
) -> None:
    """Print final per-seed measurements and their across-seed aggregate."""
    final_rows = [row for row in rows if int(row["checkpoint"]) == 100]
    summaries: list[tuple[str, float, float, float, float]] = []
    print("", flush=True)
    print("FINAL FROZEN-POLICY EVALUATION", flush=True)
    print(
        f"  {'Algorithm':<23} {'Seed':>4} {'Mean return':>12} {'Success':>9}",
        flush=True,
    )
    print(f"  {'-' * 52}", flush=True)

    for algorithm in algorithms:
        algorithm_rows = [row for row in final_rows if row["algorithm"] == algorithm]
        seed_returns: list[float] = []
        seed_successes: list[float] = []
        for seed in sorted({int(row["seed"]) for row in algorithm_rows}):
            seed_rows = [row for row in algorithm_rows if int(row["seed"]) == seed]
            mean_return = float(
                np.mean([float(row["episode_return"]) for row in seed_rows])
            )
            success_rate = float(np.mean([float(row["success"]) for row in seed_rows]))
            seed_returns.append(mean_return)
            seed_successes.append(success_rate)
            print(
                f"  {ALGORITHM_LABELS[algorithm]:<23} {seed:>4} "
                f"{mean_return:>12.1f} {success_rate:>8.0%}",
                flush=True,
            )

        if not seed_returns:
            continue
        return_std = (
            float(np.std(seed_returns, ddof=1)) if len(seed_returns) > 1 else 0.0
        )
        success_std = (
            float(np.std(seed_successes, ddof=1)) if len(seed_successes) > 1 else 0.0
        )
        summaries.append(
            (
                ALGORITHM_LABELS[algorithm],
                float(np.mean(seed_returns)),
                return_std,
                float(np.mean(seed_successes)),
                success_std,
            )
        )

    print("", flush=True)
    print("ACROSS-SEED SUMMARY", flush=True)
    print(
        f"  {'Algorithm':<23} {'Return mean +/- SD':>20} {'Success mean +/- SD':>21}",
        flush=True,
    )
    print(f"  {'-' * 66}", flush=True)
    for label, return_mean, return_std, success_mean, success_std in summaries:
        print(
            f"  {label:<23} {return_mean:>8.1f} +/- {return_std:<7.1f} "
            f"{success_mean:>8.0%} +/- {success_std:.0%}",
            flush=True,
        )
    print("", flush=True)


def aggregate_rows(
    rows: list[dict[str, object]],
    algorithms: Sequence[str] = ALGORITHMS,
) -> list[dict[str, object]]:
    """Aggregate episode measurements after first averaging within each seed."""
    aggregate: list[dict[str, object]] = []
    metrics = ("episode_return", "episode_length", "success", "truncated")
    for algorithm in algorithms:
        for checkpoint in EVALUATION_CHECKPOINTS:
            selected = [
                row
                for row in rows
                if row["algorithm"] == algorithm and row["checkpoint"] == checkpoint
            ]
            if not selected:
                continue

            seed_means: dict[str, list[float]] = {metric: [] for metric in metrics}
            for seed in sorted({int(row["seed"]) for row in selected}):
                seed_rows = [row for row in selected if int(row["seed"]) == seed]
                for metric in metrics:
                    seed_means[metric].append(
                        float(np.mean([float(row[metric]) for row in seed_rows]))
                    )

            result: dict[str, object] = {
                "algorithm": algorithm,
                "checkpoint": checkpoint,
                "seeds": len(seed_means["episode_return"]),
            }
            for metric, values_list in seed_means.items():
                values = np.asarray(values_list, dtype=float)
                result[metric] = float(np.mean(values))
                result[f"{metric}_std"] = (
                    float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                )
            aggregate.append(result)
    return aggregate


def write_figure(path: Path, aggregate: list[dict[str, object]]) -> None:
    """Plot frozen evaluation return and success over training checkpoints."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for algorithm in ALGORITHMS:
        selected = [row for row in aggregate if row["algorithm"] == algorithm]
        if not selected:
            continue
        checkpoints = np.asarray([row["checkpoint"] for row in selected], dtype=float)
        for axis, metric in zip(
            axes,
            ("episode_return", "success"),
            strict=True,
        ):
            means = np.asarray([row[metric] for row in selected], dtype=float)
            standard_errors = np.asarray(
                [
                    float(row[f"{metric}_std"]) / np.sqrt(float(row["seeds"]))
                    for row in selected
                ]
            )
            (line,) = axis.plot(
                checkpoints,
                means,
                marker="o",
                label=ALGORITHM_LABELS[algorithm],
            )
            axis.fill_between(
                checkpoints,
                means - 1.96 * standard_errors,
                means + 1.96 * standard_errors,
                color=line.get_color(),
                alpha=0.12,
            )

    axes[0].set_ylabel("Episode return")
    axes[1].set_ylabel("Success rate")
    axes[1].set_ylim(-0.02, 1.02)
    for axis in axes:
        axis.set_xlabel("Training completed (%)")
        axis.set_xticks(EVALUATION_CHECKPOINTS)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Frozen stochastic-policy evaluation")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def mean_and_std(row: dict[str, object], metric: str) -> str:
    return f"{float(row[metric]):.3f} +/- {float(row[f'{metric}_std']):.3f}"


def final_seed_table(rows: list[dict[str, object]]) -> SummaryTable:
    """Show the final frozen evaluation separately for every seed."""
    values = []
    for algorithm in ALGORITHMS:
        for seed in sorted({int(row["seed"]) for row in rows}):
            selected = [
                row
                for row in rows
                if row["algorithm"] == algorithm
                and int(row["seed"]) == seed
                and int(row["checkpoint"]) == 100
            ]
            if selected:
                mean_return = np.mean(
                    [float(row["episode_return"]) for row in selected]
                )
                mean_success = np.mean([float(row["success"]) for row in selected])
                mean_length = np.mean(
                    [float(row["episode_length"]) for row in selected]
                )
                values.append(
                    (
                        ALGORITHM_LABELS[algorithm],
                        seed,
                        f"{mean_return:.3f}",
                        f"{mean_success:.3f}",
                        f"{mean_length:.2f}",
                    )
                )
    return SummaryTable(
        "Final evaluation by seed",
        ("Algorithm", "Seed", "Mean return", "Success rate", "Episode length"),
        tuple(values),
    )


def serialized_bounds(bounds: np.ndarray) -> list[float | str]:
    """Represent unbounded Box limits without non-standard JSON numbers."""
    result: list[float | str] = []
    for value in bounds.reshape(-1):
        if np.isneginf(value):
            result.append("-Infinity")
        elif np.isposinf(value):
            result.append("Infinity")
        else:
            result.append(float(value))
    return result


def write_outputs(
    output: Path,
    config: ExperimentConfig,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    """Write final raw data, measured figures, and the deterministic summary."""
    write_csv(output / "metrics.csv", CSV_FIELDS, rows)
    write_metadata(output / "metadata.json", metadata)

    aggregate = aggregate_rows(rows, config.algorithms)
    write_figure(output / "figures" / "checkpoint_evaluation.png", aggregate)
    table_rows = tuple(
        (
            ALGORITHM_LABELS[str(row["algorithm"])],
            f"{row['checkpoint']}%",
            mean_and_std(row, "episode_return"),
            mean_and_std(row, "episode_length"),
            mean_and_std(row, "success"),
            mean_and_std(row, "truncated"),
        )
        for row in aggregate
    )
    recordings = tuple(
        SummaryMedia(
            recording_title(path, metadata["recording_seed_by_algorithm"]),
            path.relative_to(output),
        )
        for path in sorted((output / "recordings").glob("*.gif"))
    )
    write_summary(
        output / "summary.html",
        title=f"Policy gradients on {config.environment}",
        metadata={
            "Environment": config.environment,
            "Preset": config.preset,
            "Training episodes": config.training_episodes,
            "Evaluation": (
                f"{config.evaluation_episodes} episodes per checkpoint and seed"
            ),
            "Seeds": config.seeds,
            "Network": (
                f"actor MLP with hidden sizes {list(config.hidden_sizes)}; "
                "baseline and A2C critics use the same sizes"
            ),
            "Optimizer": {
                "adam": "Adam",
                "adamw": "AdamW",
                "sgd": "SGD",
            }[config.optimizer],
            "Actor learning rate": (
                f"{config.actor_learning_rate} to "
                f"{config.actor_minimum_learning_rate} (cosine)"
            ),
            "Critic learning rate": (
                f"{config.critic_learning_rate} to "
                f"{config.critic_minimum_learning_rate} (cosine)"
            ),
            "Weight decay": config.weight_decay,
            "Warmup": (
                f"{config.warmup_episodes} episodes from "
                f"{config.warmup_start_factor:g}x peak learning rate"
                if config.warmup_episodes
                else "disabled"
            ),
            "Discount": config.discount,
            "A2C rollout": f"up to {config.a2c_rollout_steps} transitions",
            "A2C entropy coefficient": config.entropy_coefficient,
            "Training diagnostics": (
                "training_diagnostics.csv" if config.diagnostics else "disabled"
            ),
            "Evaluation policy": "frozen stochastic policy",
            "Success": metadata["success_definition"],
            "Observations": "raw, flattened float32",
            "Recorded seed by algorithm": metadata["recording_seed_by_algorithm"],
        },
        tables=(
            SummaryTable(
                "Frozen-policy evaluation",
                (
                    "Algorithm",
                    "Training budget",
                    "Episode return",
                    "Episode length",
                    "Success rate",
                    "Truncation rate",
                ),
                table_rows,
            ),
            final_seed_table(rows),
        ),
        figures=(
            SummaryMedia(
                "Evaluation return and success",
                Path("figures/checkpoint_evaluation.png"),
            ),
        ),
        recordings=recordings,
    )


def initial_metadata(config: ExperimentConfig) -> dict[str, object]:
    """Build the runner-owned part of the reproducibility metadata."""
    inspection_env = make_environment(config.environment)
    assert isinstance(inspection_env.observation_space, gym.spaces.Box)
    assert isinstance(inspection_env.action_space, gym.spaces.Discrete)
    max_episode_steps = (
        inspection_env.spec.max_episode_steps
        if inspection_env.spec is not None
        else None
    )
    observation_shape = list(inspection_env.observation_space.shape)
    observation_low = serialized_bounds(inspection_env.observation_space.low)
    observation_high = serialized_bounds(inspection_env.observation_space.high)
    number_of_actions = int(inspection_env.action_space.n)
    inspection_env.close()

    success_definition = (
        "reaches the time limit without true termination"
        if config.environment == "CartPole-v1"
        else "true termination after the Acrobot reaches the target height"
    )
    metadata = asdict(config)
    metadata.update(
        {
            "status": "running",
            "algorithms": list(config.algorithms),
            "evaluation_checkpoints": list(EVALUATION_CHECKPOINTS),
            "recording_checkpoints": list(config.recording_checkpoints),
            "observation_shape": observation_shape,
            "observation_low": observation_low,
            "observation_high": observation_high,
            "observation_processing": "convert to float32 and flatten; no scaling",
            "number_of_actions": number_of_actions,
            "max_episode_steps": max_episode_steps,
            "training_environment_seed": "seed * training_episodes + episode",
            "training_action_seed": ("10000000 + seed * training_episodes + episode"),
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
            "recording_frame_stride": (4 if config.recording_checkpoints else None),
            "evaluation_policy": "frozen stochastic categorical actor",
            "success_definition": success_definition,
            "episode_stopping": "termination or truncation",
            "truncation_target": (
                "Monte Carlo trajectories end without bootstrap; A2C bootstraps "
                "the final observation after truncation"
            ),
            "paired_initialization": (
                "same actor initialization for all algorithms within each seed; "
                "baseline and A2C also share critic initialization"
            ),
            "paired_environment_seeds": True,
            "algorithm_updates": {
                "reinforce": "one Monte Carlo policy update after each episode",
                "reinforce_with_baseline": (
                    "one Monte Carlo actor and critic update after each episode"
                ),
                "a2c": ("actor and critic updates after each bounded n-step rollout"),
            },
            "learning_rate_schedule": (
                "independent linear warmup then cosine annealing for actor "
                "and critic; stepped once per training episode"
            ),
            "training_diagnostics_file": (
                "training_diagnostics.csv" if config.diagnostics else None
            ),
            "training_diagnostics_definition": (
                "one row per completed training episode; A2C losses are "
                "transition-weighted means across its rollout updates"
                if config.diagnostics
                else None
            ),
            "variability": "standard deviation across seed evaluation means",
            "confidence_interval": "mean +/- 1.96 * standard error",
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


def run_policy_gradient_experiment(
    config: ExperimentConfig,
    output: Path,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    """Train and evaluate all algorithms with paired reproducible seeds."""
    rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    metrics_path = output / "metrics.csv"
    diagnostics_path = (
        output / "training_diagnostics.csv" if config.diagnostics else None
    )
    if diagnostics_path is not None:
        write_csv(diagnostics_path, DIAGNOSTIC_FIELDS, diagnostic_rows)
    selected_recording_seeds: dict[str, int] = {}

    for algorithm in config.algorithms:
        print("", flush=True)
        print("=" * 72, flush=True)
        print(f"ALGORITHM: {ALGORITHM_LABELS[algorithm]}", flush=True)
        print("=" * 72, flush=True)
        recording_states: dict[int, dict[int, ModelState]] = {}
        final_scores: dict[int, tuple[float, float]] = {}
        recording_agent: Agent | None = None
        for seed in range(config.seeds):
            print("", flush=True)
            print(
                f"RUN {seed + 1}/{config.seeds} | seed={seed} | fresh network",
                flush=True,
            )
            print("-" * 72, flush=True)
            training_env = make_environment(config.environment)
            evaluation_env = make_environment(config.environment)
            agent = make_agent(
                algorithm,
                training_env,
                actor_learning_rate=config.actor_learning_rate,
                critic_learning_rate=config.critic_learning_rate,
                optimizer_name=config.optimizer,
                weight_decay=config.weight_decay,
                discount=config.discount,
                entropy_coefficient=config.entropy_coefficient,
                hidden_sizes=config.hidden_sizes,
                seed=seed,
            )
            actor_scheduler = make_learning_rate_scheduler(
                agent.actor_optimizer,
                training_episodes=config.training_episodes,
                minimum_learning_rate=config.actor_minimum_learning_rate,
                warmup_episodes=config.warmup_episodes,
                warmup_start_factor=config.warmup_start_factor,
            )
            agent_critic_optimizer = critic_optimizer(agent)
            critic_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
            if agent_critic_optimizer is not None:
                critic_scheduler = make_learning_rate_scheduler(
                    agent_critic_optimizer,
                    training_episodes=config.training_episodes,
                    minimum_learning_rate=config.critic_minimum_learning_rate,
                    warmup_episodes=config.warmup_episodes,
                    warmup_start_factor=config.warmup_start_factor,
                )
            completed_episodes = 0
            try:
                for checkpoint in EVALUATION_CHECKPOINTS:
                    target_episodes = checkpoint_episode_target(
                        config.training_episodes,
                        checkpoint,
                    )
                    starting_episode = completed_episodes
                    if target_episodes > starting_episode:
                        print(
                            f"  CHECKPOINT {checkpoint:>3}% | train episodes "
                            f"{starting_episode + 1}-{target_episodes}",
                            flush=True,
                        )
                    else:
                        print(
                            f"  CHECKPOINT {checkpoint:>3}% | "
                            + (
                                "evaluate initial policy before training"
                                if checkpoint == 0
                                else "evaluate current policy; no additional training"
                            ),
                            flush=True,
                        )
                    training_results: list[TrainingEpisodeResult] = []
                    diagnostic_segment_start = len(diagnostic_rows)
                    while completed_episodes < target_episodes:
                        episode_index = completed_episodes
                        actor_learning_rate = optimizer_learning_rate(
                            agent.actor_optimizer
                        )
                        critic_learning_rate = (
                            optimizer_learning_rate(agent_critic_optimizer)
                            if agent_critic_optimizer is not None
                            else None
                        )
                        result = train_episode(
                            training_env,
                            agent,
                            rollout_steps=config.a2c_rollout_steps,
                            environment_seed=(
                                seed * config.training_episodes + episode_index
                            ),
                            action_seed=(
                                10_000_000
                                + seed * config.training_episodes
                                + episode_index
                            ),
                            collect_diagnostics=config.diagnostics,
                        )
                        training_results.append(result)
                        if diagnostics_path is not None:
                            diagnostic_rows.append(
                                diagnostic_row(
                                    algorithm,
                                    seed,
                                    episode_index + 1,
                                    actor_learning_rate,
                                    critic_learning_rate,
                                    result,
                                )
                            )
                        actor_scheduler.step()
                        if critic_scheduler is not None:
                            critic_scheduler.step()
                        completed_episodes += 1

                    if training_results:
                        mean_training_return = float(
                            np.mean([r.episode_return for r in training_results])
                        )
                        mean_training_length = float(
                            np.mean([r.episode_length for r in training_results])
                        )
                        mean_updates = float(
                            np.mean([r.updates for r in training_results])
                        )
                        print(
                            "    TRAIN | "
                            f"mean return={mean_training_return:.1f} | "
                            f"mean length={mean_training_length:.1f} | "
                            f"updates/episode={mean_updates:.2f} | "
                            "actor lr="
                            f"{optimizer_learning_rate(agent.actor_optimizer):.3g}"
                            + (
                                " | critic lr="
                                f"{optimizer_learning_rate(agent_critic_optimizer):.3g}"
                                if agent_critic_optimizer is not None
                                else ""
                            ),
                            flush=True,
                        )
                    if diagnostics_path is not None:
                        diagnostic_segment = diagnostic_rows[diagnostic_segment_start:]
                        print_diagnostic_summary(diagnostic_segment)
                        write_csv(
                            diagnostics_path,
                            DIAGNOSTIC_FIELDS,
                            diagnostic_rows,
                        )

                    checkpoint_results: list[EvaluationResult] = []
                    for evaluation_episode in range(config.evaluation_episodes):
                        environment_seed = (
                            1_000_000
                            + seed * 100_000
                            + checkpoint * 1_000
                            + evaluation_episode
                        )
                        action_seed = (
                            11_000_000
                            + seed * 100_000
                            + checkpoint * 1_000
                            + evaluation_episode
                        )
                        result = evaluate_episode(
                            evaluation_env,
                            config.environment,
                            agent,
                            environment_seed=environment_seed,
                            action_seed=action_seed,
                        )
                        checkpoint_results.append(result)
                        rows.append(
                            evaluation_row(
                                algorithm,
                                seed,
                                checkpoint,
                                evaluation_episode,
                                result,
                            )
                        )

                    mean_return = float(
                        np.mean(
                            [result.episode_return for result in checkpoint_results]
                        )
                    )
                    success_rate = float(
                        np.mean([result.success for result in checkpoint_results])
                    )
                    print(
                        f"    EVAL  | mean return={mean_return:.1f} | "
                        f"success={success_rate:.0%} | "
                        f"episodes={config.evaluation_episodes}",
                        flush=True,
                    )
                    write_csv(metrics_path, CSV_FIELDS, rows)

                    if checkpoint in config.recording_checkpoints:
                        recording_states.setdefault(seed, {})[checkpoint] = {
                            name: value.detach().clone()
                            for name, value in agent.actor_model.state_dict().items()
                        }
                    if checkpoint == 100:
                        final_scores[seed] = (mean_return, success_rate)
            finally:
                training_env.close()
                evaluation_env.close()
            recording_agent = agent

        if config.recording_checkpoints:
            selected_seed = max(final_scores, key=final_scores.__getitem__)
            selected_recording_seeds[algorithm] = selected_seed
            assert recording_agent is not None
            print(
                f"  RECORDINGS | selected seed={selected_seed} from final evaluation",
                flush=True,
            )
            for checkpoint in config.recording_checkpoints:
                recording_agent.actor_model.load_state_dict(
                    recording_states[selected_seed][checkpoint]
                )
                record_evaluation(
                    output
                    / "recordings"
                    / f"{algorithm}_checkpoint_{checkpoint:03d}.gif",
                    config.environment,
                    recording_agent,
                    environment_seed=2_000_000 + checkpoint,
                    action_seed=12_000_000 + checkpoint,
                    frame_stride=4,
                )
                print(f"    recorded checkpoint {checkpoint}%", flush=True)
    return rows, selected_recording_seeds


def main(argv: Sequence[str] | None = None) -> None:
    config = parse_config(argv)
    output = create_run_directory("policy_gradient", config.environment)
    (output / "figures").mkdir()
    if config.recording_checkpoints:
        (output / "recordings").mkdir()

    metadata = initial_metadata(config)
    write_metadata(output / "metadata.json", metadata)
    write_csv(output / "metrics.csv", CSV_FIELDS, [])
    print_experiment_header(config)
    try:
        rows, selected_recording_seeds = run_policy_gradient_experiment(config, output)
        metadata["recording_seed_selection"] = (
            "highest final mean evaluation return; success rate breaks ties"
        )
        metadata["recording_seed_by_algorithm"] = selected_recording_seeds
        write_outputs(output, config, rows, metadata)
    except Exception:
        metadata["status"] = "failed"
        write_metadata(output / "metadata.json", metadata)
        raise

    metadata["status"] = "complete"
    write_metadata(output / "metadata.json", metadata)
    print_final_evaluation(rows, config.algorithms)
    print("EXPERIMENT COMPLETE", flush=True)
    print(f"  Output  : {output}", flush=True)
    print(f"  Summary : {output / 'summary.html'}", flush=True)
    print(f"  Metrics : {output / 'metrics.csv'}", flush=True)
    if config.diagnostics:
        print(f"  Diag    : {output / 'training_diagnostics.csv'}", flush=True)


if __name__ == "__main__":
    main()
