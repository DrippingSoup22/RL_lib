"""Run tabular TD prediction and control on Gymnasium environments."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from experiments.common import (
    EVALUATION_CHECKPOINTS,
    RECORDING_CHECKPOINTS,
    annotated_frame,
    checkpoint_episode_target,
    create_run_directory,
    write_csv,
    write_metadata,
)
from experiments.summary import SummaryMedia, SummaryTable, write_summary
from rl_lib.algorithms.temporal_difference import (
    SARSA,
    QLearning,
    TDZeroPrediction,
)

CSV_FIELDS = (
    "phase",
    "algorithm",
    "seed",
    "checkpoint",
    "state",
    "V",
    "visits",
    "mean_return",
    "success_rate",
    "mean_episode_length",
    "truncation_rate",
    "mean_cliff_falls",
)


def experiment_defaults(environment: str, preset: str) -> dict[str, int]:
    if preset == "quick":
        return {
            "prediction_episodes": 50,
            "training_episodes": 100,
            "evaluation_episodes": 10,
            "seeds": 1,
        }
    if environment == "FrozenLake-v1":
        return {
            "prediction_episodes": 1_000,
            "training_episodes": 5_000,
            "evaluation_episodes": 200,
            "seeds": 3,
        }
    return {
        "prediction_episodes": 500,
        "training_episodes": 1_000,
        "evaluation_episodes": 100,
        "seeds": 3,
    }


def environment_configuration(
    environment: str,
    seeds: int,
    map_size: int,
    safe_probability: float,
    slippery: bool,
    max_episode_steps: int | None,
) -> tuple[list[dict[str, object]], int, int, int]:
    if environment == "FrozenLake-v1":
        resolved_max_steps = max_episode_steps or (100 if map_size <= 4 else 200)
        maps = [
            generate_random_map(size=map_size, p=safe_probability, seed=seed)
            for seed in range(seeds)
        ]
        configurations = [
            {
                "desc": map_description,
                "is_slippery": slippery,
                "max_episode_steps": resolved_max_steps,
            }
            for map_description in maps
        ]
        return configurations, map_size * map_size, 4, resolved_max_steps

    resolved_max_steps = max_episode_steps or 200
    configurations = [
        {"max_episode_steps": resolved_max_steps} for _seed in range(seeds)
    ]
    return configurations, 48, 4, resolved_max_steps


def make_environment(
    environment: str,
    configuration: dict[str, object],
    *,
    render_mode: str | None = None,
) -> gym.Env:
    kwargs = dict(configuration)
    max_episode_steps = int(kwargs.pop("max_episode_steps"))
    return gym.make(
        environment,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        **kwargs,
    )


def fixed_prediction_action(
    environment: str,
    rng: np.random.Generator,
    state: int,
) -> int:
    if environment == "CliffWalking-v1":
        row, column = divmod(state, 12)
        if row == 3 and column < 11:
            return 0  # up
        if column < 11:
            return 1  # right
        return 2  # down
    return int(rng.integers(4))


def greedy_action(Q: np.ndarray, rng: np.random.Generator, state: int) -> int:
    values = Q[state]
    return int(rng.choice(np.flatnonzero(values == np.max(values))))


def run_prediction(
    environment: str,
    configuration: dict[str, object],
    number_of_states: int,
    episodes: int,
    learning_rate: float,
    discount: float,
) -> list[dict[str, object]]:
    estimator = TDZeroPrediction(
        number_of_states,
        learning_rate=learning_rate,
        discount=discount,
    )
    visits = np.zeros(number_of_states, dtype=int)
    rng = np.random.default_rng(0)
    env = make_environment(environment, configuration)
    for episode in range(episodes):
        observation, _ = env.reset(seed=episode)
        state = int(observation)
        terminated = truncated = False
        while not (terminated or truncated):
            action = fixed_prediction_action(environment, rng, state)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_observation)
            estimator.update(state, float(reward), next_state, terminated)
            visits[state] += 1
            state = next_state
    env.close()

    return [
        {
            "phase": "prediction",
            "algorithm": "td_zero",
            "seed": 0,
            "checkpoint": 100,
            "state": state,
            "V": float(estimator.V[state]),
            "visits": int(visits[state]),
            "mean_return": "",
            "success_rate": "",
            "mean_episode_length": "",
            "truncation_rate": "",
            "mean_cliff_falls": "",
        }
        for state in range(number_of_states)
    ]


def train_episode(
    env: gym.Env,
    agent: SARSA | QLearning,
    algorithm: str,
    seed: int,
) -> None:
    observation, _ = env.reset(seed=seed)
    state = int(observation)
    action = agent.select_action(state)
    terminated = truncated = False
    while not (terminated or truncated):
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_state = int(next_observation)
        if algorithm == "sarsa":
            next_action = None if terminated else agent.select_action(next_state)
            agent.update(
                state,
                action,
                float(reward),
                next_state,
                next_action,
                terminated,
            )
        else:
            agent.update(state, action, float(reward), next_state, terminated)
            next_action = None if terminated else agent.select_action(next_state)

        if not (terminated or truncated):
            state = next_state
            action = next_action


def evaluate(
    env: gym.Env,
    environment: str,
    Q: np.ndarray,
    episodes: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    returns: list[float] = []
    successes: list[float] = []
    lengths: list[float] = []
    truncations: list[float] = []
    cliff_falls: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed + episode)
        state = int(observation)
        terminated = truncated = False
        total_reward = 0.0
        length = 0
        falls = 0
        while not (terminated or truncated):
            action = greedy_action(Q, rng, state)
            observation, reward, terminated, truncated, _ = env.step(action)
            state = int(observation)
            total_reward += float(reward)
            length += 1
            falls += environment == "CliffWalking-v1" and reward == -100
        returns.append(total_reward)
        success = terminated and (environment == "CliffWalking-v1" or float(reward) > 0)
        successes.append(float(success))
        lengths.append(float(length))
        truncations.append(float(truncated))
        cliff_falls.append(float(falls))
    return {
        "mean_return": float(np.mean(returns)),
        "success_rate": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(lengths)),
        "truncation_rate": float(np.mean(truncations)),
        "mean_cliff_falls": float(np.mean(cliff_falls)),
    }


def action_name(environment: str, action: int) -> str:
    if environment == "CliffWalking-v1":
        return ("up", "right", "down", "left")[action]
    return ("left", "down", "right", "up")[action]


def record_evaluation(
    path: Path,
    environment: str,
    configuration: dict[str, object],
    Q: np.ndarray,
    seed: int,
) -> None:
    env = make_environment(environment, configuration, render_mode="rgb_array")
    rng = np.random.default_rng(seed)
    observation, _ = env.reset(seed=seed)
    state = int(observation)
    frames = [annotated_frame(env.render(), "Evaluation: start")]
    durations = [150]
    terminated = truncated = False
    step = 0
    while not (terminated or truncated):
        action = greedy_action(Q, rng, state)
        observation, reward, terminated, truncated, _ = env.step(action)
        state = int(observation)
        step += 1
        frames.append(
            annotated_frame(
                env.render(),
                f"Step {step}: {action_name(environment, action)}, "
                f"reward {float(reward):g}",
            )
        )
        durations.append(1_000 if terminated or truncated else 150)
    env.close()
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
    )


def run_control(
    environment: str,
    configurations: list[dict[str, object]],
    number_of_states: int,
    number_of_actions: int,
    training_episodes: int,
    evaluation_episodes: int,
    learning_rate: float,
    discount: float,
    epsilon: float,
    recordings_directory: Path,
    metrics_path: Path,
    initial_rows: list[dict[str, object]],
    recording_checkpoints: tuple[int, ...],
) -> list[dict[str, object]]:
    rows = initial_rows.copy()
    algorithms: tuple[tuple[str, type[SARSA] | type[QLearning]], ...] = (
        ("sarsa", SARSA),
        ("q_learning", QLearning),
    )
    for algorithm, agent_class in algorithms:
        for seed, configuration in enumerate(configurations):
            print(f"{algorithm}: seed {seed + 1}/{len(configurations)}", flush=True)
            env = make_environment(environment, configuration)
            agent = agent_class(
                number_of_states,
                number_of_actions,
                learning_rate=learning_rate,
                discount=discount,
                epsilon=epsilon,
                seed=seed,
            )
            completed_episodes = 0
            for checkpoint in EVALUATION_CHECKPOINTS:
                target_episodes = checkpoint_episode_target(
                    training_episodes,
                    checkpoint,
                )
                print(
                    f"  checkpoint {checkpoint}%: train to {target_episodes}, "
                    f"then evaluate {evaluation_episodes} episodes",
                    flush=True,
                )
                while completed_episodes < target_episodes:
                    train_episode(
                        env,
                        agent,
                        algorithm,
                        seed=seed * training_episodes + completed_episodes,
                    )
                    completed_episodes += 1

                measurements = evaluate(
                    env,
                    environment,
                    agent.Q,
                    evaluation_episodes,
                    seed=1_000_000 + checkpoint * 10_000 + seed * evaluation_episodes,
                )
                rows.append(
                    {
                        "phase": "control",
                        "algorithm": algorithm,
                        "seed": seed,
                        "checkpoint": checkpoint,
                        "state": "",
                        "V": "",
                        "visits": "",
                        **measurements,
                    }
                )
                write_csv(metrics_path, CSV_FIELDS, rows)
                if seed == 0 and checkpoint in recording_checkpoints:
                    record_evaluation(
                        recordings_directory
                        / f"{algorithm}_checkpoint_{checkpoint:03d}.gif",
                        environment,
                        configuration,
                        agent.Q,
                        seed=2_000_000 + checkpoint,
                    )
            env.close()
    return rows


def aggregate_control_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    aggregate = []
    for algorithm in ("sarsa", "q_learning"):
        for checkpoint in EVALUATION_CHECKPOINTS:
            selected = [
                row
                for row in rows
                if row["phase"] == "control"
                and row["algorithm"] == algorithm
                and row["checkpoint"] == checkpoint
            ]
            result: dict[str, object] = {
                "algorithm": algorithm,
                "checkpoint": checkpoint,
            }
            for metric in (
                "mean_return",
                "success_rate",
                "mean_episode_length",
                "truncation_rate",
                "mean_cliff_falls",
            ):
                values = np.asarray([row[metric] for row in selected], dtype=float)
                result[metric] = float(np.mean(values))
                result[f"{metric}_std"] = (
                    float(np.std(values, ddof=1)) if values.size > 1 else 0.0
                )
            aggregate.append(result)
    return aggregate


def write_figures(
    output: Path,
    environment: str,
    rows: list[dict[str, object]],
    number_of_states: int,
) -> None:
    figures = output / "figures"
    figures.mkdir()
    prediction = [row for row in rows if row["phase"] == "prediction"]
    V = np.asarray([row["V"] for row in prediction], dtype=float)
    columns = 12 if environment == "CliffWalking-v1" else int(np.sqrt(number_of_states))
    value_grid = V.reshape(-1, columns)
    figure, axis = plt.subplots(figsize=(9, 3.5))
    image = axis.imshow(value_grid, cmap="viridis", aspect="auto")
    axis.set_title("TD(0) state-value estimates")
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    figure.colorbar(image, ax=axis, label="V(s)")
    figure.tight_layout()
    figure.savefig(figures / "prediction_values.png", dpi=150)
    plt.close(figure)

    aggregate = aggregate_control_rows(rows)
    second_metric = (
        "mean_episode_length" if environment == "CliffWalking-v1" else "success_rate"
    )
    second_label = (
        "Mean episode length" if environment == "CliffWalking-v1" else "Success rate"
    )
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for algorithm in ("sarsa", "q_learning"):
        selected = [row for row in aggregate if row["algorithm"] == algorithm]
        checkpoints = [row["checkpoint"] for row in selected]
        axes[0].plot(
            checkpoints,
            [row["mean_return"] for row in selected],
            marker="o",
            label=algorithm,
        )
        axes[1].plot(
            checkpoints,
            [row[second_metric] for row in selected],
            marker="o",
            label=algorithm,
        )
    axes[0].set_ylabel("Mean evaluation return")
    axes[1].set_ylabel(second_label)
    for axis in axes:
        axis.set_xlabel("Training completed (%)")
        axis.set_xticks(EVALUATION_CHECKPOINTS)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "control_learning.png", dpi=150)
    plt.close(figure)


def write_outputs(
    output: Path,
    rows: list[dict[str, object]],
    metadata: dict[str, object],
) -> None:
    write_csv(output / "metrics.csv", CSV_FIELDS, rows)
    write_metadata(output / "metadata.json", metadata)
    environment = str(metadata["environment"])
    write_figures(output, environment, rows, int(metadata["number_of_states"]))
    aggregate = aggregate_control_rows(rows)
    control_table = []
    for row in aggregate:
        values = (
            row["algorithm"],
            f"{row['checkpoint']}%",
            f"{float(row['mean_return']):.3f}",
            f"{float(row['success_rate']):.3f}",
            f"{float(row['mean_episode_length']):.2f}",
            f"{float(row['truncation_rate']):.3f}",
        )
        if environment == "CliffWalking-v1":
            values += (f"{float(row['mean_cliff_falls']):.3f}",)
        control_table.append(values)
    control_headers = (
        "Algorithm",
        "Training",
        "Mean return",
        "Success rate",
        "Episode length",
        "Truncation rate",
    )
    if environment == "CliffWalking-v1":
        control_headers += ("Cliff falls",)
    prediction_rows = [row for row in rows if row["phase"] == "prediction"]
    visited_values = [
        float(row["V"]) for row in prediction_rows if int(row["visits"]) > 0
    ]
    recordings = tuple(
        SummaryMedia(
            path.stem.replace("_checkpoint_", " - ").replace("_", " ") + "%",
            path.relative_to(output),
        )
        for path in sorted((output / "recordings").glob("*.gif"))
    )
    write_summary(
        output / "summary.html",
        title=f"Temporal difference on {environment}",
        metadata={
            "Environment": environment,
            "Preset": metadata["preset"],
            "TD(0) prediction episodes": metadata["prediction_episodes"],
            "Control training episodes": metadata["training_episodes"],
            "Evaluation": (
                f"{metadata['evaluation_episodes']} episodes per checkpoint and seed"
            ),
            "Seeds": metadata["seeds"],
            "Learning rate": metadata["learning_rate"],
            "Discount": metadata["discount"],
            "Training epsilon": metadata["epsilon"],
        },
        tables=(
            SummaryTable(
                "TD(0) prediction coverage",
                ("States", "Visited states", "Total visits", "Value range"),
                (
                    (
                        len(prediction_rows),
                        len(visited_values),
                        sum(int(row["visits"]) for row in prediction_rows),
                        (
                            f"{min(visited_values):.3f} to {max(visited_values):.3f}"
                            if visited_values
                            else "not visited"
                        ),
                    ),
                ),
            ),
            SummaryTable(
                "Greedy checkpoint control evaluation",
                control_headers,
                control_table,
            ),
        ),
        figures=(
            SummaryMedia(
                "Control checkpoint evaluation", Path("figures/control_learning.png")
            ),
            SummaryMedia("TD(0) state values", Path("figures/prediction_values.png")),
        ),
        recordings=recordings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment",
        choices=("CliffWalking-v1", "FrozenLake-v1"),
        default="CliffWalking-v1",
    )
    parser.add_argument("--preset", choices=("quick", "standard"), default="standard")
    parser.add_argument("--prediction-episodes", type=int)
    parser.add_argument("--training-episodes", type=int)
    parser.add_argument("--evaluation-episodes", type=int)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--map-size", type=int, default=4)
    parser.add_argument("--safe-probability", type=float, default=0.8)
    parser.add_argument("--non-slippery", action="store_true")
    parser.add_argument("--max-episode-steps", type=int)
    args = parser.parse_args()
    defaults = experiment_defaults(args.environment, args.preset)
    prediction_episodes = (
        args.prediction_episodes
        if args.prediction_episodes is not None
        else defaults["prediction_episodes"]
    )
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
    if min(prediction_episodes, training_episodes, evaluation_episodes, seeds) < 1:
        parser.error("episode counts and seeds must be positive")
    if not np.isfinite(args.learning_rate) or not 0 < args.learning_rate <= 1:
        parser.error("learning rate must be finite and in (0, 1]")
    if not np.isfinite(args.discount) or not 0 <= args.discount <= 1:
        parser.error("discount must be finite and in [0, 1]")
    if not np.isfinite(args.epsilon) or not 0 <= args.epsilon <= 1:
        parser.error("epsilon must be finite and in [0, 1]")
    if args.map_size < 2:
        parser.error("map size must be at least 2")
    if not 0 < args.safe_probability <= 1:
        parser.error("safe probability must be in (0, 1]")
    if args.max_episode_steps is not None and args.max_episode_steps < 1:
        parser.error("max episode steps must be positive")

    configurations, states, actions, max_steps = environment_configuration(
        args.environment,
        seeds,
        args.map_size,
        args.safe_probability,
        not args.non_slippery,
        args.max_episode_steps,
    )
    output = create_run_directory("temporal_difference", args.environment)
    recordings = output / "recordings"
    recordings.mkdir()
    metadata: dict[str, object] = {
        "status": "running",
        "environment": args.environment,
        "preset": args.preset,
        "prediction_episodes": prediction_episodes,
        "training_episodes": training_episodes,
        "evaluation_episodes": evaluation_episodes,
        "seeds": seeds,
        "number_of_states": states,
        "number_of_actions": actions,
        "learning_rate": args.learning_rate,
        "discount": args.discount,
        "epsilon": args.epsilon,
        "max_episode_steps": max_steps,
        "evaluation_checkpoints": list(EVALUATION_CHECKPOINTS),
        "recording_checkpoints": (
            [100] if args.preset == "quick" else list(RECORDING_CHECKPOINTS)
        ),
        "evaluation_policy": "frozen greedy",
        "prediction_policy": (
            "fixed safe path"
            if args.environment == "CliffWalking-v1"
            else "uniform random"
        ),
        "truncation_target": "bootstrap then stop interaction",
        "variability": "standard deviation across seed evaluation means",
        "smoothing_window": 1,
        "frozen_lake": (
            {
                "map_size": args.map_size,
                "safe_probability": args.safe_probability,
                "is_slippery": not args.non_slippery,
                "maps": [configuration["desc"] for configuration in configurations],
            }
            if args.environment == "FrozenLake-v1"
            else None
        ),
    }
    write_metadata(output / "metadata.json", metadata)
    prediction_rows = run_prediction(
        args.environment,
        configurations[0],
        states,
        prediction_episodes,
        args.learning_rate,
        args.discount,
    )
    write_csv(output / "metrics.csv", CSV_FIELDS, prediction_rows)
    rows = run_control(
        args.environment,
        configurations,
        states,
        actions,
        training_episodes,
        evaluation_episodes,
        args.learning_rate,
        args.discount,
        args.epsilon,
        recordings,
        output / "metrics.csv",
        prediction_rows,
        (100,) if args.preset == "quick" else RECORDING_CHECKPOINTS,
    )
    write_outputs(output, rows, metadata)
    metadata["status"] = "complete"
    write_metadata(output / "metadata.json", metadata)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
