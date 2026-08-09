"""Episode generation for Monte Carlo experiments."""

from typing import Protocol

import gymnasium as gym

from rl_lib.data.episode import Episode, EpisodeStep


class ActionPolicy(Protocol):
    """Policy behavior required to generate a discrete episode."""

    def select_action(self, state: int) -> int: ...


def generate_episode(
    environment: gym.Env[int, int],
    policy: ActionPolicy,
    *,
    seed: int | None = None,
) -> Episode:
    """Generate one complete episode using ``policy`` from a normal reset."""
    state, _ = environment.reset(seed=seed)
    steps: list[EpisodeStep] = []

    while True:
        action = policy.select_action(state)
        next_state, reward, terminated, truncated, _ = environment.step(action)
        steps.append(
            EpisodeStep(
                state=int(state),
                action=int(action),
                reward=float(reward),
            )
        )
        state = next_state

        if terminated or truncated:
            return Episode(
                steps=tuple(steps),
                final_state=int(state),
                terminated=terminated,
                truncated=truncated,
            )
