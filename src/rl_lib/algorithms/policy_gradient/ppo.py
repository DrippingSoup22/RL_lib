"""Clipped proximal policy optimization, on batches of tensors."""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from rl_lib.networks import (
    CategoricalPolicyNetwork,
    GaussianPolicyNetwork,
    StateValueNetwork,
)
from rl_lib.optimization import clip_gradients, validate_max_gradient_norm
from rl_lib.policies import CategoricalPolicy, SquashedGaussianPolicy
from rl_lib.policies.neural import COLORED_NOISE_STEPS


@dataclass(frozen=True)
class PPOActionSample:
    """One sampled action per observation, with what learning keeps about it.

    ``environment_action`` is what the environment receives. ``policy_action``
    is what ``update`` re-evaluates later: the latent Gaussian sample for
    continuous actions, and the action itself for categorical ones.
    ``log_probability`` is the sampling policy's log-probability of it,
    ``(B,)``, and ``value`` the critic's value of the observation, ``(B,)``,
    or with several critics each one's, ``(B, K)``.
    """

    environment_action: torch.Tensor
    policy_action: torch.Tensor
    log_probability: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class PPOUpdateSummary:
    """How one update went.

    The first five values are means over all minibatches of the update; with
    several critics, ``critic_loss`` is the sum of theirs.
    ``explained_variance`` describes the critic that collected the batch,
    before the update: ``1 - Var(return_targets - values) / Var(return_targets)``
    over the batch, or with action masks over the samples that acted, where
    the actor learns from its values. It is 1 when the values predicted the
    return targets exactly, 0 when they explained none of their variation,
    negative when they
    did worse than predicting the mean, and NaN when every return target is the
    same. With several critics it is the first critic's, and
    ``explained_variances`` holds every critic's, in order.
    """

    actor_loss: float
    critic_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    explained_variance: float
    explained_variances: tuple[float, ...] = ()


class PPO:
    """PPO with a separate actor and critic, for categorical or bounded actions.

    Everything works on batches of tensors on the networks' device: the caller
    moves the networks there before creating their optimizers and this PPO, and
    passes observations and stored data already on it. Nothing makes the CPU
    wait for the GPU, except reading the summary once at the end of an update.

    The critic may be several critics, one for each of several rewards the
    task pays, as in the hybrid reward architecture (van Seijen et al.,
    NeurIPS 2017), which learns one value function per reward component: a
    critic that predicts one reward alone learns it from what it depends on,
    instead of seeing it drowned in another's variation. Each critic then has
    its own rewards, values, advantages and return targets, the last
    dimension, ``K``, of each. The actor learns from one advantage: each
    critic's, normalised over the batch on its own, times its weight in
    ``advantage_weights``, added up. So the weights set how much each reward
    counts against the others, whatever its scale.
    """

    def __init__(
        self,
        actor_network: CategoricalPolicyNetwork | GaussianPolicyNetwork,
        actor_optimizer: torch.optim.Optimizer,
        critic_network: StateValueNetwork | Sequence[StateValueNetwork],
        critic_optimizer: torch.optim.Optimizer,
        clip_ratio: float = 0.2,
        entropy_coefficient: float = 0.0,
        seed: int | None = None,
        *,
        max_gradient_norm: float | None = None,
        action_low: torch.Tensor | Sequence[float] | None = None,
        action_high: torch.Tensor | Sequence[float] | None = None,
        noise_beta: float = 0.0,
        noise_sequence_steps: int = COLORED_NOISE_STEPS,
        temporal_smoothness_coefficient: float = 0.0,
        advantage_weights: Sequence[float] | None = None,
    ) -> None:
        """Check the settings once and choose the policy from the actor network.

        ``seed`` starts ``generator``, the one source of everything random this
        PPO does: sampling actions, estimating the continuous entropy, and
        shuffling minibatches. Without it the generator starts from a random
        seed. Continuous actors need ``action_low`` and ``action_high``, and may
        color their exploration noise with ``noise_beta``, drawn
        ``noise_sequence_steps`` steps at a time (``SquashedGaussianPolicy``);
        categorical actors must not get bounds or a noise color.

        ``temporal_smoothness_coefficient`` adds the temporal term of CAPS
        (Mysore et al., ICRA 2021) to the actor's loss, continuous actors only:
        that coefficient times the mean distance between the policy's mean
        actions in each state and in the state after it, so that ``update``
        then needs ``next_observations``.

        ``critic_network`` may be a sequence of several critics, with one
        ``critic_optimizer`` for all their parameters and one weight each in
        ``advantage_weights`` (1 each by default); a single critic takes no
        weights.
        """
        several_critics = not isinstance(critic_network, StateValueNetwork)
        critic_networks = list(critic_network) if several_critics else [critic_network]
        if not critic_networks or any(
            actor_network.observation_size != critic.observation_size
            for critic in critic_networks
        ):
            raise ValueError("Actor and critic networks must have the same input size")
        if advantage_weights is not None and (
            not several_critics
            or len(advantage_weights) != len(critic_networks)
            or not all(math.isfinite(weight) for weight in advantage_weights)
        ):
            raise ValueError(
                "Advantage weights need several critics, one finite weight each"
            )
        if not math.isfinite(clip_ratio) or not 0 < clip_ratio < 1:
            raise ValueError("Clip ratio must be finite and in (0, 1)")
        if not math.isfinite(entropy_coefficient) or entropy_coefficient < 0:
            raise ValueError("Entropy coefficient must be finite and nonnegative")
        if (
            not math.isfinite(temporal_smoothness_coefficient)
            or temporal_smoothness_coefficient < 0
        ):
            raise ValueError(
                "Temporal smoothness coefficient must be finite and nonnegative"
            )

        self.actor_network = actor_network
        self.actor_optimizer = actor_optimizer
        self.critic_network = critic_network
        self.critic_networks = critic_networks
        self.several_critics = several_critics
        self.critic_optimizer = critic_optimizer
        self.clip_ratio = clip_ratio
        self.entropy_coefficient = entropy_coefficient
        self.temporal_smoothness_coefficient = temporal_smoothness_coefficient
        self.max_gradient_norm = validate_max_gradient_norm(max_gradient_norm)

        # The networks' device is where all data must be, and where every
        # random number is drawn.
        self.device = next(actor_network.parameters()).device
        self.generator = torch.Generator(device=self.device)
        if seed is None:
            self.generator.seed()
        else:
            self.generator.manual_seed(seed)
        self.advantage_weights = torch.tensor(
            advantage_weights or [1.0] * len(critic_networks), device=self.device
        )

        self.policy: CategoricalPolicy | SquashedGaussianPolicy
        if isinstance(actor_network, CategoricalPolicyNetwork):
            if action_low is not None or action_high is not None:
                raise ValueError("Categorical PPO must not receive action bounds")
            if noise_beta:
                raise ValueError("Categorical PPO has no Gaussian noise to color")
            if temporal_smoothness_coefficient:
                raise ValueError("Categorical PPO has no mean action to smooth")
            self.policy = CategoricalPolicy(actor_network, generator=self.generator)
        elif isinstance(actor_network, GaussianPolicyNetwork):
            if action_low is None or action_high is None:
                raise ValueError(
                    "Continuous PPO requires lower and upper action bounds"
                )
            self.policy = SquashedGaussianPolicy(
                actor_network,
                action_low,
                action_high,
                generator=self.generator,
                noise_beta=noise_beta,
                noise_sequence_steps=noise_sequence_steps,
            )
        else:
            raise TypeError("Actor network must be categorical or Gaussian")

    def sample_action(
        self,
        observations: torch.Tensor,
        noise_scales: torch.Tensor | None = None,
        action_mask: torch.Tensor | None = None,
    ) -> PPOActionSample:
        """Sample one action per observation, for collecting data to learn from.

        ``observations`` is ``(B, observation_size)``. Nothing keeps gradients:
        the sample is a fixed record that ``update`` later compares with the
        changed policy. For continuous actions, ``noise_scales``
        ``(B, action_size)`` scales each sample's exploration, and
        ``action_mask`` ``(B, action_size)`` marks the components that will
        take effect (``SquashedGaussianPolicy``); ``update`` then needs the
        same scales and masks.
        """
        if (noise_scales is not None or action_mask is not None) and isinstance(
            self.policy, CategoricalPolicy
        ):
            raise ValueError("Noise scales and masks apply to continuous actions only")
        with torch.no_grad():
            value = self._values(observations)
            if isinstance(self.policy, CategoricalPolicy):
                action, log_probability = self.policy.sample(observations)
                # A categorical action is re-evaluated exactly as it was sent.
                environment_action = policy_action = action
            else:
                environment_action, policy_action, log_probability = self.policy.sample(
                    observations, noise_scales, action_mask
                )
        return PPOActionSample(
            environment_action=environment_action,
            policy_action=policy_action,
            log_probability=log_probability,
            value=value,
        )

    def select_action(
        self,
        observations: torch.Tensor,
        *,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """One environment action per observation, without the critic.

        For evaluation: ``deterministic`` takes the most likely categorical
        action, or the squashed Gaussian mean, instead of sampling.
        """
        with torch.no_grad():
            if deterministic:
                return self.policy.deterministic_action(observations)
            if isinstance(self.policy, CategoricalPolicy):
                action, _ = self.policy.sample(observations)
                return action
            environment_action, _, _ = self.policy.sample(observations)
            return environment_action

    def state_value(self, observations: torch.Tensor) -> torch.Tensor:
        """The critic's value of each observation, ``(B,)``, or with several
        critics each one's, ``(B, K)``, for bootstrapping."""
        with torch.no_grad():
            return self._values(observations)

    def update(
        self,
        observations: torch.Tensor,
        policy_actions: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        advantages: torch.Tensor,
        return_targets: torch.Tensor,
        *,
        update_epochs: int,
        minibatch_size: int,
        next_observations: torch.Tensor | None = None,
        noise_scales: torch.Tensor | None = None,
        action_masks: torch.Tensor | None = None,
    ) -> PPOUpdateSummary:
        """Learn from one collected batch over several shuffled epochs.

        The batch holds ``N`` samples on the networks' device: ``observations``
        ``(N, observation_size)``; ``policy_actions`` ``(N,)`` for categorical
        or ``(N, action_size)`` for continuous PPO, as stored from
        ``PPOActionSample.policy_action``; ``old_log_probabilities`` ``(N,)``;
        and ``advantages`` and ``return_targets``, ``(N,)``, or with several
        critics ``(N, K)``. Advantages are normalised once over the whole
        batch, each critic's on its own, and then weighted and added up. Each
        epoch uses every sample once, in a new random order, in minibatches
        of ``minibatch_size``. With a temporal smoothness coefficient,
        ``next_observations`` holds the observation after each sample's,
        ``(N, observation_size)``. Samples drawn with noise scales or action
        masks need them again, ``(N, action_size)``, so that each is evaluated
        under the distribution it was drawn from. With masks, only the
        samples that acted at all, those with a component marked, teach the
        actor: its loss, entropy, KL and clip fraction are means over them,
        and the advantages are normalised over them; every sample teaches the
        critics.
        """
        if update_epochs < 1:
            raise ValueError("Update epochs must be positive")
        if minibatch_size < 1:
            raise ValueError("Minibatch size must be positive")
        if self.temporal_smoothness_coefficient and next_observations is None:
            raise ValueError("Temporal smoothness needs the next observations")
        if (noise_scales is not None or action_masks is not None) and isinstance(
            self.policy, CategoricalPolicy
        ):
            raise ValueError("Noise scales and masks apply to continuous actions only")
        batch_size = observations.shape[0]
        if batch_size == 0 or any(
            values.shape[0] != batch_size
            for values in (
                policy_actions,
                old_log_probabilities,
                advantages,
                return_targets,
                *(() if noise_scales is None else (noise_scales,)),
                *(() if action_masks is None else (action_masks,)),
            )
        ):
            raise ValueError("Every input must hold the same nonzero number of samples")
        if advantages.shape != return_targets.shape or advantages.shape[1:] != (
            (len(self.critic_networks),) if self.several_critics else ()
        ):
            raise ValueError("Advantages and return targets need one column per critic")

        # Each critic's measures and normalisation run over a column of its
        # own; a single critic's batch is the one column.
        advantages = advantages.reshape(batch_size, -1)
        column_targets = return_targets.reshape(batch_size, -1)

        # The samples that teach the actor: with masks, those that acted.
        acting = (
            torch.ones(batch_size, device=self.device)
            if action_masks is None
            else action_masks.any(dim=-1).float()
        )
        acting_count = acting.sum().clamp(min=1.0)

        # How well the critics that collected the batch predicted their
        # returns where the actor learns from them. Return targets are
        # advantages plus values, so the part the values left unexplained is
        # the advantages, taken before normalisation.
        def acting_mean(columns: torch.Tensor) -> torch.Tensor:
            return (columns * acting[:, None]).sum(dim=0) / acting_count

        def acting_variance(columns: torch.Tensor) -> torch.Tensor:
            return acting_mean((columns - acting_mean(columns)).square())

        return_target_variance = acting_variance(column_targets)
        explained_variances = torch.where(
            return_target_variance > 0,
            1 - acting_variance(advantages) / return_target_variance,
            math.nan,
        )

        # Normalize once over the full batch, never independently per
        # minibatch, and only over the samples that teach the actor. Nearly
        # constant advantages are kept as they are rather than divided by
        # almost zero; torch.where makes that choice on the device.
        advantage_mean = acting_mean(advantages)
        advantage_standard_deviation = acting_variance(advantages).sqrt()
        advantages = torch.where(
            advantage_standard_deviation > torch.finfo(advantages.dtype).eps,
            (advantages - advantage_mean) / advantage_standard_deviation,
            advantages,
        )
        # One advantage for the actor: each critic's, weighted.
        combined_advantages = advantages @ self.advantage_weights

        # The five measurements, added up over all minibatches on the device.
        measurement_totals = torch.zeros(5, device=self.device)
        minibatch_count = 0
        for _ in range(update_epochs):
            # Every epoch uses every transition once, in a new random order.
            shuffled_indices = torch.randperm(
                batch_size, generator=self.generator, device=self.device
            )
            for minibatch_start in range(0, batch_size, minibatch_size):
                minibatch_indices = shuffled_indices[
                    minibatch_start : minibatch_start + minibatch_size
                ]
                measurement_totals += self._update_minibatch(
                    observations[minibatch_indices],
                    policy_actions[minibatch_indices],
                    old_log_probabilities[minibatch_indices],
                    combined_advantages[minibatch_indices],
                    return_targets[minibatch_indices],
                    None
                    if next_observations is None
                    else next_observations[minibatch_indices],
                    None if noise_scales is None else noise_scales[minibatch_indices],
                    None if action_masks is None else action_masks[minibatch_indices],
                )
                minibatch_count += 1

        # The only conversion to Python numbers: once, after all minibatches.
        summary = torch.cat(
            (measurement_totals / minibatch_count, explained_variances)
        ).tolist()
        return PPOUpdateSummary(
            *summary[:5],
            explained_variance=summary[5],
            explained_variances=tuple(summary[5:]),
        )

    def state_dict(self) -> dict[str, Any]:
        """Everything training changes, for checkpoints.

        Both networks, both optimizers, and the random generator; several
        critics as a list, in order. As with PyTorch's own ``state_dict``, the
        tensors are the live ones: save the result right away, or copy it.
        The settings are not included; restore into a PPO built with the same
        settings and network sizes.
        """
        critic_states = [critic.state_dict() for critic in self.critic_networks]
        return {
            "actor_network": self.actor_network.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_network": critic_states
            if self.several_critics
            else critic_states[0],
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "generator_device_type": self.device.type,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore what ``state_dict`` saved, onto this PPO's device.

        The generator is restored only if it was saved on the same kind of
        device: CPU and CUDA generators draw different sequences, so on another
        device it keeps the sequence this PPO was created with.
        """
        self.actor_network.load_state_dict(state["actor_network"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_states = state["critic_network"]
        if not self.several_critics:
            critic_states = [critic_states]
        for critic, critic_state in zip(
            self.critic_networks, critic_states, strict=True
        ):
            critic.load_state_dict(critic_state)
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if state["generator_device_type"] == self.device.type:
            self.generator.set_state(state["generator"].cpu())

    def _update_minibatch(
        self,
        observations: torch.Tensor,
        policy_actions: torch.Tensor,
        old_log_probabilities: torch.Tensor,
        advantages: torch.Tensor,
        return_targets: torch.Tensor,
        next_observations: torch.Tensor | None = None,
        noise_scales: torch.Tensor | None = None,
        action_masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Take one clipped actor step and one step of every critic on a
        minibatch.

        ``advantages`` is the actor's, ``(B,)``; ``return_targets`` the
        critics', ``(B, K)``, or ``(B,)`` for a single critic. Returns the
        five minibatch measurements, in the order of the first five fields of
        ``PPOUpdateSummary``, as one detached tensor, so that ``update`` can
        add them up on the device without waiting for it.
        """

        # Re-evaluate the collected actions under the current, changing policy,
        # with the noise scales they were drawn with and the components that
        # acted.
        if noise_scales is None and action_masks is None:
            new_log_probabilities, entropy_values = self.policy.evaluate_actions(
                observations, policy_actions
            )
        else:
            new_log_probabilities, entropy_values = self.policy.evaluate_actions(
                observations, policy_actions, noise_scales, action_masks
            )
        # The actor's measures are means over the samples that acted.
        acting = None if action_masks is None else action_masks.any(dim=-1).float()

        def actor_mean(values: torch.Tensor) -> torch.Tensor:
            if acting is None:
                return values.mean()
            return (values * acting).sum() / acting.sum().clamp(min=1.0)

        log_probability_ratios = new_log_probabilities - old_log_probabilities
        probability_ratios = torch.exp(log_probability_ratios)
        clipped_probability_ratios = torch.clamp(
            probability_ratios,
            1 - self.clip_ratio,
            1 + self.clip_ratio,
        )

        unclipped_objectives = probability_ratios * advantages
        clipped_objectives = clipped_probability_ratios * advantages

        # The lower objective removes the reward for an excessively large change.
        surrogate_objectives = torch.minimum(
            unclipped_objectives,
            clipped_objectives,
        )

        mean_entropy = actor_mean(entropy_values)
        approximate_kl = actor_mean(probability_ratios - 1.0 - log_probability_ratios)
        clip_fraction = actor_mean(
            (torch.abs(probability_ratios - 1.0) > self.clip_ratio).float()
        )
        actor_loss = (
            -actor_mean(surrogate_objectives) - self.entropy_coefficient * mean_entropy
        )
        if next_observations is not None and self.temporal_smoothness_coefficient:
            # CAPS's temporal term: the Euclidean distance between the mean
            # actions in a state and in the state after it, kept differentiable
            # where they are equal by a tiny constant under the square root.
            assert isinstance(self.policy, SquashedGaussianPolicy)
            action_change = self.policy.deterministic_action(
                observations
            ) - self.policy.deterministic_action(next_observations)
            distance = torch.sqrt(action_change.square().sum(dim=-1) + 1e-12)
            actor_loss = actor_loss + self.temporal_smoothness_coefficient * (
                distance.mean()
            )

        # Return targets were calculated before optimization and remain fixed.
        # Each critic learns its own; their losses add up, and since they
        # share no parameter, each gets only its own gradient.
        value_errors = return_targets - self._values(observations)
        critic_loss = (
            0.5 * value_errors.square().reshape(len(observations), -1).mean(dim=0).sum()
        )

        # Actor and critics have independent networks; the critics' gradients
        # are clipped each on its own.
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        clip_gradients(self.actor_network.parameters(), self.max_gradient_norm)
        for critic in self.critic_networks:
            clip_gradients(critic.parameters(), self.max_gradient_norm)
        self.actor_optimizer.step()
        self.critic_optimizer.step()

        return torch.stack(
            (actor_loss, critic_loss, mean_entropy, approximate_kl, clip_fraction)
        ).detach()

    def _values(self, observations: torch.Tensor) -> torch.Tensor:
        """The critic's values, ``(B,)``, or every critic's, ``(B, K)``."""
        if not self.several_critics:
            return self.critic_networks[0](observations)
        return torch.stack(
            [critic(observations) for critic in self.critic_networks], dim=-1
        )
