from abc import abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.normal import Normal

import gymnasium as gym
from util.gymnastics import DEVICE


# ---------------------
#     General Algos
# ---------------------
def soft_update_model_params(src: nn.Module, dest: nn.Module, tau=1e-3):
    """Soft updates model parameters (θ_dest = τ * θ_src + (1 - τ) * θ_src)."""
    for dest_param, src_param in zip(dest.parameters(), src.parameters()):
        dest_param.data.copy_(tau * src_param.data + (1.0 - tau) * dest_param.data)


@dataclass
class TrajectorySegment:
    """A trajectory segment for a gym vectorized environment of N bots."""

    obs: torch.Tensor  # (B, N, O_dim)
    actions: torch.Tensor  # (B, N, A_dim)
    logprobs: torch.Tensor  # (B, N)
    values: torch.Tensor  # (B, N) or (B, N, 2) if with intrinsic values.
    rewards: torch.Tensor  # (B, N)
    dones: torch.Tensor  # (B, N)

    next_start_obs: torch.Tensor  # (N, O_dim)

    def next_obs(self) -> torch.Tensor:
        return torch.cat((self.obs[1:], self.next_start_obs.unsqueeze(0)), dim=0)

    def __len__(self):
        return self.obs.shape[0]


def flatten_and_shuffle(*tensors: torch.Tensor) -> tuple[torch.Tensor]:
    """Flattens and shuffles tensors of shape (B, N, ...) where B is the batch size and N the
    number of bots in the vectorized environment (rest the dimension of the single tensor)."""
    batch_size = tensors[0].shape[0]
    n_envs = tensors[0].shape[1]
    shuffled_indices = torch.randperm(batch_size * n_envs)
    return (t.flatten(0, 1)[shuffled_indices] for t in tensors)


class BaseAgent:
    """Naive base agent class for policy gradient algorithms."""

    @torch.no_grad
    def act(self, obs: np.ndarray) -> torch.Tensor:
        """Returns the action to take based on the given observation."""
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        action, _, _ = self.eval(obs)
        return action

    @abstractmethod
    def eval(self, obs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Evaluates an observation: returns action, logprob of the action, and the state value."""
        pass

    @abstractmethod
    def learn(self, segment: TrajectorySegment) -> dict:
        """Learns using the input trajectory segment."""
        pass


def collect_trajectory_segment(
    env: gym.vector.VectorEnv,
    agent: BaseAgent,
    start_obs: torch.Tensor,
    rollout_size=128,
    with_intrinsic_values=False,
) -> TrajectorySegment:
    """Collects a TrajectorySegment on a gym vectorized environments using a BaseAgent.

    The value tensor is shaped differently depending on whether intrinsic rewards are used.
    """
    num_bots = env.observation_space.shape[0]
    obs_shape = env.observation_space.shape[1:]
    action_shape = env.action_space.shape[1:]
    batch_dim = (rollout_size, num_bots)

    s_obs = torch.zeros(batch_dim + obs_shape).to(DEVICE)
    s_actions = torch.zeros(batch_dim + action_shape).to(DEVICE)
    s_logprobs = torch.zeros(batch_dim).to(DEVICE)
    s_values = torch.zeros(batch_dim + (2,) if with_intrinsic_values else ()).to(DEVICE)
    s_rewards = torch.zeros(batch_dim).to(DEVICE)
    s_dones = torch.zeros(batch_dim).to(DEVICE)

    obs = start_obs
    for step in range(rollout_size):
        with torch.no_grad():
            action, logprob, values = agent.eval(obs)
        next_obs, reward, term, trunc, _ = env.step(action.cpu().numpy())

        s_obs[step] = obs
        s_actions[step] = action
        s_logprobs[step] = logprob
        s_values[step] = values.to(DEVICE)
        s_rewards[step] = torch.Tensor(reward).to(DEVICE)
        s_dones[step] = torch.Tensor(np.logical_or(term, trunc)).to(DEVICE)

        obs = torch.Tensor(next_obs).to(DEVICE)

    return TrajectorySegment(
        s_obs, s_actions, s_logprobs, s_values, s_rewards, s_dones, next_start_obs=obs
    )


def compute_advantages_and_returns(
    rewards, dones, values, next_value, gamma=0.99, gae_enabled=True, gae_lambda=0.98
):
    """Computes advantages and returns using GAE, or a simpler 'standard' computation."""
    segment_len = len(rewards)
    if gae_enabled:
        advantages = torch.zeros_like(rewards).to(DEVICE)
        last_gae_lambda = 0
        for t in reversed(range(segment_len)):
            next_non_terminal = 1.0 - dones[t]
            td_error = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = td_error + gamma * gae_lambda * last_gae_lambda * next_non_terminal
            next_value = values[t]
            last_gae_lambda = advantages[t] * next_non_terminal
        returns = advantages + values
    else:
        returns = torch.zeros_like(rewards).to(DEVICE)
        running_return = next_value
        for t in reversed(range(segment_len)):
            next_non_terminal = 1.0 - dones[t]
            running_return = rewards[t] + gamma * running_return * next_non_terminal
            returns[t] = running_return
        advantages = returns - values
    return advantages, returns


# Just used to detect a default argument set or not below in ReplayBuffer.
_sentinel = object()


class Experience(NamedTuple):
    """A single step / experience of an agent stored in the replay buffer."""

    state: np.array
    action: np.array
    reward: float
    next_state: np.array
    done: bool


class ReplayBuffer:
    """Simple replay buffer for off-policy deep reinforcement learning algorithms.

    IMPORTANT: This ReplayBuffer is specifically tuned for algorithms in these lectures. For
    example, the action space is continuous (float) instead of discrete. If you want to adapt
    those algorithms to different environments, you will need to update this code accordingly.
    """

    def __init__(self, buffer_size: int = 1e5, sample_size: int = 128):
        """Initializes the buffer with an internal deque of size `buffer_size`."""
        self.memory = deque(maxlen=int(buffer_size))
        self.sample_size = sample_size

    def add(self, state, action, reward, next_state, done):
        """Stores a single step / experience of an agent."""
        stored_action = np.atleast_1d(action)
        e = Experience(state, stored_action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, sample_size=_sentinel):
        """Randomly selects `batch_size` items from the buffer, stacks them, and returns them."""
        actual_sample_size = self.sample_size if sample_size is _sentinel else sample_size
        all_indices = np.arange(len(self.memory))
        selection = np.random.choice(all_indices, size=actual_sample_size)
        experiences = [e for i in selection if (e := self.memory[i]) is not None]
        return ReplayBuffer.unpack(experiences)

    def sample_all(self):
        """Returns all samples available in the buffer."""
        return self.sample(sample_size=len(self))

    @staticmethod
    def unpack(experiences: list[Experience]):
        """Given the selection of `experiences`, returns them as a tuple of stacked values.

        This is convenient for the usage in the various learning algorithms so that they don't have
        to do it themselves.
        """
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.stack(states)).float().to(DEVICE)
        actions = torch.from_numpy(np.stack(actions)).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(DEVICE)
        next_states = torch.from_numpy(np.stack(next_states)).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack(dones, dtype=np.uint8)).float().to(DEVICE)
        return (states, actions, rewards, next_states, dones)

    def clear(self):
        """Clears the buffer."""
        self.memory.clear()

    def __len__(self):
        return len(self.memory)


# ---------------------
#         SAC
# ---------------------


class CriticNetworkSAC(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_size=256):
        super(CriticNetworkSAC, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class ActorNetworkSAC(nn.Module):
    def __init__(self, state_size, action_size, action_scale=1.0, action_bias=0.0, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mean = nn.Linear(hidden_size, action_size)
        self.fc_logstd = nn.Linear(hidden_size, action_size)
        self.action_scale = action_scale
        self.action_bias = action_bias

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std_min, log_std_max = (-20, 2)
        adjusted_log_std = torch.clamp(log_std, min=log_std_min, max=log_std_max)
        return mean, adjusted_log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        sample = normal.rsample()
        output = torch.tanh(sample)
        action = output * self.action_scale + self.action_bias
        log_prob = normal.log_prob(sample)
        log_prob -= torch.log(self.action_scale * (1 - output.pow(2)) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob


class AgentSAC:
    def __init__(
        self,
        state_size,
        action_size,
        start_mem_size=2e3,
        gamma=0.995,
        tau=0.005,
        lr_actor=3e-4,
        lr_critic=3e-4,
        max_norm=0.5,
        policy_freq=2,
        target_update_freq=2,
        buffer_size=1e6,
        sample_size=256,
        alpha=0.2,
        auto_entropy_tuning=True,
        action_scale=1.0,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.start_mem_size = start_mem_size
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.max_norm = max_norm
        self.policy_freq = policy_freq
        self.target_update_freq = target_update_freq
        self.sample_size = sample_size
        self.auto_entropy_tuning = auto_entropy_tuning
        self.t_step = 0

        self.actor = ActorNetworkSAC(state_size, action_size, action_scale).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.twin_critic_1 = CriticNetworkSAC(state_size, action_size).to(DEVICE)
        self.twin_critic_target_1 = CriticNetworkSAC(state_size, action_size).to(DEVICE)
        self.twin_critic_target_1.load_state_dict(self.twin_critic_1.state_dict())
        self.twin_critic_target_1.eval()

        self.twin_critic_2 = CriticNetworkSAC(state_size, action_size).to(DEVICE)
        self.twin_critic_target_2 = CriticNetworkSAC(state_size, action_size).to(DEVICE)
        self.twin_critic_target_2.load_state_dict(self.twin_critic_2.state_dict())
        self.twin_critic_target_2.eval()

        self.critic_optimizer = optim.Adam(
            list(self.twin_critic_1.parameters()) + list(self.twin_critic_2.parameters()),
            lr=lr_critic,
        )

        # Do automatic entropy tuning only with Gaussian policy...
        if self.auto_entropy_tuning == True:
            self.target_entropy = -torch.prod(torch.Tensor((action_size,)).to(DEVICE)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr_critic)
            self.alpha = self.log_alpha.exp().item()

        self.memory = ReplayBuffer(buffer_size, sample_size)
        self.learning_started = False

    @torch.no_grad
    def act(self, states):
        states = torch.from_numpy(states).float().to(DEVICE)
        actions, _ = self.actor.get_action(states)
        return actions.cpu().numpy()

    def add_experience(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)  # For vectorized training.

    def step(self, state, action, reward, next_state, done):
        self.t_step += 1
        self.memory.add(state, action, reward, next_state, done)
        if len(self.memory) > self.start_mem_size:
            self.learning_started = True
            experiences = self.memory.sample()
            self.learn(experiences, self.t_step)

    def learn(self, experiences, step):
        states, actions, rewards, next_states, dones = experiences

        with torch.no_grad():
            actions_next, log_pi_next_st = self.actor.get_action(next_states)
            entropy_term = self.alpha * log_pi_next_st

            Q_targets_next_1 = self.twin_critic_target_1(next_states, actions_next)
            Q_targets_next_2 = self.twin_critic_target_2(next_states, actions_next)
            Q_targets_next = torch.min(Q_targets_next_1, Q_targets_next_2) - entropy_term
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        Q_expected_1 = self.twin_critic_1(states, actions)
        Q_expected_2 = self.twin_critic_2(states, actions)
        critic_loss = F.mse_loss(Q_expected_1, Q_targets) + F.mse_loss(Q_expected_2, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_params = list(self.twin_critic_1.parameters()) + list(
            self.twin_critic_2.parameters()
        )
        nn.utils.clip_grad_norm_(critic_params, max_norm=self.max_norm)
        self.critic_optimizer.step()

        if step % self.policy_freq == 0:
            action, log_pi = self.actor.get_action(states)
            entropy_term = self.alpha * log_pi

            Q_values_1 = self.twin_critic_1(states, action)
            Q_values_2 = self.twin_critic_2(states, action)
            Q_targets = torch.min(Q_values_1, Q_values_2)
            actor_loss = (entropy_term - Q_targets).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.max_norm)
            self.actor_optimizer.step()

            if self.auto_entropy_tuning:
                negative_entropy_error = (log_pi + self.target_entropy).detach()
                log_alpha_loss = -(self.log_alpha.exp() * negative_entropy_error).mean()

                self.alpha_optim.zero_grad()
                log_alpha_loss.backward()
                self.alpha_optim.step()

                # Clamp alpha in the interval [0.01, 1.0], via log_alpha for the optimizer.
                with torch.no_grad():
                    self.log_alpha.clamp_(min=-4.5, max=0.0)
                self.alpha = self.log_alpha.exp().detach().item()

        if step % self.target_update_freq == 0:
            soft_update_model_params(self.twin_critic_1, self.twin_critic_target_1, tau=self.tau)
            soft_update_model_params(self.twin_critic_2, self.twin_critic_target_2, tau=self.tau)

    def save(self, filename: str):
        loaded_params = {
            "actor": self.actor.state_dict(),
            "twin_critic_1": self.twin_critic_1.state_dict(),
            "twin_critic_target_1": self.twin_critic_target_1.state_dict(),
            "twin_critic_2": self.twin_critic_2.state_dict(),
            "twin_critic_target_2": self.twin_critic_target_2.state_dict(),
            "auto_entropy_tuning": self.auto_entropy_tuning,
        }
        if self.auto_entropy_tuning:
            loaded_params["target_entropy"] = self.target_entropy
            loaded_params["log_alpha"] = self.log_alpha
        torch.save(loaded_params, filename)

    def load(self, filename: str):
        loaded_params = torch.load(filename, map_location=DEVICE)
        self.actor.load_state_dict(loaded_params["actor"])
        self.twin_critic_1.load_state_dict(loaded_params["twin_critic_1"])
        self.twin_critic_target_1.load_state_dict(loaded_params["twin_critic_target_1"])
        self.twin_critic_2.load_state_dict(loaded_params["twin_critic_2"])
        self.twin_critic_target_2.load_state_dict(loaded_params["twin_critic_target_2"])
        self.auto_entropy_tuning = loaded_params["auto_entropy_tuning"]
        if self.auto_entropy_tuning:
            self.target_entropy = loaded_params["target_entropy"]
            self.log_alpha = loaded_params["log_alpha"]
            self.alpha = self.log_alpha.exp().item()


class SAC:
    def __init__(
        self,
        env: gym.Env | gym.vector.VectorEnv,
        agent: AgentSAC,
        solved_score=215.0,
        max_episodes=23_300,
        max_t=1_000,
    ):
        self.env = env
        self.agent = agent
        self.solved_score = solved_score
        self.max_episodes = max_episodes
        self.max_t = max_t

    def train(self):
        if isinstance(self.env, gym.vector.VectorEnv):
            return self._train_vectorized()
        else:
            return self._train_single()

    def _train_single(self):
        scores = []
        for i_episode in range(1, self.max_episodes + 1):
            state, _ = self.env.reset()
            score = 0
            for _ in range(self.max_t - 1):
                action = (
                    self.agent.act(state)
                    if self.agent.learning_started
                    else self.env.action_space.sample()
                )
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores.append(score)
            avg_score = np.mean(scores[-100:])
            print(
                f"\rEpisode {i_episode}\tAverage Score: {avg_score:.2f}",
                end="\n" if i_episode % 25 == 0 else "",
            )
            if avg_score >= self.solved_score:
                print(f"\rEpisode {i_episode} solved environment!\tAverage Score: {avg_score:.2f}")
                break

        return scores

    def _train_vectorized(self):
        scores = []
        scores_window = deque(maxlen=100)
        num_envs = self.env.num_envs
        total_timesteps = self.max_episodes * self.max_t
        episode_scores = np.zeros(num_envs, dtype=np.float32)
        states, _ = self.env.reset()

        print("Start collecting data...", end="")
        for global_step in range(total_timesteps // num_envs):
            actions = (
                self.agent.act(states)
                if self.agent.learning_started
                else self.env.action_space.sample()
            )
            next_states, rewards, terminateds, truncateds, _ = self.env.step(actions)
            dones = terminateds | truncateds
            for i in range(num_envs):
                self.agent.add_experience(
                    states[i], actions[i], rewards[i], next_states[i], dones[i]
                )

            self.agent.t_step += num_envs
            if len(self.agent.memory) > self.agent.start_mem_size:
                if not self.agent.learning_started:
                    self.agent.learning_started = True
                    print("\rLearning started...          ", end="")
                experiences = self.agent.memory.sample()
                self.agent.learn(experiences, global_step)

            states = next_states
            episode_scores += rewards
            for i, done in enumerate(dones):
                if done:
                    finished_score = episode_scores[i]
                    scores.append(finished_score)
                    scores_window.append(finished_score)
                    episode_scores[i] = 0.0
                    avg_score = np.mean(scores_window)
                    num_episodes_done = len(scores)
                    print(
                        f"\rEpisode {num_episodes_done}\tAverage Score: {avg_score:.2f}",
                        end="\n" if num_episodes_done % 25 == 0 else "",
                    )
                    if avg_score >= self.solved_score:
                        print(
                            f"\nEnvironment solved in {num_episodes_done} episodes!"
                            + f"\tAverage Score: {avg_score:.2f}"
                        )
                        return scores
        return scores
