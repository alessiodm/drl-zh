import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import tempfile
import time
import torch
import moviepy
import warnings

from IPython.display import Image
from gymnasium.wrappers import RecordVideo

from pettingzoo.utils.env import AgentID, AECEnv
from pettingzoo.classic import tictactoe_v3, connect_four_v3
import pygame

from typing import Any, Callable


# The device to use for PyTorch. Just defined here for convenience.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The default random seed used in the lectures.
DEFAULT_RANDOM_SEED = 10


def init_random(env: gym.Env = None, seed=DEFAULT_RANDOM_SEED):
    """Initializes all the random generators used by frameworks to a default value.

    If `env` is specified, it sets also the env random generator and return the env itself. That is
    just a convenient setup for the notebooks, even if arguably not the best structured code setup.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if env is not None:
        env.reset(seed=seed)
        try:  # For pettingzoo compatibility, ignore any failures
            env.action_space.seed(seed)
        except:
            pass
    return env


def epsilon_gen(eps_start=1.0, eps_decay=0.99999, eps_min=0.05):
    """Generator function for Ɛ and its decay (e.g., exploration via Ɛ-greedy policy)."""
    eps = eps_start
    while True:
        yield eps
        eps = max(eps * eps_decay, eps_min)


def show_gym_video_recording(name_prefix: str = "rl-video", folder: str = None):
    """Shows the recorded video of episode 0 of a gym environment."""
    if folder == None:
        folder = tempfile.tempdir
    video_file = os.path.join(folder, f"{name_prefix}-episode-0.mp4")
    video_clip = moviepy.VideoFileClip(video_file)
    gif_file = os.path.join(tempfile.tempdir, "rl-video.gif")
    video_clip.write_gif(gif_file, fps=15)
    video_clip.close()
    return Image(open(gif_file, "rb").read())


def plot_scores(scores, rolling_window=25, ylabel="Score", xlabel="Episode #"):
    """Plots training scores and their running average."""
    avgs = pd.Series(scores).rolling(rolling_window).mean()
    x = np.arange(len(scores))
    plt.figure("Scores")
    plt.plot(x, scores, label="Scores")
    plt.plot(x, avgs, "r", label="Running average")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return plt.show()


def gym_simulation(
    env: str | gym.Env,
    agent=None,
    max_t=1_000,
    env_kwargs={},
    seed=DEFAULT_RANDOM_SEED,
    wrappers=None,
):
    """Runs a simulation of an agent in an initialized gym environment and plays the video."""
    if isinstance(env, str):
        sim_env = gym.make(env, render_mode="rgb_array_list", **env_kwargs)
    else:
        assert (
            env.render_mode == "rgb_array_list"
        ), "The environment render_mode must be 'rgb_array_list'"
        if len(env_kwargs) > 0:
            warnings.warn("Custom env_kwargs are ignored b/c a fully-constructed env was passed.")
        sim_env = env
    if wrappers is not None:
        for wrapper in wrappers:
            sim_env = wrapper(sim_env)
    sim_env = init_random(sim_env, seed)
    sim_env = RecordVideo(sim_env, tempfile.tempdir, lambda i: i == 0)
    state, _ = sim_env.reset()
    for _ in range(max_t):
        action = agent.act(state) if agent is not None else sim_env.action_space.sample()
        next_state, _, terminated, truncated, _ = sim_env.step(action)
        if terminated or truncated:
            break
        state = next_state
    sim_env.close()
    return show_gym_video_recording()


def pettingzoo_simulation(
    pettingzoo_env, agent=None, max_t=150, seed=DEFAULT_RANDOM_SEED, wrappers=None, **env_kwargs
):
    """Runs a simulation of an agent in a pettingzoo environment and plays the video."""
    sim_env = pettingzoo_env.env(render_mode="rgb_array", **env_kwargs)
    if wrappers is not None:
        for wrapper in wrappers:
            sim_env = wrapper(sim_env)
    sim_env = init_random(sim_env, seed)
    t = 0
    frames = []
    global_t = max_t * len(sim_env.agents)
    for agent_id in sim_env.agent_iter():
        obs, _, termination, truncation, _ = sim_env.last()
        if termination or truncation:
            action = None
        elif agent is not None:
            action = agent.eval_act(agent_id, obs)
        else:
            action = sim_env.action_space(agent_id).sample()
        sim_env.step(action)
        frame = sim_env.render()
        frames.append(frame)
        t += 1
        if t > global_t:
            break
    sim_env.close()
    video_file = os.path.join(tempfile.tempdir, "rl-video-episode-0.mp4")
    imageio.mimsave(video_file, frames, fps=15)
    return show_gym_video_recording()


# Simple types for PettingZoo play methods.
PzooState = Any
PzooAction = int
PolicyCallback = Callable[[PzooState], PzooAction]


def _pettingzoo_game_loop(
    env: AECEnv,
    player: AgentID,
    get_action_fn: Callable[[int, int], int],
    policy_callback: PolicyCallback,
):
    try:
        if player not in env.possible_agents:
            print(f"*** Invalid player. Available players: {env.possible_agents} ***")
            return
        for agent in env.agent_iter():
            ob, _, term, trunc, _ = env.last()
            ob["player"] = agent
            if term or trunc:
                action = None
                break
            else:
                mask = ob["action_mask"]
                if agent == player:
                    waiting = True
                    while waiting:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                waiting = False
                            elif event.type == pygame.MOUSEBUTTONDOWN:
                                mouse_x, mouse_y = pygame.mouse.get_pos()
                                action = get_action_fn(mouse_x, mouse_y)
                                if not mask[action]:
                                    continue
                                waiting = False
                else:
                    if policy_callback is None:
                        mask = ob["action_mask"]
                        action = env.action_space(agent).sample(mask)
                    else:
                        # this is where you would insert your policy
                        action = policy_callback(ob)
                env.step(action)
    finally:
        time.sleep(2)
        env.close()
        pygame.quit()


def pettingzoo_tictactoe_play(player: AgentID, policy_callback: PolicyCallback = None, seed=None):
    """Play an interactive play of tic-tac-toe, against a trained policy (if specified)."""
    env: AECEnv = tictactoe_v3.env(render_mode="human")
    env.reset(seed=seed)

    # https://pettingzoo.farama.org/environments/classic/tictactoe/#action-space
    def get_action(x, y):
        width, height = pygame.display.get_surface().get_size()
        if x < width / 3:  # A column
            if y < height / 3:  # 3rd row
                return 0
            elif y > height / 3 * 2:  # 1st row
                return 2
            else:  # 2nd row
                return 1
        elif x > width / 3 * 2:  # C column
            if y < height / 3:  # 3rd row
                return 6
            elif y > height / 3 * 2:  # 1st row
                return 8
            else:  # 2nd row
                return 7
        else:  # B column
            if y < height / 3:  # 3rd row
                return 3
            elif y > height / 3 * 2:  # 1st row
                return 5
            else:  # 2nd row
                return 4

    _pettingzoo_game_loop(env, player, get_action, policy_callback)


def pettingzoo_connect4_play(player: AgentID, policy_callback: PolicyCallback = None, seed=None):
    """Play an interactive play of connect-4, against a trained policy (if specified)."""
    env: AECEnv = connect_four_v3.env(render_mode="human")
    env.reset(seed=seed)

    # https://pettingzoo.farama.org/environments/classic/connect_four/#action-space
    def get_action(x, y):
        width, _ = pygame.display.get_surface().get_size()
        col_size = width / 7
        return int(x / col_size)

    _pettingzoo_game_loop(env, player, get_action, policy_callback)


def check_grid_run(board: np.array, player: int, n: int = 3) -> bool:
    """Check n-in-a-row patterns in a board for 'player'.
    Method used to check tic-tac-toe and connect-four winning scenarios."""
    rows, cols = board.shape

    # Define directions for checking: (row_delta, col_delta)
    directions = [
        (0, 1),  # Horizontal
        (1, 0),  # Vertical
        (1, 1),  # Diagonal (down-right)
        (1, -1),  # Diagonal (down-left)
    ]

    # Iterate through every cell on the board
    for r in range(rows):
        for c in range(cols):
            # If the current cell doesn't belong to the player, no line can start here
            if board[r, c] != player:
                continue

            # Check in all four directions from this cell
            for dr, dc in directions:
                count = 0
                for i in range(n):
                    curr_r, curr_c = r + i * dr, c + i * dc

                    # Check boundaries and if the piece belongs to the player
                    if (
                        0 <= curr_r < rows
                        and 0 <= curr_c < cols
                        and board[curr_r, curr_c] == player
                    ):
                        count += 1
                    else:
                        # Line broken or out of bounds
                        break

                if count == n:
                    return True  # Win found!

    return False  # No win found after checking all possibilities


class RLHF:
    """Convenient class of constants for the RLHF notebook."""

    PROMPTS = [
        # Generic starters
        "Once upon a time",
        "The story starts",
        "There once was",
        "It all started when",
        "Nobody knew why the",
        "The smallest creature decided to",
        "In a cozy little home lived",
        "The secret path led to",
        "One day, looking up at the sky,",
        "Down by the old fence,",
        "The smallest door in the house led to",
        # Forest / woodland
        "Deep in the shady forest,",
        "A rustle in the leaves meant",
        "The tallest tree in the wood was home to",
        "Following the mossy trail,",
        # Ocean / water
        "Far below the ocean waves lived",
        "A shiny fish swam quickly past",
        "On the sandy seabed,",
        "The tide pool was full of",
        "Floating gently on the water",
        # Savanna / grassland / desert
        "Across the savanna,",
        "Under the hot sun, searching for",
        "The lion stretched out near",
        "Hiding in the tall grass",
    ]

    # List of words associated with animals (lowercase)
    ANIMAL_WORDS = [
        "fox",
        "owl",
        "mule",
        "butterfly",
        "bug",
        "reindeer",
        "animal",
        "bird",
        "bee",
        "rabbit",
        "bunny",
        "lion",
        "mouse",
        "cat",
        "dog",
        "bear",
        "horse",
        "squirrel",
        "mice",
        "monkey",
        "tiger",
        "wolf",
        "frog",
        "rhinoceros",
        "giraffe",
        "elephant",
        "fish",
        "shark",
        "turtle",
        "crab",
        "dolphin",
        "coral",
        "snail",
        "whale",
        "duck",
        "octopus",
        "puppy",
        "penguin",
        "cheetah",
        "tree",
        "cub",
        "kitten",
        "chick",
        "lizard",
        "spider",
        "caterpillar",
        "worm",
    ]

    # List of words associated with humans/non-animals (lowercase)
    HUMAN_WORDS = [
        "girl",
        "boy",
        "man",
        "men",
        "woman",
        "women",
        "child",
        "children",
        "dad",
        "daddy",
        "mom",
        "mommy",
        "lady",
        "uncle",
        "aunt",
        "santa",
        "explorer",
        "she",
        "he",
        "his",
        "her",
        "person",
        "people",
        "king",
        "queen",
        "prince",
        "princess",
        "teacher",
        "doctor",
        "friend",
        "family",
        "Benny",
        "Bella",
        "Daisy",
        "Sarah",
        "Susie",
        "Sally",
        "Sophie",
        "Lily",
        "Lucy",
        "Mary",
        "Sue",
        "Mia",
        "Lola",
        "Molly",
        "Jane",
        "Jack",
        "Timmy",
        "Max",
        "Tim",
        "Sam",
        "Billy",
    ]
