import os
import random
import tempfile

from enum import Enum, StrEnum
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, TypeAlias

import numpy as np
from numpy.testing import assert_almost_equal

from matplotlib import animation
import matplotlib.pyplot as plt
from matplotlib import colors
from IPython.display import Image

class Cell(StrEnum):
    START  = 'S'
    TARGET = 'T'
    EMPTY  = 'E'
    WALL   = 'W'
    BOMB   = 'B'
    GLORY  = 'G'
    NUKE   = 'N'

class Grid:
    def __init__(self, spec: list[str]):
        self.height = len(spec)
        self.width  = len(spec[0])
        self.cells  = [[Cell(col) for col in [*row]] for row in spec]

    def __str__(self) -> str:
        return '\n'.join([' '.join([col.value for col in row]) for row in self.cells]).strip()

    def __getitem__(self, key) -> Cell:
        x, y = key
        return self.cells[self.height - y - 1][x]

class Action(Enum):
    UP    = 1
    RIGHT = 2
    DOWN  = 3
    LEFT  = 4

@dataclass(frozen=True)
class State:
    x: int = 0
    y: int = 0

    def pos(self) -> tuple[int]:
        return (self.x, self.y)

class GridMDP:
    def __init__(self, grid: Grid, start = State(), gamma = 1.0):
        self.grid = grid
        self.start = start
        self.gamma = gamma
        self.all_actions = list(Action)
        # Arguably, we could filter out the non-reachable states here - but we keep things simple.
        self.all_states = [State(x, y) for x in range(0, self.grid.width) 
                           for y in range(0, self.grid.height)]

    def is_terminal(self, state: State) -> bool:
        cell = self.grid[state.pos()]
        return cell in [Cell.TARGET, Cell.GLORY, Cell.BOMB, Cell.NUKE]

    def is_reachable(self, n: State) -> bool:
        return n.x >= 0 and n.x < self.grid.width \
                and n.y >= 0 and n.y < self.grid.height \
                and self.grid[n.pos()] != Cell.WALL

    def reward(self, state: State, action: Action, next_state: State) -> float:
        match self.grid[next_state.pos()]:
            case Cell.TARGET:
                return 1.0 * self.gamma
            case Cell.GLORY:
                return 10.0 * self.gamma
            case Cell.BOMB:
                return -1.0 * self.gamma
            case Cell.NUKE:
                return -10.0 * self.gamma
            case _:
                return 0.0

    def transition(self, state: State, action: Action, noise = 0.0) -> dict[State, float]:
        if not self.is_reachable(state) or self.is_terminal(state):
            return {}

        def landing(candidate: State) -> State:
            return candidate if self.is_reachable(candidate) else state

        action_landings = {
            Action.UP:    landing(State(state.x, state.y + 1)),
            Action.RIGHT: landing(State(state.x + 1, state.y)),
            Action.DOWN:  landing(State(state.x, state.y - 1)),
            Action.LEFT:  landing(State(state.x - 1, state.y)),
        }
        mistaken_actions = [Action.UP, Action.DOWN] if action in [Action.LEFT, Action.RIGHT] \
                                                    else [Action.LEFT, Action.RIGHT]

        next_state = action_landings[action]
        mistakes = [action_landings[m] for m in mistaken_actions]

        probs = defaultdict(lambda: 0.0)
        probs[next_state] = 1.0 - noise
        for m in mistakes:
            probs[m] += noise / len(mistakes)

        assert sum(probs.values()) <= 1.0
        assert_almost_equal(sum(probs.values()), 1.0)

        return probs

class GridEnv:
    def __init__(self, mdp: GridMDP):
        self.mdp = mdp
        self.state = State()
        self.terminated = False

    def reset(self) -> State:
        self.state = State()
        self.terminated = False
        return self.state

    def step(self, action: Action) -> tuple[State, float, bool]:
        if self.terminated:
            raise Exception('Environment episode completed, please call reset.')
        state_probs = [(s, p) for s, p in self.mdp.transition(self.state, action).items()]
        probs = [x[1] for x in state_probs]
        next_state_idx = np.random.choice(len(probs), p=probs)
        next_state = state_probs[next_state_idx][0]
        reward = self.mdp.reward(self.state, action, next_state)
        done = self.mdp.is_terminal(next_state)
        step = (next_state, reward, done)
        self.state = next_state
        self.terminated = done
        return step

class QTable:
    def __init__(self, states: list[State], actions: list[Action]):
        self.states = states
        self.actions = actions
        self.nA = len(actions)
        # Why a dict of dict vs. a single dict indexed by tuple (State, Action)?
        # Because we want to lookup all action values for a state, and that's more convenient.
        self.table: dict[State, dict[Action, float]] = \
            { s : { a : 0.0 for a in actions } for s in states }

    def __getitem__(self, key: tuple[State, Action]) -> float:
        state, action = key
        return self.table[state][action]

    def __setitem__(self, key: tuple[State, Action], value: float):
        state, action = key
        self.table[state][action] = value

    def value(self, state: State) -> float:
        return max(self.table[state].values())

    def best_action(self, state: State) -> Action:
        best_action = None
        best_v = float('-inf')
        actions = list(self.table[state].keys())
        random.shuffle(actions) # Random shuffle in case actions have same value...
        for a in actions:
            v = self[state, a]
            if v > best_v:
                best_action = a
                best_v = v
        return best_action

Policy: TypeAlias = Callable[[State], Action]

DEFAULT_GRID = Grid([
    'EEET',
    'EWEB',
    'SEEE',
])
GRID_WORLD_MDP = GridMDP(DEFAULT_GRID, gamma=0.9)
RANDOM_POLICY = lambda _: np.random.choice(list(Action))

# TODO: possibly rename this class. Also action == None means it is terminal...
# Unclear if this is actually a good-enough implementation for the sake of the examples.
@dataclass
class Step:
    state: State
    action: Action
    reward: float

# Keep this outside MDP so we can simulate the user MDP class.
def simulate_mdp(mdp, policy, max_iterations=20) -> list[Step]:
    steps = []
    state = mdp.start
    current_iteration = 0
    while True:
        current_iteration += 1
        action = policy(state)
        state_probs = [(s, p) for s, p in mdp.transition(state, action).items()]
        probs = [x[1] for x in state_probs]
        next_state = np.random.choice(len(probs), p=probs)
        next_state = state_probs[next_state][0]
        reward = mdp.reward(state, action, next_state)
        steps.append(Step(state, action, reward))
        state = next_state
        if current_iteration == max_iterations or mdp.is_terminal(state):
            steps.append(Step(next_state, None, 0.0))
            break
    return steps

#--------------------------
# Plotting utilities below
#--------------------------

plt.rcParams["animation.html"] = "jshtml"
plt.rcParams['figure.dpi'] = 100  
# plt.ion()

CMAP = colors.ListedColormap(['lavender', 'palegreen', 'white', 'gray', 'lightpink', 'limegreen', 'red'])
CMAP_BOUNDS = [0, 1, 2, 3, 4, 5, 6, 7]
NORM = colors.BoundaryNorm(CMAP_BOUNDS, CMAP.N)
CELL_VALUES = { 'S': 0, 'T': 1, 'E': 2, 'W': 3, 'B': 4, 'G': 5, 'N': 6 }

class PlotData:
    def __init__(self, grid, ax, agent_marker, state_value_texts):
        self.grid = grid
        self.ax = ax
        self.agent_marker = agent_marker
        self.state_value_texts = state_value_texts

def plot_grid(grid, qtable = None, agent_pos: tuple[int, int] = None):
    plt.rcParams["figure.figsize"] = [3.5, 2.5]
    # plt.rcParams["figure.autolayout"] = True
    colors_matrix = np.array([[CELL_VALUES[marker] for marker in row] for row in grid.cells])
    fig, ax = plt.subplots()
    ax.imshow(colors_matrix, cmap=CMAP, norm=NORM)
    state_value_texts = [[None for _ in range(grid.height)] for _ in range(grid.width)] 
    if qtable is not None:
        for x in range(grid.width):
            for y in range(grid.height):
                # To make sure there is no conflict with the notebook types.
                state = next(filter(lambda s: s.x == x and s.y == y, qtable.table.keys()))
                txt = ax.text(x, grid.height - y - 1, f'{qtable.value(state):.2f}',
                              ha='center', va='center', fontsize=12, color='black')
                state_value_texts[x][y] = txt
    plt.scatter(0, 2, s=1000, c='white', marker='*')
    agent_marker = None if agent_pos is None else \
        plt.scatter(agent_pos[0], grid.height - agent_pos[1] -1, s=1000, 
                    c='blue', marker='*', animated=True)
    ax.axes.get_xaxis().set_ticks(np.arange(grid.width) + 0.5)
    ax.axes.get_yaxis().set_ticks(np.arange(grid.height) + 0.5)
    ax.axes.get_xaxis().set_ticklabels([])
    ax.axes.get_yaxis().set_ticklabels([])
    ax.grid()
    # plt.show()
    return fig, PlotData(grid, ax, agent_marker, state_value_texts)

def run_simulation(mdp, policy, max_iterations=20, frames_per_state=10):
    steps = simulate_mdp(mdp, policy, max_iterations)

    # Compute cumulative returns for convenience
    returns = [0.0]
    for i, s in enumerate(steps):
        returns.append(returns[i] + s.reward)

    fig, data = plot_grid(mdp.grid, agent_pos=mdp.start.pos())
    plt.close(fig)

    def animate(frame):
        state_frame = frame // frames_per_state

        state = steps[state_frame].state
        previous_state = None if state_frame == 0 else steps[state_frame - 1].state
        stayed_in_place = (state == previous_state)

        new_x, new_y = (state.x, mdp.grid.height - state.y - 1)

        if frame < frames_per_state:
            data.agent_marker.set_offsets((new_x, new_y))
            return

        old_x, old_y = data.agent_marker.get_offsets()[0]
        delta_x = float(new_x - old_x) / 4.
        delta_y = float(new_y - old_y) / 4.

        anim_frame = frame % frames_per_state

        if anim_frame < 4:
            if stayed_in_place:
                cur_x = new_x + 0.1 * ((-1) ** (anim_frame % 2))
                cur_y = new_y
            else:
                cur_x = old_x + delta_x * anim_frame
                cur_y = old_y + delta_y * anim_frame
        else:
            cur_x = new_x
            cur_y = new_y

        data.ax.set_title(f'Return: {returns[state_frame]}')
        data.agent_marker.set_offsets((cur_x, cur_y))

    anim = animation.FuncAnimation(fig, animate, frames=len(steps) * frames_per_state, interval=50)

    # Temporary workaround to avoid:
    #   UserWarning: Animation was deleted without rendering anything.
    f = os.path.join(tempfile.tempdir, 'rl_animation.gif')
    writergif = animation.PillowWriter(fps=20)
    anim.save(f, writer=writergif)
    plt.close()
    return Image(open(f, 'rb').read())
