import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class GridSoccerEnv(gym.Env):
    def __init__(self, grid_size=(8, 8), max_steps=100):
        super(GridSoccerEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.action_space = spaces.MultiDiscrete([5, 5])
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(low=0, high=max(grid_size), shape=(5,), dtype=np.int32),
                spaces.Box(low=0, high=max(grid_size), shape=(5,), dtype=np.int32),
            )
        )
        self.reset()

    def reset(self):
        self.steps = 0
        self.player1_pos = self._random_position(half="left")
        self.player2_pos = self._random_position(half="right")
        self.ball_control = random.randint(1, 2)
        return self._get_obs()

    def _random_position(self, half):
        x = (
            random.randint(0, self.grid_size[0] // 2 - 1)
            if half == "left"
            else random.randint(self.grid_size[0] // 2, self.grid_size[0] - 1)
        )
        y = random.randint(0, self.grid_size[1] - 1)
        return (x, y)

    def _get_obs(self):
        player1_obs = np.array(
            [*self.player1_pos, *self.player2_pos, 1 if self.ball_control == 1 else 0],
            dtype=np.int32,
        )
        player2_obs = np.array(
            [
                self.grid_size[0] - 1 - self.player2_pos[0],
                self.player2_pos[1],
                self.grid_size[0] - 1 - self.player1_pos[0],
                self.player1_pos[1],
                1 if self.ball_control == 2 else 0,
            ],
            dtype=np.int32,
        )
        return player1_obs, player2_obs

    def step(self, action):
        self.steps += 1
        if random.choice([True, False]):
            self._move_player(1, action[0])
            self._move_player(2, action[1])
        else:
            self._move_player(2, action[1])
            self._move_player(1, action[0])
        reward, done = self._check_goal()
        truncated = self.steps >= self.max_steps
        return self._get_obs(), reward, done, truncated, {}

    def _move_player(self, player, action):
        if player == 1:
            pos = list(self.player1_pos)
            opponent_pos = self.player2_pos
        else:
            pos = list(self.player2_pos)
            opponent_pos = self.player1_pos

        if action == 1:  # up
            pos[1] = max(0, pos[1] - 1)
        elif action == 2:  # down
            pos[1] = min(self.grid_size[1] - 1, pos[1] + 1)
        elif action == 3:  # left
            pos[0] = max(0, pos[0] - 1)
        elif action == 4:  # right
            pos[0] = min(self.grid_size[0] - 1, pos[0] + 1)

        if tuple(pos) != opponent_pos:
            if player == 1:
                self.player1_pos = tuple(pos)
            else:
                self.player2_pos = tuple(pos)
        else:
            self.ball_control = player

    def _check_goal(self):
        reward = 0
        done = False
        if (
            self.ball_control == 1
            and self.player1_pos[0] == self.grid_size[0] - 1
            and self.player1_pos[1]
            in [self.grid_size[1] // 2 - 1, self.grid_size[1] // 2]
        ):
            reward = 1
            done = True
        elif (
            self.ball_control == 2
            and self.player2_pos[0] == 0
            and self.player2_pos[1]
            in [self.grid_size[1] // 2 - 1, self.grid_size[1] // 2]
        ):
            reward = -1
            done = True
        return reward, done
