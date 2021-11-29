import numpy as np
import json
import torch


class Maze:
    def __init__(self, grid_path):
        with open(grid_path, "r") as f:
            self.grid = np.asarray(json.load(f))
        self.D = self.grid.shape[0]
        self.start_pos = np.array([idx[0] for idx in np.where(self.grid == 2)])
        self.grid[tuple(self.start_pos)] = 0  # current position only added when get_current_state() called
        self.goal_pos = np.array([idx[0] for idx in np.where(self.grid == 3)])
        self.curr_pos = np.copy(self.start_pos)
        self.last_pos = np.copy(self.curr_pos)

    def reset(self):
        self.curr_pos = np.copy(self.start_pos)
        self.last_pos = np.copy(self.start_pos)

    def get_current_state(self):
        frame = np.copy(self.grid)
        frame[tuple(self.last_pos)] = 5
        frame[tuple(self.curr_pos)] = 7
        frame = np.expand_dims(frame, 0)  # add single channel dim
        frame = np.expand_dims(frame, 0)  # batch size of 1
        return frame

    def step(self, action):
        assert action in [0, 1, 2, 3], "action must be 0 (down), 1 (up), 2 (right) or 3 (left)"
        self.last_pos = np.copy(self.curr_pos)
        if action == 0 and self.grid[self.curr_pos[0] + 1, self.curr_pos[1]] != 1:  # legal move down
            self.curr_pos[0] += 1
        elif action == 1 and self.grid[self.curr_pos[0] - 1, self.curr_pos[1]] != 1:  # legal move up
            self.curr_pos[0] -= 1
        elif action == 2 and self.grid[self.curr_pos[0], self.curr_pos[1] + 1] != 1:  # legal move right
            self.curr_pos[1] += 1
        elif action == 3 and self.grid[self.curr_pos[0], self.curr_pos[1] - 1] != 1:  # legal move left
            self.curr_pos[1] -= 1
        return self.get_current_state()


def run(env, policy, T, deterministic):
    env.reset()
    S = env.get_current_state()
    for t in range(T):
        S = torch.from_numpy(S).float()
        if deterministic:
            A = torch.argmax(policy(S))
        else:
            A = torch.distributions.Categorical(policy(S)).sample()
        S = env.step(A)
    return env.curr_pos


if __name__ == "__main__":
    maze = Maze("saved_mazes/grid_hard_20.json")