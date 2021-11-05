import numpy as np
import json
import copy
import matplotlib.pyplot as plt


class Maze:
    def __init__(self, grid_path):
        with open(grid_path, "r") as f:
            self.grid = np.asarray(json.load(f))
        self.D = self.grid.shape[0]
        self.start_pos = np.array([idx[0] for idx in np.where(self.grid == 2)])
        self.grid[tuple(self.start_pos)] = 0
        self.goal_pos = np.array([idx[0] for idx in np.where(self.grid == 3)])
        self.curr_pos = np.copy(self.start_pos)

    def reset(self):
        self.curr_pos = np.copy(self.start_pos)

    def get_current_state(self):
        state = copy.deepcopy(self.grid)
        state[tuple(self.curr_pos)] = 2
        return state.flatten()

    def step(self, action):
        assert action in [0, 1, 2, 3], "action must be 0 (up), 1 (down), 2 (right) or 3 (left)"
        if action == 0 and self.grid[self.curr_pos[0] + 1, self.curr_pos[1]] != 1:  # legal move down
            self.curr_pos[0] += 1
        elif action == 1 and self.grid[self.curr_pos[0] - 1, self.curr_pos[1]] != 1:  # legal move up
            self.curr_pos[0] -= 1
        elif action == 2 and self.grid[self.curr_pos[0], self.curr_pos[1] + 1] != 1:  # legal move right
            self.curr_pos[1] += 1
        elif action == 3 and self.grid[self.curr_pos[0], self.curr_pos[1] - 1] != 1:  # legal move left
            self.curr_pos[1] -= 1
        return self.get_current_state()


if __name__ == "__main__":
    maze = Maze("saved_mazes/grid_hard_20.json")