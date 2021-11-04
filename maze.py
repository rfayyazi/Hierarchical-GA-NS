import numpy as np
import json
import matplotlib.pyplot as plt


class Maze:

    def __init__(self, grid_path, view_rad):
        """
        :param grid_path: path to json file containing DxD grid
        :param view_rad: radius of agent's view horizon
        """
        self.t_max = 100
        self.t_cur = 1
        self.view_rad = view_rad
        with open(grid_path, "r") as f:
            grid = json.load(f)
        self.D = len(grid)  # grid dimension, doesn't include outer walls
        self.grid = np.asarray(add_bumpers(grid, view_rad))
        self.start_pos = [idx[0] for idx in np.where(self.grid == 2)]
        self.goal_pos = [idx[0] for idx in np.where(self.grid == 3)]
        self.curr_pos = self.start_pos[:]

    def get_view(self):
        view = self.grid[self.curr_pos[0] - self.view_rad:self.curr_pos[0] + self.view_rad + 1,
               self.curr_pos[1] - self.view_rad:self.curr_pos[1] + self.view_rad + 1]
        return view

    def p_s_s_a(self, action):
        assert action in [0, 1, 2, 3], "action must be 0 (up), 1 (down), 2 (right) or 3 (left)"
        self.grid[tuple(self.curr_pos)] = 0
        if action == 0 and self.grid[self.curr_pos[0] + 1, self.curr_pos[1]] != 1:  # legal move down
            self.curr_pos[0] += 1
        elif action == 1 and self.grid[self.curr_pos[0] - 1, self.curr_pos[1]] != 1:  # legal move up
            self.curr_pos[0] -= 1
        elif action == 2 and self.grid[self.curr_pos[0], self.curr_pos[1] + 1] != 1:  # legal move right
            self.curr_pos[1] += 1
        elif action == 3 and self.grid[self.curr_pos[0], self.curr_pos[1] - 1] != 1:  # legal move left
            self.curr_pos[1] -= 1
        self.grid[tuple(self.curr_pos)] = 2
        view = self.get_view()
        view = view.flatten()
        return view


def add_bumpers(grid, view_rad):
    D_old = len(grid)
    x_bumper = [1 for _ in range(view_rad)]
    for i in range(D_old):
        grid[i] = x_bumper + grid[i] + x_bumper
    new_D = D_old + (2 * view_rad)
    y_bumper = [[1 for _ in range(new_D)] for _ in range(view_rad)]
    grid = y_bumper + grid + y_bumper
    return grid


if __name__ == "__main__":
    maze = Maze("saved_mazes/maze_hard_grid.json", 2)