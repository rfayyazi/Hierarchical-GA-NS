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
            self.grid = np.asarray(json.load(f))
        self.D = self.grid.shape[0]  # grid dimension, doesn't include outer walls
        self.start_pos = [idx[0] for idx in np.where(self.grid == 2)]
        self.goal_pos = [idx[0] for idx in np.where(self.grid == 3)]
        self.curr_pos = self.start_pos[:]

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
        return self.grid
        # return self.grid.flatten()


if __name__ == "__main__":
    maze = Maze("saved_mazes/grid_hard.json", 2)
    # plt.imshow(maze.grid)
    # plt.show()
    # s = maze.p_s_s_a(0)
    # plt.imshow(s)
    # plt.show()
    # s = maze.p_s_s_a(1)
    # plt.imshow(s)
    # plt.show()
    # s = maze.p_s_s_a(1)
    # plt.imshow(s)
    # plt.show()
    # s = maze.p_s_s_a(2)
    # plt.imshow(s)
    # plt.show()
    # s = maze.p_s_s_a(3)
    # plt.imshow(s)
    # plt.show()