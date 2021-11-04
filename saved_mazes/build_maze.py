"""
Maze builder GUI based on array-backed grid from
    http://programarcadegames.com/index.php?lang=en&chapter=array_backed_grids#step_03

Instructions: Set dimension N of square grid and run script. Click to change square type. Blue = wall, green = start
position, red = target position. Press s at any time to save current maze to json file. Outer walls will be added later.
"""

import pygame
import numpy as np
import copy
import time
import json
import matplotlib.pyplot as plt


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
SQUARE_DIM = 20
MARGIN = 5


def init_grid(D):
    grid = np.zeros((D, D), dtype=int)
    # grid[0, :] = 10
    # grid[-1, :] = 10
    # grid[:, 0] = 10
    # grid[:, -1] = 10
    return grid


def save_grid(grid):
    new_grid = copy.deepcopy(grid)
    # new_grid[0, :] = 1
    # new_grid[-1, :] = 1
    # new_grid[:, 0] = 1
    # new_grid[:, -1] = 1
    new_grid = new_grid.astype("int").tolist()
    file_name = "maze_" + time.strftime("%y%m%d") + "_" + time.strftime("%H%M%S") + ".json"
    with open(file_name, "w") as f:
        json.dump(new_grid, f)


def plot_maze(file_name):
    json_path = file_name + ".json"
    with open(json_path, "r") as f:
        grid = np.asarray(json.load(f))
    plt.imshow(grid, cmap="hot")
    plt.savefig(file_name + ".png")
    plt.show()


def make_maze(D=10):
    grid = init_grid(D)
    window_dim = (D * SQUARE_DIM) + ((D + 1) * MARGIN)
    pygame.init()
    pygame.display.set_caption("Maze Builder")
    screen = pygame.display.set_mode([window_dim, window_dim])
    clock = pygame.time.Clock()

    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    save_grid(grid)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                column = pos[0] // (SQUARE_DIM + MARGIN)
                row = pos[1] // (SQUARE_DIM + MARGIN)
                if grid[row, column] != 10:  # can't change border walls
                    grid[row, column] = int((grid[row, column]+1) % 4)

        screen.fill(BLACK)

        for row in range(D):
            for column in range(D):
                # if grid[row, column] == 10 or grid[row, column] == 1:
                if grid[row, column] == 1:
                    c = BLUE
                elif grid[row, column] == 2:
                    c = GREEN
                elif grid[row, column] == 3:
                    c = RED
                else:
                    c = WHITE
                pygame.draw.rect(screen, c,
                                 [(MARGIN + SQUARE_DIM) * column + MARGIN,
                                  (MARGIN + SQUARE_DIM) * row + MARGIN,
                                  SQUARE_DIM,
                                  SQUARE_DIM])
        clock.tick(60)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    # make_maze(D=18)
    plot_maze("maze_hard_grid")