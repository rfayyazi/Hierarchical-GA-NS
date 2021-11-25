import matplotlib.pyplot as plt
import os


def plot_loss(loss, results_path):
    plt.plot(loss)
    plt.savefig(os.path.join(results_path, "loss.png"))
    plt.clf()


def plot_grid(env, behaviours, results_path, plot_name):
    env.reset()
    env.grid[tuple(env.start_pos)] = 2
    plt.imshow(env.grid)
    x = [bc[1] for bc in behaviours]
    y = [bc[0] for bc in behaviours]
    plt.scatter(x, y, c="red", s=0.5)
    plt.savefig(os.path.join(results_path, plot_name + ".png"))
    plt.clf()