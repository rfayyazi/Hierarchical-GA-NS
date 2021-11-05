import os
import argparse
import copy
import time
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from maze import Maze
from agent import Primitive


def run(env, policy, T):
    env.reset()
    S = env.get_current_state()
    for t in range(T):
        S = torch.from_numpy(S).float()
        pi = torch.distributions.Categorical(policy(S))
        A = pi.sample()
        S = env.step(A)
    return env.curr_pos


def train(args, env):

    # track performance (distance to goal) and behaviour (final position) of most performant individual in each gen
    best_performances = []
    best_behaviours = []
    all_behaviours = []

    elite = None
    population = []

    for g in tqdm(range(args.G)):
        new_population = []
        performance = []
        for i in range(1, args.N):

            if g == 0:
                policy = Primitive(env.D**2, 4, args.hidden_dims)
                for theta in policy.parameters():
                    theta.requires_grad = False
                population.append(policy)
            else:
                t = np.random.randint(args.T)
                policy = population[t]
                for theta in policy.parameters():
                    theta += args.sigma * torch.normal(0.0, 1.0, size=theta.shape)
            new_population.append(policy)
            final_pos = run(env, policy, args.t_max)
            performance.append(np.linalg.norm(env.goal_pos - final_pos))
            all_behaviours.append(final_pos)

        if g > 0:
            population = [copy.deepcopy(policy) for policy in new_population]

        order = np.argsort(performance)  # performance is dist to goal, so smaller is better
        population = [population[i] for i in order]

        if g == 0:
            C = [population[i] for i in range(args.n_candidates)]
        else:
            C = [population[i] for i in range(args.n_candidates-1)]
            C.append(elite)

        candidate_performance = []
        candidate_behaviour = []
        for c in C:
            sum_performance = 0
            best_performance = np.inf
            best_behaviour = None
            for _ in range(30):
                final_pos = run(env, c, args.t_max)
                performance = np.linalg.norm(env.goal_pos - final_pos)
                sum_performance += performance
                if performance < best_performance:
                    best_performance = performance
                    best_behaviour = final_pos

            candidate_performance.append(sum_performance / 30.0)
            candidate_behaviour.append(best_behaviour)

        elite_idx = np.argmax(candidate_performance)
        best_performances.append(candidate_performance[elite_idx])
        best_behaviours.append(candidate_behaviour[elite_idx])

        if elite_idx != args.n_candidates-1:
            elite = C[elite_idx]
            del population[elite_idx]

        population.insert(0, elite)

    return best_performances, best_behaviours, all_behaviours


def plot_grid(env, behaviours, results_path, plot_name):
    env.reset()
    env.grid[tuple(env.start_pos)] = 2
    plt.imshow(env.grid)
    x = [bc[1] for bc in behaviours]
    y = [bc[0] for bc in behaviours]
    plt.scatter(x, y, c="red")
    plt.savefig(os.path.join(results_path, plot_name + ".png"))
    plt.clf()


def plot_loss(loss, results_path):
    plt.plot(loss)
    plt.savefig(os.path.join(results_path, "loss.png"))
    plt.clf()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_max", default=150, help="time limit for episode", type=int)
    parser.add_argument("--G", default=200, help="number of generations", type=int)
    parser.add_argument("--N", default=100, help="population size", type=int)
    parser.add_argument("--T", default=30, help="truncation size", type=int)
    parser.add_argument("--n_candidates", default=10, help="num of best performers to consider candidates", type=int)
    parser.add_argument("--sigma", default=0.005, help="parameter mutation standard deviation", type=float)
    parser.add_argument("--hidden_dims", default=[128, 128], help="list of 2 hidden dims of policy network", nargs="+")
    args = parser.parse_args()

    assert args.N > args.T, "population size (N) must be greater than truncation size (T)"
    args.exp_tag = "GA"
    args.run_name = args.exp_tag + "-" + time.strftime("%y%m%d") + "-" + time.strftime("%H%M%S")
    args.results_path = os.path.join("results", args.run_name)
    return args


def main():
    args = get_args()
    os.mkdir(args.results_path)
    env = Maze("saved_mazes/grid_hard_20.json")
    best_ps, best_bs, all_bs = train(args, env)
    plot_grid(env, best_bs, args.results_path, "best_behaviours")
    plot_grid(env, all_bs, args.results_path, "all_behaviours")
    plot_loss(best_ps, args.results_path)


if __name__ == "__main__":
    main()