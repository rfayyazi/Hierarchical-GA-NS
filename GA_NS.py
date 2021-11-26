import os
import sys
import argparse
import copy
import time
from tqdm import tqdm
import torch
from sklearn.neighbors import NearestNeighbors
import numpy as np
import wandb

from maze import Maze
from agent import Primitive
from plotting import plot_grid, plot_loss
from utils import save_args

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
    best_performances = []    # track performance (distance to goal) of most performant individual in each generation
    best_behaviours = []      # track behaviour (final position) of most performant individual in each generation
    novelest_behaviours = []  # track behaviour of most novel individual in each generation
    all_behaviours = []       # track behaviour of all individuals in all generations

    solution_found = False
    archive = []
    population = []
    BC = []

    for g in tqdm(range(args.G)):
        new_population = []
        new_BC = []
        for i in range(1, args.N):
            if g == 0:
                policy = Primitive(env.D**2, 4, args.hidden_dims)
                for theta in policy.parameters():
                    theta.requires_grad = False
            else:
                policy = copy.deepcopy(population[np.random.randint(args.T)])
                for theta in policy.parameters():
                    theta += args.sigma * torch.normal(0.0, 1.0, size=theta.shape)
            new_population.append(policy)
            new_BC.append(run(env, policy, args.t_max))

        if g == 0:
            population = [copy.deepcopy(policy) for policy in new_population]
            BC = [copy.deepcopy(bc) for bc in new_BC]
        else:
            population = [population[0]] + [copy.deepcopy(policy) for policy in new_population]
            BC = [BC[0]] + [copy.deepcopy(bc) for bc in new_BC]

        novelty = []
        best_performance = np.inf
        best_behaviour = np.array([0, 0])
        for i in range(len(population)):
            others = np.asarray(BC[:i] + BC[i+1:] + archive)
            neighbours = NearestNeighbors(n_neighbors=args.k, algorithm="kd_tree", metric="euclidean").fit(others)
            distances, _ = neighbours.kneighbors(np.asarray([BC[i]]))
            novelty.append(sum(distances[0]) / args.k)

            performance = np.linalg.norm(env.goal_pos - BC[i])
            if performance < best_performance:
                best_performance = performance
                best_behaviour = BC[i]

            if i > 0 and np.random.binomial(1, args.p):
                archive.append(BC[i])

            all_behaviours.append(BC[i])

        best_performances.append(best_performance)
        best_behaviours.append(best_behaviour)

        order = np.argsort(novelty)[::-1]
        population = [population[i] for i in order]
        novelest_behaviours.append(BC[order[0]])

    return best_performances, best_behaviours, novelest_behaviours, all_behaviours


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t_max", default=300, help="time limit for episode", type=int)
    parser.add_argument("--G", default=500, help="number of generations", type=int)
    parser.add_argument("--N", default=100, help="population size", type=int)
    parser.add_argument("--T", default=30, help="truncation size", type=int)
    parser.add_argument("--sigma", default=0.005, help="parameter mutation standard deviation", type=float)
    parser.add_argument("--hidden_dims", default=[128, 128], help="list of 2 hidden dims of policy network", nargs="+")
    parser.add_argument("--k", default=25, help="number of nearest neighbours for novelty", type=int)
    parser.add_argument("--p", default=0.05, help="archive probability", type=float)
    parser.add_argument("--D", default=40, help="maze dim, 20 or 40 or 84", type=int)
    args = parser.parse_args()

    assert args.N > args.T, "population size (N) must be greater than truncation size (T)"
    args.exp_tag = "GA-NS"
    args.run_name = args.exp_tag + "-" + time.strftime("%y%m%d") + "-" + time.strftime("%H%M%S")
    args.results_path = os.path.join("results", args.run_name)
    return args


def main():
    args = get_args()
    os.mkdir(args.results_path)

    if "--unobserve" in sys.argv:
        sys.argv.remove("--unobserve")
        os.environ["WANDB_MODE"] = "dryrun"
    os.environ["WANDB_CONSOLE"] = "off"
    wandb.init(project="CS532J_Final", entity="rfayyazi", config=args, name=args.run_name, tags=[args.exp_tag])

    env = Maze("saved_mazes/grid_hard_" + str(args.D) + ".json")
    args.reward_bias = np.linalg.norm(env.goal_pos - np.array([args.D, args.D]))
    best_ps, best_bs, novel_bs, all_bs = train(args, env)

    plot_grid(env, best_bs, args.results_path, "best_behaviours")
    plot_grid(env, novel_bs, args.results_path, "novelest_behaviours")
    plot_grid(env, all_bs, args.results_path, "all_behaviours")
    plot_loss(best_ps, args.results_path)

    save_args(args)


if __name__ == "__main__":
    main()