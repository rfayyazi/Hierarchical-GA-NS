import os
import copy
import argparse
import time
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
import pickle
import json
from tqdm import tqdm

from maze import Maze, run
from agent import Primitive
from plotting import plot_grid, plot_loss


def train(args, env):
    PR = {"best": []}  # performance (distance to goal) results
    BR = {"best": [], "novel": [], "all": []}  # behaviour (final position) results

    A = []  # archive
    P = []  # population
    B = []  # behaviour characteristics

    for g in tqdm(range(args.G)):

        if g != 0:
            P0 = copy.deepcopy(P[0])
            B0 = np.copy(B[0])

        P_new = []
        B_new = []
        for i in range(1, args.N):
            if g == 0:
                policy = Primitive(4, args.D)
                for theta in policy.parameters():
                    theta.requires_grad = False
            else:
                policy = copy.deepcopy(P[np.random.randint(args.T)])
                for theta in policy.parameters():
                    theta += args.sigma * torch.normal(0.0, 1.0, size=theta.shape)
            P_new.append(policy)
            B_new.append(run(env, policy, args.t_max, args.det_policy))

        P = P_new
        B = B_new
        if g != 0:
            P.insert(0, P0)
            B.insert(0, B0)

        novelty = []
        best_per = np.inf  # best performance so far
        best_beh = None    # behaviour characterization of best performer so far
        for i in range(len(P)):
            others = np.asarray(B[:i] + B[i + 1:] + A)
            neighbours = NearestNeighbors(n_neighbors=args.k, algorithm="kd_tree", metric="euclidean").fit(others)
            distances, _ = neighbours.kneighbors(np.asarray([B[i]]))
            novelty.append(sum(distances[0]) / args.k)

            performance = np.linalg.norm(env.goal_pos - B[i])
            if performance < best_per:
                best_per = performance
                best_beh = B[i]

            if i > 0 and np.random.binomial(1, args.p):
                A.append(np.copy(B[i]))

            BR["all"].append(B[i])

        PR["best"].append(best_per)
        BR["best"].append(best_beh)

        order = np.argsort(novelty)[::-1]  # most to least novel (larger is better)
        P = [P[i] for i in order]
        B = [B[i] for i in order]
        BR["novel"].append(B[0])

    return BR, PR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--G",           default=250,   help="number of generations",                 type=int)
    parser.add_argument("--N",           default=300,   help="population size",                       type=int)
    parser.add_argument("--T",           default=30,    help="truncation size",                       type=int)
    parser.add_argument("--sigma",       default=0.005, help="parameter mutation standard dev",       type=float)
    parser.add_argument("--p",           default=1.0,   help="archive probability",                   type=float)
    parser.add_argument("--k",           default=30,    help="num of nearest neighbours for novelty", type=int)
    parser.add_argument("--det_policy",  default=True,  help="deterministic policy?",                 type=bool)
    parser.add_argument("--D",           default=84,    help="maze dim, 20 or 40 or 84",              type=int)
    parser.add_argument("--t_max",       default=200,   help="time limit for episode",                type=int)
    parser.add_argument("--tiny_exp",    default=False, help="if true, run tiny experiment (small G, N, t_max)")
    args = parser.parse_args()

    if args.tiny_exp:
        args.G = 10
        args.T = 5
        args.k = 5
        args.N = 20
        args.t_max = 10

    assert args.N > args.T, "population size (N) must be greater than truncation size (T)"
    args.exp_tag = "N"
    args.run_name = args.exp_tag + "-" + time.strftime("%y%m%d") + "-" + time.strftime("%H%M%S")
    args.results_path = os.path.join("results", args.run_name)
    return args


def main():
    args = get_args()
    os.mkdir(args.results_path)

    env = Maze("saved_mazes/grid_hard_" + str(args.D) + ".json")
    behaviour_results, performance_results = train(args, env)

    plot_grid(env, behaviour_results["best"], args.results_path, "best_behav")
    plot_grid(env, behaviour_results["novel"], args.results_path, "novelest_behav")
    plot_grid(env, behaviour_results["all"], args.results_path, "all_behav")
    plot_loss(performance_results["best"], args.results_path)

    json.dump(vars(args), open(os.path.join(args.results_path, "args.json"), "w"))
    pickle.dump(behaviour_results, open(os.path.join(args.results_path, "behav_res.p"), "wb"))
    pickle.dump(performance_results, open(os.path.join(args.results_path, "perf_res.p"), "wb"))


if __name__ == "__main__":
    main()
