import os
import copy
import argparse
import time
import random
from sklearn.neighbors import NearestNeighbors
import torch
import numpy as np
import pickle
import json
from tqdm import tqdm

from maze import Maze
from policy import PrimSmall, Controller, Hierarchical
from plotting import plot_grid, plot_loss


def run_primitive(env, policy, T, deterministic):
    S = env.get_current_state()
    for t in range(T):
        S = torch.from_numpy(S).float()
        if deterministic:
            A = torch.argmax(policy(S))
        else:
            A = torch.distributions.Categorical(policy(S)).sample()
        S = env.step(A)
    return env


def run_hierarchy(env, policy, T, deterministic):
    S = env.get_current_state()
    S = torch.from_numpy(S).float()
    out = policy.controller(S, T)
    t1 = int(np.ceil(out[1]*T))
    t2 = int(np.floor((1-out[1])*T))
    p1 = np.random.binomial(1, out[0])
    if p1:  # run sub0 first, then sub1
        if policy.submodules[0].type == "primitive":
            env = run_primitive(env, policy.submodules[0], t1, deterministic)
        else:
            env = run_hierarchy(env, policy.submodules[0], t1, deterministic)
        if policy.submodules[1].type == "primitive":
            env = run_primitive(env, policy.submodules[1], t2, deterministic)
        else:
            env = run_hierarchy(env, policy.submodules[1], t2, deterministic)
    else:  # run sub1 first, then sub0
        if policy.submodules[1].type == "primitive":
            env = run_primitive(env, policy.submodules[1], t1, deterministic)
        else:
            env = run_hierarchy(env, policy.submodules[1], t1, deterministic)
        if policy.submodules[0].type == "primitive":
            env = run_primitive(env, policy.submodules[0], t1, deterministic)
        else:
            env = run_hierarchy(env, policy.submodules[0], t1, deterministic)
    return env


def conjunction(depth, P):
    HP = Hierarchical(depth)
    HP.controller = Controller()
    for theta in HP.controller.parameters():
        theta.requires_grad = False
    HP.submodules.append(copy.deepcopy(random.choice(P[depth-1])))
    if np.random.binomial(1, 0.5):
        HP.submodules.append(random.choice(P[np.random.randint(depth)]))
    else:
        HP.submodules.insert(0, random.choice(P[np.random.randint(depth)]))
    return HP


def train(args, env):
    PR = {"best": []}  # performance (distance to goal) results
    BR = {"best": [], "novel": [], "all": []}  # behaviour (final position) results

    A = []  # archive
    P = [[] for _ in range(args.H)]  # population
    B = [[] for _ in range(args.H)]  # behaviour characteristics

    for _ in range(args.N[0]):
        policy = PrimSmall()
        for theta in policy.parameters():
            theta.requires_grad = False
        P[0].append(policy)
        env.reset()
        env = run_primitive(env, policy, args.t_max, args.det_policy)
        B[0].append(env.curr_pos)

    for h in range(1, args.H):
        for _ in range(args.N[h]):
            policy = conjunction(h, P)
            P[h].append(policy)
            env.reset()
            env = run_hierarchy(env, policy, args.t_max, args.det_policy)
            B[h].append(env.curr_pos)

    for g in tqdm(range(args.G)):

        if g != 0:
            P0 = [copy.deepcopy(P[h][0]) for h in range(args.H)]
            B0 = [np.copy(B[h][0]) for h in range(args.H)]

        P_new = [[] for _ in range(args.H)]
        B_new = [[] for _ in range(args.H)]
        for h in range(args.H):
            m = np.floor(args.M[h])
            for i in range(1, args.N[h]):

                if g != 0:
                    if h == 0:
                        policy = copy.deepcopy(P[h][np.random.randint(args.T)])
                        for theta in policy.parameters():
                            theta += args.sigma * torch.normal(0.0, 1.0, size=theta.shape)
                        env.reset()
                        env = run_primitive(env, policy, args.t_max, args.det_policy)
                    else:
                        if i <= m:
                            policy = copy.deepcopy(P[h][np.random.randint(args.T)])
                            for theta in policy.controller.parameters():
                                theta += args.sigma * torch.normal(0.0, 1.0, size=theta.shape)
                        else:
                            policy = conjunction(h, P)
                        env.reset()
                        env = run_hierarchy(env, policy, args.t_max, args.det_policy)

                    P_new[h].append(policy)
                    B_new[h].append(env.curr_pos)

        if g != 0:
            P = P_new
            B = B_new
            for h in range(args.H):
                P[h].insert(0, P0[h])
                B[h].insert(0, B0[h])

        novelty = [[] for _ in range(args.H)]
        best_per = np.inf  # best performance so far
        best_beh = None  # behaviour characterization of best performer so far
        highest_novelty = 0
        novelest_beh = None

        B_flat = []
        for h in range(args.H):
            for b in B[h]:
                B_flat.append(b)

        i = 0
        for h in range(args.H):
            for n in range(len(B[h])):
                others = [list(o) for o in B_flat[:i] + B_flat[i+1:] + A]
                neighbours = NearestNeighbors(n_neighbors=args.k, algorithm="kd_tree", metric="euclidean").fit(others)
                distances, _ = neighbours.kneighbors([list(B_flat[i])])
                nov = sum(distances[0]) / args.k
                novelty[h].append(nov)
                if nov > highest_novelty:
                    highest_novelty = nov
                    novelest_beh = B_flat[i]

                performance = np.linalg.norm(env.goal_pos - B_flat[i])
                if performance < best_per:
                    best_per = performance
                    best_beh = B_flat[i]

                if i > 0 and np.random.binomial(1, args.p):
                    A.append(np.copy(B_flat[i]))

                BR["all"].append(B_flat[i])
                i += 1

        PR["best"].append(best_per)
        BR["best"].append(best_beh)

        order = [np.argsort(novelty[h])[::-1] for h in range(args.H)]  # most to least novel (larger is better)
        for h in range(args.H):
            P[h] = [P[h][j] for j in order[h]]
            B[h] = [B[h][j] for j in order[h]]
        BR["novel"].append(novelest_beh)

    return BR, PR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--G",          default=400,             help="number of generations", type=int)
    parser.add_argument("--N",          default=[600, 250, 150], help="pop size for each hierarchy depth", nargs="+")
    parser.add_argument("--M",          default=[1, 0.5, 0.5],   help="% new pop to be made via mutation", nargs="+")
    parser.add_argument("--T",          default=30,              help="truncation size", type=int)
    parser.add_argument("--sigma",      default=0.005,           help="parameter mutation standard dev", type=float)
    parser.add_argument("--p",          default=0.05,            help="archive probability", type=float)
    parser.add_argument("--k",          default=20,              help="num of nearest neighbours for novelty", type=int)
    parser.add_argument("--det_policy", default=True,            help="deterministic policy?", type=bool)
    parser.add_argument("--D",          default=40,              help="maze dim, 20 or 40 or 84", type=int)
    parser.add_argument("--t_max",      default=250,             help="time limit for episode", type=int)
    args = parser.parse_args()

    for n in args.N:
        assert n > args.T, "population size (n) must be greater than truncation size (T)"

    args.H = len(args.N)
    args.exp_tag = "H"
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