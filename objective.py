import os
import sys
import argparse
import copy
import time
from tqdm import tqdm
import torch
import numpy as np
import wandb

from maze import Maze
from agent import Primitive
from plotting import plot_grid, plot_loss
from utils import save_args


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


def train(args, env):
    PR = {"best": []}  # performance (distance to goal) results
    BR = {"best": [], "novel": [], "all": []}  # behaviour (final position) results

    elite = None
    P = []  # population

    for g in tqdm(range(args.G)):

        P_new = []
        performance = []
        behaviour = []
        for i in range(1, args.N):

            if g == 0:
                policy = Primitive(4)
                for theta in policy.parameters():
                    theta.requires_grad = False
            else:
                policy = copy.deepcopy(P[np.random.randint(args.T)])
                for theta in policy.parameters():
                    theta += args.sigma * torch.normal(0.0, 1.0, size=theta.shape)

            P_new.append(policy)
            final_pos = run(env, policy, args.t_max, args.det_policy)
            performance.append(np.linalg.norm(env.goal_pos - final_pos))
            behaviour.append(final_pos)
            BR["all"].append(final_pos)

        order = np.argsort(performance)  # best to worst performance (smaller distance to goal is better)
        P = [P_new[i] for i in order]

        if args.det_policy:
            PR["best"].append(performance[order[0]])
            BR["best"].append(behaviour[order[0]])

        else:

            if g == 0:
                C = [P[i] for i in range(args.n_candidates)]
            else:
                C = [elite] + [P[i] for i in range(args.n_candidates-1)]

            candidate_performance = []
            candidate_behaviour = []
            for c in C:

                best_performance = np.inf
                best_behaviour = None
                sum_performance = 0
                for _ in range(args.R):

                    final_pos = run(env, c, args.t_max, args.det_policy)
                    performance = np.linalg.norm(env.goal_pos - final_pos)
                    sum_performance += performance

                    if performance < best_performance:
                        best_performance = performance
                        best_behaviour = final_pos

                candidate_performance.append(sum_performance / args.R)
                candidate_behaviour.append(best_behaviour)

            elite_idx = np.argmax(candidate_performance)
            PR["best"].append(candidate_performance[elite_idx])  # average performance of elite over R runs
            BR["best"].append(candidate_behaviour[elite_idx])      # best behaviour of elite over R runs

            if elite_idx != 0:
                elite = C[elite_idx]
                del P[elite_idx]

            P.insert(0, elite)

    return BR, PR


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--G",            default=300,   help="number of generations", type=int)
    parser.add_argument("--N",            default=400,   help="population size", type=int)
    parser.add_argument("--T",            default=10,    help="truncation size", type=int)
    parser.add_argument("--n_candidates", default=10,    help="num of best performers to consider candidates", type=int)
    parser.add_argument("--R",            default=10,    help="number of repeats for candidate evaluation", type=int)
    parser.add_argument("--sigma",        default=0.005, help="parameter mutation standard deviation", type=float)
    parser.add_argument("--det_policy",   default=True,  help="deterministic policy?", type=bool)
    parser.add_argument("--D",            default=40,    help="maze dim, 20 or 40 or 84", type=int)
    parser.add_argument("--t_max",        default=200,   help="time limit for episode", type=int)
    args = parser.parse_args()

    assert args.N > args.T, "population size (N) must be greater than truncation size (T)"
    args.exp_tag = "O"
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

    behaviour_results, performance_results = train(args, env)

    plot_grid(env, behaviour_results["best"], args.results_path, "best_behaviours")
    plot_grid(env, behaviour_results["all"], args.results_path, "all_behaviours")
    plot_loss(performance_results["best"], args.results_path)

    save_args(args)


if __name__ == "__main__":
    main()