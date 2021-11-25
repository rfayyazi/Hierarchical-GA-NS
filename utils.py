import json
import os
import wandb


def save_args(args):
    with open(os.path.join(args.results_path, "args.json"), "w") as f:
        args.wandb_run_id = wandb.run.id
        json.dump(vars(args), f)