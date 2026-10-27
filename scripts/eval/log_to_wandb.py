"""Attach eval metrics and rollout videos to a training run's W&B page (after training has finished).

The W&B run is found by its name, which rsl_rl sets to the log directory name, and resumed. Logged:
    eval/<metric>                 per checkpoint, with eval/iteration as x-axis
    eval_perturbed/<metric>       same with test-time delay + joint offset (if metrics_perturbed.json exists)
    eval/video_<checkpoint>       MP4 rendered by scripts/replay/render_rollout.py
    summary eval_final/<metric>   metrics of the last checkpoint

Usage:
    python scripts/eval/log_to_wandb.py logs/rsl_rl/orca_v2_repose/<run> [--project orca_v2_repose]
"""

import argparse
import glob
import json
import os
import re

import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--project", default="orca_v2_repose")
    p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "rwr-fast"))
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir.rstrip("/"))
    name = os.path.basename(run_dir)
    runs = wandb.Api().runs(f"{args.entity}/{args.project}", filters={"display_name": name})
    if len(runs) == 0:
        raise SystemExit(f"no W&B run named {name} in {args.entity}/{args.project}")
    run_id = runs[0].id

    results = json.load(open(os.path.join(run_dir, "eval", "metrics.json")))
    perturbed_path = os.path.join(run_dir, "eval", "metrics_perturbed.json")
    perturbed = json.load(open(perturbed_path)) if os.path.exists(perturbed_path) else {}
    ckpts = sorted(results, key=lambda c: int(re.findall(r"\d+", c)[-1]))
    run = wandb.init(project=args.project, entity=args.entity, id=run_id, resume="must")
    run.define_metric("eval/iteration")
    run.define_metric("eval/*", step_metric="eval/iteration")
    run.define_metric("eval_perturbed/*", step_metric="eval/iteration")
    for c in ckpts:
        m = results[c]
        it = int(re.findall(r"\d+", c)[-1])
        log = {"eval/iteration": it, **{f"eval/{k}": m[k] for k in ("rpm", "ra", "dfsr", "dc")}}
        if c in perturbed:
            log.update({f"eval_perturbed/{k}": perturbed[c][k] for k in ("rpm", "ra", "dfsr", "dc")})
        video = os.path.join(run_dir, "eval", c.replace(".pt", ".mp4"))
        if os.path.exists(video):
            log[f"eval/video_{it:05d}"] = wandb.Video(video, format="mp4", caption=f"{name} {c}")
        run.log(log)
    run.summary.update({f"eval_final/{k}": results[ckpts[-1]][k] for k in ("rpm", "ra", "dfsr", "dc")})
    if ckpts[-1] in perturbed:
        run.summary.update({f"eval_final_perturbed/{k}": perturbed[ckpts[-1]][k] for k in ("rpm", "ra", "dfsr", "dc")})
    run.finish()
    print(f"logged {len(ckpts)} checkpoints to {args.entity}/{args.project}/{run_id} ({name})")


if __name__ == "__main__":
    main()
