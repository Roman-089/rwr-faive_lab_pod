# Faive Lab

## Overview

This repository contains the Isaac Lab environment for the ORCA v2 hand.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

- [Tutorial Notion page](https://www.notion.so/Getting-started-with-faive_lab-1c2bcc457b5880178f2dfdf40e2c9e0e): Task names and paths in this page are outdated, but you may still find some useful commands and additional tips not listed in this readme.

## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):
    1. make sure you have [Git LFS installed](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage).
    2. after installing Git LFS, clone your team's repository: `git clone <your-team-repo-SSH-URL>`
       (if you cloned before installing Git LFS, run `git lfs pull` inside the repo, otherwise the `.usd` hand models are only small placeholder files and Isaac Sim fails to load them)


- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/faive_lab
    ```

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        ```bash
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        python scripts/rsl_rl/train.py --task=orca-v2-inhand-repose-v0 --num_envs 2
        ```

## Quick reference

Run all commands from the repository root.

### Train

```bash
python scripts/rsl_rl/train.py --task=orca-v2-inhand-repose-v0 --headless --num_envs 8192 --logger wandb --video --video_interval 4000
```

- Run `wandb login` once before using `--logger wandb`. Runs go to your personal W&B account; to log to a shared W&B team instead, `export WANDB_USERNAME=<team name>` first.
- Each run is saved to `logs/rsl_rl/orca_v2_repose/<date_time>/`: checkpoints (`model_<iteration>.pt`), training videos (`videos/train/`) and the exact env/agent config used (`params/`).
- In W&B, each reward term appears as `Episode_Reward/<term name>` and the rotation command's metrics (position error, angular velocity error, ...) as `Metrics/continuous_rotation/<metric>`. To log something new, add it to `self.metrics` in `_update_metrics()` of `tasks/orca_inhand/mdp/commands/continuous_command.py`.
- Other useful flags for the command above: `--max_iterations`, `--seed`, `--run_name` (suffix for the run folder), `--log_project_name` (W&B project name).

### Play and export the policy

```bash
python scripts/rsl_rl/play.py --task=orca-v2-inhand-repose-v0 --num_envs 16
```

- Loads the latest checkpoint of the latest run. Use `--checkpoint <path/to/model_XXXX.pt>` (full path) for a specific checkpoint.
- Exports `policy.pt` (TorchScript) and `policy.onnx` to `exported/` inside that run's folder. These are the files you deploy on the real hand with faive_system.
- `--headless --video` records a video to `videos/play/` in the run's folder and exits after `--video_length` steps (default 200).

### Where things are

| What | Where |
|---|---|
| Task: observations, rewards, domain randomization, control frequency (20 Hz) | `source/faive_lab/faive_lab/tasks/orca_inhand/orca_v2_repose_env_cfg.py` |
| Custom observation / reward / command functions | `source/faive_lab/faive_lab/tasks/orca_inhand/mdp/` |
| PPO hyperparameters | `source/faive_lab/faive_lab/tasks/orca_inhand/agents/rsl_rl_ppo_cfg.py` |
| ORCA v2 joint parameters | `source/faive_lab/faive_lab/assets/orca_v2.py` |


## Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/faive_lab"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```