# Remote GPU instance (MicroAGI RTX 6000 Pro)

Extra compute for the ablation study ([docs/ablation/infrastructure.md](../../docs/ablation/infrastructure.md)).
The instance is a container with one RTX 6000 Pro (Blackwell, 96 GB), reachable only over the Cloudflare WARP VPN,
so it **pulls** work from this repo and **pushes** results to the branch `remote-results`.

| File | Purpose |
|---|---|
| `setup_instance.sh` | installs our extras into the Isaac Lab 2.3.2 image (re-run after every instance restart, ~1 min) |
| `run_queue.sh` | loop: `git pull`, start new lines of `queue.txt` (up to N at once on the GPU), train + eval, push status |
| `eval_run.sh` | same end-of-training eval as DTU (metrics, perturbed eval, MuJoCo videos, W&B) |
| `queue.txt` | the runs to do; add a line and push to queue one |

## Access

The pod has a private address on the MicroAGI cluster. As of 2026-10-29 the team reaches it from inside the ETH
network through the SRL gateway (`ssh rwr-gpu` = ProxyJump via `rwr_team4@srl-dsk-004.ethz.ch`, keys from the team
instructions; WARP is not needed). DTU cannot reach it, which is fine: the queue below only needs the pod's outbound
access to GitHub and W&B. The pod keeps running after you log out; its address changes when it restarts.

## One-time setup

1. Platform page: add your SSH public key; start an instance with **GPU rtx6000 ×1** and
   **Advanced → Image** `europe-west4-docker.pkg.dev/micro-research-shared/nvcr-mirror/nvidia/isaac-lab:2.3.2`.
2. SSH in, then:
   ```bash
   sudo -i                                    # root login shell: the NVIDIA driver is injected via /etc/profile.d
   ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519 && cat ~/.ssh/id_ed25519.pub   # add to GitHub (push access)
   git clone -b ablation-study git@github.com:rwr-fast/rwr_faive_lab.git ~/rwr_faive_lab
   bash ~/rwr_faive_lab/scripts/remote/setup_instance.sh
   wandb login
   tmux new -s queue 'bash ~/rwr_faive_lab/scripts/remote/run_queue.sh 2'
   ```
   Check that `~` is the persistent home (the platform keeps home files across restarts; container changes are not
   kept with a non-stock image). After a restart: `sudo -i`, `setup_instance.sh`, start the tmux line again; runs
   already marked in `logs/remote_state/` are not repeated.

## Reading results

`git fetch origin remote-results && git show origin/remote-results:STATUS.md` (per run: status, last iteration);
`runs/<run dir>/metrics*.json` (eval) and `runs/<name>.tail.txt` (last log lines). Videos and curves are on W&B.
