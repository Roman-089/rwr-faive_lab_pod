#!/bin/bash
# Eval + videos + W&B upload for one finished run on the remote instance; same protocol as DTU's hpc/eval_run.sh
# (docs/ablation/eval_protocol.md), without the home-quota prune.
# Usage: PY="<isaac python>" bash scripts/remote/eval_run.sh RUN_DIR TASK   (cwd = repo root)
RUN_DIR=$1; TASK=$2; PY=${PY:-python}
KITROOT=/tmp/rwr_kit_eval_$$; mkdir -p $KITROOT
# checkpoints at ~1/5, 2/5, 3/5, 4/5 of training + the last one
mapfile -t ALL < <(ls $RUN_DIR/model_*.pt | sort -V)
N=${#ALL[@]}; SEL=()
for f in 5 10 15 20; do SEL+=("${ALL[$(( (N-1) * f / 25 ))]}"); done; SEL+=("${ALL[$((N-1))]}")
SEL=($(printf "%s\n" "${SEL[@]}" | awk '!seen[$0]++'))
echo "[eval_run] $RUN_DIR ($TASK): ${SEL[*]##*/}"
$PY scripts/eval/eval_policy.py --task $TASK --headless --num_envs 64 --steps 1200 \
    --checkpoints "${SEL[@]}" --kit_args "--portable-root $KITROOT" || { rm -rf $KITROOT; exit 1; }
$PY scripts/eval/eval_policy.py --task $TASK --headless --num_envs 64 --steps 1200 --perturb \
    --checkpoints "${SEL[@]}" --kit_args "--portable-root $KITROOT" || echo "[eval_run] perturbed eval failed"
rm -rf $KITROOT
for c in "${SEL[@]}"; do
  MUJOCO_GL=egl $PY scripts/replay/render_rollout.py $RUN_DIR/eval/$(basename $c .pt).npz --frames 600 --size 360 \
    || MUJOCO_GL=osmesa $PY scripts/replay/render_rollout.py $RUN_DIR/eval/$(basename $c .pt).npz --frames 600 --size 360
done
$PY scripts/eval/log_to_wandb.py $RUN_DIR
rm -f $RUN_DIR/eval/*_perturbed.npz
