#!/bin/bash
# Pull-based run queue for the RTX 6000 Pro instance: runs the lines of scripts/remote/queue.txt (train + the same
# end-of-training eval as on DTU) with up to SLOTS runs sharing the GPU, and pushes status + eval metrics to the
# branch `remote-results` so they can be read from anywhere (the instance is only reachable over WARP).
#
# Usage (root login shell, after setup_instance.sh):  tmux new -s queue 'bash ~/rwr_faive_lab/scripts/remote/run_queue.sh 2'
# Queue line format:  RUN_NAME TASK ITERS [extra train.py args ...]     ('#' comments; a run is started once)
# Needs git push access from the instance (an SSH key on the instance that is added to GitHub).
SLOTS=${1:-2}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ISAACLAB=${ISAACLAB:-/workspace/isaaclab}
PY="$ISAACLAB/isaaclab.sh -p"
STATE=$REPO/logs/remote_state; RES=$HOME/rwr_results
export OMNI_KIT_ALLOW_ROOT=1 OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
export WANDB_ENTITY=rwr-fast WANDB_USERNAME=rwr-fast WANDB_DIR=$REPO/logs
mkdir -p "$STATE"
cd "$REPO"

# results branch in its own clone, so the working repo stays on ablation-study
if [ ! -d "$RES/.git" ]; then
  git clone -q "$(git remote get-url origin)" "$RES"
  git -C "$RES" config user.name Roman-089
  git -C "$RES" config user.email 209470582+Roman-089@users.noreply.github.com
  (cd "$RES" && { git checkout -q remote-results 2>/dev/null || { git checkout -q --orphan remote-results && git rm -rqf . ; }; })
fi

run_one() {  # RUN_NAME TASK ITERS EXTRA...
  local name=$1 task=$2 iters=$3; shift 3
  local kit=/tmp/rwr_kit_$name; mkdir -p "$kit"
  $PY scripts/rsl_rl/train.py --task="$task" --headless --num_envs 8192 --max_iterations "$iters" \
      --logger wandb --log_project_name orca_v2_repose --kit_args "--portable-root $kit" --run_name "$name" \
      agent.save_interval=500 "$@" || return 1
  local run_dir; run_dir=$(ls -dt "$REPO"/logs/rsl_rl/orca_v2_repose/*_"$name" | head -1)
  PY="$PY" bash scripts/remote/eval_run.sh "$run_dir" "$task" || return 2
}

publish() {  # copy status + metrics into the results branch and push
  mkdir -p "$RES/runs"
  {
    echo "# remote queue status ($(date -u +%FT%TZ), $(nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader))"
    for f in "$STATE"/*.status; do [ -e "$f" ] || continue
      n=$(basename "$f" .status); it=$(grep -aoE 'Learning iteration +[0-9]+/[0-9]+' "$STATE/$n.log" 2>/dev/null | tail -1)
      echo "$n $(cat "$f") ${it}"
    done
  } > "$RES/STATUS.md"
  for d in "$REPO"/logs/rsl_rl/orca_v2_repose/*; do
    n=${d##*/}; [ -f "$d/eval/metrics.json" ] || continue
    mkdir -p "$RES/runs/$n"; cp "$d"/eval/metrics*.json "$RES/runs/$n/"
  done
  for f in "$STATE"/*.log; do [ -e "$f" ] && tail -n 40 "$f" > "$RES/runs/$(basename "$f" .log).tail.txt"; done
  (cd "$RES" && git add -A && { git diff --cached --quiet || git commit -qm "remote status $(date -u +%FT%TZ)"; } \
     && git push -q origin remote-results 2>&1 | tail -1)
}

declare -A PID
loop=0
while true; do
  changed=0
  git pull -q --ff-only origin ablation-study || echo "[queue] git pull failed"
  for n in "${!PID[@]}"; do   # reap finished runs
    if ! kill -0 "${PID[$n]}" 2>/dev/null; then
      wait "${PID[$n]}"; rc=$?
      case $rc in 0) s=done ;; 1) s=failed_train ;; *) s=failed_eval ;; esac
      echo "$s" > "$STATE/$n.status"; unset "PID[$n]"; echo "[queue] $n -> $s"; changed=1
    fi
  done
  while read -r name task iters extra; do
    [[ -z "$name" || "$name" == \#* ]] && continue
    [ -e "$STATE/$name.status" ] && continue
    [ ${#PID[@]} -ge "$SLOTS" ] && break
    echo running > "$STATE/$name.status"
    # shellcheck disable=SC2086
    ( run_one "$name" "$task" "$iters" $extra ) > "$STATE/$name.log" 2>&1 &
    PID[$name]=$!; echo "[queue] started $name ($task, $iters it) pid ${PID[$name]}"; changed=1
  done < scripts/remote/queue.txt
  # push when a run starts / ends, else every 30 min (keeps remote-results readable)
  { [ $changed = 1 ] || (( loop % 6 == 0 )); } && publish
  loop=$((loop + 1))
  sleep 300
done
