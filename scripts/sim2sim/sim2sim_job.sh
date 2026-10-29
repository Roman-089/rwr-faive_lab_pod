#!/bin/bash
# Sim2sim eval of exported policies in MuJoCo on CPU (no GPU, no Isaac Sim): 64 trials x 60 s per policy, with
# nominal physics and with Isaac's DR, 4 single-threaded processes in parallel (~25 min).
# Submit from the repo root:   bsub < scripts/sim2sim/sim2sim_job.sh
# DTU's bsub does not forward shell variables: edit RUNS / TRIALS below. Policies must be exported first
# (scripts/export/export_history_policy.py --run <run>, CPU, ~10 s). Results: <run>/sim2sim/metrics_dr{none,isaac}.json
#BSUB -J rwr_sim2sim
#BSUB -q hpc
#BSUB -n 4
#BSUB -R "rusage[mem=2GB] span[hosts=1]"
#BSUB -W 01:30
#BSUB -o /zhome/b3/e/222334/rwr_faive_lab/hpc/logs/sim2sim_%J.out
#BSUB -e /zhome/b3/e/222334/rwr_faive_lab/hpc/logs/sim2sim_%J.err

TRIALS=64
RUNS="2026-10-28_21-11-45_abl_B_rotlin 2026-10-29_04-06-33_abl_B_droppen 2026-10-28_09-29-28_abl_B_asym_ent5e-3 2026-10-28_22-01-33_student_pt"

source ~/rwr_faive_lab/activate_rwr.sh  # conda env rwr, cd ~/rwr_faive_lab
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p hpc/logs

jobs_run=0
for run in $RUNS; do
    for dr in none isaac; do
        python scripts/sim2sim/run_sim2sim.py --policy logs/rsl_rl/orca_v2_repose/$run --trials $TRIALS --dr $dr \
            > hpc/logs/sim2sim_${LSB_JOBID}_${run}_dr${dr}.log 2>&1 &
        jobs_run=$((jobs_run + 1))
        if [ $((jobs_run % 4)) -eq 0 ]; then wait; fi
    done
done
wait
grep -h '^\[sim2sim\] {' hpc/logs/sim2sim_${LSB_JOBID}_*.log
