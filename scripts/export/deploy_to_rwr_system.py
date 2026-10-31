"""Add one exported policy to rwr-fast/rwr_system, branch deploy-policies-2026-10-31, folder deploy_policies/.

Usage: python deploy_to_rwr_system.py spec.json [--push]
spec.json: {"name", "run_dir" (local run dir with exported/), "rank", "kind", "what", "task", "checkpoint",
            "training_run", "isaac": {"isaac_measured_friction": {"rpm","dfsr","dc"}, "isaac_own_training_task": {...}},
            "sim_wrist_speed", "readme_what", "readme_isaac", "metrics_files": [paths], "commit_msg"}
Without --push: prepares the commit in the clone and stops (dry run).
"""
import hashlib, json, os, shutil, subprocess, sys, tempfile, time

CLONE = os.environ.get("RWR_SYSTEM_DEPLOY_CLONE", "/tmp/rwr_system_deploy")  # clone of the deploy branch
URL, BRANCH = "git@github.com:rwr-fast/rwr_system.git", "deploy-policies-2026-10-31"
GIT_ID = ["-c", "user.name=Roman-089", "-c", "user.email=209470582+Roman-089@users.noreply.github.com"]


def sh(*cmd, cwd=CLONE, retries=1):
    for i in range(retries):
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout
        time.sleep(6)
    raise RuntimeError(f"{' '.join(cmd)}: {r.stderr.strip()[-400:]}")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    spec = json.load(open(sys.argv[1]))
    push = "--push" in sys.argv
    sh("git", "lfs", "install", "--local")
    sh("git", "fetch", "-q", "--depth", "1", "origin", BRANCH, retries=6)
    sh("git", "reset", "-q", "--hard", f"origin/{BRANCH}")
    dp = os.path.join(CLONE, "deploy_policies")
    name = spec["name"]
    dst = os.path.join(dp, name)
    if os.path.exists(dst):
        raise SystemExit(f"{dst} exists: never overwrite a policy folder")
    os.makedirs(dst)
    exp = os.path.join(spec["run_dir"], "exported")
    shutil.copy(os.path.join(exp, "policy_history.pt"), os.path.join(dst, "policy.pt"))
    shutil.copy(os.path.join(exp, "policy_history.json"), os.path.join(dst, "policy.json"))
    for f in spec.get("metrics_files", []):
        shutil.copy(f, dst)
    digest = sha(os.path.join(dst, "policy.pt"))
    pj = json.load(open(os.path.join(dst, "policy.json")))
    assert pj["sha256"] == digest and pj["obs_dim"] == 74 and pj["action_dim"] == 17, "export metadata mismatch"
    import torch
    m = torch.jit.load(os.path.join(dst, "policy.pt"), map_location="cpu").eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 74))
    assert out.shape == (1, 17) and torch.isfinite(out).all(), "self-check failed"
    hist = pj["history_length"]
    if hist > 0:
        raise SystemExit("history > 0: needs the patched node in tools/ (not automated here)")
    man_p = os.path.join(dp, "manifest.json")
    man = json.load(open(man_p))
    tmpl = dict(man["policies"][0])
    isaac = spec["isaac"]
    entry = {k: tmpl[k] for k in tmpl}
    entry.update(rank=spec["rank"], folder=name, kind=spec["kind"], sha256=digest, history_length=hist,
                 needs_reset_line=False, goal_ang_vel=1.5, goal_flip_period_s=0.0, training_run=spec["training_run"],
                 checkpoint=spec["checkpoint"], task=spec["task"], description=spec["what"], isaac=isaac,
                 sim_wrist_speed=spec["sim_wrist_speed"])
    entry["real_hand_2026-10-31"] = "not run yet"
    pols = [p for p in man["policies"]]
    for p in pols:
        if p["rank"] >= spec["rank"]:
            p["rank"] += 1
    pols.append(entry)
    man["policies"] = sorted(pols, key=lambda p: p["rank"])
    json.dump(man, open(man_p, "w"), indent=1)
    rows = [f"| {k.replace('_', ' ')} | {v['rpm']:.2f} | {v['dfsr']:.1f} % | {v['dc']:.3f} |" for k, v in isaac.items()]
    open(os.path.join(dst, "SPEC.md"), "w").write(f"""# {name}

**{spec['kind']}** (rank {spec['rank']} in this folder)

{spec['what']}

| | |
|---|---|
| File | `policy.pt`, sha256 `{digest}` |
| Format | TorchScript, `forward(obs[1, 74]) -> actions[1, 17]`, observation normalizer inside |
| Interface | the course default `policy.pt` interface: 74 obs, 17 actions, 20 Hz, EMA alpha 0.2 in the node, memoryless (history 0, no `reset()` line) |
| Direction | +z only |
| `goal_ang_vel` | **1.5** (flip period 0) |
| Training | run `{spec['training_run']}`, checkpoint `{spec['checkpoint']}`, task `{spec['task']}` |
| Sim wrist speed | {spec['sim_wrist_speed']} |

## Isaac evaluation

| Setting | RPM | drop-free | drops / trial |
|---|---|---|---|
{chr(10).join(rows)}

## Real hand (ORCA v2 right, flat hand + -30 deg wrist offset relay)

not run yet

## Run

```bash
cd deploy_policies
./run_policy.sh {name}             # flat hand: starts the -30 deg wrist relay (see README)
./run_policy.sh {name} --tilted    # hand mounted 30 deg tilted as in training: no relay
```
""")
    # README: insert the row at its rank, renumber the "Which one" table
    rd_p = os.path.join(dp, "README.md")
    lines = open(rd_p).read().split("\n")
    i0 = next(i for i, l in enumerate(lines) if l.startswith("| Rank | Folder |"))
    i = i0 + 2
    body = []
    while i < len(lines) and lines[i].startswith("|"):
        body.append(lines[i]); i += 1
    new = f"| {spec['rank']} | `{name}` | {spec['readme_what']} | {spec['readme_isaac']} | not run yet |"
    body.insert(spec["rank"] - 1, new)
    renum = []
    for r, l in enumerate(body, 1):
        cells = l.split("|")
        cells[1] = f" **{r}** " if r == 1 else f" {r} "
        if r == 1 and not cells[2].strip().startswith("**"):
            cells[2] = f" **{cells[2].strip()}** "
        if r != 1 and cells[2].strip().startswith("**"):
            cells[2] = " " + cells[2].strip().strip("*") + " "
        renum.append("|".join(cells))
    lines[i0 + 2:i] = renum
    if spec.get("suggested_order"):  # replace the first sentence of the "Suggested order on the hand" line
        j = next(k for k, l in enumerate(lines) if l.startswith("Suggested order on the hand:"))
        rest = lines[j].split(". ", 1)[1] if ". " in lines[j] else ""
        lines[j] = f"Suggested order on the hand: {spec['suggested_order']}." + (f" {rest}" if rest else "")
    open(rd_p, "w").write("\n".join(lines))
    sh("git", "add", "-A", "deploy_policies")
    sh("git", *GIT_ID, "commit", "-q", "-m", spec["commit_msg"])
    lfs = sh("git", "lfs", "ls-files")
    assert f"deploy_policies/{name}/policy.pt" in lfs, "policy.pt is not an LFS object"
    print("prepared commit:", sh("git", "log", "--oneline", "-1").strip())
    print(sh("git", "show", "--stat", "HEAD").strip()[-600:])
    if not push:
        print("dry run: not pushed")
        return
    sh("git", "push", "-q", "origin", f"HEAD:{BRANCH}", retries=8)
    # verify in a fresh shallow clone
    with tempfile.TemporaryDirectory() as tmp:
        sh("git", "clone", "-q", "--depth", "1", "-b", BRANCH, URL, tmp, cwd="/tmp", retries=8)
        sh("git", "lfs", "pull", "--include", f"deploy_policies/{name}/policy.pt", cwd=tmp, retries=6)
        man2 = json.load(open(os.path.join(tmp, "deploy_policies", "manifest.json")))
        bad = []
        for p in man2["policies"]:
            f = os.path.join(tmp, "deploy_policies", p["folder"], "policy.pt")
            if p["folder"] == name:
                ok = os.path.getsize(f) > 1000 and sha(f) == p["sha256"]
                print(f"verify {name}: size {os.path.getsize(f)} sha match {sha(f) == p['sha256']}")
                if not ok:
                    bad.append(name)
        if bad:
            raise SystemExit(f"verification failed: {bad}")
    print("pushed and verified")


if __name__ == "__main__":
    main()
