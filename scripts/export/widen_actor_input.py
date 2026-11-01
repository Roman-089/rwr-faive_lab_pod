"""Widen a trained memoryless MLP policy's input for new observations appended at the end (Subtask 2: + tracked cube pose).

The new columns of the actor's first layer start at zero, so the widened policy acts exactly like the old one until
training gives the new inputs weight (net2net-style function-preserving growth). The actor's observation normalizer gets
prior statistics for the new inputs (it updates them during training like the old ones); the Adam moments of the first
layer get zeros for the new columns, so the checkpoint resumes with its optimizer state. The critic is unchanged.

Default new inputs (tracker_obs.TrackedCubePose, 8): object position in the env frame (mean = where the 06 line holds
the cube, std 1 cm), symmetry-canonical quaternion (w ~ 0.95, x/y/z ~ 0 with std 0.15 / 0.15 / 0.35 for the spin about z),
tracked flag (mean 0.9, std 0.3).

Usage: python scripts/export/widen_actor_input.py IN.pt OUT.pt [--hold_pos X Y Z]
Resume with --resume True --load_run <OUT's run dir> --checkpoint <OUT's name> on a -cube task.
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--hold_pos", type=float, nargs=3, default=(-0.01, 0.01, -0.02),
                    help="prior mean of the cube position input (env frame, m); default: where the 06 line holds the cube "
                         "(palm-circle centre in x / y, resting height in z)")
    args = ap.parse_args()

    d = torch.load(args.inp, map_location="cpu", weights_only=False)
    sd = d["model_state_dict"]
    w = sd["actor.0.weight"]
    n_old = w.shape[1]
    mean = torch.tensor([*args.hold_pos, 0.95, 0.0, 0.0, 0.0, 0.9])
    std = torch.tensor([0.01, 0.01, 0.01, 0.05, 0.15, 0.15, 0.35, 0.3])
    n_new = mean.numel()
    sd["actor.0.weight"] = torch.cat((w, torch.zeros(w.shape[0], n_new, dtype=w.dtype)), dim=1)
    if "actor_obs_normalizer._mean" in sd:
        sd["actor_obs_normalizer._mean"] = torch.cat((sd["actor_obs_normalizer._mean"], mean[None]), dim=1)
        sd["actor_obs_normalizer._var"] = torch.cat((sd["actor_obs_normalizer._var"], (std**2)[None]), dim=1)
        sd["actor_obs_normalizer._std"] = torch.cat((sd["actor_obs_normalizer._std"], std[None]), dim=1)
    opt = d.get("optimizer_state_dict")
    widened = 0
    if opt is not None:
        for st in opt["state"].values():
            for k in ("exp_avg", "exp_avg_sq"):
                t = st.get(k)
                if t is not None and tuple(t.shape) == (w.shape[0], n_old):
                    st[k] = torch.cat((t, torch.zeros(t.shape[0], n_new, dtype=t.dtype)), dim=1)
                    widened += 1
    assert opt is None or widened == 2, f"expected to widen exp_avg + exp_avg_sq of one (512, {n_old}) tensor, did {widened}"
    torch.save(d, args.out)
    print(f"{args.inp} -> {args.out}: actor input {n_old} -> {n_old + n_new} (new columns zero), "
          f"normalizer {'widened' if 'actor_obs_normalizer._mean' in sd else 'absent'}, Adam moments widened {widened}")


if __name__ == "__main__":
    main()
