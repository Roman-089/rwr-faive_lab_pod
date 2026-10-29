"""Extract the ORCA v2 hand as Isaac Lab simulates it from its USD into a small JSON for the MuJoCo sim2sim scene.

What is extracted (everything PhysX uses, nothing else): per link the mass, center of mass, principal inertia and
principal axes (physics:MassAPI); per joint the two local joint frames, axis and limits (physics:RevoluteJoint /
FixedJoint); per link the convex collision pieces (physics:approximation = convexHull, 64-vertex hulls), as their
unique vertices in the link frame. Drive gains, friction and effort limits are NOT taken from the USD: Isaac Lab
overwrites them from source/faive_lab/faive_lab/assets/orca_v2.py, which scene.py reads directly.

Needs pxr, which on the cluster only exists inside Isaac Sim. The bundled USD libraries work on a CPU login node:

    L=$CONDA_PREFIX/lib/python3.11/site-packages/isaacsim/extscache/omni.usd.libs-1.0.1+69cbf6ad.lx64.r.cp311
    PYTHONPATH=$L LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$L/bin:$L/bin/deps python scripts/sim2sim/extract_usd_model.py

(conda env ``rwr``). Rerun after the USD changes; the committed JSON records the sha256 of the USD layers it came from,
and scene.py warns when they no longer match.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

REPO_ROOT = Path(__file__).resolve().parents[2]
USD_PATH = REPO_ROOT / "source/faive_lab/faive_lab/assets/data/orca_v2/orcahand_right.usd"
OUT_PATH = Path(__file__).resolve().parent / "orca_v2_right_usd.json"


def quat_wxyz(q) -> list[float]:
    return [float(q.GetReal()), *map(float, q.GetImaginary())]


def vec(v) -> list[float]:
    return [float(x) for x in v]


def usd_layers(path: Path) -> dict[str, str]:
    """sha256 of the USD file and its configuration/ layers (what the JSON was extracted from)."""
    files = [path, *sorted((path.parent / "configuration").glob("*.usd"))]
    return {str(f.relative_to(REPO_ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def link_of(prim: Usd.Prim) -> Usd.Prim:
    while not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        prim = prim.GetParent()
    return prim


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--usd", type=Path, default=USD_PATH)
    p.add_argument("--out", type=Path, default=OUT_PATH)
    args = p.parse_args()

    stage = Usd.Stage.Open(str(args.usd))
    t = Usd.TimeCode.Default()
    links, joints, hulls = {}, [], {}
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            m = UsdPhysics.MassAPI(prim)
            world = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(t)).T  # column vectors
            links[prim.GetName()] = {
                "mass": float(m.GetMassAttr().Get()),
                "com": vec(m.GetCenterOfMassAttr().Get()),
                "diag_inertia": vec(m.GetDiagonalInertiaAttr().Get()),
                "principal_axes": quat_wxyz(m.GetPrincipalAxesAttr().Get()),
                "usd_world_pos": world[:3, 3].tolist(),  # stage pose at q = 0, for a sanity check of the tree
            }
        elif prim.IsA(UsdPhysics.Joint):
            j = UsdPhysics.Joint(prim)
            body0, body1 = j.GetBody0Rel().GetTargets(), j.GetBody1Rel().GetTargets()
            entry = {
                "name": prim.GetName(),
                "type": "revolute" if prim.IsA(UsdPhysics.RevoluteJoint) else "fixed",
                "parent": body0[0].name if body0 else None,
                "child": body1[0].name,
                "pos0": vec(j.GetLocalPos0Attr().Get()),
                "rot0": quat_wxyz(j.GetLocalRot0Attr().Get()),
                "pos1": vec(j.GetLocalPos1Attr().Get()),
                "rot1": quat_wxyz(j.GetLocalRot1Attr().Get()),
            }
            if entry["type"] == "revolute":
                r = UsdPhysics.RevoluteJoint(prim)
                entry["axis"] = r.GetAxisAttr().Get() or "X"
                entry["limits_deg"] = [float(r.GetLowerLimitAttr().Get()), float(r.GetUpperLimitAttr().Get())]
            elif not prim.IsA(UsdPhysics.FixedJoint):
                raise ValueError(f"unsupported joint type {prim.GetTypeName()} at {prim.GetPath()}")
            joints.append(entry)
        elif prim.HasAPI(UsdPhysics.CollisionAPI):
            approx = prim.GetAttribute("physics:approximation").Get()
            if approx != "convexHull":
                raise ValueError(f"{prim.GetPath()}: approximation {approx}, only convexHull is supported")
            link = link_of(prim)
            link_world = np.array(UsdGeom.Xformable(link).ComputeLocalToWorldTransform(t))
            pts = []
            for mesh in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
                if mesh.IsA(UsdGeom.Mesh):
                    mesh_world = np.array(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(t))
                    v = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(t), dtype=np.float64)
                    pts.append((np.c_[v, np.ones(len(v))] @ (mesh_world @ np.linalg.inv(link_world)))[:, :3])
            v = np.unique(np.round(np.concatenate(pts), 7), axis=0)  # meshes are unindexed triangle soups
            # name: the piece's Xform below "collisions" (e.g. R_Carpals_heel); repeated names get the link prefix
            piece = prim.GetPath().pathString.split("/collisions/")[1].split("/")[0]
            hulls.setdefault(link.GetName(), []).append({"name": piece, "vertices": np.round(v, 7).tolist()})

    out = {
        "source": {"usd": str(args.usd.relative_to(REPO_ROOT)), "sha256": usd_layers(args.usd)},
        "units": "m, kg, kg m^2, quaternions w x y z, joint limits in degrees, link frame = USD link prim",
        "links": links,
        "joints": joints,
        "collision_hulls": hulls,
    }
    args.out.write_text(json.dumps(out, indent=1))
    n_hulls = sum(len(h) for h in hulls.values())
    print(f"wrote {args.out}: {len(links)} links, {len(joints)} joints, {n_hulls} collision hulls")


if __name__ == "__main__":
    main()
