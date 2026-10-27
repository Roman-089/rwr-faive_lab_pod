# Owner: X (Task X / FAST)
"""Procedural USD for the screw fixture: a fixed base + a bolt on a continuous revolute joint.

Real threads (helical contact) are not simulated – far too expensive for thousands of parallel envs.
The bolt is a rigid body that can only spin about its axis; thread friction is modelled by the joint's
damping/friction (actuator 'thread' in the env cfg, randomized per episode). The axial travel of a real
thread (e.g. M8: 1.25 mm/rev) is small and is tracked as 'progress' = angle * pitch in the reward/metrics.

The file is generated at import time (no binary asset in git, no LFS) into a cache folder.

Frames: bolt link origin = centre of the bolt head; bolt axis = local +X (head on +X side, shaft on -X).
"""
from __future__ import annotations

import hashlib
import math
import os
import tempfile
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class BoltGeometry:
    head_across_flats_m: float = 0.013      # M8 hex head
    head_height_m: float = 0.0055
    shaft_radius_m: float = 0.004
    shaft_length_m: float = 0.02
    bolt_mass_kg: float = 0.015
    base_size_m: float = 0.03               # fixture block the shaft sits in


def _hex_prism(stage, path, across_flats, height):
    from pxr import Gf, UsdGeom

    r = across_flats / math.sqrt(3)         # circumradius
    pts, idx, cnt = [], [], []
    for x in (-height / 2, height / 2):
        for k in range(6):
            a = math.pi / 6 + k * math.pi / 3
            pts.append(Gf.Vec3f(x, r * math.cos(a), r * math.sin(a)))
    for k in range(6):                       # side quads
        n = (k + 1) % 6
        idx += [k, n, 6 + n, 6 + k]
        cnt.append(4)
    idx += list(range(5, -1, -1)); cnt.append(6)   # -x cap
    idx += list(range(6, 12)); cnt.append(6)       # +x cap
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateFaceVertexCountsAttr(cnt)
    return m


def write_bolt_usd(path: str, g: BoltGeometry = BoltGeometry()) -> str:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/fixture")
    stage.SetDefaultPrim(root.GetPrim())

    # base block, welded to the world
    base = UsdGeom.Xform.Define(stage, "/fixture/base")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.MassAPI.Apply(base.GetPrim()).CreateMassAttr(0.2)
    bx = UsdGeom.Cube.Define(stage, "/fixture/base/collision")
    bx.CreateSizeAttr(1.0)
    off = g.shaft_length_m + g.head_height_m / 2 + g.base_size_m / 2
    bx.AddTranslateOp().Set(Gf.Vec3d(-off, 0, 0))
    bx.AddScaleOp().Set(Gf.Vec3f(g.base_size_m, g.base_size_m, g.base_size_m))
    UsdPhysics.CollisionAPI.Apply(bx.GetPrim())
    world_fix = UsdPhysics.FixedJoint.Define(stage, "/fixture/world_joint")
    world_fix.CreateBody1Rel().SetTargets([base.GetPath()])
    # fixed-base articulation: root API on the world joint, same layout as orcahand_right.usd
    UsdPhysics.ArticulationRootAPI.Apply(world_fix.GetPrim())

    # bolt: hex head (convex hull) + cylindrical shaft along -X
    bolt = UsdGeom.Xform.Define(stage, "/fixture/bolt")
    UsdPhysics.RigidBodyAPI.Apply(bolt.GetPrim())
    UsdPhysics.MassAPI.Apply(bolt.GetPrim()).CreateMassAttr(g.bolt_mass_kg)
    head = _hex_prism(stage, "/fixture/bolt/head", g.head_across_flats_m, g.head_height_m)
    UsdPhysics.CollisionAPI.Apply(head.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(head.GetPrim()).CreateApproximationAttr("convexHull")
    shaft = UsdGeom.Cylinder.Define(stage, "/fixture/bolt/shaft")
    shaft.CreateAxisAttr("X")
    shaft.CreateRadiusAttr(g.shaft_radius_m)
    shaft.CreateHeightAttr(g.shaft_length_m)
    shaft.AddTranslateOp().Set(Gf.Vec3d(-(g.head_height_m + g.shaft_length_m) / 2, 0, 0))
    UsdPhysics.CollisionAPI.Apply(shaft.GetPrim())

    # continuous revolute joint about X (the bolt axis), anchored at the head centre
    j = UsdPhysics.RevoluteJoint.Define(stage, "/fixture/bolt_joint")
    j.CreateBody0Rel().SetTargets([base.GetPath()])
    j.CreateBody1Rel().SetTargets([bolt.GetPath()])
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(off, 0, 0))      # head centre in the base frame
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    # no lower/upper limit -> continuous
    UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")  # gains set by the env's 'thread' actuator

    stage.GetRootLayer().customLayerData = {"generator": "faive_lab.tasks.fast_screw.bolt_usd", **{k: v for k, v in asdict(g).items()}}
    stage.GetRootLayer().Save()
    return path


def bolt_usd_path(g: BoltGeometry = BoltGeometry()) -> str:
    """Generate (once) and return the cached USD path for this geometry."""
    key = hashlib.sha1(repr(asdict(g)).encode()).hexdigest()[:10]
    folder = os.path.join(tempfile.gettempdir(), "faive_lab_fast")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"bolt_{key}.usda")
    if not os.path.exists(path):
        write_bolt_usd(path, g)
    return path


if __name__ == "__main__":
    print(bolt_usd_path())
