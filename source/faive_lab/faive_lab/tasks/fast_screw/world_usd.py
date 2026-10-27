# Owner: X (Task X / FAST)
"""Procedural USD of the screw world (geometry from world.py) – generated at import, no binary asset in git.

Articulation 'fixture' (fixed to the world):
    base   = plate + bolt shaft (static colliders)
      └─ nut_lift  prismatic along z, 0 .. nut_travel   (driven: target = thread height, see mdp.ScrewCommand)
          carrier  (massless helper, no collider)
            └─ nut_spin  revolute about z, unlimited    (passive: thread friction = damping + friction)
                nut      hex prism collider

Real thread contact is not simulated (far too expensive for thousands of envs). Instead the nut height is
coupled to its angle in software every control step: dz = d(angle) * pitch / 2pi. Self-collisions inside the
fixture are off, so the nut collider may overlap the bolt shaft; the fingers collide with both.
"""
from __future__ import annotations

import hashlib
import math
import os
import tempfile

from .world import WorldGeometry


def _hex_prism(stage, path, across_flats, height):
    from pxr import Gf, UsdGeom

    r = across_flats / math.sqrt(3)
    pts, idx, cnt = [], [], []
    for z in (-height / 2, height / 2):
        for k in range(6):
            a = k * math.pi / 3
            pts.append(Gf.Vec3f(r * math.cos(a), r * math.sin(a), z))
    for k in range(6):
        n = (k + 1) % 6
        idx += [k, n, 6 + n, 6 + k]
        cnt.append(4)
    idx += list(range(5, -1, -1)); cnt.append(6)
    idx += list(range(6, 12)); cnt.append(6)
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(pts)
    m.CreateFaceVertexIndicesAttr(idx)
    m.CreateFaceVertexCountsAttr(cnt)
    return m


def write_world_usd(path: str, g: WorldGeometry = WorldGeometry()) -> str:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/fixture")
    stage.SetDefaultPrim(root.GetPrim())

    # base: plate + bolt shaft, welded to the world
    base = UsdGeom.Xform.Define(stage, "/fixture/base")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.MassAPI.Apply(base.GetPrim()).CreateMassAttr(0.1)
    plate = UsdGeom.Cube.Define(stage, "/fixture/base/plate")
    plate.CreateSizeAttr(1.0)
    plate.AddTranslateOp().Set(Gf.Vec3d(0, 0, g.plate_t / 2))
    plate.AddScaleOp().Set(Gf.Vec3f(g.plate_x, g.plate_y, g.plate_t))
    UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
    shaft = UsdGeom.Cylinder.Define(stage, "/fixture/base/bolt_shaft")
    shaft.CreateAxisAttr("Z")
    shaft.CreateRadiusAttr(g.bolt_d / 2)
    shaft.CreateHeightAttr(g.bolt_above_plate)
    shaft.AddTranslateOp().Set(Gf.Vec3d(0, 0, g.plate_t + g.bolt_above_plate / 2))
    UsdPhysics.CollisionAPI.Apply(shaft.GetPrim())
    world_joint = UsdPhysics.FixedJoint.Define(stage, "/fixture/world_joint")
    world_joint.CreateBody1Rel().SetTargets([base.GetPath()])
    UsdPhysics.ArticulationRootAPI.Apply(world_joint.GetPrim())   # same layout as orcahand_right.usd

    # carrier (lift DOF)
    carrier = UsdGeom.Xform.Define(stage, "/fixture/carrier")
    carrier.AddTranslateOp().Set(Gf.Vec3d(0, 0, g.nut_z_min))
    UsdPhysics.RigidBodyAPI.Apply(carrier.GetPrim())
    mc = UsdPhysics.MassAPI.Apply(carrier.GetPrim())
    mc.CreateMassAttr(0.001)
    mc.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-8, 1e-8, 1e-8))
    lift = UsdPhysics.PrismaticJoint.Define(stage, "/fixture/nut_lift")
    lift.CreateBody0Rel().SetTargets([base.GetPath()])
    lift.CreateBody1Rel().SetTargets([carrier.GetPath()])
    lift.CreateAxisAttr("Z")
    lift.CreateLocalPos0Attr(Gf.Vec3f(0, 0, g.nut_z_min))
    lift.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    lift.CreateLowerLimitAttr(0.0)
    lift.CreateUpperLimitAttr(g.nut_travel)
    UsdPhysics.DriveAPI.Apply(lift.GetPrim(), "linear")

    # nut (spin DOF)
    nut = UsdGeom.Xform.Define(stage, "/fixture/nut")
    nut.AddTranslateOp().Set(Gf.Vec3d(0, 0, g.nut_z_min))
    UsdPhysics.RigidBodyAPI.Apply(nut.GetPrim())
    UsdPhysics.MassAPI.Apply(nut.GetPrim()).CreateMassAttr(g.nut_mass)
    hexm = _hex_prism(stage, "/fixture/nut/hex", g.nut_af, g.nut_h)
    UsdPhysics.CollisionAPI.Apply(hexm.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(hexm.GetPrim()).CreateApproximationAttr("convexHull")
    UsdPhysics.FilteredPairsAPI.Apply(nut.GetPrim()).CreateFilteredPairsRel().SetTargets([base.GetPath()])
    spin = UsdPhysics.RevoluteJoint.Define(stage, "/fixture/nut_spin")
    spin.CreateBody0Rel().SetTargets([carrier.GetPath()])
    spin.CreateBody1Rel().SetTargets([nut.GetPath()])
    spin.CreateAxisAttr("Z")
    spin.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    spin.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    UsdPhysics.DriveAPI.Apply(spin.GetPrim(), "angular")

    stage.GetRootLayer().customLayerData = {"generator": "faive_lab.tasks.fast_screw.world_usd",
                                            "geometry": repr(g.asdict())}
    stage.GetRootLayer().Save()
    return path


def world_usd_path(g: WorldGeometry = WorldGeometry()) -> str:
    key = hashlib.sha1(repr(g.asdict()).encode()).hexdigest()[:10]
    folder = os.path.join(tempfile.gettempdir(), "faive_lab_fast")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"screw_world_{key}.usda")
    if not os.path.exists(path):
        write_world_usd(path, g)
    return path
