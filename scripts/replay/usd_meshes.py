"""Export the visual meshes of the hand links and the cube size from the USD stage (for offline MuJoCo replay).

Import only after the Isaac Sim app has been launched (needs pxr / omni.usd).
"""

import numpy as np

import omni.usd
from pxr import Usd, UsdGeom


def mesh_in_frame(mesh: UsdGeom.Mesh, frame_to_world: np.ndarray, time) -> tuple[np.ndarray, np.ndarray]:
    """Vertices of a USD mesh expressed in a parent frame, and its faces fan-triangulated."""
    pts = np.asarray(mesh.GetPointsAttr().Get(time), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(time))
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(time))
    # USD uses row vectors: p_world = p_local @ M
    mesh_to_world = np.array(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(time))
    rel = mesh_to_world @ np.linalg.inv(frame_to_world)
    pts = np.c_[pts, np.ones(len(pts))] @ rel
    tris = []
    start = 0
    for c in counts:
        for k in range(1, c - 1):
            tris.append((idx[start], idx[start + k], idx[start + k + 1]))
        start += c
    return pts[:, :3].astype(np.float32), np.asarray(tris, dtype=np.int32)


def export_meshes(link_paths: list[str]) -> dict[str, np.ndarray]:
    """Visual meshes of every link, in that link's frame. Collision meshes are skipped."""
    stage = omni.usd.get_context().get_stage()
    time = Usd.TimeCode.Default()
    out = {}
    for li, path in enumerate(link_paths):
        link = stage.GetPrimAtPath(path)
        link_to_world = np.array(UsdGeom.Xformable(link).ComputeLocalToWorldTransform(time))
        verts, faces, n = [], [], 0
        for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            p = str(prim.GetPath())
            if prim != link and p in link_paths:  # a nested link has its own entry
                continue
            if not prim.IsA(UsdGeom.Mesh) or "collision" in p.lower():
                continue
            if UsdGeom.Imageable(prim).ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
                continue
            v, f = mesh_in_frame(UsdGeom.Mesh(prim), link_to_world, time)
            verts.append(v)
            faces.append(f + n)
            n += len(v)
        if verts:
            out[f"mesh_verts_{li}"] = np.concatenate(verts)
            out[f"mesh_faces_{li}"] = np.concatenate(faces)
            print(f"[replay] link {path}: {n} verts, {len(out[f'mesh_faces_{li}'])} tris")
        else:
            print(f"[replay] link {path}: no visual mesh")
    return out


def object_size(prim_path: str) -> float:
    """Edge length of the (cube) object from its USD bounding box."""
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    box = cache.ComputeUntransformedBound(stage.GetPrimAtPath(prim_path)).ComputeAlignedRange()
    size = np.array(box.GetSize())
    xf = np.array(UsdGeom.Xformable(stage.GetPrimAtPath(prim_path)).GetLocalTransformation())
    scale = np.linalg.norm(xf[:3, :3], axis=1)
    return float(np.max(size * scale))


