"""Bake a GLB from a saved TRELLIS.2 mesh, in its own process so the handler can stop it.

o_voxel.postprocess.to_glb (remesh, UV unwrap, texture bake) usually takes 40–110 s, but on some shapes
(fragmented ones: hair, fur) it runs for tens of minutes. A CUDA / C++ call cannot be interrupted from
Python, so the handler runs it here, under a time limit, and kills the process when it is exceeded.

usage: python bake.py <mesh.pt> <out.glb> <decimation_target> <texture_size> <mode>
  mode  remesh   the official bake, as app.py (to_glb, remesh=True)
        plain    to_glb without the remesh
        vertex   no UV unwrap and no texture: the simplified mesh with its colour per vertex — the last
                 resort, it always finishes in seconds
Prints one JSON line {"faces": int, "vertices": int} on success.
"""
from __future__ import annotations

import json
import sys


def bake_textured(m, out: str, faces: int, texture: int, remesh: bool) -> dict:
    import o_voxel

    glb = o_voxel.postprocess.to_glb(
        vertices=m["vertices"],
        faces=m["faces"],
        attr_volume=m["attrs"],
        coords=m["coords"],
        attr_layout=m["layout"],
        voxel_size=m["voxel_size"],
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=faces,
        texture_size=texture,
        remesh=remesh,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(out, extension_webp=True)
    return {"faces": int(len(glb.faces)), "vertices": int(len(glb.vertices))}


def bake_vertex_colors(m, out: str, faces: int) -> dict:
    import cumesh
    import numpy as np
    import torch
    import trimesh
    from flex_gemm.ops.grid_sample import grid_sample_3d

    mesh = cumesh.CuMesh()
    mesh.init(m["vertices"].cuda(), m["faces"].cuda())
    print("vertex bake: simplifying", flush=True)
    mesh.simplify(faces, verbose=True)
    mesh.remove_duplicate_faces()
    mesh.remove_small_connected_components(1e-5)
    mesh.compute_vertex_normals()
    vertices, tris = mesh.read()
    normals = mesh.read_vertex_normals()

    print("vertex bake: sampling colours", flush=True)
    attrs, coords, layout = m["attrs"].cuda(), m["coords"].cuda(), m["layout"]
    voxel = float(m["voxel_size"])
    res = int(round(1.0 / voxel))
    sampled = grid_sample_3d(
        attrs,
        torch.cat([torch.zeros_like(coords[:, :1]), coords], dim=-1),
        shape=torch.Size([1, attrs.shape[1], res, res, res]),
        grid=((vertices.float() + 0.5) / voxel).reshape(1, -1, 3),
        mode="trilinear",
    ).reshape(-1, attrs.shape[1]).float()
    base = (sampled[:, layout["base_color"]].clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy()
    rgba = np.concatenate([base, np.full((len(base), 1), 255, np.uint8)], axis=1)

    v = vertices.float().cpu().numpy()
    n = normals.float().cpu().numpy()
    # Same axes as to_glb: TRELLIS z-up → glTF y-up.
    v[:, 1], v[:, 2] = v[:, 2].copy(), -v[:, 1].copy()
    n[:, 1], n[:, 2] = n[:, 2].copy(), -n[:, 1].copy()
    out_mesh = trimesh.Trimesh(vertices=v, faces=tris.cpu().numpy(), vertex_normals=n, vertex_colors=rgba, process=False)
    out_mesh.export(out, file_type="glb")
    return {"faces": int(len(out_mesh.faces)), "vertices": int(len(out_mesh.vertices))}


def main() -> None:
    src, out, faces, texture, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    import torch

    m = torch.load(src, map_location="cuda", weights_only=False)
    if mode == "vertex":
        counts = bake_vertex_colors(m, out, faces)
    else:
        counts = bake_textured(m, out, faces, texture, remesh=(mode == "remesh"))
    print(json.dumps(counts), flush=True)


if __name__ == "__main__":
    main()
