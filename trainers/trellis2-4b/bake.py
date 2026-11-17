"""Bake a GLB from a saved TRELLIS.2 mesh, in its own process so the handler can stop it.

o_voxel.postprocess.to_glb (remesh, UV unwrap, texture bake) usually takes 40–110 s, but on some shapes
it runs for tens of minutes. A CUDA / C++ call cannot be interrupted from Python, so the handler runs it
here, under a time limit, and kills the process when it is exceeded.

usage: python bake.py <mesh.pt> <out.glb> <decimation_target> <texture_size> <remesh 0|1>
Prints one JSON line {"faces": int, "vertices": int} on success.
"""
from __future__ import annotations

import json
import sys


def main() -> None:
    src, out, faces, texture, remesh = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] == "1"
    import torch
    import o_voxel

    m = torch.load(src, map_location="cuda", weights_only=False)
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
    print(json.dumps({"faces": int(len(glb.faces)), "vertices": int(len(glb.vertices))}), flush=True)


if __name__ == "__main__":
    main()
