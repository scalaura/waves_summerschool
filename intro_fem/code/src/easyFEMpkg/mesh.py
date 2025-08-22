from __future__ import annotations
import numpy as np
from typing import Tuple
try:
    import meshpy.triangle as triangle
except Exception:
    triangle = None

def build_mesh(geometry: str = "square", max_area: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from .geometry import make_geometry
    if triangle is None:
        raise RuntimeError("meshpy.triangle not available. Please install 'meshpy'.")
    pts, facets = make_geometry(geometry)
    info = triangle.MeshInfo()
    info.set_points(pts.tolist())
    info.set_facets(facets.tolist())
    # robust API
    try:
        mesh = triangle.build(info, min_angle=30.0, max_volume=max_area)
    except TypeError:
        mesh = triangle.build(info, min_angle=30.0, max_area=max_area)

    p = np.asarray(mesh.points, float)
    t = np.asarray(mesh.elements, int)

    # boundary edges = edges appearing exactly once
    edges = {}
    for tri in t:
        for a,b in [(tri[0],tri[1]), (tri[1],tri[2]), (tri[2],tri[0])]:
            i,j = (a,b) if a<b else (b,a)
            edges[(i,j)] = edges.get((i,j), 0) + 1
    boundary_edges = np.array([[i,j] for (i,j),c in edges.items() if c==1], dtype=int)

    return p, t, boundary_edges
