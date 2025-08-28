from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

try:
    import meshpy.triangle as triangle
except Exception:
    triangle = None


def _rect_points_facets(xmin: float, xmax: float, ymin: float, ymax: float):
    pts = np.array([[xmin, ymin],
                    [xmax, ymin],
                    [xmax, ymax],
                    [xmin, ymax]], dtype=float)
    facets = np.array([[0,1],[1,2],[2,3],[3,0]], dtype=int)
    return pts, facets


def _build_pslg_mesh(pts: np.ndarray, facets: np.ndarray, max_area: float):
    if triangle is None:
        raise RuntimeError("meshpy.triangle not available. Please install 'meshpy'.")
    info = triangle.MeshInfo()
    info.set_points(pts.tolist())
    info.set_facets(facets.tolist())
    # robust API across meshpy builds
    try:
        mesh = triangle.build(info, min_angle=30.0, max_volume=max_area)
    except TypeError:
        mesh = triangle.build(info, min_angle=30.0, max_area=max_area)
    p = np.asarray(mesh.points, float)      # (N,2)
    t = np.asarray(mesh.elements, int)      # (M,3)

    # boundary edges = edges that appear exactly once among all triangle edges
    edges = {}
    for tri in t:
        for a,b in ((tri[0],tri[1]), (tri[1],tri[2]), (tri[2],tri[0])):
            i,j = (a,b) if a<b else (b,a)
            edges[(i,j)] = edges.get((i,j), 0) + 1
    boundary_edges = np.array([[i,j] for (i,j), c in edges.items() if c == 1], dtype=int)
    return p, t, boundary_edges


def build_mesh(geometry: str = "square", max_area: float = 0.05, box=(-1,1,-1,1)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backwards-compatible builder for named geometries.
    geometry ∈ {"square", "circle", "lshape"} (names without trailing 'g').
    For arbitrary squares / PML use build_square_mesh or build_square_with_pml.
    """
    geom = geometry.lower()
    if geom == "square":
        xi, xa, yi, ya = box
        pts, facets = _rect_points_facets(xi, xa, yi, ya)
        return _build_pslg_mesh(pts, facets, max_area)
    elif geom == "circle":
        # polygonal circle
        num = 96
        ang = np.linspace(0.0, 2.0*np.pi, num, endpoint=False)
        pts = np.c_[np.cos(ang), np.sin(ang)].astype(float)
        facets = np.c_[np.arange(num), (np.arange(num)+1) % num].astype(int)
        return _build_pslg_mesh(pts, facets, max_area)
    elif geom == "lshape":
        pts = np.array([[0,0],[1,0],[1,1],[0.5,1],[0.5,0.5],[0,0.5]], float)
        facets = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]], int)
        return _build_pslg_mesh(pts, facets, max_area)
    else:
        raise ValueError(f"Unknown geometry '{geometry}'. Use 'square', 'circle', or 'lshape'.")


def build_square_mesh(xmin: float, xmax: float, ymin: float, ymax: float,
                      max_area: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mesh an arbitrary axis-aligned square/rectangle.
    Returns (points, triangles, boundary_edges).
    """
    pts, facets = _rect_points_facets(xmin, xmax, ymin, ymax)
    return _build_pslg_mesh(pts, facets, max_area)


def build_square_with_pml(inner_box: Tuple[float,float,float,float] = (-1.0, 1.0, -1.0, 1.0),
                          pml_thickness: float = 0.3,
                          max_area: float = 0.05,
                          return_tags: bool = True):
    """Build a mesh on an expanded outer box that surrounds the physical square with a PML layer.

    Parameters
    ----------
    inner_box : (xmin, xmax, ymin, ymax)
        Physical domain box. Example: (-1, 1, -1, 1).
    pml_thickness : float
        Thickness added on *each side* of the inner box.
        The outer mesh box becomes [xmin-L, xmax+L] × [ymin-L, ymax+L].
    max_area : float
        Target triangle area (smaller => finer mesh).
    return_tags : bool
        If True, also return:
          - elem_region: int array of length ntri (0=core, 1=PML), by centroid test
          - outer_boundary_edges: subset of boundary_edges that lie on the outer box.

    Returns
    -------
    p, t, boundary_edges  [and optionally elem_region, outer_boundary_edges]
    """
    xi, xa, yi, ya = inner_box
    L = float(pml_thickness)
    xo_min, xo_max = xi - L, xa + L
    yo_min, yo_max = yi - L, ya + L

    # build outer box mesh
    p, t, boundary_edges = build_square_mesh(xo_min, xo_max, yo_min, yo_max, max_area=max_area)

    if not return_tags:
        return p, t, boundary_edges

    # element tagging: centroid inside inner box => core (0), else PML (1)
    xy_tri = p[t]                      # (ntri, 3, 2)
    centroids = xy_tri.mean(axis=1)    # (ntri, 2)
    cx, cy = centroids[:,0], centroids[:,1]
    tol = 1e-12
    in_core = (cx >= xi - tol) & (cx <= xa + tol) & (cy >= yi - tol) & (cy <= ya + tol)
    elem_region = np.where(in_core, 0, 1).astype(int)

    # outer boundary edges: both nodes lie on outer rectangle within tolerance
    xb = (np.isclose(p[:,0], xo_min, atol=1e-10) |
          np.isclose(p[:,0], xo_max, atol=1e-10))
    yb = (np.isclose(p[:,1], yo_min, atol=1e-10) |
          np.isclose(p[:,1], yo_max, atol=1e-10))
    on_outer = xb | yb
    mask_outer_edges = np.array([on_outer[i] and on_outer[j] for (i,j) in boundary_edges], dtype=bool)
    outer_boundary_edges = boundary_edges[mask_outer_edges]

    return p, t, boundary_edges, elem_region, outer_boundary_edges
