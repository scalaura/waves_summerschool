from __future__ import annotations
import numpy as np
from typing import Callable, Tuple, Optional, Literal
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

BCMode = Literal["dirichlet", "neumann", "robin"]

def _triangle_area_and_bg(xy: np.ndarray):
    (x1,y1),(x2,y2),(x3,y3) = xy
    Delta = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    area = abs(Delta)/2.0
    beta  = np.array([y2-y3, y3-y1, y1-y2], dtype=float) / Delta
    gamma = np.array([x3-x2, x1-x3, x2-x1], dtype=float) / Delta
    return area, beta, gamma, Delta

def element_stiffness(p: np.ndarray, tri: np.ndarray, coeffs: Callable[[float,float], complex]) -> np.ndarray:
    xy = p[tri]
    area, beta, gamma, Delta = _triangle_area_and_bg(xy)
    Selem = (abs(Delta)/2.0) * (np.outer(beta,beta) + np.outer(gamma,gamma))
    Selem = Selem.astype(np.complex128)  # allow complex coeffs
    for i_loc, gidx in enumerate(tri):
        x_i, y_i = p[gidx]
        aux = coeffs(x_i, y_i)  # may be complex
        Selem[i_loc, i_loc] += aux * (abs(Delta)/6.0)  # lumped reaction
    return Selem

def element_load(p: np.ndarray, tri: np.ndarray, rhs: Optional[Callable[[float,float], complex]]) -> np.ndarray:
    """Consistent load vector for piecewise-constant RHS via centroid rule.
    fe_i = f(xc,yc) * |Delta|/6  (since ∫_T φ_i dA = |T|/3 and |Delta|=2|T|)
    """
    if rhs is None:
        return np.zeros(3, dtype=np.complex128)
    xy = p[tri]
    _, _, _, Delta = _triangle_area_and_bg(xy)
    xc = xy[:,0].mean(); yc = xy[:,1].mean()
    val = rhs(xc, yc)  # may be complex
    fe = np.full(3, val*(abs(Delta)/6.0), dtype=np.complex128)
    return fe

def _assemble_interior(p: np.ndarray, t: np.ndarray, coeffs, rhs) -> Tuple[csr_matrix, np.ndarray]:
    n = len(p)
    rows, cols, data = [], [], []
    b = np.zeros(n, dtype=np.complex128)
    for tri in t:
        Selem = element_stiffness(p, tri, coeffs)
        felem = element_load(p, tri, rhs)
        for i_loc, I in enumerate(tri):
            b[I] += felem[i_loc]
            for j_loc, J in enumerate(tri):
                rows.append(I); cols.append(J); data.append(Selem[i_loc, j_loc])
    S = coo_matrix((np.asarray(data, dtype=np.complex128), (rows, cols)), shape=(n, n)).tocsr()
    return S, b

def _apply_dirichlet(S: csr_matrix, f: np.ndarray, p: np.ndarray, boundary_edges: np.ndarray, g: Callable[[float,float], complex]):
    bnodes = np.unique(boundary_edges.reshape(-1)) if boundary_edges.size else np.array([], dtype=int)
    if bnodes.size:
        x = p[bnodes,0]; y = p[bnodes,1]
        f_vals = np.array([g(xi, yi) for xi, yi in zip(x, y)], dtype=np.complex128)
        f[bnodes] = f_vals
        S[bnodes,:] = 0.0
        S[bnodes,bnodes] = 1.0
    return S, f

def _apply_neumann(f: np.ndarray, p: np.ndarray, boundary_edges: np.ndarray, q: Callable[[float,float], complex]):
    if boundary_edges.size == 0:
        return f
    for i,j in boundary_edges:
        pi, pj = p[i], p[j]
        L = float(np.linalg.norm(pj-pi))
        xm, ym = (pi+pj)/2.0
        val = q(xm, ym)
        f[i] += val*(L/2.0)
        f[j] += val*(L/2.0)
    return f

def _apply_robin(S: csr_matrix, f: np.ndarray, p: np.ndarray, boundary_edges: np.ndarray, h, u_inf):
    if boundary_edges.size == 0:
        return S, f
    for i,j in boundary_edges:
        pi, pj = p[i], p[j]
        L = float(np.linalg.norm(pj-pi))
        xm, ym = (pi+pj)/2.0
        add = h(xm, ym)*(L/2.0)
        f[i] += add*u_inf(xm, ym)
        f[j] += add*u_inf(xm, ym)
        S[i,i] += add
        S[j,j] += add
    return S, f

def assemble(p: np.ndarray, t: np.ndarray, boundary_edges: np.ndarray,
             coeffs: Callable[[float,float], complex],
             rhs: Optional[Callable[[float,float], complex]] = None,
             bc_mode: BCMode = "dirichlet",
             g: Optional[Callable[[float,float], complex]] = None,
             q: Optional[Callable[[float,float], complex]] = None,
             h: Optional[Callable[[float,float], complex]] = None,
             u_inf: Optional[Callable[[float,float], complex]] = None) -> Tuple[csr_matrix, np.ndarray]:
    S, f = _assemble_interior(p, t, coeffs, rhs)
    if bc_mode == "dirichlet":
        if g is None: raise ValueError("Dirichlet requires g(x,y)")
        S, f = _apply_dirichlet(S, f, p, boundary_edges, g)
    elif bc_mode == "neumann":
        if q is None: raise ValueError("Neumann requires q(x,y)")
        f = _apply_neumann(f, p, boundary_edges, q)
    elif bc_mode == "robin":
        if h is None or u_inf is None: raise ValueError("Robin requires h and u_inf")
        S, f = _apply_robin(S, f, p, boundary_edges, h, u_inf)
    else:
        raise ValueError(f"Unknown bc_mode '{bc_mode}'")
    return S, f

def assemble_dirichlet(p, t, boundary_edges, coeffs, g, rhs=None):
    return assemble(p, t, boundary_edges, coeffs, rhs=rhs, bc_mode="dirichlet", g=g)

def assemble_neumann(p, t, boundary_edges, coeffs, q, rhs=None):
    return assemble(p, t, boundary_edges, coeffs, rhs=rhs, bc_mode="neumann", q=q)

def assemble_robin(p, t, boundary_edges, coeffs, h, u_inf, rhs=None):
    return assemble(p, t, boundary_edges, coeffs, rhs=rhs, bc_mode="robin", h=h, u_inf=u_inf)

def solve_system(S: csr_matrix, f: np.ndarray) -> np.ndarray:
    return spsolve(S, f)
