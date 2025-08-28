from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from dataclasses import dataclass

## Cartesian PML
@dataclass
class CartesianPML:
    xmin: float; xmax: float
    ymin: float; ymax: float
    Lx: float; Ly: float
    omega: float
    m: int = 2
    sigma_max_x: float = 20.0
    sigma_max_y: float = 20.0
    alpha_x: float = 1.0
    alpha_y: float = 1.0

    def _sigma_1d(self, x: float, a: float, b: float, L: float, sigma_max: float) -> float:
        if L <= 0: return 0.0
        if x < a + L:
            xi = (a + L - x) / L
            return sigma_max * (xi ** self.m)
        if x > b - L:
            xi = (x - (b - L)) / L
            return sigma_max * (xi ** self.m)
        return 0.0

    def s_factors(self, x: float, y: float):
        sigx = self._sigma_1d(x, self.xmin, self.xmax, self.Lx, self.sigma_max_x)
        sigy = self._sigma_1d(y, self.ymin, self.ymax, self.Ly, self.sigma_max_y)
        sx = self.alpha_x + 1j * (sigx / self.omega if self.omega != 0 else sigx)
        sy = self.alpha_y + 1j * (sigy / self.omega if self.omega != 0 else sigy)
        return sx, sy

    def A_B_at(self, x: float, y: float):
        sx, sy = self.s_factors(x, y)
        A = np.array([[sy/sx, 0.0], [0.0, sx/sy]], dtype=complex)
        B = sx * sy
        return A, B

## Tools for PML Helmholtz assembly (P1 triangles)    
def _triangle_metrics(xy: np.ndarray):
    (x1,y1),(x2,y2),(x3,y3) = xy
    Delta = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    area = 0.5*abs(Delta)
    beta  = np.array([y2-y3, y3-y1, y1-y2]) / Delta
    gamma = np.array([x3-x2, x1-x3, x2-x1]) / Delta
    return area, Delta, beta, gamma

def _consistent_mass(area: float) -> np.ndarray:
    M = np.array([[2,1,1],[1,2,1],[1,1,2]], dtype=float)
    return (area/12.0) * M

def element_matrices_helmholtz_pml(p: np.ndarray, tri: np.ndarray, k: float,
                                   A_B_func,
                                   mass: str = 'consistent') -> np.ndarray:
    xy = p[tri]
    area, _, beta, gamma = _triangle_metrics(xy)
    xc, yc = xy[:,0].mean(), xy[:,1].mean()
    A, B = A_B_func(xc, yc)
    G = np.stack([beta, gamma], axis=1).astype(complex)  # (3,2)
    Se = area * (G @ A @ G.conj().T)
    if mass == 'consistent':
        Me = _consistent_mass(area).astype(complex)
    else:
        Me = np.diag(np.full(3, area/3.0, dtype=float)).astype(complex)
    Se += (-(k**2) * B) * Me
    return Se

def assemble_helmholtz_pml(p: np.ndarray, t: np.ndarray, boundary_edges: np.ndarray,
                            k: float, A_B_func,
                            rhs=None,
                            dirichlet_outer=None,
                            mass: str = 'consistent'):
    n = len(p)
    rows, cols, data = [], [], []
    b = np.zeros(n, dtype=np.complex128)
    for tri in t:
        Se = element_matrices_helmholtz_pml(p, tri, k, A_B_func, mass=mass)
        if rhs is not None:
            xy = p[tri]
            _, Delta, _, _ = _triangle_metrics(xy)
            xc, yc = xy[:,0].mean(), xy[:,1].mean()
            val = rhs(xc, yc)
            fe = np.full(3, val*(abs(Delta)/6.0), dtype=np.complex128)
            for i_loc, I in enumerate(tri):
                b[I] += fe[i_loc]
        for i_loc, I in enumerate(tri):
            for j_loc, J in enumerate(tri):
                rows.append(I); cols.append(J); data.append(Se[i_loc, j_loc])
    S = coo_matrix((np.asarray(data, dtype=np.complex128), (rows, cols)), shape=(n, n)).tocsr()

    if dirichlet_outer is not None and boundary_edges.size:
        bnodes = np.unique(boundary_edges.reshape(-1))
        xv, yv = p[bnodes,0], p[bnodes,1]
        vals = np.array([dirichlet_outer(xi, yi) for xi, yi in zip(xv, yv)], dtype=np.complex128)
        b[bnodes] = vals
        S[bnodes, :] = 0.0
        S[bnodes, bnodes] = 1.0
    return S, b
