from __future__ import annotations
import numpy as np
from typing import Callable, Optional
from .mesh import build_mesh
from .assemble import assemble_dirichlet, solve_system
from .plotting import show_mesh, trisurf_like

def default_coeffs(x: float, y: float):
    return 0.0

def default_g(x: float, y: float):
    return x + y

def default_rhs(x: float, y: float):
    return 0.0

def run(geometry: str = "circle", max_area: float = 0.02,
        coeffs: Callable[[float,float], complex] = default_coeffs,
        g: Callable[[float,float], complex] = default_g,
        rhs: Optional[Callable[[float,float], complex]] = None,
        show_plots: bool = True):
    p, t, bnd = build_mesh(geometry=geometry, max_area=max_area)
    if show_plots:
        show_mesh(p, t, f"{geometry} mesh")
    S, f = assemble_dirichlet(p, t, bnd, coeffs, g, rhs=rhs)
    u = solve_system(S, f)
    if show_plots:
        trisurf_like(p, t, np.real(u), title='real(u)')
        trisurf_like(p, t, np.imag(u), title='imag(u)')
    return p, t, bnd, u


def compute_error_norms(p, t, u_h, u_exact, grad_exact=None):
    """Compute FEM-style L2 and H1 errors.
    
    u_exact(x,y): exact solution
    grad_exact(x,y): returns (ux, uy) gradient. If None, will approximate by FD.
    """
    L2_err2 = 0.0
    H1_err2 = 0.0

    for tri in t:
        xy = p[tri]
        x1,y1 = xy[0]; x2,y2 = xy[1]; x3,y3 = xy[2]
        area = 0.5*abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))

        # FE solution values and exact values at vertices
        uh_vals = u_h[tri]
        ue_vals = np.array([u_exact(x,y) for (x,y) in xy])

        # --- L2 element contribution (vertex quadrature, exact for P1 error) ---
        diff2 = (ue_vals - uh_vals)**2
        L2_err2 += area * np.mean(diff2)

        # --- Gradients ---
        # FE gradient
        Delta = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
        beta  = np.array([y2-y3, y3-y1, y1-y2]) / Delta
        gamma = np.array([x3-x2, x1-x3, x2-x1]) / Delta
        grad_phi = np.stack([beta, gamma], axis=1)  # shape (3,2)
        grad_uh = uh_vals @ grad_phi   # linear combination

        # Exact gradient (at centroid)
        xc,yc = xy.mean(axis=0)
        if grad_exact is not None:
            grad_ue = np.array(grad_exact(xc,yc))
        else:
            # fallback FD approx
            eps = 1e-6
            grad_ue = np.array([
                (u_exact(xc+eps,yc) - u_exact(xc-eps,yc))/(2*eps),
                (u_exact(xc,yc+eps) - u_exact(xc,yc-eps))/(2*eps),
            ])
        
        grad_diff = grad_ue - grad_uh
        H1_err2 += area * (grad_diff @ grad_diff)

    return np.sqrt(L2_err2), np.sqrt(H1_err2)
