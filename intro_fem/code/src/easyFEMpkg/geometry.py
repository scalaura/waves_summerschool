from __future__ import annotations
import numpy as np
from typing import Tuple

def make_geometry(name: str = "square") -> Tuple[np.ndarray, np.ndarray]:
    name = name.lower()
    if name == "square":
        pts = np.array([[-1, -1], [ 1, -1], [ 1,  1], [-1,  1]], float)
        fac = np.array([[0,1],[1,2],[2,3],[3,0]], int)
    elif name == "circle":
        num = 64
        ang = np.linspace(0.0, 2.0*np.pi, num, endpoint=False)
        pts = np.c_[np.cos(ang), np.sin(ang)].astype(float)
        fac = np.c_[np.arange(num), (np.arange(num)+1)%num]
    elif name == "lshape":
        pts = np.array([[0,0],[1,0],[1,1],[0.5,1],[0.5,0.5],[0,0.5]], float)
        fac = np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[5,0]], int)
    else:
        raise ValueError(f"Unknown geometry '{name}'")
    return pts, fac
