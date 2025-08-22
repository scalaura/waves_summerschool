from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

def show_mesh(p: np.ndarray, t: np.ndarray, title: str = "Mesh"):
    plt.triplot(p[:,0], p[:,1], t)
    plt.gca().set_aspect('equal')
    plt.title(title)
    plt.xlabel('x'); plt.ylabel('y')
    plt.show()

def trisurf_like(p: np.ndarray, t: np.ndarray, z: np.ndarray, title: str = ""):
    plt.tripcolor(p[:,0], p[:,1], t, z, shading='gouraud')
    plt.gca().set_aspect('equal')
    if title: plt.title(title)
    plt.colorbar(); plt.xlabel('x'); plt.ylabel('y')
    plt.show()
    
def plot_meshNsol(p: np.ndarray, t: np.ndarray, z: np.ndarray, title: str = ""):
    # Using subplots
    #f, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'height_ratios': [1, 1]})  # row 1, column 2
    # Plot mesh
    #a0.triplot(p[:,0], p[:,1], t)
    #a0.gca().set_aspect('equal')
    #a0.title("Mesh")
    #a0.xlabel('x'); a0.ylabel('y')
    #Plot solution  
    #a1.tripcolor(p[:,0], p[:,1], t, z, shading='gouraud')
    #a1.gca().set_aspect('equal')
    #if title: a1.title(title)
    #a1.colorbar(shrink=0.5); a1.xlabel('x'); a1.ylabel('y')
    # Space between the plots
    #f.tight_layout()
    #f.show()
    plt.subplot(1, 2, 1)
    # Plot mesh
    plt.triplot(p[:,0], p[:,1], t)
    plt.gca().set_aspect('equal')
    plt.title("Mesh")
    plt.xlabel('x'); plt.ylabel('y')
    #Plot solution
    plt.subplot(1, 2, 2)
    plt.tripcolor(p[:,0], p[:,1], t, z, shading='gouraud')
    plt.gca().set_aspect('equal')
    if title: plt.title(title)
    plt.colorbar(shrink=0.5); plt.xlabel('x'); plt.ylabel('y')
    # Space between the plots
    plt.tight_layout()
    plt.show()
    
    
def mesh_and_solution_side_by_side(p: np.ndarray, t: np.ndarray, z: np.ndarray, title: str = ""):
    """Plot mesh and solution side-by-side with equal panel sizes.
    Uses a GridSpec with a dedicated colorbar axis so the solution panel
    doesn't shrink. Both panels share identical axis limits and aspect.
    """

    fig = plt.figure(constrained_layout=True, figsize=(10, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05])

    ax_mesh = fig.add_subplot(gs[0, 0])
    ax_sol  = fig.add_subplot(gs[0, 1])
    cax     = fig.add_subplot(gs[0, 2])

    # Mesh
    ax_mesh.triplot(p[:,0], p[:,1], t)
    ax_mesh.set_aspect('equal')
    ax_mesh.set_title("Mesh")
    ax_mesh.set_xlabel('x'); ax_mesh.set_ylabel('y')

    # Solution
    tri = mtri.Triangulation(p[:,0], p[:,1], t)
    pc = ax_sol.tripcolor(tri, z, shading='gouraud')
    ax_sol.set_aspect('equal')
    if title:
        ax_sol.set_title(title)
    ax_sol.set_xlabel('x'); ax_sol.set_ylabel('y')

    # Keep both panels same extents
    xmin, xmax = p[:,0].min(), p[:,0].max()
    ymin, ymax = p[:,1].min(), p[:,1].max()
    for ax in (ax_mesh, ax_sol):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # Colorbar in its own axis (no shrinking)
    cb = fig.colorbar(pc, cax=cax)
    cb.ax.set_ylabel('value')

    plt.show()

    
def trisurf_side_by_side(p, t, z1, z2, titles=("Field 1","Field 2")):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4), constrained_layout=True)
    tri = mtri.Triangulation(p[:,0], p[:,1], t)

    vmin = min(z1.min(), z2.min())
    vmax = max(z1.max(), z2.max())

    pc1 = ax1.tripcolor(tri, z1, shading='gouraud', vmin=vmin, vmax=vmax)
    ax1.set_aspect('equal'); ax1.set_title(titles[0])

    pc2 = ax2.tripcolor(tri, z2, shading='gouraud', vmin=vmin, vmax=vmax)
    ax2.set_aspect('equal'); ax2.set_title(titles[1])

    # one shared colorbar spanning both
    fig.colorbar(pc2, ax=[ax1, ax2], orientation="vertical", fraction=0.046, pad=0.04)
    plt.show()


    