## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .            # optional, editable install

python scripts/run_demo.py  # runs with default example (g(x,y)=x+y, coeffs=0)
```

## Package layout

```
src/easyFEMpkg/
  geometry.py   # defines geometries: square, circle, lshape
  mesh.py       # uses meshpy.triangle to generate/refine a mesh
  assemble.py   # element matrices, global assembly, Dirichlet BCs
  plotting.py   # mesh and field plotting helpers
  __init__.py
scripts/
  run_demo.py   # main script.
```

