## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .            # optional, editable install

python scripts/run_demo.py  # runs with default example (g(x,y)=x+y, coeffs=0)
```
## Running the code with colab to avoid installation in your machine 

1) Open the jupyter notebook WavesSummerSchool_Helmholtz.ipynb on colab
    https://colab.research.google.com/drive/1zrLV_yszNN_yODu9dwvatAWCXqTca8nb?usp=sharing

2) Run the very first cell that installs the package.
3) Click Runtime → Run all.

Please note that this requires a google account and that you will not be able to save changes unless you upload the jupyter notebook to your own colab (i.e., make "your own copy").

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



