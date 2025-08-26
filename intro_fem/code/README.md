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
   __Lecture 2:__
   [https://colab.research.google.com/drive/11vBvBOsGyKMMxNm5lFx8NzwDV-GpRlBA?usp=sharing](https://colab.research.google.com/drive/11vBvBOsGyKMMxNm5lFx8NzwDV-GpRlBA?usp=sharing)
   __Lecture 3:__
   [https://colab.research.google.com/drive/1-crJghr60I6i52noBVCYAMrPcC18_fyT?usp=sharing](https://colab.research.google.com/drive/1-crJghr60I6i52noBVCYAMrPcC18_fyT?usp=sharing)

3) Run the very first cell that installs the package.
4) Click Runtime → Run all.

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



