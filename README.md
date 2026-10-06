# Optimal Control Lecture Examples (dymos)

Interactive Jupyter notebooks accompanying the MAE 546 lectures on numerical
methods for optimal control, built on [dymos](https://openmdao.github.io/dymos/)
and [OpenMDAO](https://openmdao.org/). The notebooks live under `lectures/`
(L9 shooting, L10 collocation); shared helpers live in `oclectures/`.

## Setup

Requires [conda](https://docs.conda.io) (Miniconda/Anaconda/Miniforge) and an
editor that can run Jupyter notebooks (e.g. VS Code with the Jupyter
extension). From the repository root:

```bash
conda env create -f environment.yml
conda activate dymos-lectures
pip install -e .    # the oclectures helper package used by the notebooks
```

Then open any notebook in `lectures/`, select the `dymos-lectures` environment
as the kernel (in VS Code: kernel picker in the top right → Python
Environments → `dymos-lectures`), and run it top to bottom. No external
optimizer is needed — all problems use SciPy's SLSQP.

If a notebook fails on its first cell with `ModuleNotFoundError`, the wrong
kernel is selected or the `pip install -e .` step was skipped.
