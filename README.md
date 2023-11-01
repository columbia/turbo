# Turbo

*Turbo* is an effective cache for linear query workloads over DP databases. Turbo builds upon private multiplicative weights (PMW), a DP mechanism that is powerful in theory but very ineffective in practice, and transforms it into a highly effective caching object, namely PMW-Bypass. A description of this project can be found on our paper, titled [Turbo: Effective caching in Differentially-Private Databases](https://arxiv.org/abs/2306.16163) and published at SOSP '23.


## Repo Structure

- `datasets`: Contains a small version of *CitiBike*, a dataset we used consistently for the evaluation of Turbo (see `turbo-suite/turbo-sosp-artifact`).

- `turbo-lib`: Contains the main functionality of Turbo e.g. implementaion of caches, formulas for converting *privacy budget* to *accuracy*, a default configuration for Turbo and the main `Turbo` class which users need to instantiate in order to integrate with Turbo.

- `turbo-tumult`: An integration of Turbo with Tumult. See `turbo-tumult/example.ipynb` for an example on how to use Tumult with Turbo.

- `turbo-sql`: A simplistic library with some basic functionality that facilitates the integration of Turbo with a SQL dp-engine.

- `dummy-dp-engine`: An implementation of a dummy SQL DP engine and Turbo's integration with it. 
See `dummy-dp-engine/example.ipynb` for an example on how it works.

- `turbo-sosp-artifact`: The official artifact for SOSP23. This version of Turbo contains a lot of additional functionality apart from the main caching components. It comprises a *workload generator* for running experiments on simulations, the *datasets* and *benchmarks* (automated scripts) that we used for evaluation and a lightweight, internal *DP executor* for running queries. It provides instructions on how to reproduce all the experiments found in the evaluation section of the paper.

## Setup

Start a new Python virtual environment (e.g. with `poetry shell`).

Install local packages and dependencies:
```bash
pip3 install -r requirements.txt
```

Check installation:
```bash
python3 turbo-tumult/tmlt/turbo/example.py
```

