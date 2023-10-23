# turbo-lib
Effective caching for linear query workloads over DP databases. Turbo builds upon private multiplicative weights (PMW), a DP mechanism that is powerful in theory but very ineffective in practice, and transforms it into a highly effective caching object, namely PMW-Bypass. A description of this project can be found on our paper, titled [Turbo: Effective caching in Differentially-Private Databases](https://arxiv.org/abs/2306.16163) and published as SOSP '23.

## Repo Structure

- `api`: Contains part of the Turbo interface. Users who want to integrate with Turbo need to
implement the abstract classes `TurboQuery` and `DPEngineHook`. Those implementations depend on the underlying DP engine and the expected query format that it expects. 
E.g. the example in `turbo-suite/dummy-dp-engine` yields implementations for a dummy dp engine that expects queries in a SQL format while the example in `turbo-suite/tumult-turbo` yields implementations for `Tumult` which expects queries in its own `Spark` format.

- `core`: Contains the main functionality of Turbo e.g. implementaion of caches, formulas for converting `privacy budget` to `accuracy`, a default configuration for Turbo and the main `Turbo` class which users need to instantiate in order to integrate with Turbo.

- `utilities`: utility functions used upon queries to derive special info needed by Turbo (e.g. converting queries to tensors etc.)