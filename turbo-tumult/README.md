## Turbo integration with Tumult

An integration of Turbo with Tumult. 
Refer to `tumult-turbo/tmlt/turbo/example.ipynb` for a detailed example of how to use `Tumult` with `Turbo`.
Also, refer to `tumult-turbo/tmlt/turbo/fallback-or-fail-examples.ipynb` to check how the execution falls back to Tumult when we hit cases currently not supported by Turbo.

The integration does not change neither the `tumult-analytics` nor the `tumult-core` packages.
It simply extends the class `Measurement` multiple times to allow implicit interaction with the `privacy_accountant`.

The measurements allow us to decouple operations that happen atomically within Tumult like: 

1. retrieving the true result of a query
2. retrieving the noisy result of a query or
3. subtracting budget.

These operations are performed internally in Turbo.

The integration also extends the class `Session` in `tumult-analytics` with `TurboSession`.
`TurboSession` instantiates `Turbo` with a user-defined configuration and overwrites the `evaluate` function so that it executes a query using `Turbo`.

## Repo Structure

- `api.py`: Implementing Turbo's abstract classes: `TurboQuery`, and `DPEngineHook` to support integration.
- `measurements.py`: `tumult-core:Measurement` class extensions.
- `query_visitors.py`: Helper functions that use the visitor pattern to extract info from `tumult-analytics:QueryExpr` objects.
- `session.py`: An extension of `tumult-analytis:Session` 
- [`neighborhood_definitions.md`](tmlt/turbo/neighborhood_definitions.md): Some notes to read along with the code, explaining how we convert from Turbo's DP semantics to the different definition used by Tumult.
