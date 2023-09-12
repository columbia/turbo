# Package documentation

- `budget`: various implementations for the privacy budget
- `task.py`: the Task specification
- `cache`: deterministic / probabilistic caches for storing results. Implemented as Redis key-value stores. There are 'Mock' versions for both where the key value stores are implemented as simple in-memory dictionaries. (redis version pending for probabilistic)
- `planner`: max_cuts / min_cuts / ILP - our three versions of the planner
- `budget_accountant.py`: Redis key-value store for storing the budgets of blocks. There is a 'Mock' version where the key value store is implemented as a simple in-memory dictionary.
- `psql`: an API for storing block data to TimeScaleDB using and running SQL queries using a PSQL client.  There is a 'Mock' version that stores block data in a in-memory dictionary as "histograms" instead, and runs queries using tensor operations.
- `server_blocks.py`: a Blocks Server listening to a socket for new requests for adding block data (both in TimeScaleDB and in the budget_accountant).
- `server_tasks.py`: a Tasks Server listening to a socket for new requests for running a query.
- `client_blocks.py`: API for sending requests to the Blocks Server
- `server_blocks.py`: API for sending requests to the Tasks Server
- `query_processor.py`: Finds a DP plan for the query if possible, runs it, consumes budget if necessary and stores metadata.
- `executor.py`: executes the DP plan of a query using the caches and the PSQL module.
- `simulator`: a simulation of the execution of turbo implemented using Simpy. It generates a workload and data blocks given the configuration in `turbo.json`. It bypasses the blocks/tasks servers API and directly uses the rest of the package modules to execute queries and store data blocks and block budgets. 
All [experiments](https://github.com/columbia/turbo/blob/artifact/experiments/runner.cli.caching.py) in the evaluation use this module to simulate requests on Turbo.
- `run_simulation.py`: entrypoint for running turbo in a simulation.
- `turbo.json`: configuration file to setup the turbo execution. Contains configuration for the simulation as well. To run everything using the mock modules set the flag `"mock": true`.
