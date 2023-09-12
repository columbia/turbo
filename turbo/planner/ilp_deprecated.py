import math
import os
import time
from collections import namedtuple
from multiprocessing import Manager, Process

import gurobipy as gp
from gurobipy import GRB
from loguru import logger
from sortedcollections import OrderedSet
from termcolor import colored

from turbo.budget.curves import ZeroCurve
from turbo.executor import A, RDet, RProb
from turbo.planner.planner import Planner

# from turbo.utils.utils import get_blocks_size
from turbo.utils.utility_theorems import (
    deterministic_compute_utility_curve,
    probabilistic_compute_utility_curve,
)

DeterministicChunk = namedtuple("DeterministicChunk", ["index", "noise_std"])
ProbabilisticChunk = namedtuple("ProbabilisticChunk", ["index", "alpha_beta"])
ILPArgs = namedtuple(
    "ILPArgs", ["task", "indices", "cache", "budget_accountant", "chunk_sizes"]
)

SolveArgs = namedtuple(
    "SolveArgs",
    [
        "task",
        "cache",
        "cache_type",
        "budget_accountant",
        "probabilistic_cfg",
        "blocks_metadata",
    ],
)


def get_chunk_indices(blocks):
    offset = blocks[0]
    n = blocks[1] - blocks[0] + 1
    indices = OrderedSet()
    for i in range(n):
        for j in range(i, n):
            if satisfies_constraint((i + offset, j + offset)):
                indices.add(((i, j)))
    return indices


def get_min_cuts_plan_indices(blocks):
    offset = blocks[0]
    n = blocks[1] - blocks[0] + 1
    indices = OrderedSet()

    # for k in range(n):  # For all possible k
    # Stop when you find the first plan that satisfies the binary constraint

    return indices


def satisfies_constraint(blocks, branching_factor=2):
    n = blocks[1] - blocks[0] + 1
    if not math.log(n, branching_factor).is_integer():
        return False
    if (blocks[0] % n) != 0:
        return False
    return True


def get_chunk_sizes(indices, blocks_metadata):
    # Assuming all blocks have equal size
    block_size = blocks_metadata["block_size"]
    chunk_sizes = {}
    for (i, j) in indices:
        chunk_sizes[(i, j)] = block_size * (j - i + 1)
    return chunk_sizes


class ILP(Planner):
    def __init__(self, cache, budget_accountant, config):
        super().__init__(cache, budget_accountant, config)

    def get_execution_plan(self, task):

        num_blocks = task.blocks[1] - task.blocks[0] + 1

        solve_args = SolveArgs(
            task,
            self.cache,
            self.cache_type,
            self.budget_accountant,
            self.probabilistic_cfg,
            self.blocks_metadata,
        )

        # Choose a planning method
        if self.config.planner.method == "min_cuts":
            """Picks a plan with minimal number of cuts that satisfies the binary constraint.
            If that plan can't be executed we don't look for another one
            """
            # We don't really need an ILP solution for this one but the ILP
            # infrastructure will return if the plan is eligible or not
            return_dict = {}
            indices = get_min_cuts_plan_indices(task.blocks)
            chunk_sizes = get_chunk_sizes(indices, self.blocks_metadata)
            k = len(indices)
            solve(k, k, indices, chunk_sizes, return_dict, solve_args)

        elif self.config.planner.method == "max_cuts":
            """Picks a plan with the maximum number of cuts.
            If that plan can't be executed we don't look for another one
            """
            # We don't really need an ILP solution for this one but the ILP
            # infrastructure will return if the plan is eligible or not
            return_dict = {}
            indices = OrderedSet()
            for i in range(num_blocks):
                indices.add(((i, i)))
            chunk_sizes = get_chunk_sizes(indices, self.blocks_metadata)
            k = len(indices)
            solve(k, k, indices, chunk_sizes, return_dict, solve_args)

        elif self.config.planner.method == "optimal_cuts":
            """Uses ILP to select a plan that satisfies the binary constraint and at the same time tries to
            minimize budget consumption for the given query.
            If no plan is returned it means that there is not enough budget and enough content in the cache
            to accommodate the request.
            """
            manager = Manager()
            return_dict = manager.dict()
            indices = get_chunk_indices(task.blocks)
            chunk_sizes = get_chunk_sizes(task.blocks, self.blocks_metadata)
            max_chunks = num_blocks

            # Running in Parallel
            processes = []
            num_processes = 1  # min(os.cpu_count(), max_chunks)

            k = max_chunks // num_processes
            for i in range(num_processes):
                k_start = i * k + 1
                k_end = i * k + k if i * k + k < max_chunks else max_chunks
                processes.append(
                    Process(
                        target=solve,
                        args=(
                            k_start,
                            k_end,
                            indices,
                            chunk_sizes,
                            return_dict,
                            solve_args,
                        ),
                    )
                )
                processes[i].start()
            for i in range(num_processes):
                processes[i].join()

        # Find the best solution
        plan = None
        best_objvalue = math.inf
        for _, (chunks, objvalue) in return_dict.items():
            if objvalue < best_objvalue:
                run_ops = []
                for chunk in chunks:
                    if isinstance(chunk, DeterministicChunk):
                        (i, j), noise_std = chunk
                        run_ops += [
                            RDet((i + task.blocks[0], j + task.blocks[0]), noise_std)
                        ]
                    elif isinstance(chunk, ProbabilisticChunk):
                        (i, j), (alpha, beta) = chunk
                        run_ops += [
                            RProb((i + task.blocks[0], j + task.blocks[0]), alpha, beta)
                        ]
                plan = A(l=run_ops, cost=objvalue)
                best_objvalue = objvalue
        return plan


def solve(kmin, kmax, indices, chunk_sizes, return_dict, solve_args):
    t = time.time()

    ilp_args = ILPArgs(
        solve_args.task,
        indices,
        solve_args.cache,
        solve_args.budget_accountant,
        chunk_sizes,
    )
    # print("Kmin - Kmax", kmin, kmax, indices)
    if solve_args.cache_type == "DeterministicCache":
        for k in range(kmin, kmax + 1):
            chunks, objval = deterministic_optimize(k, ilp_args)
            if chunks:
                return_dict[k] = (chunks, objval)

    elif solve_args.cache_type == "ProbabilisticCache":
        for k in range(kmin, kmax + 1):
            # print("K", k)

            # Don't allow more than max_pmw_k PMW aggregations
            if k > solve_args.probabilistic_cfg.max_pmw_k:
                break
            chunks, objval = probabilistic_optimize(k, ilp_args)
            if chunks:
                return_dict[k] = (chunks, objval)

    elif solve_args.cache_type == "CombinedCache":
        best_objvalue = math.inf
        for k in range(kmin, kmax + 1):
            # print("K", k)

            for k_prob in range(k + 1):
                # print("K prob", k_prob)

                # Don't allow more than max_pmw_k PMW aggregations
                if k_prob > solve_args.probabilistic_cfg.max_pmw_k:
                    break
                k_det = k - k_prob

                num_blocks = solve_args.task.blocks[1] - solve_args.task.blocks[0] + 1
                # Assuming all blocks have equal size
                block_size = solve_args.blocks_metadata["block_size"]

                # We find the largest possible n_det. We assume that the K_prob probabilistic chunks get assigned
                # with 1 block each in the extreme case. Then the K_det deterministic chunks hold a total of
                # (num_blocks - K_prob) blocks.
                num_det_blocks_max = num_blocks - k_prob if k_det > 0 else 0
                n_det_max = num_det_blocks_max * block_size
                # We find the smallest possible n_det. We assume that the K_det deterministic chunks get assigned
                # with 1 block each in the extreme case. Then the K_det deterministic chunks hold a total of
                # (K_det) blocks.
                num_det_blocks_min = k_det if k_prob > 0 else num_blocks
                n_det_min = num_det_blocks_min * block_size

                # Do a grid search over [n_det_min, n_det_max]
                grid_len = 10  # The larger the grid search the better the results
                tmp = int((n_det_max - n_det_min) / grid_len)
                step = tmp if tmp > 0 else 1
                for n_det_lower_bound in range(n_det_min, n_det_max + 1, step):
                    # print("k", k, "kdet",  k_det, "n_detmax", n_det_max, "n_det_min", n_det_min, "n_det_lower_bound", n_det_lower_bound)
                    chunks, objval = combined_optimize(
                        n_det_lower_bound, block_size, k_det, k_prob, ilp_args
                    )
                    if chunks and best_objvalue > objval:
                        # Return only one solution from this batch
                        return_dict[k] = (chunks, objval)
                        best_objvalue = objval

    t = (time.time() - t) / (kmax + 1 - kmin)
    # print(f"    Took:  {t}")


def deterministic_optimize(k, ilp_args):
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(env=env)
        m.Params.OutputFlag = 0

        # Initialize from args
        indices = ilp_args.indices
        blocks = list(range(ilp_args.task.blocks[0], ilp_args.task.blocks[1] + 1))
        deterministic_cache = ilp_args.cache.deterministic_cache

        ba = ilp_args.budget_accountant
        a = ilp_args.task.utility
        b = ilp_args.task.utility_beta
        num_blocks = len(blocks)
        n = ilp_args.chunk_sizes[
            (0, num_blocks - 1)
        ]  # Total size of data across all requested blocks

        run_budget_per_chunk = {}  # We want to minimize this
        noise_std_per_chunk = {}  # To return this for creating the plan

        # A variable per chunk
        x = m.addVars(
            [(i, j) for (i, j) in indices],
            vtype=GRB.INTEGER,
            lb=0,
            ub=1,
            name="x",
        )

        for (i, j) in indices:
            # Using the contents of the deterministic cache, check how much fresh budget
            # we need to spend across the blocks of the (i,j) chunk in order to reach noise-std.
            noise_std = deterministic_compute_utility_curve(
                a, b, n, ilp_args.chunk_sizes[(i, j)], k
            )
            noise_std_per_chunk[(i, j)] = noise_std

            run_budget = deterministic_cache.estimate_run_budget(
                ilp_args.task.query_id,
                (blocks[i], blocks[j]),
                noise_std,
            )
            # print("K", k, "noise-std", noise_std, run_budget.epsilons)

            # print(run_budget)
            # Enough budget in blocks constraint to accommodate "run_budget"
            if not ba.can_run((blocks[i], blocks[j]), run_budget):
                m.addConstr(x[i, j] == 0)
                break

            # For the Objective Function let's reduce the renyi orders to one number by averaging them
            run_budget_per_chunk[(i, j)] = (
                sum(run_budget.epsilons) / len(run_budget.epsilons)
            ) * (j - i + 1)

        # No overlapping chunks constraint
        for b in range(num_blocks):  # For every block
            m.addConstr(
                (
                    gp.quicksum(
                        x[i, j]
                        for i in range(b + 1)
                        for j in range(b, num_blocks)
                        if (i, j) in indices
                    )
                )
                == 1
            )

        # Computations aggregated must be equal to K
        m.addConstr((gp.quicksum(x[i, j] for (i, j) in indices)) == k)

        # Objective function
        m.setObjective(x.prod(run_budget_per_chunk))
        m.ModelSense = GRB.MINIMIZE
        m.optimize()

        if m.status == GRB.OPTIMAL:
            chunks = []
            for (i, j) in indices:
                if int((abs(x[i, j].x - 1) < 1e-4)) == 1:
                    chunks.append(
                        DeterministicChunk((i, j), noise_std_per_chunk[(i, j)])
                    )
            return chunks, m.ObjVal
        return [], math.inf


def probabilistic_optimize(k, ilp_args):
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(env=env)
        m.Params.OutputFlag = 0

        # Initialize from args
        indices = ilp_args.indices
        blocks = list(range(ilp_args.task.blocks[0], ilp_args.task.blocks[1] + 1))
        probabilistic_cache = ilp_args.cache.probabilistic_cache
        ba = ilp_args.budget_accountant
        a = ilp_args.task.utility
        b = ilp_args.task.utility_beta
        num_blocks = len(blocks)

        run_budget_per_chunk = {}  # We want to minimize this
        alpha_beta_per_chunk = {}  # To return this for creating the plan

        # A variable per chunk
        x = m.addVars(
            [(i, j) for (i, j) in indices],
            vtype=GRB.INTEGER,
            lb=0,
            ub=1,
            name="x",
        )

        for (i, j) in indices:
            # Using the contents of the probabilistic cache, estimate how much fresh budget
            # we need to spend across the blocks of the (i,j) chunk.
            alpha, beta = probabilistic_compute_utility_curve(a, b, k)
            # print("alpha", alpha, "beta", beta)
            alpha_beta_per_chunk[(i, j)] = (alpha, beta)
            # print("a, b", alpha, beta)
            run_budget, worst_run_budget = probabilistic_cache.estimate_run_budget(
                ilp_args.task.query, (blocks[i], blocks[j]), alpha, beta
            )
            # print("run budget", run_budget)

            # Enough budget in blocks constraint to accommodate "run_budget" and "worst_budget"
            if (
                run_budget == ZeroCurve()
                and not ba.can_run((blocks[i], blocks[j]), worst_run_budget)
            ) or not ba.can_run((blocks[i], blocks[j]), run_budget):
                m.addConstr(x[i, j] == 0)
                break

            # For the Objective Function let's reduce the renyi orders to one number by averaging them
            run_budget_per_chunk[(i, j)] = (
                sum(run_budget.epsilons) / len(run_budget.epsilons)
            ) * (j - i + 1)

        # No overlapping chunks constraint
        for b in range(num_blocks):  # For every block
            m.addConstr(
                (
                    gp.quicksum(
                        x[i, j]
                        for i in range(b + 1)
                        for j in range(b, num_blocks)
                        if (i, j) in indices
                    )
                )
                == 1
            )

        # Computations aggregated must be equal to K
        m.addConstr((gp.quicksum(x[i, j] for (i, j) in indices)) == k)

        # Objective function
        m.setObjective(x.prod(run_budget_per_chunk))
        m.ModelSense = GRB.MINIMIZE
        m.optimize()

        if m.status == GRB.OPTIMAL:
            chunks = []
            for (i, j) in indices:
                if int((abs(x[i, j].x - 1) < 1e-4)) == 1:
                    chunks.append(
                        ProbabilisticChunk((i, j), alpha_beta_per_chunk[(i, j)])
                    )
            return chunks, m.ObjVal
        return [], math.inf


# or deterministic, too, but let's do it when we feel more comfortable with this implementation.
def combined_optimize(n_det_lower_bound, block_size, k_det, k_prob, ilp_args):
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
        m = gp.Model(env=env)
        m.Params.OutputFlag = 0

        # Initialize from args
        indices = ilp_args.indices
        blocks = list(range(ilp_args.task.blocks[0], ilp_args.task.blocks[1] + 1))
        deterministic_cache = ilp_args.cache.deterministic_cache
        probabilistic_cache = ilp_args.cache.probabilistic_cache
        chunk_sizes = ilp_args.chunk_sizes
        ba = ilp_args.budget_accountant
        a = ilp_args.task.utility
        b = ilp_args.task.utility_beta
        num_blocks = len(blocks)

        # If we have both cache types we need to compute b after the union bound
        # Same b for deterministic and probabilistic
        if k_det > 0 and k_prob > 0:
            b = 1 - math.sqrt(1 - b)
        # aa, bb = probabilistic_compute_utility_curve(a, b, k_prob)
        # print("alpha", aa, "beta", bb)
        run_budget_per_det_chunk = {}
        run_budget_per_prob_chunk = {}
        noise_std_per_chunk = {}
        alpha_beta_per_chunk = {}

        # A variable per chunk for deterministic
        x = m.addVars(
            [(i, j) for (i, j) in indices],
            vtype=GRB.INTEGER,
            lb=0,
            ub=1,
            name="x",
        )

        # A variable per chunk for probabilistic
        y = m.addVars(
            [(i, j) for (i, j) in indices],
            vtype=GRB.INTEGER,
            lb=0,
            ub=1,
            name="y",
        )

        for (i, j) in indices:
            blocks_ij = (blocks[i], blocks[j])

            if k_det > 0:
                noise_std = deterministic_compute_utility_curve(
                    a, b, n_det_lower_bound, chunk_sizes[(i, j)], k_det
                )
                noise_std_per_chunk[(i, j)] = noise_std
                run_budget_det = deterministic_cache.estimate_run_budget(
                    ilp_args.task.query_id,
                    blocks_ij,
                    noise_std,
                )
                if not ba.can_run(blocks_ij, run_budget_det):
                    m.addConstr(x[i, j] == 0)
                    break
                run_budget_per_det_chunk[(i, j)] = (
                    sum(run_budget_det.epsilons) / len(run_budget_det.epsilons)
                ) * (j - i + 1)

                if run_budget_per_det_chunk[(i, j)] == 0:
                    # In this case we should prefer the deterministic cache for this chunk no matter what.
                    m.addConstr(y[i, j] == 0)
            else:
                run_budget_per_det_chunk[(i, j)] = 10000  # Setting a very high value

            if k_prob > 0:
                alpha, beta = probabilistic_compute_utility_curve(a, b, k_prob)
                alpha_beta_per_chunk[(i, j)] = (alpha, beta)
                (
                    run_budget_prob,
                    worst_run_budget,
                ) = probabilistic_cache.estimate_run_budget(
                    ilp_args.task.query, blocks_ij, alpha, beta
                )
                # Enough budget in blocks constraint to accommodate "run_budget"
                if (
                    run_budget_prob == ZeroCurve()
                    and not ba.can_run((blocks[i], blocks[j]), worst_run_budget)
                ) or not ba.can_run((blocks[i], blocks[j]), run_budget_prob):
                    m.addConstr(y[i, j] == 0)
                    break

                run_budget_per_prob_chunk[(i, j)] = (
                    sum(run_budget_prob.epsilons) / len(run_budget_prob.epsilons)
                ) * (j - i + 1)
            else:
                run_budget_per_prob_chunk[(i, j)] = 10000  # Setting a very high value

        # Actual n_det must not be smaller than n_det_lower_bound
        # Assuming all blocks have equal size
        m.addConstr(
            (gp.quicksum(x[i, j] * (j - i + 1) * block_size for (i, j) in indices))
            >= n_det_lower_bound
        )
        # No overlapping chunks constraint
        for b in range(num_blocks):  # For every block
            m.addConstr(
                (
                    gp.quicksum(
                        x[i, j]
                        for i in range(b + 1)
                        for j in range(b, num_blocks)
                        if (i, j) in indices
                    )
                )
                <= 1
            )
            m.addConstr(
                (
                    gp.quicksum(
                        y[i, j]
                        for i in range(b + 1)
                        for j in range(b, num_blocks)
                        if (i, j) in indices
                    )
                )
                <= 1
            )
        for (i, j) in indices:
            m.addConstr(x[i, j] + y[i, j] <= 1)

        # Computations aggregated must be equal to K_det/K_prob
        m.addConstr((gp.quicksum(x[i, j] for (i, j) in indices)) == k_det)
        m.addConstr((gp.quicksum(y[i, j] for (i, j) in indices)) == k_prob)

        # Objective function
        m.setObjective(
            x.prod(run_budget_per_det_chunk) + y.prod(run_budget_per_prob_chunk)
        )
        m.ModelSense = GRB.MINIMIZE
        m.optimize()

        if m.status == GRB.OPTIMAL:
            chunks = []
            for (i, j) in indices:
                if int((abs(x[i, j].x - 1) < 1e-4)) == 1:
                    chunks.append(
                        DeterministicChunk((i, j), noise_std_per_chunk[(i, j)])
                    )
                if int((abs(y[i, j].x - 1) < 1e-4)) == 1:
                    chunks.append(
                        ProbabilisticChunk((i, j), alpha_beta_per_chunk[(i, j)])
                    )
            return chunks, m.ObjVal
        return [], math.inf
