import math
from typing import Dict, Tuple

from turbo.executor import A, RunHistogram, RunLaplace, RunPMW
from turbo.planner.planner import Planner
from turbo.utils.utility_theorems import (
    get_epsilon_isotropic_laplace_concentration,
    get_epsilon_isotropic_laplace_monte_carlo,
    get_pmw_epsilon,
)
from turbo.utils.utils import get_blocks_size, satisfies_constraint


class MaxCuts(Planner):
    def __init__(self, cache, budget_accountant, config):
        super().__init__(cache, budget_accountant, config)

        if config.planner.monte_carlo_cache:
            self.monte_carlo_cache: Dict[Tuple[float, float, int, int], float] = {}
        else:
            self.monte_carlo_cache = None

    def get_max_cuts(self, blocks):
        """
        Returns the minimum number of nodes in the binary tree that can construct <blocks>
        """
        indices = [(i, i) for i in range(blocks[0], blocks[1] + 1)]
        return indices

    def get_execution_plan(self, task, force_laplace=False):
        """
        Picks a plan with minimal number of cuts that satisfies the binary constraint.
        If that plan can't be executed we don't look for another one
        """

        subqueries = self.get_max_cuts(task.blocks)

        n = get_blocks_size(task.blocks, self.config.blocks_metadata)
        k = len(subqueries)

        # NOTE: System wide accuracy for now
        alpha = self.config.alpha  # task.utility
        beta = self.config.beta  # task.utility_beta

        if self.mechanism_type == "Laplace" or force_laplace:
            min_epsilon = get_epsilon_isotropic_laplace_monte_carlo(
                alpha,
                beta,
                n,
                k,
                N=self.config.planner.monte_carlo_N,
                monte_carlo_cache=self.monte_carlo_cache,
            )

            run_ops = []
            for (i, j) in subqueries:
                node_size = get_blocks_size((i, j), self.config.blocks_metadata)
                sensitivity = 1 / node_size
                laplace_scale = sensitivity / min_epsilon
                noise_std = math.sqrt(2) * laplace_scale
                run_ops += [RunLaplace((i, j), noise_std)]
            plan = A(l=run_ops, sv_check=False, cost=0)

        elif self.mechanism_type == "PMW":
            # Works only in monoblock setting
            assert len(subqueries) == 1
            (i, j) = subqueries[0]
            node_size = get_blocks_size((i, j), self.config.blocks_metadata)
            epsilon = get_pmw_epsilon(alpha, beta, node_size)
            run_ops = [RunPMW((i, j), alpha, epsilon)]
            plan = A(l=run_ops, sv_check=False, cost=0)

        elif self.mechanism_type == "Hybrid":
            # Assign a Mechanism to each subquery
            # Using the Laplace Utility bound get the minimum epsilon that should be used by each subquery
            # In case a subquery is assigned to a Histogram run instead of a Laplace run
            # a final check must be done by a SV on the aggregated output to assess its quality.
            min_epsilon = get_epsilon_isotropic_laplace_monte_carlo(
                alpha,
                beta,
                n,
                k,
                N=self.config.planner.monte_carlo_N,
                monte_carlo_cache=self.monte_carlo_cache,
            )

            sv_check = False
            run_ops = []
            for (i, j) in subqueries:
                # Measure the expected additional budget needed for a Laplace run.
                cache_entry = (
                    self.cache.exact_match_cache.read_entry(task.query_id, (i, j))
                    if self.config.exact_match_caching
                    else None
                )

                node_size = get_blocks_size((i, j), self.config.blocks_metadata)
                sensitivity = 1 / node_size
                laplace_scale = sensitivity / min_epsilon
                noise_std = math.sqrt(2) * laplace_scale

                if (
                    (cache_entry and noise_std >= cache_entry.noise_std)
                ) or self.cache.histogram_cache.is_query_hard(task.query, (i, j)):
                    # If we have a good enough estimate in the cache choose Laplace because it will pay nothing.
                    # Also choose the Laplace if the histogram is not well trained according to our heuristic
                    run_ops += [RunLaplace((i, j), noise_std)]
                else:
                    sv_check = True
                    run_ops += [RunHistogram((i, j))]

            plan = A(l=run_ops, sv_check=sv_check, cost=0)

        elif self.mechanism_type == "TimestampsPMW":
            raise NotImplementedError
        return plan
