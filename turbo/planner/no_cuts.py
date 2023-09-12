import math

from turbo.executor import A, RunHistogram, RunLaplace, RunPMW
from turbo.planner.planner import Planner
from turbo.utils.utility_theorems import (
    get_epsilon_isotropic_laplace_concentration,
    get_pmw_epsilon,
)
from turbo.utils.utils import get_blocks_size


class NoCuts(Planner):
    def __init__(self, cache, budget_accountant, config):
        super().__init__(cache, budget_accountant, config)

    def get_execution_plan(self, task, force_laplace=False):
        # NOTE: System wide accuracy for now
        alpha = self.config.alpha  # task.utility
        beta = self.config.beta  # task.utility_beta
        node_size = get_blocks_size(task.blocks, self.config.blocks_metadata)

        if self.mechanism_type == "Laplace" or force_laplace:
            min_epsilon = get_epsilon_isotropic_laplace_concentration(
                alpha, beta, node_size, 1
            )

            sensitivity = 1 / node_size
            laplace_scale = sensitivity / min_epsilon
            noise_std = math.sqrt(2) * laplace_scale
            plan = A(l=[RunLaplace(task.blocks, noise_std)], sv_check=False, cost=0)

        elif self.mechanism_type == "PMW":
            # NOTE: This is PMW.To be used only in the Monoblock case
            epsilon = get_pmw_epsilon(alpha, beta, node_size)
            plan = A(l=[RunPMW(task.blocks, alpha, epsilon)], sv_check=False, cost=0)

        elif self.mechanism_type == "Hybrid":
            # Assign a Mechanism to each subquery
            # Using the Laplace Utility bound get the minimum epsilon that should be used by each subquery
            # In case a subquery is assigned to a Histogram run instead of a Laplace run
            # a final check must be done by a SV on the aggregated output to assess its quality.
            min_epsilon = get_epsilon_isotropic_laplace_concentration(
                alpha, beta, node_size, 1
            )
            sv_check = False

            # Measure the expected additional budget needed for a Laplace run.
            node_size = get_blocks_size(task.blocks, self.config.blocks_metadata)
            sensitivity = 1 / node_size
            laplace_scale = sensitivity / min_epsilon
            noise_std = math.sqrt(2) * laplace_scale

            cache_entry = (
                None
                if not self.config.exact_match_caching
                else self.cache.exact_match_cache.read_entry(task.query_id, task.blocks)
            )
            if (
                (cache_entry and noise_std >= cache_entry.noise_std)
            ) or self.cache.histogram_cache.is_query_hard(task.query, task.blocks):
                # If we have a good enough estimate in the cache choose Laplace because it will pay nothing.
                # Also choose the Laplace if the histogram is not well trained according to our heuristic
                run_ops = [RunLaplace(task.blocks, noise_std)]
            else:
                sv_check = True
                run_ops = [RunHistogram(task.blocks)]

            plan = A(l=run_ops, sv_check=sv_check, cost=0)

        return plan
