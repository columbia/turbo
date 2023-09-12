import math
from typing import Dict, List, Tuple

from loguru import logger
from sortedcollections import OrderedSet

from turbo.executor import A, RunHistogram, RunLaplace, RunPMW
from turbo.planner.planner import Planner
from turbo.utils.utility_theorems import (
    get_epsilon_isotropic_laplace_monte_carlo,
    get_pmw_epsilon,
)
from turbo.utils.utils import get_blocks_size, satisfies_constraint


class MinCuts(Planner):
    def __init__(self, cache, budget_accountant, config):
        super().__init__(cache, budget_accountant, config)

        if config.planner.monte_carlo_cache:
            self.monte_carlo_cache: Dict[Tuple[float, float, int, int], float] = {}
        else:
            self.monte_carlo_cache = None

    def get_min_cuts(self, blocks):
        """
        Returns the minimum number of nodes in the binary tree that can construct <blocks>
        """
        indices = OrderedSet()
        start, end = blocks
        chunk_end = start
        while chunk_end <= end:
            i = 1
            chunk_start = chunk_end
            while chunk_end <= end:
                x = chunk_start + 2**i - 1
                i += 1
                if x <= end and satisfies_constraint((chunk_start, x)):
                    chunk_end = x
                else:
                    indices.add((chunk_start, chunk_end))
                    chunk_end += 1
                    break
        return indices

    def get_execution_plan(self, task, force_laplace=False):
        """
        Picks a plan with minimal number of cuts that satisfies the binary constraint.
        If that plan can't be executed we don't look for another one
        """

        subqueries = self.get_min_cuts(task.blocks)
        n = get_blocks_size(task.blocks, self.config.blocks_metadata)

        # NOTE: System wide accuracy for now
        alpha = self.config.alpha  # task.utility
        beta = self.config.beta  # task.utility_beta

        if self.mechanism_type == "Laplace" or force_laplace:
            min_epsilon = get_epsilon_isotropic_laplace_monte_carlo(
                alpha,
                beta,
                n,
                k=len(subqueries),
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

            plan = self.get_hybrid_plan_separate_sv(task, subqueries, n)
        return plan

    def get_hybrid_plan_separate_sv(self, task, subqueries, n):
        beta_laplace = self.config.beta / 2

        # Laplace only if the histogram is not well trained, Histogram otherwise
        sv_check = False
        run_ops = []
        for (i, j) in subqueries:
            # We don't try to be smart and pick Laplace when the cache is good, because we don't exactly know how many Laplace we will end up aggregating
            if self.cache.histogram_cache.is_query_hard(task.query, (i, j)):
                run_ops += [RunLaplace((i, j), noise_std="TBD")]
            else:
                sv_check = True
                run_ops += [RunHistogram((i, j))]

        # We need to force some queries back to Laplace and recompute a better epsilon with the new number of aggregations
        if sv_check == False:
            # If no SV, then give everything to Laplace. Doesn't work the other way around (beta SV is fixed)
            laplace_subqueries = list(range(len(subqueries)))
            n_laplace = n
            beta_laplace = self.config.beta
            largest_contiguous_histogram = None

        else:
            # Only keep the largest contiguous chunk

            largest_contiguous_histogram = get_longest_contiguous_chunk(run_ops)
            longest_start, longest_end = largest_contiguous_histogram
            laplace_subqueries = []
            n_laplace = 0
            for k, run_op in enumerate(run_ops):
                if (
                    isinstance(run_op, RunHistogram)
                    and (k < longest_start or k > longest_end)
                ) or isinstance(run_op, RunLaplace):
                    laplace_subqueries.append(k)
                    n_laplace += get_blocks_size(
                        run_op.blocks, self.config.blocks_metadata
                    )

            # print(
            #     f"Original histograms: {[r.blocks for r in run_ops if isinstance(r, RunHistogram)]}"
            # )

        original_histograms = [r.blocks for r in run_ops if isinstance(r, RunHistogram)]

        # Now we now the exact number of Laplace subqueries, we can compute a tigther epsilon
        # (lower epsilon, so more noise, but still satisfying the utility constraint)
        # Smaller n and less aggregations
        if laplace_subqueries:
            laplace_epsilon = get_epsilon_isotropic_laplace_monte_carlo(
                a=self.config.alpha,
                b=beta_laplace,
                n=n_laplace,
                k=len(laplace_subqueries),
                N=self.config.planner.monte_carlo_N,
                monte_carlo_cache=self.monte_carlo_cache,
            )

        for subquery_number in laplace_subqueries:
            (i, j) = subqueries[subquery_number]
            node_size = get_blocks_size((i, j), self.config.blocks_metadata)
            sensitivity = 1 / node_size
            laplace_scale = sensitivity / laplace_epsilon
            noise_std = math.sqrt(2) * laplace_scale

            # Overwrite the previous run_op
            run_ops[subquery_number] = RunLaplace((i, j), noise_std)

        new_histograms = [r.blocks for r in run_ops if isinstance(r, RunHistogram)]

        if original_histograms != new_histograms:
            logger.debug(
                f"Forced some subqueries to Laplace to transform {original_histograms} to continuous {new_histograms}"
            )

        plan = A(
            l=run_ops,
            sv_check=sv_check,
            cost=0,
            largest_contiguous_histogram=largest_contiguous_histogram,
        )

        return plan


def get_longest_contiguous_chunk(run_ops: List) -> Tuple[int, int]:
    """
    Returns the start and end index of the longest contiguous chunk of RunHistograms
    """
    longest_width = 0
    longest_start = -1
    longest_end = -1

    def get_width(subquery):
        (i, j) = run_ops[subquery].blocks
        return j - i + 1

    start = 0

    while start < len(run_ops):
        while not isinstance(run_ops[start], RunHistogram):
            start += 1
            if start >= len(run_ops):
                return (longest_start, longest_end)

        end = start
        width = 0
        while isinstance(run_ops[end], RunHistogram):
            width += get_width(end)
            if width > longest_width:
                longest_width = width
                longest_start = start
                longest_end = end
            end += 1
            if end >= len(run_ops):
                return (longest_start, longest_end)

        start = end + 1

    return (longest_start, longest_end)
