import numpy as np
import torch
from loguru import logger

from turbo.budget import BasicBudget, Budget
from turbo.budget.curves import (
    BoundedOneShotSVT,
    GaussianCurve,
    LaplaceCurve,
    PureDPtoRDP,
    ZeroCurve,
)
from turbo.cache.histogram import DenseHistogram, flat_indices
from turbo.utils.utils import mlflow_log

"""
Trimmed-down implementation of PMW, following Salil's pseudocode
"""


class PMW:
    def __init__(
        self,
        alpha,  # Max error guarantee, expressed as fraction.
        epsilon,  # Not the global budget - internal Laplace will be Lap(1/(epsilon*n))
        n,  # Number of samples
        domain_size,  # From blocks_metadata
        config,
        id="",  # Name to log results
    ):

        # Core PMW parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.n = n
        self.domain_size = domain_size
        self.config = config
        self.histogram = DenseHistogram(domain_size)
        self.b = 1 / (self.n * self.epsilon)
        # Logging
        self.queries_ran = 0
        self.hard_queries_ran = 0
        self.id = id

        # We'll initialize the noisy threshold and pay right before we use the SV
        self.noisy_threshold = None
        self.local_svt_queries_ran = 0

    def run(self, query, true_output):
        assert isinstance(query, torch.Tensor)

        run_metadata = {}
        run_budget = BasicBudget(0) if self.config.puredp else ZeroCurve()

        # Pay the initialization budget if it's the first call
        if self.local_svt_queries_ran == 0:
            self.noisy_threshold = self.alpha / 2 + np.random.laplace(
                loc=0, scale=self.b
            )
            run_budget += (
                BasicBudget(3 * self.epsilon)
                if self.config.puredp
                else PureDPtoRDP(epsilon=3 * self.epsilon)
            )

        # Check the public histogram for free. Always normalized, outputs fractions
        predicted_output = self.histogram.run(query)
        logger.debug("noisy result", predicted_output)

        # Add the sparse vector noise
        true_error = abs(true_output - predicted_output)
        error_noise = np.random.laplace(loc=0, scale=self.b)
        noisy_error = true_error + error_noise
        self.queries_ran += 1
        self.local_svt_queries_ran += 1

        # Do the sparse vector check
        if noisy_error < self.noisy_threshold:
            # Easy query, just output the histogram prediction
            output = predicted_output
            run_metadata["hard_query"] = False
            logger.debug(
                f"Easy query - Predicted: {predicted_output}, true: {true_output}, true error: {true_error}, noisy error: {noisy_error}, epsilon: {self.epsilon}"
            )
        else:
            # Hard query, run a fresh Laplace estimate
            output = true_output + np.random.laplace(loc=0, scale=self.b)
            run_budget += (
                BasicBudget(self.epsilon)
                if self.config.puredp
                else LaplaceCurve(laplace_noise=1 / self.epsilon)
            )

            # Increase weights iff predicted_output is too small
            lr = self.alpha / 8
            if output < predicted_output:
                lr *= -1

            # Multiplicative weights update for the relevant bins
            query_tensor_dense = query  # .to_dense()
            self.histogram.tensor = torch.mul(
                self.histogram.tensor, torch.exp(query_tensor_dense * lr)
            )
            self.histogram.normalize()

            # We'll start a new sparse vector at the beginning of the next query (and pay for it)
            run_metadata["hard_query"] = True
            logger.debug(
                f"Hard query - Predicted: {predicted_output}, true: {true_output}"
            )
            self.local_svt_queries_ran = 0
            self.hard_queries_ran += 1

        run_metadata["true_error_fraction"] = abs(output - true_output)
        return output, run_budget, run_metadata

    def mlflow_log_run(self, output, true_output):
        mlflow_log(f"{self.id}/queries_ran", self.queries_ran, self.queries_ran)
        mlflow_log(
            f"{self.id}/hard_queries_ran", self.hard_queries_ran, self.queries_ran
        )
        mlflow_log(
            f"{self.id}/true_error_fraction",
            abs(output - true_output),
            self.queries_ran,
        )
        mlflow_log(
            f"{self.id}/true_error_count",
            self.n * abs(output - true_output),
            self.queries_ran,
        )
