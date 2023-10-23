from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, Tuple

from loguru import logger
from omegaconf import OmegaConf
from termcolor import colored
from turbo.api import TurboQuery, DPEngineHook
from turbo.core import Accuracy, calibrate_budget_pmwbypass
from turbo.core.cache import (
    ExactMatchCache,
    HistogramCache,
    MockExactMatchCache,
    MockHistogramCache,
    MockSparseVectors,
    SparseVectors,
)
from turbo.utilities import get_data_domain_info, hash, to_tensor

DEFAULT_CONFIG = Path(__file__).parent.joinpath("config.json")


class ProbeStatus(IntEnum):
    BYPASS = 0
    SV_HARD_QUERY = 1


class Turbo:
    """
    config : A configuration for Turbo
    _exact_cache: A public key-value store in which we save the DP results of queries so that we don't resample noise to re-compute them when those queries re-appear
    _histogram_cache: A public key-value store in which we save an approximation of the dataset-view in the form of a Histogram
    _sparse_vectors: A private key-value store in which we save Sparse Vectors. Every SV corresponds to a histogram
    """

    def __init__(
        self,
        config: Dict[str, Any],
        dp_engine_hook: DPEngineHook,
    ) -> "Turbo":
        """Turbo Constructor."""

        assert dp_engine_hook is not None
        self.dp_engine_hook = dp_engine_hook

        assert config is not None
        self.config = OmegaConf.load(DEFAULT_CONFIG)
        custom_config = OmegaConf.create(config)
        self.config = OmegaConf.merge(self.config, custom_config)
        # print(self.config)

        self.config.data_domain_info = get_data_domain_info(self.config.attributes_info)
        self.accuracy = Accuracy(self.config.alpha, self.config.beta)

        # Caching components
        self._exact_cache = MockExactMatchCache(self.config)
        self._histogram_cache = MockHistogramCache(self.config)
        self._sparse_vectors = MockSparseVectors(self.config)

    def run(self, query: TurboQuery, accuracy: Accuracy) -> Tuple[Any, float]:
        """Runs a query using the Turbo cache and returns the dp result
        Args:
            query: TurboQuery
            accuracy: Accuracy target
        """

        # Extract query info needed by Turbo
        aggregation_type = query.get_aggregation_type()
        filter_clause = query.get_filter_clause()
        data_view_id = query.get_data_view_id()
        data_view_size = self.dp_engine_hook.get_data_view_size(query)

        # Look first in the level 1 cache (exact-cache)
        query_id = hash(aggregation_type, filter_clause)
        answer, overhead_budget = self.probeL1(query_id, data_view_id, accuracy)
        assert overhead_budget == 0

        if not answer:
            # Simple cache look-up failed - look in the level 2 cache (PMW Bypass)
            tensor_query = to_tensor(self.config.data_domain_info, filter_clause)
            is_bypass = self.is_bypass(data_view_id, tensor_query)
            if not is_bypass:
                # Obtain true output
                true_answer = self.dp_engine_hook.executeNPQuery(query)
                answer, overhead_budget = self.probeL2(
                    data_view_size, data_view_id, tensor_query, true_answer
                )
                # At this point we used the SV - if we need to pay for it this happens here
                self.dp_engine_hook.consume_budget(overhead_budget)

            if not answer:
                # PMW-Bypass cannot help either (we either bypassed or sv check failed); run as per usual
                # Run the query with the executor, under BDP semantics
                computation_budget = calibrate_budget_pmwbypass(
                    1, accuracy.alpha, accuracy.beta, data_view_size
                )
                # `computation_budget` could be ignored by a larger privacy budget
                #  depending  on how `executeDPQuery` is implemented by the user
                answer = (
                    self.dp_engine_hook.executeDPQuery(
                        query, computation_budget, true_answer
                    )
                    if not is_bypass
                    else self.dp_engine_hook.executeDPQuery(query, computation_budget)
                )

                # Help Turbo back: use the dp-result to update the cache!
                self.update_cache(
                    query_id,
                    data_view_id,
                    data_view_size,
                    accuracy,
                    tensor_query,
                    answer,
                    use_safety_margin=True if is_bypass else False,
                )
        return answer

    def is_bypass(self, data_view_id, tensor_query) -> bool:
        """Checks whether the histogram is ready. It returns true or false.
        Args:
            data_view_id: the unique ID of the data view on which the query will operate
            tensor_query: A tensor representation of the query - to be used for histogram operations
        """
        if not self._histogram_cache.is_histogram_ready(data_view_id, tensor_query):
            return True  # Bypass histogram
        return False

    def probeL1(self, query_id, data_view_id, accuracy) -> Tuple[Any, float]:
        """Looks up for query's DP result in exact-cache. If a DP result exists it checks if it is accurate enough.
        Returns a TurboResponse.

        Args:
            tr: TurboRequest
        """
        cache_entry = self._exact_cache.read_entry(query_id, data_view_id)
        if not cache_entry or not cache_entry.accuracy.check_accuracy(accuracy):
            return None, 0.0
        return cache_entry.dp_result, 0.0

    def probeL2(
        self, data_view_size, data_view_id, tensor_query, true_result: float
    ) -> Tuple[Any, float]:
        """Looks up for query's DP result inside the histogram. There are three possibilities:
        a) The histogram is trained to answer accurately - paying zero privacy budget
        b) The histogram is bypassed by Turbo's heuristics - paying initial privacy budget
        c) The heuristic missed and the histogram did not answer accurately - paying initial privacy budget + budget for SV initialization (cache overhead)
        It returns a TurboResponse.

        Args:
            data_view_size: the size of the data view on which the query will operate
            data_view_id: the unique ID of the data view on which the query will operate
            tensor_query: A tensor representation of the query - to be used for histogram operations
            true_result: the true result (without added noise) of the query evaluated by the DP system.
            It is required for the SV accuracy check. It is of type float (we can't have histogram runs on multidimentional queries right now)
        """
        cache_entry = self._histogram_cache.read_entry(data_view_id)
        if not cache_entry:
            cache_entry = self._histogram_cache.create_new_entry()
            self._histogram_cache.write_entry(data_view_id, cache_entry)

        # Run histogram to get the predicted output
        dp_result = cache_entry.histogram.run(tensor_query)
        logger.debug(colored(f"dp_result, {dp_result}", "yellow"))

        # divide true_result by population size
        true_result = true_result / data_view_size
        logger.debug(colored(f"true_result, {true_result}", "yellow"))
        status, overhead_budget = self._run_sv_check(
            dp_result, true_result, data_view_id, data_view_size
        )
        if status:
            dp_result = dp_result * data_view_size
            return dp_result, overhead_budget

        self._histogram_cache.update_entry_threshold(data_view_id, tensor_query)
        return None, overhead_budget

    def _run_sv_check(
        self, dp_result, true_result, data_view_id, data_view_size
    ) -> Tuple[bool, float]:
        """Runs the SV check and flags the SV as uninitialized if check failed."""
        sv = self._sparse_vectors.read_entry(data_view_id)
        if not sv:
            sv = self._sparse_vectors.create_new_entry(data_view_size)

        overhead_budget = 0.0

        if not sv.initialized:
            sv.initialize()
            overhead_budget = 3 * sv.epsilon  # Initialization budget
            logger.debug(colored(f"SV_init_budget, {overhead_budget}", "red"))

        # Now check whether we pass or fail the SV check
        if sv.check(true_result, dp_result) == False:
            sv_check_status = False
            # Flag SV as uninitialized so that we pay again for its initialization next time we use it
            sv.initialized = False
        else:
            sv_check_status = True
        self._sparse_vectors.write_entry(sv)
        return sv_check_status, overhead_budget

    def update_cache(
        self,
        query_id,
        data_view_id,
        data_view_size,
        accuracy,
        tensor_query,
        dp_result: float,
        use_safety_margin: bool,
    ) -> None:
        """Uses the dp_result to update the exact-cache and the train the histogram.
        Note that, the dp_result used for the updates must not be the output of a Turbo 'probe' method.
        It must have been computed with freshly sampled noise.

        Args:
            query_id: an ID that uniquely identifies the query
            data_view_id: the unique ID of the data view on which the query will operate
            data_view_size: the size of the data view on which the query will operate
            tensor_query: A tensor representation of the query - to be used for histogram operations
            dp_result: the DP result of a query (measurement) to be used for the private update of the caches.
        """

        # Update Exact-Cache
        cache_entry = self._exact_cache.read_entry(query_id, data_view_id)
        assert not cache_entry

        cache_entry = self._exact_cache.create_new_entry(accuracy, dp_result)
        self._exact_cache.write_entry(query_id, data_view_id, cache_entry)

        # Update Histogram
        dp_result = dp_result / data_view_size

        _ = self._histogram_cache.update_entry_histogram(
            tensor_query, data_view_id, dp_result, use_safety_margin
        )
