import time
from copy import deepcopy

import redisai as rai
import torch
from loguru import logger

from turbo.cache.histogram import DenseHistogram
from turbo.utils.utils import get_blocks_size


class CacheKey:
    def __init__(self, blocks):
        self.key = str(blocks)


class CacheEntry:
    def __init__(self, histogram, bin_updates, bin_thresholds) -> None:
        self.histogram = histogram
        self.bin_updates = bin_updates
        self.bin_thresholds = bin_thresholds


class HistogramCache:
    def __init__(self, config):
        self.kv_store = self.get_kv_store(config)
        self.config = config
        self.blocks_metadata = self.config.blocks_metadata
        self.domain_size = self.blocks_metadata["domain_size"]
        heuristic = config.mechanism.probabilistic_cfg.heuristic
        self.heuristic_type, heuristic_params = heuristic.split(":")
        heuristic_params = heuristic_params.split("-")

        if len(heuristic_params) == 2:
            threshold, step = heuristic_params
            self.bin_threshold_max = None
            self.quantile_threshold = 0
        else:
            threshold, step, bin_threshold_max, quantile_threshold = heuristic_params
            if bin_threshold_max != "0":
                self.bin_threshold_max = float(bin_threshold_max)
            else:
                self.bin_threshold_max = None
            self.quantile_threshold = float(quantile_threshold)

        self.bin_thershold = float(threshold)
        self.bin_thershold_step = float(step)
        logger.debug(
            f"Using bin threshold {self.bin_thershold} and step {self.bin_thershold_step}"
        )

        self.learning_rate = config.mechanism.probabilistic_cfg.learning_rate
        if isinstance(self.learning_rate, str):
            lrs = {}
            for lr in self.learning_rate.split("_"):
                x = lr.split(":")
                lrs[float(x[0])] = float(x[1])
            self.learning_rate = lrs

    def get_kv_store(self, config):
        return rai.Client(host=config.cache.host, port=config.cache.port, db=0)

    def write_entry(self, blocks, cache_entry):
        key = CacheKey(blocks).key
        self.kv_store.tensorset(
            key + ":histogram",
            cache_entry.histogram.tensor.numpy(),
            self.domain_size,
            torch.float64,
        )
        self.kv_store.tensorset(
            key + ":bin_updates",
            cache_entry.bin_updates.numpy(),
            self.domain_size,
            torch.float64,
        )
        self.kv_store.tensorset(
            key + ":bin_thresholds",
            cache_entry.bin_thresholds.numpy(),
            self.domain_size,
            torch.float64,
        )

    def read_entry(self, blocks):
        key = CacheKey(blocks).key
        try:
            entry_histogram_tensor = self.kv_store.tensorget(key + ":histogram")
            entry_bin_updates = self.kv_store.tensorget(key + ":bin_updates")
            entry_bin_thresholds = self.kv_store.tensorget(key + ":bin_thresholds")
        except:
            return None

        entry_histogram_tensor = torch.tensor(entry_histogram_tensor)
        entry_bin_updates = torch.tensor(entry_bin_updates)
        entry_bin_thresholds = torch.tensor(entry_bin_thresholds)
        entry_histogram = DenseHistogram(
            domain_size=self.domain_size, tensor=entry_histogram_tensor
        )
        return CacheEntry(entry_histogram, entry_bin_updates, entry_bin_thresholds)

    def create_new_entry(self, blocks):

        cache_entry = None
        if self.config.mechanism.probabilistic_cfg.bootstrapping == True:
            # Bootstrapping: creating a histogram for a new block or node and
            # initializing it with the histogram of the previous block or the children nodes
            (i, j) = blocks
            node_size = j - i + 1
            if node_size == 1 and i > 0:  # leaf node
                # Find the first previous block in cache to initialize from
                # for x in reversed(range(i)):
                for x in reversed(range(i)):
                    cache_entry = self.read_entry((x, x))
                    if cache_entry is not None:
                        break
            else:  # not leaf node - aggregate children
                # Get children nodes
                left_child = (i, i + node_size / 2 - 1)
                right_child = (i + node_size / 2, j)
                left_child_entry = self.read_entry((left_child[0], left_child[1]))
                right_child_entry = self.read_entry((right_child[0], right_child[1]))
                if left_child_entry and right_child_entry:
                    new_histogram = DenseHistogram(self.domain_size)
                    new_histogram.tensor = torch.div(
                        torch.add(
                            left_child_entry.histogram.tensor,
                            right_child_entry.histogram.tensor,
                        ),
                        2,
                    )
                    new_bin_updates = torch.div(
                        torch.add(
                            left_child_entry.bin_updates, right_child_entry.bin_updates
                        ),
                        2,
                    )
                    new_bin_thresholds = torch.div(
                        torch.add(
                            left_child_entry.bin_thresholds,
                            right_child_entry.bin_thresholds,
                        ),
                        2,
                    )
                    cache_entry = CacheEntry(
                        histogram=new_histogram,
                        bin_updates=new_bin_updates,
                        bin_thresholds=new_bin_thresholds,
                    )

        if cache_entry:
            new_cache_entry = CacheEntry(
                histogram=deepcopy(cache_entry.histogram),
                bin_updates=deepcopy(cache_entry.bin_updates),
                bin_thresholds=deepcopy(cache_entry.bin_thresholds),
            )
        else:
            new_cache_entry = CacheEntry(
                histogram=DenseHistogram(self.domain_size),
                bin_updates=torch.zeros(
                    size=(1, self.domain_size), dtype=torch.float64
                ),
                bin_thresholds=torch.ones(
                    size=(1, self.domain_size), dtype=torch.float64
                )
                * self.bin_thershold,
            )
        return new_cache_entry

    def update_entry_histogram(self, query, blocks, noisy_result, epsilon) -> int:
        """_summary_

        Args:
            query (_type_): _description_
            blocks (_type_): _description_
            noisy_result (_type_): _description_
            epsilon (_type_): _description_

        Returns:
            int: -1 or 1 depending on the direction
                0 if no external update (skipped because of the check)
        """

        if self.config.mechanism.probabilistic_cfg.external_update_on_cached_results:
            # epsilon is now the target epsilon (never 0 even if lapace run was cached)
            assert epsilon > 0
        else:
            # External updates only with fresh noise
            if epsilon == 0:
                return

        cache_entry = self.read_entry(blocks)
        if not cache_entry:
            cache_entry = self.create_new_entry(blocks)

        query_tensor_dense = query

        # Do External Update on the histogram - update bin counts too
        predicted_output = cache_entry.histogram.run(query)
        # n = get_blocks_size(blocks, self.blocks_metadata)

        # Parse learning rate
        learning_rate = self.learning_rate
        if isinstance(self.learning_rate, dict):
            min_num_updates = torch.min(
                cache_entry.bin_updates[query_tensor_dense > 0]
            ).item()
            for t in reversed(sorted(list(self.learning_rate.keys()))):
                if min_num_updates >= t:
                    learning_rate = learning_rate[t]
                    break

        tau = self.config.mechanism.probabilistic_cfg.tau
        safety_margin = tau * self.config.alpha
        if noisy_result > predicted_output + safety_margin:
            # Increase weights if predicted_output is too small
            lr = learning_rate / 8
        elif noisy_result < predicted_output - safety_margin:
            lr = -learning_rate / 8
        else:
            return 0

        # Multiplicative weights update for the relevant bins
        cache_entry.histogram.tensor = torch.mul(
            cache_entry.histogram.tensor, torch.exp(query_tensor_dense * lr)
        )

        if self.heuristic_type == "global_visits":
            query_tensor_dense = torch.ones(
                size=(1, self.domain_size), dtype=torch.float64
            )
        # This depends on Query Values being 1 (counts queries only) for now
        cache_entry.bin_updates = torch.add(cache_entry.bin_updates, query_tensor_dense)
        cache_entry.histogram.normalize()

        # Write updated entry
        self.write_entry(blocks, cache_entry)

        return int(lr / abs(lr))

    def update_entry_merged_histograms(self, query, histogram_run_ops, sign) -> int:
        query_tensor_dense = query
        cache_entries = []
        learning_rates = []

        # Get all the histograms and learning rates
        for run_op in histogram_run_ops:
            blocks = run_op.blocks
            cache_entry = self.read_entry(blocks)
            if not cache_entry:
                cache_entry = self.create_new_entry(blocks)
            cache_entries.append(cache_entry)

            # Parse learning rate
            learning_rate = self.learning_rate
            if isinstance(self.learning_rate, dict):
                min_num_updates = torch.min(
                    cache_entry.bin_updates[query_tensor_dense > 0]
                ).item()
                for t in reversed(sorted(list(self.learning_rate.keys()))):
                    if min_num_updates >= t:
                        learning_rate = learning_rate[t]
                        break
            learning_rates.append(learning_rate)

        # Take the minimum learning rate across histograms, to be safe
        learning_rate = min(learning_rates)

        # No need for safety margin after a SV check
        lr = sign * learning_rate / 8

        for cache_entry in cache_entries:
            # Multiplicative weights update for the relevant bins
            cache_entry.histogram.tensor = torch.mul(
                cache_entry.histogram.tensor, torch.exp(query_tensor_dense * lr)
            )
            cache_entry.bin_updates = torch.add(
                cache_entry.bin_updates, query_tensor_dense
            )
            cache_entry.histogram.normalize()

            # Write updated entry
            self.write_entry(blocks, cache_entry)

        return int(lr / abs(lr))

    def update_entry_threshold(self, blocks, query):
        cache_entry = self.read_entry(blocks)
        assert cache_entry is not None

        if self.heuristic_type == "global_visits":
            query_tensor_dense = torch.ones(
                size=(1, self.domain_size), dtype=torch.float64
            )
        else:
            # TODO: This depends on Query Values being 1 (counts queries only) for now
            query_tensor_dense = query

        new_threshold = (
            torch.min(cache_entry.bin_updates[query_tensor_dense > 0])
            + self.bin_thershold_step
        )

        if self.bin_threshold_max:
            new_threshold = min(new_threshold, self.bin_threshold_max)

        # Keep irrelevant bins as they are - set the rest to 0 and add to them the new threshold
        bin_thresholds_mask = (query_tensor_dense == 0).int()
        cache_entry.bin_thresholds = torch.add(
            torch.mul(cache_entry.bin_thresholds, bin_thresholds_mask),
            query_tensor_dense * new_threshold,
        )

        logger.debug(f"qieru: {query_tensor_dense}")
        logger.debug(f"Updating bin thresholds to {cache_entry.bin_thresholds}")

        # Write updated entry
        self.write_entry(blocks, cache_entry)

    def is_query_hard(self, query, blocks):

        cache_entry = self.read_entry(blocks)
        if not cache_entry:
            return True

        # If each bin has been updated at least <bin-threshold> times the query is easy
        query_tensor_dense = query
        bin_updates_query = torch.mul(cache_entry.bin_updates, query_tensor_dense)
        bin_thresholds_query = torch.mul(cache_entry.bin_thresholds, query_tensor_dense)
        comparisons = bin_updates_query < bin_thresholds_query

        if self.quantile_threshold:
            # Tolerate a small fraction of untrained bins
            untrained_fraction = comparisons.float().sum() / query_tensor_dense.sum()
            if untrained_fraction > self.quantile_threshold:
                return True
        else:
            if torch.any(comparisons).item():
                return True
        return False


class MockHistogramCache(HistogramCache):
    def __init__(self, config):
        super().__init__(config)

    def get_kv_store(self, config):
        return {}

    def write_entry(self, blocks, cache_entry):
        key = CacheKey(blocks).key
        self.kv_store[key] = {
            "histogram": cache_entry.histogram,
            "bin_updates": cache_entry.bin_updates,
            "bin_thresholds": cache_entry.bin_thresholds,
        }

    def read_entry(self, blocks):
        key = CacheKey(blocks).key
        if key in self.kv_store:
            entry = self.kv_store[key]
            return CacheEntry(
                entry["histogram"], entry["bin_updates"], entry["bin_thresholds"]
            )
        return None
