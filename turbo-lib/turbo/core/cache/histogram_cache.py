import math

import torch

# import redisai as rai
from turbo.core.cache.histogram import DenseHistogram


class CacheKey:
    def __init__(self, data_view_identifier):
        self.key = str(data_view_identifier)


class CacheEntry:
    def __init__(self, histogram, bin_updates, bin_thresholds) -> None:
        self.histogram = histogram
        self.bin_updates = bin_updates
        self.bin_thresholds = bin_thresholds


class HistogramCache:
    def __init__(self, config):
        self.config = config

        self.kv_store = self.get_kv_store(self.config)

        data_domain_info = self.config.data_domain_info
        # attribute_names = list(data_domain_info.keys())
        attributes_domain_sizes = [
            attr["domain_size"] for attr in data_domain_info.values()
        ]
        self.domain_size = math.prod(attributes_domain_sizes)

        _, heuristic_params = self.config.histogram_cfg.heuristic.split(":")
        threshold, step = heuristic_params.split("-")
        self.bin_thershold = int(threshold)
        self.bin_thershold_step = int(step)

        self.learning_rate = self.config.histogram_cfg.learning_rate
        if isinstance(self.learning_rate, str):
            lrs = {}
            for lr in self.learning_rate.split("_"):
                x = lr.split(":")
                lrs[float(x[0])] = float(x[1])
            self.learning_rate = lrs

    def get_kv_store(self, config):
        return rai.Client(host=config.cache.host, port=config.cache.port, db=0)

    def write_entry(self, data_view_identifier, cache_entry):
        key = CacheKey(data_view_identifier).key
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

    def read_entry(self, data_view_identifier):
        key = CacheKey(data_view_identifier).key
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

    def create_new_entry(
        self,
    ):
        new_cache_entry = CacheEntry(
            histogram=DenseHistogram(self.domain_size),
            bin_updates=torch.zeros(size=(1, self.domain_size), dtype=torch.float64),
            bin_thresholds=torch.ones(size=(1, self.domain_size), dtype=torch.float64)
            * self.bin_thershold,
        )
        return new_cache_entry

    def update_entry_histogram(
        self,
        query: torch.Tensor,
        data_view_identifier: str,
        noisy_result: float,
        use_safety_margin: bool = True,
    ) -> int:
        cache_entry = self.read_entry(data_view_identifier)
        if not cache_entry:
            cache_entry = self.create_new_entry()

        # Do External Update on the histogram - update bin counts too
        predicted_output = cache_entry.histogram.run(query)

        # Parse learning rate
        learning_rate = self.learning_rate
        if isinstance(self.learning_rate, dict):
            min_num_updates = torch.min(cache_entry.bin_updates[query > 0]).item()
            for t in reversed(sorted(list(self.learning_rate.keys()))):
                if min_num_updates >= t:
                    learning_rate = learning_rate[t]
                    break

        safety_margin = (
            self.config.histogram_cfg.tau * self.config.alpha
            if use_safety_margin
            else 0
        )
        if noisy_result > predicted_output + safety_margin:
            # Increase weights if predicted_output is too small
            lr = learning_rate / 8
        elif noisy_result < predicted_output - safety_margin:
            lr = -learning_rate / 8
        else:
            return 0

        # Multiplicative weights update for the relevant bins
        cache_entry.histogram.tensor = torch.mul(
            cache_entry.histogram.tensor, torch.exp(query * lr)
        )
        # NOTE: This depends on Query Values being 1 (counts queries only) for now
        cache_entry.bin_updates = torch.add(cache_entry.bin_updates, query)
        cache_entry.histogram.normalize()

        # Write updated entry
        self.write_entry(data_view_identifier, cache_entry)

        return int(lr / abs(lr))

    def update_entry_threshold(self, data_view_identifier: str, query: torch.Tensor):
        cache_entry = self.read_entry(data_view_identifier)
        assert cache_entry is not None

        # NOTE: This depends on Query Values being 1 (counts queries only) for now
        new_threshold = (
            torch.min(cache_entry.bin_updates[query > 0]) + self.bin_thershold_step
        )
        bin_thresholds_mask = (query == 0).int()
        cache_entry.bin_thresholds = torch.add(
            torch.mul(cache_entry.bin_thresholds, bin_thresholds_mask),
            query * new_threshold,
        )
        # Write updated entry
        self.write_entry(data_view_identifier, cache_entry)

    def is_histogram_ready(self, data_view_identifier: str, query: torch.Tensor):
        cache_entry = self.read_entry(data_view_identifier)
        if not cache_entry:
            cache_entry = self.create_new_entry()
            self.write_entry(data_view_identifier, cache_entry)

        # If each bin has been updated at least <bin-threshold> times the query is easy
        bin_updates_query = torch.mul(cache_entry.bin_updates, query)
        bin_thresholds_query = torch.mul(cache_entry.bin_thresholds, query)
        comparisons = bin_updates_query < bin_thresholds_query
        if torch.any(comparisons).item():
            return False
        return True


class MockHistogramCache(HistogramCache):
    def __init__(self, config):
        super().__init__(config)

    def get_kv_store(self, config):
        return {}

    def write_entry(self, data_view_identifier, cache_entry):
        key = CacheKey(data_view_identifier).key
        self.kv_store[key] = {
            "histogram": cache_entry.histogram,
            "bin_updates": cache_entry.bin_updates,
            "bin_thresholds": cache_entry.bin_thresholds,
        }

    def read_entry(self, data_view_identifier):
        key = CacheKey(data_view_identifier).key
        if key in self.kv_store:
            entry = self.kv_store[key]
            return CacheEntry(
                entry["histogram"], entry["bin_updates"], entry["bin_thresholds"]
            )
        return None
