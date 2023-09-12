from turbo.cache.pmw_cache import MockPMWCache
from turbo.cache.sparse_vectors import MockSparseVectors, SparseVectors
from turbo.cache.exact_match_cache import MockExactMatchCache, ExactMatchCache
from turbo.cache.histogram_cache import MockHistogramCache, HistogramCache


class Cache:
    def __init__(self, config):
        self.config = config
        self.mechanism_type = config.mechanism.type

        if self.mechanism_type == "Laplace":
            if self.config.exact_match_caching:
                self.exact_match_cache = ExactMatchCache(config)
        elif self.mechanism_type == "Hybrid":
            if self.config.exact_match_caching:
                self.exact_match_cache = ExactMatchCache(config)
            self.histogram_cache = HistogramCache(config)
            self.sparse_vectors = SparseVectors(config)


class MockCache:
    def __init__(self, config):
        self.config = config
        self.mechanism_type = config.mechanism.type

        if self.mechanism_type == "Laplace":
            if self.config.exact_match_caching:
                self.exact_match_cache = MockExactMatchCache(config)
        elif self.mechanism_type in {"PMW", "TimestampsPMW"}:
            self.pmw_cache = MockPMWCache(config)
        elif self.mechanism_type == "Hybrid":
            if self.config.exact_match_caching:
                self.exact_match_cache = MockExactMatchCache(config)
            self.histogram_cache = MockHistogramCache(config)
            self.sparse_vectors = MockSparseVectors(config)
