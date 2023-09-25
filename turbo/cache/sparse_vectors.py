import time

import numpy as np
import redis
from loguru import logger
from termcolor import colored

from turbo.utils.utility_theorems import get_pmw_epsilon, get_sv_epsilon
from turbo.utils.utils import get_blocks_size


class SparseVector:
    def __init__(self, id, alpha=None, beta=None, n=None, sv_state=None) -> None:
        self.n = n
        self.id = id
        self.beta = beta

        if not sv_state:
            self.alpha = alpha
            self.epsilon = get_sv_epsilon(self.alpha, self.beta, self.n)
            self.b = 1 / (self.n * self.epsilon)
            self.noisy_threshold = None
            self.initialized = False
        else:
            self.alpha = sv_state["alpha"]
            self.epsilon = sv_state["epsilon"]
            self.b = sv_state["b"]
            self.noisy_threshold = sv_state["noisy_threshold"]
            self.initialized = sv_state["initialized"]

    def initialize(self):
        self.noisy_threshold = self.alpha / 2 + np.random.laplace(loc=0, scale=self.b)
        self.initialized = True

    def check(self, true_output, noisy_output):
        assert self.noisy_threshold is not None
        true_error = abs(true_output - noisy_output)
        logger.debug(colored(f"true_error, {true_error}", "yellow"))
        error_noise = np.random.laplace(loc=0, scale=self.b)
        noisy_error = true_error + error_noise
        logger.debug(
            colored(
                f"noisy_error, {noisy_error}, noisy_threshold, {self.noisy_threshold}",
                "yellow",
            )
        )
        if noisy_error < self.noisy_threshold:
            return True
        else:
            return False


class CacheKey:
    def __init__(self, node_id):
        self.key = str(node_id)


class SparseVectors:
    def __init__(self, config):
        self.config = config
        self.kv_store = self.get_kv_store(config)
        self.blocks_metadata = self.config.blocks_metadata

    def get_kv_store(self, config):
        return redis.Redis(host=config.cache.host, port=config.cache.port, db=0)

    def create_new_entry(self, node_id, extra_laplace=True, half_beta=True):
        # node_id covers exactly the requested blocks
        (i, j) = node_id
        n = get_blocks_size((i, j), self.blocks_metadata)
        beta_sv = self.config.beta / 2 if half_beta else self.config.beta

        if extra_laplace:
            # SV with update, like a pmw basically
            epsilon = get_pmw_epsilon(alpha=self.config.alpha, beta=beta_sv, n=n)
            sv_state = {
                "alpha": self.config.alpha,
                "epsilon": epsilon,
                "b": 1 / (n * epsilon),
                "noisy_threshold": None,
                "initialized": False,
            }
            return SparseVector(id=node_id, sv_state=sv_state)

        # SV-on-top
        return SparseVector(
            id=node_id,
            beta=self.config.beta,
            alpha=self.config.alpha,
            n=n,
        )

    def write_entry(self, cache_entry):
        key = CacheKey(cache_entry.id).key
        self.kv_store.hset(key + ":sparse_vector", "epsilon", cache_entry.epsilon)
        self.kv_store.hset(key + ":sparse_vector", "b", cache_entry.b)
        self.kv_store.hset(key + ":sparse_vector", "alpha", cache_entry.alpha)
        self.kv_store.hset(
            key + ":sparse_vector", "noisy_threshold", str(cache_entry.noisy_threshold)
        )
        self.kv_store.hset(
            key + ":sparse_vector", "initialized", int(cache_entry.initialized)
        )

    def read_entry(self, node_id):
        key = CacheKey(node_id).key
        sv_state = {}
        sv_info = self.kv_store.hgetall(key + ":sparse_vector")
        if sv_info:
            sv_state["epsilon"] = float(sv_info[b"epsilon"])
            sv_state["b"] = float(sv_info[b"b"])
            sv_state["alpha"] = float(sv_info[b"alpha"])
            sv_state["noisy_threshold"] = float(sv_info[b"noisy_threshold"])
            sv_state["initialized"] = sv_info[b"initialized"].decode() == "1"
        # print("sv state", sv_state)
        if sv_state:
            return SparseVector(id=node_id, sv_state=sv_state)
        return None


class MockSparseVectors(SparseVectors):
    def __init__(self, config):
        super().__init__(config)

    def get_kv_store(self, config):
        return {}

    def write_entry(self, cache_entry):
        self.kv_store[cache_entry.id] = cache_entry

    def read_entry(self, node_id):
        if node_id in self.kv_store:
            return self.kv_store[node_id]
        return None
