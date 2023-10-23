import redis
import yaml

from turbo.core import Accuracy


def cache_key(query_id, data_view_identifier):
    return f"{query_id}:{data_view_identifier}"


class CacheKey:
    def __init__(self, query_id, data_view_identifier):
        self.key = cache_key(query_id, data_view_identifier)


class CacheEntry:
    def __init__(self, accuracy, dp_result):
        self.accuracy = (
            accuracy  # accuracy guarantee for which we computed the dp-result
        )
        self.dp_result = dp_result  # the dp-result


class ExactMatchCache:
    def __init__(self, config):
        self.config = config
        self.kv_store = self.get_kv_store(config)

    def get_kv_store(self, config):
        return redis.Redis(host=config.cache.host, port=config.cache.port, db=0)

    def create_new_entry(self, accuracy, dp_result):
        new_cache_entry = CacheEntry(accuracy, dp_result)
        return new_cache_entry

    def write_entry(self, query_id, data_view_identifier, cache_entry):
        key = CacheKey(query_id, data_view_identifier).key
        self.kv_store.hset(key, "dp_result", cache_entry.dp_result)
        self.kv_store.hset(key, "accuracy_alpha", cache_entry.accuracy.alpha)
        self.kv_store.hset(key, "accuracy_beta", cache_entry.accuracy.beta)

    def read_entry(self, query_id, data_view_identifier):
        key = CacheKey(query_id, data_view_identifier).key
        entry = self.kv_store.hgetall(key)
        if entry:
            return CacheEntry(
                Accuracy(
                    float(entry[b"accuracy_alpha"]), float(entry[b"accuracy_beta"])
                ),
                float(entry[b"dp_result"]),
            )
        return None

    def dump(self):
        pass


class MockExactMatchCache(ExactMatchCache):
    def __init__(self, config):
        super().__init__(config)

    def get_kv_store(self, config):
        return {}

    def write_entry(self, query_id, data_view_identifier, cache_entry):
        key = cache_key(query_id, data_view_identifier)
        self.kv_store[key] = cache_entry

    def read_entry(self, query_id, data_view_identifier):
        key = cache_key(query_id, data_view_identifier)
        return self.kv_store.get(key, None)

    # def dump(self):
    #     print("Cache", yaml.dump(self.key_values))
