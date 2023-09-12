import redis
import yaml


def cache_key(query_id, blocks):
    return f"{query_id}:{blocks}"


class CacheKey:
    def __init__(self, query_id, blocks):
        self.key = cache_key(query_id, blocks)


class CacheEntry:
    def __init__(self, result, noise_std, noise):
        self.result = result  # True result without noise
        self.noise_std = noise_std  # std of Laplace distribution
        self.noise = noise  # The actual noise sampled from the distribution


class ExactMatchCache:
    def __init__(self, config):
        self.config = config
        self.kv_store = self.get_kv_store(config)

    def get_kv_store(self, config):
        return redis.Redis(host=config.cache.host, port=config.cache.port, db=0)

    def write_entry(self, query_id, blocks, cache_entry):
        key = CacheKey(query_id, blocks).key
        self.kv_store.hset(key, "result", cache_entry.result)
        self.kv_store.hset(key, "noise_std", cache_entry.noise_std)
        self.kv_store.hset(key, "noise", cache_entry.noise)

    def read_entry(self, query_id, blocks):
        key = CacheKey(query_id, blocks).key
        entry = self.kv_store.hgetall(key)
        if entry:
            return CacheEntry(
                float(entry[b"result"]),
                float(entry[b"noise_std"]),
                float(entry[b"noise"]),
            )
        return None

    def dump(self):
        pass


class MockExactMatchCache(ExactMatchCache):
    def __init__(self, config):
        super().__init__(config)

    def get_kv_store(self, config):
        return {}

    def write_entry(self, query_id, blocks, cache_entry):
        key = cache_key(query_id, blocks)
        self.kv_store[key] = cache_entry

    def read_entry(self, query_id, blocks):
        key = cache_key(query_id, blocks)
        return self.kv_store.get(key, None)

    def dump(self):
        print("Cache", yaml.dump(self.key_values))
