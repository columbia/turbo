from turbo.cache.histogram import build_sparse_tensor


class QueryConverter:
    def __init__(self, config) -> None:
        self.config = config
        self.attribute_names = config.blocks_metadata["attribute_names"]
        self.attribute_domain_sizes = config.blocks_metadata["attributes_domain_sizes"]

    def convert_to_sql(self, query_vector, blocks):
        p = [set() for _ in self.attribute_names]

        for entry in query_vector:
            for i, val in enumerate(entry):
                p[i].add(val)

        in_clauses = []
        for i, s in enumerate(p):
            domain_size = self.attribute_domain_sizes[i]
            if len(s) != domain_size:
                if len(s) == 1:
                    in_clauses += [f"{self.attribute_names[i]} = {tuple(s)[0]} "]
                else:
                    in_clauses += [f"{self.attribute_names[i]} IN {tuple(s)} "]
        where_clause = "AND ".join(in_clauses)
        if not where_clause:
            where_clause = " TRUE"
        sql = f"SELECT COUNT(*) FROM {self.config.postgres.database}_data WHERE {where_clause}"  # {time_window_clause};"
        return sql

    def convert_to_sparse_tensor(self, query_vector, attribute_domain_sizes=None):
        if attribute_domain_sizes is None:
            attribute_domain_sizes = self.attribute_domain_sizes
        # print("length query", len(query_vector))
        tensor = build_sparse_tensor(
            bin_indices=query_vector,
            values=[1.0] * len(query_vector),
            attribute_sizes=attribute_domain_sizes,
        )
        return tensor

    def convert_to_dense_tensor(self, query_vector):
        # print("length query", len(query_vector))
        # print(f"Converting: {query_vector} of type {type(query_vector)}")
        pass
