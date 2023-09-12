import json
from itertools import chain, combinations, product
from pathlib import Path

import typer
from loguru import logger

from turbo.cache.histogram import k_way_marginal_query_list
from turbo.utils.utils import REPO_ROOT

app = typer.Typer()


def powerset(iter):
    s = list(iter)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


# Compact dictionary representation
def create_all_queries(attributes_domain_sizes):
    queries = []
    attr_values = []
    for domain_size in attributes_domain_sizes:
        attr_values += [list(powerset(range(domain_size)))[1:]]  # Exclude empty set
    queries = list(product(*attr_values))
    query_dicts = []
    for query in queries:
        query_dict = {}
        for attr, value in enumerate(query):
            value = value[0] if len(value) == 1 else list(value)
            query_dict[str(attr)] = value
        query_dicts.append(query_dict)

    return query_dicts


def create_all_point_queries(attributes_domain_sizes):
    attribute_values = [
        list(range(attribute_size)) for attribute_size in attributes_domain_sizes
    ]
    query_tensors = []
    # Take the cartesian product
    for query_tuple in product(*attribute_values):
        # (0,0,1) -> [0,0,1]
        query_tensors.append(list(query_tuple))
    return query_tensors


def create_all_2way_marginals(attributes_domain_sizes):
    query_tensors = []
    dicts = []
    # Conjunctions of 0 with something else, then 1 with something else than 0, etc.
    for i in range(4):
        for a in range(attributes_domain_sizes[i]):
            for j in range(i + 1, 4):
                for b in range(attributes_domain_sizes[j]):
                    print(f"{i}: {a}, {j}: {b}")
                    dicts.append({i: a, j: b})
    # Only 84 2-way marginals! Pretty small, let's see if that's enough for PMW to shine
    print(f"We have {len(dicts)} 2-way marginals. Computing the tensors...")
    for d in dicts:
        query_tensors.append(k_way_marginal_query_list(d, attributes_domain_sizes))
    return query_tensors


def create_2way_marginal_mix(attributes_domain_sizes):
    query_tensors = []
    dicts = [
        {0: 1, 1: 0},
        {0: 1, 1: 1},
        {0: 1, 2: 0},
        {0: 1, 2: 1},
        {1: 0, 2: 0},
        {1: 1, 2: 1},
    ]
    for d in dicts:
        query_tensors.append(k_way_marginal_query_list(d, attributes_domain_sizes))
    return query_tensors


def create_all_3way_marginals(attributes_domain_sizes):
    query_tensors = []
    dicts = []
    # Conjunctions of 0 with something else, then 1 with something else than 0, etc.
    for i in range(4):
        for a in range(attributes_domain_sizes[i]):
            for j in range(i + 1, 4):
                for b in range(attributes_domain_sizes[j]):
                    for k in range(j + 1, 4):
                        for c in range(attributes_domain_sizes[k]):
                            print(f"{i}: {a}, {j}: {b}, {k}: {c}")
                            dicts.append({i: a, j: b, k: c})
    # 176 only
    print(f"We have {len(dicts)} 3-way marginals. Computing the tensors...")
    for d in dicts:
        query_tensors.append(k_way_marginal_query_list(d, attributes_domain_sizes))
    return query_tensors


def create_8_queries():
    # number of people who tested positive AND are females AND 18-49 AND any ethnicity
    q1 = [[1], [1], [1], [0, 1, 2, 3, 4, 5, 6, 7]]
    # are positive AND are either female or male AND are either one of the 4 age groups AND are either one of the 8 ethnicities
    q2 = [[1], [0, 1], [0, 1, 2, 3], [0, 1, 2, 3, 4, 5, 6, 7]]
    # got tested AND male or female and 18-49 and white
    q3 = [[0, 1], [0, 1], [1], [6]]
    # number of people who tested positive AND are male AND 50-64 AND American Indian, Latino, Black,  or Multi-Race
    q4 = [[1], [0], [2], [0, 2, 3, 7]]
    # number of people who tested positive AND are female AND any age AND American Indian, Latino, Black, or Multi-Race
    q5 = [[1], [1], [0, 1, 2, 3], [0, 2, 3, 7]]
    # number of people who tested negative AND are male AND 18-49 AND white or asian
    q6 = [[0], [0], [1], [1, 6]]
    # got tested AND male and 18-49 and white
    q7 = [[0, 1], [0], [1], [1, 6]]
    # number of people who got tested  AND are female AND any age AND American Indian, Latino, Black, or Multi-Race
    q8 = [[[0, 1], [1], [0, 1, 2, 3], [0, 2, 3, 7]]]
    queries = [q1, q2, q3, q4, q5, q6, q7, q8]
    query_tensors = []
    for query in queries:
        query = [list(tup) for tup in query]
        query = [list(tup) for tup in product(*query)]
        query_tensors.append(query)
    return query_tensors


def write_queries(queries_dir, workload, query_tensors):
    queries_path = Path(queries_dir).joinpath(f"{workload}.queries.json")

    queries = []
    # Write queries to a json file
    queries = {}
    for i, query in enumerate(query_tensors):
        queries[i] = {"query": query}
    json_object = json.dumps(queries, indent=4)
    # Writing to queries.json
    with open(queries_path, "w") as outfile:
        outfile.write(json_object)


def main(
    queries_dir: str = REPO_ROOT.joinpath("data/covid19/covid19_queries"),
    blocks_metadata_path: str = REPO_ROOT.joinpath(
        "data/covid19/covid19_data/blocks/metadata.json"
    ),
):

    try:
        with open(blocks_metadata_path) as f:
            blocks_metadata = json.load(f)
    except NameError:
        logger.error("Dataset metadata must have be created first..")
        exit(1)

    attributes_domain_sizes = blocks_metadata["attributes_domain_sizes"]

    # Create all types
    # query_tensors = create_8_queries()
    # write_queries(queries_dir, "8", query_tensors)

    # query_tensors = create_all_point_queries(attributes_domain_sizes)
    # write_queries(queries_dir, "all_point_queries", query_tensors)

    # query_tensors = create_all_2way_marginals(attributes_domain_sizes)
    # write_queries(queries_dir, "all_2way_marginals", query_tensors)

    # query_tensors = create_2way_marginal_mix(attributes_domain_sizes)
    # write_queries(queries_dir, "2way_marginal_mix", query_tensors)

    # query_tensors = create_all_3way_marginals(attributes_domain_sizes)
    # write_queries(queries_dir, "all_3way_marginals", query_tensors)

    # query_tensors = create_all_2way_marginals(attributes_domain_sizes)
    # query_tensors += create_all_3way_marginals(attributes_domain_sizes)
    # write_queries(queries_dir, "all_2way_3way_marginals", query_tensors)

    query_tensors = create_all_queries(attributes_domain_sizes)
    write_queries(queries_dir, "all", query_tensors)

    # query_tuples = create_compact_marginals(attributes_domain_sizes)
    # write_queries(queries_dir, "compact_marginals", query_tuples)


if __name__ == "__main__":
    typer.run(main)
