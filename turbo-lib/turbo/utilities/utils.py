import math
import torch
from typing import List, Dict
from itertools import product


def filter_dict_to_vector(
    attribute_to_value: Dict[str, List[int]],
    attr_domain_sizes: List[int],
):
    """
    Takes a dictionary of {attribute: value} or {attribute: [value_1, ..., value_n]}
    Outputs a query, as a list of bins on which the query has value 1.
    Examples for `attribute_to_value`:
    {0:1} to count all positive cases. Will output something like:
        [[0,0], [0,1], [0,2]] if you just have 2 attributes
    {0:0, 1:0} to count all negative males
    {0:0, 1:0, 3:1} to count all negative asian males
    {0:0, 1:0, 3:[1,6]} to count all negative asian or white males
    """
    # List of domains. E.g. positive = [1], gender = [0,1], ethnicity = [1,2,3]

    domain_per_attribute = []
    for attribute, size in enumerate(attr_domain_sizes):
        if str(attribute) in attribute_to_value:
            # This attribute can only take values from the list
            values = attribute_to_value[str(attribute)]
            if not isinstance(values, list):
                values = [values]
            domain_per_attribute.append(values)
        else:
            # This attribute can take any value
            domain_per_attribute.append(list(range(size)))
    # Now we take the cartesian product of the attributes domains
    return list(product(*domain_per_attribute))


def filter_dict_to_tensor(
    where_dict: Dict[str, List[int]], data_domain_info: any
) -> torch.Tensor:

    # Get attribute domain sizes
    attr_domain_sizes = [attr["domain_size"] for attr in data_domain_info.values()]

    # Convert where dict to query vector
    query_vector = filter_dict_to_vector(where_dict, attr_domain_sizes)

    # Convert query vector to tensor
    tensor = build_sparse_tensor(
        bin_indices=query_vector,
        values=[1.0] * len(query_vector),
        attribute_sizes=attr_domain_sizes,
    )
    return tensor


def get_domain_size(attribute_sizes: List[int]) -> int:
    return math.prod(attribute_sizes)


def get_flat_bin_index(
    multidim_bin_index: List[int], attribute_sizes: List[int]
) -> int:
    index = 0
    size = 1
    for dim in range(len(attribute_sizes) - 1, -1, -1):
        index += multidim_bin_index[dim] * size
        size *= attribute_sizes[dim]
    return index


def build_sparse_tensor(
    bin_indices: List[List[int]], values: List, attribute_sizes: List[int]
):
    # One row only
    column_ids = []
    column_values = []

    for b, v in zip(bin_indices, values):
        column_ids.append(get_flat_bin_index(b, attribute_sizes))
        column_values.append(v)  # In case we lose the correct order?

    return torch.sparse_coo_tensor(
        [[0] * len(column_ids), column_ids],
        column_values,
        size=(1, get_domain_size(attribute_sizes)),
        dtype=torch.float64,
    ).coalesce()  # Sums duplicated bins & puts indices in lexicographic order


def to_tensor(data_domain_info, filter_clause):

    query_dict = {}

    for attr_name, attr_values in filter_clause.items():
        attribute_position = str(data_domain_info[attr_name]["position"])
        encoding = data_domain_info[attr_name]["encoding"]
        for attr_value in attr_values:
            attribute_value_position = int(encoding[attr_value])
            if attribute_position not in query_dict:
                query_dict[attribute_position] = [attribute_value_position]
            else:
                query_dict[attribute_position].append([attribute_value_position])

    for k, v in query_dict.items():
        query_dict[k] = sorted(v)

    # NOTE: only counts for now
    query_tensor = filter_dict_to_tensor(query_dict, data_domain_info).to_dense()

    return query_tensor


def hash(aggregation_type, filter_clause):
    # Normalize the attribute names and hash the where dict to get a unique query id
    # "Query id example: count-{'5': [0], '6': [2], '7': [0]}"

    sorted_keys = sorted(list(filter_clause.keys()))
    sorted_filter_dict = {key: filter_clause[key] for key in sorted_keys}
    return f"{aggregation_type}-{str(sorted_filter_dict)}"


def get_data_domain_info(attributes_info):
    # Convert attributes_info to a more convenient format
    data_domain_info = {}
    for i, (attribute, values) in enumerate(attributes_info):
        encoding = {value: j for j, value in enumerate(values)}
        data_domain_info[attribute] = {
            "position": i,
            "domain_size": len(values),
            "encoding": encoding,
        }
    return data_domain_info
