import math
from itertools import product
from typing import Dict, List, Optional
import torch
from loguru import logger

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


class DenseHistogram:  # We use it to represent the PMW Histogram
    def __init__(
        self,
        domain_size: Optional[int] = None,
        tensor=None,
        attribute_sizes: Optional[List[int]] = None,
    ) -> None:
        # TODO: optimize this later, maybe we only need to store the "diff", which is sparse
        # TODO: keep a multidimensional tensor for efficient marginals
        # TODO: store counts instead of probabilities (int). Potentially with a separate field for the sum
        self.N = domain_size if domain_size else get_domain_size(attribute_sizes)
        self.tensor = (
            tensor
            if tensor is not None
            else torch.ones(
                size=(1, self.N),
                dtype=torch.float64,
            )
        )
        self.normalize()

    def normalize(self) -> None:
        F.normalize(self.tensor, p=1, out=self.tensor)

    def multiply(self, tensor) -> None:
        # elementwise multiplication
        torch.mul(self.tensor, tensor, out=self.tensor)

    def run(self, query: torch.Tensor) -> float:
        # sparse (1,N) x dense (N,1)
        # return torch.smm(query, self.tensor.t()).item()
        return torch.mm(query, self.tensor.t()).item()

    # TODO: Extra optimization: don't even create the query vector
    def run_rectangle(self, marginal_query: Dict[int, int]):
        # Multiple values for a single attribute?
        # attribute_ids, values = marginal_query.items()
        # Torch sum over the histogram
        pass


class SparseHistogram:  # We use it to represent the queries
    def __init__(
        self, bin_indices: List[List[int]], values: List, attribute_sizes: List[int]
    ) -> None:
        # Flat representation of shape (1, N)
        self.tensor = build_sparse_tensor(
            bin_indices=bin_indices,
            values=np.array(values) / sum(values),
            attribute_sizes=attribute_sizes,
        )
        self.domain_size = self.tensor.shape[1]  # N

    @classmethod
    def from_dataframe(
        cls,
        df,
        attribute_domain_sizes,
    ) -> "SparseHistogram":

        cols = list(df.columns)
        df = df.groupby(cols).size()
        return cls(
            bin_indices=list(df.index),  # [(0, 0, 1), (1, 0, 5), (0, 1, 2)],
            values=list(df.values),  # [4, 1, 2],
            attribute_sizes=attribute_domain_sizes,  # [2, 2, 10],
        )

    def dump(self):
        return {
            "id": self.id,
            "initial_budget": self.initial_budget.dump(),
            "budget": self.budget.dump(),
        }

    def run(self, query: torch.Tensor) -> float:
        # `query` has shape (1, N), we need the dot product, or matrix mult with (1,N)x(N,1)
        # return torch.mm(self.tensor, query.t()).item()
        return torch.sparse.mm(self.tensor, query.t()).item()


# ------------- Helper functions ------------- #
def get_flat_bin_index(
    multidim_bin_index: List[int], attribute_sizes: List[int]
) -> int:
    index = 0
    size = 1
    # TODO: write the inverse conversion
    # Row-major order like PyTorch (inner rows first)
    for dim in range(len(attribute_sizes) - 1, -1, -1):
        index += multidim_bin_index[dim] * size
        size *= attribute_sizes[dim]
    return index


def flat_items(sparse_tensor):
    # Iterates through (bin_index, value) pairs
    for index, value in zip(sparse_tensor.indices()[1], sparse_tensor.values()):
        # indices()[0] is always 0 (row tensor)
        yield index, value


def flat_indices(sparse_tensor):
    # Iterates through bin_indices
    for index in sparse_tensor.indices()[1]:
        # indices()[0] is always 0 (row tensor)
        yield (0, index)


def get_domain_size(attribute_sizes: List[int]) -> int:
    return math.prod(attribute_sizes)


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


def build_sparse_tensor_multidim(
    bin_indices: List[List[int]], values: List, attribute_sizes: List[int]
):
    return torch.sparse_coo_tensor(
        list(zip(*bin_indices)),
        values,
        size=attribute_sizes,
        dtype=torch.float64,
    ).coalesce()


def k_way_marginal_query_list(
    attribute_to_value: Dict[int, int],
    attribute_sizes: List[int],
):
    """

    Outputs a query, as a list of bins on which the query has value 1.

    Examples for `attribute_to_value`:
    {0:1} to count all positive cases. Will output something like:
        [[0,0], [0,1], [0,2]] if you just have 2 attributes
    {0:0, 1:0} to count all negative males
    {0:0, 1:0, 3:2} to count all negative asian males

    TODO: if the queries start to be too big, we can implement them with a special class
    """
    # List of domains. E.g. positive = [1], gender = [0,1], ethnicity = [1,2,3]
    domain_per_attribute = []
    for attribute, size in enumerate(attribute_sizes):
        if str(attribute) in attribute_to_value:
            # This attribute is a marginal, we force one value
            domain_per_attribute.append([attribute_to_value[str(attribute)]])
        else:
            # This attribute can take any value
            domain_per_attribute.append(list(range(size)))

    # Now we take the cartesian product of the attributes domains
    return list(product(*domain_per_attribute))


def query_dict_to_list(
    attribute_to_value: Dict[int, int],
    attribute_sizes: List[int],
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
    # Convert to str in case the json isn't parsed properly (shouldn't happen, but better be safe)
    for k in list(attribute_to_value.keys()):
        if not isinstance(k, str):
            attribute_to_value[str(k)] = attribute_to_value.pop(k)
            logger.info(f"Converting key {k} to str")

    # List of domains. E.g. positive = [1], gender = [0,1], ethnicity = [1,2,3]
    domain_per_attribute = []
    for attribute, size in enumerate(attribute_sizes):
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


def query_dict_to_list(
    attribute_to_value: Dict[int, int],
    attribute_sizes: List[int],
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
    # Convert to str in case the json isn't parsed properly (shouldn't happen, but better be safe)
    for k in list(attribute_to_value.keys()):
        if not isinstance(k, str):
            attribute_to_value[str(k)] = attribute_to_value.pop(k)
            logger.info(f"Converting key {k} to str")

    # List of domains. E.g. positive = [1], gender = [0,1], ethnicity = [1,2,3]
    domain_per_attribute = []
    for attribute, size in enumerate(attribute_sizes):
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


# ------------- / Help functions ------------- #
