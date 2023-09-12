import copy
import pickle
import random
import time

import numpy as np
from loguru import logger

from turbo.cache.histogram import query_dict_to_list
from turbo.query_converter import QueryConverter
from turbo.task import Task
from turbo.utils.utils import parse_block_requests_pattern


def Zipf(a: np.float64, min: np.uint64, max: np.uint64, size=None):
    """
    Generate Zipf-like random variables,
    but in inclusive [min...max] interval
    """
    if min == 0:
        raise ZeroDivisionError("")

    v = np.arange(min, max + 1)  # values to sample
    p = 1.0 / np.power(v, a)  # probabilities
    p /= np.sum(p)
    return np.random.choice(v, size=size, replace=True, p=p)


class TaskGenerator:
    def __init__(self, df_tasks, config) -> None:
        self.config = config
        self.tasks = df_tasks
        self.query_converter = QueryConverter(self.config)

    def sample_task_row(self, config):
        raise NotImplementedError("Must override")

    def create_task(self, task_id, num_blocks):
        task_row = self.sample_task_row(self.config).squeeze()
        eligible_block_requests = [
            n for n in self.block_requests_pattern if n <= num_blocks
        ]
        if eligible_block_requests == []:
            logger.error(
                f"There are no tasks in the workload requesting less than {num_blocks} blocks to sample from. \
                    This workload requires at least {self.tasks['n_blocks'].min()} initial blocks"
            )
            exit(1)

        num_requested_blocks = random.choice(eligible_block_requests)
        query_id = int(task_row["query_id"])
        name = task_id if "task_name" not in task_row else task_row["task_name"]

        if self.config.tasks.block_selection_policy == "LatestBlocks":
            requested_blocks = (num_blocks - num_requested_blocks, num_blocks - 1)
        elif self.config.tasks.block_selection_policy == "RandomBlocks":
            start = np.random.randint(0, num_blocks - num_requested_blocks + 1)
            requested_blocks = (start, start + num_requested_blocks - 1)

        query = eval(task_row["query"])

        # Read compressed rectangle or PyTorch slice, output a query vector
        attribute_sizes = self.config.blocks_metadata.attributes_domain_sizes
        query_vector = query_dict_to_list(query, attribute_sizes=attribute_sizes)
        # Query format for running on histograms
        if "query_path" in task_row:
            # Load tensor/query from disk if stored
            with open(task_row["query_path"], "rb") as f:
                query_tensor = pickle.load(f)
        else:
            query_tensor = self.query_converter.convert_to_sparse_tensor(query_vector)
        query_tensor = query_tensor.to_dense()

        # Query format for running using PSQL module (runs on blocks)
        query_db_format = (
            query_tensor
            if self.config.mock
            else self.query_converter.convert_to_sql(query_vector, requested_blocks)
        )

        query = query_tensor

        task = Task(
            id=task_id,
            query_id=query_id,
            query_type=task_row["query_type"],
            query=query,
            query_db_format=query_db_format,
            blocks=requested_blocks,
            n_blocks=num_requested_blocks,
            utility=float(task_row["utility"]),
            utility_beta=float(task_row["utility_beta"]),
            name=name,
        )
        # print("\nTask", task.dump())
        return task


class PoissonTaskGenerator(TaskGenerator):
    def __init__(self, df_tasks, avg_num_tasks_per_block, config) -> None:
        super().__init__(df_tasks, config)
        self.avg_num_tasks_per_block = avg_num_tasks_per_block
        self.tasks = self.tasks.sample(
            frac=1, random_state=config.global_seed
        ).reset_index()

        self.block_requests_pattern = parse_block_requests_pattern(
            self.config.blocks.block_requests_pattern
        )

        logger.debug(f"Parsed block requests pattern:{self.block_requests_pattern}")

        def zipf():
            query_pool_size = len(self.tasks["query_id"].unique())
            min = np.uint64(1)
            max = np.uint64(query_pool_size)
            samples = Zipf(config.tasks.zipf_k, min, max, int(config.tasks.max_num))
            for sample in samples:
                yield sample
            # return samples

        self.zipf_sampling = zipf()
        # print("\n\n\nUnique queries", len([*set(self.zipf_sampling)]))
        # exit()

    def sample_task_row(self, config):
        next_sample_idx = int(next(self.zipf_sampling)) - 1
        return self.tasks.iloc[[next_sample_idx]]

    def get_task_arrival_interval_time(self):
        return random.expovariate(self.avg_num_tasks_per_block)


class CSVTaskGenerator(TaskGenerator):
    def __init__(self, df_tasks, config) -> None:
        super().__init__(df_tasks, config)

    def sample_task_row(self, config):
        yield self.tasks.iterrows()

    def get_task_arrival_interval_time(self):
        yield self.tasks["relative_submit_time"].iteritems()
