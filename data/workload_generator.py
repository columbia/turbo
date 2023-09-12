import json
import math
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from turbo.utils.utils import REPO_ROOT

app = typer.Typer()


class Task:
    def __init__(
        self,
        n_blocks,
        query_id,
        utility,
        utility_beta,
        query_type,
        query,
        start_time=None,
    ):
        self.n_blocks = n_blocks
        self.query_id = query_id
        self.utility = utility
        self.utility_beta = utility_beta
        self.query_type = query_type
        self.query = query
        self.start_time = start_time


class QueryPool:
    def __init__(self, attribute_domain_sizes, queries_path) -> None:
        self.attribute_domain_sizes = attribute_domain_sizes
        self.domain_size = math.prod(attribute_domain_sizes)
        self.queries = None
        with open(queries_path) as f:
            self.queries = json.load(f)

    def get_query(self, query_id: int):
        query_id_str = str(query_id)
        if query_id_str in self.queries:
            q = self.queries[query_id_str]
            query = q["query"]
            query_path = q["query_path"] if "query_path" in q else None
        assert query is not None
        return query, query_path


class PrivacyWorkload:
    """
    csv-based privacy workload.
    """

    def __init__(self, blocks_metadata_path, queries):
        try:
            with open(blocks_metadata_path) as f:
                blocks_metadata = json.load(f)
        except NameError:
            logger.error("Dataset metadata must have be created first..")
            exit(1)
        attribute_domain_sizes = blocks_metadata["attributes_domain_sizes"]
        self.query_pool = QueryPool(attribute_domain_sizes, queries)

    def create_dp_task(self, task) -> dict:
        task_name = f"task-{task.query_id}-{task.n_blocks}"
        query, query_path = self.query_pool.get_query(task.query_id)
        dp_task = {
            "query_id": task.query_id,
            "query_type": task.query_type,
            "n_blocks": task.n_blocks,
            "utility": task.utility,
            "utility_beta": task.utility_beta,
            "task_name": task_name,
            "query": query,
        }
        if query_path:
            dp_task.update({"query_path": query_path})
        if task.start_time:
            dp_task.update({"submit_time": task.start_time})
        return dp_task

    def generate_nblocks(self, n_queries, rangelist, utility, utility_beta):
        # Simply lists all the queries, the sampling will happen in the simulator
        self.tasks = []

        # # Every workload has monoblocks
        for b in rangelist:
            for query_id in range(n_queries):
                self.tasks.append(
                    Task(
                        n_blocks=b,
                        query_id=query_id,
                        utility=utility,
                        utility_beta=utility_beta,
                        query_type="linear",
                        query=self.query_pool.get_query(query_id),
                    )
                )
        dp_tasks = [self.create_dp_task(t) for t in self.tasks]
        logger.info(f"Collecting results in a dataframe...")

        self.tasks = pd.DataFrame(dp_tasks)

        logger.info(self.tasks.head())

    def dump(self, path):
        logger.info("Saving the privacy workload...")
        self.tasks.to_csv(path, index=False)
        logger.info(f"Saved {len(self.tasks)} tasks at {path}.")


def main(
    requests_type: str = "1",  # 1:1:1:2:4:8:16:32  # 3/8 to select 1 block
    utility: float = 0.05,
    utility_beta: float = 0.001,
    queries: str = "data/covid19/covid19_queries/all.queries.json",
    workload_dir: str = "data/covid19/covid19_workload",
    blocks_metadata_path: str = "data/covid19/covid19_data/blocks/metadata.json",
) -> None:

    queries = REPO_ROOT.joinpath(queries)
    blocks_metadata_path = REPO_ROOT.joinpath(blocks_metadata_path)
    workload_dir = REPO_ROOT.joinpath(workload_dir)
    workload_dir.mkdir(parents=True, exist_ok=True)

    privacy_workload = PrivacyWorkload(blocks_metadata_path, queries)
    rangelist = list(requests_type.split(":"))
    n_different_queries = len(json.load(open(queries, "r")))
    privacy_workload.generate_nblocks(
        n_different_queries, rangelist, utility, utility_beta
    )
    path = f"{workload_dir}/{n_different_queries}queries.privacy_tasks.csv"
    privacy_workload.dump(path=path)


if __name__ == "__main__":
    typer.run(main)
