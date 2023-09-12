import json
import typer
from loguru import logger
from turbo.task import Task
from omegaconf import OmegaConf
from turbo.query_processor import QueryProcessor
from turbo.cache.cache import Cache
from turbo.budget_accountant import BudgetAccountant
from turbo.psql import PSQL
from turbo.query_converter import QueryConverter
from turbo.cache.histogram import query_dict_to_list

from turbo.planner.min_cuts import MinCuts

from turbo.utils.utils import DEFAULT_CONFIG_FILE

test = typer.Typer()


@test.command()
def test(
    omegaconf: str = "turbo/config/turbo.json",
):
    omegaconf = OmegaConf.load(omegaconf)
    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    omegaconf = OmegaConf.create(omegaconf)
    config = OmegaConf.merge(default_config, omegaconf)

    try:
        with open(config.blocks.block_metadata_path) as f:
            blocks_metadata = json.load(f)
    except NameError:
        logger.error("Dataset metadata must have be created first..")
    assert blocks_metadata is not None
    config.update({"blocks_metadata": blocks_metadata})

    query = {"0": 0, "1": 0, "2": 0, "3": 5}

    db = PSQL(config)
    budget_accountant = BudgetAccountant(config=config)
    cache = Cache(config)
    planner = MinCuts(cache, budget_accountant, config)
    query_processor = QueryProcessor(db, cache, planner, budget_accountant, config)

    # Initialize Task
    block_data_path = config.blocks.block_data_path + "/block_0.csv"
    db.add_new_block(block_data_path)
    budget_accountant.add_new_block_budget()

    num_requested_blocks = 1
    num_blocks = budget_accountant.get_blocks_count()
    assert num_blocks > 0

    # Latest Blocks first
    requested_blocks = (num_blocks - num_requested_blocks, num_blocks - 1)
    print(requested_blocks)

    utility = 0.05
    utility_beta = 0.001

    attribute_sizes = config.blocks_metadata.attributes_domain_sizes
    query_vector = query_dict_to_list(query, attribute_sizes=attribute_sizes)
    query_converter = QueryConverter(config.blocks_metadata)
    query_tensor = query_converter.convert_to_sparse_tensor(query_vector)
    query_tensor = query_tensor.to_dense()
    query_db_format = query_converter.convert_to_sql(query_vector, requested_blocks)

    task = Task(
        id=0,
        query_id=0,
        query_type="linear",
        query=query_tensor,
        query_db_format=query_db_format,
        blocks=requested_blocks,
        n_blocks=num_requested_blocks,
        utility=utility,
        utility_beta=utility_beta,
        name=0,
    )

    run_metadata = query_processor.try_run_task(task)

    # task = Task(
    #     id=0,
    #     query_id=1,
    #     query_type="linear",
    #     query=query_vector,
    #     blocks=requested_blocks,
    #     n_blocks=num_requested_blocks,
    #     utility=utility,
    #     utility_beta=utility_beta,
    #     name=0,
    # )

    # run_metadata = query_processor.try_run_task(task)
    print(run_metadata)


if __name__ == "__main__":
    test()
