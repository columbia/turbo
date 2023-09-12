import os
import sys
import typer
import json
import random
import numpy as np
from loguru import logger
from pathlib import Path
from omegaconf import OmegaConf
import multiprocessing

from turbo.server_tasks import TasksServer
from turbo.server_blocks import BlocksServer
from turbo.query_processor import QueryProcessor
from turbo.psql import PSQL, MockPSQL
from turbo.planner.max_cuts import MaxCuts
from turbo.planner.min_cuts import MinCuts
from turbo.planner.no_cuts import NoCuts
from turbo.cache.cache import Cache, MockCache
from turbo.budget_accountant import BudgetAccountant, MockBudgetAccountant

from turbo.utils.utils import (
    DEFAULT_CONFIG_FILE,
    REPO_ROOT,
)


app = typer.Typer()


def turbo(omegaconf):
    # Initialize configuration
    default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
    omegaconf = OmegaConf.create(omegaconf)
    config = OmegaConf.merge(default_config, omegaconf)
  
    for a, b in [
        ("blocks", "block_data_path"),
        ("blocks", "block_metadata_path"),
        ("tasks", "path"),
    ]:
        p = Path(config[a][b])
        config[a][b] = str(REPO_ROOT.joinpath(p))

    try:
        with open(config.blocks.block_metadata_path) as f:
            blocks_metadata = json.load(f)
    except Exception as e:
        logger.error("Dataset metadata must have been created first..")
        raise e

    assert blocks_metadata is not None
    config.update({"blocks_metadata": blocks_metadata})

    pmw_attribute_names = config.blocks_metadata.attribute_names
    pmw_attributes_domain_sizes = (
        config.blocks_metadata.attributes_domain_sizes
    )
    pmw_domain_size = config.blocks_metadata.domain_size

    config.blocks_metadata.update(
        {
            "pmw_attribute_names": pmw_attribute_names,
            "pmw_attributes_domain_sizes": pmw_attributes_domain_sizes,
            "pmw_domain_size": pmw_domain_size,
        }
    )

    if config.enable_random_seed:
        random.seed(None)
        np.random.seed(None)
    else:
        random.seed(config.global_seed)
        np.random.seed(config.global_seed)

    if config.mock:
        db = MockPSQL(config)
        budget_accountant = MockBudgetAccountant(config)
        cache = MockCache(config)
    else:
        db = PSQL(config)
        budget_accountant = BudgetAccountant(config)
        cache = Cache(config)

    if config.planner.method == "MinCuts":
        planner = MinCuts(cache, budget_accountant, config)
    elif config.planner.method == "NoCuts":
        planner = NoCuts(cache, budget_accountant, config)
    elif config.planner.method == "MaxCuts":
        planner = MaxCuts(cache, budget_accountant, config)

    query_processor = QueryProcessor(
        db, cache, planner, budget_accountant, config
    )

    blocks_server = multiprocessing.Process(target=BlocksServer(db, budget_accountant, config).run())
    tasks_server = multiprocessing.Process(target=TasksServer(query_processor, budget_accountant, config).run())

    # Start the processes
    blocks_server.start()
    tasks_server.start()

    # Wait for both processes to finish
    blocks_server.join()
    tasks_server.join()


@app.command()
def run(
    omegaconf: str = "turbo/config/turbo_server.json",
    loguru_level: str = "INFO",
):
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    omegaconf = OmegaConf.load(omegaconf)
    turbo(omegaconf)


if __name__ == "__main__":
    app()
