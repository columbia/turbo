import json
import os
import random
import time
from pathlib import Path

import mlflow
import numpy as np
import simpy
import typer
from loguru import logger
from omegaconf import OmegaConf

from turbo.budget_accountant import BudgetAccountant, MockBudgetAccountant
from turbo.cache.cache import Cache, MockCache
from turbo.planner.max_cuts import MaxCuts
from turbo.planner.min_cuts import MinCuts
from turbo.planner.no_cuts import NoCuts
from turbo.psql import PSQL, MockPSQL
from turbo.query_processor import QueryProcessor
from turbo.simulator import Blocks, ResourceManager, Tasks
from turbo.utils.utils import (
    DEFAULT_CONFIG_FILE,
    LOGS_PATH,
    REPO_ROOT,
    get_logs,
    mlflow_log,
    save_logs,
    set_run_key,
)

app = typer.Typer()


class Simulator:
    def __init__(self, omegaconf):
        self.env = simpy.Environment()

        # Initialize configuration
        default_config = OmegaConf.load(DEFAULT_CONFIG_FILE)
        omegaconf = OmegaConf.create(omegaconf)
        self.config = OmegaConf.merge(default_config, omegaconf)
        # logger.info(f"Configuration: {self.config}")

        if self.config.logs.print_pid:
            # PID for profiling, sleep a bit to give time to attach the profiler
            print(f"PID: {os.getpid()}")
            time.sleep(3)

        if self.config.logs.mlflow:
            os.environ["MLFLOW_TRACKING_URI"] = str(LOGS_PATH.joinpath("mlruns"))
            try:
                mlflow.set_experiment(
                    experiment_name=self.config.logs.mlflow_experiment_id
                )
            except Exception:
                experiment_id = mlflow.create_experiment(
                    name=self.config.logs.mlflow_experiment_id
                )
                print(f"New MLflow experiment created: {experiment_id}")

        for a, b in [
            ("blocks", "block_data_path"),
            ("blocks", "block_metadata_path"),
            ("tasks", "path"),
        ]:
            p = Path(self.config[a][b])
            self.config[a][b] = str(REPO_ROOT.joinpath(p))

        try:
            with open(self.config.blocks.block_metadata_path) as f:
                blocks_metadata = json.load(f)
        except Exception as e:
            logger.error("Dataset metadata must have been created first..")
            raise e

        assert blocks_metadata is not None
        self.config.update({"blocks_metadata": blocks_metadata})

        pmw_attribute_names = self.config.blocks_metadata.attribute_names
        pmw_attributes_domain_sizes = (
            self.config.blocks_metadata.attributes_domain_sizes
        )
        pmw_domain_size = self.config.blocks_metadata.domain_size

        self.config.blocks_metadata.update(
            {
                "pmw_attribute_names": pmw_attribute_names,
                "pmw_attributes_domain_sizes": pmw_attributes_domain_sizes,
                "pmw_domain_size": pmw_domain_size,
            }
        )

        if self.config.enable_random_seed:
            random.seed(None)
            np.random.seed(None)
        else:
            random.seed(self.config.global_seed)
            np.random.seed(self.config.global_seed)

        # Initialize all components
        if self.config.mock:
            db = MockPSQL(self.config)
            budget_accountant = MockBudgetAccountant(self.config)
            cache = MockCache(self.config)
        else:
            db = PSQL(self.config)
            budget_accountant = BudgetAccountant(self.config)
            cache = Cache(self.config)

        if self.config.planner.method == "MinCuts":
            planner = MinCuts(cache, budget_accountant, self.config)
        elif self.config.planner.method == "NoCuts":
            planner = NoCuts(cache, budget_accountant, self.config)
        elif self.config.planner.method == "MaxCuts":
            planner = MaxCuts(cache, budget_accountant, self.config)

        query_processor = QueryProcessor(
            db, cache, planner, budget_accountant, self.config
        )

        # Start the block and tasks consumers
        self.rm = ResourceManager(
            self.env, db, budget_accountant, query_processor, self.config
        )
        self.env.process(self.rm.start())

        # Start the block and tasks producers
        Blocks(self.env, self.rm)
        Tasks(self.env, self.rm)

    def run(self):
        logs = None
        config = OmegaConf.to_object(self.config)
        config["blocks_metadata"] = {}
        config["blocks"]["block_requests_pattern"] = {}

        key, _, _, _, _, _, _ = set_run_key(config)
        key += "_zip_" + str(config["tasks"]["zipf_k"])

        def _run():
            self.env.run()
            logs = get_logs(
                self.rm.query_processor.tasks_info,
                self.rm.budget_accountant.dump(),
                config,
            )
            if self.config.logs.save:
                save_dir = (
                    self.config.logs.save_dir if self.config.logs.save_dir else ""
                )
                save_logs(logs, save_dir)
            return logs

        if self.config.logs.mlflow:
            with mlflow.start_run(run_name=key):
                mlflow.log_params(config)
                for (k, v) in self.config.mechanism.probabilistic_cfg.items():
                    mlflow.log_param(k, v)
                # mlflow.log_param(
                #     "lr", self.config.mechanism.probabilistic_cfg.learning_rate
                # )
                # mlflow.log_param(
                #     "heuristic", self.config.mechanism.probabilistic_cfg.heuristic
                # )
                mlflow.log_param(
                    "block_requests_pattern",
                    str(self.config.blocks.block_requests_pattern),
                )

                if isinstance(self.config.blocks.block_requests_pattern, str):
                    (
                        distribution,
                        std,
                        mean,
                    ) = self.config.blocks.block_requests_pattern.split("-")
                    std, mean = int(std), int(mean)
                    assert distribution == "dgaussian"
                    mlflow.log_param("block_requests_pattern_std", std)
                    mlflow.log_param("block_requests_pattern_mean", mean)

                mlflow.log_param(
                    "block_selection_policy", self.config.tasks.block_selection_policy
                )
                mlflow.log_param("planner_method", self.config.planner.method)
                mlflow.log_param("zipf_k", self.config.tasks.zipf_k)

                logs = _run()
        else:
            logs = _run()
        return logs


@app.command()
def run_simulation(
    omegaconf: str = "turbo/config/turbo.json",
    loguru_level: str = "ERROR",
):
    os.environ["LOGURU_LEVEL"] = loguru_level
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

    omegaconf = OmegaConf.load(omegaconf)
    logs = Simulator(omegaconf).run()
    return logs


if __name__ == "__main__":
    app()
