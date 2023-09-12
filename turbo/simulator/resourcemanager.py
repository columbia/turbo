import pandas as pd
import simpy
from loguru import logger

from turbo.simulator.task_generator import PoissonTaskGenerator
from turbo.utils.utils import mlflow_log


class LastItem:
    def __init__(self):
        return


class ResourceManager:
    """
    Managing blocks and tasks arrival and schedules incoming tasks.
    """

    def __init__(self, environment, db, budget_accountant, query_processor, config):
        self.env = environment
        self.config = config

        self.query_processor = query_processor
        self.budget_accountant = budget_accountant
        self.db = db

        # # To store the incoming tasks and blocks
        self.new_tasks_queue = simpy.Store(self.env)
        self.new_blocks_queue = simpy.Store(self.env)

        self.blocks_initialized = self.env.event()

        # Stopping conditions
        self.block_production_terminated = self.env.event()
        self.task_production_terminated = self.env.event()
        self.block_consumption_terminated = self.env.event()
        self.task_consumption_terminated = self.env.event()

        # Dirty validation metadata
        self.score_thresholds = {
            0.5: False,
            0.9: False,
            0.95: False,
            0.99: False,
        }

    def start(self):
        self.daemon_clock = self.env.process(self.daemon_clock())

        self.env.process(self.block_consumer())
        self.env.process(self.task_consumer())

        # Termination conditions
        yield self.block_production_terminated
        yield self.task_production_terminated
        yield self.block_consumption_terminated
        yield self.task_consumption_terminated
        self.daemon_clock.interrupt()
        logger.info(f"Terminating the simulation at {self.env.now}. Closing...")

    def daemon_clock(self):
        while True:
            try:
                yield self.env.timeout(1)
                logger.info(f"Simulation Time is: {self.env.now}")
            except simpy.Interrupt as i:
                return

    def block_consumer(self):
        while True:
            block_message = yield self.new_blocks_queue.get()

            if isinstance(block_message, LastItem):
                logger.info("Done consuming blocks.")
                self.block_consumption_terminated.succeed()
                return

            block_id = block_message
            block_data_path = self.config.blocks.block_data_path + f"/block_{block_id}"
            self.db.add_new_block(block_data_path)
            self.budget_accountant.add_new_block_budget()

            if self.config.blocks.initial_num == block_id + 1:
                self.blocks_initialized.succeed()

    def task_consumer(self):
        while True:
            task_message = yield self.new_tasks_queue.get()

            if isinstance(task_message, LastItem):
                logger.info("Done consuming tasks")
                self.task_consumption_terminated.succeed()
                return

            task = task_message
            self.query_processor.try_run_task(task)

            # Validation. Pretty ugly way to plug things...
            if (
                self.config.logs.validation_interval
                and (task.id % self.config.logs.validation_interval) == 0
                and task.id > 0
            ):
                if not hasattr(self, "validation_task_pool"):
                    # Generate tasks only once
                    self.validation_task_pool = self.generate_validation_task_pool()
                validation_hit_rate = self.query_processor.validate(
                    self.validation_task_pool
                )
                mlflow_log("validation_hit_rate", validation_hit_rate, task.id)

                for score_threshold in self.score_thresholds.keys():
                    if (
                        self.score_thresholds[score_threshold] == False
                        and validation_hit_rate >= score_threshold
                    ):
                        # We passed a threshold for the first time
                        mlflow_log(
                            f"validation_hit_rate_threshold_{score_threshold}_time",
                            task.id,
                            0,
                        )
                        mlflow_log(
                            f"validation_hit_rate_threshold_{score_threshold}_budget",
                            self.query_processor.total_budget_spent_all_blocks,
                            0,
                        )

                        self.score_thresholds[score_threshold] = True

    def generate_validation_task_pool(self):
        # Dummy version of task_producer
        task_pool = []
        tasks_df = pd.read_csv(self.config.tasks.path)

        task_generator = PoissonTaskGenerator(
            tasks_df, self.config.tasks.avg_num_tasks_per_block, self.config
        )
        for _ in range(self.config.logs.max_validation_tasks):

            blocks_count = self.budget_accountant.get_blocks_count()
            task_id = -1
            task = task_generator.create_task(task_id, blocks_count)
            task_pool.append(task)
        return task_pool
