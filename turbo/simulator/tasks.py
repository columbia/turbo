import pandas as pd
from loguru import logger
from itertools import count
from turbo.simulator.resourcemanager import LastItem
from turbo.simulator.task_generator import (
    CSVTaskGenerator,
    PoissonTaskGenerator,
)


class Tasks:
    """Model task arrival rate and privacy demands."""

    def __init__(self, environment, resource_manager):
        self.env = environment
        self.resource_manager = resource_manager
        self.config = resource_manager.config
        self.task_count = count()

        self.tasks_df = pd.read_csv(self.config.tasks.path)

        if "submit_time" in self.tasks_df:
            logger.info("Reading tasks in order with hardcoded arrival times.")
            self.config.tasks.initial_num = 0
            self.config.tasks.max_num = len(self.tasks_df)
            self.task_generator = CSVTaskGenerator(self.tasks_df, self.config)
        else:
            logger.info("Poisson sampling.")
            self.task_generator = PoissonTaskGenerator(
                self.tasks_df, self.config.tasks.avg_num_tasks_per_block, self.config
            )
            assert self.config.tasks.max_num is not None

        self.env.process(self.task_producer())

    def task_producer(self) -> None:
        """Generate tasks."""

        # Wait till blocks initialization is completed
        yield self.resource_manager.blocks_initialized

        task_id = next(self.task_count)

        # Produce initial tasks
        for _ in range(self.config.tasks.initial_num):
            self.task(task_id)
            task_id = next(self.task_count)
        logger.debug("Done producing all the initial tasks.")

        while self.config.tasks.max_num > task_id:
            # No task can arrive after the end of the simulation
            # so we force them to appear right before the end of the last block
            task_arrival_interval = (
                0
                if self.resource_manager.block_production_terminated.triggered
                else self.task_generator.get_task_arrival_interval_time()
            )

            self.task(task_id)
            yield self.env.timeout(task_arrival_interval)
            task_id = next(self.task_count)

        self.resource_manager.task_production_terminated.succeed()
        self.resource_manager.new_tasks_queue.put(LastItem())

        logger.info(
            f"Done generating tasks at time {self.env.now}. Current count is: {task_id}"
        )

    def task(self, task_id: int) -> None:
        """Task behavior. Sets its own demand, notifies resource manager of its existence"""

        blocks_count = self.resource_manager.budget_accountant.get_blocks_count()
        task = self.task_generator.create_task(task_id, blocks_count)

        logger.debug(
            f"Task: {task_id} generated at {self.env.now}. Name: {task.name}. Blocks: {task.blocks}"
        )
        self.resource_manager.new_tasks_queue.put(task)
