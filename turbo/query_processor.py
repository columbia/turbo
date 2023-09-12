import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
from loguru import logger
from termcolor import colored

from turbo.executor import A, Executor, RunHistogram
from turbo.task import Task, TaskInfo
from turbo.utils.logs import compute_hit_scores
from turbo.utils.utils import (
    FAILED,
    FINISHED,
    HISTOGRAM_RUNTYPE,
    LAPLACE_RUNTYPE,
    LOGS_PATH,
    PMW_RUNTYPE,
    mlflow_log,
)

SLIDING_WINDOW_LENGTH = 1_000


class QueryProcessor:
    def __init__(self, db, cache, planner, budget_accountant, config):
        self.config = config
        self.db = db
        self.cache = cache
        self.planner = planner
        self.budget_accountant = budget_accountant
        self.executor = Executor(self.cache, self.db, self.budget_accountant, config)

        self.tasks_info = []
        self.total_budget_spent_all_blocks = 0  # ZeroCurve()

        self.score_counters = defaultdict(int)
        self.score_sliding_windows = defaultdict(list)
        self.counter = 0

        self.score_thresholds = {
            0.5: False,
            0.9: False,
            0.95: False,
            0.99: False,
        }

    def try_run_task(self, task: Task) -> Optional[Dict]:
        """
        Try to run the task.
        If it can run, returns a metadata dict. Otherwise, returns None.
        """

        round = 0
        result = None
        status = None
        run_metadata = {
            "sv_check_status": [],
            "sv_node_id": [],
            "run_types": [],
            "budget_per_block": [],
            "laplace_hits": [],
            "external_updates": [],
            "pmw_hits": [],
            "db_runtimes": [],
            "true_result_per_node": {},
        }

        runtime_start = time.time()

        # Execute the plan to run the query
        while result is None and (not status or status == "sv_failed"):
            start_planning = time.time()

            # If the SV failed, don't risk it and go with Laplace everywhere
            force_laplace = False if round == 0 else True
            if force_laplace:
                logger.debug("Forcing Laplace because SV failed")

            # Get a DP execution plan for query.
            plan = self.planner.get_execution_plan(task, force_laplace=force_laplace)
            assert plan is not None
            planning_time = time.time() - start_planning
            # print("Planning", planning_time)

            # NOTE: if status is sth else like "out-of-budget" then it stops
            result, status = self.executor.execute_plan(plan, task, run_metadata)

            # logger.info(
            #     colored(
            #         f"Task: {task.id}, Query: {task.query_id}, on blocks: {task.blocks}, Plan: {plan}.",
            #         "green",
            #     )
            # )
            round += 1

        query_runtime = time.time() - runtime_start
        run_metadata["runtime"] = query_runtime

        if result is not None:

            # if self.config.logs.mlflow:

            hit_scores = compute_hit_scores(
                sv_check_status=run_metadata["sv_check_status"],
                laplace_hits=run_metadata["laplace_hits"],
                pmw_hits=run_metadata["pmw_hits"],
                run_types=run_metadata["run_types"],
                node_sizes=run_metadata["node_sizes"],
                total_size=run_metadata["total_size"],
                external_updates=run_metadata["external_updates"],
            )

            for score_name, score_value in hit_scores.items():
                if not np.isnan(score_value):
                    # Not so meaningful for cumulative SV or Laplace score
                    self.score_counters[f"cumulative_{score_name}"] += score_value

                # Sliding averages
                if len(self.score_sliding_windows[score_name]) >= SLIDING_WINDOW_LENGTH:
                    self.score_sliding_windows[score_name].pop(0)  # Drop oldest score
                self.score_sliding_windows[score_name].append(score_value)

            budget_per_block_list = run_metadata["budget_per_block"]
            for budget_per_block in budget_per_block_list:
                for _, budget in budget_per_block.items():
                    self.total_budget_spent_all_blocks += budget.epsilon

            if self.config.blocks.max_num == 1:
                block_key = "(0, 0)"

                assert len(run_metadata["run_types"][0]) == 1, run_metadata

                run_type = run_metadata["run_types"][0][block_key]
                if run_type == LAPLACE_RUNTYPE:
                    update = (
                        1
                        if run_metadata["external_updates"][0].get(block_key, 0) != 0
                        else 0
                    )

                elif run_type == HISTOGRAM_RUNTYPE:
                    update = 1 if run_metadata["sv_check_status"][0] == False else 0
                elif run_type == PMW_RUNTYPE:
                    update = (
                        1
                        if run_metadata["pmw_hits"][0].get(block_key, None) == 0
                        else 0
                    )
                else:
                    raise NotImplementedError(f"Run type {run_type} not implemented")
                self.score_counters[f"num_updates_monoblock"] += update

            if self.counter % 1000 == 0:
                mlflow_log(f"AllBlocks", self.total_budget_spent_all_blocks, task.id)

                for score_name in hit_scores.keys():
                    # Hopefully at least one score is not Nan over the window, otherwise the mean is NaN
                    non_nan_scores = np.array(
                        [
                            score
                            for score in self.score_sliding_windows[score_name]
                            if not np.isnan(score)
                        ]
                    )
                    if len(non_nan_scores) > 0:
                        sliding_score = np.mean(non_nan_scores)

                        for score_threshold in self.score_thresholds.keys():
                            if (
                                self.score_thresholds[score_threshold] == False
                                and sliding_score >= score_threshold
                            ):
                                # # We passed a threshold for the first time
                                # mlflow_log(
                                #     f"sliding_{score_name}_threshold_{score_threshold}",
                                #     self.counter,
                                #     task.id,
                                # )
                                self.score_thresholds[score_threshold] = True

                        # mlflow_log(f"sliding_{score_name}", sliding_score, task.id)
                    # mlflow_log(
                    #     f"cumulative_{score_name}",
                    #     self.score_counters[f"cumulative_{score_name}"],
                    #     task.id,
                    # )

                # Each miss is an update (either SV update or Laplace update, without the check)
                num_updates_monoblock = (
                    task.id + 1 - self.score_counters[f"cumulative_total_hit_score"]
                )
                run_metadata["num_updates_monoblock"] = num_updates_monoblock
                # mlflow_log(f"num_updates_monoblock", num_updates_monoblock, task.id)
                # mlflow_log(
                #     f"num_updates_monoblock",
                #     self.score_counters[f"num_updates_monoblock"],
                #     task.id,
                # )

            status = FINISHED
            # logger.info(
            #     colored(
            #         f"Task: {task.id}, Query: {task.query_id}, Cost of plan: {plan.cost}, on blocks: {task.blocks}, Plan: {plan}. ",
            #         "green",
            #     )
            # )
        else:
            status = FAILED
            # logger.info(
            #     colored(
            #         f"Task: {task.id}, Query: {task.query_id}, on blocks: {task.blocks}, can't run query.",
            #         "red",
            #     )
            # )
        self.counter += 1
        self.tasks_info.append(
            TaskInfo(task, status, planning_time, run_metadata, result).dump()
        )
        return run_metadata

    def validate(self, task_pool: List[Task]):

        # Perform a fake SV check on each subquery histogram
        n_hits = 0
        for task in task_pool:
            (i, j) = task.blocks
            assert i == j

            run_op = RunHistogram((i, j))
            run_return_value, _ = self.executor.run_histogram(
                run_op, task.query, task.query_db_format
            )

            true_error = abs(
                run_return_value.true_result - run_return_value.noisy_result
            )
            if true_error < self.config.alpha / 2:
                n_hits += 1

        validation_hit_rate = n_hits / len(task_pool)
        return validation_hit_rate
