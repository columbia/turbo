from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import typer
from plotly.offline import iplot

from experiments.ray.analysis import load_ray_experiment
from turbo.utils.utils import LOGS_PATH

app = typer.Typer()


from turbo.utils.utils import REPO_ROOT


def get_paths(dataset):
    tasks_path_prefix = REPO_ROOT.joinpath(f"data/{dataset}/{dataset}_workload/")
    blocks_path_prefix = REPO_ROOT.joinpath(f"data/{dataset}/{dataset}_data/")
    blocks_metadata = str(blocks_path_prefix.joinpath("blocks/metadata.json"))
    blocks_path = str(blocks_path_prefix.joinpath("blocks"))
    return blocks_path, blocks_metadata, tasks_path_prefix


def get_df(path):
    logs = LOGS_PATH.joinpath(path)
    df = load_ray_experiment(logs)
    return df


def get_runtimes(df):

    for (tasks,) in df[
        [
            "tasks_info",
        ]
    ].values:
        query_execution_path_runtimes = {}
        query_execution_path_db_runtimes = {}

        for i, task in enumerate(tasks):
            run_metadata = task["run_metadata"]
            laplace_hits_list = run_metadata["laplace_hits"]
            sv_check_status_list = run_metadata["sv_check_status"]
            # simplifying function for now
            assert len(sv_check_status_list) <= 1

            key = (
                "sv_check_status"
                + str([sv_check_status for sv_check_status in sv_check_status_list])
                + "laplace_hit"
                + str([laplace_hit for laplace_hit in laplace_hits_list])
            )

            # Messy patch
            if key == "sv_check_status[]laplace_hit[{'(0, 0)': 1}]":
                key = "Exact-Cache hit"
            elif key == "sv_check_status[True]laplace_hit[{}]":
                key = "R1 output path"
            elif key == "sv_check_status[]laplace_hit[{'(0, 0)': 0}]":
                key = "R3 output path"
            else:
                key = "R2 output path"

            if key not in query_execution_path_runtimes:
                query_execution_path_runtimes[key] = []
            query_execution_path_runtimes[key].append(run_metadata["runtime"])

            if key not in query_execution_path_db_runtimes:
                query_execution_path_db_runtimes[key] = []
            query_execution_path_db_runtimes[key].append(run_metadata["db_runtimes"])

    for key in query_execution_path_runtimes:
        runtimes = query_execution_path_runtimes[key]
        query_execution_path_runtimes[key] = sum(runtimes) / len(runtimes)

    for key in query_execution_path_db_runtimes:
        # print(query_execution_path_db_runtimes[key])
        runtimes = [
            list(x[0].values())[0] for x in query_execution_path_db_runtimes[key]
        ]
        # print(runtimes)
        query_execution_path_db_runtimes[key] = sum(runtimes) / len(runtimes)

    # print("\nquery_execution_path_db_runtimes", query_execution_path_db_runtimes)
    return query_execution_path_runtimes


def get_convergence_num_updates(df):

    dfs = []

    for (tasks, key, learning_rate, heuristic,) in df[
        [
            "tasks_info",
            "key",
            "learning_rate",
            "heuristic",
        ]
    ].values:
        for task in reversed(tasks):
            run_metadata = task["run_metadata"]
            if "num_updates_monoblock" in run_metadata:
                num_updates = run_metadata["num_updates_monoblock"]
                dfs.append(
                    pd.DataFrame(
                        [
                            {
                                "key": key,
                                "learning_rate": learning_rate,
                                "heuristic": heuristic,
                                "num_updates": num_updates,
                            }
                        ]
                    )
                )
                break
    if dfs:
        dfs = pd.concat(dfs).reset_index()
        return dfs


def get_budgets_information(df, num_blocks):
    dfs = []
    global_dfs = []

    for (
        tasks,
        key,
        zipf_k,
        mechanism,
        learning_rate,
        warmup,
        planner,
        heuristic,
        config,
    ) in df[
        [
            "tasks_info",
            "key",
            "zipf_k",
            "mechanism",
            "learning_rate",
            "warmup",
            "planner",
            "heuristic",
            "config",
        ]
    ].values:

        for i, task in enumerate(tasks):
            # if not (i % 100 == 0 or i == len(tasks) - 1):
            # continue

            run_metadata = task["run_metadata"]
            task_budget_per_block = {}
            budget_per_block = run_metadata["budget_per_block"]
            for block, budget in budget_per_block.items():
                task_budget_per_block[block] = budget["epsilon"]

            sum_budget_across_blocks = 0
            average_budget_across_blocks = 0
            cumulative_budget_per_block = run_metadata["cumulative_budget_per_block"]
            for block, budget in cumulative_budget_per_block.items():
                cumulative_budget = budget["epsilon"]
                sum_budget_across_blocks += cumulative_budget
                task_block_budget = (
                    task_budget_per_block[block]
                    if block in task_budget_per_block
                    else 0
                )
                dfs.append(
                    pd.DataFrame(
                        [
                            {
                                "task": task["id"],
                                "cumulative_budget": cumulative_budget,
                                "mechanism": mechanism,
                                "zipf_k": zipf_k,
                                "block": str(block),
                                "budget": task_block_budget,
                                "key": key,
                                "learning_rate": learning_rate,
                                "warmup": warmup,
                                "planner": planner,
                                "heuristic": heuristic,
                                "tau": config["mechanism"]["probabilistic_cfg"]["tau"],
                                "external_update_on_cached_results": config[
                                    "mechanism"
                                ]["probabilistic_cfg"][
                                    "external_update_on_cached_results"
                                ],
                            }
                        ]
                    )
                )

            average_budget_across_blocks = sum_budget_across_blocks / num_blocks
            global_dfs.append(
                pd.DataFrame(
                    [
                        {
                            "task": task["id"],
                            "sum_budget_across_blocks": sum_budget_across_blocks,
                            "average_budget_across_blocks": average_budget_across_blocks,
                            "mechanism": mechanism,
                            "zipf_k": zipf_k,
                            "key": key,
                            "learning_rate": learning_rate,
                            "warmup": warmup,
                            "planner": planner,
                            "heuristic": heuristic,
                            "tau": config["mechanism"]["probabilistic_cfg"]["tau"],
                            "external_update_on_cached_results": config["mechanism"][
                                "probabilistic_cfg"
                            ]["external_update_on_cached_results"],
                            # "error": run_metadata["error"],
                        }
                    ]
                )
            )

    if dfs:
        time_budgets = pd.concat(dfs).reset_index()
        time_budgets["cumulative_budget"] = time_budgets["cumulative_budget"].astype(
            float
        )
        time_budgets = time_budgets.drop(columns="index")
        global_time_budgets = pd.concat(global_dfs).reset_index()
        global_time_budgets = global_time_budgets.drop(columns="index")
        return time_budgets, global_time_budgets
    return None


def get_blocks_information(df):
    dfs = []

    for (
        blocks,
        initial_budget,
        key,
        zipf_k,
        total_tasks,
        mechanism,
        learning_rate,
        warmup,
        planner,
        heuristic,
        config,
    ) in df[
        [
            "block_budgets_info",
            "blocks_initial_budget",
            "key",
            "zipf_k",
            "total_tasks",
            "mechanism",
            "learning_rate",
            "warmup",
            "planner",
            "heuristic",
            "config",
        ]
    ].values:
        for block_id, budget in blocks:
            dfs.append(
                pd.DataFrame(
                    [
                        {
                            "block": block_id,
                            "budget": budget["epsilon"],
                            "mechanism": mechanism,
                            "zipf_k": zipf_k,
                            "initial_budget": initial_budget,
                            "key": key,
                            "total_tasks": total_tasks,
                            "learning_rate": learning_rate,
                            "warmup": warmup,
                            "planner": planner,
                            "heuristic": heuristic,
                            "tau": config["mechanism"]["probabilistic_cfg"]["tau"],
                            "external_update_on_cached_results": config["mechanism"][
                                "probabilistic_cfg"
                            ]["external_update_on_cached_results"],
                        }
                    ]
                )
            )
    if dfs:
        blocks = pd.concat(dfs)
        return blocks
    return None


def get_sv_misses_information(df):
    dfs = []
    for (sv_misses, key, zipf_k) in df[["sv_misses", "key", "zipf_k"]].values:
        # print(sv_misses)
        for sv_node_id, misses in sv_misses.items():
            dfs.append(
                pd.DataFrame(
                    [
                        {
                            "sv_node_id": sv_node_id,
                            "misses": misses,
                            "key": key,
                            "zipf_k": zipf_k,
                        }
                    ]
                )
            )
    if dfs:
        dfs = pd.concat(dfs)
        return dfs
    return None


def analyze_monoblock(experiment_path):
    def plot_num_allocated_tasks(df, total_tasks):
        fig = px.bar(
            df,
            x="zipf_k",
            y="n_allocated_tasks",
            color="key",
            barmode="group",
            title=f"Num allocated tasks - total tasks={total_tasks}",
            width=900,
            height=500,
        )
        fig.write_image(
            LOGS_PATH.joinpath(f"{experiment_path}/num_allocated_tasks.png")
        )
        return fig

    def plot_total_sv_checks(df, total_tasks):

        fig = px.bar(
            df,
            x="zipf_k",
            y="total_sv_checks",
            color="key",
            barmode="group",
            title=f"Total SV checks - total tasks={total_tasks}",
            width=900,
            height=500,
        )
        return fig

    def plot_total_sv_misses(df, total_tasks):

        fig = px.bar(
            df,
            x="zipf_k",
            y="total_sv_misses",
            color="key",
            barmode="group",
            title=f"Total SV failures - total tasks={total_tasks}",
            width=900,
            height=500,
        )
        return fig

    def plot_budget_utilization(df, total_tasks):
        df["budget"] = df["initial_budget"] - df["budget"]
        # / df[
        # "initial_budget"
        # ]
        pivoted_df = df.pivot_table(
            columns="mechanism",
            index=[df.index.values, "zipf_k"],
            values="budget",
            aggfunc=np.sum,
        ).reset_index("zipf_k")
        pivoted_df.to_csv(
            LOGS_PATH.joinpath(f"{experiment_path}/budget_utilization.csv"),
            index=False,
        )

        fig = px.bar(
            df,
            x="zipf_k",
            y="budget",
            color="key",
            barmode="group",
            title=f"Budget Consumption - total tasks-{total_tasks}",
            width=900,
            height=500,
        )
        fig.write_image(LOGS_PATH.joinpath(f"{experiment_path}/budget_utilization.png"))
        return fig

    def plot_cumulative_budget_utilization_time(df, total_tasks):
        df.to_csv(
            LOGS_PATH.joinpath(f"{experiment_path}/cumulative_budget_utilization.csv"),
            index=False,
        )
        fig = px.line(
            df,
            x="task",
            y="cumulative_budget",
            color="key",
            title=f"Cumulative Budget Consumption - total tasks-{total_tasks}",
            # category_orders=category_orders,
            width=1000,
            height=600,
            facet_row="zipf_k",
        )
        fig.write_image(
            LOGS_PATH.joinpath(f"{experiment_path}/cumulative_budget_utilization.png")
        )
        return fig

    def plot_budget_utilization_time(df, total_tasks):
        fig = px.scatter(
            df,
            x="task",
            y="budget",
            color="key",
            title=f"Budget Consumption per task/time - total tasks-{total_tasks}",
            width=2500,
            height=800,
            facet_row="zipf_k",
        )
        return fig

    df = get_df(experiment_path)
    df["zipf_k"] = df["zipf_k"].astype(float)
    df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)
    total_tasks = df["total_tasks"].max()
    blocks = get_blocks_information(df)

    iplot(plot_num_allocated_tasks(df, total_tasks))
    iplot(plot_total_sv_checks(df, total_tasks))
    iplot(plot_total_sv_misses(df, total_tasks))
    iplot(plot_budget_utilization(blocks, total_tasks))
    time_budgets, _ = get_budgets_information(df, 1)
    iplot(plot_cumulative_budget_utilization_time(time_budgets, total_tasks))
    iplot(plot_budget_utilization_time(time_budgets, total_tasks))

    # analyze_mechanism_type_use(df)
    # analyze_mechanism_type_use_bar(df)


def analyze_convergence(experiment_path):
    def plot_convergence(df):
        df.to_csv(
            LOGS_PATH.joinpath(f"{experiment_path}/cumulative_budget_utilization.csv"),
            index=False,
        )
        fig = px.line(
            df,
            x="learning_rate",
            y="num_updates",
            color="key",
            title=f"Empirical Convergence",
            width=1000,
            height=600,
            log_x=True,
            range_x=[0.001250, 1],
            range_y=[0, 25000],
        )
        fig.write_image(
            LOGS_PATH.joinpath(f"{experiment_path}/empirical_convergence.png")
        )
        return fig

    df = get_df(experiment_path)
    df["learning_rate"] = df["learning_rate"].astype(float)
    df["learning_rate"] = df["learning_rate"] / 8
    df["key"] = df["heuristic"]
    # print(df)
    for index, _ in df.iterrows():
        df.loc[index, "key"] = (
            "PMW" if df.loc[index, "heuristic"] == "bin_visits:0-0" else "PMW-Bypass"
        )

    df.sort_values(["key", "learning_rate"], ascending=[True, True], inplace=True)
    convergence_df = get_convergence_num_updates(df)
    iplot(plot_convergence(convergence_df))


def analyze_multiblock(experiment_path):
    def plot_num_allocated_tasks(df, total_tasks):
        fig = px.bar(
            df,
            x="zipf_k",
            y="n_allocated_tasks",
            color="key",
            barmode="group",
            title=f"Num allocated tasks - total tasks={total_tasks}",
            width=900,
            height=500,
        )
        return fig

    def plot_budget_utilization(df, total_tasks):
        df["budget"] = df["initial_budget"] - df["budget"]
        # / df[
        #     "initial_budget"
        # ]
        grouped = df.groupby("zipf_k")
        for key, group in grouped:
            group_df = pd.DataFrame(group)
            pivoted_df = group_df.pivot_table(
                columns="mechanism",
                index=[group_df.index.values, "block"],
                values="budget",
                aggfunc=np.sum,
            ).reset_index(["block"])
            pivoted_df["block"] = pivoted_df["block"].astype(int)
            pivoted_df = pivoted_df.sort_values(["block"])
            pivoted_df.to_csv(
                LOGS_PATH.joinpath(
                    f"{experiment_path}/budget_utilization_zipf_{key}.csv"
                ),
                index=False,
            )

        fig = px.bar(
            df,
            x="block",
            y="budget",
            color="key",
            barmode="group",
            title=f"Absolute Budget Consumption - total tasks-{total_tasks}",
            width=1200,
            height=600,
            facet_row="zipf_k",
        )
        fig.write_image(LOGS_PATH.joinpath(f"{experiment_path}/budget_utilization.png"))
        return fig

    def plot_total_sv_checks(df, total_tasks):
        fig = px.bar(
            df,
            x="zipf_k",
            y="total_sv_checks",
            color="key",
            barmode="group",
            title=f"Total SV checks - total tasks={total_tasks}",
            width=900,
            height=500,
        )
        return fig

    def plot_total_sv_misses(df, total_tasks):
        fig = px.bar(
            df,
            x="zipf_k",
            y="total_sv_misses",
            color="key",
            barmode="group",
            title=f"Total SV failures - total tasks={total_tasks}",
            width=900,
            height=500,
        )
        return fig

    def plot_global_budget_utilization_time(df, total_tasks):
        df["cumulative_budget"] = df["average_budget_across_blocks"]
        df.to_csv(
            LOGS_PATH.joinpath(f"{experiment_path}/cumulative_budget_utilization.csv"),
            index=False,
        )

        fig = px.line(
            df,
            x="task",
            y="cumulative_budget",
            color="key",
            title=f"Cumulative Absolute Budget Consumption across all blocks - total tasks-{total_tasks}",
            width=1000,
            height=600,
            facet_row="zipf_k",
        )
        fig.write_image(
            LOGS_PATH.joinpath(f"{experiment_path}/cumulative_budget_utilization.png")
        )
        return fig

    df = get_df(experiment_path)
    total_tasks = df["total_tasks"].max()
    df["zipf_k"] = df["zipf_k"].astype(float)
    df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)
    iplot(plot_num_allocated_tasks(df, total_tasks))
    blocks = get_blocks_information(df)
    num_blocks = blocks["block"].nunique()
    print(num_blocks)

    iplot(
        plot_budget_utilization(
            blocks,
            total_tasks,
        )
    )
    iplot(plot_total_sv_checks(df, total_tasks))
    iplot(plot_total_sv_misses(df, total_tasks))

    _, global_time_budgets = get_budgets_information(df, num_blocks)
    iplot(plot_global_budget_utilization_time(global_time_budgets, total_tasks))


def analyze_heuristics_2(experiment_path):
    def plot_budget_utilization_c0(df, total_tasks):
        df["budget"] = df["initial_budget"] - df["budget"]

        df.to_csv(
            LOGS_PATH.joinpath(f"{experiment_path}/budget_utilization_c0.csv"),
            index=False,
        )
        fig = px.line(
            df,
            x="c0",
            y="budget",
            color="key",
            title=f"Budget Consumption - total tasks-{total_tasks}",
            # category_orders=category_orders,
            width=1000,
            height=600,
            facet_row="zipf_k",
        )
        fig.write_image(
            LOGS_PATH.joinpath(f"{experiment_path}/budget_utilization_c0.png")
        )
        return fig

    df = get_df(experiment_path)
    df["zipf_k"] = df["zipf_k"].astype(float)

    df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)
    total_tasks = df["total_tasks"].max()
    blocks = get_blocks_information(df)

    def f(x):
        heuristic_name, params = x.split(":")
        c0, s0 = params.split("-")
        return f"{heuristic_name}-S0={s0}", c0

    blocks["key"], blocks["c0"] = zip(*blocks["heuristic"].apply(f))
    iplot(plot_budget_utilization_c0(blocks, total_tasks))


def analyze_sv_misses(experiment_path):
    def plot_sv_misses_per_node(df, total_tasks):
        df["sparse_vector_node"] = df["sv_node_id"]
        df["number_of_misses"] = df["misses"]

        fig = px.bar(
            df,
            x="sparse_vector_node",
            y="number_of_misses",
            color="key",
            barmode="group",
            title=f"Number of misses per Sparse Vector - total tasks={total_tasks}",
            width=900,
            height=500,
            facet_row="zipf_k",
        )
        return fig

    df = get_df(experiment_path)
    df = df.query("mechanism == 'Hybrid'")
    total_tasks = df["total_tasks"].max()
    sv_misses = get_sv_misses_information(df)
    iplot(plot_sv_misses_per_node(sv_misses, total_tasks))


def analyze_error(experiment_path):
    def plot_error(df):
        fig = px.scatter(
            df,
            x="task",
            y="error",
            color="key",
            title=f"error",
            width=1000,
            height=600,
            facet_row="zipf_k",
        )
        return fig

    df = get_df(experiment_path)
    total_tasks = df["total_tasks"].max()
    df["zipf_k"] = df["zipf_k"].astype(float)
    df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)
    _, errors = get_budgets_information(df, 50)
    iplot(plot_error(errors))


def analyze_runtime(experiment_path):
    # def plot_error(df):
    #     fig = px.scatter(
    #         df,
    #         x="task",
    #         y="error",
    #         color="key",
    #         title=f"error",
    #         width=1000,
    #         height=600,
    #         facet_row="zipf_k",
    #     )
    #     return fig

    df = get_df(experiment_path)
    total_tasks = df["total_tasks"].max()
    df["zipf_k"] = df["zipf_k"].astype(float)
    df.sort_values(["key", "zipf_k"], ascending=[True, True], inplace=True)
    runtimes = get_runtimes(df)
    print(f"Avg. Query Runtime per execution path for {experiment_path}:")
    for k, v in runtimes.items():
        print("\t", k, v)


def analyze_num_cuts(experiment_path):
    def plot_num_cuts(df):
        df["node_size"] = df["chunk_size"]
        df["node"] = df["chunk"]

        fig = px.histogram(df, x="node", color="node_size", width=750, height=400)
        return fig

    df = get_df(experiment_path)
    # df = df.query("zipf_k == 1 and key=='Laplace+TreeCache'")

    chunks_list = []
    chunk_sizes = []
    chunks = df["chunks"][0]
    for chunk, occurences in chunks.items():
        ch = chunk[1:-1]
        b1, b2 = ch.split(",")
        for _ in range(occurences):
            chunks_list.append(ch)
            chunk_sizes.append(int(b2) - int(b1) + 1)

    df1 = pd.DataFrame(chunks_list, columns=["chunk"])
    df2 = pd.DataFrame(chunk_sizes, columns=["chunk_size"])
    df = pd.concat([df1, df2], axis=1)
    df.sort_values(["chunk_size"], ascending=[False], inplace=True)

    iplot(plot_num_cuts(df))


@app.command()
def main(function: str = "", experiment_path: str = ""):
    globals()[f"{function}"](experiment_path)


if __name__ == "__main__":
    app()
