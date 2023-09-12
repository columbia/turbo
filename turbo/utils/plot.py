from collections import defaultdict
from typing import Dict, List, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.missing_ipywidgets import FigureWidget

from turbo.budget import Budget, RenyiBudget


def df_normalized_curves(curves: Dict[str, Budget], epsilon=10, delta=1e-6):
    d = defaultdict(list)
    block = RenyiBudget.from_epsilon_delta(epsilon=epsilon, delta=delta)
    for name, curve in curves.items():
        normalized_curve = curve.normalize_by(block)
        normalized_curve, curve = RenyiBudget.same_support(normalized_curve, curve)
        d["alpha"].extend(curve.alphas)
        d["rdp_epsilon"].extend(curve.epsilons)
        d["normalized_rdp_epsilon"].extend(normalized_curve.epsilons)
        d["mech_type"].extend([name.split("-")[0]] * len(curve.alphas))
        d["mech_name"].extend([name] * len(curve.alphas))
    return pd.DataFrame(d)


def plot_budgets(
    budgets: Union[List[Budget], Budget], log_x=False, log_y=False
) -> FigureWidget:
    if isinstance(budgets, Budget):
        budgets = [budgets]

    data = defaultdict(list)
    for i, budget in enumerate(budgets):
        for alpha, epsilon in zip(budget.alphas, budget.epsilons):
            data["alpha"].append(alpha)
            data["epsilon"].append(epsilon)
            data["id"].append(i)

    df = pd.DataFrame(data=data)
    if not df.empty:
        fig = px.line(
            df,
            x="alpha",
            y="epsilon",
            color="id",
            log_x=log_x,
            log_y=log_y,
        )
    else:
        fig = px.area(
            log_x=log_x,
            log_y=log_y,
        )

    return fig


def plot_budget_utilization_per_block(
    block_log: List, best_alpha: float = 8
) -> FigureWidget:
    d = defaultdict(list)
    for b in block_log:
        d["id"].append(b["id"])
        d["eps"].append(
            b["budget"]["orders"][best_alpha]
            / b["initial_budget"]["orders"][best_alpha]
        )
    fig = px.bar(
        pd.DataFrame(d),
        x="id",
        y="eps",
        title=f"Normalized remaining budget for alpha={best_alpha}",
    )
    return fig


def plot_task_status(task_log: List, rejected_error: float = 0) -> FigureWidget:
    d = defaultdict(list)
    for t in task_log:
        d["task_id"].append(t["id"])
        d["query_id"].append(t["query_id"])
        d["size"].append(1)  # In case you want to encode something here

        if not t["allocated"]:
            d["hard_query"].append("Rejected")
            d["true_error_fraction"].append(rejected_error)
        else:
            d["hard_query"].append("Hard" if t["hard_query"] else "Easy")
            d["true_error_fraction"].append(t["true_error_fraction"])

    df = pd.DataFrame(d).sort_values(["query_id", "hard_query"])
    fig = px.scatter(
        df,
        x="task_id",
        y="true_error_fraction",
        symbol="hard_query",
        color="query_id",
        size="size",
        title=f"True error of the linear query in [0,1]",
    )
    fig.update_layout(legend_orientation="h")
    return fig
