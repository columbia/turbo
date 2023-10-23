import pandas as pd
import numpy as np
from pandasql import sqldf


class DummyDPExecutor:
    def __init__(self, dataset_path):
        self.citibike = pd.read_csv(dataset_path)

    def get_true_result(self, sql_query):
        citibike = self.citibike
        result_df = sqldf(sql_query)
        # print(result_df)
        return result_df.iloc[0][0]

    def get_noisy_result(self, true_result, budget, sensitivity=1):
        laplace_scale = sensitivity / budget
        noise = np.random.laplace(scale=laplace_scale)
        return true_result + noise


class DummyBudgetAccountant:
    def __init__(self, initial_budget):
        self.remaining_budget = initial_budget

    def consume_budget(self, budget):
        self.remaining_budget -= budget

    def remaining_budget(self):
        return self.remaining_budget


def main():
    dataset_path = "datasets/citibike.csv"

    executor = DummyDPExecutor(dataset_path)
    budget_accountant = DummyBudgetAccountant(1)

    sql_query = "SELECT count(*) FROM citibike WHERE gender = 'male'"

    budget = 0.01
    true_result = executor.get_true_result(sql_query)
    noisy_result = executor.get_noisy_result(true_result, budget)
    budget_accountant.consume_budget(budget)
    print(true_result, noisy_result, budget_accountant.remaining_budget())


if __name__ == "__main__":
    main()
