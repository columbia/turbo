from turbo.api import DPEngineHook

from turbo.sql import SQLTurboQuery, sql_drop_where

from dummy.dp_engine import DummyDPExecutor, DummyBudgetAccountant


class DummyDPEngineHook(DPEngineHook):
    def __init__(
        self, executor: DummyDPExecutor, privacy_accountant: DummyBudgetAccountant
    ):
        super().__init__()
        self.executor = executor
        self.privacy_accountant = privacy_accountant

    def executeNPQuery(self, query: SQLTurboQuery):
        return self.executor.get_true_result(query.sql_query)

    def executeDPQuery(
        self, query: SQLTurboQuery, budget: float, true_result: float = None
    ):
        if not true_result:
            true_result = self.executor.get_true_result(query.sql_query)
        noisy_result = self.executor.get_noisy_result(true_result, budget)
        try:
            self.consume(budget)
        except:
            raise ValueError("Insufficient Privacy Budget")
        return noisy_result

    def consume_budget(self, privacy_budget: float):
        return self.privacy_accountant.consume_budget(privacy_budget)

    def get_data_view_size(self, query: SQLTurboQuery):
        sql_without_where = sql_drop_where(query.sql_query)
        data_view_size = self.executor.get_true_result(sql_without_where)
        return data_view_size


def main():
    dataset_path = "datasets/citibike.csv"

    executor = DummyDPExecutor(dataset_path)
    budget_accountant = DummyBudgetAccountant(1)

    sql_query = "SELECT count(*) FROM citibike WHERE gender = 'male'"

    turbo_query = SQLTurboQuery(sql_query)
    dummy_dp_engine_hook = DummyDPEngineHook(executor, budget_accountant)
    print(dummy_dp_engine_hook.get_data_view_size(turbo_query))


if __name__ == "__main__":
    main()
