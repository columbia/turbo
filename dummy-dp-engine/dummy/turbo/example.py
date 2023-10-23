from turbo.core import Turbo
from turbo.sql import (
    SQLTurboQuery,
)
from dummy.turbo import DummyDPEngineHook
from dummy import DummyDPExecutor, DummyBudgetAccountant

turbo_config = {
    "alpha": 0.05,
    "beta": 0.001,
    "histogram_cfg": {"learning_rate": 4, "heuristic": "bin_visits:5-1", "tau": 0.01},
    "attributes_info": [
        (
            "weekday",
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        ),
        (
            "hour",
            [
                "00:00-4:00",
                "4:00-8:00",
                "8:00-12:00",
                "12:00-16:00",
                "16:00-20:00",
                "20:00-00:00",
            ],
        ),
        (
            "duration_minutes",
            ["0'-20'", "20'-40'", "40'-60'", "60'-80'", "80'-100'", "100'-120'"],
        ),
        ("start_station", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]),
        ("end_station", ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]),
        ("usertype", ["customer", "subscriber"]),
        ("gender", ["unknown", "male", "female"]),
        ("age", ["0-17", "18-49", "50-64", "65+"]),
    ],
}

# Create the dummy dp engine executor and dummy budget accountant.
dataset_path = "~/turbo-suite/datasets/citibike.csv"
executor = DummyDPExecutor(dataset_path)
budget_accountant = DummyBudgetAccountant(1)

# Create SQLTurboQuery
sql_query = "SELECT count(*) FROM citibike WHERE gender = 'male'"
turbo_query = SQLTurboQuery(sql_query)

# Create Turbo and initialize it with DummyTurboExecutorHook
turbo = Turbo(
    config=turbo_config,
    dp_engine_hook=DummyDPEngineHook(executor, budget_accountant),
)

# Run query using Turbo
answer = turbo.run(turbo_query, turbo.accuracy)
