import warnings
from pathlib import Path

warnings.filterwarnings(action="ignore", category=UserWarning)

from pyspark import SparkFiles
from pyspark.sql import SparkSession
from tmlt.analytics.privacy_budget import PureDPBudget
from tmlt.analytics.protected_change import AddMaxRows
from tmlt.analytics.query_builder import QueryBuilder
from tmlt.turbo import TurboSession
from turbo.core import Accuracy

turbo_config = {
    "alpha": 0.05,
    "beta": 0.001,
    "histogram_cfg": {"learning_rate": 4, "heuristic": "bin_visits:0-1", "tau": 0.01},
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

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setLogLevel("OFF")

turbo_suite_path = Path(__file__).parent.parent.parent.parent.resolve()
citibike_path = turbo_suite_path.joinpath("datasets/citibike.csv")
spark.sparkContext.addFile(str(citibike_path))

citibike_df = spark.read.csv(
    SparkFiles.get("citibike.csv"), header=True, inferSchema=True
)

# Build TurboSession
session = TurboSession.from_dataframe(
    privacy_budget=PureDPBudget(1),
    source_id="citibike",
    dataframe=citibike_df,
    protected_change=AddMaxRows(2),
    turbo_config=turbo_config,
)

# Build Query
query = QueryBuilder("citibike").filter("gender = 'male'").count()

# Evaluate Query
count = session.evaluate(query, Accuracy(turbo_config["alpha"], turbo_config["beta"]))
count.show()
print(session.remaining_privacy_budget)
