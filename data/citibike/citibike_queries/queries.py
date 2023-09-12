import os
import json
import typer
import pickle
from loguru import logger
from pathlib import Path
from turbo.utils.utils import REPO_ROOT
from turbo.cache.histogram import k_way_marginal_query_list
from turbo.cache.histogram import build_sparse_tensor
from multiprocessing import Process

app = typer.Typer()

# A total of 7,804 stories

"""
# Story 1: https://public.tableau.com/app/profile/james.jeffrey/viz/CitiBikeRideAnalyzer/CitiBikeRdeAnalyzer
    
    # What is the gender of riders
    # Are more subscribers or customers riding?
    # Total number of rides
    # Average Ride Time
    # What is the age distribution of riders
    # Peak Riding times within a week across all hours
    # Which stations are the most active?

# Story 2: https://public.tableau.com/app/profile/ryann.green3508/viz/CitibikeAnalysis10_26/CitibikeStory

    # Top starting stations
    # Top ending stations
    # Most popular biking routes in NYC
    # Number of rides within a specific time period.
    # Subscribers vs Customers by month broken down by gender
    # Number of Riders by month
    # Number of Riders by age group
    # Do males or females ride more?
    # Male vs Female / Customers vs Subscribers spring vs summer
    # Ridership breakdown by age and gender

# Story 3: https://public.tableau.com/app/profile/art.tucker/viz/NYCCitiBikeAnalysischallenge/NYCCitiBikeAnalysis

    # Number of Customers vs Subscribers
    # Number of riders
    # Number of riders per gender
    # Number of Rides per Start station
    # Peak hours of usage - number of riders per hour
    # Average trip duration by birth year / age
    # Number of trips per 
    # Average trip duration
    # Average trip duration by gender
    # Number of trips per hour, by weekday, by gender
    # Number of trips per user type, by weekday, by gender


# Story 4: https://public.tableau.com/app/profile/trilogy4111/viz/CitiBike_12/StartStations

    # Number of rides per start station
    # Number of rides per end station
    # Average trip duration per age
    # Number of rides per hour per gender for an entire month (June)
    # Average age of riders per start station
    # Number of riders female vs male
    # Average trip duration per gender
    # Total trip duration per bike id
    # Total number of riders per bike id

"""


def any_gender():
    return list(range(3))


def any_user_type():
    return list(range(2))


def any_age():
    return list(range(4))


def any_weekday():
    return list(range(7))


def any_hour():
    return list(range(6))


def any_duration_minutes():
    return list(range(6))


def any_start_station():
    return list(range(10))


def any_end_station():
    return list(range(10))


"""
Attributes:
    "weekday": 7
    "hour": 6
    "duration_minutes": 6
    "start_station": 10
    "end_station": 10
    "user_type": 2
    "gender": 3
    "age": 4
"""

attribute_position = {
    "weekday": "0",
    "hour": "1",
    "duration_minutes": "2",
    "start_station": "3",
    "end_station": "4",
    "user_type": "5",
    "gender": "6",
    "age": "7",
}

# What is the gender of riders
def query_1():
    # Count riders [male, female, unknown]
    queries = []
    key = attribute_position["gender"]
    for value in any_gender():
        queries.append({key: value})
    return queries


# Are more subscribers or customers riding
def query_2():
    # Count riders [subscribers, customers]
    queries = []
    key = attribute_position["user_type"]
    for value in any_user_type():
        queries.append({key: value})
    return queries


# Subscribers vs Customers broken down by gender
def query_3():
    # Count riders [usertype, gender]
    queries = []
    key1 = attribute_position["user_type"]
    key2 = attribute_position["gender"]
    for value1 in any_user_type():
        for value2 in any_gender():
            queries.append({key1: value1, key2: value2})
    return queries


# Total number of rides
def query_4():
    # Count riders total
    queries = []
    queries.append({})
    return queries


# What is the age distribution of riders
def query_5():
    # Count riders per [age]
    queries = []
    key = attribute_position["age"]
    for value in any_age():
        queries.append({key: value})
    return queries


# Top starting stations
def query_6():
    # Count riders per [start station]
    queries = []
    key = attribute_position["start_station"]
    for value in any_start_station():
        queries.append({key: value})
    return queries


# Top ending stations
def query_7():
    # Count riders per [end station]
    queries = []
    key = attribute_position["end_station"]
    for value in any_end_station():
        queries.append({key: value})
    return queries


# Peak Riding times across all hours
def query_8():
    # Count riders per [hour]
    queries = []
    key = attribute_position["hour"]
    for value in any_hour():
        queries.append({key: value})
    return queries


# Most popular biking routes in NYC
def query_9():
    # Count riders per [start/end station]
    queries = []
    key1 = attribute_position["start_station"]
    key2 = attribute_position["end_station"]
    for value1 in any_start_station():
        for value2 in any_end_station():
            queries.append({key1: value1, key2: value2})
    return queries


# Ridership breakdown by age and gender
def query_10():
    # Count riders [gender, age]
    queries = []
    key1 = attribute_position["gender"]
    key2 = attribute_position["age"]
    for value1 in any_gender():
        for value2 in any_age():
            queries.append({key1: value1, key2: value2})
    return queries


# Count riders by trip-duration
def query_11():
    # Count riders [duration-minutes]
    queries = []
    key = attribute_position["duration_minutes"]
    for value in any_duration_minutes():
        queries.append({key: value})
    return queries


# Count riders by trip-duration, by age
def query_12():
    # Count riders [duration-minutes, age]
    queries = []
    key1 = attribute_position["duration_minutes"]
    key2 = attribute_position["age"]
    for value1 in any_duration_minutes():
        for value2 in any_age():
            queries.append({key1: value1, key2: value2})
    return queries


# Count riders by trip-duration, by gender
def query_13():
    # Count riders [duration-minutes, gender]
    queries = []
    key1 = attribute_position["duration_minutes"]
    key2 = attribute_position["gender"]
    for value1 in any_duration_minutes():
        for value2 in any_gender():
            queries.append({key1: value1, key2: value2})
    return queries


# Count riders by trip-duration, by gender by age
def query_14():
    # Count riders [duration-minutes, gender, age]
    queries = []
    key1 = attribute_position["duration_minutes"]
    key2 = attribute_position["gender"]
    key3 = attribute_position["age"]

    for value1 in any_duration_minutes():
        for value2 in any_gender():
            for value3 in any_age():
                queries.append({key1: value1, key2: value2, key3: value3})
    return queries


# Count riders by trip-duration, by user-type
def query_15():
    # Count riders [duration-minutes, usertype]
    queries = []
    key1 = attribute_position["duration_minutes"]
    key2 = attribute_position["user_type"]

    for value1 in any_duration_minutes():
        for value2 in any_user_type():
            queries.append({key1: value1, key2: value2})
    return queries


# Count riders by age, by start-station (weird one)
def query_16():
    # Count riders [age, start-station]
    queries = []
    key1 = attribute_position["start_station"]
    key2 = attribute_position["age"]
    for value1 in any_start_station():
        for value2 in any_age():
            queries.append({key1: value1, key2: value2})
    return queries


# Count riders by age, by end-station (weird one)
def query_17():
    # Count riders [age, end-station]
    queries = []
    key1 = attribute_position["end_station"]
    key2 = attribute_position["age"]
    for value1 in any_end_station():
        for value2 in any_age():
            queries.append({key1: value1, key2: value2})
    return queries


# Additional

# Subscribers vs Customers broken down by age
def query_18():
    # Count riders [usertype, age]
    queries = []
    key1 = attribute_position["user_type"]
    key2 = attribute_position["age"]
    for value1 in any_user_type():
        for value2 in any_age():
            queries.append({key1: value1, key2: value2})
    return queries


# Riders broken down by weekday
def query_19():
    # Count riders [weekday]
    queries = []
    key = attribute_position["weekday"]
    for value in any_weekday():
        queries.append({key: value})
    return queries


# Count riders by trip-duration, by weekday
def query_20():
    # Count riders [duration-minutes, weekday]
    queries = []
    key1 = attribute_position["weekday"]
    key2 = attribute_position["duration_minutes"]

    for value1 in any_weekday():
        for value2 in any_duration_minutes():
            queries.append({key1: value1, key2: value2})
    return queries


# Count riders by trip-duration, by hour
def query_21():
    # Count riders [duration-minutes, hour]
    queries = []
    key1 = attribute_position["hour"]
    key2 = attribute_position["duration_minutes"]

    for value1 in any_hour():
        for value2 in any_duration_minutes():
            queries.append({key1: value1, key2: value2})
    return queries


# Count riders by hour, by weekday
def query_22():
    # Count riders [hour, weekday]
    queries = []
    key1 = attribute_position["weekday"]
    key2 = attribute_position["hour"]

    for value1 in any_weekday():
        for value2 in any_hour():
            queries.append({key1: value1, key2: value2})
    return queries


# Count riders by user_type, by gender by age
def query_23():
    # Count riders [user_type, gender, age]
    queries = []
    key1 = attribute_position["user_type"]
    key2 = attribute_position["gender"]
    key3 = attribute_position["age"]

    for value1 in any_user_type():
        for value2 in any_gender():
            for value3 in any_age():
                queries.append({key1: value1, key2: value2, key3: value3})
    return queries


# Count riders by hour, start-station
def query_24():
    # Count riders [start_station, hour]
    queries = []
    key1 = attribute_position["hour"]
    key2 = attribute_position["start_station"]

    for value1 in any_hour():
        for value2 in any_start_station():
            queries.append({key1: value1, key2: value2})
    return queries


# Count riders by hour, end-station
def query_25():
    # Count riders [end_station, hour]
    queries = []
    key1 = attribute_position["hour"]
    key2 = attribute_position["end_station"]

    for value1 in any_hour():
        for value2 in any_end_station():
            queries.append({key1: value1, key2: value2})
    return queries


# Count riders by weekday, start-station
def query_26():
    # Count riders [start_station, weekday]
    queries = []
    key1 = attribute_position["weekday"]
    key2 = attribute_position["start_station"]

    for value1 in any_weekday():
        for value2 in any_start_station():
            queries.append({key1: value1, key2: value2})
    return queries


# Count riders by weekday, end-station
def query_27():
    # Count riders [end_station, weekday]
    queries = []
    key1 = attribute_position["weekday"]
    key2 = attribute_position["end_station"]

    for value1 in any_weekday():
        for value2 in any_end_station():
            queries.append({key1: value1, key2: value2})
    return queries


# Most popular biking routes in NYC per weekday
def query_28():
    # Count riders per [weekday, start/end station]
    queries = []
    key1 = attribute_position["weekday"]
    key2 = attribute_position["start_station"]
    key3 = attribute_position["end_station"]
    for value1 in any_weekday():
        for value2 in any_start_station():
            for value3 in any_end_station():
                queries.append({key1: value1, key2: value2, key3: value3})
    return queries


# Most popular biking routes in NYC per hour
def query_29():
    # Count riders per [hour, start/end station]
    queries = []
    key1 = attribute_position["hour"]
    key2 = attribute_position["start_station"]
    key3 = attribute_position["end_station"]
    for value1 in any_hour():
        for value2 in any_start_station():
            for value3 in any_end_station():
                queries.append({key1: value1, key2: value2, key3: value3})
    return queries


# Most popular biking routes in NYC per age
def query_30():
    # Count riders per [age, start/end station]
    queries = []
    key1 = attribute_position["age"]
    key2 = attribute_position["start_station"]
    key3 = attribute_position["end_station"]
    for value1 in any_age():
        for value2 in any_start_station():
            for value3 in any_end_station():
                queries.append({key1: value1, key2: value2, key3: value3})
    return queries


def create_queries():
    queries = []
    for i in range(1, 31):
        queries += globals()[f"query_{i}"]()
    return queries


def write_queries(queries_dir, workload, queries, query_tensors_path):
    query_paths = {
        query_id: {
            "query": queries[query_id],
            "query_path": str(query_tensors_path.joinpath(f"{query_id}.pkl")),
        }
        for query_id in range(len(queries))
    }
    queries_path = Path(queries_dir).joinpath(f"{workload}.queries.json")
    with open(queries_path, "w") as outfile:
        outfile.write(json.dumps(query_paths, indent=4))


def save_query_tensors(queries, attribute_sizes, query_tensors_path):
    def convert_queries_to_tensors(start_query_id, queries, query_tensors_path):
        for query in queries:
            query_vector = k_way_marginal_query_list(query, attribute_sizes)
            query_tensor = build_sparse_tensor(
                bin_indices=query_vector,
                values=[1.0] * len(query_vector),
                attribute_sizes=attribute_sizes,
            )
            with open(query_tensors_path.joinpath(f"{start_query_id}.pkl"), "wb") as f:
                pickle.dump(query_tensor, f)
            start_query_id += 1

    # Running in Parallel
    processes = []
    num_processes = os.cpu_count()
    k = len(queries) // num_processes
    for n in range(num_processes - 1):
        processes.append(
            Process(
                target=convert_queries_to_tensors,
                args=(n * k, queries[n * k : n * k + k], query_tensors_path),
            )
        )
        processes[n].start()
    n += 1
    processes.append(
        Process(
            target=convert_queries_to_tensors,
            args=(n * k, queries[n * k :], query_tensors_path),
        )
    )
    processes[n].start()

    for n in range(num_processes):
        processes[n].join()


def main(
    queries_dir: str = REPO_ROOT.joinpath("data/citibike/citibike_queries"),
    blocks_metadata_path: str = REPO_ROOT.joinpath(
        "data/citibike/citibike_data/blocks/metadata.json"
    ),
):

    try:
        with open(blocks_metadata_path) as f:
            blocks_metadata = json.load(f)
    except NameError:
        logger.error("Dataset metadata must have be created first..")
        exit(1)
    attribute_domain_sizes = blocks_metadata["attributes_domain_sizes"]

    workload = "stories"

    # Create all queries from stories
    queries = create_queries()

    # Create and save query tensors from queries
    query_tensors_path = Path(queries_dir).joinpath(f"{workload}_sparse_queries")
    os.makedirs(query_tensors_path, exist_ok=True)
    save_query_tensors(queries, attribute_domain_sizes, query_tensors_path)

    # Write query paths to queries.json
    os.makedirs(queries_dir, exist_ok=True)
    write_queries(queries_dir, workload, queries, query_tensors_path)


if __name__ == "__main__":
    typer.run(main)
