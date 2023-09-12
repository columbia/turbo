import json
import os
import pickle
from multiprocessing import Manager, Process
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

# import modin.pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray

# from geopy import distance
from loguru import logger
from scipy.cluster.vq import kmeans2, whiten

from turbo.cache import SparseHistogram
from turbo.cache.histogram import get_domain_size
from turbo.utils.utils import REPO_ROOT

# ray.init(runtime_env={"env_vars": {"__MODIN_AUTOIMPORT_PANDAS__": "1"}})


ATTRIBUTES = [
    "weekday",
    "hour",
    "duration_minutes",
    "start_station",
    "end_station",
    "usertype",
    "gender",
    "age",
]


# def compute_distance(row):
#     # Units are in decimal degrees
#     # https://geohack.toolforge.org/geohack.php?pagename=New_York_City&params=40_42_46_N_74_00_22_W_region:US-NY_type:city(8804190)
#     start = row["start station latitude"], row["start station longitude"]
#     end = row["end station latitude"], row["end station longitude"]
#     row["distance_meters"] = int(distance.distance(start, end).m)
#     return row


def year_month_iterator():
    # age/gender are present until at least Jan 2021. In 2017 the column names are different
    start_year = 2018
    end_year = 2020
    names = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            name = f"{year}{month:02d}"
            names.append(name)
    return names


def cluster_stations(months_dir):
    print("Clustering stations..")

    stations_lat_long = []
    # Collect all stations' geolocations
    for name in year_month_iterator():
        df = pd.read_csv(months_dir.joinpath(f"{name}.csv"))
        tmpdf = pd.DataFrame(columns=["station_lat", "station_long"])
        tmpdf["station_lat"] = (
            df["start_latitude"].tolist() + df["end_latitude"].tolist()
        )
        tmpdf["station_long"] = (
            df["start_longitude"].tolist() + df["end_longitude"].tolist()
        )
        tmpdf = tmpdf.drop_duplicates(
            subset=["station_lat", "station_long"], keep="last"
        ).reset_index(drop=True)
        stations_lat_long.append(tmpdf)
    stations_lat_long = pd.concat(stations_lat_long)
    stations_lat_long = stations_lat_long.drop_duplicates(
        subset=["station_lat", "station_long"], keep="last"
    ).reset_index(drop=True)

    # Cluster start/end stations to K predefined clusters
    stations_lat_long = stations_lat_long.to_numpy()
    _, label = kmeans2(whiten(stations_lat_long), 10, iter=20)
    plt.scatter(stations_lat_long[:, 0], stations_lat_long[:, 1], c=label)
    plt.savefig("clusters.png")

    station_ids = {}
    for i, row in enumerate(stations_lat_long):
        station_ids[f"{row[0]}:{row[1]}"] = label[i]

    print("Update station attributes..")

    def update_station_attributes(name, months_dir):
        print("\tProcessing month", name)
        df = pd.read_csv(months_dir.joinpath(f"{name}.csv"))
        df["start_latitude"] = df["start_latitude"].astype("str")
        df["start_longitude"] = df["start_longitude"].astype("str")
        df["end_latitude"] = df["end_latitude"].astype("str")
        df["end_longitude"] = df["end_longitude"].astype("str")
        df["start_station"] = df["start_latitude"] + ":" + df["start_longitude"]
        df = df.drop(
            columns=[
                "start_latitude",
                "start_longitude",
            ]
        )
        df["end_station"] = df["end_latitude"] + ":" + df["end_longitude"]
        df = df.drop(
            columns=[
                "end_latitude",
                "end_longitude",
            ]
        )
        df["start_station"] = df.start_station.map(lambda x: station_ids[x])
        df["end_station"] = df.end_station.map(lambda x: station_ids[x])
        df.to_csv(months_dir.joinpath(f"{name}.csv"), index=False)

    # names = year_month_iterator()
    # for i, name in enumerate(names):
    #     update_station_attributes(name, months_dir)

    # # Running in Parallel
    def p(start, end, names, months_dir):
        for i in range(start, end):
            update_station_attributes(names[i], months_dir)

    processes = []
    num_processes = 12

    names = list(year_month_iterator())
    k = len(names) // num_processes
    for n in range(num_processes - 1):
        processes.append(
            Process(
                target=p,
                args=(n * k, n * k + k, names, months_dir),
            )
        )
        processes[n].start()
    n += 1

    processes.append(Process(target=p, args=(n * k, len(names), names, months_dir)))
    processes[n].start()

    for n in range(num_processes):
        processes[n].join()

    # processes = []
    # names = year_month_iterator()
    # for i, name in enumerate(names):
    #     processes.append(
    #         Process(
    #             target=update_station_attributes,
    #             args=(name, months_dir),
    #         )
    #     )
    #     processes[i].start()
    # for process in processes:
    #     process.join()


def preprocess_month_data(name, months_dir):
    # Doesn't work for all years
    # df = pd.read_csv(
    #     f"https://s3.amazonaws.com/tripdata/{name}-citibike-tripdata.csv.zip"
    # )
    print("Processing month", name)
    csv_name = f"{name}-citibike-tripdata.csv"
    zip_path = months_dir.joinpath(f"{csv_name}.zip")

    urlretrieve(
        f"https://s3.amazonaws.com/tripdata/{csv_name}.zip",
        zip_path,
    )

    df = pd.read_csv(ZipFile(zip_path).open(f"{name}-citibike-tripdata.csv"))

    zip_path.unlink()

    df["starttime"] = pd.to_datetime(df["starttime"])
    # ISO: (year, week, weekday)
    df["year"] = df.starttime.map(lambda x: x.isocalendar()[0])
    df["week"] = df.starttime.map(lambda x: x.isocalendar()[1])
    df["weekday"] = df.starttime.map(lambda x: x.isocalendar()[2] - 1)

    # Also day data
    df["hour"] = df.starttime.map(lambda x: x.hour)
    # df["minute"] = df.starttime.map(lambda x: x.minute)

    # General cleanup
    df["duration_minutes"] = df.tripduration.map(lambda x: int(x // 60))
    df["usertype"] = df["usertype"].map(lambda x: 0 if x == "Customer" else 1)

    df = df.rename(
        columns={
            "start station latitude": "start_latitude",
            "start station longitude": "start_longitude",
            "end station latitude": "end_latitude",
            "end station longitude": "end_longitude",
            "birth year": "birth_year",
        }
    )
    # Chop off a weird outlier
    df["start_latitude"] = df["start_latitude"].map(lambda x: x if x < 42 else np.nan)
    df["end_latitude"] = df["start_latitude"].map(lambda x: x if x < 42 else np.nan)
    df["start_longitude"] = df["start_longitude"].map(
        lambda x: x if x < -73.6 else np.nan
    )
    df["end_longitude"] = df["end_longitude"].map(lambda x: x if x < -73.6 else np.nan)

    df = df[
        [
            "year",
            "week",
            "weekday",
            "hour",
            "duration_minutes",
            "usertype",
            "start_latitude",
            "start_longitude",
            "end_latitude",
            "end_longitude",
            "gender",
            "birth_year",
        ]
    ]

    df = df.dropna()
    df.to_csv(months_dir.joinpath(f"{name}.csv"), index=False)


def preprocess_months(months_dir):
    # Running in Parallel
    processes = []
    names = year_month_iterator()
    for i, name in enumerate(names):
        processes.append(
            Process(
                target=preprocess_month_data,
                args=(name, months_dir),
            )
        )
        processes[i].start()
    for process in processes:
        process.join()


def age_groups(birthYear, currYear):
    # "0-17": 0, "18-49": 1, "50-64": 2,  "65+": 3
    if birthYear > 1920:
        x = currYear - birthYear
        if x <= 17:
            return 0
        if x <= 49:
            return 1
        if x > 49 and x <= 64:
            return 2
        if x > 64:
            return 3
    return np.NaN


def split_months_into_week_blocks(months_dir, blocks_dir):
    K = 300000
    hours_granularity = 4
    duration_minutes_granularity = 20
    duration_minutes_max = 120
    attributes_domains_sizes = {
        "weekday": 7,
        "hour": 24 // hours_granularity,
        "duration_minutes": duration_minutes_max // duration_minutes_granularity,
        "start_station": 10,
        "end_station": 10,
        "user_type": 2,
        "gender": 3,
        "age": 4,
    }

    def bucketize_and_drop(
        name,
        week_counter,
        month_num_weeks,
        metadata,
        months_dir,
        blocks_dir,
    ):
        print("Processing blocks", name, month_num_weeks)
        month_df = pd.read_csv(months_dir.joinpath(f"{name}.csv"))
        month_df = month_df.groupby("week").filter(lambda x: len(x) >= K)
        month_df = month_df.reset_index()

        for week_number in month_df.week.unique():
            df = month_df[month_df.week == week_number]
            # The december block can have some points from the first ISO week of the next year
            year = df.year.unique()[0]
            block_id = week_counter

            df = df.copy()
            df["age"] = df["birth_year"]
            for index, row in df.iterrows():
                df.loc[index, "age"] = age_groups(row["birth_year"], row["year"])

            df["hour"] = df.hour.map(lambda x: x // hours_granularity)
            df["duration_minutes"] = df.duration_minutes.map(
                lambda x: x // duration_minutes_granularity
                if x < duration_minutes_max
                else np.NaN
            )
            # This info lives in the block IDs now
            df = df.drop(columns=["year", "week", "birth_year"])
            logger.info(f"Number of NaN per column: {df.isna().sum()}")
            df = df.dropna()
            df = df.astype("int64")
            df = df[ATTRIBUTES]

            with open(blocks_dir.joinpath(f"block_{block_id}.pkl"), "wb") as f:
                histogram_data = SparseHistogram.from_dataframe(
                    df, list(attributes_domains_sizes.values())
                )
                pickle.dump(histogram_data, f)

            df.insert(0, "time", block_id)
            df.to_csv(blocks_dir.joinpath(f"block_{block_id}.csv"), index=False)

            metadata[block_id] = (f"{year}-{week_number}", df.shape[0])
            week_counter += 1

    # Running in Parallel
    processes = []
    manager = Manager()
    return_dict = manager.dict()

    week_counter = 0
    for i, name in enumerate(year_month_iterator()):
        month_df = pd.read_csv(months_dir.joinpath(f"{name}.csv"))
        month_df = month_df.groupby("week").filter(lambda x: len(x) >= K)
        month_num_weeks = len(month_df.week.unique())
        processes.append(
            Process(
                target=bucketize_and_drop,
                args=(
                    name,
                    week_counter,
                    month_num_weeks,
                    return_dict,
                    months_dir,
                    blocks_dir,
                ),
            )
        )
        processes[i].start()
        week_counter += month_num_weeks

    for process in processes:
        process.join()

    # Write blocks metadata
    metadata = {
        "domain_size": get_domain_size(list(attributes_domains_sizes.values())),
        "attribute_names": ATTRIBUTES,
        "attributes_domain_sizes": list(attributes_domains_sizes.values()),
    }
    metadata_path = Path(REPO_ROOT).joinpath(
        "data/citibike/citibike_data/blocks/metadata.json"
    )
    metadata["blocks"] = dict()
    for idx, key in enumerate(sorted(list(return_dict.keys()))):
        (week, size) = return_dict[key]
        metadata["blocks"][idx] = dict()
        metadata["blocks"][idx]["week"] = week
        metadata["blocks"][idx]["size"] = size
    json_object = json.dumps(metadata, indent=4)
    with open(metadata_path, "w") as outfile:
        outfile.write(json_object)


def main(re_preprocess_months=True):
    months_dir = Path(REPO_ROOT).joinpath("data/citibike/citibike_data/months")
    months_dir.mkdir(parents=True, exist_ok=True)

    if re_preprocess_months:
        preprocess_months(months_dir)
        cluster_stations(months_dir)

    blocks_dir = Path(REPO_ROOT).joinpath("data/citibike/citibike_data/blocks")
    blocks_dir.mkdir(parents=True, exist_ok=True)
    split_months_into_week_blocks(months_dir, blocks_dir)


if __name__ == "__main__":
    main()
