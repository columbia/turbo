import pickle
import psycopg2
import pandas as pd
from loguru import logger
from collections import namedtuple
from turbo.cache import SparseHistogram
from turbo.utils.utils import get_blocks_size


class PSQL:
    def __init__(self, config) -> None:
        self.config = config

        # Initialize the PSQL connection
        try:
            # Connect to the PostgreSQL database server
            self.psql_conn = psycopg2.connect(
                host=self.config.postgres.host,
                port=self.config.postgres.port,
                database=self.config.postgres.database,
                user=self.config.postgres.username,
                password=self.config.postgres.password,
            )
        except (Exception, psycopg2.DatabaseError) as error:
            logger.info(error)
            exit(1)

    def add_new_block(self, block_data_path):
        status = b"success"
        try:
            cur = self.psql_conn.cursor()
            with open(f'{block_data_path}.csv', 'r') as file:
                next(file)  # Skip the header
                if self.config.postgres.database == "covid":
                    cur.copy_from(file, 'covid_data', sep=',', columns=['time', 'positive', 'gender', 'age', 'ethnicity'])
                elif self.config.postgres.database == "citibike":
                    cur.copy_from(file, 'citibike_data', sep=',', columns=['time', 'weekday', 'hour', 'duration_minutes', 'start_station', 'end_station', 'usertype', 'gender', 'age'])
            cur.close()
            self.psql_conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            status = b"failed"
            logger.info(error)
        return status

    def run_query(self, query, blocks):

        sql_query = query + f" AND time>={blocks[0]} AND time<={blocks[1]};"
        try:
            cur = self.psql_conn.cursor()
            cur.execute(sql_query)
            true_result = float(cur.fetchone()[0])
            # print("query", sql_query, "true result", true_result)
            cur.close()
        except (Exception, psycopg2.DatabaseError) as error:
            logger.info(error)
            exit(1)

        blocks_size = get_blocks_size(blocks, self.config.blocks_metadata)
        true_result /= blocks_size
        # print("result:", true_result, "total-size:", blocks_size, "\n")
        return true_result

    def close(self):
        if self.psql_conn is not None:
            self.psql_conn.close()
            logger.info("Database connection closed.")


Block = namedtuple("Block", ["size", "histogram"])


class MockPSQL:
    def __init__(self, config) -> None:
        self.config = config

        # Blocks are in-memory histograms
        self.blocks = {}
        self.blocks_count = 0

        self.attributes_domain_sizes = self.config.blocks_metadata[
            "attributes_domain_sizes"
        ]
        self.domain_size = float(self.config.blocks_metadata["domain_size"])

    def add_new_block(self, block_data_path):
        histogram_data = None
        try:
            with open(f"{block_data_path}.pkl", "rb") as f:
                histogram_data = pickle.load(f)
        except:
            pass
        if histogram_data is None:
            raw_data = pd.read_csv(f"{block_data_path}.csv").drop(columns=["time"])
            histogram_data = SparseHistogram.from_dataframe(
                raw_data, self.attributes_domain_sizes
            )
        block_id = self.blocks_count
        block_size = get_blocks_size(block_id, self.config.blocks_metadata)
        block = Block(block_size, histogram_data)
        self.blocks[block_id] = block
        self.blocks_count += 1

    def run_query(self, query, blocks):
        true_result = 0
        blocks_size = 0
        for block_id in range(blocks[0], blocks[1] + 1):
            block = self.blocks[block_id]
            true_result += block.size * block.histogram.run(query)
            blocks_size += block.size
        # print("true result abs", true_result, "block size", blocks_size)
        true_result /= blocks_size
        # print("true result:", true_result, "total-size:", blocks_size)
        return true_result

    def close(self):
        pass
