import json
import socket

from loguru import logger

from turbo.cache.histogram import query_dict_to_list
from turbo.query_converter import QueryConverter
from turbo.task import Task


class TasksServer:

    """Entrypoint for sending new user requests/tasks to the system."""

    def __init__(self, query_processor, budget_accountant, config) -> None:
        self.config = config
        self.host = self.config.tasks_server.host
        self.port = self.config.tasks_server.port
        print(config)
        self.tasks_count = 0

        self.query_processor = query_processor
        self.budget_accountant = budget_accountant

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            logger.info(f"Tasks server listening at {self.host}:{self.port}")
            s.listen()
            while True:
                conn, addr = s.accept()
                with conn:
                    logger.info(f"Connected by {addr}")
                    # Simple blocking connection
                    data = conn.recv(4096)
                    if not data:
                        continue
                    deserialized_data = json.loads(data).decode("utf-8")
                    print(deserialized_data)
                    response = self.serve_request(deserialized_data)
                    conn.sendall(response)  # Send response

    def serve_request(self, data):
        task_id = self.tasks_count
        self.tasks_count += 1

        num_requested_blocks = int(data["nblocks"])
        num_blocks = self.budget_accountant.get_blocks_count()

        if num_requested_blocks > num_blocks:
            logger.info("There are not that many blocks in the system")
            return

        # Latest Blocks first
        requested_blocks = (num_blocks - num_requested_blocks, num_blocks - 1)
        name = task_id if "task_name" not in data else data["task_name"]

        query_id = int(data["query_id"])
        utility = (float(data["utility"]),)
        utility_beta = (float(data["utility_beta"]),)

        # query = "{'0': 0, '1': 0, '2': 0, '3': 0}"
        query = eval(data["query"])
        # Read compressed rectangle or PyTorch slice, output a query vector
        attribute_sizes = self.config.blocks_metadata.attributes_domain_sizes
        query_vector = query_dict_to_list(query, attribute_sizes=attribute_sizes)
        query_tensor = QueryConverter(self.config).convert_to_sparse_tensor(
            query_vector
        )
        query_tensor = query_tensor.to_dense()
        query_db_format = (
            query_tensor
            if self.config.mock
            else self.query_converter.convert_to_sql(query_vector, requested_blocks)
        )
        query = query_tensor

        # At this point user's request should be translated to a collection of block/chunk ids
        task = Task(
            id=task_id,
            query_id=query_id,
            query_type="linear",
            query=query,
            query_db_format=query_db_format,
            blocks=requested_blocks,
            n_blocks=num_requested_blocks,
            utility=utility,
            utility_beta=utility_beta,
            name=name,
        )

        run_metadata = self.query_processor.try_run_task(task)
        return run_metadata
