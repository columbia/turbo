import json
import socket


class TasksClient:
    def __init__(self, config) -> None:
        self.config = config
        self.host = self.config.tasks_server.host
        self.port = self.config.tasks_server.port

    def send_request(self, task):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.host, self.port))
            serialized_data = json.dumps(task).encode("utf-8")
            s.sendall(serialized_data)
            data = s.recv(4096)
        print(f"Received {data!r}")


task = {
    "nblocks": 1,
    "utility": 0.05,
    "utility_beta": 0.001,
    "query": {'0': [0, 1], '1': [0, 1], '2': [0, 2, 3], '3': [0, 2, 4, 5, 6]}
}