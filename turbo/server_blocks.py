import socket

from loguru import logger


class BlocksServer:

    """Entrypoint for adding new blocks in Postgres and the 'budget_accountant' KV store."""

    def __init__(self, db, budget_accountant, config) -> None:
        self.db = db
        self.config = config
        self.host = self.config.blocks_server.host
        self.port = self.config.blocks_server.port
        self.budget_accountant = budget_accountant

    def run(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:

            s.bind((self.host, self.port))
            logger.info(f"Blocks server listening at {self.host}:{self.port}")
            s.listen()
            while True:
                conn, addr = s.accept()
                with conn:
                    logger.info(f"Connected by {addr}")
                    try:
                        # Simple blocking connection
                        data = conn.recv(1024)
                        if not data:
                            continue
                        response = self.serve_request(data.decode())
                        conn.sendall(response)
                    except (Exception) as error:
                        logger.info(error)
                        exit(1)

    def serve_request(self, block_data_path):
        print(block_data_path)
        status = b"success"
        try:
            # Add the block in the database as a new chunk of data
            self.db.add_new_block(block_data_path)
            # Add the block budget in KV store
            self.budget_accountant.add_new_block_budget()
        except (Exception) as error:
            status = b"fail"
            logger.info(error)
            exit(1)
        return status
