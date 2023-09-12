from itertools import count
from loguru import logger
from turbo.simulator.resourcemanager import LastItem


class Blocks:
    """
    Model block arrival.
    """

    def __init__(self, environment, resource_manager):
        self.env = environment
        self.resource_manager = resource_manager
        self.config = self.resource_manager.config
        self.blocks_count = count()
        self.env.process(self.block_producer())

    def block_producer(self):
        """Generate blocks."""

        # Produce initial blocks
        for _ in range(self.config.blocks.initial_num):
            self.block(next(self.blocks_count))

        for _ in range(self.config.blocks.max_num - self.config.blocks.initial_num):
            self.block(next(self.blocks_count))
            yield self.env.timeout(self.config.blocks.arrival_interval)

        self.resource_manager.new_blocks_queue.put(LastItem())
        self.resource_manager.block_production_terminated.succeed()

    def block(self, block_id):
        """
        Block behavior.  Notifies resource manager of its existence,
        waits till it gets generated
        """

        self.resource_manager.new_blocks_queue.put(block_id)
        logger.debug(f"Block: {block_id} generated at {self.env.now}")
