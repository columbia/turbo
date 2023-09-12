import redis
from loguru import logger
from turbo.budget import RenyiBudget, BasicBudget


class BudgetAccountantKey:
    def __init__(self, block):
        self.key = f"{block}"


class BudgetAccountant:
    def __init__(self, config) -> None:
        self.config = config
        self.kv_store = self.get_kv_store(config)
        self.epsilon = float(self.config.budget_accountant.epsilon)
        self.delta = float(self.config.budget_accountant.delta)
        self.alphas = self.config.budget_accountant.alphas

    def get_kv_store(self, config):
        return redis.Redis(
            host=config.budget_accountant.host, port=config.budget_accountant.port, db=0
        )

    def get_blocks_count(self):
        return len(self.kv_store.keys("*"))

    def update_block_budget(self, block, budget):
        key = BudgetAccountantKey(block).key
        # Add budget in the key value store
        if self.config.puredp:
            self.kv_store.hset(key, "epsilon", str(budget.epsilon))
        else:
            for alpha in budget.alphas:
                self.kv_store.hset(key, str(alpha), str(budget.epsilon(alpha)))

    def add_new_block_budget(self):
        block = self.get_blocks_count()
        if self.config.puredp:
            budget = BasicBudget(self.epsilon)
        else:
            # Initialize block's budget from epsilon and delta
            budget = RenyiBudget.from_epsilon_delta(
                epsilon=self.epsilon, delta=self.delta
            )
        self.update_block_budget(block, budget)

    def get_block_budget(self, block):
        """Returns the remaining block budget"""
        key = BudgetAccountantKey(block).key
        budget = self.kv_store.hgetall(key)
        if self.config.puredp:
            budget = BasicBudget(float(budget[b"epsilon"]))
        else:
            alphas = [float(alpha) for alpha in budget.keys()]
            epsilons = [float(epsilon) for epsilon in budget.values()]
            budget = RenyiBudget.from_epsilon_list(epsilons, alphas)
        return budget

    def get_all_block_budgets(self):
        block_budgets = {}
        keys = self.kv_store.keys("*")
        for block in keys:
            b = int(block)
            budget = self.get_block_budget(b)
            block_budgets[b] = budget
        return block_budgets.items()

    def can_run(self, blocks, run_budget):
        for block in range(blocks[0], blocks[1] + 1):
            budget = self.get_block_budget(block)
            if not budget.can_allocate(run_budget):
                return False
        return True

    def consume_block_budget(self, block, run_budget):
        """Consumes 'run_budget' from the remaining block budget"""
        budget = self.get_block_budget(block)
        budget -= run_budget
        # Re-write the budget in the KV store
        self.update_block_budget(block, budget)

    def dump(self):
        budgets = [
            (block, budget.dump()) for (block, budget) in self.get_all_block_budgets()
        ]
        return budgets


class MockBudgetAccountant(BudgetAccountant):
    def __init__(self, config) -> None:
        super().__init__(config)

    def get_kv_store(self, config):
        return {}

    def get_blocks_count(self):
        return len(self.kv_store.keys())

    def update_block_budget(self, block, budget):
        key = BudgetAccountantKey(block).key
        # Add budget in the key value store
        self.kv_store[key] = budget

    def get_block_budget(self, block):
        """Returns the remaining budget of block"""
        key = BudgetAccountantKey(block).key
        if key in self.kv_store:
            budget = self.kv_store[key]
            return budget
        # logger.info(f"Block {block} does not exist")
        return None

    def get_all_block_budgets(self):
        return self.kv_store.items()
