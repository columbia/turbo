from collections import namedtuple
from typing import Dict, List, Tuple

import numpy as np

from turbo.budget.budget import Budget

# Default values for some datasets
DELTA_MNIST = 1e-5
DELTA_CIFAR10 = 1e-5
DELTA_IMAGENET = 1e-7

MAX_DUMP_DIGITS = 50


DPBudget = namedtuple("ConvertedDPBudget", ["epsilon", "delta"])


class BasicBudget:
    # This means PureDP
    def __init__(self, epsilon) -> None:
        self.epsilon = epsilon
        self.pure_epsilon = epsilon

    def is_positive(self) -> bool:
        if self.epsilon >= 0:
            return True
        return False

    def dp_budget(self, delta: float = DELTA_MNIST) -> DPBudget:
        if hasattr(self, "dp_budget_cached"):
            return self.dp_budget_cached

        # Cache the result
        self.dp_budget_cached = DPBudget(epsilon=self.epsilon, delta=delta)
        return self.dp_budget_cached

    def add_with_threshold(self, other: "Budget", threshold: "Budget"):
        """
        Increases every budget-epsilon by "amount".
        The maximum value a budget-epsilon can take is threshold-epsilon.
        """
        return BasicBudget(min(self.epsilon + other.epsilon, threshold.epsilon))

    def compute_noise(
        self,
    ):
        sensitivity = 1
        return np.random.laplace(scale=sensitivity / self.epsilon)

    def can_allocate(self, demand_budget: "Budget") -> bool:
        assert demand_budget.is_positive()
        diff = self - demand_budget
        if diff.epsilon >= 0:
            return True
        return False

    def __eq__(self, other):
        if other.epsilon != self.epsilon:
            return False
        return True

    def __sub__(self, other):
        return BasicBudget(self.epsilon - other.epsilon)

    def __add__(self, other):
        return BasicBudget(self.epsilon + other.epsilon)

    def normalize_by(self, other: "Budget"):
        if other.epsilon > 0:
            return BasicBudget(self.epsilon / other.epsilon)

    def __mul__(self, n: float):
        return BasicBudget(self.epsilon * n)

    def __truediv__(self, n: int):
        return BasicBudget(self.epsilon / n)

    def __repr__(self) -> str:
        return "BasicBudget({})".format(self.epsilon)

    def __ge__(self, other) -> bool:
        diff = self - other
        return diff.is_positive()

    def copy(self):
        return BasicBudget(self.epsilon.copy())

    def dump(self):
        budget_info = {"epsilon": self.epsilon}
        return budget_info

    def __repr__(self):
        return f"epsilon: {self.epsilon}"

    def __str__(self):
        return f"epsilon: {self.epsilon}"
