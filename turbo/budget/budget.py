# Default values for some datasets
DELTA_MNIST = 1e-5
DELTA_CIFAR10 = 1e-5
DELTA_IMAGENET = 1e-7

MAX_DUMP_DIGITS = 50


class Budget:
    def __init__(self) -> None:
        pass

    def is_positive(self) -> bool:
        pass

    def epsilon(self, alpha: float) -> float:
        pass

    def add_with_threshold(self, other: "Budget", threshold: "Budget"):
        pass

    def can_allocate(self, demand_budget: "Budget") -> bool:
        pass

    def positive(self) -> "Budget":
        pass

    def __eq__(self, other):
        pass

    def __sub__(self, other):
        pass

    def __add__(self, other):
        pass

    def normalize_by(self, other: "Budget"):
        pass

    def __mul__(self, n: float):
        pass

    def __truediv__(self, n: int):
        pass

    def __repr__(self) -> str:
        pass

    def __ge__(self, other) -> bool:
        pass

    def copy(self):
        pass

    def dump(self):
        pass
