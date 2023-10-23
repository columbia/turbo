from collections import namedtuple

Response = namedtuple("Response", ["dp_result", "overhead_budget"])


class Accuracy:
    def __init__(self, alpha: float, beta: float):
        """Helps specify an accuracy requirement for user's query

        Args:
            alpha: Relative error with respect to the population size (e.g. 0.05).
            beta: Probability that the accuracy requirement fails.
        """
        self.alpha = alpha  # 0.05
        self.beta = beta  # 0.001

    def check_accuracy(self, other):
        if other.alpha < self.alpha or other.beta < self.beta:
            return False
        return True
