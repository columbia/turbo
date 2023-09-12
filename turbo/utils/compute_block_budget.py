import argparse
from privacypacking.budget import Budget
import numpy as np


def compute_budget(epsilon, delta):
    alphas = [
        1.5,
        1.75,
        2,
        2.5,
        3,
        4,
        5,
        6,
        8,
        16,
        32,
        64,
    ]

    def from_epsilon_delta():
        orders = {}
        for alpha in alphas:
            orders[alpha] = max(epsilon + np.log(delta) / (alpha - 1), 0)
        return orders

    return from_epsilon_delta()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", dest="epsilon", type=float)
    parser.add_argument("--delta", dest="delta", type=float)
    args = parser.parse_args()

    return compute_budget(args.epsilon, args.delta)


if __name__ == "__main__":
    orders = main()
    for order in orders.values():
        print("- ", order)
