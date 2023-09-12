import argparse
import numpy as np


def compute_laplace_demands(laplace_noise):
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

    orders = {}
    λ = laplace_noise
    for α in alphas:
        ε = (1 / (α - 1)) * np.log(
            (α / (2 * α - 1)) * np.exp((α - 1) / λ)
            + ((α - 1) / (2 * α - 1)) * np.exp(-α / λ)
        )
        orders[α] = float(ε)

    return orders.values()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma", dest="sigma", type=float)

    args = parser.parse_args()

    return compute_laplace_demands(args.sigma)


if __name__ == "__main__":
    demands = main()
    for demand in demands:
        print("- ", demand)
