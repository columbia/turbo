import math
import warnings
from typing import List

import numpy as np

# from autodp.mechanism_zoo import LaplaceMechanism
# from autodp.transformer_zoo import AmplificationBySampling
from opacus.accountants.analysis.rdp import compute_rdp

from turbo.budget import ALPHAS, RenyiBudget


class ZeroCurve(RenyiBudget):
    def __init__(self, alpha_list: List[float] = ALPHAS) -> None:
        orders = {alpha: 0 for alpha in alpha_list}
        super().__init__(orders)


class SyntheticPolynomialCurve(RenyiBudget):
    def __init__(
        self,
        best_alpha,
        epsilon_min,
        epsilon_left,
        epsilon_right,
        alpha_list: List[float] = ALPHAS,
        block_epsilon=10,
        block_delta=1e-8,
    ) -> None:
        def lagrange_3(x):
            x_0 = alpha_list[0]
            x_2 = alpha_list[-1]
            x_1 = best_alpha
            return (
                epsilon_left * (x - x_1) * (x - x_2) / (x_0 - x_1) * (x_0 - x_2)
                + epsilon_min * (x - x_0) * (x - x_2) / (x_1 - x_0) * (x_1 - x_2)
                + epsilon_right * (x - x_0) * (x - x_1) / (x_2 - x_0) * (x_2 - x_1)
            )

        # if best_alpha not in [epsilon_left, epsilon_right]:
        #     orders = {alpha: lagrange_3(alpha) for alpha in alpha_list}

        block = RenyiBudget.from_epsilon_delta(epsilon=block_epsilon, delta=block_delta)

        non_zero_alphas = [alpha for alpha in block.alphas if block.epsilon(alpha) > 0]
        zero_alphas = [alpha for alpha in block.alphas if block.epsilon(alpha) == 0]

        # x = [non_zero_alphas[0], best_alpha, non_zero_alphas[-2], non_zero_alphas[-1]]
        # y = [
        #     epsilon_left,
        #     epsilon_min,
        #     (epsilon_min + epsilon_right) / 2,
        #     epsilon_right,
        # ]

        # print(x, y)
        # spl = splrep(x, y, k=3)

        # rdp_epsilons = splev(non_zero_alphas, spl)

        # orders = {
        #     alpha: epsilon for alpha, epsilon in zip(non_zero_alphas, rdp_epsilons)
        # }
        x = [non_zero_alphas[0], best_alpha, non_zero_alphas[-1]]
        y = [
            epsilon_left,
            epsilon_min,
            epsilon_right,
        ]
        f = interp1d(x=x, y=y, kind="slinear")
        orders = {alpha: f(alpha) * block.epsilon(alpha) for alpha in non_zero_alphas}
        for alpha in zero_alphas:
            orders[alpha] = 1
        super().__init__(orders)


def rdp_curve(α, λ):
    """
    See Table II of the RDP paper (https://arxiv.org/pdf/1702.07476.pdf)
    lambda is the std of the mechanism for sensitivity 1 (the noise multiplier otherwise).
    """
    with np.errstate(over="raise", under="raise"):
        try:
            ε = (1 / (α - 1)) * np.log(
                (α / (2 * α - 1)) * np.exp((α - 1) / λ)
                + ((α - 1) / (2 * α - 1)) * np.exp(-α / λ)
            )
        except FloatingPointError:
            # It means that alpha/lambda is too large (under or overflow)
            # We just drop the negative exponential (≃0) and simplify the log
            ε = (1 / (α - 1)) * (np.log(α / (2 * α - 1)) + (α - 1) / λ)
    return float(ε)


class LaplaceCurve(RenyiBudget):
    """
    RDP curve for a Laplace mechanism with sensitivity 1.
    """

    def __init__(self, laplace_noise: float, alpha_list: List[float] = ALPHAS) -> None:
        """Computes the Laplace RDP curve.


        Args:
            laplace_noise (float): lambda, the std of the mechanism for sensitivity 1 (the noise multiplier otherwise).
            alpha_list (List[float], optional): RDP orders. Defaults to ALPHAS.
        """
        self.laplace_noise = laplace_noise
        self.pure_epsilon = 1 / laplace_noise
        orders = {}
        λ = laplace_noise
        for α in alpha_list:
            orders[α] = rdp_curve(α, λ)
        super().__init__(orders)


class GaussianCurve(RenyiBudget):
    def __init__(self, sigma: float, alpha_list: List[float] = ALPHAS) -> None:
        orders = {alpha: alpha / (2 * (sigma**2)) for alpha in alpha_list}
        self.sigma = sigma
        super().__init__(orders)


class BoundedOneShotSVT(RenyiBudget):
    def __init__(
        self, nu: float, ro: float, kmax: int, alpha_list: List[float] = ALPHAS
    ) -> None:
        # Remark on top of page 5 after Theorem 8 in the RDP SVT paper
        # (https://papers.nips.cc/paper/2020/file/e9bf14a419d77534105016f5ec122d62-Paper.pdf)
        l = np.log(1 + kmax)
        g = 1 / (2 * (nu**2)) + 1 / (2 * (ro**2))
        orders = {alpha: alpha * g + l / (alpha - 1) for alpha in alpha_list}
        super().__init__(orders)


class PureDPtoRDP(RenyiBudget):
    # https://papers.nips.cc/paper/2020/file/e9bf14a419d77534105016f5ec122d62-Paper.pdf
    # Originally from the CDP paper by Bun and Steinke 2016
    def __init__(self, epsilon: float, alpha_list: List[float] = ALPHAS) -> None:

        orders = {}
        for alpha in alpha_list:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try:
                    eps = (
                        np.log(
                            np.sinh(alpha * epsilon) - np.sinh((alpha - 1) * epsilon)
                        )
                        - np.log(np.sinh(epsilon))
                    ) / (alpha - 1)
                except Warning as e:
                    # Something went wrong. To be on the safe side, use a coarser upper bound.
                    eps = alpha * (epsilon**2) / 2
            orders[alpha] = eps
        self.pure_epsilon = epsilon
        super().__init__(orders)


class PureDPtoRDP_loose(RenyiBudget):
    # Loose upper bound
    def __init__(self, epsilon: float, alpha_list: List[float] = ALPHAS) -> None:
        orders = {alpha: alpha * epsilon**2 / 2 for alpha in alpha_list}
        self.pure_epsilon = epsilon
        super().__init__(orders)


class LaplaceSVCurve(RenyiBudget):
    def __init__(self, epsilon_1, epsilon_2=None, alpha_list=ALPHAS):
        """
        - Algorithm 2 with c = 1 from https://papers.nips.cc/paper/2020/file/e9bf14a419d77534105016f5ec122d62-Paper.pdf
        - By default we set $\epsilon_2 = 2\epsilon_1$ like Salil
        - $M_\rho = Lap(\Delta/\epsilon_1)$ is $\epsilon_\rho(\alpha) = \epsilon_{\lambda_1}(\alpha)$-RDP for queries with sensitivity $\Delta$, with  $\epsilon_\lambda(\alpha)$ the Laplace RDP curve with $\lambda_1 = 1/\epsilon_1$.
        - $M_\nu = Lap(2\Delta/\epsilon_2)$ is $\epsilon_\nu(\alpha) = \epsilon_{\lambda_2}(\alpha)$-RDP for queries with sensitivity $2\Delta$, with $\lambda_2 = 1/\epsilon_2$.
        - Since $\epsilon_\rho(\infty) = \epsilon_2 < \infty$ we can use Point 3 of Theorem 8, so SV is $\epsilon_\nu(\alpha) +  \epsilon_2$-RDP
        """
        epsilon_2 = epsilon_2 if epsilon_2 else 2 * epsilon_1
        orders = {
            alpha: rdp_curve(alpha, 1 / epsilon_1) + epsilon_2 for alpha in alpha_list
        }
        super().__init__(orders)


class SubsampledGaussianCurve(RenyiBudget):
    def __init__(
        self,
        sampling_probability: float,
        sigma: float,
        steps: float,
        alpha_list: List[float] = ALPHAS,
    ) -> None:
        rdp = compute_rdp(
            q=sampling_probability,
            noise_multiplier=sigma,
            steps=steps,
            orders=alpha_list,
        )

        orders = {alpha: epsilon for (alpha, epsilon) in zip(alpha_list, rdp)}
        super().__init__(orders)

    @classmethod
    def from_training_parameters(
        cls,
        dataset_size: int,
        batch_size: int,
        epochs: int,
        sigma: float,
        alpha_list: List[float] = ALPHAS,
    ) -> "SubsampledGaussianCurve":
        """Helper function to build the SGM curve with more intuitive parameters."""

        sampling_probability = batch_size / dataset_size
        steps = (dataset_size * epochs) // batch_size
        return cls(sampling_probability, sigma, steps, alpha_list)


# class SubsampledLaplaceCurve(RenyiBudget):
#     def __init__(
#         self,
#         sampling_probability: float,
#         noise_multiplier: float,
#         steps: int,
#         alpha_list: List[float] = ALPHAS,
#     ) -> None:

#         curve = AmplificationBySampling(PoissonSampling=True)(
#             LaplaceMechanism(b=noise_multiplier), sampling_probability
#         )

#         orders = {alpha: curve.get_RDP(alpha) * steps for alpha in alpha_list}
#         super().__init__(orders)
