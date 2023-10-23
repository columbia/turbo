import math

import numpy as np


def calibrate_budget_pmwbypass(s, a, b, n):
    """
    Loose upper bound in Alg 1. of the Turbo paper
    It works for both SV and Bypass
    """
    epsilon = 4 * s * math.log(1 / b) / (n * a)
    return epsilon


def get_epsilon_from_accuracy(s, a, b, n):
    """
    Returns the smallest epsilon such that Pr[|X| > na] < b where X ~ Lap(s/epsilon)
    """
    epsilon = s * math.log(1 / b) / (n * a)
    return epsilon


def get_sv_epsilon(alpha, beta, n, l=1 / 2):
    """
    SV threshold = alpha/2
    Takes only beta_SV, the desired upper bound on the false positive rate for the SV.
    You need to do a union bound with beta_Laplace to get a global beta = beta_SV + beta_Laplace
    that holds even for hard queries.
    """
    binary_eps = binary_search_epsilon(
        alpha=alpha, beta=beta, n=n, l=l, beta_tolerance=1e-5, extra_laplace=False
    )
    real_beta = sum_laplace_beta(binary_eps, n, alpha, l, extra_laplace=False)

    # Make sure that we didn't accidentatlly overspend budget
    assert real_beta < beta

    return binary_eps


def loose_epsilon(alpha, beta, n, l):
    return 2 * np.log(1 / beta) / (l * alpha * n)


def sum_laplace_beta(epsilon, n, alpha, l=1 / 2, extra_laplace=False):
    """
    If extra_laplace = True, we add an extra failure probability coming from the hard query Laplace noise
    See "Concentrated per-query accuracy guarantees for a single PMW" lemma.
    """
    e = l * alpha * n * epsilon
    beta_sv = (1 / 2 + e / 4) * np.exp(-e)
    beta_laplace = np.exp(-2 * e) if extra_laplace else 0
    return beta_laplace + beta_sv


def binary_search_epsilon(alpha, beta, n, l, beta_tolerance=1e-5, extra_laplace=False):
    """
    Find the lowest epsilon that satisfies the failure probability guarantee.
    If extra_laplace = True, this is for a full PMW. Otherwise, it's just for a single SV.
    """
    eps_low = 0
    eps_high = loose_epsilon(alpha, beta, n, l)
    # Make sure that the initial upper bound is large enough
    assert sum_laplace_beta(eps_high, n, alpha, l=l, extra_laplace=extra_laplace) < beta

    real_beta = 0

    # Bring real_beta close to beta, but from below (conservative)
    while real_beta < beta - beta_tolerance:
        eps_mid = (eps_low + eps_high) / 2
        beta_mid = sum_laplace_beta(eps_mid, n, alpha, l=l, extra_laplace=extra_laplace)

        if beta_mid < beta:
            eps_high = eps_mid
            real_beta = beta_mid
        else:
            # Don't update the real_beta, you can only exit the loop if real_beta < beta - beta_tolerance
            eps_low = eps_mid

    return eps_high
