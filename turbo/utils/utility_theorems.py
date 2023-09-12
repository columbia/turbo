import math
from functools import partial
from multiprocessing import Pool
from typing import Dict, Tuple

import numpy as np
from loguru import logger

# from ray.util.multiprocessing import Pool


def single_process_toplevel(arg_tuple):
    N_local, n_chunks, epsilons, alpha, n = arg_tuple
    chunk_noises = np.zeros((N_local, n_chunks))
    for chunk_id in range(n_chunks):
        # The final laplace scale (Q_ij), already scaled by n_i/n * eps^2/sum(eps^2)
        single_chunk_laplace_scale = epsilons[chunk_id] / (
            n * np.sum(epsilons[chunk_id] ** 2)
        )
        # print(f"Chunk {chunk_id} laplace scale: {single_chunk_laplace_scale}")
        laplace_scale = np.repeat([single_chunk_laplace_scale], N_local, axis=0)
        laplace_noises = np.random.laplace(scale=laplace_scale)

        # Optimal average for that chunk, N times
        chunk_noises[:, chunk_id] = np.sum(laplace_noises, axis=1)

    aggregated_noise_total = np.sum(chunk_noises, axis=1)
    beta = (
        np.sum(aggregated_noise_total > alpha) + np.sum(aggregated_noise_total < -alpha)
    ) / N_local

    return beta


def monte_carlo_beta(existing_epsilons, chunk_sizes, fresh_epsilon, alpha, N=1_000_000):

    fresh_epsilons = [fresh_epsilon] * len(existing_epsilons)
    return monte_carlo_beta_multieps(
        existing_epsilons=existing_epsilons,
        chunk_sizes=chunk_sizes,
        fresh_epsilons=fresh_epsilons,
        alpha=alpha,
        N=N,
    )


def monte_carlo_beta_multieps(
    existing_epsilons, chunk_sizes, fresh_epsilons, alpha, N=1_000_000, n_processes=1
):

    # Add fresh epsilons, ignore chunks where fresh_epsilon=0
    epsilons = [
        np.append(eps_by_chunk, fresh_eps_by_chunk)
        if fresh_eps_by_chunk > 0
        else eps_by_chunk
        for eps_by_chunk, fresh_eps_by_chunk in zip(existing_epsilons, fresh_epsilons)
    ]

    # Vectorized code with a batch dimension corresponding to N
    n_chunks = len(epsilons)
    n = sum(chunk_sizes)

    def single_process_closure(N_local):
        chunk_noises = np.zeros((N_local, n_chunks))
        for chunk_id in range(n_chunks):
            # The final laplace scale (Q_ij), already scaled by n_i/n * eps^2/sum(eps^2)
            single_chunk_laplace_scale = epsilons[chunk_id] / (
                n * np.sum(epsilons[chunk_id] ** 2)
            )
            # print(f"Chunk {chunk_id} laplace scale: {single_chunk_laplace_scale}")
            laplace_scale = np.repeat([single_chunk_laplace_scale], N_local, axis=0)
            laplace_noises = np.random.laplace(scale=laplace_scale)

            # Optimal average for that chunk, N times
            chunk_noises[:, chunk_id] = np.sum(laplace_noises, axis=1)

        aggregated_noise_total = np.sum(chunk_noises, axis=1)
        # beta = np.sum(aggregated_noise_total > alpha) / N_local
        beta = (
            np.sum(aggregated_noise_total > alpha)
            + np.sum(aggregated_noise_total < -alpha)
        ) / N_local
        return beta

    if n_processes == 1:
        # This seems faster most of the time, don't bother with multiprocessing
        beta = single_process_closure(N)
    elif False:
        pool = Pool(processes=n_processes)
        # Might have slightly *more* samples than N
        N_local = math.ceil(N / n_processes)
        args = [N_local] * n_processes
        betas = pool.map(single_process, args)
        beta = sum(betas) / n_processes
    else:
        pool = Pool(processes=n_processes)
        # Might have slightly *more* samples than N
        N_local = math.ceil(N / n_processes)
        args = [(N_local, n_chunks, epsilons, alpha, n)] * n_processes
        betas = pool.map(single_process_toplevel, args)
        beta = sum(betas) / n_processes
    return beta


def get_beta_isotropic_laplace_monte_carlo(epsilon, alpha, n, k, N=1_000_000):
    """
    Simplified version of monte_carlo when there is no variance reduction.
    Isotropic because we use the same budget on each chunk
    Take k chunks of size n_i with \sum n_i = n
    Output \sum n_i/n Lap(1/n_i*eps) ~ (1/n) \sum Lap(1/eps)
    Btw, |(1/n) \sum Lap(1/eps)| > alpha iff |\sum Lap(1/eps)| > n*alpha (for the cache)
    """
    laplace_scale = 1 / (n * epsilon)
    laplace_noises = np.random.laplace(scale=laplace_scale, size=(N, k))
    aggregated_noise_total = np.sum(laplace_noises, axis=1)
    beta = np.sum(np.abs(aggregated_noise_total) > alpha) / N
    return beta


def get_beta_noisedown_montecarlo(
    existing_epsilons,
    new_epsilons,
    existing_noises,
    chunk_sizes,
    alpha,
    N=100_000,
):
    """
    Take m chunks of size n_i, with existing epsilons existing_epsilons_i, and existing noise x_i
    We apply NoiseDown on each chunk so that each chunk spends new_epsilons_i in terms of DP budget
    It gives us a new_noise_i for each chunk
    When we aggregate them, we can check whether the new noise is below alpha
    Repeat many times in parallel to compute beta


    Arbitrary sensitivity: the NoiseDown distribution works with epsilon-Lipschitz private mechanisms.

    Prop 6: An eps-Lipschitz private mechanism is esp/n-DP for the following adjacency relation:
        u and u' are adjacent iif d(u,u') <= 1/n

    Their setting is a bit weird, you release the whole datapoint, not a query with some sensitivity.
    x and x' are neighboring databases => d(q(x), q(x')) <= 1/n. But the reverse is not true.

    It's not a problem, forget about Prop 6 and prove what you need by hand:

    X -> U -> Y
    x -> u -> y

    If Q: u |-> u + V is espilon-Lipschitz private
    then M: x |-> q(x) + V is epsilon/n-DP

    Indeed, for all x, x' neighboring databases and output S, we have d(q(x), q(x')) <= 1/n
    so |ln P[u + V \in S] - ln P[u' + V \in S]| <= epsilon * 1/n
    """

    chunk_noises = []
    for chunk_id, n_i in enumerate(chunk_sizes):
        e1 = existing_epsilons[chunk_id] * n_i
        e2 = new_epsilons[chunk_id] * n_i

        assert e2 > e1, "You can only increase epsilon"

        x = existing_noises[chunk_id]
        p = np.random.random(N)

        # Compute masks, for a given trial n \in [N] exactly one mask is True, the others are False
        threshold_1 = (e1 / e2) * np.exp(-(e2 - e1) * abs(x))
        threshold_2 = threshold_1 + (e2 - e1) / (2 * e2)
        threshold_3 = threshold_2 + (e2 - e1) / (2 * e2) * np.exp(-(e2 - e1) * abs(x))

        print(f"Thresholds: {threshold_1}, {threshold_2}, {threshold_3}")

        below_1 = p < threshold_1
        below_2 = p < threshold_2
        below_3 = p < threshold_3
        mask_1 = below_1
        mask_2 = below_2 & ~below_1
        mask_3 = below_3 & ~below_2
        mask_4 = ~below_3

        # Possible outputs
        # NOTE: the partial pdfs in Algorithm 1 are not normalized. Should we normalize before sampling?
        # I assume yes? Reusing the same transformation as Cache DP, assuming they are correct

        # We compute the CDF and use https://en.wikipedia.org/wiki/Inverse_transform_sampling#Formal_statement
        u = np.random.uniform(0, 1, N)
        output_2 = np.log(u) / (e1 + e2)
        output_3 = np.log(u * (np.exp(abs(x) * (e1 - e2)) - 1.0) + 1.0) / (e1 - e2)
        output_4 = abs(x) - np.log(1.0 - u) / (e2 + e1)

        # Vectorized switch statement
        y = (
            mask_1 * x
            + mask_2 * output_2 * np.sign(x)
            + mask_3 * output_3 * np.sign(x)
            + mask_4 * output_4 * np.sign(x)
        )
        chunk_noises.append(y)

    # Concatenate to get shape (n_chunks, N)
    chunk_noises = np.array(chunk_noises)

    # Sum across chunks for each N, shape (N,) now
    aggregated_noise_total = np.sum(chunk_noises, axis=0)
    print(f"aggregated_noise_total.shape: {aggregated_noise_total.shape}")
    beta = (
        np.sum(aggregated_noise_total > alpha) + np.sum(aggregated_noise_total < -alpha)
    ) / N
    return beta


def get_epsilon_isotropic_laplace_monte_carlo(a, b, n, k, N, monte_carlo_cache=None):

    if k == 1:
        # We have a closed-form solution
        return get_epsilon_isotropic_laplace_concentration(a=a, b=b, n=n, k=k)

    if monte_carlo_cache is not None and (a, b, n, k) in monte_carlo_cache:
        # Cache stored in the planner
        n_hits = monte_carlo_cache.get("n_hits", 0) + 1
        monte_carlo_cache["n_hits"] = n_hits

        # Check how we are doing, once in a while
        if n_hits % 100 == 0:
            n_calls = (
                len(monte_carlo_cache) - 1 + n_hits
            )  # One entry is for the hits counter
            logger.debug(
                f"Monte Carlo cache hits: {n_hits}. Calls to utility function: {n_calls}. Hit rate: {n_hits / n_calls}"
            )

        return monte_carlo_cache[(a, b, n, k)]

    get_beta_fn = lambda eps: get_beta_isotropic_laplace_monte_carlo(
        epsilon=eps, alpha=a, n=n, k=k, N=N
    )

    epsilon_high = get_epsilon_isotropic_laplace_concentration(a=a, b=b, n=n, k=k)

    epsilon = binary_search(get_beta_fn=get_beta_fn, beta=b, epsilon_high=epsilon_high)

    if monte_carlo_cache is not None:
        monte_carlo_cache[(a, b, n, k)] = epsilon

    return epsilon


def get_epsilon_vr_monte_carlo(
    existing_epsilons, chunk_sizes, alpha, beta, N=100_000, n_processes=1
):

    # Loose bound with Chebyshev (Pr[|X| > a] <= Var[X]/a^2)
    # (we can tolerate higher variance than that because we know we have Laplace noise)
    target_var = alpha * beta**2

    # Final variance is 2/n**2 * sum_i 1/(sum_j eps_{ij}**2)
    # Sufficient condition to achieve target_var: 1/(sum_{ij} eps_{ij}**2) <= target_var * n / 2
    # Heuristic: we don't spend budget on chunks that satisfy this condition.

    n = sum(chunk_sizes)
    fresh_epsilon_mask = np.ones(len(chunk_sizes))
    epsilon_high = 0
    for i in range(len(chunk_sizes)):
        sq_eps_sum = np.sum(existing_epsilons[i] ** 2)
        if sq_eps_sum > 0 and 1 / sq_eps_sum <= target_var * n / 2:
            fresh_epsilon_mask[i] = 0
        else:
            sufficient_fresh_eps = np.sqrt(2 / (target_var * n) - sq_eps_sum)
            # print(f"Sufficient fresh eps for chunk {i}: {sufficient_fresh_eps}")
            epsilon_high = max(epsilon_high, sufficient_fresh_eps)

    def get_beta_fn(eps):
        fresh_epsilons = eps * fresh_epsilon_mask
        # print(f"Fresh epsilons: {fresh_epsilons}")
        return monte_carlo_beta_multieps(
            existing_epsilons=existing_epsilons,
            chunk_sizes=chunk_sizes,
            fresh_epsilons=fresh_epsilons,
            alpha=alpha,
            N=N,
            n_processes=n_processes,
        )

    if sum(fresh_epsilon_mask) == 0:
        print("No fresh epsilon needed.")
        return 0, fresh_epsilon_mask

    epsilon = binary_search(
        get_beta_fn=get_beta_fn, beta=beta, epsilon_high=epsilon_high
    )

    # NOTE: if we want to be really optimal, we can search for epsilons close to eps_tolerance (e.g. 1e-10)
    # and try to completely remove them from the list of fresh epsilons.

    return epsilon, fresh_epsilon_mask


def get_laplace_epsilon(a, b, n, k):
    # For retrocompatility
    return get_epsilon_isotropic_laplace_monte_carlo(a, b, n, k)


def get_epsilon_isotropic_laplace_concentration(a, b, n, k):
    if k == 1:
        epsilon = math.log(1 / b) / (n * a)
    elif k >= math.log(2 / b):
        # Concentration branch
        epsilon = math.sqrt(k * 8 * math.log(2 / b)) / (n * a)
    else:
        # b_M branch
        epsilon = (math.log(2 / b) * math.sqrt(8)) / (n * a)
    return epsilon


def get_sv_epsilon(alpha, beta, n, l=1 / 2):
    """
    l=1/2 for SV threshold = alpha/2.
    Outputs only the beta_SV from Overleaf.
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


def get_pmw_epsilon(alpha, beta, n):
    l = 1 / 2  # We can't change l in the vanilla PMW
    binary_eps = binary_search_epsilon(
        alpha=alpha, beta=beta, n=n, l=l, beta_tolerance=1e-5, extra_laplace=True
    )
    real_beta = sum_laplace_beta(binary_eps, n, alpha, l, extra_laplace=True)

    # Make sure that we didn't accidentatlly overspend budget
    assert real_beta < beta

    return binary_eps


def get_pmw_epsilon_loose(alpha, beta, n, max_pmw_k):
    """
    Outputs the smallest epsilon that gives error at most alpha with proba at least 1- beta/max_pmw_k
    See "Per-query accuracy guarantees for aggregated PMWs" lemma.
    """
    return 4 * math.log(max_pmw_k / beta) / (alpha * n)


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


def binary_search_sum_lap(alpha, beta, n, l):
    get_beta_fn = partial(sum_laplace_beta, n=n, alpha=alpha, l=l, extra_laplace=False)
    epsilon_high = loose_epsilon(alpha, beta, n, l)
    return binary_search(get_beta_fn, beta, epsilon_high)


def binary_search(
    get_beta_fn, beta, epsilon_high, beta_tolerance=1e-5, eps_tolerance=1e-10
):
    """
    Find the lowest epsilon that satisfies the failure probability guarantee.
    If extra_laplace = True, this is for a full PMW. Otherwise, it's just for a single SV.
    """
    eps_low = 0
    eps_high = epsilon_high

    # Make sure that the initial upper bound is large enough
    if get_beta_fn(eps_high) >= beta:
        logger.warning(
            "Epsilon high was not initialized properly, or Monte Carlo failed. I'll try to double it until it works and let you know if it fails again."
        )
        for _ in range(3):
            eps_high *= 2
            if get_beta_fn(eps_high) < beta:
                break
        if get_beta_fn(eps_high) >= beta:
            logger.error(
                f"Going yolo with epsilon high={eps_high} that gives {get_beta_fn(eps_high)} while we want {beta}"
            )

    real_beta = 0

    # Bring real_beta close to beta, but from below (conservative)
    while real_beta < beta - beta_tolerance:
        eps_mid = (eps_low + eps_high) / 2
        beta_mid = get_beta_fn(eps_mid)
        # print(
        #     f"{eps_low} < {eps_mid} < {eps_high} gives beta={beta_mid}. Target {beta}"
        # )

        if beta_mid < beta:
            eps_high = eps_mid
            real_beta = beta_mid
        else:
            # Don't update the real_beta, you can only exit the loop if real_beta < beta - beta_tolerance
            eps_low = eps_mid

        if (eps_high - eps_low < eps_tolerance) and (real_beta < beta - beta_tolerance):
            # If the epsilon estimate is close enough, stop (wasting up to `eps_tolerance` budget)
            # Helpful to avoid infinite loops when the true epsilon is actually 0
            # (e.g. because we had enough past epsilons, but not enough to detect it with Chebyshev)
            break

    return eps_high
