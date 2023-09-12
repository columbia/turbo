# from opacus.accountants.analysis.rdp import compute_rdp, get_privacy_spent

from opacus.privacy_analysis import compute_rdp, get_privacy_spent

from turbo.budget import ALPHAS


def compute_noise_from_target_epsilon(
    target_epsilon,
    target_delta,
    epochs,
    batch_size,
    dataset_size,
    alphas=None,
    approx_ratio=0.01,
    min_noise=0.001,
    max_noise=1000,
):
    """
    Takes a target epsilon (eps) and some hyperparameters.
    Returns a noise scale that gives an epsilon in [0.99 eps, eps].
    The approximation ratio can be tuned.
    If alphas is None, we'll explore orders.
    """
    steps = epochs * dataset_size // batch_size
    sampling_rate = batch_size / dataset_size
    if alphas is None:
        alphas = ALPHAS

    def get_eps(noise):
        rdp = compute_rdp(sampling_rate, noise, steps, alphas)
        (
            epsilon,
            _,
        ) = get_privacy_spent(alphas, rdp, delta=target_delta)
        return epsilon

    # Binary search bounds
    noise_min = min_noise
    noise_max = max_noise

    # Start with the smallest epsilon possible with reasonable noise
    candidate_noise = noise_max
    candidate_eps = get_eps(candidate_noise)
    if candidate_eps > target_epsilon:
        raise ("Cannot reach target eps. Try to increase MAX_NOISE.")

    # Search up to approx ratio
    while (
        candidate_eps < (1 - approx_ratio) * target_epsilon
        or candidate_eps > target_epsilon
    ):
        if candidate_eps < (1 - approx_ratio) * target_epsilon:
            noise_max = candidate_noise
        else:
            noise_min = candidate_noise
        candidate_noise = (noise_max + noise_min) / 2
        candidate_eps = get_eps(candidate_noise)

    return candidate_noise
