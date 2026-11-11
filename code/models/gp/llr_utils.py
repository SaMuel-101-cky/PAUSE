import torch
from torch import Tensor


def _scaled_logits_and_log_prior(logits, log_prior=None, temperature=None):
    num_classes = logits.shape[-1]
    if log_prior is None:
        log_prior = torch.zeros(
            num_classes,
            device=logits.device,
            dtype=logits.dtype,
        )
    else:
        log_prior = log_prior.to(device=logits.device, dtype=logits.dtype)

    scaled_logits = logits
    if temperature is not None:
        temperature = temperature.to(device=logits.device, dtype=logits.dtype)
        scaled_logits = logits / temperature.view(1, -1, 1)
    return scaled_logits, log_prior


def calibrated_posteriors_from_logits(logits, log_prior=None, temperature=None):
    """Return calibrated MAP posteriors without materializing LLRs."""
    scaled_logits, log_prior = _scaled_logits_and_log_prior(
        logits,
        log_prior,
        temperature,
    )
    return torch.softmax(scaled_logits + log_prior.view(1, 1, -1), dim=-1)


def extract_llrs_from_logits(
    logits,
    labels,
    log_prior=None,
    temperature=None,
    return_posteriors=False,
):
    """Convert logits to prior-free evidence LLRs.

    Temperature is applied before both branches. The returned ``llrs`` are
    evidence differences and do not include class priors. Optional posteriors
    are calibrated with ``softmax(scaled_logits + log_prior)`` for stopping
    risk and predicted labels.
    """
    scaled_logits, log_prior = _scaled_logits_and_log_prior(
        logits,
        log_prior,
        temperature,
    )

    llrs = scaled_logits.unsqueeze(-1) - scaled_logits.unsqueeze(-2)
    posteriors = torch.softmax(scaled_logits + log_prior.view(1, 1, -1), dim=-1)

    if return_posteriors:
        return llrs, labels, 0, logits.shape[0], logits.shape[1], posteriors
    return llrs, labels, 0, logits.shape[0], logits.shape[1]


def posterior_grid_to_llrs_upperdiag(x_new):
    assert x_new.ndim == 2
    x_new += 1e-16
    batch_size, num_classes = x_new.shape
    device = x_new.device
    kC2 = int(num_classes * (num_classes - 1) / 2)
    llrs_upperdiag = torch.zeros(batch_size, kC2, device=device)
    ind = 0
    for i in range(num_classes):
        for j in range(num_classes):
            if j <= i:
                continue
            llrs_upperdiag[:, ind] = x_new[:, i] / x_new[:, j]
            ind += 1

    return torch.log(llrs_upperdiag)

def retrieve_posteriors_from_llrm(llrm: Tensor, prior_ratio_matrix=None):
    """
    Args:
    - llrm: pytorch Tensor of size (batch_size, time_steps, num_classes, num_classes).
            Strictly antisymmetric matrix, diagnonal must be all-zero.
            lambda_kl = log(p(x | label_k) / p(x | label_l))
            note: (batch_size, num_classes, num_classes) will also work.
    - prior_ratio_matrix: pytorch Tensor of size (num_classes, num_classes)
                          Strictly antisymmetric matrix, diagnonal must be all-one.
                          chi_kl = p(label_k) / p(label_l)
    Returns:
    - posterior: A tensor of size (batch_size, time_steps, num_classes)

    """
    if prior_ratio_matrix is None:
        prior_ratio_matrix = torch.ones_like(llrm)
    else:
        llrm_shape = llrm.shape
        prior_ratio_matrix = prior_ratio_matrix.repeat(llrm_shape[0], llrm_shape[1], 1, 1)

    # No need to neither add 1 to the denominator or avoid the diagonal summation,
    # because the diagonal is explog(p_k/p_k) == 1 and will cancel out
    posterior = 1 / torch.sum(torch.exp(llrm) * prior_ratio_matrix, dim=-2)
    return posterior


def get_upper_triangle(scores_full):
    """
    Get the upper triangle of the scores matrix or llr matrix.

    Args:
        scores_full: A PyTorch Tensor with shape
            (batch_size (optional), num thresholds (optional), time_steps (optional), num_classes, num_classes).
    Returns:
        scores_vec: A PyTorch Tensor with shape
            (batch_size (optional), num thresholds (optional), time_steps (optional) , kC2) where k is the num_classes.
    """
    ndim = scores_full.ndim
    assert ndim > 1, "The input must have at least two dimensions."
    num_classes = scores_full.shape[-1]

    upper_triangle_indices = torch.triu_indices(num_classes, num_classes, offset=1)

    scores_vec = scores_full[..., upper_triangle_indices[0], upper_triangle_indices[1]]
    return scores_vec


def _truncate_to_upper_horizon(llrs, posteriors, upper_horizon):
    if llrs is None:
        if posteriors is None:
            raise ValueError("at least one of llrs or posteriors is required")
        horizon = _resolve_upper_horizon(posteriors.shape[1], upper_horizon)
    else:
        horizon = _resolve_upper_horizon(llrs.shape[1], upper_horizon)
        llrs = llrs[:, :horizon]
    if posteriors is not None:
        posteriors = posteriors[:, :horizon]
    return llrs, posteriors, horizon

def _resolve_upper_horizon(time_steps, upper_horizon):
    if upper_horizon is None:
        return time_steps
    if isinstance(upper_horizon, bool) or not isinstance(upper_horizon, int):
        raise ValueError("upper_horizon must be an integer or None")
    if upper_horizon < 2 or upper_horizon > time_steps:
        raise ValueError(
            f"upper_horizon must be in [2, {time_steps}], got {upper_horizon}"
        )
    return upper_horizon

