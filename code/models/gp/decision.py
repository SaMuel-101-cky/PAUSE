import gpytorch
import torch


from .llr_utils import (
    retrieve_posteriors_from_llrm,
    get_upper_triangle,
    _truncate_to_upper_horizon
)

from .backward_induction import G_st



def decision_helper(
    model_pool,
    likelihood_pool,
    penalty,
    cost,
    llrs,
    is_llr_as_sufficient_statistic=False,
    posteriors=None,
    upper_horizon=None,
):
    """
    Args:
    - model_pool (list): of length time_steps containing GPyTorch models.
    - likelihood_pool (list): of length time_steps containing GPyTorch likelihoods.
    - cost (float): sampling cost
    - llrs (Tensor): of size (batch_size, time_steps, num_classes, num_classes)

    Returns:
    - (masked) hitting_time (Tensor): of size (batch_size, time_steps) of binary elements.
       1. indicates the hit time. Only the first hit time is selected.
    """
    llrs, posteriors, horizon = _truncate_to_upper_horizon(
        llrs, posteriors, upper_horizon
    )
    batch_size, time_steps, num_classes, num_classes_ = llrs.shape
    assert num_classes == num_classes_
    assert horizon == time_steps
    assert horizon == len(model_pool) + 1
    assert horizon == len(likelihood_pool) + 1

    # posteriors: A tensor of size (batch_size, time_steps, num_classes).
    # It may differ from retrieve_posteriors_from_llrm(llrs) when llrs are
    # prior-free evidence and stopping risk should use MAP posterior.
    if posteriors is None:
        posteriors = retrieve_posteriors_from_llrm(llrs)
    else:
        posteriors = posteriors.to(llrs.device)
        assert posteriors.shape == (batch_size, time_steps, num_classes)
    assert posteriors.shape[1] == time_steps

    # stopping risk and predicted labels: both size of (batch_size, time_steps)
    Gst, predictions = G_st(posteriors, penalty)

    if is_llr_as_sufficient_statistic:
        sufficient_statistic = get_upper_triangle(llrs)
    else:
        sufficient_statistic = posteriors

    hitting_time = torch.zeros(batch_size, time_steps, device=llrs.device).int()
    for t in range(1, time_steps):
        # current time: t-1
        current_statistic = sufficient_statistic[:, t - 1, :]    # input at time t-1
        current_Gst = Gst[:, t - 1]

        # continuation risk at time t
        with torch.no_grad(), gpytorch.settings.fast_pred_var():

            Gtilde = likelihood_pool[-t](model_pool[-t](current_statistic)).mean + cost

        hitting_time[:, t - 1] = current_Gst <= Gtilde

    hitting_time[:, t] = torch.ones(batch_size, device=llrs.device)
    # _, predictions[:, t] = G_st(posteriors[:, t, :], penalty)

    hitting_time = torch.argmax(hitting_time, dim=1).float()
    predictions = predictions[range(len(predictions)), hitting_time.int()]

    hitting_time += 1
    assert torch.min(hitting_time) > 0 and torch.max(hitting_time) <= time_steps

    return hitting_time, predictions


def predictive_continuation_risk(x_new, likelihood, model, cost):
    likelihood.eval()
    model.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        return likelihood(model(x_new)).mean.detach().cpu().numpy() + cost