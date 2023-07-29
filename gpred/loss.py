import typing

import torch
# from torch.nn.functional import binary_cross_entropy_with_logits


def sigmoid(x):
    """Computes :math:`\sigma(x) = \frac{1}{1 + \exp(-x)}`."""
    if type(x) is torch.Tensor:
        return 1 / 1 + torch.exp(-x)
    else:
        import numpy as np

        return 1 / (1 + np.exp(-x))


def ce(x, y, logits=True):
    """Computes :math:`-y \log(\sigma(x)) = y \log(1 + \exp(-x))`.

    Clips the output of log to a minimum of -100, like Torch.
    """
    import numpy as np

    if logits:
        return y * (1 + torch.exp((-x).clamp(max=80))).log()
    else:
        return -y * x.clamp(min=np.exp(-100)).log()


def binary_cross_entropy_with_logits(x, y, weights=None):
    """Reimplementation of torch.nn.functional.binary_cross_entropy_with_logits().

    Numerically more stable than cross entropy computed on sigmoid outputs.
    Clips the output of log to a minimum of -100, like Torch.

    .. math::

        CE(x, y) &= -y \log(\sigma(x)) - (1 - y) \log(1 - \sigma(x)) \\
                 &= -y \log(\sigma(x)) - (1 - y) \log(\sigma(-x)) \\
                 &= y \log(1 + \exp(-x)) + (1 - y) \log(1 + \exp(x))

        1 - \sigma(x) &= 1 - \frac{1}{1 + \exp(-x)} \\
                      &= \frac{\exp(-x)}{1 + \exp(-x)} \\
                      &= \frac{1}{1 + \exp(x)} \\
                      &= \sigma(-x)

        \log(\sigma(x)) &= \log \frac{1}{1 + \exp(-x)} \\
                        &= -\log(1 + \exp(-x))

    Args:
        x (torch.Tensor, (-1, N)): Cross entropy input.
        y (torch.Tensor, (-1, N)): Cross entropy target.
    Returns:
        (torch.Tensor, (1,)): Binary cross entropy scalar.
    """
    if weights is not None:
        return (weights[0] * ce(x, y) + weights[1] * ce(-x, 1 - y)).sum(dim=1).mean()
    return (ce(x, y) + ce(-x, 1 - y)).sum(dim=1).mean()


def dnf_cross_entropy(
    s_predict: torch.Tensor,
    s_dnf: torch.Tensor,
    weights: typing.Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Args:
        s_predict: Logit predictions as a [-1, N] (size_batch, num_props) float array.
        s_dnf: Partial state as a [-1, 2, N] (size_batch, pos/neg, num_props) float array.
        weights: Class weights as a [2, N] (pos/neg, num_props) float array.
    Returns:
        [1] Binary cross entropy scalar as a singleton float array.
    """
    # [-1, 2, N] -> [-1, N] (num_props)
    s_pos = s_dnf[:, 0, :]
    s_neg = s_dnf[:, 1, :]

    # [-1, N] (num_props)
    if weights is not None:
        ce_pos = weights[0:1] * ce(s_predict, s_pos)
        ce_neg = weights[1:2] * ce(-s_predict, s_neg)
    else:
        ce_pos = ce(s_predict, s_pos)
        ce_neg = ce(-s_predict, s_neg)

    # [-1, N] -> [1]
    ce_combined = ce_pos.sum() + ce_neg.sum()

    return ce_combined


def _dnf_cross_entropy(s_predict, dnf, mask, idx_static):
    """
    Args:
        s_predict (torch.Tensor, float, (-1, N)): Symbolic state as tensor of
            proposition logits.
        dnf (torch.Tensor, bool, (-1, 2, N, M)): Tensor of M conjunctions of N
            positive and N negative propositions.
        mask (torch.Tensor, bool, (-1, M)): Boolean mask over active
            conjunctions for each example.
        idx_static (torch.Tensor, bool, (N,)): Boolean mask over static propositions.
    Returns:
        (torch.Tensor, float, (1,)): Binary cross entropy scalar.
    """
    # [-1, 2, M]
    c_mask = mask.view(-1, 2, mask.shape[-1])

    # [-1, M]
    c_mask_pre = c_mask[:, 0, :]
    c_mask_post = c_mask[:, 1, :]

    # [-1, 2, 2, N, M]
    c = dnf.view(-1, 2, *dnf.shape[1:])

    # [-1, 2, N, M]
    c_pre = c[:, 0, :, :, :]
    c_post = c[:, 1, :, :, :]

    # [-1, 2, N]
    # Find common propositions across pre conjunctions
    s_pre = c_pre.sum(dim=-1) == c_mask_pre.sum(dim=-1)[:, None, None]

    # [-1, N]
    s_pre_pos = s_pre[:, 0, :]
    s_pre_neg = s_pre[:, 1, :]

    # [-1, N, M]
    c_post_pos = c_post[:, 0, :, :]
    c_post_neg = c_post[:, 1, :, :]

    # [-1, M]
    # Mark conjunctions in post that modify static propositions in pre as violations
    mask_post_pos_violations = (c_post_pos & (s_pre_neg & idx_static)[:, :, None]).any(
        dim=1
    )
    mask_post_neg_violations = (c_post_neg & (s_pre_pos & idx_static)[:, :, None]).any(
        dim=1
    )
    mask_post_violations = mask_post_pos_violations | mask_post_neg_violations

    # [-1, 2, N, M]
    # Zero out violating conjunctions
    c_post &= ~mask_post_violations[:, None, None, :]

    # [-1, M]
    # Filter out violating conjunctions from mask
    c_mask_post &= ~mask_post_violations

    # [-1, 2, N]
    # Find common propositions across post conjunctions
    s_post = c_post.sum(dim=-1) == c_mask_post.sum(dim=-1)[:, None, None]

    # [-1, N]
    # Filter out propositions in post that conflict with pre
    s_pre_pos = s_pre[:, 0, :]
    s_pre_neg = s_pre[:, 1, :]
    s_post_pos = s_post[:, 0, :]
    s_post_neg = s_post[:, 1, :]
    s_post_pos |= s_pre_pos & ~s_post_neg
    s_post_neg |= s_pre_neg & ~s_post_pos

    # [-1, 2, N]
    # Concatenate pre and post states
    s = torch.stack((s_pre, s_post), dim=1).view(-1, *s_post.shape[1:])

    # [-1, N]
    s_pos = s[:, 0, :]
    s_neg = s[:, 1, :]
    mask_constant = ~(s_pos | s_neg)

    # # [-1]
    # # Total number of conjunctions for each dnf
    # s_mask = mask.sum(dim=1)

    # [-1]
    # Compute cross entropy for all propositions
    ce_pos = ce(s_predict, s_pos).sum(dim=1)
    ce_neg = ce(-s_predict, s_neg).sum(dim=1)

    # Only count cross entropy for propositions that have nonzero dnf sizes
    # ce_combined = (ce_pos + ce_neg) * torch.sign(s_mask)
    ce_combined = ce_pos + ce_neg

    # [-1, N]
    s_constant = s_predict * mask_constant.float()

    # [-1, 2, N]
    s_constant = s_constant.view(-1, 2, *s_constant.shape[1:])

    # [-1, N]
    s_constant_pre = torch.sigmoid(s_constant[:, 0, :])
    s_constant_post = torch.sigmoid(s_constant[:, 1, :])

    # [-1]
    l2_constant = torch.nn.functional.pairwise_distance(s_constant_pre, s_constant_post)
    # l2_norm = torch.norm(0.5 * (s_constant_pre + s_constant_post), p=2, dim=1)

    return ce_combined.sum() + 0.001 * l2_constant.mean()  # + 0.001 * l2_norm.mean()


def min_cross_entropy(s_predict, dnf, mask, logits=True):
    """Compute the minimum cross-entropy between the predicted symbolic state
    and pre/post-condition disjunctive normal form.

    The loss is averaged across the batch.

    .. math::

        ce_{min_{pre}} = \min_{j \in [1 \dots M]} \sum_{i=1}^N p(s_i^{(j)}) \log I[dnf_i^{(j)}]
        ce_{min_{post}} = \min_{j \in [1 \dots M]} \sum_{i=1}^N p(s_i^{(j)}) \log I[dnf_i^{(j)}]
        ce = ce_{min_{pre}} + ce_{min_{post}}

    Args:
        s_predict (torch.Tensor, (-1, N)): Symbolic state as tensor of
            proposition probabilities.
        dnf (torch.Tensor, (-1, 2, N, M)): Tensor of M conjunctions of N
            positive and N negative propositions.
        mask (torch.Tensor, (-1, M)): Boolean mask over active conjunctions for each example.
    Returns:
        (torch.Tensor, (-1,)): Minimum cross-entropy score.
    """
    # [-1, N, M]
    c_pos = dnf[:, 0, :, :]
    c_neg = dnf[:, 1, :, :]

    # [-1, M]
    ce_pos = ce(s_predict[:, :, None], c_pos, logits=logits).sum(dim=1)
    if logits:
        ce_neg = ce(-s_predict[:, :, None], c_neg, logits=True).sum(dim=1)
    else:
        ce_neg = ce(1 - s_predict[:, :, None], c_neg, logits=False).sum(dim=1)

    ce_pos[~mask] = float("inf")
    ce_neg[~mask] = float("inf")

    # ce_pos[~mask] = 0
    # ce_neg[~mask] = 0

    # [-1]
    ce_min, _ = (ce_pos + ce_neg).min(dim=1)
    # print(ce_min)
    # ce_min = (ce_pos + ce_neg).sum(dim=1) / mask.sum(dim=1)
    # print(ce_min)

    # if (ce_min == float("inf")).any() or (ce_min != ce_min).any():
    #     raise ValueError(f"Bad loss: ce_min={ce_min}")

    return ce_min.mean()
