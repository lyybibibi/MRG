import torch
import numpy as np


def penalty_builder(penalty_config):
    if penalty_config == '':
        return lambda x, y: y
    pen_type, alpha = penalty_config.split('_')
    alpha = float(alpha)
    if pen_type == 'wu':
        return lambda x, y: length_wu(x, y, alpha)
    if pen_type == 'avg':
        return lambda x, y: length_average(x, y, alpha)


def length_wu(length, logprobs, alpha=0.):
    """
    NMT length re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`.
    """

    modifier = (((5 + length) ** alpha) /
                ((5 + 1) ** alpha))
    return logprobs / modifier


def length_average(length, logprobs, alpha=0.):
    """
    Returns the average probability of tokens in a sequence.
    """
    return logprobs / length


def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x

def tile(x, n_tile, dim=0):
    """
    Repeat tensor x along dimension dim by n_tile times.

    This is used in beam search sampling to expand tensors from [B, ...] to
    [B * beam_size, ...] (e.g., k, fl, etc.).
    """
    if x is None:
        return None
    if not torch.is_tensor(x):
        raise TypeError("tensor_utils.tile expects a torch.Tensor, got {}".format(type(x)))

    if dim < 0:
        dim = x.dim() + dim

    # repeat factors for each dimension
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x_rep = x.repeat(*repeat_idx)

    # reorder to make tiled blocks contiguous like TF tile behavior
    init_dim = x.size(dim)
    order_index = torch.arange(init_dim, device=x.device).repeat(n_tile)
    for i in range(1, n_tile):
        order_index[i * init_dim:(i + 1) * init_dim] = order_index[:init_dim] + i * init_dim

    return torch.index_select(x_rep, dim, order_index)

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
