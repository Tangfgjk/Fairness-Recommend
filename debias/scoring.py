from contextlib import contextmanager
import torch


@contextmanager
def fixed_statistics(model):
    """Inference-mode BN/dropout behavior WITH autograd; restore mixed submodule modes."""
    states = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, state in states:
            module.training = state


def pair_logits(model, heads, relation, high, low):
    # cuDNN GRU inference forward does not retain the reserve space needed by backward.
    # Use native differentiable CUDA kernels only for this auxiliary forward; keep
    # frozen BN/dropout behavior and leave the original BCE backend untouched.
    with fixed_statistics(model), torch.backends.cudnn.flags(enabled=False):
        logits = model.score_logits(torch.cat((heads, heads)), torch.cat((relation, relation)),
                                    torch.cat((high, low)))
    return logits[:len(heads)], logits[len(heads):]
