"""Fixed-mass smooth Top-N with implicit differentiation of the threshold."""
import math
import numpy as np
import torch


class _SmoothTopN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, n, temperature):
        # Center first: preserves precision and makes large common offsets harmless.
        z = logits.double()
        z = z - z.mean(-1, keepdim=True)
        lo = z.min(-1, keepdim=True).values - 50 * temperature
        hi = z.max(-1, keepdim=True).values + 50 * temperature
        for _ in range(80):
            t = (lo+hi)/2
            mass = torch.sigmoid((z-t)/temperature).sum(-1, keepdim=True)
            lo = torch.where(mass > n, t, lo)
            hi = torch.where(mass > n, hi, t)
        w = torch.sigmoid((z-(lo+hi)/2)/temperature)
        a = w*(1-w)/temperature
        ctx.save_for_backward(a)
        ctx.dtype = logits.dtype
        return w.to(logits.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (a,) = ctx.saved_tensors
        g = grad_output.double()
        denominator = a.sum(-1, keepdim=True)
        # In the fully saturated limit the local derivative is zero.
        safe = denominator.clamp_min(torch.finfo(torch.float64).tiny)
        grad = a*(g-(a*g).sum(-1, keepdim=True)/safe)
        return grad.to(ctx.dtype), None, None


def smooth_top_n(logits, n, temperature):
    if logits.ndim != 2 or not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("Expected finite 2-D floating logits")
    if not isinstance(n, int) or not 0 < n < logits.shape[1]:
        raise ValueError("Top-N must be between zero and catalog size exclusively")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    return _SmoothTopN.apply(logits, n, temperature)


def rule_targets(distance, ns):
    distance = np.asarray(distance, float)
    if distance.ndim != 2 or len(distance) == 0 or not np.isfinite(distance).all():
        raise ValueError("Invalid training educational distance")
    rank = np.argsort(distance, axis=1, kind='stable')
    output = []
    for n in ns:
        if not 0 < n < distance.shape[1]:
            raise ValueError("Invalid N")
        output.append(np.bincount(rank[:, :n].ravel(), minlength=distance.shape[1])/(len(rank)*n))
    return np.array(output)


def exposure_loss(logits, targets, ns, temperature):
    target = torch.as_tensor(targets, dtype=logits.dtype, device=logits.device)
    if target.shape != (len(ns), logits.shape[1]) or not torch.isfinite(target).all() or (target < 0).any():
        raise ValueError("Bad exposure target")
    if not torch.allclose(target.sum(1), torch.ones(len(ns), device=target.device, dtype=target.dtype), atol=1e-6):
        raise ValueError("Each exposure target must have unit mass")
    per_n = torch.stack([.5*(smooth_top_n(logits, int(n), temperature).mean(0)/n-target[i]).abs().sum()
                         for i, n in enumerate(ns)])
    return per_n.mean(), per_n


def hard_exposure(logits, ns):
    z = np.asarray(logits, float)
    rank = np.argsort(-z, axis=1, kind='stable')
    return np.array([np.bincount(rank[:, :n].ravel(), minlength=z.shape[1])/(len(z)*n) for n in ns])


def exposure_snapshot(logits, targets, ns, temperature):
    with torch.no_grad():
        loss, per_n = exposure_loss(torch.as_tensor(logits, dtype=torch.float64), targets, ns, temperature)
    hard = hard_exposure(logits, ns)
    return {'soft_mean_TV': float(loss), 'soft_TV': per_n.tolist(),
            'hard_TV': (.5*np.abs(hard-targets).sum(1)).tolist()}
