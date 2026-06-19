import math


def cosine_warmup_lambda(epoch, *, lr_warmup, epochs):
    """Cosine learning rate scheduler."""
    if epoch < lr_warmup:
        return epoch / lr_warmup
    else:
        return 0.5 * (
            1.0 + math.cos(math.pi * (epoch - lr_warmup) / (epochs - lr_warmup + 1))
        )
