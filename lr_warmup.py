import math
import torch

class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, d_model, warmup_steps=4000, last_epoch  = -1) -> None:
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(1, self._step_count)
        scale = (self.d_model ** -0.5) * min(
            step ** -0.5, step * (self.warmup_steps ** -1.5)
        )
        return [base_lr * scale for base_lr in self.base_lrs]