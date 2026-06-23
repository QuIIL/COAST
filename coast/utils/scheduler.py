import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineScheduler(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        epoch = self.last_epoch + 1

        if epoch < self.warmup_epochs:
            warmup_factor = epoch / float(self.warmup_epochs)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        progress = (epoch - self.warmup_epochs) / float(self.max_epochs - self.warmup_epochs)
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))

        return [
            self.min_lr + (base_lr - self.min_lr) * cosine_factor
            for base_lr in self.base_lrs
        ]
