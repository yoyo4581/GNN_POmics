from torch.optim.lr_scheduler import LRScheduler
from collections import deque
import math

class LinearWarmup(LRScheduler):
    def __init__(self, optimizer, warmup_epochs: int, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        if self.last_epoch < self.warmup_epochs:
            scale = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * scale for base_lr in self.base_lrs]
        return self.base_lrs

class CosineDescent(LRScheduler):
    def __init__(self, optimizer, window: int=7, last_epoch=-1):
        self.min_loss = None
        self.amplitude = 1
        self.step_cosine = 0
        self.loss_queue = deque(maxlen=window)
        super().__init__(optimizer, last_epoch)

    def step(self, loss: float = None) -> None:
        if loss is None:
          super().step()
          return
        
        # First loss then append
        if self.min_loss is None:
          self.min_loss = loss
        
        # if the new loss is lower than the global minima then store it.
        self.min_loss = loss if loss < self.min_loss else self.min_loss
        self.loss_queue.append(loss)

        super().step()  # this triggers get_lr() as normal

    def descent_cosine(self, base_lr):
        # if the warmup window period is complete
        if len(self.loss_queue) == self.loss_queue.maxlen:
          # if the loss_queue does not hold a minimum less than or equal the global minima then descent
          if min(self.loss_queue) > self.min_loss:
            self.loss_queue.clear()

            self.step_cosine = 0
            self.min_loss = None

            self.amplitude = self.amplitude * 0.85
            scale = self.amplitude
            return base_lr * scale

        # 10 degree increment descending
        self.step_cosine += 1

        degree = 90 * (1 - 0.9 ** self.step_cosine)
        min_scale = self.amplitude/10
        scale = min_scale + (self.amplitude - min_scale) * math.cos(degree * (math.pi/180))
        return base_lr * scale
          

    def get_lr(self) -> list[float]:
        if not self.loss_queue:
          return self.base_lrs
          
        return [self.descent_cosine(base_lr)
                for base_lr in self.base_lrs]

# warmup = LinearWarmup(optimizer, warmup_epochs=10)
# cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=190)

# scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])