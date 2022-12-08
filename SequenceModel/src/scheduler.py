from custom_scheduler import CosineAnnealingWarmUpRestarts
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import get_linear_schedule_with_warmup


def get_scheduler(optimizer, args):
    if args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, mode="max", verbose=True
        )
    elif args.scheduler == "linear_warmup":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.total_steps,
        )
    elif args.scheduler == "cosine_annealing_warmup":
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer, T_0=150, T_mult=1, eta_max=0.1, T_up=10, gamma=0.5
        )
    return scheduler
