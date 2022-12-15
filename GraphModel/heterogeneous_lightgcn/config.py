import argparse
import time
from typing import Optional, Union
from torch import Tensor

from args import parse_args

class CFG:
    args, wandb_kwargs = parse_args()
    
    use_cuda_if_available = True
    user_wandb = False
    wandb_kwargs = wandb_kwargs

    # data
    basepath = args.basepath
    loader_verbose = args.loader_verbose
    use_custom = args.use_custom

    # dump
    timestr = args.timestr  # 날짜

    output_dir = args.output_dir
    pred_file = args.pred_file

    # build
    embedding_dim = args.embedding_dim  # int
    feature_num_layers = args.feature_num_layers
    alpha = args.alpha  # Optional[Union[float, Tensor]]
    build_kwargs = args.build_kwargs
    feature_aggregation_method = args.feature_aggregation_method
    if not 0 <= feature_aggregation_method <= 3:
        raise Exception("feature_aggregation_method should be 0 or 1 or 2 or 3")

    # train
    n_epoch = args.n_epoch
    learning_rate = args.learning_rate
    weight_basepath = args.weight_basepath
    patience = args.patience
    edge_dropout = args.edge_dropout
    edge_dropout_rate = args.edge_dropout_rate


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}