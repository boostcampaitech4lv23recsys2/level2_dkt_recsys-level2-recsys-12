import os
import sys

import numpy as np
import torch

import wandb
from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds

sys.path.append("/opt/ml/input/code")
import feature_engineering as fe


def main(args):

    # SETTINGS
    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # PREPROCESS & LOAD DATA
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    # MAKE EMBEDDING LAYER INPUTS SIZE
    # dataloader.py의 load_data_from_file 함수에서 처리되던
    # embedding_layer의 input 크기 결정 작업을 여기서 처리하고 모델에 보내줍니다
    args.embed_layer_input_size_list = [
        len(np.load(os.path.join(args.asset_dir, f"{col}_classes.npy")))
        for col in fe.SEQ_CATE_COLS
    ]

    # GET MODEL
    model = trainer.get_model(args).to(args.device)

    # NO K-FOLD
    if not args.kfold:
        if args.run_wandb:
            wandb.login()
            wandb.init(
                project="LastQuery", config=vars(args), entity="ai-tech-4-recsys-12"
            )
            wandb.run.name = f"{args.model}_{args.batch_size}_{args.lr}_{args.patience}"
            wandb.config = vars(args)
        train_data, valid_data = preprocess.split_data(train_data)
        trainer.run(args, train_data, valid_data, model)
    # K-FOLD
    else:
        if args.run_wandb:
            wandb.login()
        trainer.run_kfold(args, train_data, preprocess, model)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
