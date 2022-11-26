import os

import torch
import wandb
from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds


def main(args):

    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()

    model = trainer.get_model(args).to(args.device)

    if not args.kfold:
        if args.run_wandb:
            wandb.login()
            wandb.init(project="DKT_MODEL", config=vars(args), entity="ai-tech-4-recsys-12")
            wandb.run.name = (
                f"{args.batch_size}_{args.lr}_{args.patience}"
            )
            wandb.config = vars(args)
        train_data, valid_data = preprocess.split_data(train_data)
        trainer.run(args, train_data, valid_data, model)
    else:
        trainer.run_kfold(args, train_data, preprocess, model)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
