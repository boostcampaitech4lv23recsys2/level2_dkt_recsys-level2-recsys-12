import os

import torch
import wandb
from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds


def main(args):
    wandb.login()

    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
<<<<<<< HEAD

    train_data, valid_data = preprocess.split_data(train_data)
    
    wandb.init(project="dkt", config=vars(args),  name=f"{args.model}_{args.lr}_{args.batch_size}")
=======
    
    wandb.init(project="DKT_DKT", config=vars(args), entity="ai-tech-4-recsys-12")
    wandb.name=(f"{args.model}_{args.n_epochs}_{args.batch_size}_{args.lr}_{args.patience}")
>>>>>>> afb7b5b763009c8a3235a18c89e56da35e526bff
    model = trainer.get_model(args).to(args.device)
    
    if not args.kfold:
        train_data, valid_data = preprocess.split_data(train_data)
        trainer.run(args, train_data, valid_data, model)
    else:
        trainer.run_kfold(args, train_data, preprocess, model)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
