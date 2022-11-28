import os
import sys

import numpy as np
import torch

from args import parse_args
from src import trainer
from src.dataloader import Preprocess

sys.path.append("/opt/ml/input/code")
import feature_engineering as fe


def main(args):

    # SETTINGS
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # PREPROCESS & LOAD DATA
    preprocess = Preprocess(args)
    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()

    # MAKE EMBEDDING LAYER INPUTS SIZE
    # dataloader.py의 load_data_from_file 함수에서 처리되던
    # embedding_layer의 input 크기 결정 작업을 여기서 처리하고 모델에 보내줍니다
    args.embed_layer_input_size_list = [
        len(np.load(os.path.join(args.asset_dir, f"{col}_classes.npy")))
        for col in fe.SEQ_CATE_COLS
    ]

    # NO K-FOLD
    if not args.kfold:
        model = trainer.load_model(args).to(args.device)
        trainer.inference(args, test_data, model)
    # K-FOLD
    else:
        for fold in range(args.kfold):
            print(f"Inference... fold {fold}")
            model = trainer.load_model_kfold(args, fold).to(
                args.device
            )  # fold번째 pt 파일 불러옴
            trainer.inference_kfold(args, test_data, model, fold)  # pt 파일 가지고 inference


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
