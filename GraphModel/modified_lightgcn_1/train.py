import pandas as pd
import torch
from config import CFG, logging_conf

from lightgcn.datasets import prepare_dataset
from lightgcn.models import build, train
from lightgcn.utils import class2dict, get_logger

if CFG.user_wandb:
    import wandb

    wandb_name = "JS" + "_" + str(CFG.learning_rate) + "_" + str(CFG.embedding_dim)
    wandb.init(**CFG.wandb_kwargs, config=class2dict(CFG), name=wandb_name)


logger = get_logger(logging_conf)
use_cuda = torch.cuda.is_available() and CFG.use_cuda_if_available
device = torch.device("cuda" if use_cuda else "cpu")
print(device)


def main():
    logger.info("Task Started")

    logger.info("[1/3] Data Preparing - Start")
    """
        prepare_dataset:
            설명:
                train_data.csv와 test_data.csv를 concat 후 split
            CFG.basepath: "/opt/ml/input/data/"
            verbose: True
                train, test, valid에 대해 생성한 edge와 label 정보 출력
            logger: logger.getChild("data")
            isTrain:
                isTrain=True:
                    return x_train, test_data, x_valid
                isTrain=False:
                    return x_train, test_data
    """
    """
        train_data, test_data, valid_data:
            train_data_proc: {'edge':[유저,문제], 'label':문제 맞췄는지} 형태의 dict임
            test_data는 'label'이 모두 -1임
        n_node:
            answerCode=-1인 경우 포함해서 모든 유저-문제 간의 interaction 개수
    """
    train_data, test_data, valid_data, feature_sorted_list, num_features_list, n_node = prepare_dataset(
        device,
        CFG.basepath,
        verbose=CFG.loader_verbose,
        logger=logger.getChild("data"),
        isTrain=True,
    )
    logger.info("[1/3] Data Preparing - Done")

    logger.info("[2/3] Model Building - Start")
    model = build(
        n_node,
        logger=logger.getChild("build"),
        feature_sorted_list=feature_sorted_list,
        num_features_list=num_features_list,
        embedding_dim=CFG.embedding_dim,
        num_layers=CFG.num_layers,
        alpha=CFG.alpha,
        feature_weight=CFG.feature_weight,
        **CFG.build_kwargs
    )
    model.to(device)

    if CFG.user_wandb:
        wandb.watch(model)

    logger.info("[2/3] Model Building - Done")

    logger.info("[3/3] Model Training - Start")
    
    train(
        model,
        train_data,
        valid_data=valid_data,
        n_epoch=CFG.n_epoch,
        learning_rate=CFG.learning_rate,
        use_wandb=CFG.user_wandb,
        weight=CFG.weight_basepath,
        logger=logger.getChild("train"),
    )
    logger.info("[3/3] Model Training - Done")

    logger.info("Task Complete")


if __name__ == "__main__":
    main()