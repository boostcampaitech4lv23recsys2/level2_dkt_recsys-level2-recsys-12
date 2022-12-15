import argparse
import time
from typing import Optional, Union
from torch import Tensor

def parse_args():
    timestr = time.strftime("%m.%d-%H:%M:%S")
    parser = argparse.ArgumentParser()
    
    ### 옵션 ###
    parser.add_argument("--kfold", default=0, type=int, help="apply k-fold if not 0")
    parser.add_argument(
        "--run_wandb", default=False, type=bool, help="option for running wandb"
    )
    wandb_kwargs = dict(project="DKT_LGCN", entity="ai-tech-4-recsys-12")
    
    parser.add_argument(
        "--basepath",
        default="../../data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--loader_verbose",
        default=True,
        type=bool,
        help="verbose on/off",
    )
    parser.add_argument(
        "--timestr",
        default=timestr,
        type=str,
        help="timestr"
    )
    parser.add_argument(
        "--output_dir",
        default="./output/",
        type=str,
        help="output_dir"
    )
    parser.add_argument(
        "--pred_file",
        default="submission_{}.csv".format(timestr),
        type=str,
        help="pred_file"
    )    
    parser.add_argument(
        "--embedding_dim",
        default=64,
        type=int,
        help="embedding_dim"
    )    
    parser.add_argument(
        "--feature_num_layers",
        default=[3, 3, 3],
        nargs='+',
        type=int,
    )   
    parser.add_argument(
        "--alpha",
        default=None,
        type=Optional[Union[float, Tensor]],
        help="alpha"
    )    
    parser.add_argument(
        "--build_kwargs",
        default={},
        type=dict,
        help="build_kwargs"
    )
    parser.add_argument(
        "--n_epoch",
        default=1800,
        type=int,
    )
    parser.add_argument(
        "--learning_rate",
        default=0.001,
        type=float,
    )    
    parser.add_argument(
        "--weight_basepath",
        default="./weight",
        type=str,
    )
    parser.add_argument(
        "--patience",
        default=20,
        type=int,
    )
    parser.add_argument(
        "--edge_dropout",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--edge_dropout_rate",
        default=0.1,
        type=float,
    )
    parser.add_argument(
        "--feature_aggregation_method",
        default=0,
        type=float,
    )
    parser.add_argument(
        "--use_custom",
        default=1,
        type=float,
    )
    parser.add_argument(
        "--user_features",
        default= [
                "userID_answerCode_mean",
                "userID_answerCode_var",
                "userID_elapsedTime_median",
                "feature_ensemble_elo_pred",
            ],
        nargs='+',
        type=str
    )
    parser.add_argument(
        "--item_features",
        default=[        
                "dayofweek_answerCode_mean",
                "hour_answerCode_mean",
                "hour_answerCode_Level",
                "KnowledgeTag_elapsedTime_median",
                "assessmentItemID_elo_pred",
                "0", "1", "2", "3", "4", "5", "6", "7", "8",
                # "testId_answerCode_mean",
                # "testId_answerCode_var",
                "assessmentItemID_answerCode_mean",
                "assessmentItemID_answerCode_var",
                # "KnowledgeTag_answerCode_mean",
                # "KnowledgeTag_answerCode_var",
                "dayofweek_answerCode_var",
                "hour_answerCode_var",
                "month_answerCode_var",
                "assessmentItemID_elapsedTime_median",
                # "testId_elo_pred",
                # "KnowledgeTag_elo_pred",
            ],
        nargs='+',
        type=str
    )
    parser.add_argument(
        "--train_test_split_rate",
        default=0.2,
        type=float,
    )
    args = parser.parse_args()
    return args, wandb_kwargs