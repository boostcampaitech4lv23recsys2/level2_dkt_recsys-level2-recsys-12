import os

import pandas as pd
import torch
from config import CFG, logging_conf

from lightgcn.datasets import prepare_dataset
from lightgcn.models import build, inference
from lightgcn.utils import get_logger

from check_accuracy import check_accuracy

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                            recall_score, roc_auc_score)

logger = get_logger(logging_conf)
use_cuda = torch.cuda.is_available() and CFG.use_cuda_if_available
device = torch.device("cuda" if use_cuda else "cpu")

feature_aggregation_method_dict = {0:"sum_method", 1:"element_wise_method", 2:"nn.Linear_method", 3:"attention_method"}

if not os.path.exists(CFG.output_dir):
    os.makedirs(CFG.output_dir)


def main():
    logger.info("Task Started")

    logger.info("[1/4] Data Preparing - Start")
    train_data, test_data, user_problem_n_node, user_test_n_node, user_tag_n_node = prepare_dataset(
        device, CFG.basepath, verbose=CFG.loader_verbose, logger=logger.getChild("data")
    )
    logger.info("[1/4] Data Preparing - Done")

    logger.info("[2/4] Model Building - Start")
    model = build(
        user_problem_n_node, 
        user_test_n_node, 
        user_tag_n_node,
        model_type=2,
        logger=logger.getChild("build"),
        **CFG.build_kwargs
    )
    model.to(device)
    logger.info("[2/4] Model Building - Done")

    user_problem_test_data, user_test_test_data, user_tag_test_data = test_data 
    
    logger.info("[3/4] Inference - Start")
    
    pred = inference(
        model,
        user_problem_test_data,
        user_test_test_data,
        user_tag_test_data,
        logger=logger.getChild("infer"))
    
    logger.info("[3/4] Inference - Done")

    logger.info("[4/4] Result Dump - Start")
    
    pred_path = f"custom_{CFG.use_custom}_n_epoch_{CFG.n_epoch}_feature_aggregation_method_{feature_aggregation_method_dict[CFG.feature_aggregation_method]}_feature_num_layers_{''.join(str(CFG.feature_num_layers))}.csv"
    
    pred = pred.detach().cpu().numpy()
    pd.DataFrame({"prediction": pred}).to_csv(
        os.path.join(CFG.output_dir, pred_path), index_label="id"
    )
    logger.info("[4/4] Result Dump - Done")

    logger.info("Task Complete")
    logger.info("Test 성능 측정")
    auc, acc = check_accuracy(PRED_PATH = os.path.join(CFG.output_dir, pred_path))
    
    logger.info(f"Test performance: auc : {auc} \t\t acc : {acc}")
    logger.info(f"n_epoch: {CFG.n_epoch}, \t feature_num_layers: {CFG.feature_num_layers}, \t learning_rate: {CFG.learning_rate}, \t feature_aggregation_method: {feature_aggregation_method_dict[CFG.feature_aggregation_method]}")
    if CFG.edge_dropout:
        logger.info(f"edge_dropout_rate: {CFG.edge_dropout_rate}")
    else:
        logger.info(f"no edge_dropout")
        

if __name__ == "__main__":
    main()
