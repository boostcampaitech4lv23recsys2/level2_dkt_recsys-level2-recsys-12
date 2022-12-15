import os

import numpy as np
import pandas as pd
import torch
from config import CFG
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                            recall_score, roc_auc_score)

from heterogeneous_lightgcn import *

from typing import Optional, Tuple

from torch import Tensor

from check_accuracy import check_accuracy

from torch_geometric.deprecation import deprecated
from torch_geometric.typing import OptTensor

import copy

feature_aggregation_method_dict = {0:"sum_method", 1:"element_wise_method", 2:"nn.Linear_method", 3:"attention_method"}

def dropout_edge(user_problem_edge_index: Tensor,
                user_test_edge_index: Tensor,
                user_tag_edge_index: Tensor,
                p: float = 0.5,
                force_undirected: bool = False,
                training: bool = True) -> Tuple[Tensor, Tensor]:

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                        f'(got {p}')

    if not training or p == 0.0:
        edge_mask = user_problem_edge_index.new_ones(user_problem_edge_index.size(1), dtype=torch.bool)
        return user_problem_edge_index, user_test_edge_index, user_tag_edge_index, edge_mask
    
    row, col = user_problem_edge_index

    edge_mask = torch.rand(row.size(0), device=user_problem_edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    user_problem_edge_index = user_problem_edge_index[:, edge_mask]
    user_test_edge_index = user_test_edge_index[:, edge_mask]
    user_tag_edge_index = user_tag_edge_index[:, edge_mask]

    if force_undirected:
        user_problem_edge_index = torch.cat([user_problem_edge_index, user_problem_edge_index.flip(0)], dim=1)
        user_test_edge_index = torch.cat([user_test_edge_index, user_test_edge_index.flip(0)], dim=1)
        user_tag_edge_index = torch.cat([user_tag_edge_index, user_tag_edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return user_problem_edge_index, user_test_edge_index, user_tag_edge_index, edge_mask

def build(
        user_problem_n_node,
        user_test_n_node,
        user_tag_n_node,
        model_type=0,
        logger=None,
        **kwargs):
    model = Heterogeneous_LightGCN(user_problem_nodes=user_problem_n_node,
                    user_test_nodes=user_test_n_node,
                    user_tag_nodes=user_tag_n_node,
                    embedding_dim=CFG.embedding_dim,
                    feature_aggregation_method=CFG.feature_aggregation_method,
                    alpha=CFG.alpha,
                    **kwargs)
    if model_type == 0:
        return model
    
    elif model_type == 1:
        pretrained_weight_path = f"./weight/pretrained_model.pt"
        if not os.path.isfile(pretrained_weight_path):
            logger.fatal("Pretrained_weight File Not Exist")
        logger.info("Load pretrained model")
        state = torch.load(pretrained_weight_path)["model"]
        model.load_state_dict(state)
        return model
    
    elif model_type == 2:
        inference_weight_path = f"./weight/n_epoch_{CFG.n_epoch}_feature_aggregation_method_{feature_aggregation_method_dict[CFG.feature_aggregation_method]}_feature_num_layers_{''.join(str(CFG.feature_num_layers))}_model.pt"
        if not os.path.isfile(inference_weight_path):
            logger.fatal("Inference_weight File Not Exist")
        state = torch.load(inference_weight_path)["model"]
        model.load_state_dict(state)
        return model


def train(
    model,
    
    user_problem_train_data,
    user_test_train_data,
    user_tag_train_data,
    
    user_problem_valid_data=None,
    user_test_valid_data=None,
    user_tag_valid_data=None,
    
    user_problem_test_data=None,
    user_test_test_data=None,
    user_tag_test_data=None,
    
    n_epoch=50,
    learning_rate=0.001,
    use_wandb=False,
    logger=None,
    model_type=0,

):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if not os.path.exists(CFG.basepath):
        os.makedirs(CFG.basepath)

    if user_problem_valid_data is None:
        eids = np.arange(len(user_problem_train_data["label"]))
        eids = np.random.permutation(eids)[:1000]
        
        user_problem_edge, user_test_edge, user_tag_edge, label = user_problem_train_data["edge"], user_test_train_data["edge"], user_tag_train_data["edge"], user_problem_train_data["label"]
        
        label = label.to("cpu").detach().numpy()
        
        user_problem_valid_data = dict(edge=user_problem_edge[:, eids], label=label[eids])
        user_test_valid_data = dict(edge=user_test_edge[:, eids], label=label[eids])
        user_tag_valid_data = dict(edge=user_tag_edge[:, eids], label=label[eids])

    valid_label = copy.deepcopy(user_problem_valid_data["label"])
    user_problem_valid_data["label"] = (
        user_problem_valid_data["label"].to("cpu").detach().numpy()
    )
    user_test_valid_data["label"] = (
        user_test_valid_data["label"].to("cpu").detach().numpy()
    )
    user_tag_valid_data["label"] = (
        user_tag_valid_data["label"].to("cpu").detach().numpy()
    )

    logger.info(f"Training Started : n_epoch={n_epoch}")
    best_auc, best_acc = 0, 0
    for e in range(1, n_epoch + 1):
        # forward
        """
        len(train_data): 2
        len(train_data["edge"]): 2
        len(train_data["edge"][0]): 1980184 (문제 풀이 기록 개수)
        type(train_data["edge"][0]): <class 'torch.Tensor'>
        train_data["edge"][0].shape: torch.Size([1980184])
        len(train_data["edge"][1]): 1980184 (문제 풀이 기록 개수)
        type(train_data["edge"][2]): <class 'torch.Tensor'>
        train_data["edge"][0].shape: torch.Size([1980184])
        """
        if CFG.edge_dropout:
            user_problem_edge_index, user_test_edge_index, user_tag_edge_index, edge_mask = dropout_edge(user_problem_train_data["edge"], user_test_train_data["edge"], user_tag_train_data["edge"], p=CFG.edge_dropout_rate)
            train_pred = model(user_problem_edge_index, user_test_edge_index, user_tag_edge_index)
            label = torch.masked_select(user_problem_train_data["label"], edge_mask)
            loss = model.link_pred_loss(train_pred, torch.masked_select(user_problem_train_data["label"], edge_mask))
        else:
            label = user_problem_train_data["label"]
            train_pred = model(user_problem_train_data["edge"], user_test_train_data["edge"], user_tag_train_data["edge"])
            loss = model.link_pred_loss(train_pred, label)
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            prob = model.predict_link(user_problem_valid_data["edge"], user_test_valid_data["edge"], user_tag_valid_data["edge"], prob=True)
            valid_prob = copy.deepcopy(prob)
            prob = prob.detach().cpu().numpy()
            acc = accuracy_score(user_problem_valid_data["label"], prob > 0.5)
            auc = roc_auc_score(user_problem_valid_data["label"], prob)
            precision = precision_score(user_problem_valid_data["label"], prob > 0.5)
            recall = recall_score(user_problem_valid_data["label"], prob > 0.5)
            f1 = f1_score(user_problem_valid_data["label"], prob > 0.5)
            if e == 1 or e % 100 == 0:
                logger.info(
                    f" * In epoch {(e):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}, Precision={precision:.03f}, Recall={recall:.03f}, F1={f1:.03f}"
                )
                pred = inference(
                    model,
                    
                    user_problem_test_data,
                    user_test_test_data,
                    user_tag_test_data,
                    
                    logger=logger.getChild("infer"))
                pred = pred.detach().cpu().numpy()
                pred_path = os.path.join(CFG.output_dir, f"tmp_pred_{CFG.n_epoch}_{''.join(str(CFG.feature_num_layers))}_{feature_aggregation_method_dict[CFG.feature_aggregation_method]}.csv")
                pd.DataFrame({"prediction": pred}).to_csv(
                    pred_path, index_label="id"
                )
                t_auc, t_acc = check_accuracy(PRED_PATH = pred_path)
                
                logger.info(f"Test performance: auc : {t_auc} \t\t acc : {t_acc}")
                logger.info(f"n_epoch: {CFG.n_epoch}, \t feature_num_layers: {CFG.feature_num_layers}, \t learning_rate: {CFG.learning_rate}, \t feature_aggregation_method: {feature_aggregation_method_dict[CFG.feature_aggregation_method]}")
                if CFG.edge_dropout:
                    logger.info(f"edge_dropout_rate: {CFG.edge_dropout_rate}")
                else:
                    logger.info(f"no edge_dropout")
                
            if use_wandb:
                import wandb
                valid_loss = model.link_pred_loss(valid_prob, valid_label)
                label = label.to("cpu").detach().numpy()
                train_pred = train_pred.detach().cpu().numpy()
                train_auc = roc_auc_score(label, train_pred)
                
                wandb.log(
                    dict(
                        train_loss=loss,
                        valid_loss=valid_loss,
                        valid_acc=acc,
                        valid_auc=auc,
                        train_auc=train_auc,
                    )
                )
            if model_type == 0 or model_type == 1:
                if auc > best_auc and acc > best_acc:
                    torch.save(
                        {"model":  model.state_dict(), "epoch": e},
                        f"./weight/n_epoch_{CFG.n_epoch}_feature_aggregation_method_{feature_aggregation_method_dict[CFG.feature_aggregation_method]}_feature_num_layers_{''.join(str(CFG.feature_num_layers))}_model.pt"
                    )

def inference(model, user_problem_test_data, user_test_test_data, user_tag_test_data, logger=None):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(user_problem_test_data["edge"], user_test_test_data["edge"], user_tag_test_data["edge"], prob=True)
        return pred
