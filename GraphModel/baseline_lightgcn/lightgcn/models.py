import os

import numpy as np
import torch
from config import CFG
from custom_scheduler import CosineAnnealingWarmUpRestarts as wca
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from torch_geometric.nn.models import LightGCN

from typing import Optional, Tuple

from torch import Tensor

from torch_geometric.deprecation import deprecated
from torch_geometric.typing import OptTensor

def dropout_edge(edge_index: Tensor, p: float = 0.5,
                 force_undirected: bool = False,
                 training: bool = True) -> Tuple[Tensor, Tensor]:

    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                         f'(got {p}')

    if not training or p == 0.0:
        edge_mask = edge_index.new_ones(edge_index.size(1), dtype=torch.bool)
        return edge_index, edge_mask
    
    row, col = edge_index

    edge_mask = torch.rand(row.size(0), device=edge_index.device) >= p

    if force_undirected:
        edge_mask[row > col] = False

    edge_index = edge_index[:, edge_mask]

    if force_undirected:
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        edge_mask = edge_mask.nonzero().repeat((2, 1)).squeeze()

    return edge_index, edge_mask


def build(n_node, weight=None, logger=None, **kwargs):
    model = LightGCN(n_node, **kwargs)
    if weight:
        if not os.path.isfile(weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(weight)["model"]
        model.load_state_dict(state)
        return model
    else:
        logger.info("No load model")
        return model


def train(
    model,
    train_data,
    valid_data=None,
    n_epoch=50,
    learning_rate=0.001,
    use_wandb=False,
    weight=None,
    logger=None,
):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # wca(optimizer, T_0=150, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, eta_min=0.001, last_epoch=- 1, verbose=False)
    scheduler = wca(optimizer, T_0=50, T_mult=2, eta_max=0.005, T_up=5, gamma=0.5)

    if not os.path.exists(weight):
        os.makedirs(weight)

    if valid_data is None:
        eids = np.arange(len(train_data["label"]))
        eids = np.random.permutation(eids)[:1000]
        edge, label = train_data["edge"], train_data["label"]
        label = label.to("cpu").detach().numpy()
        valid_data = dict(edge=edge[:, eids], label=label[eids])

    valid_data["label"] = (
        valid_data["label"].to("cpu").detach().numpy()
    )  # convert for score calc

    logger.info(f"Training Started : n_epoch={n_epoch}")
    best_auc, best_epoch = 0, -1
    patience_check = 0
    patience_limit = CFG.patience
    for e in range(n_epoch):
        
        # forward
        if CFG.edge_dropout:
            masked_train_data, edge_mask = dropout_edge(train_data["edge"], p=CFG.edge_dropout_rate)
            # pred = model(train_data["edge"])
            # loss = model.link_pred_loss(pred, train_data["label"])
            pred = model(masked_train_data)
            loss = model.link_pred_loss(pred, torch.masked_select(train_data["label"], edge_mask))
        else:
            pred = model(train_data["edge"])
            loss = model.link_pred_loss(pred, train_data["label"])
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            prob = model.predict_link(valid_data["edge"], prob=True)
            prob = prob.detach().cpu().numpy()
            acc = accuracy_score(valid_data["label"], prob > 0.5)
            auc = roc_auc_score(valid_data["label"], prob)
            precision = precision_score(valid_data["label"], prob > 0.5)
            recall = recall_score(valid_data["label"], prob > 0.5)
            f1 = f1_score(valid_data["label"], prob > 0.5)
            if e % 500 == 0:
                logger.info(
                    f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}, Precision={precision:.03f}, Recall={recall:.03f}, F1={f1:.03f}"
                )
            if use_wandb:
                import wandb

                wandb.log(
                    dict(
                        loss=loss,
                        acc=acc,
                        auc=auc,
                        precision=precision,
                        recall=recall,
                        f1=f1,
                    )
                )

        if weight:
            if auc > best_auc:
                patience_check = 0
                if e % 500 == 0:
                    logger.info(
                        f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}, Precision={precision:.03f}, Recall={recall:.03f}, F1={f1:.03f}"
                )
                best_auc, best_epoch, best_prob = auc, e, prob
                torch.save(
                    {"model": model.state_dict(), "epoch": e + 1},
                    os.path.join(weight, f"best_model.pt"),
                )
            # else:
            #     print(f"auc : {auc}", f"best_auc : {best_auc}")
            #     patience_check += 1
            #     print(f"patience_check : {patience_check}")
            #     # early stopping
            #     """if patience_check >= patience_limit:
            #         break"""
    torch.save(
        {"model": model.state_dict(), "epoch": e + 1},
        os.path.join(weight, f"last_model.pt"),
    )
    # logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")


def inference(model, data, logger=None):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(data["edge"], prob=True)
        return pred
