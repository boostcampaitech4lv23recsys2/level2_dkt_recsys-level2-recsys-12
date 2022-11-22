import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from torch_geometric.nn.models import LightGCN

from config import CFG

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
    n_epoch=100,
    learning_rate=0.01,
    use_wandb=False,
    weight=None,
    logger=None,
):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if not os.path.exists(weight):
        os.makedirs(weight)

    if valid_data is None:
        eids = np.arange(len(train_data["label"]))
        eids = np.random.permutation(eids)[:1000]
        edge, label = train_data["edge"], train_data["label"]
        label = label.to("cpu").detach().numpy()
        valid_data = dict(edge=edge[:, eids], label=label[eids])
        
    valid_data['label'] = valid_data['label'].to('cpu').detach().numpy() # convert for score calc
        
    logger.info(f"Training Started : n_epoch={n_epoch}")
    best_auc, best_epoch = 0, -1
    patience_check = 0
    patience_limit = CFG.patience
    for e in range(n_epoch):
        # forward
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
            logger.info(
                f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}, Precision={precision:.03f}, Recall={recall:.03f}, F1={f1:.03f}"
            )
            if use_wandb:
                import wandb

                wandb.log(dict(loss=loss, acc=acc, auc=auc, precision=precision, recall=recall, f1=f1))

        if weight:
            if auc > best_auc:
                patience_check = 0
                logger.info(
                    f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}, Precision={precision:.03f}, Recall={recall:.03f}, F1={f1:.03f}, Best AUC"
                )
                best_auc, best_epoch, best_prob = auc, e, prob
                torch.save(
                    {"model": model.state_dict(), "epoch": e + 1},
                    os.path.join(weight, f"best_model.pt"),
                )
            else:
                print(f"auc : {auc}", f"best_auc : {best_auc}")
                patience_check += 1
                print(f"patience_check : {patience_check}")
                '''if patience_check >= patience_limit:
                    break'''
    torch.save(
        {"model": model.state_dict(), "epoch": e + 1},
        os.path.join(weight, f"last_model.pt"),
    )
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")


def inference(model, data, logger=None):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(data["edge"], prob=True)
        return pred

