import copy
import gc
import math
import os
import warnings
from collections import OrderedDict
from datetime import datetime

warnings.filterwarnings(action="ignore")

import numpy as np
import torch
import wandb
from sklearn.model_selection import KFold

from .criterion import get_criterion
from .dataloader import data_augmentation, get_loaders, get_loaders_kfold
from .metric import get_metric
from .model import LSTM, LSTMATTN, Bert, LastQuery
from .optimizer import get_optimizer
from .scheduler import get_scheduler


def run(args, train_data, valid_data, model, gradient=False):
    # 캐시 메모리 비우기 및 가비지 컬렉터 가동!
    torch.cuda.empty_cache()
    gc.collect()
    augmented_train_data = data_augmentation(train_data, args)
    if len(augmented_train_data) != len(train_data):
        print(
            f"Data Augmentation applied. Train data {len(train_data)} -> {len(augmented_train_data)}\n"
        )

    train_data = augmented_train_data
    train_loader, valid_loader = get_loaders(args, train_data, valid_data)

    # only when using warmup scheduler
    args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        args.n_epochs
    )
    args.warmup_steps = args.total_steps // 10

    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)

    best_auc = -1
    early_stopping_counter = 0
    for epoch in range(args.n_epochs):

        print(f"Start Training: Epoch {epoch + 1}")

        ### TRAIN
        train_auc, train_acc, train_loss = train(
            train_loader, model, optimizer, scheduler, args
        )

        ### VALID
        auc, acc = validate(valid_loader, model, args)

        ### TODO: model save or early stopping
        if args.run_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_auc": train_auc,
                    "train_acc": train_acc,
                    "valid_auc": auc,
                    "valid_acc": acc,
                }
            )
        if auc > best_auc:
            best_auc = auc
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, "module") else model
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model_to_save.state_dict(),
                },
                args.model_dir,
                "model.pt",
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(
                    f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
                )
                break

        # scheduler
        if args.scheduler == "plateau":
            scheduler.step(best_auc)


def run_kfold(args, train_data, preprocess, model):
    # 캐시 메모리 비우기 및 가비지 컬렉터 가동!
    torch.cuda.empty_cache()
    gc.collect()
    augmented_train_data = data_augmentation(train_data, args)
    if len(augmented_train_data) != len(train_data):
        print(
            f"Data Augmentation applied. Train data {len(train_data)} -> {len(augmented_train_data)}\n"
        )

    train_data = augmented_train_data
    kfold = KFold(n_splits=args.kfold, random_state=args.seed, shuffle=True)

    for fold, (train_idx, valid_idx) in enumerate(kfold.split(train_data)):

        inner_model = copy.deepcopy(model)

        train_data_fold, valid_data_fold = preprocess.split_data(train_data)
        # only when using warmup scheduler
        # args.total_steps = int(math.ceil(len(train_loader.dataset) / args.batch_size)) * (
        #     args.n_epochs
        # )
        # args.warmup_steps = args.total_steps // 10

        # reset wandb for every fold
        if args.run_wandb:
            wandb.init(
                project="DKT_LSTMATTN_KFOLD",
                config=vars(args),
                entity="ai-tech-4-recsys-12",
            )
            wandb.run.name = f"Fold:{fold}_BatchSize:{args.batch_size}_LR:{args.lr}_Patience:{args.patience}"

        # let users know which fold current fold is
        print(f"#################### Fold number {fold} ####################\n")

        optimizer = get_optimizer(inner_model, args)
        scheduler = get_scheduler(optimizer, args)

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        valid_subsampler = torch.utils.data.SubsetRandomSampler(valid_idx)

        train_loader = get_loaders_kfold(
            args,
            train_data,
            train_subsampler,
        )

        valid_loader = get_loaders_kfold(
            args,
            train_data,
            valid_subsampler,
        )

        best_auc = -1
        early_stopping_counter = 0
        for epoch in range(args.n_epochs):

            print(f"Start Training: Epoch {epoch + 1}")

            ### TRAIN
            train_auc, train_acc, train_loss = train(
                train_loader, inner_model, optimizer, scheduler, args
            )

            ### VALID
            auc, acc = validate(valid_loader, inner_model, args)

            ### TODO: model save or early stopping
            if args.run_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        "train_loss_epoch": train_loss,
                        "train_auc_epoch": train_auc,
                        "train_acc_epoch": train_acc,
                        "valid_auc_epoch": auc,
                        "valid_acc_epoch": acc,
                    }
                )
            if auc > best_auc:
                best_auc = auc
                # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
                model_to_save = (
                    inner_model.module
                    if hasattr(inner_model, "module")
                    else inner_model
                )
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model_to_save.state_dict(),
                    },
                    args.model_dir,
                    f"{args.model}_fold_{fold}.pt",
                )
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    print(
                        f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
                    )
                    break

            # scheduler
            if args.scheduler == "plateau":
                scheduler.step(best_auc)

        # finish wandb for every fold
        if args.run_wandb:
            wandb.finish()


def train(train_loader, model, optimizer, scheduler, args, gradient=False):
    model.train()

    total_preds = []
    total_targets = []
    losses = []
    for step, batch in enumerate(train_loader):
        # input[3]: correct, input[-1]: interaction, input[-2]: mask
        if args.model == "lastquery":
            input = list(map(lambda t: t.to(args.device), process_batch_lq(batch)))
        else:
            input = list(map(lambda t: t.to(args.device), process_batch(batch)))
        preds = model(input)
        # targets = input[3]  # correct
        targets = input[0]  # correct is moved to index 0

        loss = compute_loss(preds, targets)
        update_params(loss, model, optimizer, scheduler, args)

        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss: {str(loss.item())}")

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())
        losses.append(loss)

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)
    loss_avg = sum(losses) / len(losses)
    print(f"TRAIN AUC : {auc} ACC : {acc}")
    return auc, acc, loss_avg


def validate(valid_loader, model, args):
    model.eval()

    total_preds = []
    total_targets = []
    for step, batch in enumerate(valid_loader):
        # input[3]: correct, input[-1]: interaction, input[-2]: mask
        if args.model == "lastquery":
            input = list(map(lambda t: t.to(args.device), process_batch_lq(batch)))
        else:
            input = list(map(lambda t: t.to(args.device), process_batch(batch)))

        preds = model(input)
        # targets = input[3]  # correct
        targets = input[0]  # correct is moved to index 0

        # predictions
        preds = preds[:, -1]
        targets = targets[:, -1]

        total_preds.append(preds.detach())
        total_targets.append(targets.detach())

    total_preds = torch.concat(total_preds).cpu().numpy()
    total_targets = torch.concat(total_targets).cpu().numpy()

    # Train AUC / ACC
    auc, acc = get_metric(total_targets, total_preds)

    print(f"VALID AUC : {auc} ACC : {acc}\n")

    return auc, acc


def inference(args, test_data, model):

    model.eval()
    _, test_loader = get_loaders(args, None, test_data)

    total_preds = []

    for step, batch in enumerate(test_loader):
        if args.model == "lastquery":
            input = list(map(lambda t: t.to(args.device), process_batch_lq(batch)))
        else:
            input = list(map(lambda t: t.to(args.device), process_batch(batch)))

        preds = model(input)

        # predictions
        preds = preds[:, -1]
        preds = torch.nn.Sigmoid()(preds)
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)

    from datetime import datetime

    time = datetime.now().strftime("%m.%d_%H%M%S")
    write_path = os.path.join(
        args.output_dir,
        f"{time}_{args.model}_{args.n_epochs}_{args.lr}_{args.patience}_{args.batch_size}.csv",
    )
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))


def inference_kfold(args, test_data, model, fold):
    model.eval()
    _, test_loader = get_loaders(args, None, test_data)

    total_preds = []

    for step, batch in enumerate(test_loader):
        if args.model == "lastquery":
            input = list(map(lambda t: t.to(args.device), process_batch_lq(batch)))
        else:
            input = list(map(lambda t: t.to(args.device), process_batch(batch)))

        preds = model(input)

        # predictions
        preds = preds[:, -1]
        preds = torch.nn.Sigmoid()(preds)
        preds = preds.cpu().detach().numpy()
        total_preds += list(preds)

    write_path = os.path.join(args.output_dir, f"{args.model}_{fold}.csv")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(write_path, "w", encoding="utf8") as w:
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write("{},{}\n".format(id, p))


def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == "lstm":
        model = LSTM(args)
    if args.model == "lstmattn":
        model = LSTMATTN(args)
    if args.model == "bert":
        model = Bert(args)
    if args.model == "lastquery":
        model = LastQuery(args)

    return model


def get_gradient(model):
    gradient = []

    for name, param in model.named_parameters():
        grad = param.grad
        if grad != None:
            gradient.append(grad.cpu().numpy().astype(np.float16))
            # gradient.append(grad.clone().detach())
        else:
            gradient.append(None)

    return gradient


# 배치 전처리
def process_batch(batch):
    # batch[3]: correct, batch[-1]: mask

    # test, question, tag, correct, mask = batch
    (
        correct,
        test,
        question,
        tag,
        
        # features for lstmattn model
        first3,
        hour_answerCode_Level,
        elapsedTime,
        dayofweek_answerCode_median,
        KnowledgeTag_answerCode_mean,
        hour_answerCode_mean,
        KnowledgeTag_elapsedTime_median,
        userID_answerCode_mean,
        assessmentItemID_elo_pred,
        mask,
    ) = batch

    # correct, test, question, tag, mask = batch # batch = [correct, ...features..., mask]

    # change to float
    mask = mask.float()
    correct = correct.float()
    # mask = batch[-1].float()
    # correct = batch[0].float()

    # interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    interaction = correct + 1  # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)

    #  test_id, question_id, tag
    test = ((test + 1) * mask).int()
    question = ((question + 1) * mask).int()
    tag = ((tag + 1) * mask).int()
    
    # features for lstmattn model
    first3 = ((first3 + 1) * mask).int()
    hour_answerCode_Level = ((hour_answerCode_Level + 1) * mask).int()
    elapsedTime = ((elapsedTime + 1) * mask).int()
    dayofweek_answerCode_median = ((dayofweek_answerCode_median + 1) * mask).int()
    KnowledgeTag_answerCode_mean = ((KnowledgeTag_answerCode_mean + 1) * mask).int()
    hour_answerCode_mean = ((hour_answerCode_mean + 1) * mask).int()
    KnowledgeTag_elapsedTime_median = (
        (KnowledgeTag_elapsedTime_median + 1) * mask
    ).int()
    userID_answerCode_mean = ((userID_answerCode_mean + 1) * mask).int()
    assessmentItemID_elo_pred = ((assessmentItemID_elo_pred + 1) * mask).int()
    
    # features = [((feat + 1) * mask).int() for feat in batch[1 : len(batch) - 1]]

    # return (test, question, tag, correct, mask, interaction)
    return (
        correct,
        test,
        question,
        tag,
        
        # features for lstmattn model
        first3,
        hour_answerCode_Level,
        elapsedTime,
        dayofweek_answerCode_median,
        KnowledgeTag_answerCode_mean,
        hour_answerCode_mean,
        KnowledgeTag_elapsedTime_median,
        userID_answerCode_mean,
        assessmentItemID_elo_pred,
        mask,
        interaction,
    )

    # return (correct, *features, mask, interaction)


# 배치 전처리 for lastquery
def process_batch_lq(batch):
    # batch[3]: correct, batch[-1]: mask

    # test, question, tag, correct, mask = batch
    correct, test, question, tag, elapsed, mask = batch
    # correct, test, question, tag, mask = batch # batch = [correct, ...features..., mask]

    # change to float
    mask = mask.float()
    correct = correct.float()
    # mask = batch[-1].float()
    # correct = batch[0].float()

    # interaction을 임시적으로 correct를 한칸 우측으로 이동한 것으로 사용
    interaction = correct + 1  # 패딩을 위해 correct값에 1을 더해준다.
    interaction = interaction.roll(shifts=1, dims=1)
    interaction_mask = mask.roll(shifts=1, dims=1)
    interaction_mask[:, 0] = 0
    interaction = (interaction * interaction_mask).to(torch.int64)

    # categorical
    #  test_id, question_id, tag
    test = ((test + 1) * mask).int()
    question = ((question + 1) * mask).int()
    tag = ((tag + 1) * mask).int()
    # features = [((feat + 1) * mask).int() for feat in batch[1 : len(batch) - 1]]

    # continuous
    elapsed = (elapsed * mask).float()
    return (correct, test, question, tag, elapsed, mask, interaction)


# loss계산하고 parameter update!
def compute_loss(preds, targets):
    """
    Args :
        preds   : (batch_size, max_seq_len)
        targets : (batch_size, max_seq_len)

    """
    loss = get_criterion(preds, targets)

    # 마지막 시퀀드에 대한 값만 loss 계산
    loss = loss[:, -1]
    loss = torch.mean(loss)
    return loss


def update_params(loss, model, optimizer, scheduler, args):
    loss.backward()

    if args.clip_grad:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    if args.scheduler == "linear_warmup":
        scheduler.step()
    optimizer.step()
    optimizer.zero_grad()


def save_checkpoint(state, model_dir, model_filename):
    print("saving model ...")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save(state, os.path.join(model_dir, model_filename))


def load_model(args):

    model_path = os.path.join(args.model_dir, args.model_name)
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model


def load_model_kfold(args, fold_idx: int):

    model_path = os.path.join(args.model_dir, f"{args.model}_fold_{fold_idx}.pt")
    print("Loading Model from:", model_path)
    load_state = torch.load(model_path)
    model = get_model(args)

    # load model state
    model.load_state_dict(load_state["state_dict"], strict=True)

    print("Loading Model from:", model_path, "...Finished.")
    return model
