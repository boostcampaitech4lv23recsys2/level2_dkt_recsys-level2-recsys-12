import time

# ====================================================
# CFG
# learning_rate=0.005,
# eval_epoch=1,
# top_k=10,
# save_model=False,
# epochs=30,
# save_epoch=1

# ====================================================
class CFG:
    use_cuda_if_available = True
    user_wandb = False
    wandb_kwargs = dict(project="DKT_LGCN", entity="ai-tech-4-recsys-12")

    # data
    basepath = "/opt/ml/input/data/"
    loader_verbose = True

    # dump
    timestr = time.strftime("%m.%d-%H:%M:%S")  # 날짜

    output_dir = "./output/"
    pred_file = "submission_{}.csv".format(timestr)

    # build
    embedding_dim = 64  # int
    num_layers = 3  # int
    alpha = None  # Optional[Union[float, Tensor]]
    build_kwargs = {
        # "topk": 10,
    }  # other arguments
    weight = "./weight/best_model.pt"
    feature_weight = [0.6, 0.2, 0.2]

    # train
    n_epoch = 1500
    learning_rate = 0.001
    weight_basepath = "./weight"
    patience = 20


logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}
