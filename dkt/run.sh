python train.py --model lstmattn --n_epochs 50 --patience 10 --lr 0.001 --kfold 5 --batch_size 32 --max_seq_len 50 --n_layers 4
python inference.py --model lstmattn --n_epochs 50 --patience 10 --lr 0.001 --kfold 5 --batch_size 32 --max_seq_len 50 --n_layers 4
python ensemble.py --ENSEMBLE_FILES lstm_0,lstm_1,lstm_2,lstm_3,lstm_4 --ENSEMBLE_STRATEGY WEIGHTED

# ,lstmattn_5,lstmattn_6,lstmattn_7,lstmattn_8,lstmattn_9 