from dataloader import AmazonDataset
import models
from models import DistMulti, TransE, SparseTransE
from training import TrainIterater
from evaluate import Evaluater

import optuna
import numpy as np
import pickle
import time
import sys

import torch
from importlib import reload

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def time_since(runtime):
    mi = int(runtime / 60)
    sec = runtime - mi * 60
    return (mi, sec)

def objective(trial):
    start = time.time()
    import gc
    gc.collect()

    data_dir = [data_path + '/valid1', data_path + '/valid2']
    score_sum = 0

    embed_model = {'TransE': TransE, 'SparseTransE': SparseTransE}
    # hyper para
    embedding_dim = trial.suggest_discrete_uniform('embedding_dim', 16, 128, 16)

    if model_name == 'SparseTransE':
        alpha = trial.suggest_loguniform('alpha', 1e-6, 1e-2) #SparseTransEの時だけ

    batch_size = trial.suggest_int('batch_size', 128, 512, 128)
    lr= trial.suggest_loguniform('lr', 1e-4, 1e-2)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-2)
    #warmup = trial.suggest_int('warmup', 100, 500)
    #warmup = trial.suggest_int('warmup', 10, 100)
    warmup = 350
    #lr_decay_every = trial.suggest_int('lr_decay_every', 1, 10)
    lr_decay_every = 2
    lr_decay_rate = trial.suggest_uniform('lr_decay_rate', 0.5, 1)


    for dir_path in data_dir:
    # データ読み込み
        dataset = AmazonDataset(dir_path, model_name=model_name)
        relation_size = len(set(list(dataset.triplet_df['relation'].values)))
        entity_size = len(dataset.entity_list)
        #model = TransE(int(embedding_dim), relation_size, entity_size).to(device)
        model = embed_model[model_name](int(embedding_dim), relation_size, entity_size).to(device)
        iterater = TrainIterater(batch_size=int(batch_size), data_dir=dir_path, model_name=model_name)
        score =iterater.iterate_epoch(model, lr=lr, epoch=3000, weight_decay=weight_decay, warmup=warmup,
                            lr_decay_rate=lr_decay_rate, lr_decay_every=lr_decay_every, eval_every=1e+5, 
                            early_stop=True)

        score_sum += score 
    
    torch.cuda.empty_cache()

    mi, sec = time_since(time.time() - start)
    print('{}m{}sec'.format(mi, sec))
    
    return -1 * score_sum / 2


if __name__ == '__main__':
    args = sys.argv

    data_path = './data/'
    model_name = args[1]
    
    study = optuna.create_study()
    study.optimize(objective, n_trials=30)
    df = study.trials_dataframe() # pandasのDataFrame形式

    df.to_csv('result/hyparams_' + model_name + '.csv')
    with open('result/best_param_' + model_name + '.pickle', 'wb') as f:
        pickle.dump(study.best_params, f)