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


# ハイパラ
# 
# embed_dim
# batch_size
# weight_decay, lr, warmup, lr_decay_every, lr_decay_rate
# kg embed model


def load_params():
    return pickle.load(open('result/best_param_' + model_name + '.pickle', 'rb'))

def time_since(runtime):
    mi = int(runtime / 60)
    sec = runtime - mi * 60
    return (mi, sec)

if __name__ == '__main__':

    args = sys.argv

    model_name = args[1]

    params = load_params()
    print(params)

    import gc
    gc.collect()

    # dataload
    data_dir = '../' + data_path + '/test/'
    dataset = AmazonDataset(data_dir, model_name=model_name)
    
    relation_size = len(set(list(dataset.triplet_df['relation'].values)))
    entity_size = len(dataset.entity_list)
    embedding_dim = params['embedding_dim']
    alpha = params['alpha']
    model = SparseTransE(int(embedding_dim), relation_size, entity_size, alpha=alpha).to(device)
    
    batch_size = params['batch_size']
    iterater = TrainIterater(batch_size=int(batch_size), data_dir=data_dir, model_name=model_name)
    
    lr = params['lr']
    weight_decay = params['weight_decay']
    
    warmup = 350
    lr_decay_every = 2
    lr_decay_rate = params['lr_decay_rate']
    
    score =iterater.iterate_epoch(model, lr=lr, epoch=3000, weight_decay=weight_decay, warmup=warmup,
                           lr_decay_rate=lr_decay_rate, lr_decay_every=lr_decay_every, eval_every=1e+5)
    
    torch.cuda.empty_cache()

    np.savetxt('result/score_' + model_name + '.txt', np.array([score]))

