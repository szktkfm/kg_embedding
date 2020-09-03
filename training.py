import pickle
import time
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dataloader import AmazonDataset
from evaluate import Evaluater
from earlystop import EarlyStop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainIterater():


    def __init__(self, batch_size, data_dir, model_name='DistMulti'):
        #self.dataset = dataloader.AmazonDataset('./data')
        #self.dataset = AmazonDataset('./data', model_name=model_name)
        self.data_dir = data_dir
        self.dataset = AmazonDataset(self.data_dir, model_name=model_name)
        self.batch_size = batch_size
        self.model_name = model_name


    def train(self, batch, loss_func, optimizer, model):
        optimizer.zero_grad()

        if self.model_name == 'DistMulti' or self.model_name == 'Complex':
            triplet, y_train = batch
            h_entity_tensor = torch.tensor(triplet[:, 0], dtype=torch.long, device=device)
            t_entity_tensor = torch.tensor(triplet[:, 1], dtype=torch.long, device=device)
            relation_tensor = torch.tensor(triplet[:, 2], dtype=torch.long, device=device)
            y_train = torch.tensor(y_train, dtype=torch.float, device=device)

            pred = model(h_entity_tensor, t_entity_tensor, relation_tensor)
            loss = loss_func(pred, y_train)

        elif self.model_name == 'TransE':
            posi_batch, nega_batch = batch
            h = torch.tensor(posi_batch[:, 0], dtype=torch.long, device=device)
            t = torch.tensor(posi_batch[:, 1], dtype=torch.long, device=device)
            r = torch.tensor(posi_batch[:, 2], dtype=torch.long, device=device)

            n_h = torch.tensor(nega_batch[:, 0], dtype=torch.long, device=device)
            n_t = torch.tensor(nega_batch[:, 1], dtype=torch.long, device=device)
            n_r = torch.tensor(nega_batch[:, 2], dtype=torch.long, device=device)

            pred = model(h, t, r, n_h, n_t, n_r)
            loss = torch.sum(pred)

        elif self.model_name == 'SparseTransE':
            posi_batch, nega_batch, batch_user, batch_item, batch_brand = batch
            h = torch.tensor(posi_batch[:, 0], dtype=torch.long, device=device)
            t = torch.tensor(posi_batch[:, 1], dtype=torch.long, device=device)
            r = torch.tensor(posi_batch[:, 2], dtype=torch.long, device=device)

            n_h = torch.tensor(nega_batch[:, 0], dtype=torch.long, device=device)
            n_t = torch.tensor(nega_batch[:, 1], dtype=torch.long, device=device)
            n_r = torch.tensor(nega_batch[:, 2], dtype=torch.long, device=device)

            reg_user = torch.tensor(batch_user, dtype=torch.long, device=device)
            reg_item = torch.tensor(batch_item, dtype=torch.long, device=device)
            reg_brand = torch.tensor(batch_brand, dtype=torch.long, device=device)

            pred = model(h, t, r, n_h, n_t, n_r,
                         reg_user, reg_item, reg_brand)

            loss = torch.sum(pred)

        elif self.model_name == 'RegComplex':
            triplet, y_train, batch_user, batch_item, batch_brand = batch
            h_entity_tensor = torch.tensor(triplet[:, 0], dtype=torch.long, device=device)
            t_entity_tensor = torch.tensor(triplet[:, 1], dtype=torch.long, device=device)
            relation_tensor = torch.tensor(triplet[:, 2], dtype=torch.long, device=device)
            y_train = torch.tensor(y_train, dtype=torch.float, device=device)

            reg_user = torch.tensor(batch_user, dtype=torch.long, device=device)
            reg_item = torch.tensor(batch_item, dtype=torch.long, device=device)
            reg_brand = torch.tensor(batch_brand, dtype=torch.long, device=device)

            pred, reg = model(h_entity_tensor, t_entity_tensor, relation_tensor, 
                                reg_user, reg_item, reg_brand)

            loss = loss_func(pred, y_train) + reg


        loss.backward()
        optimizer.step()

        return loss


    def iterate_train(self, model, lr=0.001, weight_decay=0, print_every=2000, plot_every=50):

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        # optimizer = optim.SGD(model.parameters(), lr=lr)

        loss_func = nn.BCELoss()

        print_loss_total = 0
        plot_loss_list = []
        plot_loss_total = 0

        if self.model_name == 'DistMulti' or self.model_name == 'Complex' or self.model_name == 'RegComplex':
            train_num = len(self.dataset.triplet_df) + len(self.dataset.nega_triplet_df)
        elif self.model_name == 'TransE' or self.model_name == 'SparseTransE':
            train_num = len(self.dataset.triplet_df)

        start_time = time.time()

        for i in range(int(train_num / self.batch_size) + 1):

            batch = self.dataset.get_batch(batch_size=self.batch_size)
            loss = self.train(batch, loss_func, optimizer, model)
            print_loss_total += loss.detach()
            plot_loss_total += loss.detach()

            # print_everyごとに現在の平均のlossと、時間、dataset全体に対する進捗(%)を出力
            if (i+1) % print_every == 0:
                runtime = time.time() - start_time
                mi, sec = self.time_since(runtime)
                avg_loss = print_loss_total / print_every
                data_percent = int(i * self.batch_size / train_num * 100)
                print('train loss: {:e}    processed: {}({}%)    {}m{}sec'.format(
                    avg_loss, i*self.batch_size, data_percent, mi, sec))
                print_loss_total = 0

            # plot_everyごとplot用のlossをリストに記録しておく
            if (i+1) % plot_every == 0:
                avg_loss = plot_loss_total / plot_every
                plot_loss_list.append(avg_loss)
                plot_loss_total = 0

        return plot_loss_list


    def time_since(self, runtime):
        mi = int(runtime / 60)
        sec = int(runtime - mi * 60)
        return (mi, sec)


    def iterate_epoch(self, model, lr, epoch, weight_decay=0,
                      warmup=0, lr_decay_rate=1, lr_decay_every=10, eval_every=5, early_stop=False):
        eval_model = Evaluater(self.data_dir , model_name=self.model_name)
        es = EarlyStop(self.data_dir[0:-6] + 'early_stopping/', self.model_name, patience=6)
        plot_loss_list = []
        plot_score_list = []

        for i in range(epoch):
            plot_loss_list.extend(self.iterate_train(model, lr=lr, weight_decay=weight_decay,
                                                       print_every=10000))

            # early stop
            if early_stop:
                pre_model = es.early_stop(model)
                if pre_model:
                    print('Early Stop eposh: {}'.format(i+1))
                    return eval_model.topn_map(pre_model)

            # lrスケジューリング
            if i > warmup:
                if (i - warmup) % lr_decay_every == 0:
                    lr = lr * lr_decay_rate

            if (i+1) % eval_every == 0:
                #score = eval_model.topn_precision(model)
                #print('epoch: {}  precision: {}'.format(i, score))
                score = eval_model.topn_map(model)
                print('epoch: {}  map: {}'.format(i, score))
                plot_score_list.append(score)

        #self._plot(plot_loss_list)
        #self._plot(plot_score_list)

        #return eval_model.topn_precision(model)
        return eval_model.topn_map(model)



    def _plot(self, loss_list):
        # ここもっとちゃんと書く
        plt.plot(loss_list)
        plt.show()


