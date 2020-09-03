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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EarlyStop():

    def __init__(self, data_dir, model_name, patience):
        #self.dataset = AmazonDataset(data_dir, model_name)

        self.patience = patience
        self.model_name = model_name

        self.user_item_train_df = pd.read_csv(data_dir + 'triplet.csv')
        self.user_item_nega_df = pd.read_csv(data_dir + 'nega_triplet.csv')

        self.user_list = []
        self.item_list = []
        self.brand_list = []
        self.entity_list = []
        with open(data_dir + 'user_list.txt', 'r') as f:
            for l in f:
                self.user_list.append(l.replace('\n', ''))

        with open(data_dir + 'item_list.txt', 'r') as f:
            for l in f:
                self.item_list.append(l.replace('\n', ''))
                
        with open(data_dir + 'brand_list.txt', 'r') as f:
            for l in f:
                self.brand_list.append(l.replace('\n', ''))
                
        with open(data_dir + 'entity_list.txt', 'r') as f:
            for l in f:
                self.entity_list.append(l.replace('\n', ''))


        y_test = [1 for i in range(len(self.user_item_train_df))] \
                   + [0 for i in range(len(self.user_item_nega_df))]
        self.y_test = np.array(y_test)

        self.loss_list = []
        self.model_list = []


    def negative_sampling(self):
        implicit_feed = [list(r) for r in self.dataset.user_item_test_df.values]
        user_idx = [self.dataset.entity_list.index(u) for u in self.dataset.user_list]
        item_idx = [self.dataset.entity_list.index(i) for i in self.dataset.item_list]

        user_item_test_nega = []
        count = 0
        while count < len(self.dataset.user_item_test_df):
            uidx = np.random.randint(len(self.dataset.user_list))
            iidx = np.random.randint(len(self.dataset.item_list))
            user = user_idx[uidx]
            item = item_idx[iidx]
            ### relationはすべてuser->(buy)itemの0
            if [user, item, 0] in implicit_feed:
                continue
            if [user, item, 0] in user_item_test_nega:
                continue

            user_item_test_nega.append([user, item, 0])
            count += 1

        user_item_test_nega_df = pd.DataFrame(user_item_test_nega, columns=['reviewerID', 'asin', 'relation'])

        return user_item_test_nega_df


    def early_stop(self, model):
        loss = self.iterate_valid_loss(model, batch_size=1024)
        self.loss_list.append(loss)
        # model copy
        self.model_list.append(copy.deepcopy(model))

        flag = 0
        for i in range(len(self.loss_list) - 1):
            if self.loss_list[0] > self.loss_list[i + 1]:
                flag = 1

        if flag == 0 and len(self.loss_list) > self.patience:
            return self.model_list[0]

        if len(self.loss_list) > self.patience:
            self.loss_list.pop(0)
            self.model_list.pop(0)

        return False


    def get_batch(self, batch_size):
        if self.model_name == 'DistMulti' or self.model_name == 'Complex':
            train_num = len(self.user_item_train_df) + len(self.user_item_nega_df)
            batch_idx = np.random.permutation(train_num)[:batch_size]
            # posi_tripletとnega_tripletを連結
            batch = pd.concat([self.user_item_train_df, self.user_item_nega_df]).values[batch_idx]
            batch_y_test = self.y_test[batch_idx]
        
            return batch, batch_y_test

        elif self.model_name == 'TransE':
            batch_idx = np.random.permutation(len(self.user_item_train_df))[:batch_size]
            posi_batch = self.user_item_train_df.values[batch_idx]
            nega_batch = self.user_item_nega_df.values[batch_idx]
            
            return posi_batch, nega_batch
            
        elif self.model_name == 'SparseTransE':
            batch_idx = np.random.permutation(len(self.user_item_train_df))[:batch_size]
            posi_batch = self.user_item_train_df.values[batch_idx]
            nega_batch = self.user_item_nega_df.values[batch_idx]

            # reguralizationのためのbatch
            # entity_typeの数だけ
            batch_entity_size = int(len(self.entity_list) / (len(self.user_item_train_df) / batch_size))
            reg_batch_idx = np.random.permutation(len(self.entity_list))[:batch_entity_size]

            batch_item = reg_batch_idx[reg_batch_idx < len(self.item_list)]

            batch_user = reg_batch_idx[reg_batch_idx >= len(self.item_list)]
            batch_user = batch_user[batch_user < len(self.user_list)]

            batch_brand = reg_batch_idx[reg_batch_idx >= len(self.user_list)]
            batch_brand = batch_brand[batch_brand < len(self.brand_list)]

            return posi_batch, nega_batch, batch_user, batch_item, batch_brand

        
    def iterate_valid_loss(self, model, batch_size=1024):
        loss_func = nn.BCELoss()
        loss_total = 0

        if self.model_name == 'DistMulti' or self.model_name == 'Complex':
            train_num = len(self.user_item_train_df) + len(self.user_item_nega_df)
        elif self.model_name == 'TransE' or self.model_name == 'SparseTransE':
            train_num = len(self.user_item_train_df)

        for i in range(int(train_num / batch_size) + 1):
            batch = self.get_batch(batch_size=batch_size)
            #print(batch)
            loss = self.valid_loss(batch, loss_func, model)
            #print(loss)
            loss_total += loss.detach()

        
        return loss_total / len(self.user_item_train_df)


    def valid_loss(self, batch, loss_func, model):

        with torch.no_grad(): 
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
            
        return loss


    def valid_metric(self, model):
        return 0

if __name__ == '__main__':
    import models

    dataset = AmazonDataset('../data_beauty_2core_es/valid1/', 'TransE')
    relation_size = len(set(list(dataset.triplet_df['relation'].values)))
    entity_size = len(dataset.entity_list)
    model = models.TransE(10, relation_size, entity_size).to(device)

    es = EarlyStop('../data_beauty_2core_es/early_stopping/', 'TransE', 10)
    es.early_stop(model)