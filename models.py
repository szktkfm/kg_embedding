import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DistMulti(nn.Module):

    def __init__(self, embedding_dim, relation_size, entity_size):
        super(DistMulti, self).__init__()
        self.embedding_dim = embedding_dim

        self.entity_embed = nn.Embedding(entity_size, embedding_dim)
        self.relation_embed = nn.Embedding(relation_size, embedding_dim)
        
        
    def forward(self, head, tail, relation):
        head_embed = self.entity_embed(head)
        tail_embed = self.entity_embed(tail)
        relation_embed = self.relation_embed(relation)
        
        score = torch.sum(head_embed * tail_embed * relation_embed, axis=1)
        score = torch.sigmoid(score)
        
        return score
    
    def predict(self, head, tail, relation):
        head_embed = self.entity_embed(head)
        tail_embed = self.entity_embed(tail)
        relation_embed = self.relation_embed(relation)
        
        score = torch.sum(head_embed * tail_embed * relation_embed, axis=1)
        score = torch.sigmoid(score)
        
        return score

class TransE(nn.Module):

    def __init__(self, embedding_dim, relation_size, entity_size, gamma=1, alpha=None):
        super(TransE, self).__init__()
        self.embedding_dim = embedding_dim

        self.entity_embed = nn.Embedding(entity_size, embedding_dim)
        self.relation_embed = nn.Embedding(relation_size, embedding_dim)
        # model para init(normalize)

        # margin para
        self.gamma = gamma
        
        
    def forward(self, head, tail, relation, n_head, n_tail, n_relation):

        h = self.entity_embed(head)
        t = self.entity_embed(tail)
        r = self.relation_embed(relation)
        n_h = self.entity_embed(n_head)
        n_t = self.entity_embed(n_tail)
        n_r = self.relation_embed(n_relation)

        batch_size = h.shape[0]
        # normalize
        h = h / torch.norm(h, dim=1).view(batch_size, -1)
        t = t / torch.norm(t, dim=1).view(batch_size, -1)
        r = r / torch.norm(r, dim=1).view(batch_size, -1)
        n_h = n_h / torch.norm(n_h, dim=1).view(batch_size, -1)
        n_t = n_t / torch.norm(n_t, dim=1).view(batch_size, -1)
        n_r = n_r / torch.norm(n_r, dim=1).view(batch_size, -1)

        score = self.gamma + torch.norm((h + r - t), dim=1) - torch.norm((n_h + n_r - n_t), dim=1)
        
        return score
    
    def predict(self, head, tail, relation):
        h = self.entity_embed(head)
        t = self.entity_embed(tail)
        r = self.relation_embed(relation)

        #print(h)
        batch_size = h.shape[0]
        # normalize
        h /= torch.norm(h, dim=1).view(batch_size, -1)
        t /= torch.norm(t, dim=1).view(batch_size, -1)
        r /= torch.norm(r, dim=1).view(batch_size, -1)

        pred =  -1 * torch.norm((h + r - t), dim=1)

        return pred
    
    
class SparseTransE(nn.Module):

    def __init__(self, embedding_dim, relation_size, entity_size, gamma=1, alpha=1e-4):
        super(SparseTransE, self).__init__()
        self.embedding_dim = embedding_dim

        self.entity_embed = nn.Embedding(entity_size, embedding_dim)
        self.relation_embed = nn.Embedding(relation_size, embedding_dim)
        # model para init(normalize)

        # margin para
        self.gamma = gamma
        
        # 正則化パラメータ
        self.alpha = alpha
        
        
    def forward(self, head, tail, relation, n_head, n_tail, n_relation, 
                reg_user, reg_item, reg_brand):

        h = self.entity_embed(head)
        t = self.entity_embed(tail)
        r = self.relation_embed(relation)
        n_h = self.entity_embed(n_head)
        n_t = self.entity_embed(n_tail)
        n_r = self.relation_embed(n_relation)

        batch_size = h.shape[0]
        # normalize
        h = h / torch.norm(h, dim=1).view(batch_size, -1)
        t = t / torch.norm(t, dim=1).view(batch_size, -1)
        r = r / torch.norm(r, dim=1).view(batch_size, -1)
        n_h = n_h / torch.norm(n_h, dim=1).view(batch_size, -1)
        n_t = n_t / torch.norm(n_t, dim=1).view(batch_size, -1)
        n_r = n_r / torch.norm(n_r, dim=1).view(batch_size, -1)

        score = self.gamma + torch.norm((h + r - t), dim=1) - torch.norm((n_h + n_r - n_t), dim=1)
        
        # 正則化
        if len(reg_user) == 0:
            reg_u = torch.zeros(2, 2)
        else:
            reg_u = self.entity_embed(reg_user)
            reg_u = reg_u / torch.norm(reg_u, dim=1).view(reg_u.shape[0], -1)

        if len(reg_item) == 0:
            reg_i = torch.zeros(2, 2)
        else:
            reg_i = self.entity_embed(reg_item)
            reg_i = reg_i / torch.norm(reg_i, dim=1).view(reg_i.shape[0], -1)

        if len(reg_brand) == 0:
            reg_b = torch.zeros(2, 2)
        else:
            reg_b = self.entity_embed(reg_brand)
            reg_b = reg_b / torch.norm(reg_b, dim=1).view(reg_b.shape[0], -1)

        
        reg = torch.norm(torch.mm(reg_u, reg_u.T)) + torch.norm(torch.mm(reg_i, reg_i.T)) \
            + torch.norm(torch.mm(reg_b, reg_b.T))

        score = score + self.alpha * reg
        
        return score
    
    def predict(self, head, tail, relation):
        h = self.entity_embed(head)
        t = self.entity_embed(tail)
        r = self.relation_embed(relation)

        batch_size = h.shape[0]
        # normalize
        h /= torch.norm(h, dim=1).view(batch_size, -1)
        t /= torch.norm(t, dim=1).view(batch_size, -1)
        r /= torch.norm(r, dim=1).view(batch_size, -1)

        pred =  -1 * torch.norm((h + r - t), dim=1)

        return pred
    

class Complex(nn.Module):

    def __init__(self, embedding_dim, relation_size, entity_size):
        super(Complex, self).__init__()
        self.embedding_dim = embedding_dim
        self.entity_re = nn.Embedding(entity_size, embedding_dim)
        self.relation_re = nn.Embedding(relation_size, embedding_dim)
        self.entity_im = nn.Embedding(entity_size, embedding_dim)
        self.relation_im = nn.Embedding(relation_size, embedding_dim)
        
        
    def forward(self, head, tail, relation):
        head_re = self.entity_re(head)
        tail_re = self.entity_re(tail)
        relation_re = self.relation_re(relation)
        head_im = self.entity_im(head)
        tail_im = self.entity_im(tail)
        relation_im = self.relation_im(relation)
        
        score = torch.sum(relation_re * head_re * tail_re, axis=1) \
                + torch.sum(relation_re * head_im * tail_im,  axis=1) \
                + torch.sum(relation_im * head_re * tail_im,  axis=1) \
                - torch.sum(relation_im * head_im * tail_re,  axis=1)

        score = torch.sigmoid(score)
        
        return score
    
    def predict(self, head, tail, relation):
        head_re = self.entity_re(head)
        tail_re = self.entity_re(tail)
        relation_re = self.relation_re(relation)
        head_im = self.entity_im(head)
        tail_im = self.entity_im(tail)
        relation_im = self.relation_im(relation)
        
        score = torch.sum(relation_re * head_re * tail_re, axis=1) \
                + torch.sum(relation_re * head_im * tail_im,  axis=1) \
                + torch.sum(relation_im * head_re * tail_im,  axis=1) \
                - torch.sum(relation_im * head_im * tail_re,  axis=1)

        score = torch.sigmoid(score)
        
        return score


class RegComplex(nn.Module):

    def __init__(self, embedding_dim, relation_size, entity_size, alpha):
        super(RegComplex, self).__init__()
        self.embedding_dim = embedding_dim
        self.entity_re = nn.Embedding(entity_size, embedding_dim)
        self.relation_re = nn.Embedding(relation_size, embedding_dim)
        self.entity_im = nn.Embedding(entity_size, embedding_dim)
        self.relation_im = nn.Embedding(relation_size, embedding_dim)

        self.alpha = alpha
        
        
    def forward(self, head, tail, relation,
                reg_user, reg_item, reg_brand):
        head_re = self.entity_re(head)
        tail_re = self.entity_re(tail)
        relation_re = self.relation_re(relation)
        head_im = self.entity_im(head)
        tail_im = self.entity_im(tail)
        relation_im = self.relation_im(relation)
        
        score = torch.sum(relation_re * head_re * tail_re, axis=1) \
                + torch.sum(relation_re * head_im * tail_im,  axis=1) \
                + torch.sum(relation_im * head_re * tail_im,  axis=1) \
                - torch.sum(relation_im * head_im * tail_re,  axis=1)

        score = torch.sigmoid(score)
        
        # 正則化
        if len(reg_user) == 0:
            reg_u = torch.zeros(2, 2)
        else:
            reg_u = torch.cat([self.entity_re(reg_user) , self.entity_im(reg_user)], dim=0)
            #reg_u = reg_u / torch.norm(reg_u, dim=1).view(reg_u.shape[0], -1)

        if len(reg_item) == 0:
            reg_i = torch.zeros(2, 2)
        else:
            reg_i = torch.cat([self.entity_re(reg_item), self.entity_im(reg_item)], dim=0)
            #reg_i = reg_i / torch.norm(reg_i, dim=1).view(reg_i.shape[0], -1)

        if len(reg_brand) == 0:
            reg_b = torch.zeros(2, 2)
        else:
            reg_b = torch.cat([self.entity_re(reg_brand), self.entity_im(reg_brand)], dim=0)
            #reg_b = reg_b / torch.norm(reg_b, dim=1).view(reg_b.shape[0], -1)

        
        reg = torch.norm(torch.mm(reg_u, reg_u.T)) + torch.norm(torch.mm(reg_i, reg_i.T)) \
            + torch.norm(torch.mm(reg_b, reg_b.T))

        #score = score + self.alpha * reg
        
        return score, reg
    

    def predict(self, head, tail, relation):
        head_re = self.entity_re(head)
        tail_re = self.entity_re(tail)
        relation_re = self.relation_re(relation)
        head_im = self.entity_im(head)
        tail_im = self.entity_im(tail)
        relation_im = self.relation_im(relation)
        
        score = torch.sum(relation_re * head_re * tail_re, axis=1) \
                + torch.sum(relation_re * head_im * tail_im,  axis=1) \
                + torch.sum(relation_im * head_re * tail_im,  axis=1) \
                - torch.sum(relation_im * head_im * tail_re,  axis=1)

        score = torch.sigmoid(score)
        
        return score