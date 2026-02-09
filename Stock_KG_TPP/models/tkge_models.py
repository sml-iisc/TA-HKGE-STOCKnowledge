import os
import math
import pickle
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity


class TTransEModel(nn.Module):
    def __init__(self, config):
        super(TTransEModel, self).__init__()
        self.L1_flag = config['L1_flag']
        self.embedding_size = config['embedding_size']
        self.entity_total = config['entity_total']
        self.relation_total = config['relation_total']

        ent_weight = torch.Tensor(self.entity_total, self.embedding_size)
        rel_weight = torch.Tensor(self.relation_total, self.embedding_size)

        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_uniform_(ent_weight)
        nn.init.xavier_uniform_(rel_weight)


        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)


        self.year_embeddings    = nn.Embedding(24, self.embedding_size, padding_idx=0)
        self.month_embeddings   = nn.Embedding(13, self.embedding_size, padding_idx=0)
        self.day_embeddings     = nn.Embedding(32, self.embedding_size, padding_idx=0)
        self.hour_embeddings    = nn.Embedding(25, self.embedding_size, padding_idx=0)
        self.minutes_embeddings = nn.Embedding(61, self.embedding_size, padding_idx=0)
        self.sec_embeddings     = nn.Embedding(61, self.embedding_size, padding_idx=0)

        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)

        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)

        self.ent_embeddings.weight.data = normalize_entity_emb
        self.rel_embeddings.weight.data = normalize_relation_emb

    def regularize_embeddings(self):
        self.ent_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.rel_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.year_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.month_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.day_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.hour_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.minutes_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.sec_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def forward(self, pos_h, pos_t, pos_r, pos_tem):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_embeddings(pos_r)
        pos_tem_e = self.year_embeddings(pos_tem[:, 0]) + self.month_embeddings(pos_tem[:, 1]) + \
                    self.day_embeddings(pos_tem[:, 2]) + self.hour_embeddings(pos_tem[:, 3]) + \
                    self.minutes_embeddings(pos_tem[:, 4]) + self.sec_embeddings(pos_tem[:, 5])
 
        neg_h = torch.randint_like(pos_h, low=1, high=self.entity_total).to(pos_h.device)

        neg_h_e = self.ent_embeddings(neg_h)

        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e + pos_tem_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + pos_r_e + pos_tem_e - pos_t_e), 1)
        else:
            pos = torch.sum((pos_h_e + pos_r_e + pos_tem_e - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + pos_r_e + pos_tem_e - pos_t_e) ** 2, 1)

        loss = torch.mean(1 - pos + neg)
        return loss

class GCNTransEModel(nn.Module):
    def __init__(self, config):
        super(TTransEModel, self).__init__()
        self.L1_flag = config['L1_flag']
        self.embedding_size = config['embedding_size']
        self.entity_total = config['entity_total']+2
        self.relation_total = config['relation_total']+2

        ent_weight = torch.Tensor(self.entity_total, self.embedding_size)
        rel_weight = torch.Tensor(self.relation_total, self.embedding_size)
        year_weight = torch.Tensor(self.entity_total, self.embedding_size)
        month_weight = torch.Tensor(self.relation_total, self.embedding_size)
        day_weight = torch.Tensor(self.entity_total, self.embedding_size)
        hour_weight = torch.Tensor(self.relation_total, self.embedding_size)
        min_weight = torch.Tensor(self.entity_total, self.embedding_size)
        sec_weight = torch.Tensor(self.relation_total, self.embedding_size)

        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_uniform_(ent_weight)
        nn.init.xavier_uniform_(rel_weight)
        nn.init.xavier_uniform_(year_weight)
        nn.init.xavier_uniform_(month_weight)
        nn.init.xavier_uniform_(day_weight)
        nn.init.xavier_uniform_(hour_weight)
        nn.init.xavier_uniform_(min_weight)
        nn.init.xavier_uniform_(sec_weight)


        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_type_embeddings = nn.Embedding(config['relation_total']+2, 16)


        self.year_embeddings    = nn.Embedding(24, self.embedding_size, padding_idx=0)
        self.month_embeddings   = nn.Embedding(13, self.embedding_size, padding_idx=0)
        self.day_embeddings     = nn.Embedding(32, self.embedding_size, padding_idx=0)
        self.hour_embeddings    = nn.Embedding(25, self.embedding_size, padding_idx=0)
        self.minutes_embeddings = nn.Embedding(61, self.embedding_size, padding_idx=0)
        self.sec_embeddings     = nn.Embedding(61, self.embedding_size, padding_idx=0)

        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_type_embeddings.weight = nn.Parameter(rel_weight)
        self.year_embeddings.weight = nn.Parameter(year_weight)
        self.month_embeddings.weight = nn.Parameter(month_weight)
        self.day_embeddings.weight = nn.Parameter(day_weight)
        self.hour_embeddings.weight = nn.Parameter(hour_weight)
        self.minutes_embeddings.weight = nn.Parameter(min_weight)
        self.sec_embeddings.weight = nn.Parameter(sec_weight)

        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)

        self.ent_embeddings.weight.data = normalize_entity_emb
        self.rel_type_embeddings.weight.data = normalize_relation_emb

        self.graph_model = Sequential('x, hyperedge_index', [
                        (HypergraphConv(8, 32, dropout=0.1), 'x, hyperedge_index -> x1'),
                        nn.LeakyReLU(inplace=True),
                        (HypergraphConv(32, 32, dropout=0.1), 'x1, hyperedge_index -> x2'),
                        nn.LeakyReLU(inplace=True),
                        
                        nn.Linear(32, 20),
                    ])
        
        self.conv1 = RGATConv(16, 32, config['relation_total'], edge_dim=16, heads=1, dim=1, attention_mechanism="within-relation", mod="additive")
        self.conv2 = RGATConv(32, 64, config['relation_total'], edge_dim=16, heads=1, dim=1, attention_mechanism="within-relation", mod="additive")
        self.lin = nn.Linear(64, 20)

  

    def regularize_embeddings(self):
        self.ent_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.rel_type_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.year_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.month_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.day_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.hour_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.minutes_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.sec_embeddings.weight.data.renorm_(p=2, dim=0, maxnorm=1)

    def forward(self, pos_h, pos_t, pos_r, pos_tem):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_type_embeddings(pos_r)
        pos_tem_e = self.year_embeddings(pos_tem[:, 0]) + self.month_embeddings(pos_tem[:, 1]) + \
                    self.day_embeddings(pos_tem[:, 2]) + self.hour_embeddings(pos_tem[:, 3]) + \
                    self.minutes_embeddings(pos_tem[:, 4]) + self.sec_embeddings(pos_tem[:, 5])
 
        neg_h = torch.randint_like(pos_h, low=1, high=self.entity_total).to(pos_h.device)

        neg_h_e = self.ent_embeddings(neg_h)

        if self.L1_flag:
            pos = torch.sum(torch.abs(pos_h_e + pos_r_e + pos_tem_e - pos_t_e), 1)
            neg = torch.sum(torch.abs(neg_h_e + pos_r_e + pos_tem_e - pos_t_e), 1)
        else:
            pos = torch.sum((pos_h_e + pos_r_e + pos_tem_e - pos_t_e) ** 2, 1)
            neg = torch.sum((neg_h_e + pos_r_e + pos_tem_e - pos_t_e) ** 2, 1)

        loss = torch.mean(1 - pos + neg)
        return loss
    

class TDistmultModel(nn.Module):
    def __init__(self, config):
        super(TADistmultModel, self).__init__()
        self.L1_flag = config['L1_flag']
        self.embedding_size = config['embedding_size']
        self.entity_total = config['entity_total']
        self.relation_total = config['relation_total']

        self.criterion = nn.Softplus()
        torch.nn.BCELoss()

        self.lstm = nn.LSTM(input_size = self.embedding_size, hidden_size = self.embedding_size, num_layers = 1, batch_first = True, bidirectional = False)

        self.year_embeddings    = nn.Embedding(24, self.embedding_size, padding_idx=0)
        self.month_embeddings   = nn.Embedding(13, self.embedding_size, padding_idx=0)
        self.day_embeddings     = nn.Embedding(32, self.embedding_size, padding_idx=0)
        self.hour_embeddings    = nn.Embedding(25, self.embedding_size, padding_idx=0)
        self.minutes_embeddings = nn.Embedding(61, self.embedding_size, padding_idx=0)
        self.sec_embeddings     = nn.Embedding(61, self.embedding_size, padding_idx=0)

        ent_weight = torch.Tensor(self.entity_total, self.embedding_size)
        rel_weight = torch.Tensor(self.relation_total, self.embedding_size)
        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_uniform_(ent_weight)
        nn.init.xavier_uniform_(rel_weight)
        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)

        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        self.ent_embeddings.weight.data = normalize_entity_emb
        self.rel_embeddings.weight.data = normalize_relation_emb

    def scoring(self, h, t, r):
        return torch.sum(h * t * r, 1, False)

    def forward(self, pos_h, pos_t, pos_r, pos_tem):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_r_e = self.rel_embeddings(pos_r)
        pos_tem_e = self.year_embeddings(pos_tem[:, 0]) + self.month_embeddings(pos_tem[:, 1]) + \
                    self.day_embeddings(pos_tem[:, 2]) + self.hour_embeddings(pos_tem[:, 3]) + \
                    self.minutes_embeddings(pos_tem[:, 4]) + self.sec_embeddings(pos_tem[:, 5])
        pos_r_e += pos_tem_e
 
        neg_h = torch.randint_like(pos_h, low=1, high=self.entity_total).to(pos_h.device)
        neg_t = torch.randint_like(pos_t, low=1, high=self.relation_total).to(pos_t.device)

        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)

        pos = self.scoring(pos_h_e, pos_t_e, pos_r_e)
        neg = self.scoring(neg_h_e, neg_t_e, pos_r_e)

        loss = torch.mean(pos - neg)
        return loss


class TADistmultModel(nn.Module):
    def __init__(self, config):
        super(TADistmultModel, self).__init__()
        self.L1_flag = config['L1_flag']
        self.embedding_size = config['embedding_size']
        self.entity_total = config['entity_total']
        self.relation_total = config['relation_total']

        self.criterion = nn.Softplus()
        torch.nn.BCELoss()

        self.lstm = nn.LSTM(input_size = self.embedding_size, hidden_size = self.embedding_size, num_layers = 1, batch_first = True, bidirectional = False)

        self.year_embeddings    = nn.Embedding(24, self.embedding_size, padding_idx=0)
        self.month_embeddings   = nn.Embedding(13, self.embedding_size, padding_idx=0)
        self.day_embeddings     = nn.Embedding(32, self.embedding_size, padding_idx=0)
        self.hour_embeddings    = nn.Embedding(25, self.embedding_size, padding_idx=0)
        self.minutes_embeddings = nn.Embedding(61, self.embedding_size, padding_idx=0)
        self.sec_embeddings     = nn.Embedding(61, self.embedding_size, padding_idx=0)

        ent_weight = torch.Tensor(self.entity_total, self.embedding_size)
        rel_weight = torch.Tensor(self.relation_total, self.embedding_size)
        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_uniform_(ent_weight)
        nn.init.xavier_uniform_(rel_weight)
        self.ent_embeddings = nn.Embedding(self.entity_total, self.embedding_size)
        self.rel_embeddings = nn.Embedding(self.relation_total, self.embedding_size)
        self.ent_embeddings.weight = nn.Parameter(ent_weight)
        self.rel_embeddings.weight = nn.Parameter(rel_weight)

        normalize_entity_emb = F.normalize(self.ent_embeddings.weight.data, p=2, dim=1)
        normalize_relation_emb = F.normalize(self.rel_embeddings.weight.data, p=2, dim=1)
        self.ent_embeddings.weight.data = normalize_entity_emb
        self.rel_embeddings.weight.data = normalize_relation_emb

    def scoring(self, h, t, r):
        return torch.sum(h * t * r, 1, False)

    def forward(self, pos_h, pos_t, pos_r, pos_tem):
        pos_h_e = self.ent_embeddings(pos_h)
        pos_t_e = self.ent_embeddings(pos_t)
        pos_rseq_e = self.get_rseq(pos_r, pos_tem)
 
        neg_h = torch.randint_like(pos_h, low=1, high=self.entity_total).to(pos_h.device)
        neg_t = torch.randint_like(pos_t, low=1, high=self.relation_total).to(pos_t.device)

        neg_h_e = self.ent_embeddings(neg_h)
        neg_t_e = self.ent_embeddings(neg_t)

        pos = self.scoring(pos_h_e, pos_t_e, pos_rseq_e)
        neg = self.scoring(neg_h_e, neg_t_e, pos_rseq_e)

        loss = torch.mean(pos - neg)
        return loss

    def get_rseq(self, r, pos_tem):
        r_e = self.rel_embeddings(r)
        r_e = r_e.unsqueeze(0).transpose(0, 1)

        y_e, m_e, d_e, h_e, mi_e, s_e = self.year_embeddings(pos_tem[:, 0]), self.month_embeddings(pos_tem[:, 1]), \
                    self.day_embeddings(pos_tem[:, 2]), self.hour_embeddings(pos_tem[:, 3]), \
                    self.minutes_embeddings(pos_tem[:, 4]), self.sec_embeddings(pos_tem[:, 5])
        y_e = y_e.unsqueeze(1)
        m_e = m_e.unsqueeze(1)
        d_e = d_e.unsqueeze(1)
        h_e = h_e.unsqueeze(1)
        mi_e = mi_e.unsqueeze(1)
        s_e = s_e.unsqueeze(1)
        seq_e = torch.cat((s_e, mi_e, h_e, d_e, m_e, y_e, r_e), 1)

        hidden_tem, y = self.lstm(seq_e)
        hidden_tem = y[0].squeeze(0)
        rseq_e = hidden_tem
        return rseq_e
    
class ATISE(nn.Module):
    def __init__(self, kg, embedding_dim, batch_size, learning_rate, gamma, cmin, cmax, gpu=True):
        super(ATISE, self).__init__()
        self.gpu = gpu
        self.kg = kg
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.cmin = cmin
        self.cmax = cmax
        # Nets
        self.emb_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        self.emb_E_var = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        self.emb_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
        self.emb_R_var = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
        self.emb_TE = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        self.alpha_E = torch.nn.Embedding(self.kg.n_entity, 1, padding_idx=0)
        self.beta_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        self.omega_E = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
        self.emb_TR = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
        self.alpha_R = torch.nn.Embedding(self.kg.n_relation, 1, padding_idx=0)
        self.beta_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
        self.omega_R = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
        
    
        # Initialization
        r = 6 / np.sqrt(self.embedding_dim)
        self.emb_E.weight.data.uniform_(-r, r)
        self.emb_E_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_R.weight.data.uniform_(-r, r)
        self.emb_R_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_TE.weight.data.uniform_(-r, r)
        self.alpha_E.weight.data.uniform_(0, 0)
        self.beta_E.weight.data.uniform_(0, 0)
        self.omega_E.weight.data.uniform_(-r, r)
        self.emb_TR.weight.data.uniform_(-r, r)
        self.alpha_R.weight.data.uniform_(0, 0)
        self.beta_R.weight.data.uniform_(0, 0)
        self.omega_R.weight.data.uniform_(-r, r)

        # Regularization
        self.normalize_embeddings()
        
        if self.gpu:
            self.cuda()
            
    def forward(self, X):
        h_i, t_i, r_i, d_i = X[:, 0].astype(np.int64), X[:, 1].astype(np.int64), X[:, 2].astype(np.int64), X[:, 3].astype(np.float32)

        pi = 3.14159265358979323846
        h_mean = self.emb_E(h_i).view(-1, self.embedding_dim) + \
            d_i.view(-1, 1) * self.alpha_E(h_i).view(-1, 1) * self.emb_TE(h_i).view(-1, self.embedding_dim) \
            + self.beta_E(h_i).view(-1, self.embedding_dim) * torch.sin(
            2 * pi * self.omega_E(h_i).view(-1, self.embedding_dim) * d_i.view(-1, 1))
            
        t_mean = self.emb_E(t_i).view(-1, self.embedding_dim) + \
            d_i.view(-1, 1) * self.alpha_E(t_i).view(-1, 1) * self.emb_TE(t_i).view(-1, self.embedding_dim) \
            + self.beta_E(t_i).view(-1, self.embedding_dim) * torch.sin(
            2 * pi * self.omega_E(t_i).view(-1, self.embedding_dim) * d_i.view(-1, 1))
            
        r_mean = self.emb_R(r_i).view(-1, self.embedding_dim) + \
            d_i.view(-1, 1) * self.alpha_R(r_i).view(-1, 1) * self.emb_TR(r_i).view(-1, self.embedding_dim) \
            + self.beta_R(r_i).view(-1, self.embedding_dim) * torch.sin(
            2 * pi * self.omega_R(r_i).view(-1, self.embedding_dim) * d_i.view(-1, 1))


        h_var = self.emb_E_var(h_i).view(-1, self.embedding_dim)
        t_var = self.emb_E_var(t_i).view(-1, self.embedding_dim)
        r_var = self.emb_R_var(r_i).view(-1, self.embedding_dim)

        out1 = torch.sum((h_var+t_var)/r_var, 1)+torch.sum(((r_mean-h_mean+t_mean)**2)/r_var, 1)-self.embedding_dim
        out2 = torch.sum(r_var/(h_var+t_var), 1)+torch.sum(((h_mean-t_mean-r_mean)**2)/(h_var+t_var), 1)-self.embedding_dim
        out = (out1+out2)/4
        

        return out
    
    
    
    def log_rank_loss(self, y_pos, y_neg, temp=0):
        M = y_pos.size(0)
        N = y_neg.size(0)
        y_pos = self.gamma-y_pos
        y_neg = self.gamma-y_neg
        C = int(N / M)
        y_neg = y_neg.view(C, -1).transpose(0, 1)
        p = F.softmax(temp * y_neg)
        loss_pos = torch.sum(F.softplus(-1 * y_pos))
        loss_neg = torch.sum(p * F.softplus(y_neg))
        loss = (loss_pos + loss_neg) / 2 / M
        if self.gpu:
            loss = loss.cuda()
        return loss


    def rank_loss(self, y_pos, y_neg):
        M = y_pos.size(0)
        N = y_neg.size(0)
        C = int(N / M)
        y_pos = y_pos.repeat(C)
        target = Variable(torch.from_numpy(-np.ones(N, dtype=np.float32))).cuda()
        loss = nn.MarginRankingLoss(margin=self.gamma)
        loss = loss(y_pos, y_neg, target)
        return loss

    def normalize_embeddings(self):
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_E_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_R_var.weight.data.uniform_(self.cmin, self.cmax)
        self.emb_TE.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_TR.weight.data.renorm_(p=2, dim=0, maxnorm=1)

        
    def regularization_embeddings(self):
        lower = torch.tensor(self.cmin).float()
        upper = torch.tensor(self.cmax).float()
        if self.gpu:
            lower = lower.cuda()
            upper = upper.cuda()
        self.emb_E_var.weight.data=torch.where(self.emb_E_var.weight.data<self.cmin,lower,self.emb_E_var.weight.data)
        self.emb_E_var.weight.data=torch.where(self.emb_E_var.weight.data>self.cmax,upper,self.emb_E_var.weight.data)
        self.emb_R_var.weight.data=torch.where(self.emb_R_var.weight.data < self.cmin,lower, self.emb_R_var.weight.data)
        self.emb_R_var.weight.data=torch.where(self.emb_R_var.weight.data > self.cmax,upper, self.emb_R_var.weight.data)
        self.emb_E.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_R.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_TE.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        self.emb_TR.weight.data.renorm_(p=2, dim=0, maxnorm=1)
        

