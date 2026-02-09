import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class HHP(nn.Module):
    def __init__(self, learning_rate, batch_size, neg_size, nbr_size, node_size, node_dim, num_node_types,
                 num_edge_types, norm_rate, datasetname,device, opt='Adam'):
        super(HHP, self).__init__()
        
        self.node_size = node_size
        self.node_dim = node_dim
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.batch_size = batch_size
        self.neg_size = neg_size
        self.norm_rate = norm_rate
        self.learning_rate = learning_rate
        self.nbr_size = nbr_size
        self.datasetname = datasetname
        self.device = device

        init_range_embed = np.sqrt(3.0 / (self.node_size + self.node_dim))

        init_range_edge_type = np.sqrt(3.0 / (self.node_dim + self.node_dim))

        self.embedding = nn.Parameter(torch.empty(self.node_size, self.node_dim))

        nn.init.uniform_(self.embedding, -init_range_embed, init_range_embed)

        self.edge_type_embed = nn.Parameter(torch.empty(self.num_edge_types, self.node_dim))
        nn.init.uniform_(self.edge_type_embed, -init_range_edge_type, init_range_edge_type)

        if opt == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = optim.SGD(self.parameters(), lr=self.learning_rate)

    def forward(self, batch_data):
        e_types = batch_data[0]
                
        s_ids, s_types, s_negs, s_nbr_infos = batch_data[1]
        t_ids, t_types, t_negs, t_nbr_infos = batch_data[2]

        basic_info = self.construct_mu(s_ids, s_types, t_ids, t_types, e_types, t_negs, s_negs, neg_size=self.neg_size)
        mu, neg_mus_st, neg_mus_ts, s_embed, t_embed, neg_embed_s_list, neg_embed_t_list = basic_info
        pos_loss_st, neg_loss_st = self.construct_mutual_influence(s_embed, s_nbr_infos, t_embed, neg_embed_t_list)
        pos_loss_ts, neg_loss_ts = self.construct_mutual_influence(t_embed, t_nbr_infos, s_embed, neg_embed_s_list)

        lambda_st_pos = mu + pos_loss_st + pos_loss_ts
        lambda_st_neg = neg_mus_st + neg_loss_st
        lambda_ts_neg = neg_mus_ts + neg_loss_ts

        loss = -torch.mean(torch.log(torch.sigmoid(lambda_st_pos) + 1e-6)) \
               - torch.mean(torch.log(torch.sigmoid(-lambda_st_neg) + 1e-6)) \
               - torch.mean(torch.log(torch.sigmoid(-lambda_ts_neg) + 1e-6)) \
               + self.norm_rate * torch.sum(self.edge_type_embed ** 2)
        return loss


    def construct_node_latent_embed(self, node_ids, node_types, node_size, type_size):

        node_ids = node_ids.to(self.device).long()
        node_types = node_types.to(self.device).long()
        
        node_size = node_ids.size(0)

        indices = (torch.arange(node_size, device=self.device) * type_size + node_types).long() 

        embedding = self.embedding.index_select(0, node_ids)

        new_matrix = torch.zeros(node_size * type_size, self.node_dim, device=self.device)
        new_matrix.index_add_(0, indices, embedding)
        new_matrix = new_matrix.view(node_size, type_size, self.node_dim)

        dense_layer = nn.Linear(self.node_dim, self.node_dim).to(self.device)
        torch.nn.init.xavier_uniform_(dense_layer.weight)
        embed_typed = F.leaky_relu(dense_layer(new_matrix))

        node_final_embeds = embed_typed.view(node_size * type_size, self.node_dim)[indices]

        return node_final_embeds

    def construct_mu(self, s_ids, s_types, t_ids, t_types, e_types, t_neg_ids, s_neg_ids, neg_size):
        s_embed = self.construct_node_latent_embed(s_ids, s_types, self.batch_size, self.num_node_types)
        t_embed = self.construct_node_latent_embed(t_ids, t_types, self.batch_size, self.num_node_types)

        e_types = e_types.long()
        e_embed = self.edge_type_embed[e_types]
        mu = self.g_func(s_embed + e_embed, t_embed, 'l2')
        
        neg_mus_st = []
        neg_embed_s_list = []
        neg_embed_t_list = []
        neg_mus_ts = []
        
        for i in range(neg_size):
            neg_t_embed = self.construct_node_latent_embed(t_neg_ids[:, i], t_types, self.batch_size, self.num_node_types)
            neg_embed_t_list.append(neg_t_embed)
            neg_mu_t_i = self.g_func(s_embed, neg_t_embed, 'l2').view(-1, 1)
            neg_mus_st.append(neg_mu_t_i)
            
            neg_s_embed = self.construct_node_latent_embed(s_neg_ids[:, i], s_types, self.batch_size, self.num_node_types)
            neg_embed_s_list.append(neg_s_embed)
            neg_mu_s_i = self.g_func(t_embed, neg_s_embed, 'l2').view(-1, 1)
            neg_mus_ts.append(neg_mu_s_i)

        return mu, torch.cat(neg_mus_st, dim=-1), torch.cat(neg_mus_ts, dim=-1), s_embed, t_embed, neg_embed_s_list, neg_embed_t_list

    def construct_mutual_influence(self, node_embed, node_nbr_infos, target_embed, neg_embed):
        pos_info = []
        neg_info = []
        att_info = []
        mask = []

        for i in range(self.num_edge_types):
            nbr_ids, nbr_masks, nbr_weights, nbr_flag = node_nbr_infos[i]

            if torch.all(nbr_flag > 0):
                pos_g = torch.zeros((self.batch_size, 1)).to(node_embed.device)
                neg_g = torch.zeros((self.batch_size, self.neg_size)).to(node_embed.device)
                hete_att = torch.zeros((self.batch_size, 1)).to(node_embed.device)
            else:
                pos_g, neg_g, hete_att = self.edge_type_distance(node_embed, nbr_ids, i, nbr_weights, nbr_masks, target_embed, neg_embed)

            pos_info.append(pos_g)
            neg_info.append(neg_g)
            att_info.append(hete_att)
            mask.append(nbr_flag)

        mask = torch.cat(mask, dim=-1).view(self.batch_size, self.num_edge_types).bool().to(node_embed.device)
        att_info = torch.cat(att_info, dim=-1).to(node_embed.device)
        padding = torch.full((self.batch_size, self.num_edge_types), -2 ** 32 + 1.0).to(node_embed.device)
        padding2 = torch.full((self.batch_size, self.num_edge_types), 0.0).to(node_embed.device)
        att_v1 = F.softmax(torch.where(mask, att_info, padding), dim=1)
        norm_att = F.softmax(torch.where(mask, att_v1, padding2), dim=1)

        pos_info = torch.cat(pos_info, dim=-1).to(node_embed.device)
        neg_info = torch.cat(neg_info, dim=-1).view(self.batch_size, self.num_edge_types, self.neg_size).to(node_embed.device)
        neg_info = neg_info.permute(0, 2, 1)

        pos_loss = torch.sum(norm_att * pos_info, dim=-1)
        neg_loss = torch.sum(torch.matmul(neg_info, norm_att.unsqueeze(2)), dim=-1)
        return pos_loss, neg_loss

    def edge_type_distance(self, node_embed, ids, e_type, weight, mask, target_embed, neg_embed):
        ids = ids.long().to(self.device)  
        nbr_embed = F.leaky_relu(nn.Linear(self.node_dim, self.node_dim).to(self.device)(self.embedding[ids]))
        edge_embed = self.edge_type_embed[e_type].view(1, 1, self.node_dim).to(self.device)
        node_embed = node_embed.unsqueeze(1).to(self.device)
        
        nbr_distance = self.g_func(node_embed + edge_embed, nbr_embed, opt='l2')
        paddings = torch.full_like(nbr_distance, -2 ** 32 + 1.0).to(self.device)
        paddings2 = torch.full_like(nbr_distance, 0.0).to(self.device)
        nbr_distance2 = torch.where(mask.bool().to(self.device), nbr_distance, paddings)
        atts = F.softmax(nbr_distance2, dim=-1)
        atts_2 = torch.where(mask.bool().to(self.device), atts, paddings2)
        new_weight = atts_2 * weight.to(self.device)

        mutual_subs = self.g_func(nbr_embed, target_embed.unsqueeze(1).to(self.device), 'l2')
        mutual_neg_subs = [self.g_func(nbr_embed, neg_embed[i].unsqueeze(1).to(self.device), 'l2') for i in range(self.neg_size)]

        avg_embed = torch.sum(torch.matmul(atts_2.unsqueeze(1), nbr_embed), dim=1)
        avg_weight_1 = torch.sum(weight.to(self.device), dim=1)
        nbr_numbers = torch.clamp(torch.sum(mask.to(self.device), dim=1).float(), 1.0, self.nbr_size)
        ave_weight = (avg_weight_1 / nbr_numbers).view(-1, 1)

        avg_info = ave_weight * avg_embed

        hete_att = F.leaky_relu(nn.Linear(self.node_dim, 1).to(self.device)(avg_info))
        pos_mutual_influ = torch.sum(new_weight * mutual_subs, dim=-1)
    
        neg_mutual_influ = [torch.sum(new_weight * mutual_neg_subs[i], dim=-1).view(self.batch_size, 1) for i in range(self.neg_size)]

        batch_size = new_weight.size(0)
        neg_mutual_influ = [influ.view(batch_size, 1) for influ in neg_mutual_influ]
        
        return pos_mutual_influ.view(batch_size, 1), torch.cat(neg_mutual_influ, dim=1), hete_att.view(batch_size, 1)    
    
    def g_func(self, x, y, opt='l2'):
        if opt == 'l2':
            return -torch.sum((x - y) ** 2, dim=-1)
        elif opt == 'l1':
            return -torch.sum(torch.abs(x - y), dim=-1)
        else:
            return -torch.sum((x - y) ** 2, dim=-1)