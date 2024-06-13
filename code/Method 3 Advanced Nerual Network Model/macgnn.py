import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Dice(nn.Module):
    
    def __init__(self):
        super(Dice, self).__init__()
        self.alpha = nn.Parameter(torch.zeros((1, )))
        
    def forward(self, x):
        avg = x.mean(dim=0)
        std = x.std(dim=0)
        norm_x = (x - avg) / std
        p = torch.sigmoid(norm_x+1e-8)

        return x.mul(p) + self.alpha * x.mul(1 - p)

class NeighborAggregation(nn.Module):
    def __init__(self, embed_dim=8, hidden_dim=8):
        super(NeighborAggregation, self).__init__()
        self.Q_w = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.K_w = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.V_w = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.trans_d = math.sqrt(hidden_dim)
        self.get_score = nn.Softmax(dim=-1)

    def forward(self, query, key):
        trans_Q = self.Q_w(query)
        trans_K = self.K_w(key)
        trans_V = self.V_w(query)
        score = self.get_score(torch.bmm(trans_Q, torch.transpose(trans_K,1,2))/(self.trans_d))
        answer = torch.mul(trans_V, score)
        return answer


class MacGNN(nn.Module):

    def __init__(self, field_dims, u_group_num, i_group_num, embed_dim, recent_len, initial_embedding, tau=0.8, device='cpu'):
        super(MacGNN, self).__init__()
        self.embed_dim = embed_dim
        self.user_embed = nn.Embedding(field_dims[0], embed_dim)
        self.item_embed = nn.Embedding(field_dims[1], embed_dim)
        with torch.no_grad():
            self.user_embed.weight[:] = torch.tensor(initial_embedding[0])
            self.item_embed.weight[:] = torch.tensor(initial_embedding[1])
        
        self.user_relation_embed = nn.Embedding(field_dims[0], embed_dim)
        self.item_relation_embed = nn.Embedding(field_dims[1], embed_dim)
        
        self.cate_embed = nn.Embedding(field_dims[2], embed_dim)
        self.u_macro_embed = nn.Embedding(u_group_num + 1, embed_dim)
        self.i_macro_embed = nn.Embedding(i_group_num + 1, embed_dim)
        
        torch.nn.init.xavier_uniform_(self.user_relation_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.item_relation_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.cate_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.u_macro_embed.weight.data)
        torch.nn.init.xavier_uniform_(self.i_macro_embed.weight.data)
        self.tau = tau
        
        self.u_group_num = u_group_num + 1
        self.i_group_num = i_group_num + 1
        self.recent_len = recent_len
        
        self.u_shared_aggregator = NeighborAggregation(embed_dim, 2 * embed_dim)
        self.i_shared_aggregator = NeighborAggregation(embed_dim, 2 * embed_dim)
        self.cb_user_aggregator = NeighborAggregation(embed_dim, 2 * embed_dim)

        self.userAggregationAttention = nn.MultiheadAttention(2 * embed_dim, 4, bias=False, batch_first=True)
        self.userAggregation = nn.Conv1d(self.u_group_num, 1, 1, bias = False)
        self.itemAggregationAttention = nn.MultiheadAttention(2 * embed_dim, 4, bias=False, batch_first=True)
        self.itemAggregation = nn.Conv1d(self.i_group_num, 1, 1, bias = False)
        self.cbItemAggregation = nn.Conv1d(10, 1, 1, bias = False)
        
        self.cbUserAggregationAttention = nn.MultiheadAttention(2 * embed_dim, 4, bias=False, batch_first=True)
        self.cbUserAggregation = nn.Conv1d(10, 1, 1, bias = False)

        self.itemRecentAggregationAttention = nn.MultiheadAttention(2 * embed_dim, 4, bias=False, batch_first=True)
        self.itemRecentAggregation = nn.GRU(2 * embed_dim, 2 * embed_dim, bias = False, batch_first=True)
        
        self.userRecentAggregationAttention = nn.MultiheadAttention(2 * embed_dim, 4, bias=False, batch_first=True)
        self.userRecentAggregation = nn.GRU(2 * embed_dim, 2 * embed_dim, bias = False, batch_first=True)
        
        self.macro_weight_func = nn.Softmax(dim=1)
        self.u_gruop_slice = torch.arange(self.u_group_num, requires_grad=False).to(device)
        self.i_gruop_slice = torch.arange(self.i_group_num, requires_grad=False).to(device)
        
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * (14 + 4), 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 80),
            nn.ReLU(inplace=True),
            nn.Linear(80, 1)
        )
    
    def neighbor_aggregation_forward(self, aggregation_layer, aggregation_attn_layer, feature):
        feature, _ = aggregation_attn_layer(feature, feature, feature)
        feature = nn.functional.tanh(feature)
        combined = aggregation_layer(feature)
        return combined.squeeze(1)

    def recent_aggregation_forward(self, aggregation_layer, aggregation_attn_layer, feature):
        feature, _ = aggregation_attn_layer(feature, feature, feature)
        feature = nn.functional.tanh(feature)
        _, combined = aggregation_layer(feature)
        return combined[0]

    def linear_aggregation_forward(self, aggregation_layer, aggregation_attn_layer, feature):
        feature, _ = aggregation_attn_layer(feature, feature, feature)
        feature = nn.functional.tanh(feature)
        feature = feature.reshape(feature.shape[0], -1)
        combined = aggregation_layer(feature)
        return combined
        
    def forward(self, x):
        user_embedding = self.user_embed(x[:, 0])
        user_1ord_neighbor = x[:, 1: self.i_group_num + 1]
        user_2ord_neighbor = x[:, self.i_group_num + 1: self.i_group_num + self.u_group_num + 1]
        user_recent = x[:, self.i_group_num + self.u_group_num + 1: self.i_group_num + self.u_group_num + self.recent_len + 1]
        
        item_embedding = self.item_embed(x[:, self.i_group_num + self.u_group_num + self.recent_len + 1])
        item_1ord_neighbor = x[:, self.i_group_num + self.u_group_num + self.recent_len + 2: self.i_group_num + 2 * self.u_group_num + self.recent_len + 2]
        item_2ord_neighbor = x[:, self.i_group_num + 2 * self.u_group_num + self.recent_len + 2: 2 * self.i_group_num + 2 * self.u_group_num + self.recent_len + 2]
        item_recent = x[:, 2 * self.i_group_num + 2 * self.u_group_num + self.recent_len + 2:]

        item_cb_neighbor = x[:, 2 * self.i_group_num + 2 * self.u_group_num + self.recent_len * 2 + 2:]
        item_cb_embedding = self.item_relation_embed(item_cb_neighbor)
        
        user_relation_embedding = self.user_relation_embed(x[:, 0])
        item_relation_embedding = self.item_relation_embed(x[:, self.i_group_num + self.u_group_num + self.recent_len + 1])
        
        batch_u_gruop_slice = self.u_gruop_slice.expand(x.shape[0], self.u_group_num)
        batch_i_gruop_slice = self.i_gruop_slice.expand(x.shape[0], self.i_group_num)

        user_recent_mask = (user_recent > 0).float().unsqueeze(-1)
        item_recent_mask = (item_recent > 0).float().unsqueeze(-1)
        
        user_1ord_weight = self.macro_weight_func(torch.log(user_1ord_neighbor.float()+1) / self.tau).unsqueeze(-1)
        user_2ord_weight = self.macro_weight_func(torch.log(user_2ord_neighbor.float()+1) / self.tau).unsqueeze(-1)
        item_1ord_weight = self.macro_weight_func(torch.log(item_1ord_neighbor.float()+1) / self.tau).unsqueeze(-1)
        item_2ord_weight = self.macro_weight_func(torch.log(item_2ord_neighbor.float()+1) / self.tau).unsqueeze(-1)
        
        user_1ord_embedding = self.i_macro_embed(batch_i_gruop_slice)
        user_2ord_embedding = self.u_macro_embed(batch_u_gruop_slice)
        item_1ord_embedding = self.u_macro_embed(batch_u_gruop_slice)
        item_2ord_embedding = self.i_macro_embed(batch_i_gruop_slice)
        
        user_recent_embedding = self.item_relation_embed(user_recent)
        item_recent_embedding = self.user_relation_embed(item_recent)
        
        u_1ord_trans_emb = self.i_shared_aggregator(user_1ord_embedding, item_relation_embedding.unsqueeze(1))
        u_2ord_trans_emb = self.u_shared_aggregator(user_2ord_embedding, user_relation_embedding.unsqueeze(1))
        i_1ord_trans_emb = self.u_shared_aggregator(item_1ord_embedding, user_relation_embedding.unsqueeze(1))
        i_2ord_trans_emb = self.i_shared_aggregator(item_2ord_embedding, item_relation_embedding.unsqueeze(1))
        
        user_recent_trans_emb = self.i_shared_aggregator(user_recent_embedding, item_relation_embedding.unsqueeze(1))
        item_recent_trans_emb = self.u_shared_aggregator(item_recent_embedding, user_relation_embedding.unsqueeze(1))
        cb_item_trans_emb = self.i_shared_aggregator(item_cb_embedding, item_relation_embedding.unsqueeze(1))
        cb_user_trans_emb = self.cb_user_aggregator(item_cb_embedding, user_relation_embedding.unsqueeze(1))
        
        user_1ord_ws = torch.mul(u_1ord_trans_emb, user_1ord_weight)
        user_1ord_ws = self.neighbor_aggregation_forward(self.itemAggregation, self.itemAggregationAttention, user_1ord_ws)
        user_2ord_ws = torch.mul(u_2ord_trans_emb, user_2ord_weight)
        user_2ord_ws = self.neighbor_aggregation_forward(self.userAggregation, self.userAggregationAttention, user_2ord_ws)
        
        item_1ord_ws = torch.mul(i_1ord_trans_emb, item_1ord_weight)
        item_1ord_ws = self.neighbor_aggregation_forward(self.userAggregation, self.userAggregationAttention, item_1ord_ws)
        item_2ord_ws = torch.mul(i_2ord_trans_emb, item_2ord_weight)
        item_2ord_ws = self.neighbor_aggregation_forward(self.itemAggregation, self.itemAggregationAttention, item_2ord_ws)

        # content based
        cb_item_ws = self.neighbor_aggregation_forward(self.cbItemAggregation, self.itemAggregationAttention, cb_item_trans_emb)
        cb_user_ws = self.neighbor_aggregation_forward(self.cbUserAggregation, self.itemAggregationAttention, cb_user_trans_emb)
        
        # recent
        user_recent_ws = torch.mul(user_recent_trans_emb, user_recent_mask)
        user_recent_ws = self.recent_aggregation_forward(self.itemRecentAggregation, self.itemAggregationAttention, user_recent_ws)
        item_recent_ws = torch.mul(item_recent_trans_emb, item_recent_mask)
        item_recent_ws = self.recent_aggregation_forward(self.userRecentAggregation, self.userAggregationAttention, item_recent_ws)
        
        concated = torch.hstack([
             user_embedding, 
             user_1ord_ws, user_2ord_ws, user_recent_ws,
             item_embedding,
             item_1ord_ws, item_2ord_ws, item_recent_ws,
             cb_item_ws, cb_user_ws,
        ])
        
        output = self.mlp(concated)
        output = torch.sigmoid(output)
        return output
