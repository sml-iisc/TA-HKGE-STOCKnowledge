import argparse
import torch
from torch import autograd
from models.models2 import Transformer_Ranking, Saturation
from TPP_embeddings import generate_tpp_embeddings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle



import csv
import os, sys
import math
from queue import PriorityQueue
from tkinter import N

import matplotlib as mpl
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
# from mamba_ssm import Mamba
import pickle

mpl.rcParams['figure.dpi']= 300


from random import randint
import wandb
from utils import (mean_absolute_percentage_error,
                   load_or_create_dataset_graph,
                   mean_square_error, root_mean_square_error)

from sklearn.metrics import accuracy_score, ndcg_score
from tqdm import tqdm
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import Sequential, GCNConv, JumpingKnowledge
from torch_geometric.nn import global_mean_pool
from torchmetrics.functional import retrieval_normalized_dcg


top_k_choice = [1, 5]

def parse_args():
    parser = argparse.ArgumentParser(description="Model configuration")
    parser.add_argument('--W', type=int, default=20, help='Window size')
    parser.add_argument('--T', type=int, default=20, help='Time steps')
    parser.add_argument('--LOG', type=bool, default=False, help='Logging flag')
    parser.add_argument('--D_MODEL', type=int, default=20, help='Model dimension')
    parser.add_argument('--N_HEAD', type=int, default=5, help='Number of heads')
    parser.add_argument('--DROPOUT', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--D_FF', type=int, default=1024, help='Feedforward dimension')
    parser.add_argument('--ENC_LAYERS', type=int, default=1, help='Number of encoder layers')
    parser.add_argument('--DEC_LAYERS', type=int, default=1, help='Number of decoder layers')
    parser.add_argument('--MAX_EPOCH', type=int, default=10, help='Maximum epochs')
    parser.add_argument('--USE_POS_ENCODING', type=bool, default=False, help='Use positional encoding')
    parser.add_argument('--USE_GRAPH', type=str, default='hgat', choices=['True', 'False', 'hgat', 'rgat', 'gcn'], help='Use graph type')
    parser.add_argument('--HYPER_GRAPH', type=bool, default=False, help='Use hyper graph')
    parser.add_argument('--USE_RELATION_GRAPH', type=str, default= False, choices=['gcn', 'hypergraph', 'with_sector', 'False'], help='Use relation graph type')
    parser.add_argument('--USE_KG', type=bool, default=False, help='Use knowledge graph')
    parser.add_argument('--PREDICTION_PROBLEM', type=str, default='value', help='Prediction problem type')
    parser.add_argument('--RUN', type=int, default=randint(1, 100000), help='Run identifier')
    parser.add_argument('--PLOT', type=bool, default=False, help='Plot flag')
    parser.add_argument('--MODEL_TYPE', type=str, default='', help='random')
    parser.add_argument('--ENCODER_LAYER', type=str, default='transf', choices=['gru', 'transf', 'lstm', 'mamba'], help='Encoder layer type')
    parser.add_argument('--TPP_EMB', type=int, default=128, help='TPP embedding size')
    parser.add_argument('--tau_choices', type=int, nargs='+', default=[1, 5, 20], help='Tau choices')
    parser.add_argument('--tau_positions', type=int, nargs='+', default=[1, 5, 20], help='Tau positions')
    parser.add_argument('--FAST', type=bool, default=False, help='Fast flag')
    parser.add_argument('--risk_free_returns_in_phase', type=float, nargs='+', default=[0.09, 0.05, 0.07, 0.04, 0.07, 0.117, 0.3016, 0.4279, 0.442, 0.6843, 1.0689, 1.3382, 1.9212, 2.2711, 2.5158, 2.2441, 1.6929, 0.6449, 0.1343, 0.0824, 0.0455, 0.0830, 0.9286, 3.1276], help='Risk free returns in phase')
    parser.add_argument('--risk_free_returns_in_phase_nifty', type=float, nargs='+', default=[8.4839, 9.0665, 8.8801, 8.6328, 8.2422, 7.7184, 7.2458, 7.0708, 6.6413, 6.1594, 6.2695, 6.2345, 6.4746, 7.0454, 6.7617, 6.1305, 5.2932, 4.5420, 3.4372, 3.4249, 3.5782, 3.6773, 4.4368, 6.0270], help='Risk free returns in phase nifty')
    parser.add_argument('--MODEL', type=str, default='ours', help='Model name')
    parser.add_argument('--REL_EMB', type=int, default=16, help='Relation embedding size')
    
    parser.add_argument('--label_file', type=str, default='', help='Label file')
    parser.add_argument('--ratio', type=float, default=0.2, help='Ratio')
    parser.add_argument('--time_train', type=float, default=-1, help='Time to split train and test')
    parser.add_argument('--init_data', type=int, default=1, help='Initialize the graph')
    parser.add_argument('--task', type=str, default='LP', help='Task type')
    parser.add_argument('--tpp_batch_size', type=int, default=4096, help='TPP batch size')
    parser.add_argument('--neg_size', type=int, default=5, help='Negative size')
    parser.add_argument('--nbr_size', type=int, default=5, help='Neighbor size')
    parser.add_argument('--sample_type', type=str, default='important', help='Sample type')
    parser.add_argument('--tpp_num_epoch', type=int, default=10, help='TPP number of epochs')
    parser.add_argument('--num_node_type', type=int, default=15, help='Number of node types')
    parser.add_argument('--num_edge_type', type=int, default=54, help='Number of edge types')
    parser.add_argument('--tpp_learning_rate', type=float, default=0.0001, help='TPP learning rate')
    parser.add_argument('--norm_rate', type=float, default=0.001, help='Normalization rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index')
    parser.add_argument('--train', type=str, default='train', help='Train or test')
    parser.add_argument('--datatype', type=str, default='month', help='Data type')
    parser.add_argument('--train_test', type=int, default=-1, help='Train test split')
    parser.add_argument('--delim', type=str, default=',', help='Delimiter')
    
    parser.add_argument('--INDEX', type=str, default='nifty500', choices=['nasdaq100', 'sp500', 'nifty500'], help='Index')
    parser.add_argument('--data_file_path', type=str, help='Data file path')
    parser.add_argument('--relation_graph_path', type=str, help='Relation graph path')
    parser.add_argument('--relation_hypergraph_path', type=str, help='Relation hypergraph path')
    parser.add_argument('--kg_file_path', type=str, help='Knowledge graph file path')
    parser.add_argument('--nifty_temporal_kg_path', type=str, help='Nifty temporal knowledge graph path')
    parser.add_argument('--node_tensor_path', type=str, help='Node tensor path')
    parser.add_argument('--hpge_data_path', type=str, help='HPGE data path')
    parser.add_argument('--path', type=str, help='Path')
    parser.add_argument('--outpath', type=str, help='Output path')

    args = parser.parse_args()

    # Set default paths based on other arguments
    args.data_file_path = '../Phase-Stock-KG/data/pickle/' + args.INDEX + '/full_graph_data_correct-P25-W' + str(args.W) + '-T' + str(args.T) + '_' + args.PREDICTION_PROBLEM + '.pkl'
    args.relation_graph_path = '../Phase-Stock-KG/kg/profile_and_relationship/wikidata/relation_graph.pkl'
    args.relation_hypergraph_path = '../Phase-Stock-KG/kg/profile_and_relationship/wikidata/relation_hypergraph.pkl'
    args.kg_file_path = '../Phase-Stock-KG/kg/tkg_create/temporal_kg.pkl'
    args.nifty_temporal_kg_path = '../Phase-Stock-KG/kg/tkg_create/temporal_kg_nifty.pkl'
    args.node_tensor_path = '../Phase-Stock-KG/kg/tkg_create/node_tensor_usa.pt'
    args.hpge_data_path = '../Stock_KG_TPP/HPGE_pytorch/datas/our_data/hpge_dataset_' + args.datatype + '.csv'
    args.path = '../Stock_KG_TPP/TPP/train_files/' + args.INDEX + '/' + args.datatype + '/' 
    args.outpath = '../Stock_KG_TPP/TPP/HPP_files/' + args.INDEX + '/' + args.datatype + '/' 
    return args

def Calculate_AIRR(IRR, delta):
    ROI = IRR / 100  # Convert IRR to ROI
    AIRR = (1 + ROI) ** (252/delta) - 1  # Apply the formula for AIRR
    return AIRR

def rank_loss(prediction, ground_truth):
    all_one = torch.ones(prediction.shape[0], 1, dtype=torch.float32).to(device)
    prediction = prediction.unsqueeze(dim=1)
    ground_truth = ground_truth.unsqueeze(dim=1)
    return_ratio = prediction 
    true_return_ratio = ground_truth - 1

    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),                  # C x C
        all_one @ return_ratio.t()                   # C x C
    )
    gt_pw_dif = torch.sub(
        all_one @ true_return_ratio.t(),
        true_return_ratio @ all_one.t()
    )

    rank_loss = torch.mean(
        F.relu(-1*pre_pw_dif * gt_pw_dif )
    )
   
    return rank_loss 

def evaluate(prediction, ground_truth, bestret, worstret, K):
    return_ratio = prediction - 1
    true_return_ratio = ground_truth - 1
    bestret = bestret - 1
    worstret = worstret - 1

    target_obtained_return_ratio = torch.topk(true_return_ratio, k=K, dim=0)[0].mean()

    topk_predicted = torch.topk(return_ratio, k=K, dim=0)[1]

    obtained_return_ratio = true_return_ratio[topk_predicted].mean()
    best_return_ratio = bestret[topk_predicted].mean()
    worst_return_ratio = worstret[topk_predicted].mean()

    a_cat_b, counts = torch.cat([torch.topk(return_ratio.squeeze(), k=K, dim=0)[1], torch.topk(true_return_ratio.squeeze(), k=K, dim=0)[1]]).unique(return_counts=True)
    accuracy = a_cat_b[torch.where(counts.gt(1))].shape[0] / K

    return obtained_return_ratio, target_obtained_return_ratio, accuracy, best_return_ratio, worst_return_ratio


top_k_choice = [1, 5]

def calculate_ndcg(predict, true, k):
    tt = torch.topk(true, k, dim=0)[1]
    rel_score = torch.arange(k, 0, -1).to(device)
    true_rel = torch.zeros_like(predict).long()
    true_rel[tt] = rel_score
    return retrieval_normalized_dcg(predict, true_rel)

def approxNDCGLoss(y_pred, y_true, eps=1e-10, padded_value_indicator=-1, alpha=1.):
    """
    Loss based on approximate NDCG introduced in "A General Approximation Framework for Direct Optimization of
    Information Retrieval Measures". Please note that this method does not implement any kind of truncation.
    :param y_pred: predictions from the model, shape [batch_size, slate_length]
    :param y_true: ground truth labels, shape [batch_size, slate_length]
    :param eps: epsilon value, used for numerical stability
    :param padded_value_indicator: an indicator of the y_true index containing a padded item, e.g. -1
    :param alpha: score difference weight used in the sigmoid function
    :return: loss value, a torch.Tensor
    """
    # shuffle for randomised tie resolution
    random_indices = torch.randperm(y_pred.shape[-1])
    y_pred_shuffled = y_pred.unsqueeze(dim=0)[:, random_indices]
    y_true_shuffled = y_true.unsqueeze(dim=0)[:, random_indices]

    y_true_sorted, indices = y_true_shuffled.sort(descending=True, dim=-1)

    mask = y_true_sorted == padded_value_indicator

    preds_sorted_by_true = torch.gather(y_pred_shuffled, dim=1, index=indices)
    preds_sorted_by_true[mask] = float("-inf")

    max_pred_values, _ = preds_sorted_by_true.max(dim=1, keepdim=True)

    preds_sorted_by_true_minus_max = preds_sorted_by_true - max_pred_values

    cumsums = torch.cumsum(preds_sorted_by_true_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])

    observation_loss = torch.log(cumsums + eps) - preds_sorted_by_true_minus_max

    observation_loss[mask] = 0.0

    return torch.mean(torch.sum(observation_loss, dim=1))

def approx_rank(logits):
    """_summary_

    Args:
        logits (_type_): A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.

    Returns:
        _type_: A `Tensor` of ranks with the same shape as logits.
    """
    list_size = logits.shape[1]
    x = logits.unsqueeze(2).repeat(1, 1, list_size)
    y = logits.unsqueeze(1).repeat(1, list_size, 1)
    rank = torch.sigmoid(x - y)
    rank = torch.sum(rank, dim=-1) #+ 0.5
    return rank

def approx_ndcg_loss(logits, labels):
    """_summary_

    Args:
        logits (_type_): A `Tensor` with shape [batch_size, list_size]. Each value is the
      ranking score of the corresponding item.
        labels (_type_): A `Tensor` with shape [batch_size, list_size]. Each value is the
      relevance label of the corresponding item.

    Returns:
        _type_: A `Tensor` of ndcg loss with shape [batch_size].
    """
    rank = approx_rank(logits)
    #print("logits", torch.topk(logits, k=5, dim=-1), torch.topk(rank, k=5, dim=-1))
    return - retrieval_normalized_dcg(rank, labels)

def main():
    args = parse_args()

    print("save_path:", args.data_file_path)
    dataset, company_to_id, graph, hyper_data = load_or_create_dataset_graph(INDEX=args.INDEX, W=args.W, T=args.T, save_path=args.data_file_path, problem=args.PREDICTION_PROBLEM, fast=args.FAST)

    num_nodes = len(company_to_id.keys())
    inverse_company_to_id = {v: k for k, v in company_to_id.items()}

    if not args.HYPER_GRAPH:
        graph_nodes_batch = torch.zeros(graph.x.shape[0]).to(device)
        graph = graph.to(device)
        graph_data = {
            'x': graph.x,
            'edge_list': graph.edge_index,
            'batch': graph_nodes_batch
        }
    else:
        x, hyperedge_index = hyper_data['x'].to(device), hyper_data['hyperedge_index'].to(device)
        print("Graph details: ", x.shape, hyperedge_index.shape)
        graph_data = {
            'x': x,
            'hyperedge_index': hyperedge_index
        }

    relation_graph = None
    key = args.INDEX if args.INDEX == 'sp500' else args.INDEX[:-3]
    if args.USE_RELATION_GRAPH == 'gcn':
        with open(args.relation_graph_path, 'rb') as f:
            relation_graph = pickle.load(f)[key]
        relation_graph = relation_graph.to(device)
    elif args.USE_RELATION_GRAPH == 'hypergraph' or args.USE_RELATION_GRAPH == 'with_sector':
        with open(args.relation_hypergraph_path, 'rb') as f:
            relation_graph = pickle.load(f)[key]
        relation_graph = relation_graph.to(device)

    kg_file_name = args.kg_file_path
    if args.INDEX == 'nifty500':
        kg_file_name = args.nifty_temporal_kg_path
    with open(kg_file_name, 'rb') as f:
        pkl_file = pickle.load(f)
        if "nasdaq" in args.INDEX:
            kg_map = pkl_file['nasdaq_map']
        elif "sp" in args.INDEX:
            kg_map = pkl_file['sp_map']
        elif "nifty" in args.INDEX:
            kg_map = pkl_file['nifty_map']

    if args.USE_KG:
        relation_kg = None
    else:
        relation_kg = None

    df_hpge = pd.read_csv(args.hpge_data_path, header=None)
    if (args.INDEX == 'nasdaq100' or args.INDEX == 'sp500'):
        df_hpge = pd.read_csv('../Stock_KG_TPP/HPGE_pytorch/datas/our_data/hpge_dataset_month_nasdaq.csv', header=None)
    
    def predict(loader, desc, kg_map, risk_free_ret,inverse_company_to_id, continous_to_node, continous_to_node_type, continous_to_edge_type):
        epoch_loss = 0

        # RR is actually RoI (Return on Investment)

        rr, true_rr, accuracy, best_rr, worst_rr = torch.zeros(4).to(device), torch.zeros(4).to(device), torch.zeros(4).to(device), torch.zeros(4).to(device), torch.zeros(4).to(device)
        ndcg, sharpe_ratio = torch.zeros(4).to(device), torch.zeros(4).to(device)
        sharpe = [[], [], []]
        yb_store, yhat_store, yb_store2 = [], [], []

        test_rr_list = [[], [], []]
        for xb, yb, ticker_ids ,tkg, bestret, worstret in loader:
            head, relation, tail, ts = tkg
            head, relation, tail, ts, kg_map = head.to(device), relation.to(device), tail.to(device), ts.to(device), kg_map.to(device)

            tkg = (head, relation, tail, ts, kg_map)

            xb      = xb.to(device)
            yb      = yb.to(device)
            ticker_ids = ticker_ids.to(device)
            bestret = bestret.to(device)
            worstret = worstret.to(device)
            
            y_hat, kg_loss = model(xb, ticker_ids, phase, node_to_continous, continous_to_node_type, continous_to_edge_type, yb, graph_data, relation_kg, tkg, relation_graph)

            y_hat = y_hat.squeeze()
            y_hat = F.softmax(y_hat.squeeze(), dim = 0)
            true_return_ratio = yb.squeeze() 
            
            true_top5 = torch.topk(true_return_ratio, k=5, dim=0)
            zeros = torch.zeros_like(y_hat)
            zeros[true_top5[1]] = 1

            neg_ret_target_mask = true_return_ratio >= 1
            neg_ret_target = torch.zeros_like(y_hat)
            neg_ret_target[neg_ret_target_mask] = 1
            
            loss = F.binary_cross_entropy(y_hat, zeros) 
            loss += F.binary_cross_entropy(y_hat, neg_ret_target)  
            
            tt = torch.argsort(true_return_ratio, descending=True)
            rel_score = torch.arange(xb.shape[0], 0, -1).to(device)
            true_rel = torch.zeros_like(y_hat).long()
            true_rel[tt] = rel_score
            loss += approx_ndcg_loss(y_hat.unsqueeze(dim=0), true_rel.unsqueeze(dim=0)) 
            epoch_loss += loss.item()
            
            if args.USE_KG or args.USE_GRAPH == 'hgat':
                loss += kg_loss 

            if model.training:
                loss.backward()
                clipped_gradient_norm=nn.utils.clip_grad_norm_(model.parameters(),4.0)
                opt_c.step()
                opt_c.zero_grad()

            for index, k in enumerate(top_k_choice):
                crr, ctrr, cacc, cbr, cwr = evaluate(y_hat[:-1], true_return_ratio, bestret, worstret, k)
                true_rr[index] += ctrr
                rr[index] += crr
                sharpe[index].append(float(crr))
                accuracy[index] += cacc
                best_rr[index] += cbr
                worst_rr[index] += cwr
                ndcg[index] += calculate_ndcg(y_hat, true_return_ratio, k)

                if desc == "TESTING":
                    test_rr_list[index].append(float(crr))

            ndcg[3] += calculate_ndcg(y_hat, true_return_ratio, 25)

            if desc == "TESTING":
                plot = torch.topk(y_hat, k=5, dim=0)[1]
                plot2 = true_top5[1]
                plot1_list = []
                plot2_list = []
                for i in range(5):
                    plot1_list.append(inverse_company_to_id[plot[i].item()])
                    plot2_list.append(inverse_company_to_id[plot2[i].item()])
                print("Predicted: ", plot1_list, "Actual: ", plot2_list)

        epoch_loss /= len(loader)
        rr /= len(loader) 
        true_rr /= len(loader) 
        accuracy /= len(loader)
        ndcg /= len(loader)
        best_rr /= len(loader)
        worst_rr /= len(loader)

        tau = [252, 252/20]
        for i in range(len(top_k_choice)):
            mean = sum(sharpe[i]) / len(sharpe[i]) * 100
            variance = sum([((x*100 - mean) ** 2) for x in sharpe[i]]) / len(sharpe[i])
            res = (variance*tau[i]) ** 0.5
            sharpe_ratio[i] = (mean*tau[i] - (risk_free_ret)) / (res+0.00001)

        if desc == "TESTING":
            print(test_rr_list)
        print("\n[{0}] Epoch: {1} Loss: {2} NDCG {3}".format(desc, ep+1, epoch_loss, ndcg[3]))
        for index, k in enumerate(top_k_choice):
            print("[{0}] Top {3} NDCG: {5} Return Ratio: {1} True Return Ratio: {2} Accuracy: {4}".format(desc, rr[index], true_rr[index], k, accuracy[index], ndcg[index]))
            print("[{0}] Best RR: {1} Worst RR: {2} Sharpe Ratio: {3}".format(desc, best_rr[index], worst_rr[index], sharpe_ratio[index]))
        log = {'MSE': epoch_loss}

        if args.LOG:
            wandb.log(log)
        PLOT = False
        if PLOT:
            mpl.rcParams['figure.dpi']= 300
            plt.scatter(np.array(yb_store), np.array(yb_store2), c=np.array(yhat_store))
            plt.savefig("/plots/saturation/E"+str(ep)+"-T"+str(tau)+ ".png")
            plt.close()

        return epoch_loss, rr, true_rr, accuracy, ndcg, best_rr, worst_rr, sharpe_ratio

    for tau in args.tau_choices:
        tau_pos = args.tau_positions.index(tau)
        print("Tau: ", tau, "Tau Position: ", tau_pos)

        start_time, train_begin = 0, 0
        test_mean_rr, test_mean_trr, test_mean_err, test_mean_rrr = [[], [], []], [[], [], []], [[], [], []], [[], [], []]
        test_mean_ndcg, test_mean_acc = [[], [], [], []], [[], [], []]
        test_mean_brr, test_mean_wrr, test_mean_sharpe = torch.zeros(4).to(device), torch.zeros(4).to(device), [[], [], []]

        def collate_fn(instn):
            tkg = instn[0][1]
            instn = instn[0][0]
            ticker_ids = torch.Tensor(np.array([x[5] for x in instn])).long()
            df = torch.Tensor(np.array([x[0] for x in instn])).unsqueeze(dim=2)
            for i in range(1, 5):
                df1 = torch.Tensor(np.array([x[i] for x in instn])).unsqueeze(dim=2)
                df = torch.cat((df, df1), dim=2)
            min_val = df.min()
            max_val = df.max()
            normalized_tensor = 2 * (df - min_val) / (max_val - min_val) - 1
            target = torch.Tensor(np.array([x[7][tau_pos] for x in instn]))
            best_case, worst_case = torch.Tensor(np.array([x[11][tau_pos+1] for x in instn])), torch.Tensor(np.array([x[10][tau_pos+1] for x in instn]))
            best_case = best_case / torch.Tensor(np.array([x[10][0] for x in instn]))
            worst_case = worst_case / torch.Tensor(np.array([x[11][0] for x in instn]))
            return (normalized_tensor, target, ticker_ids, tkg, best_case, worst_case)

        for phase in range(1, 25):
            print("Phase: ", phase)
            train_loader = DataLoader(dataset[train_begin:start_time+400], 1, shuffle=True, collate_fn=collate_fn, num_workers=1)
            val_loader = DataLoader(dataset[start_time+400:start_time+450], 1, shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(dataset[start_time+450:start_time+550], 1, shuffle=False, collate_fn=collate_fn)

            all_dates = []
            for data_entry in dataset[train_begin:start_time+400]:
                cur_data, _ = data_entry
                for company_data in cur_data:
                    dates = pd.to_datetime(company_data[6])
                    all_dates.extend(dates)

            train_start_date = min(all_dates)
            train_end_date = max(all_dates)
            train_start_time_ms = int(train_start_date.timestamp() * 1000)
            train_end_time_ms = int(train_end_date.timestamp() * 1000)
            
            if args.INDEX == "nifty500":
                filtered_df = df_hpge[(df_hpge[5] >= train_start_time_ms) & (df_hpge[5] <= train_end_time_ms) | ((df_hpge[0] == 393) | (df_hpge[2] == 393))]
                graph_path = '../Stock_KG_TPP/TPP/data_files/' + args.INDEX + '/' + args.datatype + '/' + "/hpge_dataset_" + str(tau) + "_" + str(phase) + ".csv"
                num_edge_type = 53
                num_node_type = 14
            elif args.INDEX == "nasdaq100" or args.INDEX == "sp500":
                filtered_df = df_hpge[(df_hpge[5] >= train_start_time_ms) & (df_hpge[5] <= train_end_time_ms)]
                graph_path = '../Stock_KG_TPP/TPP/data_files/nasdaq100/' + args.datatype + '/' + "/hpge_dataset_" + str(tau) + "_" + str(phase) + ".csv"
                num_edge_type = 56
                num_node_type = 12

            graph = graph_path
            if not os.path.exists(graph):
                filtered_df.to_csv(graph, index=False, header=False)

            node_types = set()
            nodes = set()
            relations = set()
            with open(graph, 'r') as f:
                for line in f:
                    line = line.strip().split(',')
                    nodes.add(int(line[0]))
                    nodes.add(int(line[2]))
                    node_types.add(int(line[1]))
                    node_types.add(int(line[3]))
                    relations.add(int(line[4]))
            
            # finding the num_node_types and num_edge_types from graph
            df = pd.read_csv(graph)
            head = df.iloc[:, 0]
            head_type_col = df.iloc[:, 1]
            tail = df.iloc[:, 2]
            tail_type_col = df.iloc[:, 3]
            relation_col = df.iloc[:, 4]
            
            # Find the number of unique node types
            unique_node_types = pd.concat([head_type_col, tail_type_col]).unique()
            
            # Find the number of unique edge types
            unique_edge_types = relation_col.unique()
            
            # Create mappings for node types
            node_type_to_continous = {node_type: idx for idx, node_type in enumerate(unique_node_types)}
            continous_to_node_type = {idx: node_type for node_type, idx in node_type_to_continous.items()}
            
            # Create mappings for edge types
            edge_type_to_continous = {edge_type: idx for idx, edge_type in enumerate(unique_edge_types)}
            continous_to_edge_type = {idx: edge_type for edge_type, idx in edge_type_to_continous.items()}
            
            #create mappings for the node_ids
            node_ids = pd.concat([head, tail]).unique()
            node_to_continous = {node_id: idx for idx, node_id in enumerate(node_ids)}
            continous_to_node = {idx: node_id for node_id, idx in node_to_continous.items()}
            
            result_path = '../Stock_KG_TPP/TPP/HPP_files/nasdaq100/{}/HHP_phase-{}_{}_{}_{}_{}/'.format(args.datatype, phase, args.tpp_batch_size, args.tpp_num_epoch, args.datatype, args.TPP_EMB) + '/result.pkl'
            if not os.path.exists(result_path):
                generate_tpp_embeddings(graph, args.path, args.sample_type, num_edge_type, args.nbr_size, args.neg_size, args.task, args.datatype, 
                                    args.init_data, args.tpp_batch_size, args.tpp_num_epoch, args.tpp_learning_rate, args.outpath, args.ratio, args.train, 
                                    args.TPP_EMB, num_node_type, args.norm_rate,args.delim, phase, device, args.train_test, continous_to_edge_type, continous_to_node_type, continous_to_node, node_to_continous, node_type_to_continous, edge_type_to_continous)

            start_time += 100
            if start_time >= 300:
                train_begin += 100

            node_type = torch.load(args.node_tensor_path, weights_only=True)
            config = {
                'entity_total': 6500,
                'relation_total': 57,
                'L1_flag': False,
                'node_type': node_type,
                'num_node_type': 14
            }

            
            model = Transformer_Ranking(args.W, args.T, args.D_MODEL, args.N_HEAD, args.ENC_LAYERS, args.DEC_LAYERS, args.D_FF, args.REL_EMB,args.DROPOUT, args.TPP_EMB, args.INDEX, args.tpp_batch_size, args.tpp_num_epoch, args.datatype, args.USE_POS_ENCODING, args.USE_GRAPH, args.HYPER_GRAPH, args.USE_KG, num_nodes, config, args.ENCODER_LAYER, args.USE_RELATION_GRAPH)

            if phase == 1:
                print(model)
            model.to(device)

            opt_c = torch.optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-08)

            prev_val_loss, best_val_loss = float("infinity"), float("infinity")
            val_loss_history = []

            for ep in range(10):
                print("Epoch: " + str(ep+1))
                model.train()
                train_epoch_loss, rr, trr, accuracy, ndcg, bestr, worstr, sharpe = predict(train_loader, "TRAINING", kg_map, args.risk_free_returns_in_phase[phase-1], inverse_company_to_id, continous_to_node, continous_to_node_type, continous_to_edge_type)
                model.eval()
                with torch.no_grad():
                    val_epoch_loss, rr, trr, accuracy, ndcg, bestr, worstr, sharpe = predict(val_loader, "VALIDATION", kg_map, args.risk_free_returns_in_phase[phase-1], inverse_company_to_id, continous_to_node, continous_to_node_type, continous_to_edge_type)

                if prev_val_loss < val_epoch_loss:
                    val_loss_history.append(1)
                else:
                    val_loss_history.append(0)
                prev_val_loss = val_epoch_loss
                
                if best_val_loss >= val_epoch_loss: 
                    print("Saving Model")

                    torch.save(model.state_dict(), "models/saved_models/best_model_"+ args.INDEX+str(args.W)+"_"+str(args.T)+"_"+str(args.RUN)+".pt")
                    best_val_loss = val_epoch_loss
            
                if ep > 7 and sum(val_loss_history[-3:]) == 3:
                    print("Early Stopping")
                    break

            if args.MODEL_TYPE != 'random':
                model.load_state_dict(torch.load("models/saved_models/best_model_"+ args.INDEX+str(args.W)+"_"+str(args.T)+"_"+str(args.RUN)+".pt"))

            model.eval()
            with torch.no_grad():
                test_epoch_loss, rr, trr, accuracy, ndcg, bestr, worstr, sharpe  = predict(test_loader, "TESTING", kg_map, args.risk_free_returns_in_phase[phase-1], inverse_company_to_id, continous_to_node, continous_to_node_type, continous_to_edge_type)
                for i in range(len(top_k_choice)):
                    test_mean_rr[i].append(rr[i].item())
                    test_mean_trr[i].append(trr[i].item())
                    test_mean_ndcg[i].append(ndcg[i].item())
                    test_mean_acc[i].append(accuracy[i].item())
                    test_mean_sharpe[i].append(sharpe[i].item())
                test_mean_ndcg[3].append(ndcg[3].item())
                test_mean_brr += bestr
                test_mean_wrr += worstr
                print("NDCG: mean {0} std {1}".format(sum(test_mean_ndcg[3])/phase, np.std(np.array(test_mean_ndcg[3]))))
                for index, k in enumerate(top_k_choice):
                    RR = sum(test_mean_rr[index])/phase
                    IRR = RR*100
                    AIRR = Calculate_AIRR(IRR, tau)
                    print("[Mean - {0}] Top {3} NDCG: {5} Return Ratio: {1} True Return Ratio: {2} Accuracy: {4}".format("TESTING", sum(test_mean_rr[index])/phase, sum(test_mean_trr[index])/phase, k, sum(test_mean_acc[index])/phase, sum(test_mean_ndcg[index])/phase))
                    print("[Mean - {0}] Investment Return Ratio {1} Annualized Investment Return Ratio {2}".format("TESTING", IRR, AIRR))
                    print("[Mean - {0}] Best Return Ratio: {1} Worst Return Ratio: {2} Sharpe Ratio: {3} ".format("TESTING", test_mean_brr[index]/phase, test_mean_wrr[index]/phase, sum(test_mean_sharpe[index])/phase))
                    print("[STD - {0}] Top {3} NDCG: {5} Return Ratio: {1} True Return Ratio: {2} Accuracy: {4} Sharpe: {6}".format("TESTING", np.std(np.array(test_mean_rr[index])), np.std(np.array(test_mean_trr[index])), k, np.std(np.array(test_mean_acc[index])), np.std(np.array(test_mean_ndcg[index])), np.std(np.array(test_mean_sharpe[index]))))

            if args.LOG:
                wandb.save('model.py')
        print("Tau: ", tau)
        for index, k in enumerate(top_k_choice):
            print("[Result Copy] Top {1} {2} {3} {4}".format("TESTING", k, sum(test_mean_ndcg[index])/phase, sum(test_mean_acc[index])/phase, sum(test_mean_rr[index])/phase))
            print("[Result Copy] Top {1} {2} {3} {4}".format("TESTING", k, sum(test_mean_sharpe[index])/phase, test_mean_brr[index]/phase, test_mean_wrr[index]/phase))
            
            
if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    main()
