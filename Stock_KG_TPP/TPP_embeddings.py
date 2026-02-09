import os
import pickle as pkl
import argparse
import torch

from HPGE_pytorch.data_loader import CustomDataLoader 
from HPGE_pytorch.evaluation import Evaluation
from HPGE_pytorch.input import input_fn
from HPGE_pytorch.model import HHP

import sys
sys.setrecursionlimit(10000)
    

def train(train_file, model_path, embedding_file, num_epoch, learning_rate, batch_size, 
          neg_size, nbr_size, node_size, TPP_EMB, num_node_type, num_edge_type, norm_rate, datatype, device):
    
    if os.path.exists(model_path):
        os.system('rm -rf {}'.format(model_path))
        os.mkdir(model_path)
    else:
        os.makedirs(model_path)
    
    dataloader = input_fn(train_file, num_edge_type, batch_size, neg_size, nbr_size, num_epoch)
    
    train_model = HHP(learning_rate, batch_size, neg_size, nbr_size, node_size, TPP_EMB, num_node_type, num_edge_type, norm_rate, datatype,device, 'Adam').to(device)
        
    optimizer = torch.optim.Adam(train_model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    global_step = 0
    train_model.train()
    
    embedding = None
    edge_type_embed = None
        
    try:
        for epoch in range(num_epoch):
            for batch in dataloader:
                optimizer.zero_grad()
                loss = train_model(batch)
                
                loss.backward()
                
                optimizer.step()
                
                global_step += 1
                if global_step % 100 == 0:
                    print("TPP_step:", global_step, "TPP_loss:", loss.item())
                
                embedding = train_model.embedding.data.cpu().numpy()
                edge_type_embed = train_model.edge_type_embed.data.cpu().numpy()
            print("TPP_Epoch:", epoch, "TPP_Loss:", loss.item())
    finally:
        with open(embedding_file, "wb") as f:
            pkl.dump([embedding, edge_type_embed], f)
            

def generate_tpp_embeddings(graph, path, sample_type, num_edge_type, nbr_size, neg_size, task, datatype, init_data, batch_size, num_epoch, 
                            learning_rate, outpath, ratio, train_type, TPP_EMB, num_node_type, norm_rate, delim, phase, device, train_test,continous_to_edge_type, continous_to_node_type, continous_to_node, node_to_continous, node_type_to_continous, edge_type_to_continous):

    def move_to_device(batch, device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, list):
            return [move_to_device(item, device) for item in batch]
        else:
            return [move_to_device(item, device) for item in batch]
    
    filename = graph

    data_loader = CustomDataLoader(path, filename, node_type_to_continous, edge_type_to_continous, node_to_continous, train_test, nbr_size, neg_size, delim, num_edge_type, sample_type)
    
    train_file = path + 'train_{}_{}_{}_{}_{}.csv'.format(nbr_size, neg_size, task, datatype,TPP_EMB)

    if init_data == 1:
        data_loader.generate_training_dataset(train_file)
        
    model_path = outpath + "HHP_phase-{}_{}_{}_{}_{}/".format(phase, batch_size, num_epoch, datatype, TPP_EMB)
    embedding_file = model_path + "result.pkl"

    if train_type == 'train':
        train(train_file, model_path, embedding_file, num_epoch, learning_rate, batch_size, neg_size, nbr_size, data_loader.node_size, TPP_EMB, num_node_type, num_edge_type, norm_rate, datatype, device)

