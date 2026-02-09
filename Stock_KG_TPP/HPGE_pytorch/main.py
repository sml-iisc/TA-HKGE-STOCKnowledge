import os
import pickle as pkl
import argparse
import torch

from data_loader import DataLoader 
from evaluation import Evaluation
from input import input_fn
from model import HHP

import sys


sys.setrecursionlimit(10000)

parser = argparse.ArgumentParser()

parser.add_argument('--path', type=str, default='../Stock_KG_TPP/HPGE_pytorch/', help='path')
parser.add_argument('--outpath', type=str, default='../Stock_KG_TPP/HPGE_pytorch/', help='outpath')
parser.add_argument('--graph', type=str, default='../Stock_KG_TPP/HPGE_pytorch/datas/our_data/hpge_dataset_month.csv', help='graph')
parser.add_argument('--label_file', type=str, default='', help='labelfile')
parser.add_argument('--ratio', type=float, default=0.2, help='ratio')
parser.add_argument('--time_train', type=float, default=-1, help='time to split train and test')
parser.add_argument('--init_data', type=int, default=1, help='whether initialize the graph')#-1
parser.add_argument('--task', type=str, default='NC', help='task: node classification, clustering and link prediction')
parser.add_argument('--batch_size', type=int, default=1024, help='batch_size')#4096
parser.add_argument('--neg_size', type=int, default=5, help='neg_size')
parser.add_argument('--nbr_size', type=int, default=5, help='nbr_size')
parser.add_argument('--sample_type', type=str, default='important', help='cut_off or important')
parser.add_argument('--num_epoch', type=int, default=10, help='num_epoch')
parser.add_argument('--node_dim', type=int, default=128, help='node_dim')
parser.add_argument('--num_node_type', type=int, default=2, help='num_node_type') #2
parser.add_argument('--num_edge_type', type=int, default=2, help='num_edge_type') #2
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning_rate')
parser.add_argument('--norm_rate', type=float, default=0.001, help='norm_rate')
parser.add_argument('--use_gpu', type=str, default='7', help='use gpu')
parser.add_argument('--train', type=str, default="train", help='train/test')
parser.add_argument('--datasetname', type=str, default='monthly', help='daily, week, month,start,expiry')

args = parser.parse_args()


device = torch.device(f"cuda:{args.use_gpu}" if torch.cuda.is_available() and args.use_gpu != '-1' else "cpu")
print("device:",device)
def move_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    elif isinstance(batch, list):
        return [move_to_device(item, device) for item in batch]
    else:
        return [move_to_device(item, device) for item in batch]
    
def train(train_file, model_path, embedding_file, node_size):
    if os.path.exists(model_path):
        os.system('rm -rf {}'.format(model_path))
        os.mkdir(model_path)
    else:
        os.makedirs(model_path)
    
    dataloader = input_fn(train_file, args.num_edge_type, args.batch_size, args.neg_size, args.nbr_size, args.num_epoch)
    print("dataloader created")
    
    train_model = HHP(args.learning_rate, args.batch_size, args.neg_size, args.nbr_size, node_size,
                      args.node_dim, args.num_node_type, args.num_edge_type, args.norm_rate, args.datasetname,device, 'Adam').to(device)
    print("model created")
        
    optimizer = torch.optim.Adam(train_model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    
    global_step = 0
    train_model.train()
    
    embedding = None
    edge_type_embed = None
        
    try:
        for epoch in range(args.num_epoch):
            for batch in dataloader:
                optimizer.zero_grad()
                loss = train_model(batch)
                
                loss.backward()
                
                optimizer.step()
                
                global_step += 1
                if global_step % 100 == 0:
                    print("step:", global_step, "loss:", loss.item())
                
                embedding = train_model.embedding.data.cpu().numpy()
                edge_type_embed = train_model.edge_type_embed.data.cpu().numpy()
            print("Epoch:", epoch, "Loss:", loss.item())
    finally:
        with open(embedding_file, "wb") as f:
            pkl.dump([embedding, edge_type_embed], f)


def test(embedding_file, task, ratio):
    eva = Evaluation(args.label_file, embedding_file, args.task)

    if task == "NC":
        eva.lr_classification(ratio)
    elif task == 'LP':
        eva.link_preds(ratio)
    elif task == 'CL':
        eva.kmeans()

def main():
    print("args are:", args)

    filename = args.graph
    data_loader = DataLoader(args.path, filename, delim=",", sample_type=args.sample_type, num_edge_types=args.num_edge_type, neg_size=args.neg_size, nbr_size=args.nbr_size)
    print("creating train file")
    train_file = args.path + 'train_{}_{}_{}_{}.csv'.format(args.nbr_size, args.neg_size, args.task, args.datasetname)
    print("train file created")

    if args.init_data == 1:
        print("generating training dataset")
        data_loader.generate_training_dataset(train_file)
        print("training dataset generated")
        
    model_path = args.outpath + "HHP_{}_{}_{}/".format(args.batch_size, args.num_epoch, args.datasetname)
    embedding_file = model_path + "result.pkl"

    if args.train == 'train':
        print("training model")
        train(train_file, model_path, embedding_file, data_loader.node_size)
    else:
        test(embedding_file, args.task, args.ratio)

if __name__ == "__main__":
    main()
    