import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import logging

def read_csv_with_defaults(filename, col_delim1, num_edge_types):
    # Define default values for each column
    def_vals_s = [[-1], [-1]] + [['']] + [['']] * 2 * num_edge_types + [[-1]] * num_edge_types
    def_vals_t = [[-1], [-1]] + [['']] + [['']] * 2 * num_edge_types + [[-1]] * num_edge_types
    def_vals = [[-1]] + def_vals_s + def_vals_t

    def_vals_flat = [item for sublist in def_vals for item in sublist]

    dtype = {i: str if def_vals_flat[i] == [''] else str for i in range(len(def_vals_flat))}

    data = pd.read_csv(filename, delimiter=col_delim1, header=None, dtype=dtype, na_values=def_vals_flat)

    for col in data.columns:
        data[col] = data[col].apply(lambda x: [float(i) for i in x.split(',')] if isinstance(x, str) and ',' in x else float(x) if isinstance(x, str) and x != '' else x)
    
    for i, default in enumerate(def_vals_flat):
        if default == ['']:
            data[i] = data[i].fillna('')
        else:
            data[i] = data[i].fillna(default)

    return data


class CustomDataset(Dataset):
    def __init__(self, filename, num_edge_types, neg_size, nbr_size, col_delim1=";", col_delim2=",", col_delim3=":"):
        self.data = read_csv_with_defaults(filename, col_delim1, num_edge_types)
        self.num_edge_types = num_edge_types
        self.neg_size = neg_size
        self.nbr_size = nbr_size
        self.col_delim2 = col_delim2
        self.col_delim3 = col_delim3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        row = row.tolist()
        e_types = row[0]
        s_ids = row[1]
        s_types = row[2]
        s_negs = self.decode_neg_ids(row[3], self.neg_size)
        s_nbr_infos = self.decode_nbr_infos(self.nbr_size, row[4:4 + self.num_edge_types],
                                            row[4 + self.num_edge_types:4 + 2 * self.num_edge_types],
                                            row[4 + 2 * self.num_edge_types:4 + 3 * self.num_edge_types])
        t_ids = row[4 + 3 * self.num_edge_types]
        t_types = row[5 + 3 * self.num_edge_types]
        t_negs = self.decode_neg_ids(row[6 + 3 * self.num_edge_types], self.neg_size)
        base = 7 + 3 * self.num_edge_types
        t_nbr_infos = self.decode_nbr_infos(self.nbr_size, row[base:base + self.num_edge_types],
                                            row[base + self.num_edge_types:base + 2 * self.num_edge_types],
                                            row[base + 2 * self.num_edge_types:base + 3 * self.num_edge_types])
        return [e_types, [s_ids, s_types, s_negs, s_nbr_infos], [t_ids, t_types, t_negs, t_nbr_infos]]

    def decode_nbr_infos(self, nbr_size, ids_list, weights_list, flags):
        type_nbr_info = []
        for e_type in range(len(ids_list)):
            ids_str = ids_list[e_type]
            weights_str = weights_list[e_type]
            ids = self.decode_nbr_ids(ids_str, [nbr_size])
            mask = self.decode_nbr_mask(ids_str, [nbr_size])
            weights = self.decode_nbr_weights(weights_str, [nbr_size])
            type_nbr_info.append([ids, mask, weights, flags[e_type]])
        return type_nbr_info

    def decode_nbr_ids(self, str_tensor, shape):
        if not isinstance(str_tensor, str):
            str_tensor = str(str_tensor)
        str_tensor = str_tensor.replace('[', '').replace(']', '')
        ids = np.array([int(float(x)) if x != '' and not np.isnan(float(x)) else 0 for x in str_tensor.split(self.col_delim2)])
        dense_matrix = np.zeros(shape, dtype=np.int32)
        dense_matrix[:len(ids)] = ids
        return torch.tensor(dense_matrix)

    def decode_nbr_mask(self, str_tensor, shape):
        if not isinstance(str_tensor, str):
            str_tensor = str(str_tensor)
        str_tensor = str_tensor.replace('[', '').replace(']', '')
        ids = np.array([int(float(x)) if x != '' else 0 for x in str_tensor.split(self.col_delim2)])
        dense_matrix = np.zeros(shape, dtype=np.int32)
        dense_matrix[:len(ids)] = 1
        return torch.tensor(dense_matrix)

    def decode_nbr_weights(self, str_tensor, shape):
        if not isinstance(str_tensor, str):
            str_tensor = str(str_tensor)
        str_tensor = str_tensor.replace('[', '').replace(']', '')
        weights = np.array([float(x) if x != '' else 0.0 for x in str_tensor.split(self.col_delim2)])
        dense_matrix = np.zeros(shape, dtype=np.float32)
        dense_matrix[:len(weights)] = weights
        return torch.tensor(dense_matrix)

    def decode_neg_ids(self, str_tensor, neg_size):
        if not isinstance(str_tensor, str):
            str_tensor = str(str_tensor)
        str_tensor = str_tensor.replace('[', '').replace(']', '')
        ids = np.array([int(float(x)) for x in str_tensor.split(self.col_delim2) if x.strip() != ''])
        dense_matrix = np.zeros([neg_size], dtype=np.int32)
        dense_matrix[:len(ids)] = ids
        return torch.tensor(dense_matrix)


def input_fn(filename, num_edge_types, batch_size=128, neg_size=1, nbr_size=1, num_epochs=1, shuffle=True, col_delim1=";", col_delim2=",", col_delim3=":"):
    dataset = CustomDataset(filename, num_edge_types, neg_size, nbr_size, col_delim1, col_delim2, col_delim3)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, drop_last=True)
    return dataloader
