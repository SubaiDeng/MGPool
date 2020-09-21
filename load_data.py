import pickle
import dgl
import torch
from dgl.data import TUDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.cluster import SpectralClustering as SC
from scipy.sparse.linalg import eigsh
import math



import os

class MyDataset(Dataset):
    def __init__(self, graph_list, label_list):
        self.graph_list = graph_list
        self.label_list = label_list
        self.list_len = len(graph_list)

    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index]

    def __len__(self):
        return self.list_len



def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)


def load_data(args):
    # TUDataset.url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets'
    dataset = TUDataset(name=args.dataset)

    element_set = set([])
    for i in map(lambda x: set(list(x.ndata['node_labels'].numpy().flatten())), dataset.graph_lists):
        element_set = element_set.union(i)
    element_list = list(element_set)
    node_label_map, node_label_id, node_one_hot = Get_Onehot_Map(element_list)

    graph_list = list([])
    label_list = list(dataset.graph_labels.numpy().flatten())
    for iter, g in enumerate(dataset.graph_lists):
        node_label = list(map(lambda x: node_label_map[x.numpy()[0]], g.ndata['node_labels']))
        g.ndata['feature'] = torch.tensor(node_label)
        if args.device == 'cuda':
            g = g.to('cuda:0')
            g.ndata['feature'] = g.ndata['feature'].cuda()
        graph_list.append(g)
        # adj = g.adj().to_dense().numpy()
        # graph_eigen_list = eigen_calculation(adj, args.pool_sizes, args.H)

    fea_dim = dataset.graph_lists[0].ndata['feature'].shape[1]
    num_class = dataset.num_labels[0]

    # return dataset, num_class, fea_dim
    num_graphs = len(dataset)
    num_training = int(num_graphs * 0.9)
    # num_val = int(num_graphs * 0.1)
    num_test = num_graphs - num_training
    train_idx, test_idx = random_split(list(range(num_graphs)), [num_training, num_test])

    training_graph = [graph_list[i] for i in train_idx]
    training_label = [label_list[i] for i in train_idx]
    # validation_graph = [graph_list[i] for i in validation_idx]
    # validation_label = [label_list[i] for i in validation_idx]
    test_graph = [graph_list[i] for i in test_idx]
    test_label = [label_list[i] for i in test_idx]

    training_set = MyDataset(training_graph, training_label)
    # validation_set = MyDataset(validation_graph, validation_label)
    test_set = MyDataset(test_graph, test_label)

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    # val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate)

    return train_loader,  test_loader,  fea_dim, num_class


def Get_Onehot_Map(element_list):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(np.array(element_list))
    print(integer_encoded)
    # binary encoder
    onehot_encoded = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoded.fit_transform(integer_encoded)
    mapp = dict()
    for label, encoder in zip(element_list, onehot_encoded.tolist()):
        mapp[label] = np.array(encoder, dtype=np.float32)

    return mapp, integer_encoded.reshape(-1, ), onehot_encoded

#
# def split_subgraph(adj, node_clustering):
#     subgraph_list = list([])
#     label_list = list(set(node_clustering))
#     for label in label_list:
#         idx = [i for i in range(len(node_clustering)) if node_clustering[i] == label]
#         subgraph = adj[np.ix_(idx, idx)]
#         adj[np.ix_(idx, idx)] = 0
#         subgraph_list.append(subgraph)
#
#     return subgraph_list, label_list, adj
#
#
# def to_laplacian(adj):
#     d = np.sum(adj, 0)
#     l = np.diag(d) - adj
#
#     return l
#
#
# def subgraph_aggregate(adj_ext, node_clustering_labels, subgraph_label_list):
#     S = np.zeros((len(node_clustering_labels), len(subgraph_label_list)))
#
#     for iter, label in enumerate(subgraph_label_list):
#         idx = [i for i in range(len(node_clustering_labels)) if node_clustering_labels[i] == label]
#         S[np.ix_(idx), iter] = 1
#
#     adj_coar = np.dot(np.dot(S.T, adj_ext), S)
#
#     return adj_coar
#
# def eigen_calculation(adj, cluster_batch_list, H):
#
#     graph_pooling_eigen_list = ([])
#     cluster_batch_list = [int(x) for x in cluster_batch_list.split('-')]
#     for iter in range(len(cluster_batch_list)):
#         num_nodes = adj.shape[0]
#         num_cluster = math.ceil(num_nodes/cluster_batch_list[iter])
#         node_clustering = SC(n_clusters=num_cluster,
#                         assign_labels="discretize",
#                         affinity='precomputed',
#                         random_state=0).fit(adj)
#
#         node_clustering_labels = node_clustering.labels_
#         subgraph_list, subgraph_label_list, adj_ext = split_subgraph(adj, node_clustering_labels)
#         subgraph_eigen_list = ([])
#         for subgraph_adj in subgraph_list:
#             laplacian = to_laplacian(subgraph_adj)
#             vals, vecs = eigsh(laplacian, k=H)
#             subgraph_eigen_list.append(vecs)
#         adj_coar = subgraph_aggregate(adj_ext, node_clustering_labels, subgraph_label_list)
#
#         adj = adj_coar
#         graph_pooling_eigen_list.append(subgraph_eigen_list)
#
#     return graph_pooling_eigen_list
#
#
# def node_attach_graph():
#     pass