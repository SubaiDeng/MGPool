import dgl
import torch
from dgl.data import LegacyTUDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import EBGC
import numpy as np



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


# def load_data(args):
#     # TUDataset.url = 'https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets'
#
#     # # NCI1/NCI109/DD
#     # dataset = TUDataset(name=args.dataset)
#     # element_set = set([])
#     # for i in map(lambda x: set(list(x.ndata['node_labels'].numpy().flatten())), dataset.graph_lists):
#     #     element_set = element_set.union(i)
#     # element_list = list(element_set)
#     # node_label_map, node_label_id, node_one_hot = Get_Onehot_Map(element_list)
#
#     # PROTEINS
#     dataset = LegacyTUDataset(name=args.dataset)
#     t = dataset.graph_lists
#     for i in dataset.graph_lists:
#         print(i.ndata)
#         t = i.ndata['feat']
#         pass
#
#     graph_list = list([])
#     label_list = list(dataset.graph_labels.numpy().flatten())
#     for iter, g in enumerate(dataset.graph_lists):
#         node_label = list(map(lambda x: node_label_map[x.numpy()[0]], g.ndata['node_labels']))
#         g.ndata['feature'] = torch.tensor(node_label)
#         if args.device == 'cuda:0':
#             g = g.to('cuda:0')
#             g.ndata['feature'] = g.ndata['feature'].cuda()
#         graph_list.append(g)
#         # adj = g.adj().to_dense().numpy()
#         # graph_eigen_list = eigen_calculation(adj, args.pool_sizes, args.H)
#
#     fea_dim = dataset.graph_lists[0].ndata['feature'].shape[1]
#     num_class = dataset.num_labels[0]
#
#     return graph_list, label_list, fea_dim, num_class


def load_data(args):
    dataset = LegacyTUDataset(name=args.dataset)
    graph_list = list([])


    for i, g in enumerate(dataset.graph_lists):
        g.ndata['feat'] = g.ndata['feat'].float()
        # Entropy-based Graph Clustering
        nx_g = g.to_networkx()
        EBGC_cluster = EBGC.EBGC()
        cluster_result = EBGC_cluster.fit(nx_g)
        _, node_entropy_labels = np.nonzero(cluster_result)
        t = np.array(node_entropy_labels).reshape(-1, 1)
        g.ndata['label'] = torch.tensor(t)
        if args.device == 'cuda:0':
            g = g.to('cuda:0')
        graph_list.append(g)
        print('Load ' + str(i) + ' graph finished.')
    fea_dim = dataset.graph_lists[0].ndata['feat'].shape[1]
    num_class = dataset.num_labels

    label_list = dataset.graph_labels
    if args.device == 'cuda:0':
        label_list = label_list.to('cuda:0')
    # label_list = list(dataset.graph_labels.numpy().flatten())
    
    return graph_list, label_list, fea_dim, num_class


def split_dataset(args, train_val_idx, test_idx, graph_list, label_list):
    num_graphs = len(graph_list)
    num_training = int(num_graphs*0.8)
    num_val = len(train_val_idx) - num_training
    train_sampled_id, validation_sampled_idx = random_split(list(range(len(train_val_idx))), [num_training,num_val])

    train_idx = [train_val_idx[i] for i in train_sampled_id]
    validation_idx = [train_val_idx[i] for i in validation_sampled_idx]

    training_graph = [graph_list[i] for i in train_idx]
    training_label = [label_list[i] for i in train_idx]
    validation_graph = [graph_list[i] for i in validation_idx]
    validation_label = [label_list[i] for i in validation_idx]
    test_graph = [graph_list[i] for i in test_idx]
    test_label = [label_list[i] for i in test_idx]

    training_set = MyDataset(training_graph, training_label)
    validation_set = MyDataset(validation_graph, validation_label)
    test_set = MyDataset(test_graph, test_label)

    train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate)

    return train_loader, val_loader, test_loader