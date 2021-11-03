import dgl
import torch
from dgl.data import LegacyTUDataset
from dgl.data import TUDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import EBGC
import numpy as np
from infomap import Infomap
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain
import time
from sklearn.cluster import SpectralClustering


class MyDataset(Dataset):
    def __init__(self, graph_list, label_list):
        self.graph_list = graph_list
        self.label_list = label_list
        self.list_len = len(graph_list)

    def __getitem__(self, index):
        return self.graph_list[index], self.label_list[index]

    def __len__(self):
        return self.list_len


def draw_graph(g, node_labels):
    # pos = nx.spring_layout(g)
    # nx.draw_spring(g, with_labels=False)
    # nx.draw_networkx_edge_labels(g, pos, font_size=14, alpha=0.5, rotate=True)

    k = 1/(np.max(node_labels))
    val_map = {x: x*k for x in node_labels}
    values = [val_map.get(node_labels[node]) for node in g.nodes()]

    nx.draw(g, cmap=plt.get_cmap('viridis'), node_color=values, with_labels=True, font_color='white')

    plt.axis('off')
    plt.show()


def DD_preprocess(fea):
    n = len(fea)
    one_hot_fea = torch.zeros(n, 1337)  # 1336 = 798+538+1
    t1 = list(range(n))
    t2 = (fea.int()+538).view(-1).numpy()
    one_hot_fea[t1, t2] = 1
    return one_hot_fea


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs_list, labels = map(list, zip(*samples))
    graphs = list([x[0]for x in graphs_list])
    masked_graphs = list([x[1]for x in graphs_list])
    batched_graph = dgl.batch(graphs)
    batched_masked_graph = dgl.batch(masked_graphs)
    batched_graph_merge = (batched_graph, batched_masked_graph)
    return batched_graph_merge, torch.tensor(labels)


def cluster_graph(g, cluster_type):
    cluster_switch = {'SC': _clustering_sc,
                      'IM': _clustering_im,
                      'LV': _clustering_lv,
                      # 'OCG': _clustering_ocg,
                      }

    node_cluster_label = cluster_switch.get(cluster_type)(g)

    # if idx == 0:
    #     # Draw graph
    #     self.draw_graph(graph, node_cluster_label)
    #     BREAK()
    return node_cluster_label

def _clustering_sc(g):
    # Spectral Clustering
    k = 4
    nx_g = g.to_networkx().to_undirected()
    adj_mat = nx.to_numpy_matrix(nx_g)
    if len(adj_mat) < k:
        node_labels = np.array(list(range(len(adj_mat))))
    else:
        sc = SpectralClustering(k, affinity='precomputed', n_init=10)
        sc.fit(adj_mat)
        node_labels = sc.labels_
    return node_labels


def _clustering_lv(g):
    # Louvain
    node_labels = np.zeros([g.num_nodes()])
    nx_g = g.to_networkx().to_undirected()
    label_dict = community_louvain.best_partition(nx_g)
    for idx in range(len(node_labels)):
        node_labels[idx] = label_dict[idx]
    return node_labels


def _clustering_im(g):
    # InfoMap
    x_id, y_id = g.edges()
    node_labels = np.zeros([g.num_nodes()])
    im = Infomap("--flow-model undirected --silent")
    for iter_edge in range(len(x_id)):
        im.add_link(x_id[iter_edge], y_id[iter_edge])
    im.run()
    for node in im.tree:
        if node.is_leaf:
            node_labels[node.node_id] = node.module_id - 1
            # print(node.node_id, node.module_id)
    return node_labels


def preprocess(g, node_feature_type, cluster_type):
    # 1) degree / flat / attributes

    if node_feature_type == 'degree':
        g.ndata['feat'] = g.in_degrees().float().reshape(-1, 1)
    elif node_feature_type == 'flat':
        g.ndata['feat'] = torch.diag(torch.ones_like(g.in_degrees())).float()
    elif node_feature_type == 'attributes':
        g.ndata['feat'] = g.ndata['feat'].float()

    # h = g.ndata['feat']

    # 2) cluster graph
    node_cluster_label = cluster_graph(g, cluster_type)
    g.ndata['label'] = torch.tensor(node_cluster_label).reshape(-1, 1)
    g.edata['w'] = torch.ones(g.num_edges(), dtype=torch.int32)

    # 3) masked g
    label_list = node_cluster_label
    n_cluster = len(set(label_list))
    node_node_mask = torch.zeros([len(g), len(g)])
    for node_label in range(n_cluster):
        node_labels_id = np.array(np.where(label_list == node_label)).reshape(-1)
        for node_id in node_labels_id:
            node_node_mask[node_id, node_labels_id] = 1
    masked_g_adj = g.adj().to_dense()
    cluster_adj = torch.mul(masked_g_adj, node_node_mask)
    masked_g_edges = cluster_adj.nonzero().t()
    # built g
    masked_g = dgl.graph((masked_g_edges[0], masked_g_edges[1]), num_nodes=g.num_nodes())
    masked_g.ndata['feat'] = g.ndata['feat'].clone().detach()
    masked_g.ndata['label'] = g.ndata['label'].clone().detach()
    masked_g.ndata['_ID'] = g.ndata['_ID'].clone().detach()
    masked_g.edata['w'] = torch.ones(masked_g.num_edges(), dtype=torch.int32)

    return g, masked_g

def load_data(args):
    dataset = LegacyTUDataset(name=args.dataset)
    node_feature_type = args.node_feature_type
    cluster_type = args.cluster_type
    graph_list = list([])
    time_preprocess_start = time.time()
    for i, g in enumerate(dataset.graph_lists):
        print(f'Preprocess Graph {i}')
        preprocessed_g, masked_g = preprocess(g, node_feature_type, cluster_type)
        preprocessed_g = preprocessed_g.to(args.device)
        masked_g = masked_g.to(args.device)
        graph_list.append((preprocessed_g, masked_g))
        print('Preprocessing finished.')

    time_preprocess_end = time.time()
    time_preprocess_delta = time_preprocess_end - time_preprocess_start
    print("Preprocess Time: {:.4f}".format(time_preprocess_delta))
    # if node_feature_type == 'degree':
    #     fea_dim = dataset.graph_lists[0].ndata['feat'].shape[1]
    # else:
    fea_dim = dataset.graph_lists[0].ndata['feat'].shape[1]
    num_class = dataset.num_labels

    label_list = dataset.graph_labels.long().to(args.device)

    # # for Fran dataset
    # for i, j in enumerate(label_list):
    #     if j == 2:
    #         label_list[i] = 1

    # label_list = list(dataset.graph_labels.numpy().flatten())
    # t = set(label_list)
    # label_list = label_list[:100]
    return graph_list, label_list, fea_dim, num_class, time_preprocess_delta


def load_data_bac(args):
    dataset = LegacyTUDataset(name=args.dataset)
    # dataset = TUDataset(name=args.dataset)

    graph_list = list([])

    # graph_lists = dataset.graph_lists[:100]
    # for i, g in enumerate(graph_lists):
    for i, g in enumerate(dataset.graph_lists):

        g.ndata['feat'] = g.ndata['feat'].float()
        # # FRANKENSTEIN
        # g.ndata['feat'] = g.ndata['node_attr'].float()
        # g.ndata['feat'] = torch.tensor(np.ones([g.num_nodes(), 1])).float()

        # For DD dataset
        # g.ndata['feat'] = DD_preprocess(g.ndata['feat'].float())
        # t = g.ndata['feat'].float().numpy()
        node_labels = np.zeros([g.num_nodes()])

        # Louvain
        nx_g = g.to_networkx().to_undirected()
        node_label = community_louvain.best_partition(nx_g)
        for idx in range(len(node_label)):
            node_labels[idx] = node_label[idx]


        # # InfoMap Graph clustering
        # x_id, y_id = g.edges()
        # im = Infomap("--flow-model undirected --silent")
        # for iter_edge in range(len(x_id)):
        #     im.add_link(x_id[iter_edge], y_id[iter_edge])
        # im.run()
        # for node in im.tree:
        #     if node.is_leaf:
        #         node_labels[node.node_id] = node.module_id - 1
        #         # print(node.node_id, node.module_id)

        g.ndata['label'] = torch.tensor(np.array(node_labels).reshape(-1, 1))
        g.edata['w'] = torch.ones(g.num_edges(), dtype=torch.int32)

        # # Draw graph
        # demo_graph = g.to_networkx()
        # draw_graph(demo_graph, node_labels)

        # Masked graph
        label_list = node_labels
        n_cluster = len(set(label_list))
        node_node_mask = torch.zeros([len(g), len(g)])
        for node_label in range(n_cluster):
            node_labels_id = np.array(np.where(label_list == node_label)).reshape(-1)
            for node_id in node_labels_id:
                node_node_mask[node_id, node_labels_id] = 1
        masked_g_adj = g.adj().to_dense()
        cluster_adj = torch.mul(masked_g_adj, node_node_mask)
        masked_g_edges = cluster_adj.nonzero().t()

        masked_g = dgl.graph((masked_g_edges[0], masked_g_edges[1]), num_nodes=g.num_nodes())
        masked_g.ndata['feat'] = g.ndata['feat'].clone().detach()
        masked_g.ndata['label'] = g.ndata['label'].clone().detach()
        masked_g.ndata['_ID'] = g.ndata['_ID'].clone().detach()
        masked_g.edata['w'] = torch.ones(masked_g.num_edges(), dtype=torch.int32)

        g = g.to(args.device)
        masked_g = masked_g.to(args.device)
        graph_list.append((g, masked_g))
        print('Load ' + str(i) + ' graph finished.')

    fea_dim = dataset.graph_lists[0].ndata['feat'].shape[1]
    num_class = dataset.num_labels

    label_list = dataset.graph_labels

    # # for Fran dataset
    # for i, j in enumerate(label_list):
    #     if j == 2:
    #         label_list[i] = 1

    label_list = label_list.to(args.device)
    # label_list = list(dataset.graph_labels.numpy().flatten())
    # t = set(label_list)
    # label_list = label_list[:100]
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
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    return train_loader, val_loader, test_loader
