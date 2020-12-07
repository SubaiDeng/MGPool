import dgl
import torch
from dgl.data import LegacyTUDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
import EBGC
import numpy as np
from infomap import Infomap
import networkx as nx
import matplotlib.pyplot as plt



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

def load_data(args):
    dataset = LegacyTUDataset(name=args.dataset)
    graph_list = list([])

    # graph_lists = dataset.graph_lists[:100]
    # for i, g in enumerate(graph_lists):
    for i, g in enumerate(dataset.graph_lists):
        g.ndata['feat'] = g.ndata['feat'].float()
        t = g.ndata['feat'].float().numpy()

        # # Entropy-based Graph Clustering
        # nx_g = g.to_networkx()
        # EBGC_cluster = EBGC.EBGC()
        # cluster_result = EBGC_cluster.fit(nx_g)
        # _, node_entropy_labels = np.nonzero(cluster_result)
        # t = np.array(node_entropy_labels).reshape(-1, 1)
        # g.ndata['label'] = torch.tensor

        # InfoMap Graph clustering
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

        g.ndata['label'] = torch.tensor(np.array(node_labels).reshape(-1, 1))
        g.edata['w'] = torch.ones(g.num_edges(), dtype=torch.int32)

        # Draw graph
        demo_graph = g.to_networkx()
        draw_graph(demo_graph, node_labels)

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
    label_list = label_list.to(args.device)
    # label_list = list(dataset.graph_labels.numpy().flatten())

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
