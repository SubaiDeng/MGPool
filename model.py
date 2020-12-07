import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from scipy.sparse import coo_matrix
import EBGC
import networkx as nx


class MLP_layer(nn.Module):
    def __init__(self,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 output_dim, ):
        super(MLP_layer, self).__init__()
        self.num_layers = num_mlp_layers
        self.linears = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        if self.num_layers == 1:
            self.linears.append(nn.Linear(input_dim, output_dim))
        else:
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(self.num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(self.num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, h):
        if self.num_layers == 1:
            return self.linears[0](h)
        else:
            for layer in range(self.num_layers - 1):
                t1 = self.linears[layer](h)
                t1 = self.batch_norms[layer](t1)
                h = F.relu(t1)
        return F.relu(self.linears[self.num_layers - 1](h))


class GIN(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 feat_drop,
                 num_mlp_layers,
                 graph_pooling_type,
                 neighbor_pooling_type,
                 learn_eps,):
        super(GIN, self).__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.mlp_layers = nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.eps_list = nn.Parameter(torch.zeros(self.num_layers))
        self.graph_pooling_type = graph_pooling_type
        self.feat_drop = feat_drop
        self.neighbor_pooling_type = neighbor_pooling_type
        self.id_layers = 0
        self.learn_eps = learn_eps

        for layer in range(num_layers):
            if layer == 0:  # The first layer
                self.mlp_layers.append(MLP_layer(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlp_layers.append(MLP_layer(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def self_eps_aggregate(self, h_self, h_neigh):
        if self.learn_eps:
            h = (1 + self.eps_list[self.id_layers]) * h_self + h_neigh
        else:
            h = h_self + h_neigh
        return h

    def message_func(self, edges):
        # print("h:{}".format(edges.src['h']))
        # print("w:{}".format(edges.data['w']))
        # h = torch.mul(edges.data['w'].float().reshape(1,-1), edges.src['h'].float(),)
        h = (edges.data['w'].float() * edges.src['h'].float().t()).t()
        # h = edges.src['h'].float()
        # a_1 = edges.data['w'].float()
        # a_2 = edges.src['h'].float()
        return {'msg_h': h}

    def reduce_mean_func(self, nodes):
        h = torch.mean(nodes.mailbox['msg_h'], dim=1)
        h = self.self_eps_aggregate(nodes.data['h'], h)
        return {'h': h}

    def reduce_sum_func(self, nodes):
        h = torch.sum(nodes.mailbox['msg_h'], dim=1)
        h = self.self_eps_aggregate(nodes.data['h'], h)
        return {'h': h}

    def node_pooling(self, g):
        if self.neighbor_pooling_type == 'sum':
            g.update_all(self.message_func, self.reduce_sum_func)
        elif self.neighbor_pooling_type == 'mean':
            g.update_all(self.message_func, self.reduce_mean_func)
        return g.ndata.pop('h')

    def graph_pooling(self, g):
        h = 0
        if self.graph_pooling_type == 'max':
            hg = dgl.max_nodes(g, 'h')
        elif self.graph_pooling_type == 'mean':
            hg = dgl.mean_nodes(g, 'h')
        elif self.graph_pooling_type == 'sum':
            hg = dgl.sum_nodes(g, 'h')
        return hg

    def forward(self, g, h):

        # 0 layer
        g.ndata['h'] = h
        for layer in range(self.num_layers):
            self.id_layers = layer
            # step 1 aggregate
            h = self.node_pooling(g)
            # step 2 MLP
            h = self.mlp_layers[layer](h)
            h = self.batch_norms[layer](h)
            h = F.relu(h)
            g.ndata['h'] = h
        # h_graph = self.graph_pooling(g)
        return h


class GCNLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, feat_drop=None):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        # self.gcn_msg = fn.copy_src(src='h', out='m')
        # self.gcn_reduce = fn.sum(msg='m', out='h')
        self.feat_drop = feat_drop
        self.bn_layer = torch.nn.BatchNorm1d(hidden_dim)
        
    def message_func(self, edges):
        # print("h:{}".format(edges.src['h']))
        # print("w:{}".format(edges.data['w']))
        # h = torch.mul(edges.data['w'].float().reshape(1,-1), edges.src['h'].float(),)
        h = (edges.data['w'].float() * edges.src['h'].float().t()).t()
        # h = edges.src['h'].float()
        # a_1 = edges.data['w'].float()
        # a_2 = edges.src['h'].float()
        return {'msg_h': h}

    def reduce_func(self, nodes):
        h = torch.sum(nodes.mailbox['msg_h'], dim=1)
        h = nodes.data['h'] + h
        return {'h': h}

    def forward(self, g, h):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        # h = self.feat_drop(h)
        h = F.dropout(h, self.feat_drop, training=self.training)
        with g.local_scope():
            h = self.linear(h)
            g.ndata['h'] = h
            g.update_all(self.message_func, self.reduce_func)
            h = g.ndata.pop('h')
            return self.bn_layer(h)


class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers, feat_drop):
        super(GCN, self).__init__()
        self.gcn_layer_list = nn.ModuleList()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.feat_drop = feat_drop
        self.num_layers = num_layers

        if num_layers==1:
            self.gcn_layer_list.append(GCNLayer(self.in_dim, self.out_dim, self.feat_drop))
        else:
            self.gcn_layer_list.append(GCNLayer(self.in_dim, self.hidden_dim, self.feat_drop))  # first layer
            for i in range(1, self.num_layers-1):
                self.gcn_layer_list.append(GCNLayer(self.hidden_dim, self.hidden_dim, self.feat_drop))  # hidden layer
            self.gcn_layer_list.append(GCNLayer(self.hidden_dim, self.out_dim, self.feat_drop))  # final layer

    def forward(self, g, h):
        for i in range(self.num_layers):
            h = self.gcn_layer_list[i](g, h)
            h = F.relu(h)
            
        return h


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

        self.final_drop = args.final_drop
        self.device = args.device

        self.in_dim = args.in_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim

        self.bn_list = nn.ModuleList()
        self.feat_drop = args.feat_drop
        self.num_layers_after_cluster = args.num_layers_after_cluster

        self.GCN = GCN(self.in_dim, 
                       self.hidden_dim, 
                       self.hidden_dim, 
                       args.gcn_num_layers, 
                       self.feat_drop)
        self.GIN = GIN(self.in_dim, 
                       self.hidden_dim, 
                       self.hidden_dim, 
                       args.gin_num_layers, 
                       self.feat_drop, 
                       args.gin_mlp_num_layers, 
                       args.gin_graph_pooling_type, 
                       args.gin_neighbor_pooling_type,
                       args.learn_eps)
        self.finalGCN = GCN(self.hidden_dim*2, 
                            self.hidden_dim*2, 
                            self.hidden_dim*2, 
                            args.num_layers_after_cluster, 
                            self.feat_drop)

        self.classify = nn.Linear(self.hidden_dim*2, self.out_dim)

        # self.classify_list = nn.ModuleList()
        # for l in range(self.num_layers):
        #     self.classify_list.append(nn.Linear(self.hidden_dim, self.out_dim))
        #     # self.bn_list.append(nn.BatchNorm1d(self.hidden_dim))


    def entropy_cal(self, p):
        log_p = torch.log(p)
        H = -torch.sum(p*log_p, dim=1)
        max_H = torch.max(H)
        lambda_list = 1-(H/max_H)
        return lambda_list

    def weighted_pooling(self, seglen, lambda_list, h):
        graph_emb_list = torch.tensor([])
        # if self.device==torch.device("cuda:0"):
        #     graph_emb_list = graph_emb_list.cuda()
        graph_emb_list = graph_emb_list.to(self.device)
        idx = 0
        for num_node in seglen:
            # num_node = len(graph.g)
            weight = lambda_list[idx:idx+num_node]
            vec = h[idx:idx+num_node]
            g_emb = torch.sum(weight.unsqueeze(-1) * vec, dim=0).unsqueeze(0)
            graph_emb_list = torch.cat((graph_emb_list, g_emb), dim=0)
            idx += num_node
        return graph_emb_list

    def cluster_pooling(self, batch_g):
        g_lists = dgl.unbatch(batch_g)
        pooled_g_list = list()
        pooled_idx_list = list()
        pooled_adj_list = list()
        pooled_emb_list = list()
        for i, g in enumerate(g_lists):
            g = g.to('cpu')
            # print(str(i) + ' graph pooling')
            node_labels = g.ndata['label']
            node_emb = g.ndata['emb']
            n_cluster = int(torch.max(node_labels) + 1)
            n_node = len(node_emb)
            node_cluster = torch.zeros([n_node, n_cluster])
            adj = g.adj().to_dense()
            if n_cluster == 1:
                # One Cluster
                pooled_emb = node_emb
                pooled_adj = adj
            else:
                node_cluster[list(range(n_node)), list(node_labels)] = 1
                pooled_emb = torch.mm(node_cluster.T, g.ndata['emb'])
                pooled_adj = torch.mm(torch.mm(node_cluster.T, adj), node_cluster)

            for i in range(len(pooled_adj)):
                pooled_adj[i,i] = 1

            idx = torch.nonzero(pooled_adj)

            x_id = idx[:, 0]
            y_id = idx[:, 1]
            weight = pooled_adj[x_id,y_id]
            
            pooled_g = dgl.graph((x_id, y_id), num_nodes=len(pooled_emb))
            pooled_g.edata['w'] = weight
            pooled_g.ndata['emb'] = pooled_emb
            pooled_g = pooled_g.to(self.device)
            pooled_g_list.append(pooled_g)


        return pooled_g_list

    def forward(self, batch_g, batch_masked_g):

        score_over_layer = 0
        h_gcn = batch_g.ndata['feat']
        h_gcn = self.GCN(batch_g, h_gcn)
        
        h_gin = batch_masked_g.ndata['feat']
        h_gin = self.GIN(batch_masked_g, h_gin)
        h = torch.cat([h_gcn, h_gin], dim=1)

        batch_g.ndata['emb'] = h
        pooled_g_list = self.cluster_pooling(batch_g)

        pooled_batch_g = dgl.batch(pooled_g_list)
        h = pooled_batch_g.ndata['emb']

        h = self.finalGCN(pooled_batch_g, h)

        seglen = [len(x) for x in pooled_g_list]
        lambda_list = self.entropy_cal(F.softmax(self.classify(h), dim=1))

        pooled_h = self.weighted_pooling(seglen, lambda_list, h)
        t = self.classify(pooled_h)
        score_over_layer += F.dropout(t, self.final_drop, training=self.training)

        return score_over_layer


def accuracy(prediction, labels):
    _, indices = torch.max(prediction, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def val(model, data_loader, args):
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    acc_list = []
    loss_list = []
    for iter, (batched_graph_merge, label) in enumerate(data_loader):
        prediction = model(batched_graph_merge[0], batched_graph_merge[1])
        label = label.to(args.device)
        acc_list.append(accuracy(prediction, label))
        loss_list.append(loss_func(prediction, label).detach().item())
    acc = np.average(acc_list)
    # loss = np.average(loss_list)
    loss = np.sum(loss_list)
    return acc, loss
