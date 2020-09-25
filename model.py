import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from scipy.sparse import coo_matrix



class GCNLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, feat_drop=None):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.gcn_msg = fn.copy_src(src='h', out='m')
        self.gcn_reduce = fn.sum(msg='m', out='h')
        self.feat_drop = feat_drop
        self.bn_layer = torch.nn.BatchNorm1d(hidden_dim)

    def forward(self, g, h):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        # h = self.feat_drop(h)
        h = F.dropout(h, self.feat_drop, training=self.training)
        with g.local_scope():
            g.ndata['h'] = h
            g.update_all(self.gcn_msg, self.gcn_reduce)
            h = g.ndata.pop('h')
            h = self.linear(h)
            return self.bn_layer(h)


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.num_layers = args.num_layers
        self.in_dim = args.in_dim
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim
        self.gcn_layer_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.device = args.device
        self.feat_drop = args.feat_drop
        self.final_drop = args.final_drop

        self.gcn_layer_list.append(GCNLayer(self.in_dim, self.hidden_dim, self.feat_drop))
        for l in range(1, self.num_layers):
            self.gcn_layer_list.append(GCNLayer(self.hidden_dim, self.hidden_dim, self.feat_drop))
        self.gcn_pooling_layer = GCNLayer(self.hidden_dim, self.hidden_dim, self.feat_drop)

        # self.classify_list = nn.ModuleList()
        # for l in range(self.num_layers):
        #     self.classify_list.append(nn.Linear(self.hidden_dim, self.out_dim))
        #     # self.bn_list.append(nn.BatchNorm1d(self.hidden_dim))

        self.classify = nn.Linear(self.hidden_dim, self.out_dim)

    # def split_subgraph(self, g, fea, seglen):
    #     idx = 0
    #     adj = g.to_dense().numpy()
    #     adj_list = list([])
    #     fea_list = list([])
    #     for num_node in seglen:
    #         new_idx = idx + num_node
    #         sub_adj = adj[idx:new_idx, idx:new_idx]
    #         sub_fea = fea[idx:new_idx]
    #         adj_list.append(sub_adj)
    #         fea_list.append(sub_fea)
    #         idx = new_idx
    #
    #     return adj_list, fea_list

    # def build_graph(self, adj, landmarks_list):
    #     pool_g = dgl.DGLGraph()
    #     id_list_x = []
    #     id_list_y = []
    #     val_list = []
    #     for iter_i, row in enumerate(adj):
    #         for iter_j, element in enumerate(row):
    #             id_list_x.append(iter_i)
    #             id_list_y.append(iter_j)
    #             val_list.append(element)
    #     sp_graph = coo_matrix((val_list, (id_list_x, id_list_y)))
    #     pool_g = dgl.from_scipy(sp_graph)
    #     pool_fea = torch.tensor([])
    #     for landmark in landmarks_list:
    #         pool_fea = torch.cat((pool_fea, landmark), dim=0)
    #     pool_fea = pool_fea.reshape(len(landmarks_list), -1)
    #     return pool_g, pool_fea

    def entropy_cal(self, p):
        log_p = torch.log(p)
        H = -torch.sum(p*log_p, dim=1)
        max_H = torch.max(H)
        lambda_list = 1-(H/max_H)
        return lambda_list

    def weighted_pooling(self, batch_graph, lambda_list, h):
        graph_emb_list = torch.tensor([])
        seglen = batch_graph.batch_num_nodes().cpu().numpy()
        # if self.device==torch.device("cuda:0"):
        #     graph_emb_list = graph_emb_list.cuda()
        if self.device=='cuda:0':
            graph_emb_list = graph_emb_list.cuda()
        idx = 0
        for num_node in seglen:
            # num_node = len(graph.g)
            weight = lambda_list[idx:idx+num_node]
            vec = h[idx:idx+num_node]
            g_emb = torch.sum(weight.unsqueeze(-1) * vec, dim=0).unsqueeze(0)
            graph_emb_list = torch.cat((graph_emb_list, g_emb), dim=0)
            idx += num_node
        return graph_emb_list

    def forward(self, g):
        score_over_layer = 0
        h = g.ndata['feat']

        # Last layer for entropy Calculation
        for l in range(self.num_layers):
            h = self.gcn_layer_list[l](g, h)
            h = F.elu(h)

        lambda_list = self.entropy_cal(F.softmax(self.classify(h), dim=1))
        pooled_h = self.weighted_pooling(g, lambda_list, h)
        t = self.classify(pooled_h)
        score_over_layer += F.dropout(t, self.final_drop, training=self.training)

        # # All layers for entropy Calculation
        # for l in range(self.num_layers):
        #     h = self.gcn_layer_list[l](g, h)
        #     h = F.elu(h)
        #     # lambda_list = self.entropy_cal(F.softmax(self.classify_list[l](self.bn_list[l](h)), dim=1))
        #     lambda_list = self.entropy_cal(F.softmax(self.classify_list[l](h), dim=1))
        #     pooled_h = self.weighted_pooling(g, lambda_list, h)
        #     # t = self.classify_list[l](self.bn_list[l](pooled_h))
        #     t = self.classify_list[l](pooled_h)
        #     score_over_layer += F.dropout(t, self.final_drop, training=self.training)




        # g.ndata['h'] = h
        # seglen = g.batch_num_nodes().numpy()

        # adj = g.adj()
        # graph_list, fea_list = self.split_subgraph(adj, h, seglen)
        # readout_list = torch.tensor([])
        # for iter in range(len(seglen)):
        #
        #     adj = graph_list[iter]
        #     fea = fea_list[iter]
        #
        #     reuslt_kmeans_h = KMeans(n_clusters=2, random_state=0).fit(fea.data.numpy())
        #
        #     landmark_h = self.find_landmark(fea, reuslt_kmeans_h.labels_, reuslt_kmeans_h.cluster_centers_)
        #     t = [x.data.numpy() for x in landmark_h]
        #     landmark_similarity = euclidean_distances(fea.data.numpy(), t)
        #     pool_adj = np.dot(np.dot(landmark_similarity.T, adj), landmark_similarity)
        #     pool_g, pool_fea = self.build_graph(pool_adj, landmark_h)
        #     pool_h = self.gcn_pooling_layer(pool_g, pool_fea)
        #
        #     readout_sum = torch.sum(pool_h, 0)
        #     readout_list = torch.cat((readout_list, readout_sum),dim=0)
        #     # readout_max = torch.max(pool_h)
        #     pass
        #
        # readout_list = readout_list.reshape(len(seglen),-1)
        # hg = dgl.mean_nodes(g, 'h')
        # return self.classify(hg)
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
    for iter, (bg, label) in enumerate(data_loader):
        prediction = model(bg)
        if args.device == 'cuda:0':
            label = label.cuda()
        acc_list.append(accuracy(prediction, label))
        loss_list.append(loss_func(prediction, label).detach().item())
    acc = np.average(acc_list)
    loss = np.average(loss_list)
    return acc, loss
