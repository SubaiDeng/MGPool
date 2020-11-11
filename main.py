import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import load_data
from model import Net, accuracy, val
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np
import random
import EBGC
import os

import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = './dump/model/'


def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--dataset', type=str,
                        help='Name of the dataset: NMI1/ENZYMES')
    parser.add_argument('--batch_size', type=int,
                        help='The sizes of the batch')
    parser.add_argument('--epoch_num', type=int,
                        help='The number of the epoch')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    
    parser.add_argument('--gcn_num_layers', type=int,
                        help='The number of the gcn layers in the model')
    parser.add_argument('--gin_num_layers', type=int,
                        help='The number of the gin layers in the model')
    parser.add_argument('--gin_mlp_num_layers', type=int,
                        help='number of the layers in the mlp')
    parser.add_argument('--gin_graph_pooling_type', type=str,
                        help='The type of graph pooling in GIN')
    parser.add_argument('--gin_neighbor_pooling_type', type=str,
                        help='The type of neighbor aggregation in GIN')
    parser.add_argument('--learn_eps', action="store_true",
                        help='Whether to learn the epsilon weighting for the center nodes.')
    
    parser.add_argument('--hidden_dim', type=int,
                        help='The dimensional of the hidden layers')
    parser.add_argument('--device', type=str,
                        help='The type of the running device')
    parser.add_argument('--feat_drop', type=float,
                        help='feat_drop')
    parser.add_argument('--H', type=int,
                        help='The number of H (the top H eigenvector is reserved)')
    parser.add_argument('--num_landmarks', type=int,
                        help='The number of the landmark.')
    parser.add_argument('--num_kfold', type=int,
                        help='The number of the kFold crossing validation.')
    parser.add_argument('--num_layers_after_cluster', type=int,
                        help='num_layers_after_cluster')
    parser.add_argument('--num_random')

    parser.set_defaults(
        # dataset='PROTEINS_full',
        dataset='NCI1',
        # dataset='DD',
        batch_size=64,
        epoch_num=500,
        lr=0.005,
        hidden_dim=64,
        device='cuda',
        feat_drop=0,
        final_drop=0.5,
        num_kfold=10,
        num_random=1,
        num_layers_after_cluster=1,

        gcn_num_layers=3,
        gin_num_layers=3,
        gin_mlp_num_layers=3,
        gin_graph_pooling_type='sum',
        gin_neighbor_pooling_type='sum',
        learn_eps=True,
    )
    return parser.parse_args()


def train(args, train_loader, val_loader):

    model = Net(args).to(args.device)
    saved_epoch_id = 0

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    min_loss = 1e10
    for epoch in tqdm(range(args.epoch_num)):
        losses_list = []
        train_acc_list = []
        model.train()
        for batched_graph_merge, label in train_loader:
            prediction = model(batched_graph_merge[0], batched_graph_merge[1])
            if args.device == 'cuda:0':
                label = label.cuda()
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_list.append(loss.detach().item())
            train_acc_list.append(accuracy(prediction, label))

        loss = np.average(losses_list)
        train_acc = np.average(train_acc_list)
        val_acc, val_loss = val(model, val_loader, args)
        if val_loss < min_loss:
            torch.save(model.state_dict(), MODEL_PATH + 'Model-NCI' + '.kpl')
            min_loss = val_loss
            print("\n EPOCH:{}\t Train Loss:{:.4f}\tTrain ACC: {:.4f}\t\tVal Loss:{:.4f}\t Val ACC: {:.4f}\tModel saved at epoch {}".format(epoch, loss, train_acc, val_loss, val_acc, epoch))
            saved_epoch_id = epoch
        else:
            print("\n EPOCH:{}\t Train Loss:{:.4f}\tTrain ACC: {:.4f}\t\tVal Loss:{:.4f}\t Val ACC: {:.4f}".format(epoch, loss, train_acc, val_loss, val_acc))
    print("SUCCESS: Model Training Finished.")
    return saved_epoch_id


def test(test_loader, args):
    model = Net(args).to(args.device)
    model.load_state_dict(torch.load(MODEL_PATH + 'Model-NCI' + '.kpl'))
    test_acc, test_loss = val(model, test_loader, args)
    return test_acc


def main():
    print('hello world')
    args = arg_parse()
    if torch.cuda.is_available():
        args.device = 'cuda:0'
    graph_list, label_list, fea_dim, num_class = load_data.load_data(args)
    args.in_dim = fea_dim
    args.out_dim = num_class

    random_seed = list(range(args.num_random))
    # random_seed = list([1])
    test_acc_list = []

    for seed in random_seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        kf = KFold(n_splits=args.num_kfold, random_state=seed, shuffle=True)
        for iter, (train_index, test_index) in enumerate(kf.split(graph_list)):
            print("Fold:{}".format(iter))
            # if iter < 3:
            #     continue
            train_loader, val_loader, test_loader = load_data.split_dataset(args, train_index, test_index, graph_list, label_list)
            saved_epoch_id = train(args, train_loader, val_loader)
            test_acc = test(test_loader, args)
            test_acc_list.append(test_acc)
            print("\nFold:{} \t Test ACC:{:.4f}".format(iter, test_acc))
        acc_avg = np.average(test_acc_list)
        acc_std = np.std(test_acc_list)
        print("\n TOTAL: Test ACC:{:.4f}, Test ACC STD:{:.4f} Using the model at epoch {}".format(acc_avg, acc_std, saved_epoch_id))
        print("Test ACC list:")
        print(test_acc_list)


if __name__ == "__main__":
    main()
