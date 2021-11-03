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
import time

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
    parser.add_argument('--num_kfold', type=int,
                        help='The number of the kFold crossing validation.')
    parser.add_argument('--num_layers_after_cluster', type=int,
                        help='num_layers_after_cluster')
    parser.add_argument('--num_random')

    parser.set_defaults(
        # dataset='PROTEINS_full',
        # dataset='PROTEINS',
        # dataset='NCI1',
        # dataset='FRANKENSTEIN',
        # dataset='MUTAG',
        # dataset='DD',
        dataset='IMDB-BINARY',
        batch_size=32,
        epoch_num=200,
        lr=0.005,
        hidden_dim=64,
        device='cuda:0',
        feat_drop=0.2,
        final_drop=0.5,
        num_kfold=10,
        num_random=1,
        num_layers_after_cluster=2,
        gcn_num_layers=3,
        gin_num_layers=3,
        gin_mlp_num_layers=3,
        gin_graph_pooling_type='sum',
        gin_neighbor_pooling_type='sum',
        ratio_entropy=0.7,
        ratio_save_model=0.5,
        learn_eps=True,
        node_feature_type='degree',
        cluster_type='LV',
    )
    return parser.parse_args()


def train(args, train_loader, val_loader, test_loader):
    model = Net(args).to(args.device)
    saved_epoch_id = 0

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    min_loss = 1e10
    max_acc = 0
    for epoch in tqdm(range(args.epoch_num)):
        losses_list = []
        train_acc_list = []
        if epoch < args.epoch_num * args.ratio_entropy:
            enable_entropy = False
        else:
            enable_entropy = True
        model.train()
        for batched_graph_merge, label in train_loader:
            prediction = model(batched_graph_merge[0], batched_graph_merge[1], enable_entropy)
            label = label.to(args.device)
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses_list.append(loss.detach().item())
            train_acc_list.append(accuracy(prediction, label))

        loss = np.average(losses_list)
        train_acc = np.average(train_acc_list)
        val_acc, val_loss = val(model, val_loader, args, enable_entropy)
        test_acc, test_loss = val(model, test_loader, args, enable_entropy)
        # if val_loss < min_loss and epoch > args.epoch_num * 0.7:
        if val_acc > max_acc and epoch > args.epoch_num * args.ratio_save_model:
            torch.save(model.state_dict(), MODEL_PATH + 'Model-NCI' + '.kpl')
            min_loss = val_loss
            max_acc = val_acc
            print("\n EPOCH:{}\t "
                  "Train Loss:{:.4f}\tTrain ACC: {:.4f}\t"
                  "Val Loss:{:.4f}\t Val ACC: {:.4f}\t"
                  "Test Loss:{:.4f}\t Test ACC: {:.4f}\t"
                  "Model saved at epoch {}".format(epoch, loss, train_acc, val_loss, val_acc, test_loss, test_acc,
                                                   epoch))
            saved_epoch_id = epoch
            test_acc_result = test_acc
        else:
            print("\n EPOCH:{}\t "
                  "Train Loss:{:.4f}\tTrain ACC: {:.4f}\t"
                  "Val Loss:{:.4f}\t Val ACC: {:.4f}\t"
                  "Test Loss:{:.4f}\t Test ACC: {:.4f}\t".format(epoch, loss, train_acc, val_loss, val_acc, test_loss,
                                                                 test_acc))
    print("SUCCESS: Model Training Finished.")
    return saved_epoch_id, test_acc_result


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

    time_preprocess_list = []
    for i in range(1):
        graph_list, label_list, fea_dim, num_class, time_preprocess_delta = load_data.load_data(args)
        time_preprocess_list.append(time_preprocess_delta)
    time_preprocess_avg = np.average(time_preprocess_list)
    time_preprocess_std = np.std(time_preprocess_list)
    print("Preprocess Time AVG: {:.4f}".format(time_preprocess_avg))
    print("Preprocess Time STD: {:.4f}".format(time_preprocess_std))
    args.in_dim = fea_dim
    args.out_dim = num_class
    random_seed = list(range(args.num_random))
    # random_seed = list([2])
    test_acc_list = []

    time_train_list = []
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
            train_loader, val_loader, test_loader = load_data.split_dataset(args, train_index, test_index, graph_list,
                                                                            label_list)

            time_train_start = time.time()
            saved_epoch_id, test_acc = train(args, train_loader, val_loader, test_loader)
            time_train_end = time.time()
            time_train_delta = time_train_end - time_train_start
            # test_acc = test(test_loader, args)
            test_acc_list.append(test_acc)
            time_train_list.append(time_train_delta)
            print("\nFold:{} \t Test ACC:{:.4f}".format(iter, test_acc))
        acc_avg = np.average(test_acc_list)
        acc_std = np.std(test_acc_list)
        time_train_avg = np.average(time_train_list)
        time_train_std = np.std(time_train_list)
        print("\n TOTAL: Test ACC:{:.4f}, Test ACC STD:{:.4f} Using the model at epoch {}".format(acc_avg, acc_std,
                                                                                                  saved_epoch_id))
        print("Test ACC list:")
        print(test_acc_list)
        print("Running Time:")
        print("Preprocess Time AVG: {:.4f}".format(time_preprocess_avg))
        print("Preprocess Time STD: {:.4f}".format(time_preprocess_std))
        print("Train Time AVG: {:.4f}".format(time_train_avg))
        print("Train Time STD: {:.4f}".format(time_train_std))


if __name__ == "__main__":
    main()
