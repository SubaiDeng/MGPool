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

import warnings
warnings.filterwarnings("ignore")

MODEL_PATH = './dump/model/'


def arg_parse():
    parser = argparse.ArgumentParser(description='Arguments.')
    parser.add_argument('--dataset', type=str,
                        help='Name of the dataset: NMI1/ENZYMES')
    parser.add_argument('--pool_sizes', type=str,
                        help='The average size of each subgraph for each layers: 50-10-5')
    parser.add_argument('--batch_size', type=int,
                        help='The sizes of the batch')
    parser.add_argument('--epoch_num', type=int,
                        help='The number of the epoch')
    parser.add_argument('--lr', type=float,
                        help='Learning rate')
    parser.add_argument('--num_layers', type=int,
                        help='The number of the layers in the model')
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
    parser.add_argument('--num_random')

    parser.set_defaults(
        dataset='PROTEINS_full',
        # dataset='NCI1',
        batch_size=64,
        epoch_num=300,
        lr=0.005,
        num_layers=3,
        hidden_dim=64,
        device='cuda',
        feat_drop=0.05,
        final_drop=0.5,
        num_kfold=10,
        num_random=1,
    )
    return parser.parse_args()


def train(args, train_loader, val_loader):

    model = Net(args).to(args.device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    min_loss = 1e10
    for epoch in tqdm(range(args.epoch_num)):
        losses_list = []
        train_acc_list = []
        model.train()
        for bg, label in train_loader:
            prediction = model(bg)
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
        print("\n Train Loss:{:.4f}\tTrain ACC: {:.4f}\t\tVal Loss:{:.4f}\t Val ACC: {:.4f}".format(loss, train_acc, val_loss, val_acc))
        if val_loss < min_loss:
            torch.save(model.state_dict(), MODEL_PATH + 'Model-NCI' + '.kpl')
            min_loss = val_loss
            print("Model saved at epoch{}".format(epoch))

    print("SUCCESS: Model Training Finished.")


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
    test_acc_list = []
    for seed in random_seed:
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        kf = KFold(n_splits=args.num_kfold, random_state=seed, shuffle=True)
        for iter, (train_index, test_index) in enumerate(kf.split(graph_list)):
            train_loader, val_loader, test_loader = load_data.split_dataset(args, train_index, test_index, graph_list, label_list)
            train(args, train_loader, val_loader)
            test_acc = test(test_loader, args)
            test_acc_list.append(test_acc)
            print("\nFold:{} \t Test ACC:{:.4f}".format(iter, test_acc))
        acc_avg = np.average(test_acc_list)
        acc_std = np.std(test_acc_list)
        print("\n TOTAL: Test ACC:{:.4f}, Test ACC STD:{:.4f}".format(acc_avg, acc_std))
        print("Test ACC list:")
        print(test_acc_list)


if __name__ == "__main__":
    main()
