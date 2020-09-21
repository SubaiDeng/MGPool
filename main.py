import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import load_data
from model import Net, accuracy, val
from tqdm import tqdm
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

    parser.set_defaults(
        dataset='NCI1',
        batch_size=64,
        epoch_num=300,
        lr=0.005,
        num_layers=3,
        hidden_dim=64,
        device='cuda',
        feat_drop=0,
        final_drop=0.5,
    )
    return parser.parse_args()


def train(args, train_loader, val_loader):

    model = Net(args)
    if args.device == 'cuda':
        model = model.cuda()


    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    epoch_losses = []
    epoch_train_acc = []
    epoch_test_acc = []
    for epoch in tqdm(range(args.epoch_num)):
        epoch_loss = 0
        train_acc = 0
        model.train()
        # train_loader = tqdm(train_loader)
        iter = 0
        for bg, label in train_loader:
            prediction = model(bg)
            if args.device == 'cuda':
                label = label.cuda()
            loss = loss_func(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()
            train_acc += accuracy(prediction, label)
            iter += 1

        epoch_loss /= iter
        train_acc /= iter
        test_acc = val(model, val_loader, args)
        epoch_losses.append(epoch_loss)
        epoch_train_acc.append(train_acc)
        epoch_test_acc.append(test_acc)
        print('\n Loss:{:.4f} \nTrain ACC: {:.4f}, Test ACC: {:.4f}'.format(epoch_loss, train_acc, test_acc))
        if epoch_loss < 1e-5:
            break
    torch.save(model.state_dict(), MODEL_PATH + 'Model-NCI-' + '.kpl')
    print("SUCCESS: Model Training Finished.")
    print("Train ACC: {:.4f}".format(train_acc))


def main():
    print('hello world')
    args = arg_parse()

    train_loader,  test_loader,  fea_dim, num_class = load_data.load_data(args)
    args.in_dim = fea_dim
    args.out_dim = num_class
    train(args, train_loader, test_loader)



if __name__ == "__main__":
    main()
