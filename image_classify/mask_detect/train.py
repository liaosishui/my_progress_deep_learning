import torch
from utils.try_gpu import try_all_gpu, try_gpu
from utils.accuracy import accuracy
from torch import nn


def train(net, train_iter, test_iter, lr, num_epochs, wd, lr_period, lr_decay, predict=False):
    updater = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(updater, lr_period, lr_decay)
    loss = nn.CrossEntropyLoss(reduction='mean')
    num_batches = len(train_iter)
    train_loss_sum = []
    train_acc_sum = []
    show_gpu = False
    net = nn.DataParallel(net, device_ids=try_all_gpu()).to(try_gpu(0))
    for epoch in range(num_epochs):
        test_loss_sum = []
        test_acc_sum = []
        num_test_batches = len(test_iter)
        y_res = []
        if predict == False:
            net.train()
            for i, (X, y) in enumerate(train_iter):
                if isinstance(X, list):
                    X = [x.to(try_gpu(0)) for x in X]
                else:
                    X = X.to(try_gpu(0))
                y = y.to(try_gpu(0))
                y_hat = net(X)
                l = loss(y_hat, y)
                updater.zero_grad()
                l.sum().backward()
                updater.step()
                train_loss_sum.append(l.sum())
                train_acc_sum.append(accuracy(y_hat, y))
                if show_gpu == False:
                    print(f'this is training by {X.device, y.device}')
                    show_gpu = True
                del X
                del y
            print(
                f'the {epoch} : train_loss {sum(train_loss_sum) / num_batches}, train_accuracy {sum(train_acc_sum) / num_batches}')
            train_loss_sum.clear()
            train_acc_sum.clear()
            scheduler.step()
            with torch.no_grad():
                for i, (X, y) in enumerate(test_iter):
                    y_hat = net(X.to(try_gpu(0)))
                    l = loss(y_hat, y.to(try_gpu(0)))
                    test_loss_sum.append(l.sum())
                    test_acc_sum.append(accuracy(y_hat, y))
                    y_temp = y_hat.argmax(dim=1)
                    y_res.append(y_temp)
                print(f'test_loss {sum(test_loss_sum) / num_test_batches}, test_accuracy {sum(test_acc_sum) / num_test_batches}')
        else:
            with torch.no_grad():
                for (X, y) in test_iter:
                    y_hat = net(X.to(try_gpu(0)))
                    y_temp = y_hat.argmax(dim=1)
                    y_res += y_temp
            return y_res
    return y_res
