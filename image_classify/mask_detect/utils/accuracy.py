import torch

def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = torch.argmax(y_hat, axis=1)
    all_len = len(y_hat)
    right_y = [1 if a - b < 1e-5 else 0 for (a, b) in zip(y, y_hat)]
    right_len = sum(right_y)
    return float(right_len / all_len)
