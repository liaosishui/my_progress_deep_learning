import torch

def try_all_gpu():
    devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else torch.device(f'cpu')

def try_gpu(i=0):
    gpu_count = torch.cuda.device_count()
    assert ((gpu_count == 0) | (gpu_count-1 >= i)), f"you haven't enough gpus, you have {gpu_count} gpus"
    if gpu_count == 0:
        return torch.device('cpu')
    else:
        return torch.device(f'cuda:{i}')
