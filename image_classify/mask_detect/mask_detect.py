import matplotlib.image
import matplotlib.pyplot as plt
import torch
import torchvision
from utils.splitDatasets import split_datasets
from utils.MyDatasets import MyDataset
from utils.unzipfile import unzip_files
from train import train
from predict import pred
from PIL import Image
def run():
    # unzip_files('data/mask_detect_kaggle.zip', 'data/')
    # transforms = {
    #     'train': torchvision.transforms.Compose([
    #         torchvision.transforms.Resize((240, 240)),
    #         torchvision.transforms.RandomHorizontalFlip(),
    #         torchvision.transforms.ToTensor()]),
    #     'valid': torchvision.transforms.Compose([
    #         torchvision.transforms.Resize((240, 240)),
    #         torchvision.transforms.RandomHorizontalFlip(),
    #         torchvision.transforms.ToTensor()])
    # }
    # train_paths, train_labels, valid_paths, valid_labels = split_datasets('data/mask_detection/')
    # train_ds = MyDataset(train_paths, train_labels, transforms['train'])
    # valid_ds = MyDataset(valid_paths, valid_labels, transforms['valid'])
    # valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size=5, shuffle=True, drop_last=True)
    # train_iter = torch.utils.data.DataLoader(train_ds, batch_size=5, shuffle=True, drop_last=False)
    # res_net = torchvision.models.resnet18()
    # res_net.fc = torch.nn.Linear(512, 2, bias=True)
    # y_train_res = train(res_net, train_iter, valid_iter, 0.001, 10, 5e-4, 4, 0.9)
    # torch.save(res_net.state_dict(), 'runs/weights/resnet18-params')
    pretrained_net = torchvision.models.resnet18()
    pretrained_net.fc = torch.nn.Linear(512, 2, bias=True)
    pretrained_net.load_state_dict(torch.load('runs/weights/resnet18-params'))
    test_img_path = 'data/test.jpg'
    test_image = Image.open(test_img_path).convert('RGB')
    plt.imshow(test_image)
    pred(pretrained_net, test_image)
    plt.show()

def __main__():
    run()

if __name__ == '__main__':
    __main__()