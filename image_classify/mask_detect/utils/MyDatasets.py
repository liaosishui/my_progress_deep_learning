import os.path
import random
from PIL import Image
import torch



class MyDataset(torch.utils.data.Dataset):

    def __init__(self, images_path : list, images_classes : list, transform=None):
        self.images_path = images_path
        self.images_class = images_classes
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            img = img.convert('RGB')
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

def split_datasets(target_path, train_size=0.8):
    '''divide the datasets into train_dataset and valid_dataset'''

    '''divide the datasets into train_dataset and valid_dataset'''
    random.seed(42)
    assert os.path.exists(target_path), f'the path is not exists'
    class_names = [cls for cls in os.listdir(target_path) if os.path.isdir(os.path.join(target_path, cls))]
    class_indices = dict((k, v) for v, k in enumerate(class_names))
    class_names.sort()
    # json_str = json.dumps(dict((val, key) for key, val in class_names.items()), indent=4)

    train_images_paths = []
    train_images_labels = []
    valid_images_paths = []
    valid_images_labels = []
    supported = ['.jpg', '.png', '.JPG', '.PNG']
    _, dir_names, filenames = next(os.walk(target_path))
    print(_, dir_names, len(filenames))
    for dir_name in dir_names:
        for dir_path, _, filenames in os.walk(os.path.join(target_path, dir_name)):
            len_filenames = len(filenames)
            filename_indices = list(range(len_filenames))
            random.shuffle(filename_indices)
            for i in filename_indices[:int(len_filenames * train_size)]:
                train_images_paths.append(os.path.join(dir_path, filenames[i]))
                train_images_labels.append(class_indices[dir_name])
            for i in filename_indices[int(len_filenames * train_size):]:
                valid_images_paths.append(os.path.join(dir_path, filenames[i]))
                valid_images_labels.append(class_indices[dir_name])
    return train_images_labels, train_images_labels, valid_images_labels, valid_images_labels


