import os
import random

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
    return train_images_paths, train_images_labels, valid_images_paths, valid_images_labels


