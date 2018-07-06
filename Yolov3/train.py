import numpy as np
from utils import preprocess_batch_labels
from files_helper import annotation_reader
from PIL import Image
import math
from sklearn.utils import shuffle, resample


def batch_generator(x_file, y_file, batch_size):

    with open(x_file, 'r') as x:
        x_path = x.read().splitlines()

    with open(y_file, 'r') as y:
        y_path = y.read().splitlines()

    assert (len(x_path) == len(y_path))

    m = len(x_path)
    num_split = math.ceil( m / batch_size)

    add_num = num_split * batch_size - m

    if add_num != 0:
        x_add, y_add = resample(x_path, y_path, n_samples=add_num)
        for i in range(add_num):
            x_path.append(x_add[i])
            y_path.append(y_add[i])

    x_path, y_path = shuffle(x_path, y_path)

    result = []

    for i in range(num_split):
        idx_start = i * batch_size
        idx_end = (i+1) * batch_size

        batch_x = x_path[idx_start:idx_end]
        batch_y = y_path[idx_start:idx_end]

        result.append((batch_x, batch_y))

    return result


def read_data_from_batch(batch, resize_size, anchors, n_classes):

    img_path, label_path = batch

    b_x = []
    b_y = []

    for path in img_path:
        img = Image.open(path)
        img = img.resize(size=resize_size)
        img = np.array(img)
        
        # deal with grey scale image
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
        
        img = np.expand_dims(img, axis=0)
        b_x.append(img)

    for path in label_path:
        label = annotation_reader(path)
        label = preprocess_batch_labels(label, resize_size, anchors, n_classes)
        b_y.append(label)

    b_x = np.concatenate(b_x, axis=0)
    b_y = np.concatenate(b_y, axis=0)

    return b_x, b_y

