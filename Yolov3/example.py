from kmeans import YOLO_Kmeans
import numpy as np
import parameter
from train import train_model
from evaluate import evaluate_model
import tensorflow as tf

# specify the path to these four txt files containing the path to training images, validation images, training labels
# and validation labels.
img_train = ''
img_val = ''
label_train = ''
label_val = ''

# generate custom anchor box for dataset, and save as txt file
Kmean = YOLO_Kmeans(n_cluster=9, filename=label_train)
anchor_boxes = np.ceil(Kmean.generate_anchor() * parameter._INPUT_SHAPE)
np.savetxt('anchor_box.txt', anchor_boxes)

# call the train function
with tf.device('device:GPU:0'):
    # train model
    train_model(img_train, label_train, img_val, label_val, model_path=None, save_path='checkpoint/')

    # evaluate model
    evaluate_model(img_val, model_path='checkpoint/', label_file=label_val, save_path=None)
