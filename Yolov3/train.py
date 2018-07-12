import numpy as np
from utils import preprocess_batch_labels
from files_helper import annotation_reader
from PIL import Image
from sklearn.utils import shuffle, resample
from mpmath import math2
import tensorflow as tf
import parameter
from nets import yolo_v3
from utils import get_boxes, get_boxes_from_yolo, non_max_suppression, average_iou, load_weight
from loss import get_loss


def batch_generator(x_file, y_file, batch_size):

    with open(x_file, 'r') as x:
        x_path = x.read().splitlines()

    with open(y_file, 'r') as y:
        y_path = y.read().splitlines()

    assert (len(x_path) == len(y_path))

    m = len(x_path)
    num_split = math2.ceil( m / batch_size)

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
            # convert grayscale to rgb
            # img = np.tile(img, [1,1,3])
        
        img = np.expand_dims(img, axis=0)
        b_x.append(img)

    b_x = np.concatenate(b_x, axis=0)

    if label_path:
        for path in label_path:
            label = annotation_reader(path)
            label = preprocess_batch_labels(label, resize_size, anchors, n_classes)
            b_y.append(label)

        b_y = np.concatenate(b_y, axis=0)

    return b_x, b_y


def train_model(x_train_file, y_train_file, x_val_file, y_val_file, grayscale=False, model_path=None, save_path='/checkpoint'):
    """

    :param x_train_file: training images txt that contains all paths of images
    :param y_train_file: training labels txt
    :param x_val_file: validation images txt
    :param y_val_file: validation labels txt
    :param grayscale: whether images are grayscale
    :param model_path: path of the model checkpoint
    :param save_path: specify path to save model
    :return:
    """

    batch_size = parameter._BATCH_SIZE
    input_shape = parameter._INPUT_SHAPE
    y_dim = parameter._DIM
    n_classes = parameter._N_CLASSES
    learning_rate = parameter._LEARNING_RATE
    epochs = parameter._EPOCH
    anchors = parameter._ANCHORS
    confidence_threshold = parameter._CONFIDENCE_THRESHOLD
    iou_threshold = parameter._IOU_THRESHOLD
    yolo_weight = parameter._YOLOV3_WEIGHTS

    # generate validation set
    val_batch = []

    with open(x_val_file) as x:
        x_val = x.read().splitlines()

    with open(y_val_file) as y:
        y_val = y.read().splitlines()

    val_batch.append(x_val)
    val_batch.append(y_val)

    x_val, y_val = read_data_from_batch(val_batch, input_shape, anchors, n_classes)

    y_val_true_boxes = get_boxes_from_yolo(y_val_file, input_shape)

    # define model

    with tf.Graph().as_default():

        tf_x = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], 1 if grayscale else 3])
        tf_y = tf.placeholder(tf.float32, [None, y_dim, 5 + n_classes])

        with tf.variable_scope('detector'):
            detections, raw_output = yolo_v3(tf_x, n_classes)

            boxes = get_boxes(detections, n_classes, input_shape)

            xy_loss, wh_loss, conf_loss, cls_loss = get_loss(raw_output, tf_y, input_shape)

            loss = xy_loss + wh_loss + conf_loss + cls_loss

            train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

        with tf.Session() as sess:

            saver = tf.train.Saver()

            # initialize
            if model_path:
                saver.restore(sess, save_path=model_path)

            else:
                # use pre-train weight:
                # load_ops = load_weight(var_list=tf.global_variables(scope='detector'), weight_file=yolo_weight, for_training=True)
                # sess.run(load_ops)
                # uninitialized_variables = [str(v, encoding='utf-8') for v in sess.run(tf.report_uninitialized_variables())]
                # variables = [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]
                # sess.run(tf.variables_initializer(variables))

                sess.run(tf.global_variables_initializer())

            # start training
            print('start training')

            for epoch in range(epochs):

                batches = batch_generator(x_train_file, y_train_file, batch_size)
                n_batch = len(batches)
                train_loss = 0

                for batch in batches:
                    b_x, b_y = read_data_from_batch(batch, input_shape, anchors, n_classes)

                    _, batch_loss = sess.run([train_op, loss], feed_dict={tf_x: b_x, tf_y: b_y})

                    train_loss += batch_loss

                train_loss /= n_batch

                val_loss, y_val_pred_boxes = sess.run([loss, boxes], feed_dict={tf_x: x_val, tf_y: y_val})

                y_val_pred_boxes = non_max_suppression(y_val_pred_boxes, confidence_threshold, iou_threshold)

                avg_iou = average_iou(y_val_true_boxes, y_val_pred_boxes)

                print('epoch:', epoch, '| training loss: %.4f' % train_loss, '|val loss: %.4f' % val_loss, '| val iou: %.4f' % avg_iou)

            print('training finishes, saving model.....')

            saver.save(sess, save_path=save_path)

        print('End')
