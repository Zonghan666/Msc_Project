import tensorflow as tf
from utils import non_max_suppression, get_boxes, draw_boxes, get_boxes_from_yolo, average_iou
from train import read_data_from_batch
import parameter
from nets import yolo_v3
from PIL import Image
import os


def evaluate_model(img_file, model_path, label_file=None, grayscale=False, save_path='evaluate/'):
    """
    predict images using the pretrained model

    :param img_file: the path of a txt file that contains the path of images
    :param model_path: path to the tf checkpoint
    :param label_file: ground true label for the images, if not none, will also give the average iou. Otherwise, only
                       predicted images are given
    :param grayscale: whether images are grayscle
    :param save_path:  path the save the image which have been drawn
    :return: None
    """

    n_classes = parameter._N_CLASSES
    anchors = parameter._ANCHORS
    input_shape = parameter._INPUT_SHAPE
    confidence_threshold = parameter._CONFIDENCE_THRESHOLD
    iou_threshohld = parameter._IOU_THRESHOLD

    with open(img_file, 'r') as x:
        x_path = x.read().splitlines()

    if label_file:
        with open(label_file, 'r') as y:
            y_path = y.read().splitlines()
    else:
        y_path = []

    batch = (x_path, y_path)

    b_x, b_y = read_data_from_batch(batch,
                                    n_classes=n_classes,
                                    anchors=anchors,
                                    resize_size=input_shape)

    # define model in a graph
    graph = tf.Graph()

    with graph.as_default():

        tf_x = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], 1 if grayscale else 3])

        with tf.variable_scope('detector'):
            detections, raw_output = yolo_v3(tf_x, n_classes)

            boxes = get_boxes(detections, n_classes, input_shape)

    with tf.Session(graph=graph) as sess:

        # restore model
        saver = tf.train.Saver()
        saver.restore(sess, save_path=model_path)

        detected_boxes = sess.run(boxes, feed_dict={tf_x:b_x})

        y_pred_boxes = non_max_suppression(detected_boxes, confidence_threshold, iou_threshohld)

        m = b_x.shape[0]

        for i in range(m):
            img = Image.open(x_path[i])
            draw_boxes(y_pred_boxes[i], img, input_shape, greyscale=grayscale)
            img_name = os.path.splitext(x_path[i].split('/')[-1])[0]
            img.save(save_path+img_name+'.jpg', 'JPEG')

    if label_file:
        y_true_boxes = get_boxes_from_yolo(label_file, input_shape)
        avg_iou = average_iou(y_true_boxes, y_pred_boxes)

        print('average iou: %.4f' % avg_iou)

