import numpy as np
import tensorflow as tf
import parameter



def box_iou(box1, box2):
    """
    function to compute iou between two list of boxes

    :param box1: a tensor of shape [grid_size, grid_size, n_anchor, 4]
    :param box2: a tensor of shape [n_object, 4]
                 4 = center_x, center_y, width, height
    :return: iou: a tensor of shape [grid_size, grid_size, n_anchor, n_object]
    """

    # expand dim for broadcasting
    box1 = tf.expand_dims(box1, -2)
    box1_xy = box1[...,0:2]
    box1_wh = box1[...,2:4]
    box1_wh_half = box1_wh / 2
    box1_xy_min = box1_xy - box1_wh_half
    box1_xy_max = box1_xy + box1_wh_half
    box1_area = box1_wh[...,0] * box1_wh[...,1]

    box2_xy = box2[...,0:2]
    box2_wh = box2[...,2:4]
    box2_wh_half = box2_wh / 2
    box2_xy_min = box2_xy - box2_wh_half
    box2_xy_max = box2_xy + box2_wh_half
    box2_area = box2_wh[...,0] * box2_wh[...,1]

    interestion_min = tf.maximum(box1_xy_min, box2_xy_min)
    interestion_max = tf.minimum(box1_xy_max, box2_xy_max)
    interestion_wh = tf.maximum(interestion_max - interestion_min, 0)
    interestion_area = interestion_wh[...,0] * interestion_wh[...,1]

    iou = interestion_area / (box1_area + box2_area - interestion_area + parameter._EPLISION)

    return iou


def yolo_loss(raw_output, y_true, anchors, input_shape, ignore_threshold=0.6):
    """

    :param y_pred: a tensor of shape [batch, grid_size, grid_size, n_anchors, 5+n_classes]
    :param y_true: a array of the same shape as y_pred
    :param anchors: list of anchors used in this feature map
    :param input_shape: shape of the input tensor of the model (416,416)
    :param ignore_threshold: threshold to check whether a box has detected an object
    :return: loss, a tensor of shape (1,)
    """


    grid, raw_pred, box_xy, box_wh = raw_output

    grid_shape = grid.get_shape().as_list()[1:3]

    anchors = np.array(anchors)

    pred_boxes = tf.concat([box_xy, box_wh], axis=-1)
    raw_pred_conf = raw_pred[..., 4:5]
    raw_class_prob = raw_pred[..., 5:]

    true_boxes = y_true[..., 0:4]
    true_obj = object_mask = y_true[..., 4:5]
    true_class_prob = y_true[..., 5:]

    # process raw output to compute loss
    raw_true_xy = true_boxes[..., 0:2] * grid_shape[::-1] - grid
    raw_true_wh = tf.log(true_boxes[..., 2:4] / anchors * input_shape[::-1])
    raw_true_wh = tf.keras.backend.switch(object_mask, raw_true_wh, tf.zeros_like(raw_true_wh))

    # define different scale
    box_loss_scale = 2 - true_boxes[..., 2:3] * true_boxes[..., 3:4]
    obj_loss_scale = 1
    nobj_loss_scale = 1

    # batch size tensor
    m = tf.shape(raw_pred)[0]
    mf = tf.cast(m, raw_pred[0].dtype)

    # initialize the ignore_mask_tensor
    ignore_mask = tf.TensorArray(dtype=y_true[0].dtype, size=1, dynamic_size=True)
    object_mask_bool = tf.cast(object_mask, dtype=tf.int32)

    # iterate all images in a batch
    def loop_body(b, ignore_mask):
        true_boxes_tensor = tf.boolean_mask(tensor=true_boxes[b], mask=object_mask_bool[b, ..., 0])
        iou = box_iou(pred_boxes[b], true_boxes_tensor)
        max_iou = tf.reduce_max(iou, axis=-1)
        ignore_threshold_tensor = tf.constant(ignore_threshold, dtype=max_iou.dtype)
        mask_tensor = tf.cast((max_iou < ignore_threshold_tensor), dtype=true_boxes.dtype)
        ignore_mask = ignore_mask.write(value=mask_tensor, index=b)
        return b+1, ignore_mask
    _, ignore_mask = tf.while_loop(lambda b, *args: b<m, loop_body, [0, ignore_mask])

    # stack all tensor_array
    ignore_mask = ignore_mask.stack()
    ignore_mask = tf.expand_dims(ignore_mask, -1)

    # compute loss(cross_entropy for sigmoid value)
    xy_loss = object_mask * box_loss_scale * tf.keras.backend.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2], from_logits=True)
    wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - raw_pred[...,2:4])
    conf_loss = object_mask * obj_loss_scale * tf.keras.backend.binary_crossentropy(target=true_obj, output=raw_pred_conf, from_logits=True) + \
                (1-object_mask) *  nobj_loss_scale * tf.keras.backend.binary_crossentropy(target=true_obj, output=raw_pred_conf, from_logits= True) * ignore_mask
    class_loss = object_mask * tf.keras.backend.binary_crossentropy(true_class_prob, raw_class_prob, from_logits=True)

    xy_loss = tf.reduce_sum(xy_loss) / mf
    wh_loss = tf.reduce_sum(wh_loss) / mf
    conf_loss = tf.reduce_sum(conf_loss) / mf
    class_loss = tf.reduce_sum(class_loss) / mf

    return xy_loss, wh_loss, conf_loss, class_loss


def get_loss(raw_output, y_true, input_shape):
    """

    :param y_pred: a tensor output by yolov3 model. Should the raw_output: list of tensor
                   [raw_output_0, raw_output_1, raw_output_2]
                   raw_output is the tensor output by the detection layers, which is tuple
                   (x_y_offset, raw_pred, box_xy, box_wh)
    :param y_true: ground true label. should be output of preprocess_true_labels of size
                   [batch, 10647, 5+n_classes]
    #param input_shape: shape of the input_tensor of the model
    :return: loss: total loss of the prediction
    """

    # number of anchors in each grid
    n_anchor = 3

    # number of classes
    n_classes = y_true.get_shape().as_list()[-1] - 5

    raw_output_0, raw_output_1, raw_output_2 = raw_output

    grid_0 = raw_output_0[0].get_shape().as_list()[1:3]
    grid_1 = raw_output_1[0].get_shape().as_list()[1:3]
    grid_2 = raw_output_2[0].get_shape().as_list()[1:3]

    y_true_0, y_true_1, y_true_2 = tf.split(value=y_true,
                                            num_or_size_splits=[np.prod(grid_0)*n_anchor,
                                                                np.prod(grid_1)*n_anchor,
                                                                np.prod(grid_2)*n_anchor],
                                            axis=1)

    y_true_0 = tf.reshape(y_true_0, [-1, grid_0[0], grid_0[1], n_anchor, 5 + n_classes])
    y_true_1 = tf.reshape(y_true_1, [-1, grid_1[0], grid_1[1], n_anchor, 5 + n_classes])
    y_true_2 = tf.reshape(y_true_2, [-1, grid_2[0], grid_2[1], n_anchor, 5 + n_classes])

    xy_loss_0, wh_loss_0, confidence_loss_0, classes_loss_0 = yolo_loss(raw_output_0, y_true_0, parameter._ANCHORS[6:9], input_shape)
    xy_loss_1, wh_loss_1, confidence_loss_1, classes_loss_1 = yolo_loss(raw_output_1, y_true_1, parameter._ANCHORS[3:6], input_shape)
    xy_loss_2, wh_loss_2, confidence_loss_2, classes_loss_2 = yolo_loss(raw_output_2, y_true_2, parameter._ANCHORS[0:3], input_shape)

    xy_loss = xy_loss_0 + xy_loss_1 + xy_loss_2
    wh_loss = wh_loss_0 + wh_loss_1 + wh_loss_2
    confidence_loss = confidence_loss_0 + confidence_loss_1 + confidence_loss_2
    classes_loss = classes_loss_0 + classes_loss_1 + classes_loss_2

    return xy_loss, wh_loss, confidence_loss, classes_loss
