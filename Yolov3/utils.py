import tensorflow as tf
import numpy as np
from PIL import ImageDraw
import parameter
from files_helper import annotation_reader


def get_boxes_from_yolo(file, img_shape):
    """

    :param file: path of the label of yolo format
    :param img_shape: shape of the images
    :return: boxes, an array containing the cls, x0, y0, x1, y1 of the boxes
    """

    with open(file, 'r') as y:
        label_path = y.read().splitlines()

    boxes = []
    cls = []

    for path in label_path:
        label = annotation_reader(path)
        boxes.append(label[..., 1:])
        cls.append(label[..., 0:1])

    boxes = np.concatenate(boxes, axis=0)
    cls = np.concatenate(cls, axis=0)

    x = boxes[..., 0:1] * img_shape[0]
    y = boxes[..., 1:2] * img_shape[1]
    width = boxes[..., 2:3] * img_shape[0]
    height = boxes[..., 3:4] * img_shape[1]

    x0 = x - width / 2
    y0 = y - height / 2
    x1 = x + width / 2
    y1 = y + height / 2

    boxes = np.concatenate([x0, y0, x1, y1], axis=-1)

    boxes = np.concatenate([cls, boxes], axis=-1)

    return boxes


def get_boxes(detections, n_classes, input_shape):
    """

    :param detections: A tensor output from yolo_v3 model of size [batch, 10647, (5+n_classes)]
                      10647 = (13*13 + 26*26 + 52*52) * 3
                      5 = center_x + center_y = width + height + object_confidence
                      location information are values lyring in range [0,1]
    :param n_classes: number of predicted classes
    :param input_shape: shape of the input_img [width, height], should be [416,416]
    :return: detections: detections which contains the exact position and size of the bounding boxes
    """

    center_x, center_y, width, height, confidence, classes = tf.split(value=detections,
                                                                      num_or_size_splits=[1, 1, 1, 1, 1, n_classes],
                                                                      axis=-1)

    # conver the yolo value to exact value according to the shape of input image
    center_x *= input_shape[0]
    center_y *= input_shape[1]
    width *= input_shape[0]
    height *= input_shape[1]

    # compute the coordinates of the bounding box
    w2 = width / 2
    h2 = height / 2
    x0 = center_x - w2
    x1 = center_x + w2
    y0 = center_y - h2
    y1 = center_y + h2

    boxes = tf.concat([x0, y0, x1, y1], axis=-1)
    detections = tf.concat([boxes, confidence, classes], axis=-1)

    return detections


def get_iou(box1, box2):

    """
    Compute intersection over Union value for 2 bounding box

    :param box1: array of 4 values(top left and bottom right coordinates): [x0, y0, x1, y1]
    :param box2: array of 4 values
    :return: IoU: Intersection over Union of the two boxes
    """
    
    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2
    
    overlapped_x0 = max(b1_x0, b2_x0)
    overlapped_x1 = min(b1_x1, b2_x1)
    overlapped_y0 = max(b1_y0, b2_y0)
    overlapped_y1 = min(b1_y1, b2_y1)
    
    overlapped_area = (overlapped_x1 - overlapped_x0) * (overlapped_y1 - overlapped_y0)
    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)
    
    IoU = overlapped_area / (b1_area + b2_area - overlapped_area + 1e-5)
    
    return IoU


def average_iou(bboxes_true, bboxes_pred):
    """
    compute averge iou between ground true boxes and predicted boxes
    :param bboxes_true: true label, output of get_boxes_from_yolo
    :param bboxes_pred:  predicted label, output of non_max_suppression
    :return: avg_iou
    """

    m_images = bboxes_true.shape[0]

    avg_iou = 0

    for i in range(m_images):
        y_true = bboxes_true[i]
        y_pred = bboxes_pred[i]
        n_obj = y_true.shape[0]
        obj_iou = 0

        # iterate all true boxes in an image
        for obj in y_true:
            # box_true = y_true[1:5]
            box_true = (obj[1], obj[2], obj[3], obj[4])
            iou = 0

            # iterate all predected box and find the best one
            for _, boxes in y_pred.items():
                for box_pred, _ in boxes:
                    box_pred = (box_pred[0], box_pred[1], box_pred[2], box_pred[3])
                    iou = max(iou, get_iou(box_true, box_pred))

            obj_iou += iou

        obj_iou /= n_obj
        avg_iou += obj_iou

    avg_iou /= m_images

    return avg_iou


def non_max_suppression(detection, confidence_threshold=0.5, iou_threshold=0.4):
    """

    :param detection: numpy array of size [n_grid, (5+n_classes)]
    :param confidence_threshold: a value in range (0,1] that determines whether it is a valid bounding box
    :param iou_threshold: IoU threshold for the non_max_suppression
    :return: result: dict of format {class:[n_boxes:(boxes, score)]}
                     boxes:x,y,width, heigth
                     score:confidence score
    """

    # iter each image in the batch
    
    output = []
    
    for i, img_pred in enumerate(detection):
        result = {}
        # now the img_pred is of size [n_grid, 5+n_classes]
        mask = img_pred[:, 4] >= confidence_threshold
        img_pred = img_pred[mask]
        
        bbox_attrs = img_pred[:, 0:5]
        classes = img_pred[:, 5:]
        classes = np.argmax(classes, axis=-1)
        
        unique_class = list(set(classes.reshape(-1)))
        
        for cls in unique_class:
            
            # select boxes that are predicted as 'cls' class
            cls_mask = (classes == cls)
            cls_boxes = bbox_attrs[cls_mask]
            
            # sort the bbox_attrs according to the confidence score
            score_mark = cls_boxes[:,-1].argsort()[::-1]
            cls_boxes = cls_boxes[score_mark]
            
            # split the position information and the score of the boxes
            cls_scores = cls_boxes[:, -1]
            cls_boxes = cls_boxes[:, :-1]
            
            while len(cls_boxes) > 0:
                box = cls_boxes[0]
                score = cls_scores[0]
                if not cls in result:
                    result[cls] = []
                result[cls].append((box, score))
                cls_boxes = cls_boxes[1:]
                cls_scores = cls_scores[1:]
                ious = np.array([get_iou(box, x) for x in cls_boxes])
                iou_mask = ious < iou_threshold
                cls_boxes = cls_boxes[iou_mask]
                cls_scores = cls_scores[iou_mask]

        output.append(result)
    
    return output


def load_weight(var_list, weight_file, for_training=False):
    """
    function to load yolov3.weight to the tf model
    if for_training == False, will load all the weights
    if for_training == True, the weights of the detection layers will not be loaded


    :param var_list: list of unitialised variable, should be tf.global_variable()
    :param weight_file:  yolov3.weights
    :param for_training: whether used for transfer learning
    :return:
    """

    # The first 5 values are header information
    # 1. Major version number
    # 2. Minor Version Number
    # 3. Subversion number
    # 4,5. IMages seen

    with open(weight_file) as fp:
        _ = np.fromfile(fp, dtype=np.int32, count=5)
        weight = np.fromfile(fp, dtype=np.float32)
    
    # pointer to the weight from the file
    ptr = 0
    
    # pointer to the variable_list from the model
    i = 0
    assign_op = []
    
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i+1]
        
        # when we process a conv2d layer
        if 'conv2d' in var1.name.split('/')[2]:
            # check type of the next layers:
            if 'batch_normalization' in var2.name.split('/')[2]:
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                
                # the order is different
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.get_shape().as_list()
                    n_params = np.prod(shape)
                    var_weights = weight[ptr : ptr+n_params].reshape(shape)
                    ptr = ptr + n_params
                    assign_op.append(tf.assign(ref=var, value=var_weights))
                    
                # move on the var_list pointer by 4, beacuse we have loaded 4 variables
                i = i + 4


                # load the conv filter weights
                shape = var1.get_shape().as_list()
                n_params = np.prod(shape)

                var_weights = weight[ptr:ptr + n_params].reshape((shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, [2, 3, 1, 0])
                ptr += n_params
                assign_op.append(tf.assign(ref=var1, value=var_weights))
                i += 1
            
            elif 'bias' in var2.name.split('/')[3]:

                # bias terms only appear in the final detection layers. So, if for_training == True:
                # the bias weights as well as the following convs weights will not be loaded
                # corrseponding to my model, these three detection layers should be:

                # 1.
                # 'detector/yolo_v3/conv2d_6/kernel'
                # 'detector/yolo_v3/conv2d_6/bias'

                # 2.
                # 'detector/yolo_v3/conv2d_14/kernel'
                # 'detector/yolo_v3/conv2d_14/bias'

                # 3.
                # 'detector/yolo_v3/conv2d_22/kernel'
                # 'detector/yolo_v3/conv2d_22/bias'

                bias = var2
                shape = bias.get_shape().as_list()
                n_params = np.prod(shape)
                bias_weights = weight[ptr:ptr + n_params].reshape(shape)
                ptr = ptr + n_params

                if for_training == False:
                    assign_op.append(tf.assign(ref=bias, value=bias_weights))

                i = i + 1

                shape = var1.get_shape().as_list()
                n_params = np.prod(shape)

                var_weights = weight[ptr:ptr + n_params].reshape((shape[3], shape[2], shape[0], shape[1]))
                var_weights = np.transpose(var_weights, [2, 3, 1, 0])
                ptr += n_params

                if for_training == False:
                    assign_op.append(tf.assign(ref=var1, value=var_weights))
                
                # move on the var_list pointer by 1, beacuse we have loaded 1 variable
                i = i + 1
                
            # finally we load the weight of the conv filter(kernel):
            # size of kernel in tf model should be [kernel_size[0], kernel_size[1], in_channels, out_channels]
            # weight value of the kernel stored in file is[out_channels * in_channels * kernel_size[0] * kernel_size[1]]
            # which is 1-D vector
            
            # shape = var1.get_shape().as_list()
            # n_params = np.prod(shape)
            #
            # var_weights = weight[ptr:ptr + n_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            # var_weights = np.transpose(var_weights, [2, 3, 1, 0])
            # ptr += n_params
            # assign_op.append(tf.assign(ref=var1, value=var_weights))
            # i += 1
            
    return assign_op


def draw_boxes(boxes, img, detection_size, greyscale=False):
    draw = ImageDraw.Draw(img)

    for cls, bboxes in boxes.items():
        color = tuple(np.random.randint(0, 256, 3))
        for box, score in bboxes:
            original_img_size = np.array(img.size)
            current_img_size = np.array(detection_size)
            ratio = original_img_size / current_img_size
            box = list((box.reshape(2,2) * ratio).reshape(-1))
            
            if greyscale == False:
                draw.rectangle(xy=box, outline=color)
            elif greyscale == True:
                draw.rectangle(xy=box, outline=80000)


def preprocess_true_labels(true_labels, input_shape, grid_shape, anchors, n_classes):
    """
    process true labels to training input format for each feature map

    TODO: each image in a batch contain different numbers of object, can't handle it for the moment

    :param true_labels: array, shape = (batch, n_object, 5) in my cases, I only have one object in my dataset
    :param input_shape: shape of the input image of the model (should be 416, 416)
    :param grid_shape: shape of the feature map, output of the yolo_v3 model should one of (13*13, 26*26, 52*52)
    :param anchors: list used anchors (n, 2) (width,height), usually n = 3
    :param n_classes: number of predicted classes
    :return: y_true:: array of shape [batch, grid_shape[1], grid_shape[0], 5 + n_classes]
    """

    # convert to np.array

    n_anchor = len(anchors)

    # convert values of yolo format to the absolute values
    boxes_xy = true_labels[..., 1:3] * input_shape
    boxes_wh = true_labels[..., 3:] * input_shape

    # initialize the result
    m = true_labels.shape[0]

    y_true = np.zeros([m, grid_shape[1], grid_shape[0], n_anchor, (5 + n_classes)])

    anchors = np.array(anchors)
    anchors_max = anchors / 2
    anchors_min = -anchors_max

    # iterate all image in a batch
    for b in range(m):
        obj_wh = boxes_wh[b]
        obj_wh = np.expand_dims(obj_wh, -2)
        boxes_max = obj_wh / 2
        boxes_min = -boxes_max

        # compute iou of anchor box and true box
        interestion_min = np.maximum(anchors_min, boxes_min)
        interestion_max = np.minimum(anchors_max, boxes_max)
        interestion_wh = np.maximum(interestion_max - interestion_min, 0)
        interestion_area = interestion_wh[..., 0] * interestion_wh[..., 1]
        anchors_area = anchors[..., 0] * anchors[..., 1]
        box_area = obj_wh[..., 0] * obj_wh[..., 1]
        iou = interestion_area / (anchors_area + box_area - interestion_area + parameter._EPLISION)

        # get the idx of anchor box with highest iou
        best_iou_idx = np.argmax(iou, axis=-1)

        for t, n in enumerate(best_iou_idx):

            strides = input_shape / grid_shape
            center_x_y = (boxes_xy[b, t, 0], boxes_xy[b, t, 1])
            x_idx, y_idx = center_x_y // strides

            x_idx = x_idx.astype(int)
            y_idx = y_idx.astype(int)

            # set
            c = true_labels[b, t, 0].astype('int32')
            y_true[b, y_idx, x_idx, n, 0:4] = true_labels[b, t, 1:5]
            y_true[b, y_idx, x_idx, n, 4] = 1
            y_true[b, y_idx, x_idx, n, 5 + c] = 1

    return y_true


def preprocess_batch_labels(true_labels, input_shape, anchors, n_classes):

    input_shape = np.array(input_shape)
    grid_0 = (input_shape / 32).astype(int)
    grid_1 = (grid_0 * 2).astype(int)
    grid_2 = (grid_1 * 2).astype(int)

    m = true_labels.shape[0]

    y0 = preprocess_true_labels(true_labels, input_shape, np.array(grid_0), anchors[6:9], n_classes)
    y1 = preprocess_true_labels(true_labels, input_shape, np.array(grid_1), anchors[3:6], n_classes)
    y2 = preprocess_true_labels(true_labels, input_shape, np.array(grid_2), anchors[0:3], n_classes)

    y0 = y0.reshape([m, -1, 5+n_classes])
    y1 = y1.reshape([m, -1, 5+n_classes])
    y2 = y2.reshape([m, -1, 5+n_classes])

    y = np.concatenate([y0,y1,y2], axis=1).astype(float)

    return y














