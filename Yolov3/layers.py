import tensorflow as tf
import parameter
from tensorflow.contrib.layers import xavier_initializer


def fixed_padding(input_tensor, kernel_size, mode='CONSTANT'):

    """
    pad the input_tensor along the spatial dimension to achieve the same performance of
    'same' padding but independent of the input size

    :param input_tensor:  A tensor of size [batch, height, width, channel] (NHWC)
    :param kernel_size: The size of the kernel filter or the pool_size. Should be an positive integer
    :param mode: padding mode, default 'CONSTANT' which means padding zero value
    :return: output: A tensor after padding
    '''
    """
    
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    
    output = tf.pad(tensor=input_tensor, 
                    paddings=[[0,0], [pad_beg, pad_end], [pad_beg, pad_end], [0,0]], 
                    mode=mode)
    
    return output


def Conv2d(input_tensor, n_filter, kernel_size, strides, batch_norm=True, activation=True, use_bias=False):

    """

    :param input_tensor:  input tensor of size [batch, height, width, channels]
    :param n_filter:  number of filter used in the conv layer
    :param kernel_size: size of the filter can be an integer or a tuple of 2 elements
    :param strides: strides of the conv operation, can be an integer of tuple of 2 elements
    :param batch_norm: whether to use batch_normalisation default:True
    :param activation: whether to use the leaky_relu activation default:True
    :param use_bias: whether add bias to the kernel default: False
    :return:
    """

    stride = strides[0]
    k = kernel_size[0]
    
    if stride > 1:
        input_tensor = fixed_padding(input_tensor, k)
    
    x = tf.layers.conv2d(inputs=input_tensor,
                         filters=n_filter,
                         kernel_size=kernel_size,
                         strides=strides,
                         padding=('same' if stride == 1 else 'valid'),
                         activation=None,
                         use_bias=use_bias,
                         kernel_initializer=xavier_initializer())
    
    if batch_norm == True:
        # if we use batch_norm layers, we don't use bias terms
        batch_norm_params = {
            'momentum' : parameter._BATCH_NORM_DECAY,
            'epsilon' : parameter._BATCH_NORM_EPSILON,
            'scale' : True,
            'fused' : None,
            'trainable' : False
        }
        
        x = tf.layers.batch_normalization(inputs=x, **batch_norm_params)
        
    if activation == True:
        x = tf.nn.leaky_relu(features=x, alpha= parameter._LEAKY_RELU)
        
    return x


def upsampling(input_tensor, strides=2):
    
    # tf implementation, data_format:NHWC
    height, width = input_tensor.get_shape().as_list()[1:3]
    new_size = (height * strides, width * strides)
    x = tf.image.resize_bilinear(images=input_tensor, size=new_size)
    
    return x


def Residual_block(input_tensor, n_filter):
    
    x = Conv2d(input_tensor, n_filter, (1,1), (1,1))
    x = Conv2d(x, n_filter*2, (3,3), (1,1))
    x = x + input_tensor 
    return x


def Stack_Residual_block(input_tensor, n_filter, n_Repeat):
    x = Residual_block(input_tensor, n_filter)
        
    for i in range(n_Repeat-1):
        x = Residual_block(x, n_filter)
        
    return x


def yolo_block(input_tensor, num_filter):

    """
    yolo convolution layer followed by the darknet53

    :param input_tensor: A tensor fo size [batch, height, width, channels]
    :param num_filter: number of filter
    :return: route: a feature map used in the following conv layers
             x: detection feature map
    """

    x = Conv2d(input_tensor=input_tensor,
               n_filter=num_filter, 
               kernel_size=(1,1), 
               strides=(1,1))
        
    x = Conv2d(input_tensor=x,
               n_filter=num_filter * 2, 
               kernel_size=(3,3),
               strides=(1,1))
        
    x = Conv2d(input_tensor=x, 
               n_filter=num_filter,
               kernel_size=(1,1),
               strides=(1,1))
        
    x = Conv2d(input_tensor=x,
               n_filter=num_filter * 2,
               kernel_size=(3,3),
               strides=(1,1))
        
    x = Conv2d(input_tensor=x,
               n_filter=num_filter, 
               kernel_size=(1,1), 
               strides=(1,1))
        
    route = x
        
    x = Conv2d(input_tensor=x,
               n_filter=num_filter * 2,
               kernel_size=(3,3), 
               strides=(1,1))
        
    return route, x


def detection_layer(input_tensor, n_classes, anchors, img_size):

    """

    :param input_tensor: A tensor of size [batch, height, width, channels] (NHWC).
    :param n_classes: number of predicted classes
    :param anchors: List of tuple consist of size of the anchors (height, width)
    :param img_size: Size of the original input image
    :param cal_loss: whether used to calculate the loss
    :return: predictions: A tensor of size [batch, N*N, n_anchor*(5 + n_classes))]
                          (N:size of the feature map,
                          n_anchor: number of the anchor used in this feature map
                          n_classes: number of classes in this model)

                          location values are in range [0,1] (the same to yolo format)
    :return: raw_prediction: A tensor of size [batch, grid_size, grid_size, n_anchors, 5+n_classes
                             this tensor is the raw output of the neural net.
    """
    n_anchors = len(anchors)

    # detection conv layers. No batch_norm and activation(linear)
    predictions = Conv2d(input_tensor=input_tensor,
                         n_filter=(n_anchors * (5 + n_classes)),
                         kernel_size=(1,1),
                         strides=(1,1),
                         activation=False,
                         batch_norm=False,
                         use_bias=True)

    # get size of the feature map (height, width)
    grid_size = predictions.get_shape().as_list()[1:3]

    # save raw output for loss calculation
    raw_predictions = predictions
    raw_predictions = tf.reshape(raw_predictions, [-1, grid_size[0], grid_size[1], n_anchors, 5+n_classes])

    # total number of grids
    dim = grid_size[0] * grid_size[1]

    # total attributes of the bounding box (x,y, height, width, confidence)
    bbox_attrs = 5 + n_classes

    predictions = tf.reshape(predictions, [-1, n_anchors*dim, bbox_attrs])

    stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])

    box_centers, box_sizes, confidence, classes = tf.split(value=predictions,
                                                           num_or_size_splits=[2,2,1,n_classes],
                                                           axis=-1)

    # apply sigmoid on x, y, confidence and classes(multiple label detection)
    box_centers = tf.nn.sigmoid(box_centers)
    confidence = tf.nn.sigmoid(confidence)
    classes = tf.nn.sigmoid(classes)

    # construct a offset grid for the feature map
    grid_x = tf.range(grid_size[0], dtype=tf.float32)
    grid_y = tf.range(grid_size[1], dtype=tf.float32)

    a, b = tf.meshgrid(grid_x, grid_y)

    x_offset = tf.reshape(a, (-1,1))
    y_offset = tf.reshape(b, (-1,1))

    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, n_anchors]), [1, -1, 2])

    # output exact position of the centre
    box_centers = box_centers + x_y_offset
    box_centers = box_centers * stride

    # output exact size of the object
    anchors = tf.tile(anchors, [dim, 1])
    box_sizes = tf.exp(box_sizes) * anchors

    # normalize the location value in range [0,1]
    box_centers /= img_size
    box_sizes /= img_size

    predictions = tf.concat([box_centers, box_sizes, confidence, classes], axis=-1)

    grid = tf.reshape(x_y_offset, [-1, grid_size[0], grid_size[1], n_anchors, 2])
    box_xy = tf.reshape(box_centers, [-1, grid_size[0], grid_size[1], n_anchors, 2])
    box_wh = tf.reshape(box_sizes, [-1, grid_size[0], grid_size[1], n_anchors, 2])

    raw_data = (grid, raw_predictions, box_xy, box_wh)

    return predictions, raw_data

