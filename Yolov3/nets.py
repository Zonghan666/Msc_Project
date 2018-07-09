from layers import Conv2d, Stack_Residual_block, yolo_block, upsampling, detection_layer
import parameter
import tensorflow as tf


def darkNet53(input_tensor, classifier=False, n_classes=None):
    """
    darkNet:feature extractor of the yolov3 model or classifier

    :param input_tensor: a tensor of size [batch, input_shape[0], input_shape[0], 1 if grayscale else 3]
    :param classifier:  if the model perform as a classifier or a feature detector
    :param n_classes: number of predicted classes if classifier == True

    :return: if classifier == True, return 1-D vector (n_classes
             if classifier == False, return 3 feature maps
    """

    x = Conv2d(input_tensor=input_tensor, 
               n_filter=32, 
               kernel_size=(3,3),
               strides=(1,1))
    
    x = Conv2d(input_tensor=x,
               n_filter=64, 
               kernel_size=(3,3), 
               strides=(2,2))
    
    x = Stack_Residual_block(input_tensor=x, 
                             n_filter=32, 
                             n_Repeat=1,)
    
    x = Conv2d(input_tensor=x,
               n_filter=128,
               kernel_size=(3,3),
               strides=(2,2))
    
    x = Stack_Residual_block(input_tensor=x,
                             n_filter=64,
                             n_Repeat=2)
    
    x = Conv2d(input_tensor=x,
               n_filter=256,
               kernel_size=(3,3),
               strides=(2,2))
    
    x = Stack_Residual_block(input_tensor=x, 
                             n_filter=128,
                             n_Repeat=8)
    
    # first feature map
    route0 = x
    
    x = Conv2d(input_tensor=x,
               n_filter=512,
               kernel_size=(3,3),
               strides=(2,2))
    
    x = Stack_Residual_block(input_tensor=x,
                             n_filter=256, 
                             n_Repeat=8)
    
    # second feature map
    route1 = x
    
    x = Conv2d(input_tensor=x, 
               n_filter=1024,
               kernel_size=(3,3),
               strides=(2,2))
    
    x = Stack_Residual_block(input_tensor=x,
                             n_filter=512,
                             n_Repeat=4)
    
    # classifier == False mean only feature extraction task is perfromed
    if not classifier:
        return route0, route1, x
    
    else:
        pool_height = x.get_shape()[1].value
        pool_width = x.get_shape()[2].value
        
        # global average pooling
        x = tf.layers.average_pooling2d(inputs=x, 
                                        pool_size=(pool_height, pool_width), 
                                        strides=(1,1),
                                        name='average_pooling')
        
        # reshape logits to 2D vector
        x = tf.reshape(tensor=x, shape=[-1, x.get_shape()[-1].value])
        
        # perform linear transform without softmax
        logits = tf.layers.dense(inputs=x, units=n_classes)
        
        return logits, x


def yolo_v3(input_tensor, n_classes):
    """

    :param input_tensor: a tensor of size [batch, input_shape[0], input_shape[0], 1 if grayscale else 3]
    :param n_classes:  number of predicted classes

    :return:  detections: a tensor of size [batch, n_grids, (5 + n_classes)]

              raw_output: a tuple of three tensors used to compute loss
                         (raw_output_0, raw_output_1, raw_output_2)
                         corresponding to three feature maps
    """

    img_size = input_tensor.get_shape().as_list()[1:3]
    
    # normalize pixels value to range [0, 1]
    input_tensor = input_tensor / 255
    
    with tf.variable_scope('darknet53'):
        route_0, route_1, x = darkNet53(input_tensor=input_tensor, classifier=False)

    with tf.variable_scope('yolo_v3'):
        
        # first detection
        route, x = yolo_block(input_tensor=x, num_filter=512)

        detection_0, raw_output_0 = detection_layer(input_tensor=x,
                                                    anchors= parameter._ANCHORS[6:9],
                                                    img_size=img_size,
                                                    n_classes=n_classes)
        
        # second detection
        x = Conv2d(input_tensor=route,
                   n_filter=256,
                   kernel_size=(1,1), 
                   strides=(1,1), )

        x = upsampling(input_tensor=x, strides=2)
        x = tf.concat([x, route_1], axis=3)
        
        route, x= yolo_block(input_tensor=x, num_filter=256)
        
        detection_1, raw_output_1 = detection_layer(input_tensor=x,
                                                    anchors= parameter._ANCHORS[3:6],
                                                    img_size=img_size,
                                                    n_classes=n_classes)
        
        # last detection
        x = Conv2d(input_tensor=route, n_filter=128,
                   kernel_size=(1,1),
                   strides=(1,1),)

        x = upsampling(input_tensor=x, strides=2)
        x = tf.concat([x, route_0], axis=3)
        
        _, x = yolo_block(input_tensor=x, num_filter=128)
        
        detection_2, raw_output_2 = detection_layer(input_tensor=x,
                                                    anchors= parameter._ANCHORS[0:3],
                                                    img_size=img_size,
                                                    n_classes=n_classes)

        detections = tf.concat([detection_0, detection_1, detection_2], axis=1)
        raw_output = (raw_output_0, raw_output_1, raw_output_2)

        return detections, raw_output
        

