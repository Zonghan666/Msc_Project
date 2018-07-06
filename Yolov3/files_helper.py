import xml.dom.minidom
import numpy as np


def annotation_reader(file_path):
    """
    read all object annotations of VOC format from the xml file and convert them to YOLO format
    my task is not to detect a single class "nipple" so class{0:others, 1:nipple}

    :param file_path: file path of the  xml
    :return: narray of shape [1, n_object, 5] (yolo format)
             5 = class + center_x, center_y, width, height (all range in [0,1])
             1 stands for one image
    """

    dom = xml.dom.minidom.parse(file_path)
    root = dom.documentElement

    #load the size of the image
    size_xml = root.getElementsByTagName('size')[0]
    img_width = float(size_xml.getElementsByTagName('width')[0].firstChild.data)
    img_height = float(size_xml.getElementsByTagName('height')[0].firstChild.data)

    #iterate the object
    objects = root.getElementsByTagName('object')

    result = []

    for obj in objects:
        object_name = obj.getElementsByTagName('name')[0].firstChild.data
        xmin = float(obj.getElementsByTagName('xmin')[0].firstChild.data)
        ymin = float(obj.getElementsByTagName('ymin')[0].firstChild.data)
        xmax = float(obj.getElementsByTagName('xmax')[0].firstChild.data)
        ymax = float(obj.getElementsByTagName('ymax')[0].firstChild.data)

        obj_width = xmax - xmin
        obj_height = ymax - ymin

        center_x = xmin + obj_width / 2
        center_y = ymin + obj_height / 2

        #normalize
        center_x = center_x / img_width
        center_y = center_y / img_height
        obj_width = obj_width / img_width
        obj_height = obj_height / img_height

        #convert class to int
        if object_name == 'nipple':
            cls = 0
            
        

        result.append((cls, center_x, center_y, obj_width, obj_height))


    result = np.array(result)

    #expand the dims
    result = np.expand_dims(result, 0)


    return result
