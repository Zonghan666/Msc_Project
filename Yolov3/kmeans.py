import numpy as np
from files_helper import annotation_reader


class YOLO_Kmeans:

    def __init__(self, n_cluster, filename):
        self.n_cluster = n_cluster
        self.filename = filename

    def iou(self, boxes, cluster):
        n = boxes.shape[0]
        k = self.n_cluster

        boxes_area = boxes[:, 0] * boxes[:, 1]
        boxes_area = boxes_area.repeat(k)
        boxes_area = np.reshape(boxes_area, [n,k])

        cluster_area = cluster[:, 0] * cluster[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, [n,k])

        boxes_w_matrix = boxes[:, 0].repeat(k)
        boxes_w_matrix = np.reshape(boxes_w_matrix, [n,k])
        cluster_w_matrix = np.reshape(np.tile(cluster[:, 0], (1,n)), [n,k])
        min_w_matrix = np.minimum(cluster_w_matrix, boxes_w_matrix)

        boxes_h_matrix = boxes[:, 1].repeat(k)
        boxes_h_matrix = np.reshape(boxes_h_matrix, [n,k])
        cluster_h_matrix = np.reshape(np.tile(cluster[:, 1], [1,n]), [n,k])
        min_h_matrix = np.minimum(cluster_h_matrix, boxes_h_matrix)

        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (boxes_area + cluster_area - inter_area)

        return result

    def avg_iou(self, boxes, cluster):
        accuarcy = np.mean(np.max(self.iou(boxes, cluster), axis=1))
        return accuarcy

    def kmeans(self, boxes, k, dist=np.median):
        n_boxes = boxes.shape[0]
        last_nearst = np.zeros([n_boxes, ])
        np.random.seed()
        clusters = boxes[np.random.choice(n_boxes, k, replace=False)]

        while True:
            distance = 1 - self.iou(boxes, clusters)

            current_nearst = np.argmin(distance, axis=1)
            if (last_nearst == current_nearst).all():
                break

            for cluster in range(k):
                clusters[cluster] = dist(boxes[(current_nearst == cluster)], axis=0)

            last_nearst = current_nearst

        return clusters

    def read_boxes(self):
        with open(self.filename, 'r') as file:
            label_path = file.read().splitlines()

        boxes = []

        for path in label_path:
            label = annotation_reader(path)
            box = label[0, :, 3:]
            boxes.append(box)

        boxes = np.concatenate(boxes, axis=0)

        return boxes

    def generate_anchor(self):
        all_boxes = self.read_boxes()
        anchors = self.kmeans(all_boxes, k=self.n_cluster)

        # sort the anchors according to their area
        anchors_area = anchors[:,0] * anchors[:,1]
        order = np.argsort(anchors_area)
        anchors = anchors[order]

        print('K anchors: \n {}'.format(anchors))
        print('Accuracy: {:.2f}%'.format(self.avg_iou(all_boxes, anchors) * 100))

        return anchors
