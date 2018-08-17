# convolution layer
_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

# anchors
# yolov3
#_ANCHORS = [(10., 13.), (16., 30.), (33., 23.), (30., 61.), (62., 45.), (59., 119.), (116., 90.), (156., 198.), (373., 326.)]

# nipple(DDSM)
#_ANCHORS = [(29., 25.), (36., 22.), (33., 30.), (40., 26.), (50., 27.), (43., 31.), (58., 29.), (46., 38.), (58., 36.)]

# (608, 608)
# _ANCHORS = [(23., 25.), (29., 26.), (26., 31.), (26., 39.), (34., 30.), (30., 35.), (40., 34.), (35., 40.), (46., 43.)]

# Optimam(736, 736)
_ANCHORS = [(29., 29.), (27., 37.), (36., 33.), (32., 40.), (37., 43.), (44., 38.), (41., 50.), (51., 46.), (56., 55.)]


# commom
_EPLISION = 1e-08

# classes
_CLASSES = {'nipple': 0, 'non-nipple': 1}

# classes
#_CLASSES = {'dog': 16}

# inversed classes
_INVERSED_CLASSES = {v: k for k, v in _CLASSES.items()}

# shape
_INPUT_SHAPE = (736, 736)
y_dim_0 = _INPUT_SHAPE[0] / 32
y_dim_1 = y_dim_0 * 2
y_dim_2 = y_dim_1 * 2
_DIM = int((y_dim_0**2 + y_dim_1**2 + y_dim_2**2) * len(_ANCHORS) / 3)

# training parameter
_LEARNING_RATE = 7e-5
_BATCH_SIZE = 8
_N_CLASSES = len(_CLASSES)
_EPOCH = 5

# non_max_suppression
_CONFIDENCE_THRESHOLD = 0.4
_IOU_THRESHOLD = 0.1

# yolov3 weight path
_YOLOV3_WEIGHTS = 'weights/yolov3.weights'

# logs_path
_LOGS_PATH = 'logs/416/100%'
