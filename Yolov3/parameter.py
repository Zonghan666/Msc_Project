# convolution layer
_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

# anchors
# yolov3
_ANCHORS = [(10., 13.), (16., 30.), (33., 23.), (30., 61.), (62., 45.), (59., 119.), (116., 90.), (156., 198.), (373., 326.)]

# nipple(DDSM)
#_ANCHORS = [(29., 25.), (36., 22.), (33., 30.), (40., 26.), (50., 27.), (43., 31.), (58., 29.), (46., 38.), (58., 36.)]

# commom
_EPLISION = 1e-08

# classes
_CLASSES = {'dog': 0}

# shape
_INPUT_SHAPE = (416, 416)
y_dim_0 = _INPUT_SHAPE[0] / 32
y_dim_1 = y_dim_0 * 2
y_dim_2 = y_dim_1 * 2
_DIM = int((y_dim_0**2 + y_dim_1**2 + y_dim_2**2) * len(_ANCHORS) / 3)

# training parameter
_LEARNING_RATE = 1e-4
_BATCH_SIZE = 2
_N_CLASSES = len(_CLASSES)
_EPOCH = 30

# non_max_suppression
_CONFIDENCE_THRESHOLD = 0.5
_IOU_THRESHOLD = 0.2