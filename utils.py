import numpy as np


def get_classes(path):
    with open(path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(path):
    with open(path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    idx = [6, 7, 8, 3, 4, 5, 0, 1, 2]
    return np.array(anchors).reshape(-1, 2)[idx]


def sigmoid(x):
    return 1. / (1. + np.exp(-x))
