import tensorflow as tf
import cv2
from math import ceil
from time import time
import numpy as np

from utils import *

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class YOLO(object):
    _defaults = {
        'model_path': 'models/coco_model/coco_model.h5',
        'anchors_path': 'models/coco_model/coco_anchors.txt',
        'classes_path': 'models/coco_model/coco_classes.txt',
        'score': 0.4,
        'iou': 0.2,
        'model_size': (416, 416),
        "gpu_num": 1,
    }

    def __init__(self):
        self.__dict__.update(self._defaults)
        self.class_names = get_classes(self.classes_path)
        self.anchors = get_anchors(self.anchors_path)
        self.model = tf.keras.models.load_model(self.model_path)

        self.num_anchors = len(self.anchors) // 3
        self.num_classes = len(self.class_names)

    def detect_frame(self, image):
        input = cv2.resize(image, self.model_size)
        image_height, image_width = image.shape[0], image.shape[1]
        input_height, input_width = input.shape[0], input.shape[1]

        input = input.astype("float32")
        input /= 255.
        input = np.expand_dims(input, 0)

        output = self.model.predict(input)
        boxes = self.decode_output(output)
        boxes = self.remove_extra_boxes(boxes)
        image = self.draw_boxes(image, boxes,
                                height_scale=image_height/input_height,
                                width_scale=image_width/input_width)
        return image

    def run_image(self, path):
        image = cv2.imread(path)
        image = self.detect_frame(image)

        cv2.imwrite("output/image_output.png", image)
        cv2.imshow(path, image)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def run_video(self, path):
        cam = cv2.VideoCapture(path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("output/video_output.avi", fourcc, 12.0, (640, 480))

        while True:
            ret, frame = cam.read()
            frame = self.detect_frame(frame)

            cv2.imshow("webcam", frame)
            out.write(frame)

            if cam.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cam.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
                break

        out.release()
        cv2.destroyAllWindows()


    def run_webcam(self):
        cam = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter("output/webcam_output.avi", fourcc, 12.0, (640, 480))

        while True:
            ret, frame = cam.read()
            frame = self.detect_frame(frame)

            cv2.imshow("webcam", frame)
            out.write(frame)

            if cv2.waitKey(1) == 27:
                break

        out.release()
        cv2.destroyAllWindows()

    def draw_boxes(self, image, boxes, height_scale=1, width_scale=1):
        for i in range(len(boxes)):
            confidence = np.amax(boxes[i][5:])
            index = np.where(boxes[i][5:] == confidence)[0][0].astype('int')
            what_was_detected = self.class_names[index]
            label = what_was_detected + ' ' + str(round(confidence * 100, 2))

            cv2.rectangle(image, (int(boxes[i][0]*width_scale), int(boxes[i][1]*height_scale)),
                                 (int(boxes[i][2]*width_scale), int(boxes[i][3]*height_scale)),
                          (255, 0, 0), 5)
            cv2.putText(image, label, (int(boxes[i][0]*width_scale), int(boxes[i][1]*height_scale) + 30),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        return image



    def decode_output(self, output):
        boxes = np.array([])
        for layer in range(3):
            out = output[layer][0]
            out = out.reshape((out.shape[0], out.shape[1], self.num_anchors, self.num_classes+5))

            # calculate array for all grid position
            grid_y = np.tile(np.arange(0, out.shape[0]).reshape(-1, 1, 1),
                             (1, out.shape[1], 1))
            grid_x = np.tile(np.arange(0, out.shape[1]).reshape(1, -1, 1),
                             (out.shape[0], 1, 1))
            grid_xy = np.tile(np.expand_dims(np.concatenate([grid_x, grid_y], axis=2), axis=2), (3, 1))
            grid_wh = np.array([{0:32, 1:16, 2:8}[layer],{0:32, 1:16, 2:8}[layer]])


            out[..., :2] = (sigmoid(out[..., :2]) + grid_xy) * grid_wh
            out[..., 4:] = sigmoid(out[..., 4:])
            out[..., 5:] = out[..., 4][..., np.newaxis] * out[..., 5:]
            out[..., 5:] *= out[..., 5:] > self.score

            # calculate wh
            out[..., 2:4] = np.exp(out[..., 2:4]) * self.anchors[layer*3:layer*3+3]

            # select boxes with high object score
            out = out.reshape(-1, self.num_classes+5)
            selected = out[out[:,4] > self.score]

            # change cor to x1, y1 , x2, y2
            final = np.copy(selected)
            final[..., 0] = selected[..., 0] - selected[..., 2] / 2
            final[..., 1] = selected[..., 1] - selected[..., 3] / 2
            final[..., 2] = selected[..., 0] + selected[..., 2] / 2
            final[..., 3] = selected[..., 1] + selected[..., 3] / 2

            boxes = np.append(boxes, final)

        boxes = boxes.reshape(-1, self.num_classes + 5)
        return boxes


    def remove_extra_boxes(self, boxes):
        boxes = boxes[~np.all(boxes[:, 5:] == 0, axis=1)]
        classes = boxes[:, 5:]
        indices = np.where(classes > 0)[1]

        u, index = np.unique(indices, return_index=True)
        split = np.split(boxes, index[1:])

        boxes = []
        for s in split:
            box = self.non_max_suppression_fast(s)
            if len(box) != 0:
                boxes.append(box)

        if len(boxes) != 0:
            return np.concatenate(boxes)
        return np.array([])

    def non_max_suppression_fast(self, boxes):
        if len(boxes) == 0:
            return []

        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            overlap = (w * h) / area[idxs[:last]]

            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > self.iou)[0])))

        return boxes[pick]