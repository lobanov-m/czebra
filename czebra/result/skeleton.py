import numpy as np
import cv2


class Skeleton:
    def __init__(self, points, confidences):
        assert len(points) == 17 and len(confidences) == 17
        self.points = points
        self.confidences = confidences

    def draw(self, image):
        for pair, color in zip(self._joint_pairs(), self._pair_colors()):
            cv2.line(image, self.get_cv_point(pair[0]), self.get_cv_point(pair[1]), color, 3)

    def get_cv_point(self, i):
        x = int(np.round(self.points[i][0]))
        y = int(np.round(self.points[i][1]))
        return x, y

    @staticmethod
    def _joint_pairs():
        joint_pairs = [
            [0, 1], [1, 3], [0, 2], [2, 4],
            [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
            [5, 11], [6, 12], [11, 12],
            [11, 13], [12, 14], [13, 15], [14, 16]
        ]
        return joint_pairs

    @staticmethod
    def _pair_colors():
        colors = [
            [0, 255, 255],
            [17, 238, 255],
            [34, 221, 255],
            [51, 204, 255],
            [68, 187, 255],
            [85, 170, 255],
            [102, 153, 255],
            [119, 136, 255],
            [136, 119, 255],
            [153, 102, 255],
            [170, 85, 255],
            [187, 68, 255],
            [204, 50, 255],
            [221, 33, 255],
            [238, 16, 255],
            [255, 0, 255],
        ]
        return colors
