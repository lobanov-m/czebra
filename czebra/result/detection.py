import cv2
import numpy as np

from matplotlib import patches

from .skeleton import Skeleton
from .box3d import Box3D
from ..visualization import draw_box, draw_labels

from typing import Union, List, Optional


class Detection(object):
    """
    obj_type: str
    score: float in range [0, 1]
    box: List[float] of length 4 with format x0, y0, x1, y1. Not relative.
    box3d: Box3D
    flags: int
    contour: np.array with shape (N, 2)
    """

    def __init__(
            self,
            obj_type: Optional[str] = None,
            score: Optional[float] = None,
            box: Optional[List[float]] = None,
            box3d: Optional[Box3D] = None,
            flags: Optional[int] = None,
            contours: Optional[List[np.ndarray]] = None,
            track_id: Union[None, int, str] = None,
            detector_id: Optional[str] = None,
            classes_name: Optional[str] = None,
            skeleton: Optional[Skeleton] = None,
            lines: Optional[List[np.ndarray]] = None,
            z_order: Optional[int] = None
    ):
        self.obj_type = obj_type
        self.score = score
        self.box = box
        self.box3d = box3d
        self.flags = flags
        self.contours = contours
        self.track_id = track_id
        self.detector_id = detector_id
        self.classes_name = classes_name
        self.skeleton = skeleton
        self.lines = lines
        self.z_order = z_order

    def visualize_box3d(self, frame, calib, color=(0, 255, 0), draw_upper_cross=True, with_label=True):
        labels = None
        if with_label:
            labels = self.get_labels()
        self.box3d.draw_box3d(frame, calib, color, draw_upper_cross, labels)

    def visualize_box(self, frame, color=(0, 255, 0), thickness=1, with_label=True):
        labels = self.get_labels()
        draw_box(frame, self.box, color, thickness, labels, with_label)

    def visualize_contours(self, frame, color=(0, 255, 0), thickness=1, with_label=True, labels_position='first_point'):
        cv2.drawContours(frame, [c.astype('int64') for c in self.contours], 0, color=color, thickness=thickness)
        if with_label:
            labels = self.get_labels()
            if labels_position == 'first_point':
                labels_y = int(self.contours[0][0, 0, 1])
                labels_x = int(self.contours[0][0, 0, 0])
            elif labels_position == 'left_upper_corner':
                labels_y = int(np.min(np.concatenate([c[:, :, 1] for c in self.contours])))
                labels_x = int(np.min(np.concatenate([c[:, :, 0] for c in self.contours])))
            else:
                raise ValueError(f"Detection.visualize_contours(): Wrong labels_position option: {labels_position}")
            draw_labels(frame, labels_y, labels_x, labels, color)

    def visualize_skeleton(self, frame):
        self.skeleton.draw(frame)

    def visualize_lines(self, frame, color=(0, 150, 0), thickness=1, with_label=True):
        if len(self.lines) == 1:
            cv2.polylines(frame, self.lines, False, color, thickness)
        else:
            rails_draw = [np.around(np.array(self.lines[i]).reshape(-1, 2)).astype(np.int32) for i in range(2)]
            ml = min(len(rails_draw[0]), len(rails_draw[1]))
            for i in range(ml):
                midpnt = ((rails_draw[0][i][0] + rails_draw[1][i][0]) // 2, (rails_draw[0][i][1] + rails_draw[1][i][1]) // 2)
                cv2.line(frame, tuple(rails_draw[0][i]), midpnt, (0, 150, 0))
                cv2.line(frame, midpnt, tuple(rails_draw[1][i]), (0, 0, 150))
        if with_label:
            labels = self.get_labels()
            labels_x = self.lines[0][0, 0, 0]
            labels_y = self.lines[0][0, 0, 1]
            draw_labels(frame, labels_y, labels_x, labels, color)

    def get_labels(self):
        label_cls_score = f'{self.obj_type}' + (f' {self.score:.2f}%' if self.score is not None else '')
        labels = [label_cls_score]
        if self.track_id is not None:
            labels.append(f'Track id: {self.track_id}')
        return labels

    def get_birdeye_mpl_patches(self, color):
        color = [c / 255. for c in color]
        # BGR to RGB
        color = (color[2], color[1], color[0])
        angle = self.box3d.quaternion.angle if self.box3d.quaternion.axis[1] > 0 else -self.box3d.quaternion.angle
        angle = -angle
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        points = np.array([
            [self.box3d.w / 2, -self.box3d.w / 2, -self.box3d.w / 2,  self.box3d.w / 2, ],
            [self.box3d.l / 2,  self.box3d.l / 2, -self.box3d.l / 2, -self.box3d.l / 2, ],
        ])
        rotated_points = rotation_matrix.dot(points)
        translation = np.array([[self.box3d.x], [self.box3d.z]])
        translated_points = rotated_points + translation
        points_arrow = np.array([
            [0, 0],
            [0, 3]
        ])
        rotated_arrow = rotation_matrix.dot(points_arrow)
        translated_arrow = rotated_arrow + translation
        rect = patches.Polygon(translated_points.T, linewidth=1, edgecolor=color, facecolor='none')
        arrow = patches.Arrow(translated_arrow[0, 0], translated_arrow[1, 0],
                              translated_arrow[0, 1] - translated_arrow[0, 0],
                              translated_arrow[1, 1] - translated_arrow[1, 0], width=1)
        return rect, arrow

    def __str__(self):
        s = f"{self.obj_type}"
        if self.score is not None:
            s += f" ({self.score:0.2f})"
        if self.box is not None:
            s += f": {int(self.box[0]):4d} {int(self.box[1]):4d} " f"{int(self.box[2]):4d} {int(self.box[3]):4d}"
        return s
