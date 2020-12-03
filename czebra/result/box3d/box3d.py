import numpy as np
import cv2

from pyquaternion import Quaternion
# from scipy.spatial.transform import Rotation
from .utils import camera_to_image, image_to_camera, yaw_allocentric_to_egocentric
from ...visualization import draw_box, draw_labels


class Box3D(object):
    def __init__(self, x, y, z, w, h, l, obj_type, quaternion: Quaternion, score=1):
        """
           z
          /
         /____ x
        |
        |
        y

        Box:
                 /^
                /
              _____
            /     /|
          l/     / |
          /____ /  |
         |     |  /
        h|     | /
         |__w__|/

              2_____ 3
             /|    /|
            / |   / |
           / 1|__/__|0
         6/__/_7/  /
         |  /  |  /
         | /   | /
         |/____|/
         5     4

        (x, y, z) - координата центра коробки

        Положительный угол - по часовой
        Изначальное направление: (0, 0, 1)
        (не в кватернионе, а по логике. yaw-pitch-roll из кватерниона надо конвертировать)
        yaw - рысканье (вокруг y)
        pitch - тангаж (вокруг x)
        roll - крен (вокруг z)
        """

        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.h = h
        self.l = l
        self.obj_type = obj_type
        self.quaternion = quaternion
        self.score = score

    @property
    def rotation_matrix(self):
        return self.quaternion.rotation_matrix

    @classmethod
    def init_from_center_dimensions_quaternion(cls, obj_type, score, center, dimensions, quaternion):
        w, h, l = dimensions
        if not isinstance(quaternion, Quaternion):
            quaternion = Quaternion(quaternion)
        return cls(center[0], center[1], center[2], w, h, l, obj_type, quaternion, score)

    @classmethod
    def init_from_center_dimensions_yaw_pitch_roll(cls, obj_type, score, center, dimensions, yaw_pitch_roll):
        w, h, l = dimensions
        quaternion = cls.get_quaternion_from_yaw_pitch_roll(*yaw_pitch_roll)
        return cls(center[0], center[1], center[2], w, h, l, obj_type, quaternion, score)

    @classmethod
    def init_from_center_projection_depth_dimensions_quaternion_allocentric(
            cls, obj_type, score, center_projection, depth, dimensions, quaternion_allocentric, calib):
        center = image_to_camera(np.array([[center_projection[0], center_projection[1]]]),
                                 np.array([[depth]]), calib)[0]
        if not isinstance(quaternion_allocentric, Quaternion):
            quaternion_allocentric = Quaternion(quaternion_allocentric)
        q_ego = yaw_allocentric_to_egocentric(quaternion_allocentric, center[0], center[2])
        return cls.init_from_center_dimensions_quaternion(obj_type, score, center, dimensions, q_ego)

    def get_box3d(self):
        w, h, l = self.w, self.h, self.l
        points = np.array([
            [ w/2,  h/2,  l/2],
            [-w/2,  h/2,  l/2],
            [-w/2, -h/2,  l/2],
            [ w/2, -h/2,  l/2],
            [ w/2,  h/2, -l/2],
            [-w/2,  h/2, -l/2],
            [-w/2, -h/2, -l/2],
            [ w/2, -h/2, -l/2],
        ])
        rotation_matrix = self.rotation_matrix.T
        points_rotated = points.dot(rotation_matrix)
        points_rotated_translated = points_rotated + np.array([[self.x, self.y, self.z]])
        return points_rotated_translated

    @property
    def yaw_pitch_roll(self):
        w, x, y, z = self.quaternion.elements
        q = Quaternion(w, z, x, y)
        yaw, pitch, roll = q.yaw_pitch_roll
        return yaw, pitch, roll

    @staticmethod
    def get_quaternion_from_yaw_pitch_roll(yaw, pitch, roll):
        # yaw, pitch и roll подразумеваются смысловые, в осях Box3D (self.get_yaw_pitch_roll()),
        # а не те, что возвращаются self.quaternion.yaw_pitch_roll
        # Здесь мы сначала делаем roll, потом pitch-им то, что на-roll-или, потом yaw-им то, что сделали до этого.
        # Система координат не привязана к коробке.
        # Или можно считать, что сначала делаем yaw, потом делаем pitch и roll в с.к. коробки
        q = Quaternion(axis=(0, 1, 0), angle=yaw) * \
            Quaternion(axis=(1, 0, 0), angle=pitch) * \
            Quaternion(axis=(0, 0, 1), angle=roll)
        return q

    def get_box3d_center_projection(self, calib):
        center = np.array([[self.x, self.y, self.z]])
        center_projected = camera_to_image(center, calib)
        return center_projected[0]

    def draw_box3d(self, frame, calib, color=(0, 255, 0), draw_upper_cross=True, labels=None):
        box3d = self.get_box3d()
        self.draw_already_got_box3d(box3d, frame, calib, color, draw_upper_cross, labels)

    @staticmethod
    def draw_already_got_box3d(box3d, frame, calib, color=(0, 255, 0), labels=None):
        points_image = camera_to_image(box3d, calib)
        # start point, end point, thickness
        lines = [
            (0, 1, 3),  # front
            (1, 2, 3),  # front
            (2, 3, 3),  # front
            (3, 0, 3),  # front
            (0, 4, 1),
            (1, 5, 1),
            (2, 6, 1),
            (3, 7, 1),
            (4, 5, 1),
            (5, 6, 1),
            (6, 7, 1),
            (7, 4, 1),
            (2, 7, 1),  # upper cross
            (3, 6, 1),  # upper cross
        ]
        for i0, i1, t in lines:
            p0 = tuple(points_image[i0].astype('int32').tolist())
            p1 = tuple(points_image[i1].astype('int32').tolist())
            cv2.line(frame, p0, p1, color, thickness=t)
        if labels is not None:
            labels_x = int(np.clip(points_image[:, 0].min(), a_min=0, a_max=frame.shape[1] - 50))
            labels_y = int(np.clip(points_image[:, 1].min(), a_min=50, a_max=frame.shape[0]))
            draw_labels(frame, labels_y, labels_x, labels, color)

    def get_box2d(self, calib, image_size):
        box3d = self.get_box3d()
        points_image = camera_to_image(box3d, calib)
        x0, y0 = np.min(points_image[:, 0]), np.min(points_image[:, 1])
        x1, y1 = np.max(points_image[:, 0]), np.max(points_image[:, 1])
        x0 = np.min([np.max([x0, 0]), image_size[0]])
        y0 = np.min([np.max([y0, 0]), image_size[1]])
        x1 = np.min([np.max([x1, 0]), image_size[0]])
        y1 = np.min([np.max([y1, 0]), image_size[1]])
        return x0, y0, x1, y1

    def draw_box2d(self, frame, calib, color=(0, 0, 255), additional_strs=None):
        image_size = (frame.shape[1], frame.shape[0])
        box2d = self.get_box2d(calib, image_size)
        display_strs = [self.obj_type]
        if additional_strs is not None:
            for s in additional_strs:
                display_strs.append(s)
        draw_box(frame, box2d, color, 1, display_strs)

    def get_observation_angle_quaternion(self):
        view_angle = np.arctan(self.x / self.z)
        view_angle_q = Quaternion(axis=(0, 1, 0), angle=-view_angle)
        alpha_q = self.quaternion * view_angle_q
        return alpha_q

    def rotate(self, quaternion):
        self.quaternion *= quaternion

    def rotate_world(self, quaternion):
        self.x, self.y, self.z = np.dot(quaternion.rotation_matrix, [self.x, self.y, self.z])
        self.quaternion = quaternion * self.quaternion

    def translate(self, x, y, z):
        self.x += x
        self.y += y
        self.z += z

    def to_z_up_system(self):
        """
        z
        |  y
        | /
        |/____ x
        Машина по умолчанию смотрит в сторону x
        :return:
        """
        x = self.x
        y = self.z
        z = -self.y
        q = self.quaternion * Quaternion(angle=np.pi / 2, axis=[1, 0, 0])
        return (x, y, z), q

    def __repr__(self):
        return (f"Class: {self.obj_type}({self.score: 6.2f}); X: {self.x: 6.2f}; Y: {self.y: 6.2f}; Z: {self.z: 6.2f};"
                f" W: {self.w: 6.2f}; H: {self.h: 6.2f}; L: {self.l: 6.2f};"
                f" Angle: {self.quaternion.degrees: 6.1f}; Axis: [{self.quaternion.axis[0]: 4.2f},"
                f" {self.quaternion.axis[1]: 4.2f}, {self.quaternion.axis[2]: 4.2f}]")
