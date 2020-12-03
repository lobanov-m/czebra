import numpy as np

from pyquaternion import Quaternion


def camera_to_image(corners, projection, invalid_value=-1000):
    """
    corners: (N, 3), N points on X(right)-Y(down)-Z(front) camera plane
    projection: (3, 4), projection matrix

    points: (N, 2), N points on X-Y image plane
    """
    corners = np.array(corners).reshape(-1, 3)
    assert corners.shape[1] == 3, "Shape ({}) not fit".format(corners.shape)

    points = np.hstack([corners, np.ones((corners.shape[0], 1))]).dot(projection.T)

    # [x, y, z] -> [x/z, y/z]
    depths = points[:, 2:3]
    mask = depths > 0
    points = (points[:, :2] / depths) * mask + invalid_value * (1 - mask)

    return points


def image_to_camera(points, depths, projection):
    points = np.array(points).reshape(-1, 2)
    depths = np.array((depths)).reshape(-1, 1)
    assert points.shape[0] == depths.shape[0]
    assert depths.shape[1] == 1
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    points *= depths
    points -= projection[:, 3]
    projection_inv = np.linalg.inv(projection[:, :3].T)
    corners = points.dot(projection_inv)
    return corners


def yaw_allocentric_to_egocentric(q_allo, x, z):
    beta = np.arctan(x / z)
    # Дополнительный угол, добавляемый из-за того что угол закодирован как угол обзора, а не как глобальный угол
    q_additional_angle = Quaternion(axis=[0, 1, 0], angle=beta)
    q_ego = q_allo * q_additional_angle
    return q_ego
