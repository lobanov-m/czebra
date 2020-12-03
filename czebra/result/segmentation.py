import numpy as np

from ..visualization import visualize_segmentation

from typing import Union


class Segmentation:
    def __init__(self, segmentation_map):
        self.segmentation_map = segmentation_map

    def visualize_segmentation(self, image, color_map: Union[str, np.array] = 'default', alpha=0.8, inplace=True):
        """
        :param image: изображение, np.array с shape: [H, W, C]
        :param color_map: str - имя мэппинга или np.array с shape [256, 3] или [256, 1] -
                          трёх- или одноканальный цвет для каждого класса сегментации
        :param alpha: прозрачность
        :param inplace: возвращать новое изображение или раскрашивать переданное
        """
        assert (image.shape[0] == self.segmentation_map.shape[0] and
                image.shape[1] == self.segmentation_map.shape[1])
        visualize_segmentation(image, self.segmentation_map, color_map, alpha, inplace)