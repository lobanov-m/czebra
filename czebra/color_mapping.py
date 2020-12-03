import numpy as np


def get_segmentation_color_map(name, to_bgr=True, use_default=True):
    if name not in COLOR_MAPPINGS_SEGMENTATION:
        if use_default:
            name = 'default'
        else:
            raise ValueError(f"No mapping for classes {name}")
    color_map = np.zeros((256, 3), dtype='uint8')
    for i, cls, color in COLOR_MAPPINGS_SEGMENTATION[name]:
        if to_bgr:
            color = color[::-1]
        color_map[i] = color
    return color_map


COLOR_MAPPING_SEGMENTATION_DEFAULT = [
    (1, 'зеленый', (6, 251, 126)),
    (2, 'красный', (181, 47, 118)),
    (3, 'голубой', (30, 132, 224)),
    (4, 'оранжевый', (238, 155, 45)),
    (5, 'светло розовый', (229, 166, 246)),
    (6, 'зеленый', (50, 153, 21)),
    (7, 'фиолетовый', (136, 49, 244)),
    (8, 'синий', (19, 29, 156)),
    (9, 'морской волны', (67, 250, 250)),
    (10, 'желтый', (244, 240, 140)),
    (11, 'желтый зеленоватый', (151, 245, 10)),
    (12, 'зеленый', (14, 135, 29)),
    (255, 'белый', (255, 255, 255)),
]


COLOR_MAPPINGS_SEGMENTATION = {
    'default': COLOR_MAPPING_SEGMENTATION_DEFAULT,
}