import numpy as np
import cv2

from .color_mapping import get_segmentation_color_map


def draw_box(image, box, color=(0, 0, 255), thickness=4, labels=(), visualize_labels=True,
             font_scale=0.5, font_thickness=2, fmt='xyxy', normalized=False):
    assert fmt in ('xyxy', 'yxyx', 'xywh')
    if fmt == 'xyxy':
        x0, y0, x1, y1 = box
    elif fmt == 'yxyx':
        y0, x0, y1, x1 = box
    else:
        x0, y0, x1, y1 = box
        x1 += x0
        y1 += y0

    h, w, c = image.shape
    if normalized:
        x0 *= w
        y0 *= h
        x1 *= w
        y1 *= h

    x0 = int(np.round(x0))
    y0 = int(np.round(y0))
    x1 = int(np.round(x1))
    y1 = int(np.round(y1))

    cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
    if visualize_labels and labels is not None:
        draw_labels(image, y0, x0, labels, color, (0, 0, 0), font_scale, font_thickness)


def draw_labels(image, y, x, labels, box_color, font_color=(0, 0, 0), font_scale=0.5, font_thickness=2):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_bottom = y
    # Reverse list and print from bottom to top.
    for s in labels[::-1]:
        (text_w, text_h), base_line = cv2.getTextSize(s, font, font_scale, font_thickness)
        margin = int(np.ceil(0.1 * text_h))
        cv2.rectangle(image, (x, text_bottom - text_h - 2 * margin), (x + text_w, text_bottom), box_color, cv2.FILLED)
        cv2.putText(image, s, (x + margin, text_bottom - margin), font, thickness=font_thickness,
                    fontScale=font_scale, color=font_color)
        text_bottom -= text_h + 2 * margin


def visualize_segmentation(image, segmentation, color_map, alpha=0.8, inplace=True):
    if isinstance(color_map, str):
        color_map = get_segmentation_color_map(color_map)
    assert len(segmentation.shape) == 2
    assert image.shape[0] == segmentation.shape[0]
    assert image.shape[1] == segmentation.shape[1]
    assert color_map.shape == (256, 3)
    segmentation = apply_color_map(segmentation, color_map)
    mask_segmented = np.sum(segmentation != 0, axis=2).astype(np.bool)
    mask_segmented = np.tile(mask_segmented[:, :, np.newaxis], (1, 1, 3))
    if inplace:
        image[mask_segmented] = image[mask_segmented] * alpha + segmentation[mask_segmented] * (1 - alpha)
    else:
        _image = image.copy()
        _image[mask_segmented] = image[mask_segmented] * alpha + segmentation[mask_segmented] * (1 - alpha)
        image = _image
    return image


def apply_color_map(segmentation, color_map):
    assert segmentation.dtype == np.uint8
    assert color_map.dtype == np.uint8
    if color_map.shape == (256,):
        color = False
    elif color_map.shape == (256, 3):
        color = True
    else:
        raise ValueError('Неправильный shape у color map')
    if color:
        c_b = np.empty_like(segmentation, 'uint8')
        c_g = np.empty_like(segmentation, 'uint8')
        c_r = np.empty_like(segmentation, 'uint8')
        cv2.LUT(segmentation, color_map[:, 0], c_b)
        cv2.LUT(segmentation, color_map[:, 1], c_g)
        cv2.LUT(segmentation, color_map[:, 2], c_r)
        c = np.stack((c_b, c_g, c_r), axis=2)
        return c
    else:
        c = np.empty_like(segmentation, 'uint8')
        cv2.LUT(segmentation, color_map, c)
        return c
