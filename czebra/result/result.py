from .detection import Detection
from .segmentation import Segmentation

from typing import List, Optional


class Result(object):
    def __init__(self,
                 detections: Optional[List[Detection]] = None,
                 segmentation: Optional[Segmentation] = None):
        self.detections = detections
        self.segmentation = segmentation
