import numpy as np
import cv2
import mxnet as mx
import os

from .predictor import Predictor
from ..result import Result, Segmentation

# Typing
from typing import Tuple, List

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'


class PredictorMXNetBiSeNet(Predictor):
    def __init__(self, prefix: str, epoch: int, model_input_size: Tuple[int, int],
                 means: List[float] = (128, 128, 128), stds: List[float] = (255, 255, 255),
                 to_rgb: bool = False, **kwargs):
        """
        prefix: mxnet model prefix; for example for such weights path /home/user/mxnet_ego_lane_v7-0001.params
                prefix would be: /home/user/mxnet_ego_lane_v7
        epoch: number of weights epoch; for upper example poch would be 1.
        means: Mean of images pixes used for preprocessing in BGR format
        std: standard deviations of images pixes in BGR format
        to_rgb: if transform image to rgb before processing
        """
        super(PredictorMXNetBiSeNet, self).__init__(**kwargs)
        self.model_input_size = model_input_size
        assert len(means) == 3
        self.means = np.array(means)
        assert len(stds) == 3
        self.stds = np.array(stds)
        self.to_rgb = to_rgb

        self.module = mx.module.Module.load(prefix, epoch, context=mx.gpu(), label_names=[])
        self.module.bind([('data', (1, 3, model_input_size[1], model_input_size[0]))])

    def predict(self, data, inference_result=None):
        if isinstance(data, dict):
            frame = data['frame']
        elif isinstance(data, np.ndarray):
            frame = data
        else:
            raise ValueError()

        batch = self.preprocess(frame)
        self.module.forward(batch)
        out = self.postprocess((frame.shape[1], frame.shape[0]))

        if inference_result is None:
            inference_result = Result()
        inference_result.segmentation = out
        return inference_result

    def preprocess(self, frame):
        frame = cv2.resize(frame, self.model_input_size)
        frame = (frame.astype('float32') - self.means) / self.stds
        frame = np.moveaxis(frame, 2, 0)[::-1]
        if self.to_rgb:
            frame = frame[::-1]
        frame = frame[np.newaxis, :]
        frame = mx.nd.array(frame, mx.gpu())
        batch = mx.io.DataBatch([frame])
        return batch

    def postprocess(self, frame_size):
        out = self.module.get_outputs()[0]
        out = mx.nd.softmax(out, axis=1).argmax(axis=1)
        out = out[0].asnumpy()
        out = cv2.resize(out, frame_size, interpolation=cv2.INTER_NEAREST)
        out = out.astype('uint8')
        out = Segmentation(out)
        return out
