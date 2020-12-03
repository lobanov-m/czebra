import numpy as np

from ..result import Result

from typing import Union, Dict


class Predictor:
    def predict(self, data: Union[Dict, np.array], inference_result=None) -> Result:
        pass

