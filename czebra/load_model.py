def load_model(name):
    if name == 'mxnet_bisenet_egolane':
        from .models.predictor_mxnet_bisenet import PredictorMXNetBiSeNet
        model = PredictorMXNetBiSeNet('model', 1, model_input_size=(1280, 720), means=[128, 128, 128],
                                      stds=[255, 255, 255], to_rgb=False)
        return model
