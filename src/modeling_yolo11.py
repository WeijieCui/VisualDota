import os

from ultralytics import YOLO

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
DEFAULT_INITIAL_CONFIG = 'yolo11n-obb.yaml'
DEFAULT_CONFIG = '../models/yolo11/yolo11n-obb-dota.yaml'


def load_model(
        model_name_or_path: str = None,
        config: str = DEFAULT_INITIAL_CONFIG,
        init: bool = False,
) -> YOLO:
    """
    Load YOLO V11 Model
    :param model_name_or_path: name or path of YOLO, default: the latest trained version.
    :param config: YAML config file path
    :param init: whether it is an initial model, default: False.
     If it is true, load the trained version of yolo11n-oob.pt.
    :return: YOLO
    examples:
        load an initial model from models/yolo11/yolo11n-obb.pt: load_model(init=True)
        load a special model: load_model(model_name_or_path=path_to_model)
        load the latest model from models/yolo11/yolo11n-obb-latest.pt: load_model()
    """
    if init:
        config = 'yolo11n-obb.yaml'
        model_name_or_path = '../models/yolo11/yolo11n-obb.pt'
        model = YOLO(config).load(model_name_or_path)
    else:
        if not model_name_or_path:
            model_name_or_path = '../models/yolo11/yolo11n-obb-latest.pt'
        model = YOLO(model_name_or_path)
    print(f'load model from {model_name_or_path}, config: {config}')
    return model


def train_model(
        model: YOLO,
        config: str = DEFAULT_CONFIG,
        epochs: int = 10,
        imgsz: int = 768,
        batch: int = 2,
):
    """
    Train YOLO model
    :param model: YOLO V11
    :param config: YAML config file path
    :param epochs: epochs for training
    :param imgsz: image size
    :param batch: image number per batch
    :return: results
    """
    results = model.train(data=config, epochs=epochs, imgsz=imgsz, batch=batch)
    return results


def val_model(
        model: YOLO,
        config: str = DEFAULT_CONFIG,
        imgsz: int = 768,
        batch: int = 2,
):
    """
    validate model
    :param model: YOLO
    :param config: YAML config file path
    :param imgsz: image size
    :param batch: image number per batch
    :return: results
    """
    results = model.val(data=config, imgsz=imgsz, batch=batch)
    return results


if __name__ == '__main__':
    """Test YOLO V11"""
    _model = load_model(
        init=True,
    )
    _model = load_model(
        # init=True,
    )
    _results = train_model(_model, epochs=1)
    print(_results)
    _results = val_model(_model)
    print('final result:')
    print(_results)
