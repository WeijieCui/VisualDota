import os

import cv2
import torch
import torchvision
from matplotlib import pyplot as plt, patches
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_Weights, FasterRCNN
from torchvision import transforms

from data_loader import faster_rcnn_data_loader, DOTA_LABELS

DEVICE_DEFAULT = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
MODEL_ROOT_DEFAULT = '../models'
MODEL_PREFIX_DEFAULT = 'fasterrcnn_v'
MODEL_SUFFIX_DEFAULT = '.pth'


def _get_model(num_classes: int) -> FasterRCNN:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train=False):
    return transforms.Compose([transforms.ToTensor()])


def load_model(
        model_name: str = None,
        num_classes: int = 15,
        root: str = MODEL_ROOT_DEFAULT,
        init: bool = False,
        device: torch.device = None
) -> FasterRCNN:
    model = _get_model(num_classes)
    if not init:
        if not model_name:
            model_paths = [f for f in os.listdir(root)
                           if f.startswith(MODEL_PREFIX_DEFAULT) and f.endswith(MODEL_SUFFIX_DEFAULT)]
            if model_paths:
                model_name = sorted(model_paths)[0]
        if model_name:
            full_model_path = os.path.join(root, model_name)
            model.load_state_dict(torch.load(full_model_path))
            model.eval()
            print(f'load from local model: {full_model_path}')
    model.to(device if device else DEVICE_DEFAULT)
    return model


def save_model(
        model: FasterRCNN,
        version: int,
        model_path: str = MODEL_ROOT_DEFAULT,
):
    model_full_path = os.path.join(model_path, MODEL_PREFIX_DEFAULT + str(version) + MODEL_SUFFIX_DEFAULT)
    torch.save(model.to(torch.device('cpu')).state_dict(), model_full_path)
    print(f'saved model to {model_full_path}')


def train_model(
        model: FasterRCNN,
        data_loader: DataLoader,
        device: torch.device = None,
        train_roi_head: bool = False,
        train_box_predictor: bool = True,
        lr: float = 0.005,
        momentum: float = 0.9,
        weight_decay: float = 0.005,
        step_size: int = 3,
        gamma: float = 0.1,
        epochs: int = 10,
):
    print('training faster-rcnn model')
    device = device or DEVICE_DEFAULT
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze the ROI Heads (both box_head and box_predictor)
    if train_roi_head:
        for param in model.roi_heads.parameters():
            param.requires_grad = True
    # Unfreeze only the box_predictor
    if train_box_predictor:
        for param in model.roi_heads.box_predictor.parameters():
            param.requires_grad = True

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    try:
        # Training Loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            count = 0
            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                total_loss += losses.item()
                count += len(images)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if count % 50 == 0:
                    print(f'processed {count} images.')
            lr_scheduler.step()
            print(f"Epoch {epoch + 1} Loss: {total_loss / len(data_loader)}")
    except Exception as e:
        print(f"training interrupt: {e}")
    finally:
        # Clean by GC
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

    model.train(mode=False)


def predict(
        model: FasterRCNN,
        image: torch.Tensor = None,
        image_path: str = None,
        confidence_threshold: float = 0.1,
        show: bool = True,
        saved_img_path: str = None,
):
    # load image file
    if image_path:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = get_transform(train=False)
        image = transform(image).to(DEVICE_DEFAULT)
    # predict
    with torch.no_grad():
        prediction = model([image])

    # process results
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()

    # filter results with low confidence
    keep = scores >= confidence_threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]
    # visualization
    if show:
        fig, ax = plt.subplots(1, figsize=(12, 12))
        ax.imshow(image.cpu().permute(1, 2, 0).numpy())
        label_names = {v: k for k, v in DOTA_LABELS.items()}
        for box, score, label in zip(boxes, scores, labels):
            xmin, ymin, xmax, ymax = box
            rect = patches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(
                xmin, ymin,
                f"{label_names[label]}: {score:.2f}",
                color='white', backgroundcolor='red',
                fontsize=5,
            )
        if saved_img_path:
            plt.savefig(saved_img_path)
        else:
            plt.show()


if __name__ == '__main__':
    _model = load_model(model_name='fasterrcnn_v7.pth')
    train_loader = faster_rcnn_data_loader(
        image_dir='data/tiny_train/images',
        label_dir='data/tiny_train/labels',
        batch_size=20,
        num_workers=3,
    )
    for i in range(8, 15):
        train_model(_model, data_loader=train_loader, epochs=1)
        save_model(_model, version=i)
        predict(
            _model,
            image_path='data/tiny_train/images/P0002.png',
            saved_img_path=f'data/tiny_train/results/P0002_v{i}.png',
            confidence_threshold=0,
        )
