import os
import traceback

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
        model_name: str,
        model_path: str = MODEL_ROOT_DEFAULT,
):
    model_full_path = os.path.join(model_path, model_name)
    torch.save(model.to(torch.device('cpu')).state_dict(), model_full_path)
    print(f'saved model to {model_full_path}')


def split_into_blocks(img_tensor: torch.Tensor, n: int) -> torch.Tensor:
    """
    Split image into n * n blocks
    Parameters:
         img_tensor (torch.Tensor): Shape (3, H, W)
    Return:
        torch.Tensor: Shape (n * n, 3, H//4, W//4)
    """
    num_colors, height, weight = img_tensor.shape
    if height % n != 0 or weight % n != 0:
        img_tensor = img_tensor[:, : height - height % n, :weight - weight % n]
    block_h, block_w = height // n, weight // n
    # Unford the dimension of width and height
    blocks = img_tensor.unfold(1, block_h, block_h).unfold(2, block_w, block_w)
    # Adjust and Merge blocks (4, 4, 3, block_h, block_w) -> (16, 3, block_h, block_w)
    return blocks.permute(1, 2, 0, 3, 4).contiguous().view(-1, num_colors, block_h, block_w)


def fine_predict(model, images: [], separate: int = 1):
    if separate == 1:
        return model(images)
    results = []
    for image in images:
        img_tiles = split_into_blocks(image, separate)
        predictions = model(img_tiles)
        result_boxes, result_labels, result_scores = [], [], []
        for i, prediction in enumerate(predictions):
            xi, yi = i % separate, i // separate
            num_tiles, num_colors, height, weight = img_tiles.shape
            boxes = [(x0 + xi * weight, y0 + yi * height, x1 + xi * weight, y1 + yi * height)
                     for (x0, y0, x1, y1) in prediction['boxes']]
            result_boxes.extend(boxes)
            result_labels.extend(prediction['labels'])
            result_scores.extend(prediction['scores'])
        result = {
            'boxes': torch.tensor(result_boxes),
            'labels': torch.tensor(result_labels),
            'scores': torch.tensor(result_scores),
        }
        results.append(result)
        return results


def fine_train(model, images: [], targets: [], separate: int = 1):
    if separate == 1:
        return model(images, targets)
    results = []
    for image, target in zip(images, targets):
        img_tiles = split_into_blocks(image, separate)
        num_tiles, num_colors, height, weight = img_tiles.shape
        fine_targets = [{
            'boxes': [],
            'labels': []
        } for i in range(separate ** 2)]
        for (xmin, ymin, xmax, ymax), label in zip(target['boxes'], target['labels']):
            xi, xj, yi, yj = int(xmin // weight), int(xmax // weight), int(ymin // height), int(ymax // height)
            if xi == xj and yi == yj:
                idx = xi + yi * separate
                fine_targets[idx]['boxes'].append(
                    (xmin - xi * weight, ymin - yi * height, xmax - xi * weight, ymax - yi * height))
                fine_targets[idx]['labels'].append(label)
        available = [len(target['boxes']) > 0 for target in fine_targets]
        if not any(available):
            continue
        fine_targets = [{
            'boxes': torch.tensor(fine_target['boxes']).to(device=targets[0]['boxes'].device),
            'labels': torch.tensor(fine_target['labels']).to(device=targets[0]['labels'].device),
        }
            for fine_target in fine_targets]
        results.append(model([image for image in img_tiles[available, :, :, :]],
                             [target for i, target in enumerate(fine_targets) if available[i]]))
    return {
        'loss_box_reg': sum(result['loss_box_reg'] for result in results) / len(results),
        'loss_classifier': sum(result['loss_classifier'] for result in results) / len(results),
        'loss_objectness': sum(result['loss_objectness'] for result in results) / len(results),
        'loss_rpn_box_reg': sum(result['loss_rpn_box_reg'] for result in results) / len(results),
    }


def train_model(
        model: FasterRCNN,
        data_loader: DataLoader,
        device: torch.device = None,
        train_roi_head: bool = False,
        train_box_predictor: bool = True,
        lr: float = 0.05,
        momentum: float = 0.5,
        weight_decay: float = 0.0001,
        step_size: int = 8,
        gamma: float = 0.1,
        epochs: int = 10,
        separate: int = 1,
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
    loss_dict = None
    try:
        # Training Loop
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            count = 0
            for images, targets in data_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = fine_train(model, images, targets, separate=separate)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                total_loss += losses.item()
                count += len(images)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if count % 5 == 0:
                    print(f'processed {count} images.')
            lr_scheduler.step()
            model.train(mode=False)
            print(f"Epoch {epoch + 1} Loss: {total_loss / len(data_loader)}, loss_dict: {loss_dict}")
    except Exception as e:
        traceback.print_exc()
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
        show_label: bool = False,
        saved_img_path: str = None,
        separate: int = 4,
):
    # load image file
    if image_path:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        transform = get_transform(train=False)
        image = transform(image).to(DEVICE_DEFAULT)
    # predict
    with torch.no_grad():
        prediction = fine_predict(model, [image], separate=separate)

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
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image.cpu().permute(1, 2, 0).numpy())
    ax.axis('off')
    label_names = {v: k for k, v in DOTA_LABELS.items()}
    for box, score, label in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box
        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if show_label:
            ax.text(
                xmin, ymin,
                f"{label_names[label]}: {score:.2f}",
                color='white', backgroundcolor='red',
                fontsize=5,
            )
    if saved_img_path:
        plt.savefig(saved_img_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()


if __name__ == '__main__':
    _version = 1
    _separate = 5
    image_dir = 'data/train/images'
    test_image_dir = 'data/test/images'
    model_name = f'fasterrcnn_s{_separate}_e{_version - 1}.pth'
    result_image_dir = test_image_dir.replace('images', 'results')
    if not os.path.exists(result_image_dir):
        os.makedirs(result_image_dir)
    if _version <= 1:
        _model = load_model(
            init=True,
            num_classes=15,
        )
    else:
        _model = load_model(
            model_name=model_name,
            num_classes=15,
        )
    train_loader = faster_rcnn_data_loader(
        image_dir='data/train/images',
        label_dir='data/train/labels',
        batch_size=5,
        num_workers=3,
    )
    for epoch in range(_version, _version + 1):
        train_model(_model, data_loader=train_loader, epochs=1, separate=_separate)
        save_model(_model, model_name=f'fasterrcnn_s{_separate}_e{epoch}.pth')
        for _image in [f for f in os.listdir(test_image_dir) if f.endswith(('.jpg', '.png', '.tif'))]:
            predict(
                _model,
                image_path=os.path.join(test_image_dir, _image),
                saved_img_path=os.path.join(result_image_dir, f's{_separate}_e{epoch}_{_image}'),
                confidence_threshold=0.2,
                separate=_separate,
            )
