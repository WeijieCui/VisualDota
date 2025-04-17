# data_loader.py

import os

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DOTA_LABELS = {
    'plane': 0, 'ship': 1, 'storage-tank': 2,
    'baseball-diamond': 3, 'tennis-court': 4,
    'basketball-court': 5, 'ground-track-field': 6,
    'harbor': 7, 'bridge': 8,
    'large-vehicle': 9, 'small-vehicle': 10,
    'helicopter': 11, 'roundabout': 12,
    'soccer-ball-field': 13, 'swimming-pool': 14,
}


class DotaRawDataset(Dataset):
    """
    用于加载 DOTA 数据集的 PyTorch Dataset 类。
    """

    def __init__(
            self,
            image_dir,
            label_dir,
            transform=None,
            joint_transform=None,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.joint_transform = joint_transform
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.tif'))]
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def _parse_label_file(self, label_path):
        bboxes = []
        classes = []
        with open(label_path, 'r') as f:
            lines = f.readlines()[2:]  # 跳过头部信息
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                points = list(map(float, parts[:8]))
                cls = parts[8]
                bbox = [(points[i], points[i + 1]) for i in range(0, 8, 2)]
                bboxes.append(bbox)
                classes.append(cls)
        return bboxes, classes

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.split('.')[0] + '.txt')

        image = Image.open(image_path).convert('RGB')

        if os.path.exists(label_path):
            bboxes, classes = self._parse_label_file(label_path)
        else:
            bboxes, classes = [], []

        if self.joint_transform:
            return self.joint_transform(image, bboxes, classes)

        if self.transform:
            image = self.transform(image)
        return image, {'boxes': bboxes, 'labels': classes}


class JointTransformFasterRCNN:
    def __init__(
            self,
            size: int,
            max_size: int = None,
            to_horizontal_bbox: bool = False,
    ):
        r"""Resize the input image and the orient bounding box
        :param size(sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`.
        :param max_size (int, optional): The maximum allowed for the longer edge of
            the resized image. If the longer edge of the image is greater
            than ``max_size`` after being resized according to ``size``,
            ``size`` will be overruled so that the longer edge is equal to
            ``max_size``.
            As a result, the smaller edge may be shorter than ``size``. This
            is only supported if ``size`` is an int (or a sequence of length
            1 in torchscript mode).
        :param to_horizontal_bbox (int, default = False): If it is True, the locations of the orient bounding boxes
            will be transformed to the one of horizontal bounding boxes.
        """
        self.max_size = max_size
        self.size = size
        self.to_horizontal_bbox = to_horizontal_bbox
        self.img_transform = transforms.Compose([
            transforms.Resize(size=size, max_size=max_size),
            # transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image: Image, boxes: list = None, labels: list = None):
        r"""Resize the input image and the orient bounding box
        :param image(Image): Image data.
        :param boxes (list((x0, y0), (x1, y1), (x2, y2), (x3, y3)), optional): The orient bounding boxes of the objects.
        :param labels (list(str), optional): The labels for the image.
        """
        transformed_image = self.img_transform(image)
        ratio = transformed_image.shape[1] / image.height
        horizontal_bboxes = [(min(x0, x1, x2, x3), min(y0, y1, y2, y3), max(x0, x1, x2, x3), max(y0, y1, y2, y3),)
                             for ((x0, y0), (x1, y1), (x2, y2), (x3, y3)) in boxes]
        num_labels = [DOTA_LABELS[n] for n in labels]
        target = {
            'boxes': torch.tensor(horizontal_bboxes) * ratio,
            'labels': torch.tensor(num_labels),
        }
        return transformed_image, target


def custom_collate_fn(batch):
    """
    自定义 collate_fn，支持不规则图像尺寸与多样标签。
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


def faster_rcnn_data_loader(
        image_dir: str,
        label_dir: str,
        batch_size: int = 10,
        num_workers: int = 2,
) -> DataLoader:
    train_dataset = DotaRawDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        transform=transforms.Compose([transforms.ToTensor()]),
        joint_transform=JointTransformFasterRCNN(size=224)
    )

    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn  # ✅ 使用自定义 collate_fn
    )


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = DotaRawDataset(
        image_dir='./data/train/images',
        label_dir='./data/train/labels',
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=5,
        shuffle=True,
        num_workers=2,
        collate_fn=custom_collate_fn  # ✅ 使用自定义 collate_fn
    )

    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f"批次 {batch_idx + 1}，图像数量: {len(images)}")
        print(f"第一张图像尺寸: {images[0].shape}")
        print(f"第一张图像标签: {len(labels[0]['boxes'])}")
        # break
