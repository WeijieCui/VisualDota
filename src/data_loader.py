# data_loader.py

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class DotaRawDataset(Dataset):
    """
    用于加载 DOTA 数据集的 PyTorch Dataset 类。
    """
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
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

        if self.transform:
            image = self.transform(image)

        if os.path.exists(label_path):
            bboxes, classes = self._parse_label_file(label_path)
        else:
            bboxes, classes = [], []

        return image, {'bboxes': bboxes, 'classes': classes}


def custom_collate_fn(batch):
    """
    自定义 collate_fn，支持不规则图像尺寸与多样标签。
    """
    images, targets = zip(*batch)
    return list(images), list(targets)


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
        print(f"第一张图像标签: {len(labels[0]['bboxes'])}")
        # break
