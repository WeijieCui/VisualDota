import os
from pprint import pprint

from PIL import Image


def load_dataset(
        top_num: int = None,
        folder_path: str = './data/train/',
        img_fold: str = 'images',
        label_fold: str = 'labelTxt',
        resize: bool = True,
        size: int = 512,
        include_label: bool = True,
) -> [(Image, dict)]:
    # folder_path
    image_folder_path = os.path.join(folder_path, img_fold)
    label_folder_path = os.path.join(folder_path, label_fold)
    # extensions of supported image format
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    # list all image files
    image_files = []
    count = 0
    for filename in os.listdir(image_folder_path):
        image_file_path = os.path.join(image_folder_path, filename)
        if os.path.isfile(image_file_path):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_files.append(filename)
                if top_num:
                    count += 1
                    if count >= top_num:
                        break

    # read and resize dataset
    dataset = []
    for filename in image_files:
        image_file_path = os.path.join(image_folder_path, filename)
        label_file_path = os.path.join(label_folder_path, filename.split('.')[0] + '.txt')
        try:
            with Image.open(image_file_path) as img:
                if resize and size:
                    inner_size = min(img.width, img.height)
                    sx, sy = int((img.width - inner_size) / 2), int((img.height - inner_size) / 2)
                    ex, ey = sx + inner_size, sy + inner_size
                    img = img.crop((sx, sy, ex, ey))
                    img = img.resize((size, size))
            with open(label_file_path) as lines:
                label_strs = [line for line in lines][2:]
            labels = []
            if include_label:
                for line in label_strs:
                    sp = line.split(' ')
                    position = [(int(sp[i * 2]), int(sp[i * 2 + 1])) for i in range(4)]
                    outside = any((x, y) for (x, y) in position
                                  if x < sx or x > ex or y < sy or y > ey)
                    if not outside:
                        labels.append({
                            'position': [(int((x - sx) * size / inner_size), int((y - sy) * size / inner_size))
                                         for (x, y) in position],
                            'class': sp[8],
                            'result': bool(sp[9]),
                        })
                if labels:
                    dataset.append((img, labels))
            else:
                dataset.append((img, labels))
        except Exception as e:
            print(f"Failed to load image {filename}: {str(e)}")

    return dataset


def count_labels(label_folder_path: str = './data/train/labelTxt/') -> [(str, int)]:
    label_dict = {}
    for filename in os.listdir(label_folder_path):
        label_file_path = os.path.join(label_folder_path, filename)
        if os.path.isfile(label_file_path):
            with open(label_file_path) as lines:
                label_strs = [line for line in lines][2:]
            for line in label_strs:
                label = line.split(' ')[8]
                if label in label_dict:
                    label_dict[label] += 1
                else:
                    label_dict.setdefault(label, 1)
    labels = [(k, v) for k, v in label_dict.items()]
    labels.sort(key=lambda x: x[1], reverse=True)
    return labels


def _test_load_dataset():
    dataset = load_dataset(top_num=10)
    print(f'load {len(dataset)} dataset')


def _test_count_labels():
    labels = count_labels()
    print(f'the number of labels: {len(labels)}')
    pprint(labels)


if __name__ == '__main__':
    _test_load_dataset()
    _test_count_labels()
