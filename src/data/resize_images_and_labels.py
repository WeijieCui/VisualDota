import os
from PIL import Image


# Resize Images
def resize_image(img, target_size, max_size):
    width, height = img.size
    min_edge = min(width, height)
    ratio = target_size / min_edge  # Calculate the scale ratio for short edge
    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Check if the long edge exceed the max_size
    if max(new_width, new_height) > max_size:
        ratio = max_size / max(new_width, new_height)
        new_width = int(new_width * ratio)
        new_height = int(new_height * ratio)

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


# Batch resize images and labels
def batch_resize_images_and_labels(
        root_dir: str = 'tiny_train',
        image_dir: str = 'images',
        label_dir: str = 'labels',
        target_size: int = 500,
        max_size: int = 1000,
):
    # Process all images and labels
    for root, dirs, files in os.walk(os.path.join(root_dir, image_dir)):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                try:
                    with Image.open(file_path) as img:
                        resized_img = resize_image(img, target_size, max_size)
                        resized_img.save(file_path)  # overwrite source file
                        # print(f"Resize {file} -> {resized_img.size}")
                        ratio = resized_img.width / img.width
                    label_path = os.path.join(root_dir, label_dir, file.split('.')[0] + '.txt')
                    label_raws = []
                    with open(label_path) as label_file:
                        lines = label_file.readlines()
                        label_raws.extend(lines[:2])
                        for line in lines[2:]:
                            parts = line.strip().split()
                            if len(parts) < 9:
                                label_raws.append(line)
                                continue
                            points = list(map(lambda i: str(int(ratio * int(i))), parts[:8]))
                            label_raws.append(' '.join(points) + ' ' + parts[8] + ' ' + parts[9] + '\n')
                    if len(label_raws) <= 2:
                        print(file_path)
                    with open(label_path, 'w') as label_file:
                        label_file.writelines(label_raws)
                except Exception as e:
                    print(f"Failed to resize {file}, error: {e}")


if __name__ == '__main__':
    batch_resize_images_and_labels(
        root_dir='tiny_train',
    )
