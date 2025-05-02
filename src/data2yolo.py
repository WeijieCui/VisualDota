import os

# 定义 object_name 和 object_id 的映射关系
object_mapping = {
    "plane": 0,
    "ship": 1,
    "storage-tank": 2,
    "baseball-diamond": 3,
    "tennis-court": 4,
    "basketball-court": 5,
    "ground-track-field": 6,
    "harbor": 7,
    "bridge": 8,
    "large-vehicle": 9,
    "small-vehicle": 10,
    "helicopter": 11,
    "roundabout": 12,
    "soccer-ball-field": 13,
    "swimming-pool": 14,
}
orin_directory = "labels"
target_directory = "labels_yolo"


# 定义函数：处理单个文件
def process_file(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            data = line.strip().split()
            if len(data) != 10:
                continue  # 如果数据格式不符合要求，跳过该行
            # 提取 x1, y1, ..., x4, y4, object_name
            coords = [str(float(i) / 1024.0) for i in data[:8]]
            object_name = data[8]
            # 获取 object_id
            object_id = object_mapping.get(object_name)
            if object_id is None:
                continue  # 如果 object_name 不在映射表中，跳过该行
            # 生成新格式：object_id x1 y1 x2 y2 x3 y3 x4 y4
            new_line = f"{object_id} " + " ".join(coords) + "\n"
            outfile.write(new_line)


# 定义函数：遍历目录，处理所有 txt 文件
def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt'):
                input_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_file_path, input_dir)
                output_file_path = os.path.join(output_dir, relative_path)
                # os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                process_file(input_file_path, output_file_path)


if __name__ == '__main__':
    # 设置输入和输出目录
    data_directories = ['data/train', 'data/val']
    for data_directory in data_directories:
        # 调用函数
        process_directory(os.path.join(data_directory, orin_directory),
                          os.path.join(data_directory, target_directory))
