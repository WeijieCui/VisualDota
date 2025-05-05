import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def enhance_channel(channel, factor=1.5, max_value=255):
    """
    Enhances the brightness of a single channel by scaling it with a given factor
    """
    enhanced_channel = channel * factor
    enhanced_channel = np.clip(enhanced_channel, 0, max_value)

    return enhanced_channel.astype(np.uint8)


def parse_ground_truth(filename):
    """
    This method is to convert the validation set label files into a format suitable for mAP calculation, 
    using the original, unprocessed source val label files.
    """
    objects = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('imagesource') or line.startswith('gsd'):
            continue
        parts = line.strip().split()
        coords = list(map(float, parts[:8]))
        classname = parts[8]
        difficult = int(parts[9])
        objects.append((coords, classname, difficult))
    return objects

def parse_detections(filename):
    """
    This method is to convert the detections label files for mAP calculation,
    The format follows the OBB specification on the DOTA dataset website, the structure described as follows:
    file name:plane.txt
    file content:imgname score x1 y1 x2 y2 x3 y3 x4 y4
    """
    detections = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        image_id = parts[0]
        confidence = float(parts[1])
        coords = list(map(float, parts[2:10]))
        detections.append((image_id, confidence, coords))
    return detections

def to_polygon_coords(coords):
    return [(coords[i], coords[i + 1]) for i in range(0, 8, 2)]



def compute_iou_poly(poly1, poly2):
    """
    The KIT provided by the DOTA dataset is implemented in C. 
    Instead, Shapely was used to perform the necessary geometric computations.
    """
    try:
        p1 = Polygon(to_polygon_coords(poly1))
        p2 = Polygon(to_polygon_coords(poly2))
        if not p1.is_valid or not p2.is_valid:
            return 0.0
        intersection = p1.intersection(p2).area
        union = p1.union(p2).area
        return intersection / union if union > 0 else 0.0
    except Exception as e:
        print("Polygon construction failed:", e)
        return 0.0

def voc_eval(detpath, annopath, imagesetfile, classname, ovthresh=0.5, use_07_metric=False):
    """
    The method based on theDOTA KIT, in order to calculate mAP
    """
    with open(imagesetfile, 'r') as f:
        imagenames = [x.strip() for x in f.readlines()]

    class_recs = {}
    npos = 0
    for imagename in imagenames:
        gtfile = annopath.format(imagename)
        objects = parse_ground_truth(gtfile)
        bbox = [obj[0] for obj in objects if obj[1] == classname]
        difficult = np.array([obj[2] for obj in objects if obj[1] == classname]).astype(bool)
        det = [False] * len(bbox)
        npos += sum(~difficult)
        class_recs[imagename] = {'bbox': bbox, 'difficult': difficult, 'det': det}

    detfile = detpath.format(classname)
    detections = parse_detections(detfile)
    detections = sorted(detections, key=lambda x: -x[1])  # Sort by confidence

    nd = len(detections)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        image_id, confidence, bb = detections[d]
        R = class_recs.get(image_id, None)
        if R is None:
            fp[d] = 1.0
            continue
        BBGT = R['bbox']
        ovmax = -np.inf
        jmax = -1
        for j, gt in enumerate(BBGT):
            iou = compute_iou_poly(gt, bb)
            if iou > ovmax:
                ovmax = iou
                jmax = j
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.0
                    R['det'][jmax] = True
                else:
                    fp[d] = 1.0
            else:
                fp[d] = 1.0
        else:
            fp[d] = 1.0

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos) if npos > 0 else np.zeros_like(tp)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    if use_07_metric:
        ap = 0.0
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.0
    else:
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return rec, prec, ap


def run_dota_evaluation(
    dataset_root_dir='./data',
    detpath_template='data/val/fasterrcnn_prediction/{:s}.txt',
    annopath_template='data/val/labelTxt/{:s}.txt',
    printclassap=False
):
    """
    This method is used for mAP calculation.

    detpath_template : Template path of the prediction result files.
    annopath_template : Template path to the ground truth label files.
    printclassap : Whether to print AP for each class.
    """
    # Automatically generate predict.txt
    img_dir = os.path.join(dataset_root_dir, 'val/images')
    predict_txt_path = os.path.join(dataset_root_dir, 'val/fasterrcnn_prediction', 'predict.txt')

    img_files = [f for f in os.listdir(img_dir) if f.endswith('.png') or f.endswith('.jpg')]
    file_prefixes = [os.path.splitext(f)[0] for f in img_files]

    with open(predict_txt_path, 'w') as f:
        for prefix in file_prefixes:
            f.write(prefix + '\n')

    imagesetfile = predict_txt_path

    # Define class names
    classnames = [
        'plane', 'baseball-diamond', 'bridge', 'ground-track-field',
        'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
        'basketball-court', 'storage-tank', 'soccer-ball-field',
        'roundabout', 'harbor', 'swimming-pool', 'helicopter'
    ]

    # Calculate AP for each class
    total_ap = 0.0
    for classname in classnames:
        rec, prec, ap = voc_eval(detpath_template, annopath_template, imagesetfile, classname, ovthresh=0.5, use_07_metric=True)

        if printclassap:
            print(f"Evaluating class: {classname}")
            print(f"AP for {classname}: {ap:.4f}")

        total_ap += ap

    mean_ap = total_ap / len(classnames)
    print(f"Mean AP: {mean_ap:.4f}")




def display_images(image_filename, yolo_save_dir, faster_rcnn_dir):
    """
    Display original and channel-enhanced/reduced detection results
    from both YOLO and Faster R-CNN models.
    """
    variants = ['original', 'red_enhanced', 'red_reduced', 
                'green_enhanced', 'green_reduced', 
                'blue_enhanced', 'blue_reduced']
    
    # YOLO image paths
    image_paths_yolo = [
        os.path.join(yolo_save_dir, f'{image_filename}') if v == 'original' 
        else os.path.join(yolo_save_dir, f'{image_filename[:-4]}_{v}.png') 
        for v in variants
    ]

    # RCNN image paths
    image_paths_rcnn = [
        os.path.join(faster_rcnn_dir, f'{image_filename}') if v == 'original' 
        else os.path.join(faster_rcnn_dir, f'{image_filename[:-4]}_{v}.png') 
        for v in variants
    ]

    plt.figure(figsize=(16, 6))

    # Show Faster R-CNN images
    for i, path in enumerate(image_paths_rcnn):
        if os.path.exists(path):
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 7, i + 1)
            plt.imshow(img_rgb)
            plt.title(f'RCNN - {variants[i]}', fontsize=8)
            plt.axis('off')

    # Show YOLO images
    for i, path in enumerate(image_paths_yolo):
        if os.path.exists(path):
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(2, 7, i + 8)
            plt.imshow(img_rgb)
            plt.title(f'YOLO - {variants[i]}', fontsize=8)
            plt.axis('off')

    plt.tight_layout()
    plt.show()


def enhance_and_save_images(image_filename, dataset_root_dir, enhanced_dir, reduction_factor=0.5):
    """
    Enhances and reduces the RGB channels of an image and saves the enhanced and reduced images.

    image_filename : Name of the image to process.
    dataset_root_dir : Root directory where the image is located.
    enhanced_dir : Directory to save the enhanced and reduced images.
    reduction_factor : Factor by which to reduce the channel (default is 50% reduction).
    """
    # Ensure the directories exist
    os.makedirs(enhanced_dir, exist_ok=True)
    
    # Read the image
    image_path = os.path.join(dataset_root_dir, 'test/images/', image_filename)
    img_bgr = cv2.imread(image_path)
    
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Split into RGB channels
    b, g, r = cv2.split(img_bgr)
    
    # Enhance each channel
    r_enhanced = enhance_channel(r)
    g_enhanced = enhance_channel(g)
    b_enhanced = enhance_channel(b)
    
    # Reduce each channel by 50%
    r_reduced = (r * reduction_factor).astype(r.dtype)
    g_reduced = (g * reduction_factor).astype(g.dtype)
    b_reduced = (b * reduction_factor).astype(b.dtype)
    
    # Create enhanced images for each channel
    img_red_enhanced = cv2.merge((b, g, r_enhanced))
    img_green_enhanced = cv2.merge((b, g_enhanced, r))
    img_blue_enhanced = cv2.merge((b_enhanced, g, r))
    
    # Create reduced images for each channel (50% reduction)
    img_red_reduced = cv2.merge((b, g, r_reduced))
    img_green_reduced = cv2.merge((b, g_reduced, r))
    img_blue_reduced = cv2.merge((b_reduced, g, r))
    
    # Save the enhanced images
    base_name = os.path.splitext(image_filename)[0]
    red_enhanced_path = os.path.join(enhanced_dir, f'{base_name}_red_enhanced.png')
    green_enhanced_path = os.path.join(enhanced_dir, f'{base_name}_green_enhanced.png')
    blue_enhanced_path = os.path.join(enhanced_dir, f'{base_name}_blue_enhanced.png')

    # Save the reduced images
    red_reduced_path = os.path.join(enhanced_dir, f'{base_name}_red_reduced.png')
    green_reduced_path = os.path.join(enhanced_dir, f'{base_name}_green_reduced.png')
    blue_reduced_path = os.path.join(enhanced_dir, f'{base_name}_blue_reduced.png')

    # Write images
    cv2.imwrite(red_enhanced_path, img_red_enhanced)
    cv2.imwrite(green_enhanced_path, img_green_enhanced)
    cv2.imwrite(blue_enhanced_path, img_blue_enhanced)
    
    cv2.imwrite(red_reduced_path, img_red_reduced)
    cv2.imwrite(green_reduced_path, img_green_reduced)
    cv2.imwrite(blue_reduced_path, img_blue_reduced)

    # Return the paths of the original, enhanced, and reduced images
    return [image_path, red_enhanced_path, green_enhanced_path, blue_enhanced_path, 
            red_reduced_path, green_reduced_path, blue_reduced_path]

