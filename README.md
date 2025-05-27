# Visual DOTA

The [DOTA](https://captain-whu.github.io/DOTA/dataset.html) dataset is a large-scale benchmark
designed for multi-object detection in aerial images. 
It contains 2,800+ high-resolution images (800×800 to 20,000×20,000 pixels) from multiple sources
like Google Earth and satellites, with nearly 190,000 annotated object instances across 15 categories
(e.g., plane, ship, vehicle, sports fields).
Each object is labeled using oriented bounding boxes (OBBs) to handle complex orientations,
and tagged by detection difficulty. 
This makes DOTA ideal for testing models in real world scenarios involving scale, rotation, and dense object layouts.

![img.png](img.png)

In disaster response and rescue operations, rapid and accurate identification of critical
infrastructure and affected zones is essential. 
Implementing multi-object detection models allows automated analysis of large-scale aerial imagery
to locate damaged buildings, blocked roads, vehicles, and gathering areas for survivors. 
The DOTA dataset, with its extensive annotations of structures such as ships, vehicles, and sports fields,
offers a practical foundation for training AI systems aimed at enhancing situational awareness during emergencies.
Leveraging advanced deep learning detectors trained on this dataset can significantly prioritize rescue efforts,
and improve coordination in high-pressure rescue scenarios.

This project used three models, YOLO V1 (Building from scratch), YOLO V11 (Pretrained, Oriental Bounding Box) 
and Faster R-CNN (Pretrained + Transfering, Horizontal Bounding Box), to verify their performance on the DOTA dataset.
Practice shows that the YOLO V11 model performs best on the dota dataset.

<img alt="result_car2.png" src="src\asserts\images\yolo11\result_car2.png" width="800"/>

<img alt="result_plane.png" src="src\asserts\images\yolo11\result_plane.png" width="800"/>

Please see the [report](CSMAI_coursework_report.pdf) for details.


This project is also available in the GitHub repository https://github.com/UoR-Vision/VisualDota.

To make it easier, you can run this code on [colab](https://colab.research.google.com/github/UoR-Vision/VisualDota/blob/main/src/main.ipynb).

## 🙏 Acknowledgement
Our experiments used the [DOTA](https://captain-whu.github.io/DOTA/dataset.html) dataset.

We use the structures and weights of [YOLO V1](https://arxiv.org/pdf/1506.02640),
[YOLO V11](https://docs.ultralytics.com/models/yolo11/),
and [Faster R-CNN](https://arxiv.org/abs/1506.01497) models in our experiments.