import os
from ultralytics import YOLO

if __name__ == '__main__':

	current = os.path.dirname(os.path.realpath(__file__))
	# 初始化模型并加载权重 记得替换这里的路径
	model = YOLO(r"C:\ding\好康的\qwe\runs\obb\train57\weights\best.pt")
	# 进行推理并保存结果 记得替换这里的路径
	results = model.predict(source=r"C:\ding\DOTA_split\DOTA_split\test\images", save=False, save_txt=True, save_conf=True)