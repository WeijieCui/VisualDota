from ultralytics import YOLO

if __name__ == '__main__':

	# 记得替换下面的路径
	model = YOLO(r"D:\好康的1\好康的\qwe\runs\obb\train57\weights\best.pt")
	results = model(r"D:\AI data\DOTA\test\images\P0006.png")
	results[0].save()