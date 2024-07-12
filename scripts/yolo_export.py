from ultralytics import YOLO

m = YOLO("yolov8n.pt")
m.export(format="rknn", opset=19)
