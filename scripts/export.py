from ultralytics import YOLO

model = YOLO("models/photosheap.pt")  
model.export(format="onnx")  