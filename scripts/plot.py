import cv2
from ultralytics import YOLO

model = YOLO("models/photosheap.pt")

def plot_results(results):
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class IDs

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            label = f"Class {int(cls)}: {score:.2f}"
            cv2.rectangle(result.orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result.orig_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return results.orig_img

results = model.predict(source="data/images", show=True, save=True)
results = plot_results(results)
cv2.imwrite("output/results.jpg", results)  