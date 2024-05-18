import cv2
import numpy as np
from cvlib.object_detection import draw_bbox
from vidgear.gears import CamGear

solo
# Muat model MobileNet SSD
prototxt = 'deploy.prototxt'  
model = 'mobilenet_iter_73000.caffemodel'  

net = cv2.dnn.readNetFromCaffe(prototxt, model)

# Daftar label COCO yang sesuai dengan indeks dari MobileNet SSD
COCO_LABELS = [
    "background", "unlabeled", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair",
    "unlabeled", "unlabeled", "unlabeled", "unlabeled", "unlabeled",
    "person", "sheep", "sofa", "train", "tvmonitor"
]

COCO_LABELS[3] = "wheelchair"

stream = CamGear(source='https://youtu.be/9Ro1CvVUb_Q', stream_mode=True, logging=True).start()
count = 0

while True:
    frame = stream.read()
    count += 1
    if count % 6 != 0:
        continue

    frame = cv2.resize(frame, (1020, 600))
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    bbox = []
    label = []
    conf = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            if idx < len(COCO_LABELS):
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                bbox.append([startX, startY, endX, endY])
                label.append(COCO_LABELS[idx])
                conf.append(float(confidence))

    frame = draw_bbox(frame, bbox, label, conf)

    total_objects = len(label)
    text = f"Total Objects: {total_objects}"
    
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("FRAME", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

stream.release()
cv2.destroyAllWindows()
