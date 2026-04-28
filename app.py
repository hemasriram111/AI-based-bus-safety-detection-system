import cv2
import os

# --- 1. Load the Model and Class Labels ---
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
frozen_model = os.path.join(script_dir, 'frozen_inference_graph.pb')
labels_file = os.path.join(script_dir, 'labels.txt')

model = cv2.dnn.DetectionModel(frozen_model, config_file)
classLabels = []
with open(labels_file, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print("Model loaded successfully!")
print(f"Available classes: {classLabels[:5]}...")

# --- 2. Configure the Model ---
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# --- 3. Setup Video Capture ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

font = cv2.FONT_HERSHEY_PLAIN

# --- 4. Main Detection Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    classIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    if len(classIndex) != 0:
        for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            # NEW: Add a safety check to ensure the class index is valid
            if classInd - 1 < len(classLabels):
                className = classLabels[classInd - 1]

                # THE KEY STEP: Only proceed if the detected object is a 'person'
                if className == 'person':
                    cv2.rectangle(frame, boxes, (0, 255, 0), 2)
                    label = f'Person: {conf:.2f}'
                    cv2.putText(frame, label, (boxes[0], boxes[1] - 10), font, 2, (0, 255, 0), 2)

    cv2.imshow('Person Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 5. Clean Up ---
cap.release()
cv2.destroyAllWindows()