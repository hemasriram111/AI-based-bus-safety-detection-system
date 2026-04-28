import cv2
import os
import numpy as np
import time
import pyttsx3
import threading
import winsound
from datetime import datetime

# --- 1. Helper Function to Find Coordinates ---
def print_coords(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Coordinates: ({x}, {y})")

# --- 2. Load Model and Define ROI ---
script_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(script_dir, 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
frozen_model = os.path.join(script_dir, 'frozen_inference_graph.pb')
labels_file = os.path.join(script_dir, 'labels.txt')

model = cv2.dnn.DetectionModel(frozen_model, config_file)
classLabels = []
with open(labels_file, 'rt') as fpt: 
    classLabels = fpt.read().rstrip('\n').split('\n')

area1_coords = [(320, 350), (640, 350), (640, 480), (320, 480)]
area1 = np.array(area1_coords, np.int32)

# --- 3. Configure the Model ---
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# --- 4. Setup Video Capture ---
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

def play_alert():
    try:
        # Avoid thread COM errors on Windows
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass
        engine = pyttsx3.init()
        engine.setProperty('rate', 150) # Slower and clearer voice
        engine.say("Unsafe journey detected. Please go inside.")
        engine.runAndWait()
        
        # Play a siren sound (alternating frequencies)
        for _ in range(5):
            winsound.Beep(800, 500) # 800 Hz for 500 milliseconds
            winsound.Beep(600, 500) # 600 Hz for 500 milliseconds
            
    except Exception as e:
        print(f"Error playing sound alert: {e}")

font = cv2.FONT_HERSHEY_PLAIN
window_name = 'Person Detection in ROI'
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, print_coords)

# --- 5. Main Detection Loop ---
person_detected_start_time = None
unsafe_journey = False
alert_played = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.polylines(frame, [area1], isClosed=True, color=(0, 0, 255), thickness=2)

    classIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)

    person_in_area_this_frame = False

    if len(classIndex) != 0:
        for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
            if classInd - 1 < len(classLabels):
                className = classLabels[classInd - 1]

                if className == 'person':
                    x, y, w, h = boxes
                    
                    # MODIFIED: Convert point coordinates to float to fix the error
                    person_point = (float(x + w // 2), float(y + h))

                    is_inside = cv2.pointPolygonTest(area1, person_point, False)

                    if is_inside >= 0:
                        person_in_area_this_frame = True
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label = f'Person: {conf:.2f}'
                        cv2.putText(frame, label, (x, y - 10), font, 2, (0, 255, 0), 2)

    if person_in_area_this_frame:
        if person_detected_start_time is None:
            person_detected_start_time = time.time()
        else:
            elapsed_time = time.time() - person_detected_start_time
            if elapsed_time >= 10:
                unsafe_journey = True
                
                # Play the voice alert only once per trigger
                if not alert_played:
                    threading.Thread(target=play_alert, daemon=True).start()
                    alert_played = True
            else:
                cv2.putText(frame, f'Timer: {int(elapsed_time)}s', (10, 30), font, 2, (0, 165, 255), 2)
    else:
        if unsafe_journey and person_detected_start_time is not None:
            total_duration = int(time.time() - person_detected_start_time)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"unsafe journey detected : yes\ntime stamp : {timestamp}\nhow much time : {total_duration} sec\n--------------------------\n"
            log_file_path = os.path.join(script_dir, "unsafe_journey_logs.txt")
            with open(log_file_path, "a") as f:
                f.write(log_entry)
                
        person_detected_start_time = None
        unsafe_journey = False
        alert_played = False

    if unsafe_journey:
        cv2.putText(frame, "Person detected: Unsafe Journey", (10, 60), font, 2, (0, 0, 255), 3)

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Log if we exit while a person is still doing an unsafe journey
        if unsafe_journey and person_detected_start_time is not None:
            total_duration = int(time.time() - person_detected_start_time)
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"unsafe journey detected : yes\ntime stamp : {timestamp}\nhow much time : {total_duration} sec\n--------------------------\n"
            log_file_path = os.path.join(script_dir, "unsafe_journey_logs.txt")
            with open(log_file_path, "a") as f:
                f.write(log_entry)
        break

# --- 6. Clean Up ---
cap.release()
cv2.destroyAllWindows()