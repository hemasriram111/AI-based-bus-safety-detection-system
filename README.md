This is a professional GitHub `README.md` template based on the research paper provided. It is structured to be clear, informative, and visually organized for developers or recruiters viewing your repository.

---

# AI-Based Bus Safety Detection System

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![TensorFlow](https://img.shields.io/badge/Model-SSD--MobileNet_V3-orange)
![Status](https://img.shields.io/badge/Status-Research_Prototype-brightgreen)

An intelligent, real-time monitoring system designed to reduce public transit accidents by detecting unsafe passenger behaviors—such as footboard traveling and protruding from windows—using Computer Vision and Deep Learning.

## 📌 Overview
Public bus transportation systems often face safety risks due to overcrowding and risky passenger habits. This project introduces a proactive safety solution that shifts away from manual monitoring. By utilizing **SSD MobileNet V3** and **Region of Interest (ROI)** algorithms, the system identifies safety violations and provides instant audio-visual alerts to drivers and passengers.

### Key Features
* **Real-time Person Detection:** High-speed detection at 18–22 FPS.
* **ROI Monitoring:** Specialized tracking of restricted footboard and window zones.
* **Spatial-Temporal Logic:** A 10-second validation timer filters out transient movements to prevent false alarms.
* **Multimodal Alerts:** Integrated Text-to-Speech (TTS) warnings and high-frequency siren beeps.
* **Event Logging:** Automatic timestamped recording of violations for safety audits.

---

## 🏗 System Architecture
The system operates on an edge computing paradigm to ensure low latency and data privacy.



1.  **Data Acquisition:** Captures HD video via USB/CCTV sensors.
2.  **Pre-processing:** Frames are resized to $320 \times 320$ and normalized for the AI engine.
3.  **Inference Engine:** SSD MobileNet V3 identifies passengers with a confidence threshold of 0.55.
4.  **Decision Logic:** Uses the `pointPolygonTest` algorithm to determine if a person is within the hazardous ROI.
5.  **Alerting:** Triggers `pyttsx3` and `winsound` if a violation exceeds 10 seconds.

---

## 🛠 Technology Stack
* **Language:** Python 3.7+
* **CV Framework:** OpenCV (DNN Module)
* **Deep Learning Model:** SSD-MobileNet V3 (TensorFlow weights)
* **Audio Synthesis:** `pyttsx3`, `winsound`
* **OS Support:** Windows 10/11

---

## 🚀 Getting Started

### Prerequisites
* Python 3.7 or higher
* A webcam or IP camera
* The pre-trained SSD MobileNet V3 configuration and weight files

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/bus-safety-detection.git
   cd bus-safety-detection
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python pyttsx3 numpy
   ```
3. Ensure the model files (`ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` and `.pb`) are in the root directory.

### Running the System
```bash
python main.py
```

---

## 📊 Performance & Results
* **Accuracy:** 94% Mean Average Precision (mAP).
* **Speed:** Consistent 22 FPS on standard hardware (Intel i3/Raspberry Pi).
* **Detection Workflow:**
    | Step | Action | Outcome |
    | :--- | :--- | :--- |
    | 1 | Detection | Person enters Footboard ROI ($T > 0s$) |
    | 2 | Verification | Sustained occupancy $> 10s$ |
    | 3 | Notification | Audio/Visual triggers initiated |
    | 4 | Recording | Log saved with Time & Duration |

---

## 👥 Contributors
* **J. Bindu**
* **T. A. N. Nookesh Reddy**
* **P. N. V. Bhaskar Reddy**
* **G. Hema Sriram**
* **Guide:** Mrs. M. Suneetha MTech
* *Aditya College of Engineering and Technology, Andhra Pradesh, India*

## 📜 References
* Howard, A., et al. (2019). *Searching for MobileNetV3*. ICCV.
* Liu, W., et al. (2016). *SSD: Single Shot Multibox Detector*. ECCV.

---

### Contact
For queries, contact: [hemasriram111@gmail.com](mailto:hemasriram111@gmail.com)
